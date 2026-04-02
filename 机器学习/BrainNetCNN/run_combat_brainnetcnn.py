import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, balanced_accuracy_score

sys.path.append('/mnt/sda1/zhangyan/cpac_output/机器学习/BrainNetCNN')
from model_brainnetcnn import BrainNetCNN, BrainNetCNNSimplified

try:
    from neuroHarmonize import harmonizationLearn, harmonizationApply
    HAS_NEUROHARMONIZE = True
except ImportError:
    HAS_NEUROHARMONIZE = False
    warnings.warn("neuroHarmonize not found.")

def mixup_data(x, d, y, alpha=0.2, device='cpu'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_d = lam * d + (1 - lam) * d[index]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_d, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

def rebuild_symmetric_matrix(upper_tri_1d, n_nodes=116):
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    iu = np.triu_indices(n_nodes, k=1)
    matrix[iu] = upper_tri_1d
    matrix = matrix + matrix.T
    return matrix

def load_participants_metadata(file_path):
    df = pd.read_csv(file_path)
    if 'ScanDir ID' in df.columns:
        id_col = 'ScanDir ID'
    elif 'participant_id' in df.columns:
        id_col = 'participant_id'
    else:
        id_col = [c for c in df.columns if 'ID' in c.upper()][0]
        
    df['participant_id'] = df[id_col].astype(str).str.replace('sub-', '', regex=False)
    df['group'] = df['DX'].apply(lambda x: 0 if x == 0 else 1)
    
    for col in ['Site', 'Age', 'Gender', 'Handedness']:
        if col not in df.columns:
            for c in df.columns:
                if c.lower() == col.lower():
                    df.rename(columns={c: col}, inplace=True)
                    break
                    
    data_dict = {
        'participant_id': df['participant_id'],
        'group': df['group'],
        'Site': df['Site'].astype(str) if 'Site' in df.columns else 'Unknown',
        'Age': pd.to_numeric(df['Age'], errors='coerce') if 'Age' in df.columns else 0,
        'Gender': pd.to_numeric(df['Gender'], errors='coerce') if 'Gender' in df.columns else 0,
        'Handedness': pd.to_numeric(df['Handedness'], errors='coerce') if 'Handedness' in df.columns else 1
    }
    df_clean = pd.DataFrame(data_dict).dropna(subset=['participant_id', 'group', 'Site'])
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].mean())
    df_clean['Gender'] = df_clean['Gender'].fillna(0)
    df_clean['Handedness'] = df_clean['Handedness'].fillna(1)
    return df_clean

def find_all_fc_files(base_dir):
    fc_files = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(('.txt', '.tsv', '.csv')):
                base_name = f.split('_')[0].replace('sub-', '')
                fc_files[base_name] = os.path.join(root, f)
    return fc_files

def fisher_z_transform(r):
    r_clipped = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))

def run_experiment(use_combat=True):
    participants_path = '/mnt/sda1/zhangyan/cpac_output/机器学习/人口学/ADHD_all_phenotypic.csv'
    fc_dir = '/mnt/sda1/zhangyan/cpac_output/feature/deepprep'
    output_dir = '/mnt/sda1/zhangyan/cpac_output/机器学习/combat/BrainNetCNN/deepprep'
    os.makedirs(output_dir, exist_ok=True)
    
    participants_df = load_participants_metadata(participants_path)
    fc_file_map = find_all_fc_files(fc_dir)
    
    features, labels, valid_covars, demos = [], [], [], []
    for _, row in participants_df.iterrows():
        sid = row['participant_id']
        match_file = fc_file_map.get(sid)
        if not match_file:
            for fid, fpath in fc_file_map.items():
                if sid in fid or fid in sid:
                    match_file = fpath
                    break
        if match_file:
            try:
                fc_matrix = pd.read_csv(match_file, sep=None, header=None, engine='python').values
                if fc_matrix.shape != (116, 116):
                    continue
                iu = np.triu_indices_from(fc_matrix, k=1)
                fc_vector = fc_matrix[iu]
                fc_vector_z = fisher_z_transform(fc_vector)
                
                if np.isnan(fc_vector_z).any() or np.isinf(fc_vector_z).any():
                    continue
                    
                features.append(fc_vector_z)
                labels.append(row['group'])
                valid_covars.append({
                    'SITE': row['Site'], 'AGE': row['Age'], 'GENDER': row['Gender']
                })
                demos.append([row['Gender'], row['Age'], row['Handedness']])
            except:
                continue
                
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    demos = np.array(demos, dtype=np.float32)
    covars_df = pd.DataFrame(valid_covars)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Matched Subjects: {len(X)}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratify_key = covars_df['SITE'].astype(str) + "_" + pd.Series(y).astype(str)
    
    fold_metrics = []
    num_epochs = 250
    batch_size = 16
    lr = 1e-3
    weight_decay = 1e-2
    dropout_rate = 0.6
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, stratify_key)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        d_train, d_test = demos[train_idx], demos[test_idx]
        
        c_train = covars_df.iloc[train_idx].reset_index(drop=True)
        c_test = covars_df.iloc[test_idx].reset_index(drop=True)
        
        if use_combat and HAS_NEUROHARMONIZE:
            model_cm, X_train_adj = harmonizationLearn(X_train, c_train)
            X_test_adj = harmonizationApply(X_test, c_test, model_cm)
        else:
            X_train_adj, X_test_adj = X_train, X_test
            
        X_train_mat = np.stack([rebuild_symmetric_matrix(v) for v in X_train_adj]).astype(np.float32)
        X_test_mat = np.stack([rebuild_symmetric_matrix(v) for v in X_test_adj]).astype(np.float32)
        
        X_train_t = torch.FloatTensor(X_train_mat).unsqueeze(1)
        X_test_t = torch.FloatTensor(X_test_mat).unsqueeze(1)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
        d_train_t = torch.tensor(d_train, dtype=torch.float32)
        d_test_t = torch.tensor(d_test, dtype=torch.float32)
        
        train_ds = TensorDataset(X_train_t, y_train_t, d_train_t)
        test_ds = TensorDataset(X_test_t, y_test_t, d_test_t)
        
        class_counts = np.bincount(y_train)
        weights = [1.0/class_counts[int(l)] for l in y_train]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(y_train), replacement=True)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        # Enable enhanced BrainNetCNN features
        net = BrainNetCNNSimplified(n_nodes=116, demo_dim=3, dropout_rate=dropout_rate).to(device)
        criterion = FocalLoss(alpha=0.5, gamma=2.0)
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
        
        best_auc = 0
        best_test_metrics = None
        
        for epoch in range(num_epochs):
            net.train()
            train_loss = 0
            for bx, by, bd in train_loader:
                bx, by, bd = bx.to(device), by.to(device), bd.to(device)
                
                # Setup robust dynamic noise
                noise = torch.randn_like(bx) * 0.02
                bx = bx + noise
                
                # Apply MixUp augmentation
                bx_mixed, bd_mixed, y_a, y_b, lam = mixup_data(bx, bd, by, alpha=0.2, device=device)
                
                optimizer.zero_grad()
                out = net(bx_mixed, demo=bd_mixed)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
                
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                
            scheduler.step()
            
            # Predict
            net.eval()
            all_preds, all_probs, all_labels = [], [], []
            with torch.no_grad():
                for bx, by, bd in test_loader:
                    bx, bd = bx.to(device), bd.to(device)
                    out = net(bx, demo=bd)
                    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    preds = torch.max(out, 1)[1].cpu().numpy()
                    all_probs.extend(probs)
                    all_preds.extend(preds)
                    all_labels.extend(by.numpy())
                    
            y_true, y_pred, y_prob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
            
            if auc > best_auc or epoch == 0:
                best_auc = auc
                best_test_metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1': f1_score(y_true, y_pred, zero_division=0),
                    'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                    'auc': auc
                }
                
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:03d}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f} | Test AUC: {auc:.4f} | Test Acc: {accuracy_score(y_true, y_pred):.4f}")
                
        print(f"Fold {fold+1} Best AUC: {best_auc:.4f}, Accuracy: {best_test_metrics['accuracy']:.4f}")
        fold_metrics.append(best_test_metrics)
        
    suffix = 'combat_enhanced' if use_combat else 'nocombat_enhanced'
    summary_path = os.path.join(output_dir, f"brainnetcnn_results_{suffix}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Results Enhanced (ComBat={use_combat})\n{'='*30}\n")
        f.write("Features: FocalLoss, Mixup(alpha=0.2), GaussianNoise(0.02), CosineAnnealingLR\n")
        f.write("-" * 20 + "\n")
        df_res = pd.DataFrame(fold_metrics)
        m = df_res.mean()
        s = df_res.std()
        for col in df_res.columns:
            f.write(f"{col}: {m[col]:.4f} +- {s[col]:.4f}\n")
            
    print(f"Fold completed. Results saved to {summary_path}")

if __name__ == '__main__':
    print("=== Running Enhanced BrainNetCNN Pipeline WITHOUT ComBat ===")
    run_experiment(use_combat=False)
    
    if HAS_NEUROHARMONIZE:
        print("\n=== Running Enhanced BrainNetCNN Pipeline WITH ComBat ===")
        run_experiment(use_combat=True)
    else:
        print("\nSkipping BrainNetCNN combat because neuroHarmonize missing.")
