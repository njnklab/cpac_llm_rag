import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*torch-scatter.*")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, balanced_accuracy_score

# PyG imports
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    warnings.warn("PyTorch Geometric not found. BrainGNN cannot run.")

sys.path.append('/mnt/sda1/zhangyan/cpac_output/机器学习/BrainGNN')
from model_braingnn import BrainGNN, BrainGNNSimplified

try:
    from neuroHarmonize import harmonizationLearn, harmonizationApply
    HAS_NEUROHARMONIZE = True
except ImportError:
    HAS_NEUROHARMONIZE = False
    warnings.warn("neuroHarmonize not found.")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
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


def create_pyg_data(fc_matrix, label, demo=None, n_nodes=116, use_abs=False, threshold=0.15):
    """
    Modified to include `threshold` to sparsify the graph.
    Fully connected graphs cause GCN to over-smooth instantly.
    """
    edge_fc = fc_matrix.copy()
    np.fill_diagonal(fc_matrix, 0)
    np.fill_diagonal(edge_fc, 0)
    
    # Use Identity matrix for node features so each region has a unique embedding.
    # Passing the correlation matrix itself as features causes A*A squared smoothing.
    x = torch.eye(n_nodes, dtype=torch.float32)
    
    edges = []
    edge_weight = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                w = edge_fc[i, j]
                # Edge Sparsification: Only keep strong connections > threshold
                if abs(w) > threshold:
                    edges.append([i, j])
                    edge_weight.append(abs(w) if use_abs else w)
    
    if len(edges) == 0:
        # Fallback if graph is completely disconnected
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and abs(edge_fc[i,j]) > 0.01:
                    edges.append([i, j])
                    edge_weight.append(abs(edge_fc[i,j]) if use_abs else edge_fc[i,j])
                    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weight, dtype=torch.float32)
    
    y = torch.tensor([label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    if demo is not None:
        data.demo = torch.tensor(demo, dtype=torch.float32).unsqueeze(0)
    return data


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
    if not PYG_AVAILABLE:
        print("Cannot run experiment: PyTorch Geometric not installed.")
        return

    participants_path = '/mnt/sda1/zhangyan/cpac_output/机器学习/人口学/ADHD_all_phenotypic.csv'
    fc_dir = '/mnt/sda1/zhangyan/cpac_output/feature/deepprep'
    output_dir = '/mnt/sda1/zhangyan/cpac_output/机器学习/combat/BrainGNN/deepprep'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading demographics from {participants_path}")
    participants_df = load_participants_metadata(participants_path)
    
    print(f"Searching for FC mats in {fc_dir}")
    fc_file_map = find_all_fc_files(fc_dir)
    print(f"Found {len(fc_file_map)} FC files in directory.")
    
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
                if fc_matrix.shape != (116, 116): continue
                iu = np.triu_indices_from(fc_matrix, k=1)
                fc_vector = fc_matrix[iu]
                fc_vector_z = fisher_z_transform(fc_vector)
                if np.isnan(fc_vector_z).any() or np.isinf(fc_vector_z).any(): continue
                    
                features.append(fc_vector_z)
                labels.append(row['group'])
                valid_covars.append({'SITE': row['Site'], 'AGE': row['Age'], 'GENDER': row['Gender']})
                demos.append([row['Gender'], row['Age'], row['Handedness']])
            except: continue
                
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    demos = np.array(demos, dtype=np.float32)
    covars_df = pd.DataFrame(valid_covars)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Matched Subjects For BrainGNN: {len(X)}")
    if len(X) == 0: return
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratify_key = covars_df['SITE'].astype(str) + "_" + pd.Series(y).astype(str)
    
    fold_metrics = []
    
    num_epochs = 250
    batch_size = 16
    lr = 5e-4
    weight_decay = 1e-2
    dropout_rate = 0.5
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, stratify_key)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        d_train, d_test = demos[train_idx], demos[test_idx]
        
        c_train = covars_df.iloc[train_idx].reset_index(drop=True)
        c_test = covars_df.iloc[test_idx].reset_index(drop=True)
        
        # 1. ComBat
        if use_combat and HAS_NEUROHARMONIZE:
            combat_c_train = c_train[['SITE', 'AGE', 'GENDER']].copy()
            model_cm, X_train_adj = harmonizationLearn(X_train, combat_c_train)
            combat_c_test = c_test[['SITE', 'AGE', 'GENDER']].copy()
            X_test_adj = harmonizationApply(X_test, combat_c_test, model_cm)
        else:
            X_train_adj, X_test_adj = X_train, X_test
            
        # 2. Rebuild & PyG Graph Setup with edge threshold = 0.15 Sparsification
        train_data_list = []
        class_counts = np.bincount(y_train)
        weights = [1.0/class_counts[int(l)] for l in y_train]
        
        for i, val in enumerate(X_train_adj):
            mat = rebuild_symmetric_matrix(val)
            train_data_list.append(create_pyg_data(mat, y_train[i], d_train[i], use_abs=True, threshold=0.15))
            
        test_data_list = []
        for i, val in enumerate(X_test_adj):
            mat = rebuild_symmetric_matrix(val)
            test_data_list.append(create_pyg_data(mat, y_test[i], d_test[i], use_abs=True, threshold=0.15))
            
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(y_train), replacement=True)
        train_loader = DataLoader(train_data_list, batch_size=batch_size, sampler=sampler, drop_last=True)
        test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)
        
        # 3. Enhanced BrainGNN Model
        net = BrainGNNSimplified(in_channels=116, hidden_channels=32, num_classes=2, dropout_rate=dropout_rate, demo_dim=3).to(device)
        criterion = FocalLoss(alpha=0.5, gamma=2.0)
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
        
        best_auc = 0
        best_test_metrics = None
        
        for epoch in range(num_epochs):
            net.train()
            total_train_loss = 0
            for batch_data in train_loader:
                batch_data = batch_data.to(device)
                
                # Dynamic node features Gaussian Noise to prevent overfitting
                noise = torch.randn_like(batch_data.x) * 0.02
                batch_data.x = batch_data.x + noise
                
                optimizer.zero_grad()
                out = net(batch_data)
                
                y_batch = batch_data.y.view(-1)
                loss = criterion(out, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()
                
            scheduler.step()
            
            # Predict
            net.eval()
            all_preds, all_probs, all_labels = [], [], []
            with torch.no_grad():
                for batch_data in test_loader:
                    batch_data = batch_data.to(device)
                    out = net(batch_data)
                    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    preds = torch.max(out, 1)[1].cpu().numpy()
                    all_probs.extend(probs)
                    all_preds.extend(preds)
                    all_labels.extend(batch_data.y.view(-1).cpu().numpy())
                    
            y_true, y_pred, y_prob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
            
            if auc >= best_auc or epoch == 0:
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
                print(f"  Epoch {epoch+1:03d}/{num_epochs} - Train Loss: {total_train_loss/len(train_loader):.4f} | Test AUC: {auc:.4f} | Test Acc: {accuracy_score(y_true, y_pred):.4f}")
                
        print(f"Fold {fold+1} Best AUC: {best_auc:.4f}, Accuracy: {best_test_metrics['accuracy']:.4f}")
        fold_metrics.append(best_test_metrics)
        
    suffix = 'combat_enhanced' if use_combat else 'nocombat_enhanced'
    summary_path = os.path.join(output_dir, f"braingnn_results_{suffix}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"BrainGNN Results Enhanced (ComBat={use_combat})\n{'='*30}\n")
        f.write("Features: FocalLoss, Sparsified Graph (Threshold=0.15), PyG Form, GaussianNoise(0.02), CosineAnnealingLR\n")
        f.write("-" * 20 + "\n")
        df_res = pd.DataFrame(fold_metrics)
        m = df_res.mean()
        s = df_res.std()
        for col in df_res.columns:
            f.write(f"{col}: {m[col]:.4f} +- {s[col]:.4f}\n")
            
    print(f"Fold completed. Results saved to {summary_path}")

if __name__ == '__main__':
    if not PYG_AVAILABLE:
        print("Please ensure torch-geometric is installed.")
        sys.exit(1)
        
    print("\n=== Running Enhanced BrainGNN Pipeline WITHOUT ComBat ===")
    run_experiment(use_combat=False)
        
    if HAS_NEUROHARMONIZE:
        print("\n=== Running Enhanced BrainGNN Pipeline WITH ComBat ===")
        run_experiment(use_combat=True)
    else:
        print("\nSkipping BrainGNN combat because neuroHarmonize missing.")
