import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score,
                             balanced_accuracy_score)
import json

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. BrainGNN requires PyG.")

from model_braingnn import BrainGNN, BrainGNNSimplified
from dataset_braingnn import BrainGNNDataset, load_fc_data


def compute_class_weights(labels):
    """Compute class weights for imbalanced dataset."""
    labels = np.array(labels)
    n0 = np.sum(labels == 0)
    n1 = np.sum(labels == 1)
    total = len(labels)
    
    w0 = total / (2 * n0) if n0 > 0 else 1.0
    w1 = total / (2 * n1) if n1 > 0 else 1.0
    
    return torch.tensor([w0, w1], dtype=torch.float32)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for data in train_loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data.y)
        loss.backward()
        # Gradient clipping to prevent NaNs/Exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item() * data.y.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == data.y).sum().item()
        total += data.y.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model on given loader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    all_subject_ids = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, data.y)
            
            total_loss += loss.item() * data.y.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_subject_ids.extend(data.subject_id)
    
    avg_loss = total_loss / len(all_labels)
    
    return {
        'loss': avg_loss,
        'y_true': np.array(all_labels),
        'y_pred': np.array(all_preds),
        'y_prob': np.array(all_probs),
        'subject_ids': all_subject_ids
    }


def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics."""
    # Compute confusion matrix with fixed labels to ensure a 2x2 matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Handle potential NaNs in y_prob to prevent roc_auc_score crash
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
    }
    
    return metrics


def train_fold(model, train_loader, val_loader, criterion, optimizer, 
               device, num_epochs=100, patience=20, save_path=None):
    """Train model for one fold with early stopping."""
    best_val_auc = 0.0
    best_model_state = None
    best_val_metrics = None
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_results = evaluate(model, val_loader, criterion, device)
        val_loss = val_results['loss']
        val_metrics = compute_metrics(val_results['y_true'], 
                                       val_results['y_pred'], 
                                       val_results['y_prob'])
        val_auc = val_metrics['auc']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            best_val_metrics = val_metrics
            best_val_metrics['val_loss'] = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if save_path and best_model_state:
        torch.save(best_model_state, save_path)
    
    return best_model_state, best_val_metrics, history


def run_braingnn_cv(
    fc_paths,
    labels,
    subject_ids,
    demographics=None,
    n_splits=5,
    random_state=42,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-4,
    num_epochs=100,
    patience=20,
    dropout_rate=0.5,
    pooling_ratio=0.5,
    hidden_channels=64,
    conv_type='gcn',
    use_simplified=False,
    add_noise=False,
    noise_std=0.01,
    output_dir=None,
    device=None
):
    """
    Run 5-fold cross-validation for BrainGNN with Hybrid Demographic features.
    """
    if not PYG_AVAILABLE:
        raise RuntimeError("PyTorch Geometric is required for BrainGNN")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    labels = np.array(labels)
    n_nodes = 116
    demo_dim = demographics.shape[1] if demographics is not None else 0
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_metrics = []
    all_test_results = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(fc_paths, labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*60}")
        
        train_val_labels = labels[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, 
            test_size=0.2, 
            random_state=random_state, 
            stratify=train_val_labels
        )
        
        # Helper to split by indices
        def get_split(indices):
            paths = [fc_paths[i] for i in indices]
            lbls = labels[indices].tolist()
            subs = [subject_ids[i] for i in indices]
            demos = demographics[indices] if demographics is not None else None
            return paths, lbls, subs, demos

        train_paths, train_labels, train_subjects, train_demos = get_split(train_idx)
        val_paths, val_labels, val_subjects, val_demos = get_split(val_idx)
        test_paths, test_labels, test_subjects, test_demos = get_split(test_idx)
        
        print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        
        # Create datasets
        train_dataset = BrainGNNDataset(train_paths, train_labels, train_subjects, train_demos, n_nodes=n_nodes, add_noise=add_noise, noise_std=noise_std)
        val_dataset = BrainGNNDataset(val_paths, val_labels, val_subjects, val_demos, n_nodes=n_nodes)
        test_dataset = BrainGNNDataset(test_paths, test_labels, test_subjects, test_demos, n_nodes=n_nodes)
        
        class_counts = np.bincount(train_labels)
        class_weights_arr = 1.0 / np.where(class_counts == 0, 1.0, class_counts)
        sample_weights = [class_weights_arr[int(l)] for l in train_labels]
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)

        train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model with demo_dim
        if use_simplified:
            model = BrainGNNSimplified(
                in_channels=n_nodes, 
                hidden_channels=hidden_channels,
                num_classes=2, 
                pooling_ratio=pooling_ratio,
                dropout_rate=dropout_rate,
                conv_type=conv_type,
                demo_dim=demo_dim
            )
        else:
            model = BrainGNN(
                in_channels=n_nodes, 
                hidden_channels=hidden_channels,
                num_classes=2, 
                pooling_ratio=pooling_ratio,
                dropout_rate=dropout_rate,
                conv_type=conv_type,
                demo_dim=demo_dim
            )
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        save_path = os.path.join(output_dir, f'best_model_fold_{fold+1}.pth') if output_dir else None
        best_state, val_metrics, history = train_fold(
            model, train_loader, val_loader, criterion, optimizer,
            device, num_epochs, patience, save_path
        )
        
        model.load_state_dict(best_state)
        test_results = evaluate(model, test_loader, criterion, device)
        test_metrics = compute_metrics(test_results['y_true'], test_results['y_pred'], test_results['y_prob'])
        test_metrics['val_auc'] = val_metrics['auc']
        test_metrics['fold'] = fold + 1
        
        print(f"\nFold {fold+1} Test Results (with Demographics):")
        for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            print(f"  {k.capitalize()}: {test_metrics[k]:.4f}")
        
        fold_metrics.append(test_metrics)
        
        for i, subject_id in enumerate(test_results['subject_ids']):
            all_test_results.append({
                'fold': fold + 1,
                'subject_id': subject_id,
                'y_true': int(test_results['y_true'][i]),
                'y_pred': int(test_results['y_pred'][i]),
                'y_prob': float(test_results['y_prob'][i])
            })
    
    summary_metrics = {}
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity', 'balanced_accuracy']
    for metric_name in metric_names:
        values = [m[metric_name] for m in fold_metrics]
        summary_metrics[f'{metric_name}_mean'] = np.mean(values)
        summary_metrics[f'{metric_name}_std'] = np.std(values)
    
    total_cm = np.array([[0, 0], [0, 0]])
    for m in fold_metrics:
        total_cm += np.array(m['confusion_matrix'])
    summary_metrics['confusion_matrix_sum'] = total_cm.tolist()
    
    predictions_df = pd.DataFrame(all_test_results)
    return fold_metrics, summary_metrics, predictions_df


def print_results(fold_metrics, summary_metrics):
    """Print results in a formatted way."""
    print("\n" + "="*60)
    print("BrainGNN 5-Fold Cross-Validation (Hybrid Strategy)")
    print("="*60)
    
    metric_names = [('Accuracy', 'accuracy'), ('Precision', 'precision'), ('Recall', 'recall'), 
                    ('F1-score', 'f1'), ('AUC', 'auc')]
    
    for display_name, metric_key in metric_names:
        mean = summary_metrics[f'{metric_key}_mean']
        std = summary_metrics[f'{metric_key}_std']
        print(f"{display_name:20s}: {mean:.4f} ± {std:.4f}")
    
    print("\nConfusion Matrix (sum):")
    cm = summary_metrics['confusion_matrix_sum']
    print(f"[[{cm[0][0]:4d} {cm[0][1]:4d}]\n [{cm[1][0]:4d} {cm[1][1]:4d}]]")


def main():
    """Main function to run BrainGNN Mixed experiment."""
    if not PYG_AVAILABLE:
        print("Error: PyTorch Geometric is required.")
        return
    
    # 路径配置
    participants_path = '/mnt/sda1/zhangyan/cpac_output/机器学习/人口学/ds002748_participants.tsv'
    fc_dir = '/mnt/sda1/zhangyan/cpac_output/feature/fmriprep/ds002748/'
    output_dir = '/mnt/sda1/zhangyan/cpac_output/机器学习/BrainGNN/result/hybrid/fmriprep/ds002748/'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {fc_dir}...")
    fc_paths, labels, subject_ids, demographics = load_fc_data(participants_path, fc_dir, n_nodes=116)
    
    if len(fc_paths) == 0:
        print("No data found.")
        return
    
    print("\nStarting CV with Hybrid demographics strategy...")
    fold_metrics, summary_metrics, predictions_df = run_braingnn_cv(
        fc_paths=fc_paths,
        labels=labels,
        subject_ids=subject_ids,
        demographics=demographics,
        
        n_splits=5,
        random_state=42,
        
        batch_size=8,
        lr=5e-4,              # 稍大一点的学习率以配合新特征
        weight_decay=1e-2,
        
        num_epochs=150,       # 增加最大轮数
        patience=30,          # 【恢复早停】设为 30
        
        dropout_rate=0.5,
        hidden_channels=32,
        conv_type='graph',
        use_simplified=True,
        add_noise=True,
        noise_std=0.01,       # 【降低噪声强度】从 0.05 降到 0.01
        
        output_dir=output_dir
    )
    
    print_results(fold_metrics, summary_metrics)
    
    # Save results
    results_file = os.path.join(output_dir, 'braingnn_results.json')
    with open(results_file, 'w') as f:
        json.dump({'fold_metrics': fold_metrics, 'summary_metrics': summary_metrics}, f, indent=2)
    
    predictions_df.to_csv(os.path.join(output_dir, 'braingnn_predictions.csv'), index=False)
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
