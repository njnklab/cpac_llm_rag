import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score,
                             balanced_accuracy_score)
from tqdm import tqdm
import json

from model_brainnetcnn import BrainNetCNN, BrainNetCNNSimplified
from dataset_brainnetcnn import BrainNetCNNDataset, load_fc_data


def compute_class_weights(labels):
    """
    Compute class weights for imbalanced dataset.
    Formula: w_i = total_samples / (n_classes * n_samples_i)
    """
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
    
    for batch_data in train_loader:
        if len(batch_data) == 4:
            fc_tensors, labels, _, demo_tensors = batch_data
            demo_tensors = demo_tensors.to(device)
        else:
            fc_tensors, labels, _ = batch_data
            demo_tensors = None
            
        fc_tensors = fc_tensors.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(fc_tensors, demo=demo_tensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * fc_tensors.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
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
        for batch_data in loader:
            if len(batch_data) == 4:
                fc_tensors, labels, subject_ids, demo_tensors = batch_data
                demo_tensors = demo_tensors.to(device)
            else:
                fc_tensors, labels, subject_ids = batch_data
                demo_tensors = None
                
            fc_tensors = fc_tensors.to(device)
            labels = labels.to(device)
            
            outputs = model(fc_tensors, demo=demo_tensors)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * fc_tensors.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_subject_ids.extend(subject_ids)
    
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
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
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
    best_val_auc = 0.0
    best_model_state = None
    best_val_metrics = None
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_results = evaluate(model, val_loader, criterion, device)
        val_loss = val_results['loss']
        val_metrics = compute_metrics(val_results['y_true'], val_results['y_pred'], val_results['y_prob'])
        val_auc = val_metrics['auc']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            best_val_metrics = val_metrics
            best_val_metrics['val_loss'] = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val AUC={val_auc:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if save_path and best_model_state:
        torch.save(best_model_state, save_path)
    return best_model_state, best_val_metrics, history


def run_brainnetcnn_cv(
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
    use_simplified=False,
    add_noise=False,
    output_dir=None,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    labels = np.array(labels)
    n_nodes = 116
    demo_dim = demographics.shape[1] if demographics is not None else 0
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []
    all_test_results = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(fc_paths, labels)):
        print(f"\nFold {fold+1}/{n_splits}")
        train_val_labels = labels[train_val_idx]
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=random_state, stratify=train_val_labels)
        
        def get_split_data(indices):
            paths = [fc_paths[i] for i in indices]
            lbls = labels[indices].tolist()
            subs = [subject_ids[i] for i in indices]
            dems = demographics[indices] if demographics is not None else None
            return paths, lbls, subs, dems

        tr_paths, tr_labels, tr_subs, tr_dems = get_split_data(train_idx)
        va_paths, va_labels, va_subs, va_dems = get_split_data(val_idx)
        te_paths, te_labels, te_subs, te_dems = get_split_data(test_idx)
        
        train_dataset = BrainNetCNNDataset(tr_paths, tr_labels, tr_subs, tr_dems, n_nodes=n_nodes, add_noise=add_noise)
        val_dataset = BrainNetCNNDataset(va_paths, va_labels, va_subs, va_dems, n_nodes=n_nodes)
        test_dataset = BrainNetCNNDataset(te_paths, te_labels, te_subs, te_dems, n_nodes=n_nodes)
        
        class_counts = np.bincount(tr_labels)
        weights = [1.0/class_counts[int(l)] for l in tr_labels]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(tr_labels), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        model = (BrainNetCNNSimplified if use_simplified else BrainNetCNN)(n_nodes=n_nodes, demo_dim=demo_dim, dropout_rate=dropout_rate).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_state, val_m, _ = train_fold(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience)
        
        model.load_state_dict(best_state)
        test_res = evaluate(model, test_loader, criterion, device)
        test_m = compute_metrics(test_res['y_true'], test_res['y_pred'], test_res['y_prob'])
        test_m['fold'] = fold + 1
        
        print(f"Fold {fold+1} Accuracy: {test_m['accuracy']:.4f}, AUC: {test_m['auc']:.4f}")
        fold_metrics.append(test_m)
        
        for i, sid in enumerate(test_res['subject_ids']):
            all_test_results.append({'fold': fold+1, 'subject_id': sid, 'y_true': int(test_res['y_true'][i]), 
                                     'y_pred': int(test_res['y_pred'][i]), 'y_prob': float(test_res['y_prob'][i])})

    summary_metrics = {}
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        vals = [m[k] for m in fold_metrics]
        summary_metrics[f'{k}_mean'] = np.mean(vals)
        summary_metrics[f'{k}_std'] = np.std(vals)
    
    return fold_metrics, summary_metrics, pd.DataFrame(all_test_results)


def main():
    participants_path = '/mnt/sda1/zhangyan/cpac_output/机器学习/人口学/ds002748_participants.tsv'
    fc_dir = '/mnt/sda1/zhangyan/cpac_output/feature/fmriprep/ds002748/'
    output_dir = '/mnt/sda1/zhangyan/cpac_output/机器学习/BrainNetCNN/result/hybrid/fmriprep/ds002748/'
    os.makedirs(output_dir, exist_ok=True)
    
    fc_paths, labels, subject_ids, demographics = load_fc_data(participants_path, fc_dir)
    fold_metrics, summary_m, pred_df = run_brainnetcnn_cv(
        fc_paths, labels, subject_ids, demographics, 
        batch_size=8, lr=1e-4, weight_decay=1e-2, num_epochs=100, patience=30, 
        dropout_rate=0.5, use_simplified=True, add_noise=True, output_dir=output_dir
    )
    
    with open(os.path.join(output_dir, 'brainnetcnn_results.json'), 'w') as f:
        json.dump({'fold_metrics': fold_metrics, 'summary_metrics': summary_m}, f, indent=2)
    pred_df.to_csv(os.path.join(output_dir, 'brainnetcnn_predictions.csv'), index=False)
    print(f"\nSaved hybrid results to {output_dir}")


if __name__ == "__main__":
    main()
