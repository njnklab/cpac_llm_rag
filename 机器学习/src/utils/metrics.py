"""评估指标计算工具"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, roc_auc_score
)

def calculate_all_metrics(y_true, y_pred, y_prob=None):
    """
    计算所有评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（用于计算AUC），可选
        
    Returns:
        dict: 包含所有指标的字典
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Macro-F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Weighted-F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Specificity': specificity,
        'Confusion Matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
    }
    
    if y_prob is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['AUC'] = 0.0
    
    return metrics


def print_metrics(metrics, title="Metrics"):
    """格式化打印指标"""
    print(f"\n{'='*40}")
    print(f"{title}")
    print(f"{'='*40}")
    for key, value in metrics.items():
        if key != 'Confusion Matrix':
            print(f"{key}: {value:.4f}")
    print(f"Confusion Matrix: {metrics['Confusion Matrix']}")


def aggregate_fold_results(fold_results):
    """
    聚合多折交叉验证结果
    
    Args:
        fold_results: list of dict，每折的指标
        
    Returns:
        dict: 包含mean和std的字典
    """
    if not fold_results:
        return {}
    
    # 收集所有数值指标
    numeric_keys = [k for k in fold_results[0].keys() 
                    if k != 'Confusion Matrix' and isinstance(fold_results[0][k], (int, float))]
    
    summary = {}
    for key in numeric_keys:
        values = [r[key] for r in fold_results]
        summary[f'{key} (mean)'] = np.mean(values)
        summary[f'{key} (std)'] = np.std(values)
    
    # 汇总混淆矩阵
    total_cm = [[0, 0], [0, 0]]
    for r in fold_results:
        cm = r.get('Confusion Matrix', [[0, 0], [0, 0]])
        total_cm[0][0] += cm[0][0]
        total_cm[0][1] += cm[0][1]
        total_cm[1][0] += cm[1][0]
        total_cm[1][1] += cm[1][1]
    summary['Confusion Matrix (sum)'] = total_cm
    
    return summary