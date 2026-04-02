"""
FC矩阵分类实验主脚本
支持SVM和Random Forest，可选择是否使用ComBat
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import load_participants_metadata, find_all_fc_files
from utils.data_utils import load_fc_matrix, fc_to_feature_vector
from utils.metrics import calculate_all_metrics, aggregate_fold_results, print_metrics
from models.traditional_ml import get_classifiers

try:
    from neuroHarmonize import harmonizationLearn, harmonizationApply
    HAS_NEUROHARMONIZE = True
except ImportError:
    HAS_NEUROHARMONIZE = False
    warnings.warn("neuroHarmonize not found. ComBat functionality will be unavailable.")


def run_classification_experiment(
    participants_path: str,
    fc_dir: str,
    output_dir: str,
    use_combat: bool = True,
    n_splits: int = 5,
    random_state: int = 42
):
    """
    运行分类实验
    
    Args:
        participants_path: 人口学信息CSV路径
        fc_dir: FC矩阵目录
        output_dir: 输出目录
        use_combat: 是否使用ComBat
        n_splits: 交叉验证折数
        random_state: 随机种子
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if use_combat and not HAS_NEUROHARMONIZE:
        print("WARNING: ComBat requested but neuroHarmonize not available. Skipping ComBat.")
        use_combat = False
    
    print(f"Loading demographics from {participants_path}")
    participants_df = load_participants_metadata(participants_path)
    
    print(f"Searching for FC mats in {fc_dir}")
    fc_file_map = find_all_fc_files(fc_dir)
    print(f"Found {len(fc_file_map)} FC files in directory.")
    
    features, labels, valid_covars = [], [], []
    
    for _, row in participants_df.iterrows():
        sid = row['participant_id']
        match_file = fc_file_map.get(sid)
        if not match_file:
            for fid, fpath in fc_file_map.items():
                if sid in fid or fid in sid:
                    match_file = fpath
                    break
        
        if match_file:
            fc_matrix = load_fc_matrix(match_file)
            if fc_matrix is None:
                continue
                
            fc_vector = fc_to_feature_vector(fc_matrix, use_fisher_z=True)
            
            if np.isnan(fc_vector).any() or np.isinf(fc_vector).any():
                continue
                
            features.append(fc_vector)
            labels.append(row['group'])
            valid_covars.append({
                'SITE': row['Site'],
                'AGE': row['Age'],
                'GENDER': row['Gender'],
                'Participant_ID': sid
            })
                
    X = np.array(features)
    y = np.array(labels)
    covars_df = pd.DataFrame(valid_covars)
    
    print(f"\nFinal Matched Subjects: {len(X)}")
    print(f"Class distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    
    if len(X) == 0:
        print("No samples matched. Exiting.")
        return
        
    classifiers = get_classifiers()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    stratify_key = covars_df['SITE'].astype(str) + "_" + pd.Series(y).astype(str)
    
    results = {name: [] for name in classifiers}
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, stratify_key)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        covars_train = covars_df.iloc[train_idx].reset_index(drop=True)
        covars_test = covars_df.iloc[test_idx].reset_index(drop=True)
        
        # ComBat harmonization
        if use_combat:
            combat_c_train = covars_train[['SITE', 'AGE', 'GENDER']].copy()
            model, X_train_adj = harmonizationLearn(X_train, combat_c_train)
            combat_c_test = covars_test[['SITE', 'AGE', 'GENDER']].copy()
            X_test_adj = harmonizationApply(X_test, combat_c_test, model)
        else:
            X_train_adj, X_test_adj = X_train, X_test
            
        # Train and evaluate each classifier
        for name, clf in classifiers.items():
            clf.fit(X_train_adj, y_train)
            y_pred, y_prob = clf.predict(X_test_adj)
            
            metrics = calculate_all_metrics(y_test, y_pred, y_prob)
            results[name].append(metrics)
            
            print(f"{name} - Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1-score']:.4f}")
            
    # Save results
    suffix = 'combat' if use_combat else 'nocombat'
    summary_path = os.path.join(output_dir, f"classification_results_{suffix}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Results (ComBat={use_combat})\n{'='*50}\n")
        f.write("Pipeline: Fisher Z -> SelectKBest(k=200) -> StandardScaler -> Classifier\n")
        if use_combat:
            f.write("With ComBat harmonization\n\n")
        else:
            f.write("Without ComBat harmonization\n\n")
            
        for name in classifiers:
            f.write(f"\nModel: {name}\n{'-'*30}\n")
            summary = aggregate_fold_results(results[name])
            for key, value in summary.items():
                if 'Confusion Matrix' in key:
                    f.write(f"{key}: {value}\n")
                else:
                    f.write(f"{key}: {value:.4f}\n")
                
    print(f"\nFold completed. Results saved to {summary_path}")
    return results


if __name__ == '__main__':
    participants_path = '/mnt/sda1/zhangyan/cpac_output/机器学习/人口学/ADHD_all_phenotypic.csv'
    fc_dir = '/mnt/sda1/zhangyan/cpac_output/feature/deepprep'
    output_dir = '/mnt/sda1/zhangyan/cpac_output/机器学习/results/combat/svm_rf'
    
    print("="*60)
    print("Running Pipeline WITHOUT ComBat")
    print("="*60)
    run_classification_experiment(
        participants_path=participants_path,
        fc_dir=fc_dir,
        output_dir=output_dir,
        use_combat=False
    )
    
    if HAS_NEUROHARMONIZE:
        print("\n" + "="*60)
        print("Running Pipeline WITH ComBat")
        print("="*60)
        run_classification_experiment(
            participants_path=participants_path,
            fc_dir=fc_dir,
            output_dir=output_dir,
            use_combat=True
        )
    else:
        print("\nSkipping ComBat pipeline because neuroHarmonize is missing.")