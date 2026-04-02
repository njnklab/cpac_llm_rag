import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score

try:
    from neuroHarmonize import harmonizationLearn, harmonizationApply
    HAS_NEUROHARMONIZE = True
except ImportError:
    HAS_NEUROHARMONIZE = False
    warnings.warn("neuroHarmonize not found. Please activate your zy environment to use ComBat.")

def load_participants_metadata(file_path):
    df = pd.read_csv(file_path)
    if 'ScanDir ID' in df.columns:
        id_col = 'ScanDir ID'
    elif 'participant_id' in df.columns:
        id_col = 'participant_id'
    else:
        id_col = [c for c in df.columns if 'ID' in c.upper()][0]
        
    df['participant_id'] = df[id_col].astype(str).str.replace('sub-', '', regex=False)
    
    if 'DX' in df.columns:
        # 0 = Control (label 0), !=0 = ADHD (label 1)
        df['group'] = df['DX'].apply(lambda x: 0 if x == 0 else 1)
    else:
        raise ValueError("Cannot find DX column.")
        
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
        'Gender': pd.to_numeric(df['Gender'], errors='coerce') if 'Gender' in df.columns else 0
    }
    
    df_clean = pd.DataFrame(data_dict).dropna(subset=['participant_id', 'group', 'Site'])
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].mean())
    df_clean['Gender'] = df_clean['Gender'].fillna(0)
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

def run_classification_experiment(use_combat=True):
    participants_path = '/mnt/sda1/zhangyan/cpac_output/机器学习/人口学/ADHD_all_phenotypic.csv'
    fc_dir = '/mnt/sda1/zhangyan/cpac_output/feature/deepprep'
    output_dir = '/mnt/sda1/zhangyan/cpac_output/机器学习/combat/svm_rf/deepprep'
    
    os.makedirs(output_dir, exist_ok=True)
    
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
                    'SITE': row['Site'],
                    'AGE': row['Age'],
                    'GENDER': row['Gender'],
                    'Participant_ID': sid
                })
            except Exception as e:
                continue
                
    X = np.array(features)
    y = np.array(labels)
    covars_df = pd.DataFrame(valid_covars)
    print(f"\nFinal Matched Subjects: {len(X)}")
    print(f"Class distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    
    if len(X) == 0:
        print("No samples matched. Exiting.")
        return
        
    classifiers = {
        'Linear SVM': SVC(kernel='linear', C=0.01, class_weight='balanced', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, max_features='sqrt', min_samples_leaf=3, class_weight='balanced', random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratify_key = covars_df['SITE'].astype(str) + "_" + pd.Series(y).astype(str)
    
    results = {name: [] for name in classifiers}
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, stratify_key)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        covars_train = covars_df.iloc[train_idx].reset_index(drop=True)
        covars_test = covars_df.iloc[test_idx].reset_index(drop=True)
        
        # 1. ComBat
        if use_combat and HAS_NEUROHARMONIZE:
            combat_c_train = covars_train[['SITE', 'AGE', 'GENDER']].copy()
            model, X_train_adj = harmonizationLearn(X_train, combat_c_train)
            combat_c_test = covars_test[['SITE', 'AGE', 'GENDER']].copy()
            X_test_adj = harmonizationApply(X_test, combat_c_test, model)
        else:
            X_train_adj, X_test_adj = X_train, X_test
            
        # 2. SelectKBest (k=200)
        selector = SelectKBest(score_func=f_classif, k=200)
        X_train_sel = selector.fit_transform(X_train_adj, y_train)
        X_test_sel = selector.transform(X_test_adj)
            
        # 3. Predict & Eval
        for name, clf in classifiers.items():
            # 3.1 StandardScaler
            scaler = StandardScaler()
            X_train_final = scaler.fit_transform(X_train_sel)
            X_test_final = scaler.transform(X_test_sel)
                
            clf.fit(X_train_final, y_train)
            y_pred = clf.predict(X_test_final)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            b_acc = balanced_accuracy_score(y_test, y_pred)
            mf1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            wf1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results[name].append({
                'Accuracy': acc, 'Precision': prec, 'Recall': rec, 
                'F1-score': f1, 'Balanced Accuracy': b_acc, 
                'Macro-F1': mf1, 'Weighted-F1': wf1
            })
            
    suffix = 'combat' if use_combat else 'nocombat'
    summary_path = os.path.join(output_dir, f"classification_results_{suffix}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Results (ComBat={use_combat})\n{'='*30}\n")
        f.write("Pipeline: Fisher Z -> ComBat -> SelectKBest(k=200) -> StandardScaler -> Classifier\n\n")
        for name in classifiers:
            f.write(f"Model: {name}\n{'-'*20}\n")
            df_res = pd.DataFrame(results[name])
            m = df_res.mean()
            s = df_res.std()
            for col in df_res.columns:
                f.write(f"{col}: {m[col]:.4f} +- {s[col]:.4f}\n")
            f.write("\n")
                
    print(f"Fold completed. Results saved to {summary_path}")

if __name__ == '__main__':
    print("=== Running Pipeline without ComBat ===")
    run_classification_experiment(use_combat=False)
    
    if HAS_NEUROHARMONIZE:
        print("\n=== Running Pipeline WITH ComBat ===")
        run_classification_experiment(use_combat=True)
    else:
        print("\nSkipping ComBat pipeline because neuroHarmonize is missing.")
