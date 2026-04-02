import os
import subprocess

datasets = ['KKI', 'NeuroIMAGE', 'OHSU', 'ds002748']
methods = ['fmriprep', 'deepprep']

def run_experiment(dataset, method):
    print(f"\n\n>>> RUNNING BRAINNETCNN HYBRID EXPERIMENT: {dataset} | {method} <<<")
    
    if dataset == 'ds002748':
        participants_path = '/mnt/sda1/zhangyan/cpac_output/机器学习/人口学/ds002748_participants.tsv'
    else:
        participants_path = f'/mnt/sda1/zhangyan/cpac_output/机器学习/人口学/{dataset}_phenotypic.csv'
        
    fc_dir = f'/mnt/sda1/zhangyan/cpac_output/feature/{method}/{dataset}/'
    output_dir = f'/mnt/sda1/zhangyan/cpac_output/机器学习/BrainNetCNN/result/hybrid/{method}/{dataset}/'
    
    # Check if data exists
    if not os.path.exists(fc_dir) or not os.listdir(fc_dir):
        print(f"Skipping {dataset} | {method} - No FC files found.")
        return

    runner_code = f"""
import sys
import os
import json
import pandas as pd
import numpy as np

# Add current dir to path
sys.path.append('/mnt/sda1/zhangyan/cpac_output/机器学习/BrainNetCNN')
from train_brainnetcnn import run_brainnetcnn_cv
from dataset_brainnetcnn import load_fc_data

participants_path = '{participants_path}'
fc_dir = '{fc_dir}'
output_dir = '{output_dir}'

os.makedirs(output_dir, exist_ok=True)

try:
    fc_paths, labels, subject_ids, demographics = load_fc_data(participants_path, fc_dir, n_nodes=116)

    if len(fc_paths) > 0:
        fold_metrics, summary_metrics, predictions_df = run_brainnetcnn_cv(
            fc_paths=fc_paths,
            labels=labels,
            subject_ids=subject_ids,
            demographics=demographics,
            n_splits=5,
            random_state=42,
            batch_size=8,
            lr=5e-4,
            weight_decay=1e-2,
            num_epochs=150,
            patience=30,
            dropout_rate=0.5,
            use_simplified=True,
            add_noise=True,
            output_dir=output_dir
        )
        
        # Save results
        results_file = os.path.join(output_dir, 'brainnetcnn_results.json')
        with open(results_file, 'w') as f:
            json.dump({{'fold_metrics': fold_metrics, 'summary_metrics': summary_metrics}}, f, indent=2)
        
        predictions_df.to_csv(os.path.join(output_dir, 'brainnetcnn_predictions.csv'), index=False)
        print(f"SUCCESS: Result saved to {{results_file}}")
    else:
        print(f"No valid data found for {dataset}")
except Exception as e:
    print(f"Error running experiment for {dataset}: {{e}}")
"""
    temp_script = f"/tmp/run_bnc_{dataset}_{method}.py"
    with open(temp_script, 'w') as f:
        f.write(runner_code)
    
    subprocess.run(['python3', temp_script])

for d in datasets:
    for m in methods:
        run_experiment(d, m)
