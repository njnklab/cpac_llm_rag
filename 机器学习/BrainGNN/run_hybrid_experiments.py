import os
import subprocess

datasets = ['KKI', 'NeuroIMAGE', 'OHSU', 'ds002748']
methods = ['fmriprep', 'deepprep']

def run_experiment(dataset, method):
    print(f"\n\n>>> RUNNING EXPERIMENT: {dataset} | {method} <<<")
    
    # Configure paths
    if dataset == 'ds002748':
        participants_path = '/mnt/sda1/zhangyan/cpac_output/机器学习/人口学/ds002748_participants.tsv'
    else:
        participants_path = f'/mnt/sda1/zhangyan/cpac_output/机器学习/人口学/{dataset}_phenotypic.csv'
        
    fc_dir = f'/mnt/sda1/zhangyan/cpac_output/feature/{method}/{dataset}/'
    output_dir = f'/mnt/sda1/zhangyan/cpac_output/机器学习/BrainGNN/result/hybrid/{method}/{dataset}/'
    
    # We need to modify train_braingnn.py temporarily or pass arguments
    # For simplicity, I'll create a temporary runner script for each
    runner_code = f"""
import sys
import os
# Add current dir to path
sys.path.append('/mnt/sda1/zhangyan/cpac_output/机器学习/BrainGNN')
from train_braingnn import run_braingnn_cv, print_results
from dataset_braingnn import load_fc_data
import json

participants_path = '{participants_path}'
fc_dir = '{fc_dir}'
output_dir = '{output_dir}'

os.makedirs(output_dir, exist_ok=True)

fc_paths, labels, subject_ids, demographics = load_fc_data(participants_path, fc_dir, n_nodes=116)

if len(fc_paths) > 0:
    fold_metrics, summary_metrics, predictions_df = run_braingnn_cv(
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
        hidden_channels=32,
        conv_type='graph',
        use_simplified=True,
        add_noise=True,
        noise_std=0.01,
        output_dir=output_dir
    )
    print_results(fold_metrics, summary_metrics)
    
    # Save results
    results_file = os.path.join(output_dir, 'braingnn_results.json')
    with open(results_file, 'w') as f:
        json.dump({{'fold_metrics': fold_metrics, 'summary_metrics': summary_metrics}}, f, indent=2)
    
    predictions_df.to_csv(os.path.join(output_dir, 'braingnn_predictions.csv'), index=False)
else:
    print(f"Skipping {{participants_path}} - no data found at {{fc_dir}}")
"""
    temp_script = f"/tmp/run_{dataset}_{method}.py"
    with open(temp_script, 'w') as f:
        f.write(runner_code)
    
    subprocess.run(['python3', temp_script])

for d in datasets:
    for m in methods:
        run_experiment(d, m)
