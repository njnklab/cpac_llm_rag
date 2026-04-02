#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 全局字体设置 - 加粗加大
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['font.weight'] = 'bold'

ANALYSIS_DIR = "/mnt/sda1/zhangyan/cpac_output/质量控制分析"
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "画图")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAMEWORKS = {
    'CPAC-Default': {
        'path': 'cpac-default',
        'patterns': ['ds002748-default-qc-all.csv', 'kki-default-qc-all.csv', 
                     'neuroimage-default-qc-all.csv', 'ohsu-default-qc-all.csv'],
        'color': '#1a3353',
    },
    'CPAC-LLM': {
        'path': 'cpac-llm',
        'patterns': ['ds002748-llm-qc-all.csv', 'kki-llm-qc-all.csv', 
                     'neuroimage-llm-qc-all.csv', 'ohsu-llm-qc-all.csv'],
        'color': '#2d5a4a',
    },
    'fMRIPrep': {
        'path': 'fmriprep',
        'patterns': ['ds002748-fmriprep-qc-all.csv', 'KKI-fmriprep-qc-all.csv', 
                     'NeuroIMAGE-fmriprep-qc-all.csv', 'OHSU-fmriprep-qc-all.csv'],
        'color': '#4a2d5a',
    },
    'DeepPrep': {
        'path': 'deepprep',
        'patterns': ['ds002748-deepprep-qc-all.csv', 'KKI-deepprep-qc-all.csv', 
                     'NeuroIMAGE-deepprep-qc-all.csv', 'OHSU-deepprep-qc-all.csv'],
        'color': '#5a3d2d',
    }
}

DATASETS = {
    'ds002748': ['ds002748-default-qc-all.csv', 'ds002748-llm-qc-all.csv', 
                 'ds002748-fmriprep-qc-all.csv', 'ds002748-deepprep-qc-all.csv'],
    'KKI': ['kki-default-qc-all.csv', 'kki-llm-qc-all.csv', 
            'KKI-fmriprep-qc-all.csv', 'KKI-deepprep-qc-all.csv'],
    'NeuroIMAGE': ['neuroimage-default-qc-all.csv', 'neuroimage-llm-qc-all.csv', 
                   'NeuroIMAGE-fmriprep-qc-all.csv', 'NeuroIMAGE-deepprep-qc-all.csv'],
    'OHSU': ['ohsu-default-qc-all.csv', 'ohsu-llm-qc-all.csv', 
             'OHSU-fmriprep-qc-all.csv', 'OHSU-deepprep-qc-all.csv']
}

COLUMN_MAPPING = {
    'MeanFD_Power': 'MeanFD',
    'MeanFD': 'MeanFD',
    'MeanDVARS': 'DVARS',
    'MeanStdDVARS': 'DVARS',
    'boldSnr': 'tSNR',
    'tSNR_mean': 'tSNR',
    'CJV': 'CJV',
    'EFC': 'EFC',
    'WM2MAX': 'WM2MAX',
}


def load_dataset_data(dataset_name):
    dataset_files = DATASETS[dataset_name]
    data_dict = {}
    
    for framework_name, config in FRAMEWORKS.items():
        framework_path = os.path.join(ANALYSIS_DIR, config['path'])
        
        for pattern in config['patterns']:
            if pattern in dataset_files:
                csv_file = os.path.join(framework_path, pattern)
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        df = standardize_columns(df)
                        data_dict[framework_name] = df
                        break
                    except Exception as e:
                        print(f"    警告: 无法读取 {csv_file}: {e}")
    
    return data_dict


def standardize_columns(df):
    df = df.copy()
    new_columns = {}
    for col in df.columns:
        if col in COLUMN_MAPPING:
            new_columns[col] = COLUMN_MAPPING[col]
    df = df.rename(columns=new_columns)
    return df


def get_nice_y_range(all_values_list, metric_name=None):
    """获取美观的Y轴范围"""
    all_data = []
    for values in all_values_list:
        if len(values) > 0:
            all_data.extend(values)
    
    if not all_data:
        return 0, 1
    
    y_min_raw, y_max_raw = np.min(all_data), np.max(all_data)
    
    # 根据指标类型设置合理的范围和刻度
    if metric_name == 'CJV':
        y_min = max(0.3, np.floor(y_min_raw * 20) / 20 - 0.05)
        y_max = min(0.8, np.ceil(y_max_raw * 20) / 20 + 0.05)
        y_ticks = np.arange(y_min, y_max + 0.05, 0.1)
    elif metric_name == 'EFC':
        y_min = max(0.2, np.floor(y_min_raw * 20) / 20 - 0.05)
        y_max = min(0.6, np.ceil(y_max_raw * 20) / 20 + 0.05)
        y_ticks = np.arange(y_min, y_max + 0.05, 0.1)
    elif metric_name == 'WM2MAX':
        y_min = max(0.6, np.floor(y_min_raw * 10) / 10 - 0.05)
        y_max = min(1.0, np.ceil(y_max_raw * 10) / 10 + 0.05)
        y_ticks = np.arange(y_min, y_max + 0.05, 0.1)
    elif metric_name == 'MeanFD':
        # 根据实际数据范围动态设置Y轴
        all_values_array = np.array(all_data)
        y_max_raw = np.max(all_values_array)
        
        # 如果最大值小于0.5，Y轴上限设为0.5
        if y_max_raw < 0.5:
            y_max = 0.5
            tick_step = 0.1
        # 如果最大值在0.5-1.0之间，Y轴上限设为1.0
        elif y_max_raw < 1.0:
            y_max = 1.0
            tick_step = 0.2
        # 如果最大值在1.0-2.0之间，Y轴上限设为2.0
        elif y_max_raw < 2.0:
            y_max = 2.0
            tick_step = 0.5
        # 否则使用IQR方法处理离群值
        else:
            q1, q3 = np.percentile(all_values_array, [25, 75])
            iqr = q3 - q1
            upper_fence = q3 + 1.5 * iqr
            display_max = max(upper_fence, q3 * 1.2, 3.0)
            p99 = np.percentile(all_values_array, 99)
            y_max = min(display_max, max(p99 * 1.1, 3.0))
            y_max = np.ceil(y_max / 0.5) * 0.5
            tick_step = 0.5
        
        y_min = 0
        y_ticks = np.arange(0, y_max + tick_step/2, tick_step)
    elif metric_name == 'DVARS':
        y_min = max(0.5, np.floor(y_min_raw * 2) / 2 - 0.1)
        y_max = min(3.0, np.ceil(y_max_raw * 2) / 2 + 0.1)
        y_ticks = np.arange(y_min, y_max + 0.1, 0.3)
    elif metric_name == 'tSNR':
        y_min = max(0, np.floor(y_min_raw / 10) * 10 - 10)
        y_max = min(200, np.ceil(y_max_raw / 10) * 10 + 10)
        tick_step = 20 if (y_max - y_min) > 80 else 10
        y_ticks = np.arange(y_min, y_max + tick_step/2, tick_step)
    else:
        range_val = y_max_raw - y_min_raw
        y_min = y_min_raw - range_val * 0.05
        y_max = y_max_raw + range_val * 0.05
        y_ticks = None
    
    return y_min, y_max, y_ticks


def plot_box_comparison(data_dict, metrics, title, output_path):
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5.5 * n_metrics + 0.5, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    framework_names = list(data_dict.keys())
    colors = [FRAMEWORKS[name]['color'] for name in framework_names]
    
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        
        all_values = []
        for framework in framework_names:
            df = data_dict[framework]
            if metric in df.columns:
                values = df[metric].dropna().values
                all_values.append(values)
            else:
                all_values.append([])
        
        non_empty = [(i, v) for i, v in enumerate(all_values) if len(v) > 0]
        if not non_empty:
            ax.set_visible(False)
            continue
        
        positions = list(range(len(framework_names)))
        
        box_plot = ax.boxplot(
            all_values,
            positions=positions,
            widths=0.6,
            showmeans=True,
            patch_artist=True,
            medianprops=dict(color='white', linewidth=2.5),
            meanprops=dict(marker='D', markeredgecolor='white', markerfacecolor='lightgray', markersize=8)
        )
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)
        
        jitter = 0.08
        for pos, values in enumerate(all_values):
            if len(values) > 0:
                x_jittered = np.random.normal(loc=pos, scale=jitter, size=len(values))
                ax.scatter(x_jittered, values, alpha=0.25, color='gray', s=8, zorder=1)
        
        y_min, y_max, y_ticks = get_nice_y_range(all_values, metric)
        ax.set_ylim(y_min, y_max)
        if y_ticks is not None and len(y_ticks) > 1:
            ax.set_yticks(y_ticks)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        ax.set_xticks(positions)
        ax.set_xticklabels(framework_names, rotation=30, ha='right', fontweight='bold')
        ax.set_ylabel(metric, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_xlim(-0.5, len(framework_names) - 0.5)
        
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1, length=3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"    已保存: {output_path}")


def plot_tsnr_comparison(data_dict, title, output_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    framework_names = list(data_dict.keys())
    colors = [FRAMEWORKS[name]['color'] for name in framework_names]
    all_tsnr_values = []
    
    for framework in framework_names:
        df = data_dict[framework]
        if 'tSNR' in df.columns:
            values = df['tSNR'].dropna().values
            all_tsnr_values.append(values)
        else:
            all_tsnr_values.append([])
    
    positions = list(range(len(framework_names)))
    
    box_plot = ax.boxplot(
        all_tsnr_values,
        positions=positions,
        widths=0.6,
        showmeans=True,
        patch_artist=True,
        medianprops=dict(color='white', linewidth=2.5),
        meanprops=dict(marker='D', markeredgecolor='white', markerfacecolor='lightgray', markersize=8)
    )
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    
    jitter = 0.08
    for pos, values in enumerate(all_tsnr_values):
        if len(values) > 0:
            x_jittered = np.random.normal(loc=pos, scale=jitter, size=len(values))
            ax.scatter(x_jittered, values, alpha=0.25, color='gray', s=8, zorder=1)
    
    y_min, y_max, y_ticks = get_nice_y_range(all_tsnr_values, 'tSNR')
    ax.set_ylim(y_min, y_max)
    if y_ticks is not None and len(y_ticks) > 1:
        ax.set_yticks(y_ticks)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_xticks(positions)
    ax.set_xticklabels(framework_names, rotation=30, ha='right', fontweight='bold')
    ax.set_ylabel('tSNR', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"    已保存: {output_path}")


def calculate_summary_stats(data_dict, metrics):
    summary_rows = []
    
    for framework_name, df in data_dict.items():
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    row = {
                        'Framework': framework_name,
                        'Metric': metric,
                        'N': len(values),
                        'Mean': values.mean(),
                        'Std': values.std(),
                        'Median': values.median(),
                        'Min': values.min(),
                        'Max': values.max(),
                    }
                    summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def process_dataset(dataset_name):
    print(f"\n  数据集: {dataset_name}")
    print("  " + "-" * 50)
    
    dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    data_dict = load_dataset_data(dataset_name)
    
    if not data_dict:
        print(f"  警告: {dataset_name} 没有数据")
        return
    
    print(f"    加载了 {len(data_dict)} 个框架的数据")
    for fw, df in data_dict.items():
        print(f"      {fw}: {len(df)} 条记录")
    
    struct_metrics = ['CJV', 'EFC', 'WM2MAX']
    func_metrics = ['MeanFD', 'DVARS']
    all_metrics = struct_metrics + func_metrics + ['tSNR']
    
    plot_box_comparison(
        data_dict, 
        struct_metrics, 
        f'{dataset_name} - Structural QC Metrics',
        os.path.join(dataset_dir, '01_structural_metrics.png')
    )
    
    plot_box_comparison(
        data_dict,
        func_metrics,
        f'{dataset_name} - Functional QC Metrics',
        os.path.join(dataset_dir, '02_functional_fd_dvars.png')
    )
    
    plot_tsnr_comparison(
        data_dict,
        f'{dataset_name} - tSNR Comparison',
        os.path.join(dataset_dir, '03_tsnr.png')
    )
    
    summary_df = calculate_summary_stats(data_dict, all_metrics)
    summary_path = os.path.join(dataset_dir, 'summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"    已保存统计: summary_statistics.csv")
    
    return summary_df


def main():
    print("=" * 60)
    print("四个预处理框架 QC 指标对比 - 按数据集分组")
    print("=" * 60)
    
    all_summaries = {}
    
    for dataset_name in DATASETS.keys():
        summary = process_dataset(dataset_name)
        if summary is not None:
            all_summaries[dataset_name] = summary
    
    print("\n" + "=" * 60)
    print(f"✓ 所有结果已保存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
