#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
四个预处理框架的QC指标对比箱线图
对比: CPAC-Default, CPAC-LLM, fMRIPrep, DeepPrep
"""

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 基础路径
ANALYSIS_DIR = "/mnt/sda1/zhangyan/cpac_output/质量控制分析"
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "画图")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 四个框架的配置
FRAMEWORKS = {
    'CPAC-Default': {
        'path': 'cpac-default',
        'pattern': '*-default-qc-all.csv',
        'color': '#1f77b4',  # 蓝色
    },
    'CPAC-LLM': {
        'path': 'cpac-llm',
        'pattern': '*-llm-qc-all.csv',
        'color': '#ff7f0e',  # 橙色
    },
    'fMRIPrep': {
        'path': 'fmriprep',
        'pattern': '*-fmriprep-qc-all.csv',
        'color': '#2ca02c',  # 绿色
    },
    'DeepPrep': {
        'path': 'deepprep',
        'pattern': '*-deepprep-qc-all.csv',
        'color': '#d62728',  # 红色
    }
}

# 列名映射（统一各框架的列名）
COLUMN_MAPPING = {
    # 功能像指标
    'MeanFD_Power': 'MeanFD',
    'MeanFD': 'MeanFD',
    'MeanDVARS': 'DVARS',
    'MeanStdDVARS': 'DVARS',
    'boldSnr': 'tSNR',
    'tSNR_mean': 'tSNR',
    # 结构像指标
    'CJV': 'CJV',
    'EFC': 'EFC',
    'WM2MAX': 'WM2MAX',
}


def load_framework_data(framework_name, config):
    """加载某个框架的所有数据"""
    framework_path = os.path.join(ANALYSIS_DIR, config['path'])
    csv_files = glob.glob(os.path.join(framework_path, config['pattern']))
    
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # 提取数据集名称
            dataset = os.path.basename(csv_file).replace('-qc-all.csv', '')
            df['Dataset'] = dataset
            df['Framework'] = framework_name
            all_data.append(df)
        except Exception as e:
            print(f"  警告: 无法读取 {csv_file}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None


def standardize_columns(df):
    """标准化列名"""
    df = df.copy()
    
    # 创建新列，统一名称
    new_columns = {}
    for col in df.columns:
        if col in COLUMN_MAPPING:
            new_columns[col] = COLUMN_MAPPING[col]
    
    # 重命名列
    df = df.rename(columns=new_columns)
    
    return df


def plot_box_comparison(data_dict, metrics, title_prefix, output_name):
    """绘制多个指标的箱线图对比"""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics + 1, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    framework_names = list(data_dict.keys())
    colors = [FRAMEWORKS[name]['color'] for name in framework_names]
    
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        
        # 收集各框架的数据
        all_values = []
        for framework in framework_names:
            df = data_dict[framework]
            if metric in df.columns:
                values = df[metric].dropna().values
                all_values.append(values)
            else:
                all_values.append([])
        
        # 过滤掉空数据
        non_empty = [(i, v) for i, v in enumerate(all_values) if len(v) > 0]
        if not non_empty:
            ax.set_visible(False)
            continue
        
        positions = list(range(len(framework_names)))
        
        # 画箱线图
        box_plot = ax.boxplot(
            all_values,
            positions=positions,
            widths=0.6,
            showmeans=True,
            patch_artist=True,
            medianprops=dict(color='black', linewidth=2),
            meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=6)
        )
        
        # 设置箱线图颜色
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 添加散点
        jitter = 0.08
        for pos, values in enumerate(all_values):
            if len(values) > 0:
                x_jittered = np.random.normal(loc=pos, scale=jitter, size=len(values))
                ax.scatter(x_jittered, values, alpha=0.4, color='black', s=8, zorder=3)
        
        # 设置标签
        ax.set_xticks(positions)
        ax.set_xticklabels(framework_names, rotation=30, ha='right', fontsize=9)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_xlim(-0.5, len(framework_names) - 0.5)
    
    fig.suptitle(title_prefix, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, output_name)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {output_path}")


def calculate_summary_stats(data_dict, metrics):
    """计算各框架的汇总统计"""
    summary_rows = []
    
    for framework_name, df in data_dict.items():
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].dropna()
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


def main():
    print("=" * 60)
    print("四个预处理框架 QC 指标对比")
    print("=" * 60)
    
    # 1. 加载所有数据
    print("\n1. 加载数据...")
    data_dict = {}
    for framework_name, config in FRAMEWORKS.items():
        print(f"  {framework_name}...", end=' ')
        df = load_framework_data(framework_name, config)
        if df is not None:
            df = standardize_columns(df)
            data_dict[framework_name] = df
            print(f"共 {len(df)} 条记录")
        else:
            print("无数据")
    
    # 2. 定义指标
    # 结构像指标
    struct_metrics = ['CJV', 'EFC', 'WM2MAX']
    # 功能像指标
    func_metrics = ['MeanFD', 'DVARS', 'tSNR']
    
    # 3. 绘制结构像指标对比
    print("\n2. 绘制结构像指标...")
    plot_box_comparison(
        data_dict, 
        struct_metrics, 
        'Structural QC Metrics Comparison (CJV, EFC, WM2MAX)',
        '01_structural_metrics_comparison.png'
    )
    
    # 4. 绘制功能像指标对比 (MeanFD & DVARS)
    print("\n3. 绘制功能像指标 (MeanFD & DVARS)...")
    plot_box_comparison(
        data_dict,
        ['MeanFD', 'DVARS'],
        'Functional QC Metrics Comparison (MeanFD, DVARS)',
        '02_functional_fd_dvars_comparison.png'
    )
    
    # 5. 单独绘制 tSNR（可能有极端值）
    print("\n4. 绘制 tSNR 对比...")
    fig, ax = plt.subplots(figsize=(7, 5))
    
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
    
    # 画箱线图
    box_plot = ax.boxplot(
        all_tsnr_values,
        positions=positions,
        widths=0.6,
        showmeans=True,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=6)
    )
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 添加散点
    jitter = 0.08
    for pos, values in enumerate(all_tsnr_values):
        if len(values) > 0:
            x_jittered = np.random.normal(loc=pos, scale=jitter, size=len(values))
            ax.scatter(x_jittered, values, alpha=0.4, color='black', s=8, zorder=3)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(framework_names, rotation=30, ha='right')
    ax.set_title('tSNR Comparison Across Frameworks', fontsize=12, fontweight='bold')
    ax.set_ylabel('tSNR', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    tsnr_path = os.path.join(OUTPUT_DIR, '03_tsnr_comparison.png')
    fig.savefig(tsnr_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {tsnr_path}")
    
    # 6. 计算并保存汇总统计
    print("\n5. 生成汇总统计...")
    all_metrics = struct_metrics + func_metrics
    summary_df = calculate_summary_stats(data_dict, all_metrics)
    summary_path = os.path.join(OUTPUT_DIR, 'qc_summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"  已保存: {summary_path}")
    
    # 显示汇总
    print("\n" + "=" * 60)
    print("汇总统计表（按框架分组）:")
    print("=" * 60)
    for framework in framework_names:
        print(f"\n{framework}:")
        fw_summary = summary_df[summary_df['Framework'] == framework]
        for _, row in fw_summary.iterrows():
            print(f"  {row['Metric']:12s}: Mean={row['Mean']:.4f}, Std={row['Std']:.4f}, N={row['N']}")
    
    print("\n" + "=" * 60)
    print(f"✓ 所有结果已保存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
