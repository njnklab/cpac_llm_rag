#!/usr/bin/env python3
"""
结果分析脚本
汇总和可视化所有实验结果
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_all_results(results_dir='../results'):
    """加载所有实验结果"""
    all_results = []
    
    # 遍历所有结果目录
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.txt') or file.endswith('.csv'):
                filepath = os.path.join(root, file)
                print(f"Found result: {filepath}")
    
    return all_results

def create_comparison_plots():
    """创建对比图表"""
    # 读取对比数据
    df = pd.read_csv('../docs/total_metrics_comparison.csv')
    
    # 设置图表风格
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy对比
    sns.boxplot(data=df, x='Framework', y='Accuracy', ax=axes[0,0])
    axes[0,0].set_title('Accuracy by Framework')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # F1-Score对比
    sns.boxplot(data=df, x='Framework', y='F1-Score', ax=axes[0,1])
    axes[0,1].set_title('F1-Score by Framework')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Model性能对比
    sns.barplot(data=df, x='Model', y='Accuracy', ax=axes[1,0])
    axes[1,0].set_title('Accuracy by Model')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Dataset性能对比
    sns.barplot(data=df, x='Dataset', y='Accuracy', ax=axes[1,1])
    axes[1,1].set_title('Accuracy by Dataset')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('../docs/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved to docs/performance_comparison.png")

def print_summary():
    """打印结果摘要"""
    df = pd.read_csv('../docs/total_metrics_comparison.csv')
    
    print("\n" + "="*60)
    print("实验结果摘要")
    print("="*60)
    
    print(f"\n总实验数: {len(df)}")
    print(f"框架数: {df['Framework'].nunique()}")
    print(f"数据集数: {df['Dataset'].nunique()}")
    print(f"模型数: {df['Model'].nunique()}")
    
    print("\n各框架最佳Accuracy:")
    for framework in df['Framework'].unique():
        best = df[df['Framework'] == framework]['Accuracy'].max()
        print(f"  {framework}: {best:.4f}")
    
    print("\n各模型平均Accuracy:")
    for model in df['Model'].unique():
        avg = df[df['Model'] == model]['Accuracy'].mean()
        print(f"  {model}: {avg:.4f}")

if __name__ == '__main__':
    print_summary()
    
    try:
        create_comparison_plots()
    except Exception as e:
        print(f"\n无法生成图表: {e}")
        print("请确保已安装matplotlib和seaborn")