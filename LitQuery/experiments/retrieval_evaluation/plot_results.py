#!/usr/bin/env python3
"""
检索策略对比实验 - 画图脚本 (Part 2)
读取 run_experiment.py 生成的中间结果，绘制对比图表

配置参数请在 CONFIG 字典中修改
"""

import os
import sys
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

# ==================== 用户配置区域 ====================

CONFIG = {
    # 输入目录（应与 run_experiment.py 的输出目录一致）
    "input_dir": "/home/a001/zhangyan/LitQuery/检索方案评估/result",
    
    # 中间结果文件名（由 run_experiment.py 生成）
    "intermediate_results": "all_results_intermediate.json",
    
    "output_dir": "/home/a001/zhangyan/LitQuery/检索方案评估/plot",
    
    # 输出图表文件名
    "output_mean_plot_3metrics": "fig_mean_3metrics.png",
    "output_mean_plot_2metrics": "fig_mean_2metrics.png",
    "output_dist_plot": "fig_dist_5metrics.png",
    
    # 图表DPI设置
    "dpi": 300,
}

# ==================== 数据结构 ====================

@dataclass
class EvalResult:
    run_id: str
    dataset: str
    query_idx: int
    user_input: str
    retrieval_mode: str
    metrics: Dict[str, float]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EvalResult':
        return cls(**data)


# ==================== 画图函数 ====================

def load_intermediate_results(input_path: str) -> Dict[str, List[EvalResult]]:
    """加载中间结果JSON"""
    print(f"\n加载中间结果: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"错误: 文件不存在: {input_path}")
        print("请先运行 run_experiment.py 生成结果")
        sys.exit(1)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_results = {}
    for mode, results_list in data.items():
        all_results[mode] = [EvalResult.from_dict(r) for r in results_list]
    
    print(f"加载完成: {len(all_results)} 种策略")
    for mode, results in all_results.items():
        print(f"  - {mode}: {len(results)} 条结果")
    
    return all_results


def generate_mean_plot(all_results: Dict[str, List[EvalResult]], 
                       output_path_3metrics: str, 
                       output_path_2metrics: str, 
                       dpi: int = 300):
    """生成均值对比图（分为两张图：前3个指标0-1范围，后2个指标1-5范围）"""
    print(f"\n{'='*60}")
    print("生成均值对比图...")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.style.use('default')
    except ImportError:
        print("错误: 未安装 matplotlib 或 numpy")
        print("请运行: pip install matplotlib numpy")
        return
    
    modes = ["bm25_only", "vector_only", "fusion_0.7_0.3"]
    mode_labels = ["BM25 Only", "Vector Only", "Fusion (0.7/0.3)"]
    colors = ['#1A1A2E', '#16213E', '#4361EE']
    
    all_metric_names = ["faithfulness", "answer_relevancy", "context_utilization", 
                        "cpac_plan_quality", "evidence_uncertainty"]
    data = {mode: {metric: [] for metric in all_metric_names} for mode in modes}
    
    for mode in modes:
        if mode in all_results:
            for result in all_results[mode]:
                for metric in all_metric_names:
                    if metric in result.metrics:
                        data[mode][metric].append(result.metrics[metric])
    
    metric_names_3 = ["faithfulness", "answer_relevancy", "context_utilization"]
    metric_labels_3 = ["Faithfulness", "Answer Relevancy", "Context Utilization"]
    
    fig1, ax1 = plt.subplots(figsize=(12, 7), facecolor='white')
    ax1.set_facecolor('#f8f9fa')
    
    x1 = np.arange(len(metric_names_3))
    width = 0.25
    
    for i, mode in enumerate(modes):
        means = [np.mean(data[mode][metric]) if data[mode][metric] else 0 
                 for metric in metric_names_3]
        stds = [np.std(data[mode][metric]) if data[mode][metric] else 0 
                for metric in metric_names_3]
        
        ax1.bar(x1 + (i - 1)*width, means, width, yerr=stds, 
               label=mode_labels[i], color=colors[i], alpha=0.9, capsize=5,
               edgecolor='black', linewidth=1.5,
               error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2))
    
    ax1.set_xlabel('Metrics', fontsize=16, fontweight='bold', color='black')
    ax1.set_ylabel('Score (0-1)', fontsize=16, fontweight='bold', color='black')
    ax1.set_title('Retrieval Strategy Comparison: RAGAS Metrics (0-1 Scale)', 
                  fontsize=20, fontweight='bold', pad=20, color='black')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(metric_labels_3, fontsize=14, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelsize=14, colors='black')
    
    legend1 = ax1.legend(loc='upper left', fontsize=14, frameon=True, 
                         facecolor='white', edgecolor='black', labelcolor='black')
                         
    ax1.grid(axis='y', alpha=0.3, linestyle='--', color='gray', linewidth=1)
    ax1.set_ylim(0, 1.2)
    
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(output_path_3metrics, dpi=dpi, bbox_inches='tight', facecolor=fig1.get_facecolor())
    print(f"均值对比图(3指标)已保存: {output_path_3metrics}")
    plt.close()
    
    
    metric_names_2 = ["cpac_plan_quality", "evidence_uncertainty"]
    metric_labels_2 = ["CPAC Plan Quality", "Evidence Uncertainty"]
    
    fig2, ax2 = plt.subplots(figsize=(10, 7), facecolor='white')
    ax2.set_facecolor('#f8f9fa')
    
    x2 = np.arange(len(metric_names_2))
    
    for i, mode in enumerate(modes):
        means = [np.mean(data[mode][metric]) if data[mode][metric] else 0 
                 for metric in metric_names_2]
        stds = [np.std(data[mode][metric]) if data[mode][metric] else 0 
                for metric in metric_names_2]
        
        ax2.bar(x2 + (i - 1)*width, means, width, yerr=stds, 
               label=mode_labels[i], color=colors[i], alpha=0.9, capsize=5,
               edgecolor='black', linewidth=1.5,
               error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2))
    
    ax2.set_xlabel('Metrics', fontsize=16, fontweight='bold', color='black')
    ax2.set_ylabel('Score (1-5)', fontsize=16, fontweight='bold', color='black')
    ax2.set_title('Retrieval Strategy Comparison: Domain-Specific Rubrics (1-5 Scale)', 
                  fontsize=18, fontweight='bold', pad=20, color='black')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metric_labels_2, fontsize=14, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelsize=14, colors='black')
    
    legend2 = ax2.legend(loc='upper left', fontsize=14, frameon=True, 
                         facecolor='white', edgecolor='black', labelcolor='black')
                         
    ax2.grid(axis='y', alpha=0.3, linestyle='--', color='gray', linewidth=1)
    ax2.set_ylim(0, 5.8)
    
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(output_path_2metrics, dpi=dpi, bbox_inches='tight', facecolor=fig2.get_facecolor())
    print(f"均值对比图(2指标)已保存: {output_path_2metrics}")
    
    plt.style.use('default')
    plt.close()


def generate_dist_plot(all_results: Dict[str, List[EvalResult]], output_path: str, dpi: int = 300):
    """生成分布对比图（箱线图）"""
    print(f"\n生成箱线图...")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.style.use('default')
    except ImportError:
        print("错误: 未安装 matplotlib 或 numpy")
        return
    
    metric_names = ["faithfulness", "answer_relevancy", "context_utilization", 
                    "cpac_plan_quality", "evidence_uncertainty"]
    metric_labels = ["Faithfulness", "Answer\nRelevancy", "Context\nUtilization", 
                     "CPAC Plan\nQuality", "Evidence\nUncertainty"]
    # 按照期望的性能顺序排列：BM25 (最弱) -> Vector (中等) -> Fusion (最强)
    modes = ["bm25_only", "vector_only", "fusion_0.7_0.3"]
    mode_labels = ["BM25 Only", "Vector Only", "Fusion (0.7/0.3)"]
    
    # 方案 4：暗夜星云 (Midnight Nebula)
    colors = ['#1A1A2E', '#16213E', '#4361EE']
    
    data = {mode: {metric: [] for metric in metric_names} for mode in modes}
    
    for mode in modes:
        if mode in all_results:
            for result in all_results[mode]:
                for metric in metric_names:
                    if metric in result.metrics:
                        data[mode][metric].append(result.metrics[metric])
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), facecolor='white') 
    
    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        ax.set_facecolor('#f8f9fa') 
        
        box_data = [data[mode][metric] for mode in modes]
        bp = ax.boxplot(box_data, labels=mode_labels, patch_artist=True,
                        boxprops=dict(linewidth=1.5, color='black'),
                        capprops=dict(linewidth=1.5, color='black'),
                        whiskerprops=dict(linewidth=1.5, color='black', linestyle='--'),
                        flierprops=dict(marker='o', markerfacecolor='gray', markersize=5, alpha=0.5),
                        medianprops=dict(linewidth=2, color='#f1c40f')) # 黄色中位数线高亮
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        
        ax.set_title(label.replace('\n', ' '), fontsize=11, fontweight='bold', color='black')
        ax.grid(axis='y', alpha=0.3, color='gray', linestyle='--')
        
        if idx < 3:
            ax.set_ylim(0, 1.1)
        else:
            ax.set_ylim(1, 5.5)
        
        ax.tick_params(axis='x', rotation=15, colors='black')
        ax.tick_params(axis='y', colors='black')
        
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1)
    
    fig.suptitle('Distribution Comparison Across Retrieval Strategies', 
                 fontsize=15, fontweight='bold', y=1.03, color='black')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"箱线图已保存: {output_path}")
    
    plt.style.use('default')
    plt.close()


def print_statistics(all_results: Dict[str, List[EvalResult]]):
    """打印统计汇总"""
    print(f"\n{'='*60}")
    print("统计汇总")
    print(f"{'='*60}")
    
    try:
        import numpy as np
    except ImportError:
        print("错误: 未安装 numpy")
        return
    
    metric_names = ["faithfulness", "answer_relevancy", "context_utilization", 
                    "cpac_plan_quality", "evidence_uncertainty"]
    modes = ["vector_only", "bm25_only", "fusion_0.7_0.3"]
    
    for mode in modes:
        if mode not in all_results:
            continue
        
        print(f"\n[{mode}]")
        print("-" * 40)
        
        for metric in metric_names:
            values = [r.metrics.get(metric, 0) for r in all_results[mode] if metric in r.metrics]
            if values:
                arr = np.array(values)
                print(f"{metric:20s}: mean={np.mean(arr):.3f}, std={np.std(arr):.3f}, median={np.median(arr):.3f}")


# ==================== 主函数 ====================

def main():
    print(f"\n{'='*70}")
    print("  检索策略对比实验 - 画图脚本")
    print(f"{'='*70}\n")
    
    config = CONFIG
    
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    intermediate_path = os.path.join(input_dir, config["intermediate_results"])
    
    all_results = load_intermediate_results(intermediate_path)
    
    mean_plot_path_3 = os.path.join(output_dir, config["output_mean_plot_3metrics"])
    mean_plot_path_2 = os.path.join(output_dir, config["output_mean_plot_2metrics"])
    generate_mean_plot(all_results, mean_plot_path_3, mean_plot_path_2, dpi=config["dpi"])
    
    dist_plot_path = os.path.join(output_dir, config["output_dist_plot"])
    generate_dist_plot(all_results, dist_plot_path, dpi=config["dpi"])
    
    print_statistics(all_results)
    
    print(f"\n{'='*70}")
    print("  图表生成完成！")
    print(f"{'='*70}\n")
    print("输出文件:")
    print(f"  - {config['output_mean_plot_3metrics']}")
    print(f"  - {config['output_mean_plot_2metrics']}")
    print(f"  - {config['output_dist_plot']}")
    print(f"\n输出目录: {output_dir}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
