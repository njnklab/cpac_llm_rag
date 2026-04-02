import os
import json
import yaml
from pathlib import Path
import sys

# 设置路径以前往父目录调用逻辑
# 假设 generate_dataset_config.py 在父目录
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# 导入 generate_dataset_config.py 里的核心逻辑
from generate_dataset_config import (
    get_bids_summary, 
    get_model_short_name, 
    run_single_job,
    OLLAMA_MODELS,
    YML_TEMPLATE_PATH
)
import generate_dataset_config

# --- 实验配置 ---
BIDS_DATASET_PATH = '/mnt/sda1/zhangyan/openneuro/ds002748'
EXPERIMENT_ROOT = '/home/a001/zhangyan/cpac/llm_parameters/20250825全参数整理/基座模型对比实验'
N_REPETITIONS = 10  # 每个模型重复 10 次

# 抑郁症研究目标
RESEARCH_GOAL = (
    "The primary goal of this study is to investigate whether resting-state fMRI signals "
    "can be used to discriminate patients with mild depression from healthy controls "
    "based on functional brain network features. Using structural (anat) and functional "
    "(resting-state) data acquired under a closed-eyes condition, we aim to preprocess "
    "and extract imaging-derived features such as regional activity and connectivity metrics, "
    "and subsequently build machine learning classification models. By doing so, we seek "
    "to identify neuroimaging biomarkers that can support objective diagnosis and provide "
    "new insights into the neural mechanisms underlying depression."
)

def main():
    print(f"开始 ds002748 数据集的多轮对比实验...")
    print(f"数据存放根路径: {EXPERIMENT_ROOT}")
    
    dataset_name = Path(BIDS_DATASET_PATH).name
    summary_text = get_bids_summary(BIDS_DATASET_PATH)
    
    # 劫持 generate_dataset_config 模块里的全局变量
    generate_dataset_config.RESEARCH_GOAL = RESEARCH_GOAL
    generate_dataset_config.BIDS_DATASET_PATH = BIDS_DATASET_PATH

    # 遍历所有模型并开始多轮循环实验
    for model_name in OLLAMA_MODELS:
        model_short = get_model_short_name(model_name)
        print(f"\n" + "="*80)
        print(f"正在进行多轮实验 - 模型: {model_name}")
        print("="*80)
        
        for i in range(1, N_REPETITIONS + 1):
            print(f"  --> Round {i}/{N_REPETITIONS}")
            
            # 创建隔离的 run 目录：ds002748_runs/run_i/
            run_dir = Path(EXPERIMENT_ROOT) / f"{dataset_name}_runs" / f"run_{i}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            json_out = run_dir / f"{dataset_name}_{model_short}_modifications.json"
            yml_out = run_dir / f"{dataset_name}_{model_short}_pipeline.yml"
            
            # 执行模型推理并更新 YML
            # 这会调用底层 Ollama 推理逻辑并将结果保存至对应 run 目录
            success = run_single_job(summary_text, model_name, json_out, yml_out)
            
            if not success:
               print(f"      ✗ 轮次 {i} 指令生成失败，或无修改提出。")

    print("\n" + "="*80)
    print("实验核心推理阶段全部完成！")
    print(f"请现在直接运行: python evaluate_models.py (不要运行 evaluate_models 里的 main)")
    print("它将自动分析刚才生成的所有结果文件。")
    print("="*80)

if __name__ == '__main__':
    main()
