import sys
import os
from pathlib import Path

# --- 核心：将父目录加入 sys.path 以便导入原脚本逻辑 ---
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# 导入评估脚本中的主逻辑（它包含循环 N 次调用和评估的功能）
import evaluate_models
from ollama import ollama_cpac
import generate_dataset_config

# --- 实验补全参数微调 ---
# 1. 提高模型温度到 0.7，打破 GPT-OSS 和 Gemma-3 极其保守、不做修改的僵局
ollama_cpac.DEFAULT_OPTIONS["temperature"] = 0.7
if "seed" in ollama_cpac.DEFAULT_OPTIONS:
    del ollama_cpac.DEFAULT_OPTIONS["seed"]

# 2. 限定仅针对这两个缺失输出的模型进行补全实验
evaluate_models.OLLAMA_MODELS = [
    'gpt-oss:20b',
    'gemma3:27b'
]

if __name__ == '__main__':
    print("\n" + "="*60)
    print("补全实验启动：针对 GPT-OSS & Gemma-3 (Temperature=0.7)")
    print("="*60)
    print("注意：通过提高采样温度，我们旨在获取更具多样性的参数修改建议。")
    print(f"数据将保存至: {evaluate_models.EXPERIMENT_ROOT}/ds002748_runs/run_1...10/")
    
    # 运行评估脚本中的循环调用逻辑
    try:
        evaluate_models.main()
        print("\n" + "="*60)
        print("补全实验调用完成！")
        print("现在你可以直接运行 python evaluate_models.py (不要用本脚本) 来生成包含 5 个模型的最终对比表了。")
        print("="*60)
    except Exception as e:
        print(f"\n[ERROR] 运行补全实验时出错: {e}")
