import os
import json
import yaml
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 基础路径配置
# 由于脚本在 scripts/ 目录下，EXPERIMENT_ROOT 指向其上一级
SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_ROOT = str(SCRIPT_DIR.parent)
YML_TEMPLATE_PATH = '/home/a001/zhangyan/cpac/llm_parameters/20250825全参数整理/pipeline_config_default.yml'
DATASET_NAME = 'ds002748'

def get_nested(d, path):
    """递归获取嵌套字典里的值"""
    if not d or not path: return None
    keys = path.split('.')
    current = d
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        elif isinstance(current, list):
            try:
                idx = int(k)
                if 0 <= idx < len(current):
                    current = current[idx]
                else: return None
            except: return None
        else: return None
    return current

def evaluate_single_pair(json_path, yml_path, template_yml):
    """计算单对 JSON+YAML 的指标"""
    if not json_path.exists(): return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            mod_data = json.load(f)
    except: return None
    
    config = mod_data.get('config', {})
    modifications = config.get('modifications') or mod_data.get('modifications') or []
    
    # 1. 可执行率
    exe = 0
    generated_yml = None
    if yml_path and yml_path.exists():
        try:
            with open(yml_path, 'r', encoding='utf-8') as f:
                generated_yml = yaml.safe_load(f)
            if generated_yml and 'pipeline_setup' in generated_yml: exe = 1
        except: pass

    # 2. 字段合法率
    valid = 0
    for mod in modifications:
        field = mod.get('parameter_path') or mod.get('field', '')
        try:
            if get_nested(template_yml, field) is not None: valid += 1
        except: pass
    valid_rate = (valid / len(modifications)) if modifications else 0

    # 3. 回退率
    failed = 0
    if not modifications:
        fallback_rate = 0
    else:
        for mod in modifications:
            field = mod.get('parameter_path') or mod.get('field', '')
            tem_val = get_nested(template_yml, field)
            if not generated_yml:
                failed += 1
                continue
            gen_val = get_nested(generated_yml, field)
            if gen_val is None or str(gen_val) == str(tem_val): failed += 1
        fallback_rate = (failed / len(modifications))

    return {
        'exe': exe,
        'valid': valid_rate,
        'fallback': fallback_rate,
        'count': len(modifications)
    }

def main():
    print(f"正在扫描实验目录: {EXPERIMENT_ROOT}")
    if not os.path.exists(YML_TEMPLATE_PATH):
        print(f"错误: 找不到模板文件 {YML_TEMPLATE_PATH}")
        return
    with open(YML_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        template_yml = yaml.safe_load(f)

    all_results = []
    run_dirs = sorted(glob.glob(os.path.join(EXPERIMENT_ROOT, f"{DATASET_NAME}_runs", "run_*")))
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        for json_file in run_path.glob("*_modifications.json"):
            model_info = json_file.name.replace(f"{DATASET_NAME}_", "").replace("_modifications.json", "")
            yml_file = run_path / json_file.name.replace("_modifications.json", "_pipeline.yml")
            stats = evaluate_single_pair(json_file, yml_file, template_yml)
            if stats:
                stats['Model'] = model_info
                all_results.append(stats)

    if not all_results:
        print("未发现有效数据。")
        return

    df = pd.DataFrame(all_results)
    summary_list = []
    for model_name, group in df.groupby('Model'):
        # 计算一致性系数 (Consistency Index, CI)
        # CI = 1 - (std of count / mean of count)
        mean_cnt = group['count'].mean()
        std_cnt = group['count'].std()
        ci = 1.0 - (std_cnt / mean_cnt) if mean_cnt > 0 and not pd.isna(std_cnt) else 1.0
        if ci < 0: ci = 0 # 防止负值
        
        summary_list.append({
            'Model': model_name,
            'Executable Rate': f"{group['exe'].mean():.2f}",
            'Field Validity Rate': f"{group['valid'].mean():.2f} ± {group['valid'].std():.2f}",
            'Fallback Rate': f"{group['fallback'].mean():.2f} ± {group['fallback'].std():.2f}",
            'Avg Modified Fields': f"{group['count'].mean():.1f} ± {group['count'].std():.1f}",
            'Consistency Index (CI)': f"{ci:.2f}"
        })

    final_df = pd.DataFrame(summary_list)
    print("\n" + "="*80)
    print(f"全量化实验汇总 (数据集: {DATASET_NAME})")
    print("="*80)
    print(final_df.to_markdown(index=False))
    
    # 路径存到 reports/
    report_dir = os.path.join(EXPERIMENT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)
    csv_out = os.path.join(report_dir, f"{DATASET_NAME}_final_report.csv")
    md_out = os.path.join(report_dir, f"{DATASET_NAME}_final_report.md")
    
    final_df.to_csv(csv_out, index=False)
    with open(md_out, 'w', encoding='utf-8') as f:
        f.write(f"# {DATASET_NAME} 基座模型评估全量化结果 (N=10)\n\n")
        f.write(final_df.to_markdown(index=False))
    print(f"\n结果已保存至 reports/ 目录下。")

if __name__ == '__main__':
    main()
