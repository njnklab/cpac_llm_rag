#!/usr/bin/env python3
"""
CPAC 参数理解对比测试 - 使用本地 Ollama 模型

支持模型:
- llama3.1:70b-instruct-q4_K_M (context: 131072)
- deepseek-r1:70b (context: 131072)
- gpt-oss:20b (context: 131072)
- qwen3:32b (context: 40960)

使用 generate API，输出格式为 JSON
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from ollama_client import OllamaClient


def clean_response(text: str) -> str:
    """
    清理模型输出，去除 <think> 标签等非 JSON 内容
    
    Args:
        text: 原始模型输出
    
    Returns:
        清理后的纯净文本
    """
    # 去除 <think>...</think> 标签及其内容（qwen3 系列）
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 去除可能的 markdown 代码块标记
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    
    # 去除首尾空白
    text = text.strip()
    
    # 尝试提取 JSON 对象（如果有额外文本）
    # 找到第一个 { 和最后一个 } 之间的内容
    match = re.search(r'(\{.*\})', text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    
    return text


def normalize_to_target_schema(obj: dict, dataset_name: str, dataset_summary: str) -> dict:
    """Normalize model output to the target schema.

    Target schema:
    {
      "dataset_info": {"name": str, "summary": str},
      "modifications": [
        {
          "parameter_path": str,
          "current_default": str,
          "recommended_value": str,
          "action": "modify",
          "confidence": "high"|"medium"|"low",
          "rationale": str
        }
      ],
      "summary": {"total_modifications": int, "high_confidence": int, "medium_confidence": int, "low_confidence": int}
    }
    """
    if not isinstance(obj, dict):
        raise TypeError("model output must be a JSON object")

    # Accept both old key "parameter" and new key "parameter_path"
    mods_in = obj.get("modifications")
    if mods_in is None and isinstance(obj.get("response"), dict):
        mods_in = obj["response"].get("modifications")
    if mods_in is None:
        mods_in = []

    normalized_mods = []
    if isinstance(mods_in, list):
        for m in mods_in:
            if not isinstance(m, dict):
                continue
            parameter_path = m.get("parameter_path") or m.get("parameter")
            current_default = m.get("current_default")
            recommended_value = m.get("recommended_value")
            confidence = m.get("confidence")
            rationale = m.get("rationale")

            if parameter_path is None or current_default is None or recommended_value is None or rationale is None:
                continue

            # Filter: only keep modifications where recommended_value != current_default
            if str(current_default).strip() == str(recommended_value).strip():
                continue

            conf = str(confidence).strip().lower() if confidence is not None else "medium"
            if conf not in {"high", "medium", "low"}:
                conf = "medium"

            normalized_mods.append(
                {
                    "parameter_path": str(parameter_path),
                    "current_default": str(current_default),
                    "recommended_value": str(recommended_value),
                    "action": "modify",
                    "confidence": conf,
                    "rationale": str(rationale),
                }
            )

    counts = {"high": 0, "medium": 0, "low": 0}
    for m in normalized_mods:
        counts[m["confidence"]] += 1

    return {
        "dataset_info": {"name": dataset_name, "summary": dataset_summary},
        "modifications": normalized_mods,
        "summary": {
            "total_modifications": len(normalized_mods),
            "high_confidence": counts["high"],
            "medium_confidence": counts["medium"],
            "low_confidence": counts["low"],
        },
    }


# ============================================================================
# 模型配置
# ============================================================================
MODELS = [
    "llama3.1:70b-instruct-q4_K_M",
    "deepseek-r1:70b",
    "gpt-oss:20b",
    "qwen3:32b",
]

# 各模型的最大上下文长度（从 ollama show -v 获取）
MODEL_CONTEXT_LENGTH = {
    "llama3.1:70b-instruct-q4_K_M": 131072,
    "deepseek-r1:70b": 131072,
    "gpt-oss:20b": 131072,
    "qwen3:32b": 40960,
}


TARGET_DATASET_NAME = "ADHD-200 NYU Sample"
TARGET_DATASET_SUMMARY = "Resting-state fMRI dataset with 224 subjects, single session, 3x3x4mm voxel size, 176 timepoints, no available fieldmaps"


# ============================================================================
# "严谨参数理解"配置 - 用于 MRI/fMRI 参数分析
# ============================================================================
def get_strict_options(model: str, num_ctx: int = 32768, num_predict: int = 4096) -> dict:
    """
    获取"严谨参数理解"配置
    
    Args:
        model: 模型名称
        num_ctx: 上下文窗口大小（根据输入长度调整）
        num_predict: 最大生成 token 数
    
    Returns:
        Ollama options 字典
    """
    # 确保 num_ctx 不超过模型最大值
    max_ctx = MODEL_CONTEXT_LENGTH.get(model, 32768)
    num_ctx = min(num_ctx, max_ctx)
    
    return {
        # 上下文与生成
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        
        # GPU 利用率最大化
        "num_gpu": -1,  # 尽量全部 offload 到 GPU
        "num_batch": 512,  # 批处理大小，可根据显存调整
        
        # 采样参数 - 严谨/低随机性
        "temperature": 0.2,  # 低温度，更确定性
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.05,
        
        # 复现性
        "seed": 42,
    }


# ============================================================================
# CPAC 参数理解 Prompt（优化版）
# ============================================================================
CPAC_SYSTEM_PROMPT = """You are an expert in fMRI data analysis, specializing in configuring C-PAC (Configurable Pipeline for the Analysis of Connectomes) processing pipelines.

Your task is to analyze the provided dataset information and CPAC default parameters, then suggest **ONLY the parameters that need to be modified** from their defaults.

## Critical Rules:
1. **ONLY include parameters where recommended_value ≠ current_default**
2. **DO NOT include parameters where you recommend keeping the default**
3. For each modification, provide a clear rationale based on dataset characteristics or research goals
4. Assign confidence level: high (>90%), medium (70-90%), low (<70%)

## Output Schema (JSON only, no markdown):
Return exactly ONE JSON object with EXACT keys and types:
{
  "dataset_info": {
    "name": "<dataset name>",
    "summary": "<1-2 sentence dataset summary>"
  },
  "modifications": [
    {
      "parameter_path": "full.parameter.path",
      "current_default": "<stringified default>",
      "recommended_value": "<stringified recommended value>",
      "action": "modify",
      "confidence": "high"|"medium"|"low",
      "rationale": "<why this change is necessary>"
    }
  ],
  "summary": {
    "total_modifications": 0,
    "high_confidence": 0,
    "medium_confidence": 0,
    "low_confidence": 0
  }
}

## IMPORTANT (STRICT):
- Do NOT include any thinking process, explanations, headings, or markdown formatting
- Do NOT wrap the JSON in code blocks
- Do NOT output any keys other than: dataset_info, modifications, summary
- The only allowed value for action is: "modify"
- Only include entries in modifications where recommended_value != current_default
- Output ONLY the raw JSON object, nothing else
"""


def build_cpac_prompt(dataset_summary: str, parameters: str, research_goal: str = "") -> str:
    """
    构建 CPAC 参数理解的完整 prompt
    
    Args:
        dataset_summary: BIDS 数据集摘要
        parameters: CPAC 参数列表（path = default ; choices=...）
        research_goal: 研究目标（可选）
    
    Returns:
        完整的 prompt 字符串
    """
    prompt = f"""{CPAC_SYSTEM_PROMPT}

## Dataset Summary:
{dataset_summary}

## Research Goal:
{research_goal if research_goal else "General fMRI preprocessing for functional connectivity analysis"}

## CPAC Parameters to Review:
{parameters}

## Your Task:
Analyze the above parameters and return a JSON object with ONLY the parameters that should be modified from their defaults. Remember: if current_default == recommended_value, do NOT include that parameter.

Return ONLY the JSON object, no additional text or markdown formatting."""
    
    return prompt


# ============================================================================
# 测试用的 CPAC 参数（来自用户提供的示例）
# ============================================================================
TEST_PARAMETERS = """
Module: anatomical_preproc
Parameters (path = default ; choices=...):
- anatomical_preproc.run = On
- anatomical_preproc.run_t2 = Off
- anatomical_preproc.non_local_means_filtering.run = [Off]
- anatomical_preproc.non_local_means_filtering.noise_model = 'Gaussian' ; choices=Gaussian, Rician
- anatomical_preproc.n4_bias_field_correction.run = [Off]
- anatomical_preproc.n4_bias_field_correction.shrink_factor = 2
- anatomical_preproc.t1t2_bias_field_correction.run = Off
- anatomical_preproc.t1t2_bias_field_correction.BiasFieldSmoothingSigma = 5
- anatomical_preproc.acpc_alignment.run = Off
- anatomical_preproc.acpc_alignment.run_before_preproc = true
- anatomical_preproc.acpc_alignment.brain_size = 150
- anatomical_preproc.acpc_alignment.FOV_crop = robustfov
- anatomical_preproc.acpc_alignment.acpc_target = 'whole-head' ; choices=brain, whole-head
- anatomical_preproc.acpc_alignment.align_brain_mask = Off
- anatomical_preproc.brain_extraction.run = On
- anatomical_preproc.brain_extraction.using = ['BET'] ; choices=3dSkullStrip, BET, UNet, niworkflows-ants, FreeSurfer-ABCD, FreeSurfer-BET-Tight, FreeSurfer-BET-Loose, FreeSurfer-Brainmask
- anatomical_preproc.brain_extraction.FSL-BET.frac = 0.5
- anatomical_preproc.brain_extraction.FSL-BET.robust = On
- anatomical_preproc.brain_extraction.FSL-BET.vertical_gradient = 0.0

---
Module: functional_preproc
Parameters (path = default ; choices=...):
- functional_preproc.run = On
- functional_preproc.truncation.start_tr = 0
- functional_preproc.truncation.stop_tr = None
- functional_preproc.despiking.run = [Off]
- functional_preproc.slice_timing_correction.run = [On]
- functional_preproc.slice_timing_correction.tpattern = None
- functional_preproc.motion_estimates_and_correction.run = On
- functional_preproc.motion_estimates_and_correction.motion_correction.using = ['3dvolreg'] ; choices=3dvolreg, mcflirt
- functional_preproc.motion_estimates_and_correction.motion_correction.motion_correction_reference = ['mean'] ; choices=mean, median, selected_volume, fmriprep_reference
- functional_preproc.distortion_correction.run = [On]
- functional_preproc.distortion_correction.using = ['PhaseDiff', 'Blip'] ; choices=PhaseDiff, Blip
- functional_preproc.func_masking.run = On
- functional_preproc.func_masking.using = ['AFNI'] ; choices=AFNI, FSL, FSL_AFNI, Anatomical_Refined, Anatomical_Based, Anatomical_Resampled, CCS_Anatomical_Refined
"""

TEST_DATASET_SUMMARY = """
- Dataset: NYU (New York University)
- Number of Subjects: 224
- Number of Sessions: 1
- Modalities: anat (T1w), func (BOLD)
- Scanner: 3T Siemens
- TR: 2.0s
- Voxel size: 3x3x3 mm
- No fieldmap data available
"""


def run_single_model_test(client: OllamaClient, model: str, prompt: str, 
                          options: dict, output_dir: Path) -> dict:
    """
    对单个模型运行测试
    
    Args:
        client: OllamaClient 实例
        model: 模型名称
        prompt: 完整 prompt
        options: Ollama options
        output_dir: 输出目录
    
    Returns:
        包含结果和元信息的字典
    """
    print(f"\n{'='*60}")
    print(f"Testing model: {model}")
    print(f"Options: num_ctx={options.get('num_ctx')}, num_predict={options.get('num_predict')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            options=options,
            stream=False,
            format="json"  # 强制 JSON 输出
        )
        
        elapsed_time = time.time() - start_time
        raw_response = response.get("response", "")
        
        # 清理响应（去除 <think> 标签等）
        cleaned_response = clean_response(raw_response)
        
        # 尝试解析 JSON
        try:
            parsed_json = json.loads(cleaned_response)
            json_valid = True
        except json.JSONDecodeError as e:
            parsed_json = {"error": f"JSON parse error: {str(e)}", "raw": cleaned_response[:500]}
            json_valid = False

        # 归一化到目标 schema（即使模型返回了不同 schema，也尽量对齐）
        if json_valid:
            try:
                parsed_json = normalize_to_target_schema(
                    parsed_json,
                    dataset_name=TARGET_DATASET_NAME,
                    dataset_summary=TARGET_DATASET_SUMMARY,
                )
            except Exception as e:
                json_valid = False
                parsed_json = {"error": f"Schema normalize error: {str(e)}", "raw": cleaned_response[:500]}
        
        result = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed_time, 2),
            "options": options,
            "json_valid": json_valid,
            "response": parsed_json,
            "raw_response_length": len(raw_response),
        }
        
        print(f"✓ Completed in {elapsed_time:.2f}s")
        print(f"  JSON valid: {json_valid}")
        print(f"  Response length: {len(raw_response)} chars")
        
        if json_valid and "modifications" in parsed_json:
            print(f"  Modifications suggested: {len(parsed_json.get('modifications', []))}")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        result = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed_time, 2),
            "options": options,
            "error": str(e),
        }
        print(f"✗ Error: {e}")
    
    # 保存单个模型结果
    safe_model_name = model.replace(":", "_").replace("/", "_")

    # 1) 保存 schema-only 最终 JSON（符合目标格式，可直接使用）
    schema_output_file = output_dir / f"{safe_model_name}_modifications.json"
    if isinstance(result.get("response"), dict) and result.get("json_valid") is True:
        with open(schema_output_file, "w", encoding="utf-8") as f:
            json.dump(result["response"], f, ensure_ascii=False, indent=4)
        print(f"  Saved schema-only JSON to: {schema_output_file}")

    # 2) 保存带元信息的调试结果（便于排查模型输出/速度等）
    output_file = output_dir / f"{safe_model_name}_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved to: {output_file}")
    
    return result


def main():
    """
    主函数：对比多个模型对 CPAC 参数的理解
    """
    print("=" * 60)
    print("CPAC Parameter Understanding - Multi-Model Comparison")
    print("=" * 60)
    
    # 初始化客户端
    client = OllamaClient()
    
    # 检查可用模型
    print("\nChecking available models...")
    available = client.list_models()
    available_names = [m["name"] for m in available.get("models", [])]
    
    models_to_test = []
    for model in MODELS:
        if model in available_names:
            print(f"  ✓ {model}")
            models_to_test.append(model)
        else:
            print(f"  ✗ {model} (not found)")
    
    if not models_to_test:
        print("\nNo models available for testing!")
        return
    
    # 构建 prompt
    prompt = build_cpac_prompt(
        dataset_summary=TEST_DATASET_SUMMARY,
        parameters=TEST_PARAMETERS,
        research_goal="Resting-state functional connectivity analysis with focus on reproducibility"
    )
    
    print(f"\nPrompt length: {len(prompt)} chars")
    
    # 创建输出目录
    output_dir = Path("/home/a001/zhangyan/cpac/llm_parameters/20250825全参数整理/ollama/temp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行测试
    all_results = []
    for model in models_to_test:
        options = get_strict_options(model, num_ctx=32768, num_predict=4096)
        result = run_single_model_test(client, model, prompt, options, output_dir)
        all_results.append(result)
    
    # 保存汇总结果
    summary_file = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_time": datetime.now().isoformat(),
            "models_tested": models_to_test,
            "prompt_length": len(prompt),
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    print(f"{'='*60}")
    
    # 打印对比摘要
    print("\n## Quick Comparison:")
    for r in all_results:
        model = r["model"]
        if "error" in r:
            print(f"  {model}: ERROR - {r['error'][:50]}")
        else:
            mods = len(r.get("response", {}).get("modifications", []))
            print(f"  {model}: {r['elapsed_seconds']}s, JSON valid={r['json_valid']}, modifications={mods}")


if __name__ == "__main__":
    main()
