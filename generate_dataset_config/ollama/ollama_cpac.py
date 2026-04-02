#!/usr/bin/env python3
"""
Ollama 版 CPAC 参数配置生成器

按模块分 5 次调用 Ollama 模型，然后合并成最终 JSON。
输出格式与 generate_config.py 的 AWS 版本一致。

支持模型:
- llama3.1:70b-instruct-q4_K_M
- deepseek-r1:70b
- gpt-oss:20b
- qwen3:32b
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加父目录到 path，以便导入 generate_config 中的函数
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ollama_client import OllamaClient
except ImportError:
    from ollama.ollama_client import OllamaClient


# ============================================================================
# 配置
# ============================================================================
OLLAMA_MODELS = [
    "llama3.1:70b-instruct-q4_K_M",
    "deepseek-r1:70b",
    "gpt-oss:20b",
    "qwen3:32b",
]

CORE_MODULES = [
    'anatomical_preproc',
    'functional_preproc',
    'registration_workflows',
    'nuisance_corrections',
    'segmentation',
]

# 各模型的最大上下文长度
MODEL_CONTEXT_LENGTH = {
    "llama3.1:70b-instruct-q4_K_M": 131072,
    "deepseek-r1:70b": 131072,
    "gpt-oss:20b": 131072,
    "qwen3:32b": 40960,
}

# 默认 Ollama options（严谨参数理解配置）
DEFAULT_OPTIONS = {
    "num_ctx": 16384,      # 单模块 prompt 较短，16K 足够
    "num_predict": 2048,   # 单模块修改建议不会太长
    "num_gpu": -1,
    "num_batch": 512,
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.05,
    "seed": 42,
}


# ============================================================================
# 从 generate_config.py 复用的函数（简化版）
# ============================================================================
def load_module_defaults(modules_path: str, module_names: list) -> dict:
    """加载指定模块的默认参数 YAML 文件（原始文本）"""
    defaults = {}
    for name in module_names:
        file_path = os.path.join(modules_path, f"{name}.yml")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                defaults[name] = f.read()
        except FileNotFoundError as e:
            print(f"Warning: Could not load {file_path}. Error: {e}")
            defaults[name] = ""
    return defaults


def _parse_inline_list(value):
    v = value.strip()
    if not (v.startswith("[") and v.endswith("]")):
        return None
    inner = v[1:-1].strip()
    if not inner:
        return []
    parts = [p.strip() for p in inner.split(",")]
    cleaned = []
    for p in parts:
        p2 = p.strip().strip("\"'")
        if p2:
            cleaned.append(p2)
    return cleaned


def _extract_choices_from_comment(comment_text):
    if not comment_text:
        return None
    m = re.search(r"\b(options|using|input)\s*:\s*(.+)$", comment_text, flags=re.IGNORECASE)
    if not m:
        return None
    rhs = m.group(2).strip()
    if "[" in rhs and "]" in rhs:
        start = rhs.find("[")
        end = rhs.find("]", start)
        if end != -1:
            parsed = _parse_inline_list(rhs[start : end + 1])
            if parsed:
                return parsed
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", rhs)
    vals = [a or b for (a, b) in quoted if (a or b)]
    if vals:
        return vals
    rhs = rhs.replace(" or ", ", ")
    rough = [t.strip() for t in rhs.split(",")]
    rough = [t.strip().strip("\"'") for t in rough if t.strip()]
    return rough or None


_PATH_KEYWORDS = (
    '_path', '_template', '_mask', '_model', '_matrix', '_file',
    '/ants_template', '/ccs_template', '/cpac_templates', 's3://',
    '$FSLDIR', '.nii', '.gz', '.mat', '.model'
)


def _is_path_related(key, value):
    value_lower = value.lower() if value else ""
    for kw in _PATH_KEYWORDS:
        if kw.lower() in value_lower:
            return True
    key_lower = key.lower() if key else ""
    if any(key_lower.endswith(suffix) for suffix in ('_path', '_template', '_mask', '_file', '_matrix', '_model')):
        if value.startswith('/') or value.startswith('$') or value.startswith('s3://') or '.nii' in value:
            return True
    return False


def _summarize_module_yaml(yaml_text, max_entries=180, max_complex=25):
    lines = yaml_text.splitlines()
    stack = []
    entries = []
    complex_blocks = []
    pending_comment = []
    last_key_started = None

    def _stack_path(extra_key=None):
        keys = [k for _, k in stack]
        if extra_key is not None:
            keys.append(extra_key)
        return ".".join(keys)

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        stripped = line.lstrip(" ")
        if stripped.startswith("#"):
            pending_comment.append(stripped[1:].strip())
            continue
        indent = len(line) - len(stripped)
        if stripped.startswith("-"):
            if last_key_started and indent > last_key_started[1]:
                complex_blocks.append(last_key_started[0])
                last_key_started = None
            continue
        if ":" not in stripped:
            continue
        key_part, value_part = stripped.split(":", 1)
        key = key_part.strip()
        value = value_part.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if value == "":
            stack.append((indent, key))
            last_key_started = (_stack_path(), indent)
            pending_comment = []
            continue
        path = _stack_path(key)
        comment_text = " ".join(pending_comment).strip()
        pending_comment = []
        if _is_path_related(key, value):
            continue
        choices = None
        inline = _parse_inline_list(value)
        if inline is not None:
            if len(inline) > 1:
                choices = inline
        if choices is None:
            choices = _extract_choices_from_comment(comment_text)
        entries.append((path, value, choices))
        if len(entries) >= max_entries:
            break

    complex_unique = []
    seen = set()
    for p in complex_blocks:
        if p not in seen:
            seen.add(p)
            complex_unique.append(p)
        if len(complex_unique) >= max_complex:
            break
    return entries, complex_unique


def format_module_reference(module_name: str, yaml_text: str) -> str:
    """格式化单个模块的参数列表（用于 prompt）"""
    entries, complex_blocks = _summarize_module_yaml(yaml_text)
    out = ["---", f"Module: {module_name}", "Parameters (path = default ; choices=...):"]
    for path, value, choices in entries:
        if choices:
            choices_str = ", ".join(choices)
            out.append(f"- {path} = {value} ; choices={choices_str}")
        else:
            out.append(f"- {path} = {value}")
    for p in complex_blocks:
        out.append(f"- {p} = <complex_block>")
    return "\n".join(out)


# ============================================================================
# 从 main.py 复用的函数
# ============================================================================
def clean_response(text: str) -> str:
    """清理模型输出，去除 <think> 标签等非 JSON 内容"""
    # 去除各种 thinking 标签及其内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
    text = re.sub(r'<analysis>.*?</analysis>', '', text, flags=re.DOTALL)
    # 去除可能的 markdown 代码块标记
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = text.strip()
    # 尝试提取 JSON 对象（如果有额外文本）
    match = re.search(r'(\{.*\})', text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    return text


# ============================================================================
# 单模块 Prompt 构建
# ============================================================================
SINGLE_MODULE_PROMPT_TEMPLATE = """You are an expert in fMRI data analysis, specializing in configuring C-PAC (Configurable Pipeline for the Analysis of Connectomes) processing pipelines.

Your task is to analyze the provided dataset information and suggest **ONLY the parameters that need to be modified** from their defaults for the **{module_name}** module.

## Critical Rules:
1. **ONLY include parameters where recommended_value ≠ current_default**
2. **DO NOT include parameters where you recommend keeping the default**
3. For each modification, provide a clear rationale based on dataset characteristics or research goals
4. Assign confidence level: high (>90%), medium (70-90%), low (<70%)

## Output Schema (JSON only, no markdown):
Return exactly ONE JSON object with EXACT keys and types:
{{
  "module": "{module_name}",
  "modifications": [
    {{
      "parameter_path": "full.parameter.path",
      "current_default": "<stringified default>",
      "recommended_value": "<stringified recommended value>",
      "action": "modify",
      "confidence": "high"|"medium"|"low",
      "rationale": "<why this change is necessary>"
    }}
  ]
}}

## IMPORTANT (STRICT):
- Do NOT include any thinking process, explanations, headings, or markdown formatting
- Do NOT use <think> tags or any reasoning tags
- Do NOT output thinking steps, reasoning, or analysis process
- Do NOT wrap the JSON in code blocks
- The only allowed value for action is: "modify"
- Only include entries in modifications where recommended_value != current_default
- Output ONLY the raw JSON object, nothing else
- NO thinking tags, NO reasoning tags, ONLY pure JSON output

## Dataset Summary:
{dataset_summary}

## Research Goal:
{research_goal}

## Module Parameters to Review:
{module_parameters}

Return ONLY the JSON object for this module's modifications:"""


def build_single_module_prompt(module_name: str, module_yaml: str, 
                                dataset_summary: str, research_goal: str) -> str:
    """构建单模块的 prompt"""
    module_parameters = format_module_reference(module_name, module_yaml)
    return SINGLE_MODULE_PROMPT_TEMPLATE.format(
        module_name=module_name,
        dataset_summary=dataset_summary,
        research_goal=research_goal,
        module_parameters=module_parameters,
    )


# ============================================================================
# 核心函数：单模块调用
# ============================================================================
def query_single_module(client: OllamaClient, model: str, module_name: str,
                        module_yaml: str, dataset_summary: str, research_goal: str,
                        options: dict = None) -> dict:
    """
    对单个模块调用 Ollama 模型，返回该模块的修改建议
    
    Returns:
        dict: {"module": str, "modifications": list, "elapsed_seconds": float, "error": str or None}
    """
    if options is None:
        options = DEFAULT_OPTIONS.copy()
    
    prompt = build_single_module_prompt(module_name, module_yaml, dataset_summary, research_goal)
    
    print(f"    Querying {model} for module: {module_name}...")
    start_time = time.time()
    
    try:
        # 某些模型（如 gpt-oss、glm）不支持 format="json"，需要特殊处理
        models_without_json_format = ['gpt-oss', 'glm']
        use_json_format = not any(m in model.lower() for m in models_without_json_format)
        
        # glm-4.7-flash 需要使用 chat API 而不是 generate API
        if 'glm' in model.lower():
            chat_response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=options.get("temperature", 0.2),
                max_tokens=options.get("num_predict", 2048),
                stream=False
            )
            raw_response = chat_response.get("message", {}).get("content", "")
            # 如果 content 为空但有 thinking，可能需要从 thinking 提取
            if not raw_response and "thinking" in chat_response.get("message", {}):
                # 尝试从 thinking 中提取 JSON
                raw_response = chat_response["message"]["thinking"]
        else:
            response = client.generate(
                model=model,
                prompt=prompt,
                options=options,
                stream=False,
                format="json" if use_json_format else None
            )
            raw_response = response.get("response", "")
        elapsed = time.time() - start_time
        cleaned = clean_response(raw_response)
        
        # Debug: 如果解析失败且是 glm 模型，打印原始响应
        if 'glm' in model.lower():
            print(f"      [DEBUG] Raw response length: {len(raw_response)}")
            if len(raw_response) > 0:
                print(f"      [DEBUG] First 500 chars: {raw_response[:500]}")
        
        try:
            parsed = json.loads(cleaned)
            modifications = parsed.get("modifications", [])
            # 归一化每个 modification
            normalized_mods = []
            for m in modifications:
                if not isinstance(m, dict):
                    continue
                param_path = m.get("parameter_path") or m.get("parameter")
                current_default = m.get("current_default")
                recommended_value = m.get("recommended_value")
                rationale = m.get("rationale")
                confidence = m.get("confidence", "medium")
                
                if not all([param_path, current_default is not None, recommended_value is not None, rationale]):
                    continue
                if str(current_default).strip() == str(recommended_value).strip():
                    continue
                
                conf = str(confidence).strip().lower()
                if conf not in {"high", "medium", "low"}:
                    conf = "medium"
                
                normalized_mods.append({
                    "parameter_path": str(param_path),
                    "current_default": str(current_default),
                    "recommended_value": str(recommended_value),
                    "action": "modify",
                    "confidence": conf,
                    "rationale": str(rationale),
                })
            
            print(f"      ✓ {module_name}: {len(normalized_mods)} modifications ({elapsed:.1f}s)")
            return {
                "module": module_name,
                "modifications": normalized_mods,
                "elapsed_seconds": round(elapsed, 2),
                "error": None,
            }
        except json.JSONDecodeError as e:
            print(f"      ✗ {module_name}: JSON parse error ({elapsed:.1f}s)")
            return {
                "module": module_name,
                "modifications": [],
                "elapsed_seconds": round(elapsed, 2),
                "error": f"JSON parse error: {str(e)}",
            }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"      ✗ {module_name}: Error - {str(e)[:50]} ({elapsed:.1f}s)")
        return {
            "module": module_name,
            "modifications": [],
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e),
        }


# ============================================================================
# 核心函数：5 次调用合并
# ============================================================================
def query_all_modules(client: OllamaClient, model: str, 
                      dataset_name: str, dataset_summary: str, research_goal: str,
                      modules_path: str, modules: list = None,
                      options: dict = None) -> dict:
    """
    对所有模块依次调用 Ollama 模型，然后合并成最终 JSON
    
    Args:
        client: OllamaClient 实例
        model: 模型名称
        dataset_name: 数据集名称（用于 dataset_info）
        dataset_summary: 数据集摘要
        research_goal: 研究目标
        modules_path: 模块 YAML 文件所在目录
        modules: 要处理的模块列表，默认为 CORE_MODULES
        options: Ollama options
    
    Returns:
        dict: 最终合并的 JSON，格式与 generate_config.py 的 AWS 版一致
    """
    if modules is None:
        modules = CORE_MODULES
    if options is None:
        options = DEFAULT_OPTIONS.copy()
    
    # 加载所有模块的 YAML
    module_defaults = load_module_defaults(modules_path, modules)
    
    # 依次调用每个模块
    all_modifications = []
    module_results = []
    total_elapsed = 0
    
    print(f"\n  Processing {len(modules)} modules with {model}...")
    for module_name in modules:
        yaml_text = module_defaults.get(module_name, "")
        if not yaml_text:
            print(f"    ⚠ Skipping {module_name}: no YAML found")
            continue
        
        result = query_single_module(
            client, model, module_name, yaml_text,
            dataset_summary, research_goal, options
        )
        module_results.append(result)
        total_elapsed += result.get("elapsed_seconds", 0)
        
        # 收集 modifications
        for mod in result.get("modifications", []):
            all_modifications.append(mod)
    
    # 去重（按 parameter_path）
    seen_paths = set()
    unique_modifications = []
    for mod in all_modifications:
        path = mod.get("parameter_path")
        if path and path not in seen_paths:
            seen_paths.add(path)
            unique_modifications.append(mod)
    
    # 计算 summary
    counts = {"high": 0, "medium": 0, "low": 0}
    for mod in unique_modifications:
        conf = mod.get("confidence", "medium")
        if conf in counts:
            counts[conf] += 1
    
    # 构建最终 JSON
    final_json = {
        "dataset_info": {
            "name": dataset_name,
            "summary": dataset_summary.strip()[:200] + "..." if len(dataset_summary) > 200 else dataset_summary.strip(),
        },
        "modifications": unique_modifications,
        "summary": {
            "total_modifications": len(unique_modifications),
            "high_confidence": counts["high"],
            "medium_confidence": counts["medium"],
            "low_confidence": counts["low"],
        },
    }
    
    print(f"\n  ✓ Total: {len(unique_modifications)} modifications from {len(modules)} modules ({total_elapsed:.1f}s)")
    
    return {
        "final_json": final_json,
        "module_results": module_results,
        "total_elapsed_seconds": round(total_elapsed, 2),
    }


# ============================================================================
# 便捷函数：生成 CPAC 配置（供 generate_config.py 调用）
# ============================================================================
def generate_cpac_config_ollama(dataset_name: str, dataset_summary: str, research_goal: str,
                                 model: str, output_file: str = None,
                                 modules_path: str = None) -> dict:
    """
    使用 Ollama 模型生成 CPAC 配置（与 generate_config.py 的接口对齐）
    
    Args:
        dataset_name: 数据集名称
        dataset_summary: 数据集摘要（BIDS summary）
        research_goal: 研究目标
        model: Ollama 模型名称
        output_file: 输出 JSON 文件路径（可选）
        modules_path: 模块 YAML 文件所在目录（默认为 ../module_ymls）
    
    Returns:
        dict: {"config": final_json, "module_results": list, "elapsed_seconds": float}
    """
    if modules_path is None:
        modules_path = str(Path(__file__).parent.parent / "module_ymls")
    
    client = OllamaClient()
    
    result = query_all_modules(
        client=client,
        model=model,
        dataset_name=dataset_name,
        dataset_summary=dataset_summary,
        research_goal=research_goal,
        modules_path=modules_path,
    )
    
    final_json = result.get("final_json", {})
    
    # 保存到文件
    if output_file and final_json:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)
        print(f"  Saved to: {output_file}")
    
    return {
        "config": final_json,
        "module_results": result.get("module_results", []),
        "elapsed_seconds": result.get("total_elapsed_seconds", 0),
    }


# ============================================================================
# 测试入口
# ============================================================================
def main():
    """测试：用 NYU 数据集对所有 Ollama 模型进行测试"""
    print("=" * 60)
    print("Ollama CPAC Config Generator - Multi-Model Test")
    print("=" * 60)
    
    # 测试数据
    dataset_name = "ADHD-200 NYU Sample"
    dataset_summary = """
- Dataset: NYU (New York University)
- Number of Subjects: 224
- Number of Sessions: 1
- Modalities: anat (T1w), func (BOLD)
- Scanner: 3T Siemens
- TR: 2.0s
- Voxel size: 3x3x3 mm
- No fieldmap data available
"""
    research_goal = "Resting-state functional connectivity analysis with focus on reproducibility"
    
    # 输出目录
    output_dir = Path(__file__).parent / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查可用模型
    client = OllamaClient()
    available = client.list_models()
    available_names = [m["name"] for m in available.get("models", [])]
    
    print("\nAvailable models:")
    models_to_test = []
    for model in OLLAMA_MODELS:
        if model in available_names:
            print(f"  ✓ {model}")
            models_to_test.append(model)
        else:
            print(f"  ✗ {model} (not found)")
    
    if not models_to_test:
        print("\nNo models available for testing!")
        return
    
    # 对每个模型进行测试
    all_results = []
    for model in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing model: {model}")
        print(f"{'='*60}")
        
        safe_model_name = model.replace(":", "_").replace("/", "_")
        output_file = output_dir / f"{safe_model_name}_cpac_modifications.json"
        
        result = generate_cpac_config_ollama(
            dataset_name=dataset_name,
            dataset_summary=dataset_summary,
            research_goal=research_goal,
            model=model,
            output_file=str(output_file),
        )
        
        all_results.append({
            "model": model,
            "output_file": str(output_file),
            "total_modifications": result["config"].get("summary", {}).get("total_modifications", 0),
            "elapsed_seconds": result["elapsed_seconds"],
        })
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['model']}: {r['total_modifications']} modifications, {r['elapsed_seconds']}s")
        print(f"    → {r['output_file']}")
    
    # 保存汇总
    summary_file = output_dir / f"ollama_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_time": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "models_tested": models_to_test,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nComparison saved to: {summary_file}")


if __name__ == "__main__":
    main()
