import os
import json
import argparse
from bids_summary import get_bids_summary
import httpx
from anthropic import AnthropicBedrock
import boto3
import re

# Import configuration
from config_loader import get_config

# Load configuration
config = get_config()

# Get API configuration from config
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') or config['llm_api'].get('openai_api_key')
OPENAI_BASE_URL = config['llm_api']['openai_base_url']

# AWS / Bedrock configuration
AWS_REGION = config['llm_api']['aws_region']
HTTP_PROXY = config['llm_api']['http_proxy']
HTTPS_PROXY = config['llm_api']['https_proxy']

# Ollama configuration
OLLAMA_DEFAULT_MODEL = config['backend']['ollama_default_model']


def load_module_defaults(modules_path, module_names):
    """Loads the default parameter YAML files for specified modules as raw text."""
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


# Keywords to identify path-related parameters that should be filtered out
_PATH_KEYWORDS = (
    '_path', '_template', '_mask', '_model', '_matrix', '_file',
    '/ants_template', '/ccs_template', '/cpac_templates', 's3://',
    '$FSLDIR', '.nii', '.gz', '.mat', '.model'
)

# Keywords in parameter names that indicate path-related parameters
_PATH_PARAM_KEYWORDS = (
    'template', 'mask', 'path', 'file', 'matrix', 'model',
    'identity_matrix', 'ref_mask', 'brain_mask'
)


def _is_path_related(key, value):
    """Check if a parameter is path-related and should be filtered out."""
    # Check if value contains path indicators
    value_lower = value.lower() if value else ""
    for kw in _PATH_KEYWORDS:
        if kw.lower() in value_lower:
            return True
    
    # Check if key name suggests it's a path parameter
    key_lower = key.lower() if key else ""
    # Only filter if key ends with path-related suffix AND value looks like a path
    if any(key_lower.endswith(suffix) for suffix in ('_path', '_template', '_mask', '_file', '_matrix', '_model')):
        # Check if value looks like a path (starts with / or $ or s3:// or contains .nii)
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

        # Filter out path-related parameters
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


def _format_module_reference(module_name, yaml_text):
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


def construct_prompt(bids_summary, research_goal, module_defaults):
    """Constructs the detailed prompt for the LLM to generate parameter modification suggestions."""
    best_practices = """
    **fMRI Preprocessing Best Practices:**
    1.  **Anatomical Preprocessing**: Always run brain extraction (skull-stripping). `FSL-BET` with the `robust` option is a good default. N4 bias field correction is recommended for improving segmentation and registration.
    2.  **Functional Preprocessing**: Slice timing correction is crucial, especially for event-related designs or sequences with non-interleaved slice acquisition. Use fieldmaps for distortion correction if they are available (check BIDS summary).
    3.  **Registration**: Register functional images to the subject's anatomical image, and then register the anatomical image to a standard template (like MNI152). FSL's FLIRT (for linear) and FNIRT (for non-linear) are standard choices.
    4.  **Nuisance Correction**: For resting-state fMRI, it is critical to regress out nuisance signals. This should include motion parameters (e.g., 24-parameter model), signals from CSF and white matter, and a linear trend. For task-fMRI, the requirements might be less strict.
    5.  **Segmentation**: Segment the anatomical image into gray matter (GM), white matter (WM), and cerebrospinal fluid (CSF) to aid in registration and nuisance correction.
    """

    module_reference_part = "\n\n".join(
        _format_module_reference(name, yaml_content)
        for name, yaml_content in module_defaults.items()
    )

    output_format_example = """{
  "dataset_info": {
    "name": "<dataset_name>",
    "summary": "<brief description of dataset characteristics>"
  },
  "modifications": [
    {
      "parameter_path": "<module>.<section>.<parameter>",
      "current_default": "<default value from template>",
      "recommended_value": "<your recommended value>",
      "action": "modify",
      "confidence": "high|medium|low",
      "rationale": "<detailed explanation why this change is needed based on dataset characteristics and research goal>"
    }
  ],
  "summary": {
    "total_modifications": <number>,
    "high_confidence": <number>,
    "medium_confidence": <number>,
    "low_confidence": <number>
  }
}"""

    prompt = f"""
    You are an expert in fMRI data analysis, specializing in configuring C-PAC processing pipelines. Your task is to analyze the provided dataset information and suggest **only the parameters that need to be modified** from their defaults.

    **Instructions:**
    - Analyze the BIDS dataset summary, the user's research goal, and the fMRI best practices.
    - Review the default parameters provided and identify which ones should be changed for this specific dataset.
    - **CRITICAL: ONLY include parameters where the recommended_value is DIFFERENT from current_default.**
    - **DO NOT include parameters where you recommend keeping the default value.** If current_default == recommended_value, do NOT include that parameter.
    - For each modification, provide a clear rationale explaining WHY this change is necessary based on the dataset characteristics or research goal.
    - Assign a confidence level (high/medium/low) to each recommendation.
    - Return **ONLY** a single, valid JSON object in the format specified below. Do not include any explanatory text, comments, or markdown formatting around the JSON.

    **1. BIDS Dataset Summary:**
    ```
    {bids_summary}
    ```

    **2. User's Research Goal:**
    ```
    {research_goal}
    ```

    **3. fMRI Best Practices:**
    ```
    {best_practices}
    ```

    **4. Default Module Parameters (Reference - only suggest changes to these):**
    {module_reference_part}

    **5. Required Output Format:**
    ```json
    {output_format_example}
    ```

    **Generated Parameter Modification Suggestions (JSON only):**
    """
    return prompt


def _repair_truncated_json(truncated_json):
    """Attempt to repair a truncated JSON by closing open brackets/braces.
    
    This is useful when LLM output is cut off due to max_tokens limit.
    Only repairs if the modifications array is present and has at least one complete entry.
    """
    # Check if we have at least the modifications array started
    if '"modifications"' not in truncated_json:
        return None
    
    # Find the last complete modification entry (ends with })
    # We'll try to close the JSON properly
    try:
        # Count open brackets and braces
        open_braces = truncated_json.count('{') - truncated_json.count('}')
        open_brackets = truncated_json.count('[') - truncated_json.count(']')
        
        # Find the last complete object in modifications array
        last_complete_obj = truncated_json.rfind('}')
        if last_complete_obj == -1:
            return None
        
        # Trim to last complete object and close properly
        repaired = truncated_json[:last_complete_obj + 1]
        
        # Recount after trimming
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')
        
        # Close any remaining open brackets/braces
        repaired += ']' * open_brackets
        repaired += '}' * open_braces
        
        # Validate by parsing
        json.loads(repaired)
        return repaired
    except:
        return None


def _query_anthropic(prompt, model_name):
    """Query Anthropic models via AnthropicBedrock client."""
    proxies = {}
    if HTTP_PROXY:
        proxies["http://"] = HTTP_PROXY
    if HTTPS_PROXY:
        proxies["https://"] = HTTPS_PROXY

    http_client = None
    if proxies:
        http_client = httpx.Client(proxies=proxies)

    client_kwargs = {"aws_region": AWS_REGION}
    if http_client is not None:
        client_kwargs["http_client"] = http_client

    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if aws_access_key and aws_secret_key:
        client_kwargs["aws_access_key"] = aws_access_key
        client_kwargs["aws_secret_key"] = aws_secret_key

    client = AnthropicBedrock(**client_kwargs)
    response = client.messages.create(
        model=model_name,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    content_blocks = getattr(response, "content", [])
    raw_content = "".join(
        getattr(block, "text", "")
        for block in content_blocks
        if getattr(block, "type", "") == "text"
    )
    return raw_content


def _query_bedrock_converse(prompt, model_name):
    """Query non-Anthropic models via Bedrock Converse API (boto3)."""
    # Set proxy for boto3 if needed
    if HTTP_PROXY or HTTPS_PROXY:
        os.environ['HTTP_PROXY'] = HTTP_PROXY or ''
        os.environ['HTTPS_PROXY'] = HTTPS_PROXY or ''
    
    client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
    
    # Different models have different max_tokens limits
    # Meta Llama: 2048, Mistral: 4096+, Amazon Nova: 4096+
    if model_name.startswith('meta.'):
        max_tokens = 2048
    else:
        max_tokens = 4096
    
    response = client.converse(
        modelId=model_name,
        messages=[{
            "role": "user",
            "content": [{"text": prompt}]
        }],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": 0.7
        }
    )
    
    # Extract text from response
    output = response.get("output", {})
    message = output.get("message", {})
    content = message.get("content", [])
    
    raw_content = ""
    for block in content:
        if "text" in block:
            raw_content += block["text"]
    
    return raw_content


def query_llm(prompt, api_key, base_url, model_name='anthropic.claude-3-5-sonnet-20240620-v1:0'):
    """Sends the prompt to the LLM via AWS Bedrock and returns the parsed config and raw response.

    Supports two types of models:
    - Anthropic models (model_name starts with 'anthropic.'): Uses AnthropicBedrock client
    - Other models (Meta, Mistral, Amazon, etc.): Uses boto3 Bedrock Converse API
    
    The api_key and base_url parameters are kept for backward compatibility but are
    not used by the current Bedrock-based implementation.
    """
    raw_content = None
    try:
        # Choose the appropriate API based on model provider
        if model_name.startswith('anthropic.'):
            raw_content = _query_anthropic(prompt, model_name)
        else:
            raw_content = _query_bedrock_converse(prompt, model_name)

        # Clean the response to extract only the JSON object.
        start_index = raw_content.find('{')
        end_index = raw_content.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_content = raw_content[start_index:end_index+1]
            return json.loads(json_content), raw_content
        else:
            # Try to repair truncated JSON (common with low max_tokens models)
            if start_index != -1:
                truncated_json = raw_content[start_index:]
                repaired = _repair_truncated_json(truncated_json)
                if repaired:
                    print("  [WARNING] JSON was truncated, attempting repair...")
                    return json.loads(repaired), raw_content
            raise ValueError("Could not find a valid JSON object in the LLM response.")
    except Exception as e:
        print("\n--- An unexpected error occurred while querying the LLM via AWS Bedrock ---")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        if raw_content:
            print("LLM Response content that failed parsing:\n---\n", raw_content, "\n---")
        print("--------------------------------------------------------------------\n")
        return None, raw_content


def generate_cpac_config(summary, bids_dir, goal, model_name, api_key=None, base_url=None, output_file=None):
    """Generates a C-PAC configuration using an LLM and returns debug info.

    Returns:
        dict: A dictionary containing 'config', 'prompt', and 'raw_response'.
              Returns None if a critical error occurs.

    Note:
        The api_key and base_url arguments are retained for backward compatibility
        but are not used by the current AWS Bedrock-based implementation.
    """
    core_modules = config['core_modules']
    module_ymls_path = os.path.join(os.path.dirname(__file__), config['paths']['module_ymls_dir'])

    print("1. Loading default module parameters from YMLs...")
    module_defaults = load_module_defaults(module_ymls_path, core_modules)
    print("   ...Done.")

    print("2. Constructing prompt for LLM...")
    prompt = construct_prompt(summary, goal, module_defaults)
    print("   ...Done.")

    print(f"4. Querying LLM ({model_name}) to generate configuration...")
    llm_config, raw_response = query_llm(prompt, api_key, base_url, model_name=model_name)
    print("   ...Done.")

    if llm_config and output_file:
        try:
            print(f"5. Saving configuration to: {output_file}")
            with open(output_file, 'w') as f:
                json.dump(llm_config, f, indent=4)
            print("   ...Done.")

            # Also save the prompt used for this configuration next to the JSON file
            prompt_output_path = os.path.splitext(output_file)[0] + "_prompt.txt"
            print(f"6. Saving prompt to: {prompt_output_path}")
            with open(prompt_output_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            print("   ...Done.")
        except IOError as e:
            print(f"Error: Could not save configuration or prompt. Reason: {e}")

    return {
        "config": llm_config,
        "prompt": prompt,
        "raw_response": raw_response
    }


def generate_cpac_config_with_ollama(summary, bids_dir, goal, model_name=None, output_file=None):
    """使用 Ollama 本地模型生成 CPAC 配置（按模块分 5 次调用）。
    
    Args:
        summary: BIDS 数据集摘要
        bids_dir: BIDS 数据集路径
        goal: 研究目标
        model_name: Ollama 模型名称，默认使用配置中的 ollama_default_model
        output_file: 输出 JSON 文件路径
    
    Returns:
        dict: {"config": final_json, "module_results": list, "elapsed_seconds": float}
    """
    from pathlib import Path
    from ollama.ollama_cpac import generate_cpac_config_ollama
    
    if model_name is None:
        model_name = OLLAMA_DEFAULT_MODEL
    
    dataset_name = Path(bids_dir).name
    
    result = generate_cpac_config_ollama(
        dataset_name=dataset_name,
        dataset_summary=summary,
        research_goal=goal,
        model=model_name,
        output_file=output_file,
    )
    
    return result


def main():
    """Main function to run the configuration generation process for testing."""
    from pathlib import Path
    
    # Get configuration values
    test_bids_dir = config['paths']['bids_dataset_path']
    test_goal = """
        This dataset contains resting-state fMRI data from 51 patients with mild depression 
        and 21 healthy controls. The primary research goal is to identify neuroimaging biomarkers 
        that can distinguish depression patients from healthy controls using functional connectivity 
        analysis. We aim to extract robust features such as ReHo, fALFF, and seed-based connectivity 
        for building classification models. Special attention should be paid to motion correction 
        and nuisance regression to ensure signal quality for clinical applications.
    """
    
    # 选择使用 AWS Bedrock 还是 Ollama
    use_ollama = config['backend']['use_ollama']
    
    if use_ollama:
        # Ollama 模式：测试多个本地模型
        test_models = [config['backend']['ollama_default_model']]
        output_base_dir = Path(__file__).parent / 'dataset_yaml' / 'ollama_test'
    else:
        # AWS Bedrock 模式
        test_models = [config['backend'].get('aws_model', 'anthropic.claude-3-5-sonnet-20240620-v1:0')]
        dataset_name = Path(test_bids_dir).name
        output_base_dir = Path(__file__).parent / 'dataset_yaml' / dataset_name
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Testing Parameter Modification Suggestions")
    print(f"=" * 60)
    print(f"Dataset: {test_bids_dir}")
    print(f"Mode:    {'Ollama (Local)' if use_ollama else 'AWS Bedrock'}")
    print(f"Models:  {test_models}")
    print(f"Output:  {output_base_dir}")
    print(f"=" * 60)

    # Generate BIDS summary
    try:
        print(f"\n1. Generating BIDS summary for {test_bids_dir}...")
        summary_text = get_bids_summary(test_bids_dir)
        print("   ...Done.")
    except Exception as e:
        print(f"   Could not generate BIDS summary. Error: {e}")
        return

    # 对每个模型进行测试
    all_results = []
    for test_model in test_models:
        safe_model_name = test_model.replace(":", "_").replace("/", "_").replace(".", "_")
        test_output_file = str(output_base_dir / f'{safe_model_name}_modifications.json')
        
        print(f"\n{'='*60}")
        print(f"Testing model: {test_model}")
        print(f"Output: {test_output_file}")
        print(f"{'='*60}")
        
        if use_ollama:
            # 使用 Ollama 本地模型
            result = generate_cpac_config_with_ollama(
                summary=summary_text,
                bids_dir=test_bids_dir,
                goal=test_goal,
                model_name=test_model,
                output_file=test_output_file
            )
        else:
            # 使用 AWS Bedrock
            result = generate_cpac_config(
                summary=summary_text,
                bids_dir=test_bids_dir,
                goal=test_goal,
                model_name=test_model,
                output_file=test_output_file
            )
        
        if result and result.get("config"):
            config = result["config"]
            total_mods = len(config.get('modifications', []))
            elapsed = result.get('elapsed_seconds', 'N/A')
            print(f"\n✓ SUCCESS: {total_mods} modifications suggested")
            if elapsed != 'N/A':
                print(f"  Elapsed: {elapsed}s")
            
            all_results.append({
                "model": test_model,
                "output_file": test_output_file,
                "total_modifications": total_mods,
                "elapsed_seconds": elapsed,
            })
            
            # Print summary of modifications
            if "modifications" in config:
                print(f"\n--- Modification Summary ---")
                for i, mod in enumerate(config['modifications'][:5], 1):  # 只显示前 5 条
                    print(f"  [{i}] {mod.get('parameter_path', 'N/A')}")
                    print(f"      {mod.get('current_default', 'N/A')} -> {mod.get('recommended_value', 'N/A')}")
                if len(config['modifications']) > 5:
                    print(f"  ... and {len(config['modifications']) - 5} more modifications")
        else:
            print(f"\n✗ FAILED: Configuration generation failed for {test_model}")
            all_results.append({
                "model": test_model,
                "output_file": test_output_file,
                "total_modifications": 0,
                "error": "Generation failed",
            })
    
    # 打印汇总
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        status = "✓" if r.get('total_modifications', 0) > 0 else "✗"
        elapsed_str = f", {r.get('elapsed_seconds')}s" if r.get('elapsed_seconds') else ""
        print(f"  {status} {r['model']}: {r.get('total_modifications', 0)} modifications{elapsed_str}")
        print(f"    → {r['output_file']}")
    print(f"{'='*60}")
    print("\nProcess complete.")


if __name__ == '__main__':
    main()
