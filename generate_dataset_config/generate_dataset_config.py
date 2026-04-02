# main.py: A flexible script for generating C-PAC configurations using LLMs.
import json
import os
from pathlib import Path
from bids import BIDSLayout
import sys

# Import local modules
from bids_summary import get_bids_summary, get_subject_summary
from generate_config import generate_cpac_config, generate_cpac_config_with_ollama
from update_yml_from_json import YmlUpdaterFromJSON
from config_loader import get_config, get_research_goal

# Load configuration
config = get_config()

# Extract configuration values
USE_OLLAMA = config['backend']['use_ollama']
OLLAMA_MODELS = config['backend']['ollama_models']

# AWS Bedrock model (when not using Ollama)
if not USE_OLLAMA:
    MODELS_TO_USE = [config['backend'].get('aws_model', 'anthropic.claude-3-5-sonnet-20240620-v1:0')]
else:
    MODELS_TO_USE = OLLAMA_MODELS

# RAG Configuration
USE_RAG = config['rag']['enabled']
LITQUERY_DIR = config['rag']['litquery_dir']
RAG_DB_NAME = config['rag']['db_name']
RAG_DB_PATH = f'{LITQUERY_DIR}/{RAG_DB_NAME}'
RAG_MODEL = config['rag']['model']
RAG_EMBED_MODEL = config['rag']['embed_model']
RAG_USE_OPENAI = config['rag']['use_openai']
RAG_USE_HYBRID_SEARCH = config['rag']['use_hybrid_search']

# Path Configuration
BIDS_DATASET_PATH = config['paths']['bids_dataset_path']
YML_TEMPLATE_PATH = config['paths']['yml_template_path']
OUTPUT_ROOT_DIR = config['paths']['output_root_dir']

# Research Goal
RESEARCH_GOAL = get_research_goal(config)


def get_subject_list(bids_dir):
    """Extracts a list of subject IDs from a BIDS dataset."""
    try:
        layout = BIDSLayout(bids_dir, validate=False)
        return layout.get_subjects()
    except Exception as e:
        print(f"Error reading subject list from {bids_dir}: {e}")
        return []


def get_model_short_name(model_name):
    """Extracts a short, filesystem-safe name from the full model identifier.
    
    Examples:
        'anthropic.claude-3-5-sonnet-20240620-v1:0' -> 'claude-3-5-sonnet'
        'gpt-4.1' -> 'gpt-4.1'
        'meta.llama3-70b-instruct-v1:0' -> 'llama3-70b-instruct'
        'qwen3:32b' -> 'qwen3-32b'  (Ollama format)
        'llama3.1:70b-instruct-q4_K_M' -> 'llama3_1-70b-instruct-q4_K_M'  (Ollama format)
    """
    import re
    
    # Known provider prefixes to remove (AWS Bedrock)
    provider_prefixes = ['anthropic.', 'meta.', 'amazon.', 'cohere.', 'ai21.']
    
    name = model_name
    for prefix in provider_prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    
    # Remove version suffix (e.g., '-20240620-v1:0', '-v1:0') for AWS models
    # Pattern: remove date-like suffix and version numbers
    name = re.sub(r'-\d{8}-v\d+:\d+$', '', name)  # -20240620-v1:0
    name = re.sub(r'-v\d+:\d+$', '', name)         # -v1:0
    name = re.sub(r':\d+$', '', name)              # :0 (but not Ollama's :32b)
    
    # For Ollama models like 'qwen3:32b', replace : with -
    # But first check if it looks like an Ollama model (contains : followed by size like 32b, 70b)
    if re.search(r':\d+b', name, re.IGNORECASE):
        # This is an Ollama model format, just replace unsafe chars
        name = name.replace(':', '-').replace('.', '_')
    else:
        # Replace any remaining unsafe characters for filesystem
        name = name.replace(':', '-').replace('/', '-')
    
    return name


def get_rag_context(query):
    """跨目录调用 LitQuery 系统，根据查询获取 RAG 内容"""
    # 添加 LitQuery 的 src/rag 目录到 Python 路径
    rag_module_dir = os.path.join(LITQUERY_DIR, 'src', 'rag')
    if rag_module_dir not in sys.path:
        sys.path.append(rag_module_dir)
    
    # 添加 LitQuery 的 config 目录到 Python 路径
    config_module_dir = os.path.join(LITQUERY_DIR, 'config')
    if config_module_dir not in sys.path:
        sys.path.append(config_module_dir)
        
    try:
        from multi_db_rag import MultiDBRAG
        from chromadb_rag import OllamaRAG
    except ImportError as e:
        print(f"  - [RAG Error] 导入 RAG 模块失败: {e}. 请检查 LITQUERY_DIR 设定。")
        return ""

    print("  - [RAG] 正在初始化向量数据库和检索器...")
    multi_rag = MultiDBRAG(model_name=RAG_MODEL, ollama_base_url="http://localhost:11434")
    rag_instance = OllamaRAG(
        persist_dir=RAG_DB_PATH,
        model_name=RAG_MODEL,
        embed_model=RAG_EMBED_MODEL
    )
    multi_rag.add_rag_instance(RAG_DB_NAME, rag_instance)
    
    print(f"  - [RAG] 正在检索问题: '{query}'")
    result = multi_rag.query(
        question=query,
        n_results=5,
        use_openai=RAG_USE_OPENAI,
        use_hybrid_search=RAG_USE_HYBRID_SEARCH
    )
    return result.get("answer", "")


def main():
    """Main function to run the experiment based on the configuration."""
    dataset_name = Path(BIDS_DATASET_PATH).name
    print(f"\n{'#'*60}")
    print(f"# Starting Experiment for Dataset: {dataset_name}")
    print(f"# Total Models: {len(MODELS_TO_USE)}")
    print(f"{'#'*60}\n")

    # Get full dataset summary (只需获取一次)
    summary_text = get_bids_summary(BIDS_DATASET_PATH)
    
    # -------------------- 插入 RAG 知识阶段 --------------------
    if USE_RAG:
        print(f"\n{'='*60}")
        print("Executing RAG Knowledge Retrieval...")
        print(f"{'='*60}")
        rag_query = (
            f"Based on the following research goal, please provide expert recommendations "
            f"for fMRI preprocessing pipeline parameters (e.g., motion correction, registration, "
            f"smoothing, nuisance regression, filtering) that best suit this dataset:\n\n"
            f"Research Goal: {RESEARCH_GOAL}"
        )
        rag_knowledge = get_rag_context(rag_query)
        
        if rag_knowledge:
            rag_prompt = (
                "\n\n=======================================================\n"
                "### EXPERT LITERATURE RECOMMENDATIONS (RAG RELEVANT KNOWLEDGE) ###\n"
                "The following knowledge is retrieved from highly relevant literature regarding this dataset and analysis goal. "
                "Please carefully consider these expert recommendations and strictly incorporate them into the C-PAC YAML configuration modifications:\n\n"
                f"{rag_knowledge}\n"
                "=======================================================\n"
            )
            summary_text += rag_prompt
            print("  - [RAG] 成功提取文献知识并合并到数据集 summary 中.\n")
    # -----------------------------------------------------------
    
    # 跟踪结果
    results = {}
    
    # 循环处理每个模型
    for i, model_name in enumerate(MODELS_TO_USE, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(MODELS_TO_USE)}] Processing Model: {model_name}")
        print(f"{'='*60}")

        # Get short model name for file naming
        model_short_name = get_model_short_name(model_name)
        
        # Define output paths (organized by dataset name)
        output_dir = Path(OUTPUT_ROOT_DIR) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output file paths: {dataset}_{model}_modifications.json / _pipeline.yml
        output_json_path = output_dir / f"{dataset_name}_{model_short_name}_modifications.json"
        final_yml_path = output_dir / f"{dataset_name}_{model_short_name}_pipeline.yml"

        # Generate configuration and update YAML
        success = run_single_job(summary_text, model_name, output_json_path, final_yml_path)
        results[model_name] = success

    # 打印汇总
    print(f"\n{'='*60}")
    print("Experiment Summary:")
    print(f"{'='*60}")
    successful = sum(1 for s in results.values() if s)
    for model_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {model_name}: {status}")
    print(f"\nTotal: {successful}/{len(MODELS_TO_USE)} models successful")
    print(f"{'='*60}\n")


def run_single_job(summary_text, model_name, output_json_path, final_yml_path):
    """Generates a modifications JSON and updates the YAML for a single job.

    This function uses the new modifications-based workflow:
    1. LLM generates a JSON with only the parameters that need to be modified
    2. YAML updater applies only those modifications to the template

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"  - Generating modifications JSON: {output_json_path}")
    print(f"  - Backend: {'Ollama (Local)' if USE_OLLAMA else 'AWS Bedrock'}")

    if USE_OLLAMA:
        # 使用 Ollama 本地模型（按模块分 5 次调用）
        generated_result = generate_cpac_config_with_ollama(
            summary=summary_text,
            bids_dir=BIDS_DATASET_PATH,
            goal=RESEARCH_GOAL,
            model_name=model_name,
            output_file=str(output_json_path)
        )
    else:
        # 使用 AWS Bedrock
        generated_result = generate_cpac_config(
            summary=summary_text,
            bids_dir=BIDS_DATASET_PATH,
            goal=RESEARCH_GOAL,
            model_name=model_name,
            output_file=str(output_json_path)
        )

    # Only proceed to YAML update if the LLM returned a valid result with modifications
    if not generated_result or not generated_result.get("config"):
        print("  - Configuration generation failed or returned empty config. Skipping YAML update.")
        return False

    # Check if there are any modifications to apply
    config = generated_result.get("config", {})
    modifications = config.get("modifications", [])
    if not modifications:
        print("  - No modifications suggested by LLM. Skipping YAML update.")
        return False

    print(f"  - Found {len(modifications)} modification(s) to apply")
    print(f"  - Updating YAML from modifications: {final_yml_path}")
    try:
        yaml_updater = YmlUpdaterFromJSON(YML_TEMPLATE_PATH)
        # Use the new modifications-based method instead of the legacy method
        result = yaml_updater.process_modifications_and_save(str(output_json_path), str(final_yml_path))
        print(f"  - Update result: {result['success_count']} successful, {result['failed_count']} failed")
        return True
    except Exception as e:
        print(f"  - An error occurred during YAML update: {e}")
        return False


if __name__ == '__main__':
    main()
