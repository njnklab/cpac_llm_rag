import requests
import json
import logging
from math import ceil

# 获取logger
logger = logging.getLogger("build_embed_db")

# 不同embed模型的context length限制 (字符数，约为token数的4倍)
MODEL_MAX_LENGTH = {
    "all-minilm": 200,           # ~256 tokens, 保守设置200字符
    "mxbai-embed-large": 400,    # ~512 tokens
    "qwen3-embedding": 2000,     # ~8192 tokens
    "nomic-embed-text": 2000,    # ~8192 tokens
}

def get_max_length_for_model(model: str) -> int:
    """根据模型名称获取合适的max_length"""
    for key, length in MODEL_MAX_LENGTH.items():
        if key in model.lower():
            return length
    return 512  # 默认值

def generate_embeddings_for_fragment(fragment, base_url, model, max_length=None):
    """
    为单个文本片段生成embeddings，如果文本超过最大长度则分段处理
    """
    # 创建session并禁用环境变量代理
    session = requests.Session()
    session.trust_env = False
    
    # 如果未指定max_length，根据模型自动设置
    if max_length is None:
        max_length = get_max_length_for_model(model)
    
    if len(fragment) <= max_length:
        sub_fragments = [fragment]
    else:
        n_splits = ceil(len(fragment) / max_length)
        sub_fragments = []
        for i in range(n_splits):
            start = i * max_length
            end = min((i + 1) * max_length, len(fragment))
            sub_fragments.append(fragment[start:end])
    
    all_embeddings = []
    
    for sub_fragment in sub_fragments:
        try:
            # 禁用代理，直接连接Ollama
            response = session.post(
                f"{base_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": sub_fragment  
                }
            )
            
            if response.status_code == 200:
                try:
                    embeddings = response.json().get("embedding")
                    if embeddings:
                        all_embeddings.extend([embeddings])      # 注意这里的修改，因为每次返回一个向量
                    else:
                        logger.warning(f"子片段未生成embeddings。响应内容: {response.json()}")
                except KeyError:
                    logger.error(f"响应格式不正确。响应内容: {response.json()}")
            else:
                logger.error(f"API调用失败，状态码：{response.status_code}, 响应内容: {response.text}")
                
        except Exception as e:
            logger.error(f"处理子片段时发生错误: {e}")
            continue
            
    return all_embeddings
