import requests
import json
from math import ceil

def generate_embeddings_for_fragment(fragment, base_url, model, max_length=1024):
    """
    为单个文本片段生成embeddings，如果文本超过最大长度则分段处理
    """
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
            response = requests.post(
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
                        print(f"警告: 子片段未生成embeddings。响应内容: {response.json()}")
                except KeyError:
                    print(f"错误: 响应格式不正确。响应内容: {response.json()}")
            else:
                print(f"API调用失败，状态码：{response.status_code}, 响应内容: {response.text}")
                
        except Exception as e:
            print(f"处理子片段时发生错误: {e}")
            continue
            
    return all_embeddings
