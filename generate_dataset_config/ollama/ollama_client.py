import requests
import json

# 禁用代理，确保直接访问本地 Ollama 服务
NO_PROXY = {"http": None, "https": None}


class OllamaClient:
    """
    Ollama API客户端，用于与本地Ollama服务通信
    """
    
    def __init__(self, host="localhost", port=11434):
        """
        初始化Ollama客户端
        
        Args:
            host (str): Ollama服务器主机名
            port (int): Ollama服务器端口号
        """
        self.base_url = f"http://{host}:{port}"
    
    def list_models(self):
        """
        获取所有已安装的模型列表
        
        Returns:
            dict: 包含模型信息的字典
        """
        response = requests.get(f"{self.base_url}/api/tags", proxies=NO_PROXY)
        return response.json()
    
    def generate(self, model, prompt, options=None, stream=False, format=None):
        """
        使用指定模型生成回复
        
        Args:
            model (str): 模型名称，例如 "llama3.1:70b-instruct-q4_K_M"
            prompt (str): 提示词
            options (dict): Ollama 原生 options，支持:
                - num_ctx: 上下文窗口大小 (e.g. 32768)
                - num_predict: 最大生成 token 数 (e.g. 4096)
                - num_gpu: GPU offload 层数 (-1 表示尽量全部)
                - num_batch: 批处理大小 (e.g. 512)
                - seed: 随机种子，用于复现
                - temperature: 采样温度 (0~1)
                - top_p: nucleus sampling
                - top_k: top-k sampling
                - repeat_penalty: 重复惩罚
            stream (bool): 是否以流的方式返回结果
            format (str): 输出格式，例如 "json" 强制 JSON 输出
            
        Returns:
            如果stream=False，返回完整响应文本
            如果stream=True，返回一个生成器，可以逐步获取响应
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        if options:
            payload["options"] = options
        if format:
            payload["format"] = format
        
        if stream:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, stream=True, proxies=NO_PROXY)
            def generate_stream():
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        try:
                            json_line = json.loads(decoded_line)
                            response_part = json_line.get('response', '')
                            full_response += response_part
                            yield json_line
                        except json.JSONDecodeError:
                            print(f"警告: 无法解析JSON行: {decoded_line}")
                # 返回完整响应
                return {"full_response": full_response}
                            
            return generate_stream()
        else:
            # 非流式处理，获取完整响应
            full_response = ""
            response = requests.post(f"{self.base_url}/api/generate", json=payload, stream=True, proxies=NO_PROXY)
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        json_line = json.loads(decoded_line)
                        response_part = json_line.get('response', '')
                        full_response += response_part
                    except json.JSONDecodeError:
                        print(f"警告: 无法解析JSON行: {decoded_line}")
                        
            return {"response": full_response}
    
    def chat(self, model, messages, temperature=0.7, max_tokens=2048, stream=False):
        """
        使用指定模型进行聊天
        
        Args:
            model (str): 模型名称，例如 "gemma3:27b"
            messages (list): 消息列表，格式为 [{"role": "user", "content": "Hello"}, ...]
            temperature (float): 采样温度，控制输出的随机性
            max_tokens (int): 生成的最大token数量
            stream (bool): 是否以流的方式返回结果
            
        Returns:
            如果stream=False，返回完整响应文本
            如果stream=True，返回一个生成器，可以逐步获取响应
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        if stream:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, proxies=NO_PROXY)
            def generate_stream():
                full_content = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        try:
                            json_line = json.loads(decoded_line)
                            if 'message' in json_line:
                                content_part = json_line['message'].get('content', '')
                                full_content += content_part
                            yield json_line
                        except json.JSONDecodeError:
                            print(f"警告: 无法解析JSON行: {decoded_line}")
                # 返回完整响应
                return {"message": {"content": full_content}}
                            
            return generate_stream()
        else:
            # 非流式处理，获取完整响应
            full_content = ""
            response = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, proxies=NO_PROXY)
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        json_line = json.loads(decoded_line)
                        if 'message' in json_line:
                            content_part = json_line['message'].get('content', '')
                            full_content += content_part
                    except json.JSONDecodeError:
                        print(f"警告: 无法解析JSON行: {decoded_line}")
                        
            return {"message": {"content": full_content}}

# 使用示例
if __name__ == "__main__":
    # 创建客户端实例
    client = OllamaClient()
    
    # 列出所有模型
    models = client.list_models()
    print("已安装的模型:")
    for model in models.get("models", []):
        print(f"- {model['name']}")
    
    # 简单问答示例 (使用generate API)
    print("\n使用generate API的问答示例:")
    response = client.generate(
        model="gemma3:27b",
        prompt="什么是人工智能?"
    )
    print(f"问题: 什么是人工智能?")
    print(f"回答: {response.get('response', '')}")
    
    # 聊天示例 (使用chat API)
    print("\n使用chat API的问答示例:")
    chat_response = client.chat(
        model="gemma3:27b",
        messages=[
            {"role": "user", "content": "请用简单的语言解释量子计算"}
        ]
    )
    print(f"问题: 请用简单的语言解释量子计算")
    print(f"回答: {chat_response.get('message', {}).get('content', '')}")