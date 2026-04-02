import requests
import json

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
        response = requests.get(f"{self.base_url}/api/tags")
        return response.json()
    
    def generate(self, model, prompt, temperature=0.7, max_tokens=2048, stream=False):
        """
        使用指定模型生成回复
        
        Args:
            model (str): 模型名称，例如 "gemma3:27b"
            prompt (str): 提示词
            temperature (float): 采样温度，控制输出的随机性
            max_tokens (int): 生成的最大token数量
            stream (bool): 是否以流的方式返回结果
            
        Returns:
            如果stream=False，返回完整响应文本
            如果stream=True，返回一个生成器，可以逐步获取响应
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        if stream:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, stream=True)
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
            response = requests.post(f"{self.base_url}/api/generate", json=payload, stream=True)
            
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
            response = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True)
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
            response = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True)
            
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