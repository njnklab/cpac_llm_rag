


from openai import OpenAI
from typing import List, Dict, Any, Optional


class OpenAIClient:
    """
    简单的OpenAI API调用封装类
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        初始化OpenAI客户端
        
        Args:
            api_key (str): OpenAI API密钥
            base_url (str, optional): API基础URL
        """

        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def chat(self, messages: List[Dict[str, str]], model: str = "gpt-4o", **kwargs) -> str:
        """
        调用OpenAI聊天API
        
        Args:
            messages (List[Dict[str, str]]): 消息列表，格式如 [{"role": "user", "content": "hello"}]
            model (str): 模型名称，默认gpt-4o
            **kwargs: 其他OpenAI API参数（如temperature、max_tokens等）
            
        Returns:
            str: AI回复内容
            
        Raises:
            Exception: API调用失败时抛出异常
        """
        try:
            print("选择使用OpenAI")
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI 请求错误: {e}")


# 使用示例
if __name__ == "__main__":
    # 初始化客户端
    # OpenAI API 密钥
    OPENAI_API_KEY = "xxx"
    OPENAI_BASE_URL = "xxx"
    client = OpenAIClient(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL  
    )
    
    # 调用示例
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        response = client.chat(
            messages=messages,
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1000
        )
        
        print(f"回复: {response}")
        
    except Exception as e:
        print(f"错误: {e}")