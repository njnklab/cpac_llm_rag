import os
import json
import httpx
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'config'))
import config
import concurrent.futures
from typing import Dict, List, Union, Any
from concurrent.futures import ThreadPoolExecutor
from anthropic import AnthropicBedrock

# AWS / Bedrock configuration
AWS_REGION = "us-east-1"
HTTP_PROXY = "http://localhost:10808"
HTTPS_PROXY = "http://localhost:10808"
class MultiDBRAG:
    """
    管理多个知识库并进行并行查询的RAG系统
    """
    
    def __init__(self, rag_instances=None, model_name="gemma3:27b", 
                 ollama_base_url="http://localhost:11434"):
        """
        初始化MultiDBRAG系统
        
        Args:
            rag_instances: 一个字典，键为数据库名称，值为OllamaRAG实例
            model_name: Ollama模型名称
            ollama_base_url: Ollama API基础URL
        """
        self.rag_instances = rag_instances or {}
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        
        # 设置OpenAI API
        self.openai_api_key = os.getenv("OPENAI_API_KEY", config.OPENAI_API_KEY)
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", config.OPENAI_BASE_URL)
        
    def add_rag_instance(self, name: str, rag_instance):
        """
        添加一个RAG实例到系统中
        
        Args:
            name: 数据库名称
            rag_instance: OllamaRAG实例
        """
        self.rag_instances[name] = rag_instance
        
    def list_databases(self):
        """返回所有已添加的数据库名称列表"""
        return list(self.rag_instances.keys())
    
    def get_database_info(self):
        """获取所有数据库的信息"""
        info = {}
        for name, rag in self.rag_instances.items():
            info[name] = rag.get_database_info()
        return info
    
    def _query_bedrock(self, prompt: str, model_name: str = 'anthropic.claude-3-5-sonnet-20240620-v1:0') -> str:
        """
        通过 AWS Bedrock 调用 LLM
        
        Args:
            prompt: 提示词
            model_name: Bedrock 模型名称
            
        Returns:
            LLM 生成的回答
        """
        try:
            # Configure optional HTTP(S) proxy
            proxies = {}
            if HTTP_PROXY:
                proxies["http://"] = HTTP_PROXY
            if HTTPS_PROXY:
                proxies["https://"] = HTTPS_PROXY

            http_client = None
            if proxies:
                http_client = httpx.Client(proxy=proxies)

            client_kwargs = {
                "aws_region": AWS_REGION,
            }
            if http_client is not None:
                client_kwargs["http_client"] = http_client

            # Prefer explicit AWS credentials from environment if available
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

            # Anthropic responses contain a list of content blocks; concatenate text blocks.
            content_blocks = getattr(response, "content", [])
            raw_content = "".join(
                getattr(block, "text", "")
                for block in content_blocks
                if getattr(block, "type", "") == "text"
            )
            
            return raw_content if raw_content else "未返回答案"
            
        except Exception as e:
            print(f"\n--- AWS Bedrock 调用出错 ---")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {e}")
            return f"AWS Bedrock 请求错误: {e}"

    def _query_single_db(self, db_name: str, question: str, n_results: int = 3, 
                         use_openai: bool = False, use_hybrid_search: bool = True):
        """
        查询单个数据库
        
        Args:
            db_name: 数据库名称
            question: 用户问题
            n_results: 要检索的相关文档数量
            use_openai: 是否使用OpenAI API生成回答
            use_hybrid_search: 是否使用混合搜索
            
        Returns:
            包含答案、相关文档和数据库名称的字典
        """
        rag_instance = self.rag_instances[db_name]
        result = rag_instance.query(
            question=question,
            n_results=n_results,
            use_openai=use_openai,
            use_hybrid_search=use_hybrid_search
        )
        
        # 添加数据库名称到结果中，便于后续识别来源
        for source in result.get("chunk_sources", []):
            source["database"] = db_name
            
        return {
            "db_name": db_name,
            "answer": result["answer"],
            "relevant_chunks": result["relevant_chunks"],
            "chunk_sources": result["chunk_sources"]
        }
    
    def query(self, question: str, n_results: int = 3, use_openai: bool = False,
              use_hybrid_search: bool = True, db_names: List[str] = None, 
              max_workers: int = None) -> Dict[str, Any]:
        """
        并行查询多个知识库并合并结果
        
        Args:
            question: 用户问题
            n_results: 每个数据库要检索的相关文档数量
            use_openai: 是否使用OpenAI API生成最终答案
            use_hybrid_search: 是否使用混合搜索
            db_names: 要查询的数据库名称列表，如果为None则查询所有数据库
            max_workers: 最大并行工作线程数，None为自动决定
            
        Returns:
            合并后的查询结果
        """
        if not self.rag_instances:
            return {
                "answer": "没有可用的知识库",
                "db_results": [],
                "all_chunks": [],
                "all_sources": []
            }
        
        # 确定要查询的数据库
        dbs_to_query = db_names if db_names else list(self.rag_instances.keys())
        
        # 并行查询各数据库
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_db = {
                executor.submit(
                    self._query_single_db,
                    db_name, 
                    question, 
                    n_results, 
                    False,  # 单个数据库查询时不生成答案，仅检索文档
                    use_hybrid_search
                ): db_name for db_name in dbs_to_query
            }
            
            for future in concurrent.futures.as_completed(future_to_db):
                db_name = future_to_db[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    print(f"查询数据库 {db_name} 时出错: {e}")
        
        # 合并所有检索到的文档
        all_chunks = []
        all_sources = []
        
        for result in all_results:
            all_chunks.extend(result["relevant_chunks"])
            all_sources.extend(result["chunk_sources"])
        
        # 根据相关性对所有文档进行重新排序(可以根据需要实现更复杂的排序逻辑)
        # 这里简单地保留原有顺序，每个数据库取前n_results个结果
        
        # 根据合并后的文档生成最终答案
        if all_chunks:
            # ========== LLM调用部分开始 ==========
            # 如需修改LLM调用方式，修改此区块内的代码
            
            # 构建提示
            prompt = (
                "You are a knowledgeable assistant that combines retrieved information with your own expertise. "
                "Answer the question thoroughly while following these guidelines:\n\n"
                
                "1. Use the retrieved documents as primary sources and supplement with your own knowledge when appropriate.\n"
                "2. When referencing information from the provided documents, cite your sources using [Doc X] format.\n"
                "3. Provide comprehensive yet concise answers with clear logical structure.\n"
                "4. Feel free to make reasonable inferences beyond the documents when helpful, but clearly distinguish between document information and your own additions.\n"
                "5. If the retrieved documents contain minimal relevant information, acknowledge this and rely more on your expertise.\n\n"
                
                "Retrieved Documents:\n"
                + ('-' * 40) + "\n"
                + '\n'.join([f"[Doc {i+1}] " + chunk for i, chunk in enumerate(all_chunks)]) + "\n"
                + ('-' * 40) + "\n\n"
                
                "Question: " + question + "\n\n"
                
                "Answer:"
            )
            
            # 使用MultiDBRAG实例的配置生成最终答案
            # 默认使用 Ollama（本地运行）
            try:
                import requests
                
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={"model": self.model_name, "prompt": prompt, "stream": False}
                )
                response.raise_for_status()
                final_answer = response.json().get("response", "未返回答案")
            except requests.exceptions.RequestException as e:
                final_answer = f"Ollama 请求错误: {e}"
            
            # 如果 Ollama 失败且指定了 use_openai，回退到 OpenAI
            if final_answer.startswith("Ollama 请求错误") and use_openai:
                try:
                    import openai
                    from openai import OpenAI
                    
                    openai_client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
                    completion = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a precise and knowledgeable RAG assistant that provides accurate answers based solely on retrieved information."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    final_answer = completion.choices[0].message.content
                except Exception as e:
                    final_answer = f"OpenAI 请求错误: {e}"
            # ========== LLM调用部分结束 ==========
        else:
            final_answer = "所有知识库中均未找到相关文档"
        
        # 返回最终结果
        return {
            "answer": final_answer,
            "db_results": all_results,  # 包含各个数据库的详细结果
            "all_chunks": all_chunks,   # 所有检索到的文档块
            "all_sources": all_sources  # 所有文档来源信息
        }


def init_multi_db_rag(config):
    """
    根据配置初始化MultiDBRAG系统
    
    Args:
        config: 配置字典，包含多个数据库的配置信息
        
    Returns:
        初始化好的MultiDBRAG实例
    """
    from chromadb_rag import OllamaRAG  # 导入您的OllamaRAG类
    
    multi_rag = MultiDBRAG()
    
    for db_name, db_config in config.items():
        rag_instance = OllamaRAG(
            persist_dir=db_config["persist_dir"],
            model_name=db_config["model_name"],
            embed_model=db_config.get("embed_model", "nomic-embed-text:latest"),
            ollama_base_url=db_config.get("ollama_base_url", "http://localhost:11434"),
            # 其他必要参数
        )
        multi_rag.add_rag_instance(db_name, rag_instance)
        
    return multi_rag