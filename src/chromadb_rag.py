import chromadb
from chromadb.config import Settings
import os,time,json,requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import requests
from typing import List, Dict, Optional, Union
from embedding import generate_embeddings_for_fragment
from openai import OpenAI
from typing import Dict, List, Optional, Union, Tuple, Any

import fitz  
from concurrent.futures import ThreadPoolExecutor, as_completed

# import config          # 导入配置模块
# OpenAI API 密钥


class OllamaRAG:
    def __init__(self, persist_dir="./chroma_db", model_name="gemma3:27b", 
                 embed_model="nomic-embed-text:latest", chunk_size=1024, chunk_overlap=50,
                 ollama_base_url="http://localhost:11434"):
        """初始化RAG系统
        Args:
            persist_dir: ChromaDB持久化存储的目录
            model_name: Ollama模型名称
            embed_model: 嵌入模型名称
            chunk_size: 每个文本块的大小
            chunk_overlap: 相邻文本块的重叠部分
            ollama_base_url: Ollama API基础URL
        """

        # 设置API
        self.openai_api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
        
        self.ollama_base_url = ollama_base_url

        # 确保存储目录存在
        os.makedirs(persist_dir, exist_ok=True)
        
        # 初始化持久化的向量数据库
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                allow_reset=True,  # 允许重置集合
                is_persistent=True  # 启用持久化
            )
        )
        
        # 获取或创建collection
        try:
            self.collection = self.client.get_collection("documents")
            print(f"加载现有collection，包含{self.collection.count()}个文档")
        except:
            self.collection = self.client.create_collection("documents")
            print("创建新的collection")
        
        # 创建BM25集合 - 用于混合搜索
        try:
            self.bm25_collection = self.client.get_collection("bm25_docs")
        except:
            self.bm25_collection = self.client.create_collection("bm25_docs")
            print("创建BM25 collection用于混合搜索")



        self.persist_dir = persist_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
        )
        
        self.embeddings = OllamaEmbeddings(
            base_url="http://localhost:11434",
            model=embed_model
        )
        self.model_name = model_name
        self.embed_model = embed_model
    def get_database_info(self) -> Dict:
        """获取数据库信息"""
        return {
            "document_count": self.collection.count(),
            "persist_dir": self.persist_dir,
            "peek_documents": self.collection.peek()
        }
    
    # ------------获取知识库中所有唯一的文档ID---------------
    def get_document_ids(self) -> List[str]:
        """
        获取知识库中所有唯一的文档ID
        """
        all_ids = self.collection.get()["ids"]
        # 提取唯一的document_id（去掉fragment部分）
        unique_doc_ids = set(id.split("_fragment_")[0] for id in all_ids)
        return list(unique_doc_ids)
    
    # ------------检查文档ID是否存在---------------
    def is_pdf_embedded(self, pdf_path: str) -> bool:
        document_id = os.path.splitext(os.path.basename(pdf_path))[0]
        existing_docs = self.collection.get(where={"document_id": document_id})
        return len(existing_docs['ids']) > 0
    
    
    
    # ------------从知识库中移除指定ID的文档---------------
    def remove_document(self, document_id: str) -> None:
        """
        从知识库中移除指定ID的文档
        """
        # 获取所有与该document_id相关的fragment IDs
        all_ids = self.collection.get()["ids"]
        to_remove = [id for id in all_ids if id.startswith(f"{document_id}_fragment_")]
        
        if to_remove:
            self.collection.delete(ids=to_remove)
            print(f"已从知识库中移除文档 '{document_id}'")
        else:
            print(f"未找到文档 '{document_id}'")

        # 从BM25集合中删除
        all_bm25_ids = self.bm25_collection.get()["ids"]
        bm25_to_remove = [id for id in all_bm25_ids if id.startswith(f"{document_id}_fragment_")]
        
        if bm25_to_remove:
            self.bm25_collection.delete(ids=bm25_to_remove)
            print(f"已从BM25索引中移除文档 '{document_id}'")
        else:
            print(f"在BM25索引中未找到文档 '{document_id}'")


    # -----------------清空知识库-------------------------     
    def clear_database(self):
        """清空知识库"""
        try:
            self.client.delete_collection("documents")
            self.client.delete_collection("bm25_docs")
            self.collection = self.client.create_collection("documents")
            self.bm25_collection = self.client.create_collection("bm25_docs")
            print("知识库已清空")
        except Exception as e:
            print(f"清空数据库时出错: {e}")
    # ----------------检查PDF文件是否有效可读--------------
    def is_valid_pdf(self, pdf_path: str) -> bool:
        """检查PDF文件是否有效可读"""
        try:
            # 尝试打开并读取PDF文件
            doc = fitz.open(pdf_path)
            # 检查页数
            if doc.page_count == 0:
                print(f"警告: PDF '{pdf_path}' 没有页面")
                return False
                
            # 检查是否可以访问内容
            try:
                # 尝试读取第一页的文本
                text = doc[0].get_text()
                doc.close()
                return True
            except Exception as e:
                print(f"警告: 无法读取PDF '{pdf_path}' 的内容: {e}")
                doc.close()
                return False
                
        except Exception as e:
            print(f"警告: 无法打开PDF '{pdf_path}': {e}")
            return False
        
    # ----------向现有知识库添加新的PDF文件------------------
    def add_pdf_to_database(self, pdf_path: str, document_id: Optional[str] = None) -> None:
        """
        向现有知识库添加新的PDF文件，如果文件已存在则跳过
        
        Args:
            pdf_path: PDF文件路径
            document_id: 可选的文档ID，用于标识此PDF。如果不提供，将使用文件名
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        # 如果没有提供document_id，使用文件名（不包括扩展名）
        if document_id is None:
            document_id = os.path.splitext(os.path.basename(pdf_path))[0]

        # 检查文档是否已经存在
        existing_docs = self.collection.get(where={"document_id": document_id})
        if existing_docs['ids']:
            print(f"PDF '{document_id}' 已存在于知识库中，跳过处理")
            return

        print(f"开始处理PDF文件: {pdf_path}")

        # 验证PDF是否有效
        if not self.is_valid_pdf(pdf_path):
            raise ValueError(f"PDF文件 '{pdf_path}' 无效或已损坏")

        try:
            # 使用PyPDFLoader加载PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            print(f"PDF加载完成，开始分割文本...")

            # 将所有页面内容合并并分割
            texts = []
            metadata_list = []
            
            for page_idx, page in enumerate(pages):
                page_fragments = self.text_splitter.split_text(page.page_content)
                texts.extend(page_fragments)
                
                # 为每个文本片段创建元数据
                for _ in page_fragments:
                    metadata_list.append({
                        "source": pdf_path, 
                        "document_id": document_id,
                        "page": page_idx + 1,
                        "total_pages": len(pages),
                        "file_type": "pdf",
                        "creation_time": time.time()
                    })

            print(f"文本分割完成，共{len(texts)}个片段，开始生成向量...")

            # 存储文档片段和元数据
            successful_count = 0
            for idx, (fragment, metadata) in enumerate(zip(texts, metadata_list)):
                try:
                    # 生成当前文本片段的 embeddings
                    embeddings = generate_embeddings_for_fragment(fragment,base_url = self.ollama_base_url, model=self.embed_model)
                    
                    # 修复: 确保embeddings格式正确，如果是三维数组，提取内部的二维数组
                    if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list) and isinstance(embeddings[0][0], list):
                        # 如果是形如 [[[...]]] 的三维数组，取出内部二维数组
                        embeddings = embeddings[0]
                    
                    # 进一步确保是一维数组或二维数组
                    if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
                        # 如果是二维数组，确保只有一个向量
                        if len(embeddings) == 1:
                            embeddings = embeddings[0]  # 提取单个向量作为一维数组


                    # 存储到向量集合
                    combined_id = f"{document_id}_fragment_{idx}"
                    self.collection.add(
                        documents=[fragment],
                        embeddings=[embeddings],
                        ids=[combined_id],
                        metadatas=[metadata]
                    )
                    
                    # 同时存储到BM25集合用于混合搜索
                    self.bm25_collection.add(
                        documents=[fragment],
                        ids=[combined_id],
                        metadatas=[metadata]
                    )
                    
                    successful_count += 1
                    
                    # 每5个片段打印一次进度
                    if (idx + 1) % 5 == 0 or idx == len(texts) - 1:
                        print(f"已处理 {idx + 1}/{len(texts)} 个文档片段")
                        
                except Exception as e:
                    print(f"处理片段 {idx} 时出错: {e}")
                    # 打印嵌入向量的形状信息，帮助调试
                    try:
                        print(f"嵌入向量形状: {type(embeddings)}")
                        if isinstance(embeddings, list):
                            print(f"  第一层长度: {len(embeddings)}")
                            if len(embeddings) > 0 and isinstance(embeddings[0], list):
                                print(f"  第二层长度: {len(embeddings[0])}")
                                if len(embeddings[0]) > 0 and isinstance(embeddings[0][0], list):
                                    print(f"  第三层长度: {len(embeddings[0][0])}")
                    except:
                        print("无法打印嵌入向量信息")
                    continue  # 跳过有问题的片段，继续处理
            
            print(f"PDF '{document_id}' 处理完成，成功添加 {successful_count}/{len(texts)} 个文档片段")
            
        except Exception as e:
            print(f"处理PDF '{pdf_path}' 时出错: {e}")


    def query(self, question: str, n_results: int = 3, use_openai: bool = False,
            use_hybrid_search: bool = True) -> Dict[str, Union[str, List[str]]]:
        """
        查询知识库并生成回答
        Args:
            question: 用户问题
            n_results: 要检索的相关文档数量
            use_openai: 是否使用OpenAI API生成回答
            use_hybrid_search: 是否使用混合搜索(向量+BM25)
            
        Returns:
            包含答案和相关文档的字典
        """
        # 检索相关文档
        if use_hybrid_search:
            results = self.hybrid_search(question, n_results)
        else:
            # 常规向量搜索
            question_embedding = self.embeddings.embed_query(question)
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=n_results
            )

        if not results["documents"]:
            return {
                "answer": "没有找到相关文档", 
                "relevant_chunks": [],
                "chunk_sources": []
            }
            
        relevant_chunks = results["documents"]
        chunk_sources = [
            {
                "document_id": metadata.get("document_id", "unknown"),
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", "unknown")
            }
            for metadata in results["metadatas"]
        ]

        # 构建改进的英文提示内容
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
            + '\n'.join([f"[Doc {i+1}] " + chunk for i, chunk in enumerate(relevant_chunks)]) + "\n"
            + ('-' * 40) + "\n\n"
            
            "Question: " + question + "\n\n"
            
            "Answer:"
        )

        # 根据选择使用 Ollama 或 OpenAI
        if use_openai:
            try:
                print("选择使用OpenAI")
                openai_client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
                completion = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a precise and knowledgeable RAG assistant that provides accurate answers based solely on retrieved information."},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = completion.choices[0].message.content
            except Exception as e:
                answer = f"OpenAI 请求错误: {e}"
        else:
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={"model": self.model_name, "prompt": prompt, "stream": False}
                )
                response.raise_for_status()
                answer = response.json().get("response", "未返回答案")
            except requests.exceptions.RequestException as e:
                answer = f"Ollama 请求错误: {e}"

        # 返回结果
        return {
            "answer": answer,
            "relevant_chunks": relevant_chunks,
            "chunk_sources": chunk_sources
        }


    def hybrid_search(self, query: str, n_results: int = 3, 
                     vector_weight: float = 0.7) -> Dict[str, Any]:
        """
        结合向量搜索和BM25进行混合搜索
        
        Args:
            query: 查询文本
            n_results: 返回的结果数量
            vector_weight: 向量搜索的权重 (0-1)
            
        Returns:
            合并后的搜索结果
        """
        # 向量搜索
        query_embedding = self.embeddings.embed_query(query)
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results*2  # 检索更多结果供合并
        )
        
        # BM25搜索 (ChromaDB没有直接提供BM25，但我们可以用关键字搜索作为替代)
        bm25_results = self.bm25_collection.query(
            query_texts=[query],
            n_results=n_results*2
        )
        
        # 合并结果 - 创建得分映射
        combined_scores = {}
        
        # 处理向量结果
        if vector_results["ids"] and vector_results["ids"][0]:
            for i, doc_id in enumerate(vector_results["ids"][0]):
                # 向量距离转换为分数（距离越小越好）
                vector_score = 1.0 - vector_results["distances"][0][i] if "distances" in vector_results else 1.0
                combined_scores[doc_id] = vector_weight * vector_score
        
        # 处理BM25结果
        if bm25_results["ids"] and bm25_results["ids"][0]:
            for i, doc_id in enumerate(bm25_results["ids"][0]):
                # BM25分数（排名越前越好）
                bm25_score = 1.0 - (i / len(bm25_results["ids"][0]))
                
                # 合并分数
                if doc_id in combined_scores:
                    combined_scores[doc_id] += (1 - vector_weight) * bm25_score
                else:
                    combined_scores[doc_id] = (1 - vector_weight) * bm25_score
        
        # 排序并获取前N个结果
        top_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:n_results]
        
        if not top_ids:
            return {
                "documents": [],
                "metadatas": [],
                "ids": []
            }
        
        # 获取完整文档内容
        result_docs = []
        result_metadata = []
        
        for doc_id in top_ids:
            doc_info = self.collection.get(ids=[doc_id])
            if doc_info["documents"]:
                result_docs.append(doc_info["documents"][0])
                result_metadata.append(doc_info["metadatas"][0])
        
        return {
            "documents": result_docs,
            "metadatas": result_metadata,
            "ids": top_ids
        }
    
    def filter_by_metadata(self, 
                          document_id: Optional[str] = None,
                          min_date: Optional[float] = None,
                          max_date: Optional[float] = None,
                          page: Optional[int] = None,
                          file_type: Optional[str] = None) -> Dict[str, Any]:
        """
        根据元数据过滤知识库中的文档
        
        Args:
            document_id: 文档ID过滤
            min_date: 最早创建时间 (Unix timestamp)
            max_date: 最晚创建时间 (Unix timestamp)
            page: 特定页码
            file_type: 文件类型
            
        Returns:
            过滤后的文档集合
        """
        where_clause = {}
        
        if document_id:
            where_clause["document_id"] = document_id
        
        if page is not None:
            where_clause["page"] = page
            
        if file_type:
            where_clause["file_type"] = file_type
        
        # ChromaDB不支持范围查询，我们需要先获取所有匹配其他条件的文档
        # 然后在Python中进行日期过滤
        results = self.collection.get(where=where_clause)
        
        # 如果没有日期范围过滤，直接返回结果
        if min_date is None and max_date is None:
            return results
        
        # 过滤日期范围
        filtered_indices = []
        for i, metadata in enumerate(results["metadatas"]):
            creation_time = metadata.get("creation_time", 0)
            
            if min_date is not None and creation_time < min_date:
                continue
                
            if max_date is not None and creation_time > max_date:
                continue
                
            filtered_indices.append(i)
        
        # 应用过滤
        return {
            "ids": [results["ids"][i] for i in filtered_indices],
            "documents": [results["documents"][i] for i in filtered_indices],
            "metadatas": [results["metadatas"][i] for i in filtered_indices]
        }



class OllamaEmbeddings:
    def __init__(self, base_url="http://localhost:11434", model="nomic-embed-text:v1.5"):
        self.base_url = base_url
        self.model = model
    
    def embed_query(self, text: str) -> List[float]:
        """生成单个文本的embedding"""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        return response.json()["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成embeddings"""
        return [self.embed_query(text) for text in texts]


def call_ollama_api_generate(prompt, model, url_base="http://localhost:11434"):
    """
    调用 Ollama API 并返回生成的结果
    """
    if isinstance(prompt, list):
        prompt = str(prompt)[1:-1].replace("'", '"')  # 转换为 JSON 格式字符串

    # print(f"Calling Ollama API with prompt: {prompt}")

    url = f"{url_base}/api/generate"
    data = {
        "model": model,
        "prompt": prompt
    }

    try:
        response = requests.post(url, json=data, stream=True)
        if response.status_code != 200:
            print(f"Error response content: {response.text}")
            return f"Error: API returned status code {response.status_code}"
        
        res = ""
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if 'response' in json_response:
                    res += json_response['response']
                if json_response.get('done', False):
                    break
        return res
    except requests.RequestException as e:
        return f"Error connecting to Ollama API: {str(e)}"

def extract_keywords(question, model="gemma2:27b", url_base="http://localhost:11434"):
    """
    使用 Ollama API 解读问题并生成关键词
    
    Parameters:
    - question (str): 输入的问题
    - model (str): 使用的 Ollama 模型
    - url_base (str): Ollama API 的基础 URL
    
    Returns:
    - 提取的关键词或错误信息
    """
    prompt = f"""
    Extract searchable keywords for scientific literature related to the following question:
    {question}
    
    Provide the result in this format:
    Keywords: [(term1 OR term2) AND (term3 OR term4)] OR (another logic structure).
    """
    return call_ollama_api_generate(prompt, model, url_base)



    # def load_pdf(self, pdf_path: str, clear_existing: bool = False) -> None:
    #     """加载PDF文档并处理
        
    #     Args:
    #         pdf_path: PDF文件路径
    #         clear_existing: 是否清空现有数据
    #     """
    #     if clear_existing:
    #         self.clear_database()
        
    #     print(f"开始加载PDF文件: {pdf_path}")
        
    #     # 使用PyPDFLoader加载PDF
    #     loader = PyPDFLoader(pdf_path)
    #     pages = loader.load()
        
    #     print(f"PDF加载完成，开始分割文本...")
        
    #     # 将所有页面内容合并
    #     texts = []
    #     for page in pages:
    #         texts.extend(self.text_splitter.split_text(page.page_content))
    #     # 分割文本
    #     # texts = self.text_splitter.split_text(text)
        
    #     print(f"文本分割完成，共{len(texts)}个片段，开始生成向量...")
        
    #     # 批量生成embeddings并存储
    #     # Ollama API 设置
    #     base_url = "http://192.168.0.205:11434"
    #     model = "nomic-embed-text:v1.5"

    #     for idx, fragment in enumerate(texts):
    #         # 生成当前文本片段的 embeddings
    #         embeddings = generate_embeddings_for_fragment(
    #             fragment=fragment,
    #             base_url=base_url,
    #             model=model,
    #             max_length=1024
    #         )

    #         # 存储 embeddings 和文档
    #         self.collection.add(
    #             documents=[fragment],  # 确保文档是列表格式
    #             embeddings=embeddings,
    #             ids=[f"doc_{idx}"]  # 使用当前索引
    #         )
            
    #         print(f"已处理并持久化存储 {idx + 1}/{len(texts)} 个文档")
            
    #     print(f"文档处理完成！数据已保存至: {self.persist_dir}")