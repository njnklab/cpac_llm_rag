#!/usr/bin/env python3
"""
多向量数据库检索测试脚本

功能：
1. 使用相同的问题测试4个不同的向量数据库
2. 记录每个数据库的检索时间
3. 对比检索结果
4. 保存详细报告到"检索测试"文件夹

使用方法：
    python test_vector_dbs.py
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# 添加项目根目录到路径
sys.path.insert(0, '/home/a001/zhangyan/LitQuery')

from chromadb_rag import OllamaRAG


@dataclass
class RetrievalResult:
    """单次检索结果"""
    db_name: str
    db_path: str
    doc_count: int
    query: str
    start_time: str
    end_time: str
    duration_seconds: float
    n_results: int
    use_hybrid: bool
    retrieved_chunks: List[str]
    chunk_sources: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict:
        """转换为字典（限制chunks长度以便JSON序列化）"""
        return {
            "db_name": self.db_name,
            "db_path": self.db_path,
            "doc_count": self.doc_count,
            "query": self.query,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "n_results": self.n_results,
            "use_hybrid": self.use_hybrid,
            "retrieved_count": len(self.retrieved_chunks),
            "chunk_sources": self.chunk_sources,
        }


# ==================== 用户配置区域 ====================

CONFIG = {
    # 测试问题（从文件中读取第一条）
    "query_file": "/home/a001/zhangyan/LitQuery/queries/ds002748_queries.txt",
    
    # 要测试的数据库列表
    "databases": [
        {
            "name": "minilm",
            "path": "/home/a001/zhangyan/LitQuery/serpapi_db_minilm",
            "embed_model": "all-minilm",  # 使用实际的嵌入模型名称
        },
        {
            "name": "mxbai",
            "path": "/home/a001/zhangyan/LitQuery/serpapi_db_mxbai",
            "embed_model": "mxbai-embed-large",
        },
        {
            "name": "nomic-embed-text",
            "path": "/home/a001/zhangyan/LitQuery/serpapi_db_nomic-embed-text",
            "embed_model": "nomic-embed-text:latest",
        },
        {
            "name": "qwen3",
            "path": "/home/a001/zhangyan/LitQuery/serpapi_db_qwen3",
            "embed_model": "qwen3-embedding:latest",  # qwen3数据库实际使用的嵌入模型
        },
    ],
    
    # 检索配置
    "n_results": 5,  # 检索文档数量
    "use_hybrid_search": True,  # 是否使用混合搜索
    
    # Ollama配置
    "ollama_base_url": "http://localhost:11434",
    "model_name": "qwen3:14b",  # 用于生成回答的模型
    
    # 输出配置
    "output_dir": "/home/a001/zhangyan/LitQuery/检索测试",
}

# ==================== 配置区域结束 ====================


def read_first_query(query_file: str) -> str:
    """读取问题文件的第一条问题"""
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                return line
    return ""


def test_single_db(
    db_config: Dict,
    query: str,
    n_results: int,
    use_hybrid: bool,
    ollama_url: str,
    model_name: str,
) -> RetrievalResult:
    """
    测试单个数据库的检索性能
    
    Args:
        db_config: 数据库配置字典
        query: 查询问题
        n_results: 检索数量
        use_hybrid: 是否混合搜索
        ollama_url: Ollama地址
        model_name: 模型名称
        
    Returns:
        RetrievalResult对象
    """
    db_name = db_config["name"]
    db_path = db_config["path"]
    embed_model = db_config["embed_model"]
    
    print(f"\n{'='*70}")
    print(f"测试数据库: {db_name}")
    print(f"数据库路径: {db_path}")
    print(f"嵌入模型: {embed_model}")
    print(f"{'='*70}")
    
    # 记录开始时间
    start_time = datetime.now()
    start_time_str = start_time.isoformat()
    
    try:
        # 初始化RAG系统
        print(f"[1/3] 初始化RAG系统...")
        init_start = time.time()
        rag = OllamaRAG(
            persist_dir=db_path,
            model_name=model_name,
            embed_model=embed_model,
            ollama_base_url=ollama_url
        )
        init_duration = time.time() - init_start
        doc_count = rag.collection.count()
        print(f"      ✓ 初始化完成 ({init_duration:.2f}秒)")
        print(f"      数据库包含 {doc_count} 个文档片段")
        
        # 执行检索（不生成回答，只检索）
        print(f"[2/3] 执行检索 (n_results={n_results}, hybrid={use_hybrid})...")
        retrieval_start = time.time()
        
        # 使用hybrid_search或普通query但不生成回答
        if use_hybrid:
            results = rag.hybrid_search(query, n_results=n_results)
        else:
            from chromadb_rag import OllamaEmbeddings
            embeddings = OllamaEmbeddings(
                base_url=ollama_url,
                model=embed_model
            )
            query_embedding = embeddings.embed_query(query)
            results = rag.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
        
        retrieval_duration = time.time() - retrieval_start
        print(f"      ✓ 检索完成 ({retrieval_duration:.2f}秒)")
        
        # 提取结果
        if use_hybrid:
            retrieved_chunks = results.get("documents", [])
            chunk_sources = results.get("metadatas", [])
        else:
            retrieved_chunks = results.get("documents", [[]])[0] if results.get("documents") else []
            chunk_sources = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        
        print(f"      检索到 {len(retrieved_chunks)} 个文档片段")
        
        # 记录结束时间
        end_time = datetime.now()
        end_time_str = end_time.isoformat()
        total_duration = (end_time - start_time).total_seconds()
        
        print(f"[3/3] 总耗时: {total_duration:.2f}秒")
        
        # 打印前3个结果的摘要
        print(f"\n前{min(3, len(retrieved_chunks))}个检索结果摘要:")
        for i, (chunk, source) in enumerate(zip(retrieved_chunks[:3], chunk_sources[:3])):
            doc_id = source.get("document_id", "unknown") if isinstance(source, dict) else "unknown"
            page = source.get("page", "unknown") if isinstance(source, dict) else "unknown"
            preview = chunk[:200].replace('\n', ' ') + "..." if len(chunk) > 200 else chunk
            print(f"\n  [{i+1}] 来源: {doc_id} (页 {page})")
            print(f"      内容: {preview}")
        
        return RetrievalResult(
            db_name=db_name,
            db_path=db_path,
            doc_count=doc_count,
            query=query,
            start_time=start_time_str,
            end_time=end_time_str,
            duration_seconds=total_duration,
            n_results=n_results,
            use_hybrid=use_hybrid,
            retrieved_chunks=retrieved_chunks,
            chunk_sources=chunk_sources if isinstance(chunk_sources, list) else [],
        )
        
    except Exception as e:
        end_time = datetime.now()
        end_time_str = end_time.isoformat()
        total_duration = (end_time - start_time).total_seconds()
        
        print(f"      ✗ 错误: {str(e)}")
        
        return RetrievalResult(
            db_name=db_name,
            db_path=db_path,
            doc_count=0,
            query=query,
            start_time=start_time_str,
            end_time=end_time_str,
            duration_seconds=total_duration,
            n_results=n_results,
            use_hybrid=use_hybrid,
            retrieved_chunks=[],
            chunk_sources=[{"error": str(e)}],
        )


def generate_detailed_report(results: List[RetrievalResult], output_dir: str):
    """生成详细报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存JSON格式的完整结果
    json_path = os.path.join(output_dir, f"retrieval_test_{timestamp}.json")
    json_data = {
        "test_time": datetime.now().isoformat(),
        "total_databases": len(results),
        "query": results[0].query if results else "",
        "config": {
            "n_results": CONFIG["n_results"],
            "use_hybrid_search": CONFIG["use_hybrid_search"],
            "ollama_base_url": CONFIG["ollama_base_url"],
            "model_name": CONFIG["model_name"],
        },
        "results": [r.to_dict() for r in results],
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nJSON报告已保存: {json_path}")
    
    # 2. 生成可读性更好的文本报告
    txt_path = os.path.join(output_dir, f"retrieval_test_{timestamp}.txt")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("向量数据库检索测试报告\n")
        f.write("="*80 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试配置:\n")
        f.write(f"  - 检索数量: {CONFIG['n_results']}\n")
        f.write(f"  - 混合搜索: {CONFIG['use_hybrid_search']}\n")
        f.write(f"  - Ollama地址: {CONFIG['ollama_base_url']}\n")
        f.write(f"  - 模型: {CONFIG['model_name']}\n")
        f.write("="*80 + "\n\n")
        
        f.write("测试问题:\n")
        f.write("-"*80 + "\n")
        f.write(f"{results[0].query if results else 'N/A'}\n")
        f.write("\n" + "="*80 + "\n\n")
        
        # 性能对比表
        f.write("性能对比:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'数据库':<20} {'文档数':<10} {'检索时间':<12} {'结果数':<8}\n")
        f.write("-"*80 + "\n")
        
        for r in results:
            f.write(f"{r.db_name:<20} {r.doc_count:<10} {r.duration_seconds:>10.2f}s  {len(r.retrieved_chunks):<8}\n")
        
        f.write("-"*80 + "\n\n")
        
        # 详细结果
        for r in results:
            f.write("\n" + "="*80 + "\n")
            f.write(f"数据库: {r.db_name}\n")
            f.write(f"路径: {r.db_path}\n")
            f.write(f"文档总数: {r.doc_count}\n")
            f.write(f"检索时间: {r.duration_seconds:.2f}秒\n")
            f.write(f"检索到: {len(r.retrieved_chunks)} 个文档\n")
            f.write("="*80 + "\n\n")
            
            for i, (chunk, source) in enumerate(zip(r.retrieved_chunks, r.chunk_sources)):
                if isinstance(source, dict):
                    doc_id = source.get("document_id", "unknown")
                    page = source.get("page", "unknown")
                    src_file = source.get("source", "unknown")
                else:
                    doc_id = page = src_file = "unknown"
                
                f.write(f"[{i+1}] 来源: {doc_id} | 页: {page}\n")
                f.write(f"    文件: {src_file}\n")
                f.write(f"    内容:\n")
                
                # 格式化内容，每行缩进
                content = chunk.strip()
                for line in content.split('\n'):
                    f.write(f"      {line}\n")
                f.write("\n")
    
    print(f"文本报告已保存: {txt_path}")
    
    # 3. 打印控制台摘要
    print("\n" + "="*80)
    print("测试完成！性能对比:")
    print("="*80)
    print(f"{'数据库':<20} {'文档数':<10} {'检索时间':<12} {'结果数':<8}")
    print("-"*80)
    
    for r in results:
        print(f"{r.db_name:<20} {r.doc_count:<10} {r.duration_seconds:>10.2f}s  {len(r.retrieved_chunks):<8}")
    
    print("="*80)


def main():
    print("\n" + "="*80)
    print("多向量数据库检索测试")
    print("="*80)
    
    # 读取测试问题
    query = read_first_query(CONFIG["query_file"])
    if not query:
        print(f"错误: 无法从文件读取问题: {CONFIG['query_file']}")
        return 1
    
    print(f"\n测试问题:\n{query[:100]}...")
    
    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print(f"\n输出目录: {CONFIG['output_dir']}")
    
    # 测试所有数据库
    results = []
    print(f"\n开始测试 {len(CONFIG['databases'])} 个数据库...")
    print(f"配置: n_results={CONFIG['n_results']}, hybrid={CONFIG['use_hybrid_search']}")
    
    for db_config in CONFIG["databases"]:
        result = test_single_db(
            db_config=db_config,
            query=query,
            n_results=CONFIG["n_results"],
            use_hybrid=CONFIG["use_hybrid_search"],
            ollama_url=CONFIG["ollama_base_url"],
            model_name=CONFIG["model_name"],
        )
        results.append(result)
    
    # 生成报告
    print("\n" + "="*80)
    print("生成测试报告...")
    print("="*80)
    generate_detailed_report(results, CONFIG["output_dir"])
    
    print("\n所有测试完成！")
    print(f"请查看 {CONFIG['output_dir']} 目录中的结果文件。")
    
    return 0


if __name__ == "__main__":
    exit(main())
