import os
from multi_db_rag import MultiDBRAG
from chromadb_rag import OllamaRAG  


def read_questions_from_file(file_path):
    """
    从txt文件读取问题列表，每行一个问题
    
    Args:
        file_path: 问题文件路径
    Returns:
        问题列表
    """
    questions = []
    if not os.path.exists(file_path):
        print(f"问题文件不存在: {file_path}")
        return questions
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # 跳过空行和注释
                questions.append(line)
    
    print(f"从文件读取到 {len(questions)} 个问题")
    return questions


def scan_pdf_directory(pdf_dir):
    """
    扫描目录下的所有PDF文件
    """
    pdf_files = []
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def process_pdf_files(rag_instance, pdf_files):
    """处理一批PDF文件"""
    new_files = []
    for pdf_path in pdf_files:
        if not rag_instance.is_pdf_embedded(pdf_path):
            new_files.append(pdf_path)
    
    print(f"发现{len(new_files)}个新的PDF文件需要处理")
    
    successful = 0
    for i, pdf_path in enumerate(new_files):
        try:
            print(f"正在处理 [{i+1}/{len(new_files)}]: {pdf_path}")
            rag_instance.add_pdf_to_database(pdf_path)
            print(f"成功添加: {pdf_path}")
            successful += 1
        except Exception as e:
            print(f"处理失败: {pdf_path}")
            print(f"错误信息: {e}")
    
    return successful, len(new_files)

def main():

    # 设置查询参数 - 直接通过变量设置，无需终端交互
    use_openai = True  # 是否使用OpenAI API
    db_names = None    # None表示查询所有数据库，或指定如['serpapi_db']

    # 多数据库配置
    db_configs = {
        "serpapi_db": {
            "persist_dir": "/home/a001/zhangyan/LitQuery/serpapi_db",
            "model_name": "gemma3:27b",
            "embed_model": "nomic-embed-text:latest",
            "pdf_dir": "/home/a001/zhangyan/LitCollector/papers_pdfs"
        },
        "arxiv_db": {
            "persist_dir": "/home/a001/zhangyan/LitQuery/arxiv_db",
            "model_name": "gemma3:27b",
            "embed_model": "nomic-embed-text:latest",
            "pdf_dir": "/home/a001/zhangyan/LitCollector/arxiv_pdfs"
        }
    }
    
    # 初始化MultiDBRAG系统
    multi_rag = MultiDBRAG(
        model_name="gemma3:27b", 
        ollama_base_url="http://localhost:11434"
    )
    
    # 初始化并添加各个RAG实例
    for db_name, config in db_configs.items():
        rag_instance = OllamaRAG(
            persist_dir=config["persist_dir"],
            model_name=config["model_name"],
            embed_model=config["embed_model"]
        )
        multi_rag.add_rag_instance(db_name, rag_instance)
        print(f"\n[{db_name}] 已添加到多数据库系统")
    
    # 显示所有知识库信息
    db_info = multi_rag.get_database_info()
    print("\n所有知识库信息:")
    for db_name, info in db_info.items():
        print(f"\n--- {db_name} ---")
        print(f"- 文档数量: {info['document_count']}")
        print(f"- 存储位置: {info['persist_dir']}")
    
    # 运行多知识库查询操作
    print("\n----- 运行多知识库并行查询操作 -------")
    
    # 从txt文件读取问题（每行一个问题）
    questions_file = "/home/a001/zhangyan/LitQuery/questions.txt"
    questions = read_questions_from_file(questions_file)
    
    if not questions:
        print("没有找到问题，请在 questions.txt 中添加问题（每行一个）")
        return
    
    for idx, question in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"正在处理第 {idx+1}/{len(questions)} 个问题")
        print(f"{'='*60}")
        
        result = multi_rag.query(
            question=question,
            n_results=5,
            use_openai=use_openai,
            use_hybrid_search=True,
            db_names=db_names,
            max_workers=len(multi_rag.rag_instances) if db_names is None else len(db_names)
        )
        
        print("\n" + "="*50)
        print("查询问题:", question)
        print("="*50)
        print("\n回答:")
        print(result["answer"])
        
        print("\n" + "-"*50)
        print("参考文档片段:")
        for i, (chunk, source) in enumerate(zip(result["all_chunks"], result["all_sources"])):
            print(f"\n[{i+1}] 数据库: {source['database']}")
            print(f"来源: {source['document_id']}")
            print(f"内容概要: {chunk[:150]}..." if len(chunk) > 150 else chunk)

        # 输出各数据库查询统计
        db_stats = {}
        for db_result in result["db_results"]:
            db_name = db_result["db_name"]
            chunk_count = len(db_result["relevant_chunks"])
            db_stats[db_name] = chunk_count
        
        print("\n" + "-"*50)
        print("数据库查询统计:")
        for db_name, count in db_stats.items():
            print(f"- {db_name}: 检索到 {count} 个相关文档")

if __name__ == "__main__":
    main()