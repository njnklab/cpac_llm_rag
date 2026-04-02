import os
from typing import List
from chromadb_rag import OllamaRAG

def scan_pdf_directory(directory: str) -> List[str]:
    """扫描指定目录及其子目录下的所有PDF文件"""
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def main():
    # 配置参数
    pdf_dir = "/home/a001/zhangyan/LitCollector/papers_pdfs"
    db_dir = "/home/a001/zhangyan/LitQuery/serpapi_db"
    model_name = "gemma3:27b"
    embed_model = "nomic-embed-text:latest"
    
    # 初始化RAG系统
    rag = OllamaRAG(
        persist_dir=db_dir,
        model_name=model_name,
        embed_model=embed_model
    )
    
    # 1. 检查并处理新的PDF文件
    pdf_files = scan_pdf_directory(pdf_dir)
    print(f"找到{len(pdf_files)}个PDF文件")
    
    new_files = []
    for pdf_path in pdf_files:
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        if not rag.is_pdf_embedded(pdf_path):
            new_files.append(pdf_path)
    
    print(f"发现{len(new_files)}个新的PDF文件需要处理")
    
    if new_files:
        for i, pdf_path in enumerate(new_files):
            try:
                print(f"正在处理 [{i+1}/{len(new_files)}]: {pdf_path}")
                rag.add_pdf_to_database(pdf_path)
                print(f"成功添加: {pdf_path}")
            except Exception as e:
                print(f"处理失败: {pdf_path}")
                print(f"错误信息: {e}")
    
    # 2. 显示知识库信息
    info = rag.get_database_info()
    print(f"\n知识库信息:")
    print(f"- 文档数量: {info['document_count']}")
    print(f"- 存储位置: {info['persist_dir']}")
    
    # # 3. 运行查询操作
    # print("-----运行查询操作-------")
    # while True:
    #     question = input("\n请输入您的问题 (输入'q'退出): ")
    #     if question.lower() == 'q':
    #         break
        
    #     result = rag.query(
    #         question=question,
    #         n_results=3,
    #         use_openai=True,
    #         use_hybrid_search=True  
    #     )
        
    #     print("\n" + "="*50)
    #     print("查询问题:", question)
    #     print("="*50)
    #     print("\n回答:")
    #     print(result["answer"])
        
    #     print("\n" + "-"*50)
    #     print("参考文档片段:")
    #     for i, (chunk, source) in enumerate(zip(result["relevant_chunks"], result["chunk_sources"])):
    #         print(f"\n[{i+1}] 来源: {source['document_id']}")
    #         print(f"内容概要: {chunk[:150]}..." if len(chunk) > 150 else chunk)

if __name__ == "__main__":
    main()