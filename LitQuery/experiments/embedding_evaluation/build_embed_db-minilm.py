"""
用于构建不同embed_model知识库的脚本
使用方法: 修改下方配置参数后运行
"""
import os
import logging
from datetime import datetime
from typing import List
import sys
sys.path.insert(0, '/mnt/sda1/zhangyan/llm-cpac/LitQuery/src/rag')
from chromadb_rag import OllamaRAG

# 日志配置
LOG_DIR = "/home/a001/zhangyan/LitQuery/log"
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(embed_model: str):
    """根据embed_model名称创建日志文件"""
    model_name = embed_model.replace(":", "_").replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"build_{model_name}_{timestamp}.log")
    
    logger = logging.getLogger("build_embed_db")
    logger.setLevel(logging.INFO)
    
    # 清除已有handler
    logger.handlers.clear()
    
    # 文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志文件: {log_file}")
    return logger

def scan_pdf_directory(directory: str) -> List[str]:
    """扫描指定目录及其子目录下的所有PDF文件"""
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def main():
    # ============ 配置参数 - 请根据需要修改 ============
    
    # PDF源文件目录 (与serpapi_db使用相同的文档)
    pdf_dir = "/home/a001/zhangyan/LitQuery/LitCollector/papers_pdfs"
    
    # 知识库保存目录 - 每个embed_model使用不同目录
    db_dir = "/home/a001/zhangyan/LitQuery/向量数据库/1024-200/serpapi_db_minilm"
    
    # embed_model选择
    embed_model = "all-minilm:latest"           # 45MB, 最快
    # embed_model = "mxbai-embed-large:latest"  # 669MB, 中等
    # embed_model = "qwen3-embedding:latest"    # 4.7GB, 最慢但可能最好
    # nomic-embed-text:latest
    
    # 分块参数 
    chunk_size = 1024
    chunk_overlap = 200
    
    
    # LLM模型 (用于后续查询，构建时不使用)
    model_name = "gemma3:27b"
    # ============ 配置结束 ============
    
    # 初始化日志
    logger = setup_logger(embed_model)
    
    logger.info("=" * 60)
    logger.info("知识库构建配置:")
    logger.info(f"  PDF目录: {pdf_dir}")
    logger.info(f"  知识库目录: {db_dir}")
    logger.info(f"  Embed模型: {embed_model}")
    logger.info(f"  chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")
    logger.info("=" * 60)
    
    # 初始化RAG系统
    rag = OllamaRAG(
        persist_dir=db_dir,
        model_name=model_name,
        embed_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # 扫描PDF文件
    pdf_files = scan_pdf_directory(pdf_dir)
    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    
    # 检查哪些文件需要处理
    new_files = []
    for pdf_path in pdf_files:
        if not rag.is_pdf_embedded(pdf_path):
            new_files.append(pdf_path)
    
    logger.info(f"发现 {len(new_files)} 个新的PDF文件需要处理")
    
    if not new_files:
        logger.info("所有PDF文件已处理完毕，无需重新构建")
        info = rag.get_database_info()
        logger.info(f"知识库信息:")
        logger.info(f"- 文档片段数: {info['document_count']}")
        logger.info(f"- 存储位置: {info['persist_dir']}")
        return
    
    # 处理PDF文件
    successful = 0
    failed = 0
    failed_files = []
    start_time = datetime.now()
    
    for i, pdf_path in enumerate(new_files):
        try:
            logger.info(f"正在处理 [{i+1}/{len(new_files)}]: {os.path.basename(pdf_path)}")
            rag.add_pdf_to_database(pdf_path)
            logger.info(f"成功添加: {os.path.basename(pdf_path)}")
            successful += 1
        except Exception as e:
            logger.error(f"处理失败: {os.path.basename(pdf_path)}")
            logger.error(f"错误信息: {e}")
            failed += 1
            failed_files.append(pdf_path)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # 显示最终统计
    logger.info("=" * 60)
    logger.info("构建完成!")
    logger.info(f"  成功: {successful} 个文件")
    logger.info(f"  失败: {failed} 个文件")
    logger.info(f"  耗时: {duration}")
    
    if failed_files:
        logger.warning("失败文件列表:")
        for f in failed_files:
            logger.warning(f"  - {f}")
    
    info = rag.get_database_info()
    logger.info(f"知识库信息:")
    logger.info(f"- 文档片段数: {info['document_count']}")
    logger.info(f"- 存储位置: {info['persist_dir']}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
