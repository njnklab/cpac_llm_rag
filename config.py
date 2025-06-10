#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPAC项目配置文件
"""

# =============================================================================
# 基础配置
# =============================================================================

# 数据集名称
DATASET_NAME = "ds002748"

# 处理列表 - 指定要处理的被试
SUBJECT_LIST = ["sub-01", "sub-02", "sub-03"]  # 具体被试列表
# SUBJECT_LIST = "all"  # 或者设为 "all" 处理所有被试


# 分析需求 - 指定需要进行的分析类型
ANALYSIS_REQUIREMENTS = {
    "functional_connectivity": True,     # 功能连接分析
    "regional_homogeneity": True,        # 局部一致性分析
    "alff": False,                       # 低频振幅分析
    "reho": False,                       # 区域同质性分析
    "seed_correlation": False,           # 种子相关分析
}

# =============================================================================
# 配置模式选择
# =============================================================================

# 配置模式: "default", "llm", "rag"
CONFIG_MODE = "rag"

# LLM配置 (当CONFIG_MODE="llm"时使用)  gemma3  deepseek
LLM_MODEL = "gemma3:27b"               # 本地Ollama模型名称
OLLAMA_URL = "http://localhost:11434"  # Ollama服务地址

# RAG配置 (当CONFIG_MODE="rag"时使用)
CHROMADB_PATH = "/home/a001/zhangyan/LitQuery/serpapi_db"  # ChromaDB数据库路径
RAG_COLLECTION = "serpapi_db"                              # 知识库集合名称

# =============================================================================
# 路径配置
# =============================================================================

# 数据路径
DATA_PATH = "/home/a001/zhangyan/openneuro"
OUTPUT_PATH = "/home/a001/zhangyan/cpac/output"

# 日志文件
LOG_FILE = "cpac_processing.log"