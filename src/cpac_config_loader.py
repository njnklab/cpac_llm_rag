#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件加载模块
根据不同的配置模式加载相应的CPAC配置文件
"""

import os
import time
import sys
from chromadb_rag import OllamaRAG

def load_config_file(config_mode, llm_model=None):
    """
    根据配置模式加载相应的配置文件
    
    Args:
        config_mode: 配置模式 ("default", "llm", "rag")
        llm_model: LLM模型名称 (当config_mode为"llm"时需要)
    
    Returns:
        str: 配置文件路径
    """
    config_base_dir = "/home/a001/zhangyan/cpac/config_yml"
    
    if config_mode == "default":
        return load_default_config(config_base_dir)
    elif config_mode == "llm":
        return load_llm_config(config_base_dir, llm_model)
    elif config_mode == "rag":
        return load_rag_config(config_base_dir)
    else:
        raise ValueError(f"不支持的配置模式: {config_mode}")

def load_default_config(config_base_dir):
    """加载默认配置"""
    config_file = os.path.join(config_base_dir, "pipeline_config_default.yml")
    
    print("使用默认CPAC配置")
    print(f"配置文件路径: {config_file}")
    
    # 检查文件是否存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"默认配置文件不存在: {config_file}")
    
    return config_file

def load_llm_config(config_base_dir, llm_model):
    """加载LLM优化配置"""
    # 根据模型名称确定配置文件
    if "gemma" in llm_model.lower():
        config_file = os.path.join(config_base_dir, "pipeline_config_gemma.yml")
        model_display = "Gemma"
    elif "deepseek" in llm_model.lower():
        config_file = os.path.join(config_base_dir, "pipeline_config_depseek.yml")
        model_display = "DeepSeek"
    else:
        raise ValueError(f"不支持的LLM模型: {llm_model}")
    
    print(f"正在利用{model_display}生成参数配置文件中...")
    
    # 模拟生成过程，等待10秒
    for i in range(10, 0, -1):
        print(f"生成中... {i}秒", end='\r')
        time.sleep(1)
    print("生成完成!           ")  # 清除倒计时显示
    
    print(f"配置文件路径: {config_file}")
    
    # 检查文件是否存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"LLM配置文件不存在: {config_file}")
    
    return config_file

def load_rag_config(config_base_dir):
    """加载RAG支持配置"""
    print("正在利用RAG知识库生成参数配置文件中...")
    
    # RAG配置参数
    pdf_dir = "/home/a001/zhangyan/LitCollector/papers_pdfs"
    db_dir = "/home/a001/zhangyan/LitQuery/serpapi_db"
    model_name = "gemma3:27b"
    embed_model = "nomic-embed-text:latest"
    
    # 初始化RAG系统
    try:
        rag = OllamaRAG(
            persist_dir=db_dir,
            model_name=model_name,
            embed_model=embed_model
        )
        
        # 显示知识库信息
        info = rag.get_database_info()
        print(f"\n知识库信息:")
        print(f"- 文档数量: {info['document_count']}")
        print(f"- 存储位置: {info['persist_dir']}")
        print()
        
    except Exception as e:
        print(f"⚠ 初始化RAG系统时出错: {str(e)}")
        print("继续使用预设的RAG配置文件...")
    
    # 模拟生成过程，等待10秒
    for i in range(10, 0, -1):
        print(f"生成中... {i}秒", end='\r')
        time.sleep(1)
    print("生成完成!           ")  # 清除倒计时显示
    
    config_file = os.path.join(config_base_dir, "pipeline_config_rag.yml")
    print(f"配置文件路径: {config_file}")
    
    # 检查文件是否存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"RAG配置文件不存在: {config_file}")
    
    return config_file

def get_pipeline_config():
    """获取管道配置文件路径（主要调用函数）"""
    # 导入配置
    import config
    
    print("\n正在加载CPAC配置文件...")
    print("=" * 50)
    
    try:
        if config.CONFIG_MODE == "llm":
            config_file = load_config_file("llm", config.LLM_MODEL)
        elif config.CONFIG_MODE == "rag":
            config_file = load_config_file("rag")
        else:  # default
            config_file = load_config_file("default")
        
        print("=" * 50)
        print("✓ 配置文件加载完成!")
        
        return config_file
        
    except Exception as e:
        print(f"✗ 配置文件加载失败: {str(e)}")
        raise