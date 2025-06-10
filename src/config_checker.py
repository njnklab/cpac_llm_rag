#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置检查模块
用于验证config.py中的各项设置
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

def check_config_exists():
    """检查config.py文件是否存在"""
    if not os.path.exists("config.py"):
        raise FileNotFoundError("未找到config.py文件，请先创建配置文件")
    print("✓ 找到config.py配置文件")

def check_dataset_exists(data_path, dataset_name):
    """检查数据集文件夹是否存在"""
    dataset_path = os.path.join(data_path, dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
    print(f"✓ 数据集路径存在: {dataset_path}")
    return dataset_path

def check_subject_list(data_path, dataset_name, subject_list):
    """检查被试列表设置"""
    # 检查subject_list是列表还是"all"字符串
    if isinstance(subject_list, str) and subject_list == "all":
        print("✓ 被试列表设置为处理所有被试")
        return True
    
    if not isinstance(subject_list, list):
        raise ValueError("SUBJECT_LIST必须是列表或字符串'all'")
    
    if len(subject_list) == 0:
        raise ValueError("SUBJECT_LIST列表不能为空")
    
    print(f"✓ 被试列表为具体列表，包含{len(subject_list)}个被试")
    
    # 检查participants.tsv文件
    dataset_path = os.path.join(data_path, dataset_name)
    participants_file = os.path.join(dataset_path, "participants.tsv")
    
    if not os.path.exists(participants_file):
        raise FileNotFoundError(f"未找到participants.tsv文件: {participants_file}")
    
    # 读取participants.tsv并检查被试ID
    try:
        participants_df = pd.read_csv(participants_file, sep='\t')
        if 'participant_id' not in participants_df.columns:
            raise ValueError("participants.tsv文件中未找到participant_id列")
        
        available_subjects = participants_df['participant_id'].tolist()
        missing_subjects = [sub for sub in subject_list if sub not in available_subjects]
        
        if missing_subjects:
            raise ValueError(f"以下被试在participants.tsv中未找到: {missing_subjects}")
        
        print(f"✓ 所有被试都在participants.tsv中找到")
        
    except Exception as e:
        raise ValueError(f"读取participants.tsv文件时出错: {str(e)}")

def check_ollama_model(model_name):
    """检查Ollama模型是否存在"""
    try:
        # 运行ollama list命令
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, check=True)
        
        # 检查模型是否在输出中
        if model_name in result.stdout:
            print(f"✓ Ollama模型 {model_name} 已安装")
            return True
        else:
            raise ValueError(f"Ollama模型 {model_name} 未找到，请先安装该模型")
            
    except subprocess.CalledProcessError:
        raise RuntimeError("无法运行ollama命令，请确保Ollama已正确安装并运行")
    except FileNotFoundError:
        raise RuntimeError("未找到ollama命令，请确保Ollama已安装并添加到PATH")

def check_chromadb_path(chromadb_path):
    """检查ChromaDB路径"""
    if os.path.exists(chromadb_path):
        print(f"✓ ChromaDB路径存在: {chromadb_path}")
    else:
        print(f"⚠ ChromaDB路径不存在，将创建新的数据库: {chromadb_path}")
        # 创建目录
        os.makedirs(chromadb_path, exist_ok=True)
        print(f"✓ 已创建ChromaDB目录: {chromadb_path}")

def check_output_path(output_path):
    """检查输出路径"""
    if not os.path.exists(output_path):
        print(f"⚠ 输出路径不存在，将创建: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        print(f"✓ 已创建输出目录: {output_path}")
    else:
        print(f"✓ 输出路径存在: {output_path}")

def validate_all_configs():
    """验证所有配置项"""
    try:
        import importlib.util

        def load_config():
            config_path = "/home/a001/zhangyan/cpac/config.py"
            spec = importlib.util.spec_from_file_location("config", config_path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            return config

        config = load_config()
            
            
        # import config
        
        print("开始验证配置...")
        print("=" * 50)
        
        # 1. 检查配置文件存在
        check_config_exists()
        
        # 2. 检查数据集路径
        dataset_path = check_dataset_exists(config.DATA_PATH, config.DATASET_NAME)
        
        # 3. 检查被试列表
        check_subject_list(config.DATA_PATH, config.DATASET_NAME, config.SUBJECT_LIST)
        
        # 4. 根据配置模式进行相应检查
        if config.CONFIG_MODE == "llm":
            check_ollama_model(config.LLM_MODEL)
        elif config.CONFIG_MODE == "rag":
            check_chromadb_path(config.CHROMADB_PATH)
        elif config.CONFIG_MODE == "default":
            print("✓ 使用默认配置模式")
        else:
            raise ValueError(f"不支持的配置模式: {config.CONFIG_MODE}")
        
        # 5. 检查输出路径
        check_output_path(config.OUTPUT_PATH)
        
        print("=" * 50)
        print("✓ 所有配置验证通过！")
        return True
        
    except Exception as e:
        print("=" * 50)
        print(f"validate_all_configs() 配置验证失败: {str(e)}")
        return False