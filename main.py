#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPAC项目主函数
"""

import sys
import os
from pathlib import Path
import Tools
# 添加src路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config_checker import validate_all_configs

def run_cpac_preprocessing(config, pipeline_config_file):
    """执行CPAC预处理"""
    print("\n开始CPAC预处理...")
    print("=" * 50)
    
    # 构建BIDS数据路径
    bids_root = os.path.join(config.DATA_PATH, config.DATASET_NAME)
    
    # 获取要处理的被试列表
    if config.SUBJECT_LIST == "all":
        # 从participants.tsv获取所有被试
        import pandas as pd
        participants_file = os.path.join(bids_root, "participants.tsv")
        participants_df = pd.read_csv(participants_file, sep='\t')
        subject_list = participants_df['participant_id'].tolist()
    else:
        subject_list = config.SUBJECT_LIST
    
    print(f"准备处理 {len(subject_list)} 个被试")
    
    # 处理每个被试
    for participant_id in subject_list:
        try:
            bids_path = os.path.join(bids_root, participant_id)
            print(f"\n处理被试: {participant_id}")
            print(f"BIDS路径: {bids_path}")
            
            if not os.path.exists(bids_path):
                print(f"⚠ 被试路径不存在: {bids_path}")
                continue
            
            # 检查BIDS数据目录是否有效
            print("检查BIDS数据目录...")
            is_valid, error_msg = Tools.check_bids_data(bids_path)
            if not is_valid:
                print(f"✗ 被试 {participant_id} 数据无效: {error_msg}")
                Tools.update_processed_ids(config.processed_file, participant_id, status="invalid_data")
                continue
            
            # 创建并运行CPAC脚本
            script_path = Tools.create_cpac_script(bids_root, config.DATASET_NAME, participant_id, pipeline_config_file)
            if not script_path:
                print(f"✗ 创建CPAC脚本失败: {participant_id}")
                continue
            
            print(f"CPAC脚本: {script_path}")
            
            # # 运行CPAC脚本
            # output_dir = Tools.run_cpac_script(script_path, config.DATASET_NAME, participant_id)
            
            # # 生成被试报告
            # try:
            #     report_path = Tools.generate_subject_report(config.DATASET_NAME, participant_id)
            #     print(f"✓ 生成被试报告: {report_path}")
            # except Exception as e:
            #     print(f"⚠ 生成报告失败: {str(e)}")
            
            # 更新已处理列表
            # Tools.update_processed_ids(config.processed_file, participant_id)
            print(f"✓ 被试 {participant_id} 处理完成")
            # print(f"输出目录: {output_dir}")
            
        except Exception as e:
            print(f"✗ 处理被试 {participant_id} 时出错: {str(e)}")
            continue
    
    print("=" * 50)
    print("✓ CPAC预处理完成!")


# --------------------------------------------------



def main():
    """主函数"""
    print("CPAC预处理项目启动")
    print("=" * 60)
    
    import os
    print("当前工作目录:", os.getcwd())


    # 验证配置
    if not validate_all_configs():
        print("配置验证失败，程序退出")
        sys.exit(1)
    
    print("\n配置验证完成，准备开始处理...")
    
    # 导入配置（验证通过后）
    import config
    import dataset
    # 显示配置摘要
    print("\n当前配置摘要:")
    print("-" * 30)
    print(f"数据集名称: {config.DATASET_NAME}")
    print(f"被试数量: {len(config.SUBJECT_LIST) if isinstance(config.SUBJECT_LIST, list) else '全部'}")
    dataset.print_dataset_files(config.DATA_PATH, config.DATASET_NAME)
    print(f"参数配置模式: {config.CONFIG_MODE}")
    if config.CONFIG_MODE == "llm":
        print(f"LLM模型: {config.LLM_MODEL}")
    elif config.CONFIG_MODE == "rag":
        print(f"ChromaDB路径: {config.CHROMADB_PATH}")
        
    print(f"输出路径: {config.OUTPUT_PATH}")
    # 显示分析需求
    enabled_analyses = [k for k, v in config.ANALYSIS_REQUIREMENTS.items() if v]
    print(f"启用的分析: {', '.join(enabled_analyses)}")
    print("-" * 30)
    
    # 获取用户分析描述
    user_analysis_description = dataset.get_user_analysis_description()
    
    # 加载CPAC配置文件
    from cpac_config_loader import get_pipeline_config
    pipeline_config_file = get_pipeline_config()
    
    # 执行CPAC预处理
    print("---------------执行CPAC预处理-----------")
    run_cpac_preprocessing(config, pipeline_config_file)
    
    print("\n✓ 所有被试处理完成,生成最终报告")
    

if __name__ == "__main__":
    main()