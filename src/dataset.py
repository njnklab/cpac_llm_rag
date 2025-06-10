import os
import config

def print_dataset_files(DATA_PATH,DATASET_NAME):
    """
    自动使用 config.DATA_PATH 和 config.DATASET_NAME 拼接数据集目录，
    在该目录下查找并打印以下文件内容：
      - dataset_description.txt
      - README
      - README.md
      - README.txt
    """
    # 拼接出目标目录
    folder_path = os.path.join(DATA_PATH, DATASET_NAME)
    
    # 要查找的文件列表
    files_to_find = [
        'dataset_description.txt',
        'README',
    ]

    for filename in files_to_find:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            print(f"\n===== {filename} =====")
            try:
                # 假设文本文件使用 UTF-8 编码
                with open(file_path, 'r', encoding='utf-8') as f:
                    print(f.read())
            except Exception as e:
                print(f"读取文件 {filename} 时出错：{e}")
        else:
            print(f"未找到：{filename} （路径：{file_path}）")

def get_user_analysis_description():
    """获取用户分析描述并保存到JSON文件"""
    import json
    from datetime import datetime
    
    print("\n请描述您想要进行的分析:")
    print("(可以包括研究问题、分析目标、特殊要求等，输入完成后按回车)")
    print("-" * 50)
    
    # 获取用户输入
    user_input = input("分析描述: ")
    
    # 如果用户没有输入，给个默认值
    if not user_input.strip():
        user_input = "标准功能磁共振预处理分析"
    
    # 创建分析描述数据
    analysis_data = {
        "timestamp": datetime.now().isoformat(),
        "user_description": user_input.strip(),
        "enabled_analyses": [],  # 将从config中获取
        "dataset_info": {
            "name": "",
            "subject_count": 0
        },
        "config_mode": ""
    }
    
    # 从config中获取信息
    import config
    analysis_data["enabled_analyses"] = [k for k, v in config.ANALYSIS_REQUIREMENTS.items() if v]
    analysis_data["dataset_info"]["name"] = config.DATASET_NAME
    analysis_data["dataset_info"]["subject_count"] = len(config.SUBJECT_LIST) if isinstance(config.SUBJECT_LIST, list) else "all"
    analysis_data["config_mode"] = config.CONFIG_MODE
    
    # 确保日志目录存在
    log_dir = "/home/a001/zhangyan/cpac/log"
    os.makedirs(log_dir, exist_ok=True)
    
    # 保存到JSON文件
    json_file = os.path.join(log_dir, "user_analysis.json")
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 分析描述已保存到: {json_file}")
    except Exception as e:
        print(f"⚠ 保存分析描述时出错: {str(e)}")
    
    print(f"\n您的分析描述: {user_input}")
    
    return user_input