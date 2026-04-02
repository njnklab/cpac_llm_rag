import os
import json
import time
import pandas as pd
from serpapi import GoogleSearch
from datetime import datetime
import random

def get_start_offset():
    """
    读取已有的papers_info.xlsx文件，返回下一个起始偏移量
    """
    excel_path = "/home/a001/zhangyan/LitCollector/papers_info.xlsx"
    
    if not os.path.exists(excel_path):
        return 0
    
    try:
        df = pd.read_excel(excel_path)
        # 获取当前记录数
        current_records = len(df)
        # 返回下一个起始位置
        return current_records
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return 0


def get_google_scholar_results(q, as_ylo, as_yhi, num, page=0, api_key=None):
    """
    获取Google Scholar搜索结果
    
    参数:
    q: 查询关键词
    as_ylo: 开始年份
    as_yhi: 结束年份
    num: 每页结果数量
    page: 页码 (0为第一页)
    api_key: SerpAPI密钥，如果为None则从密钥池中随机选择
    """
    # 定义根文件夹路径
    base_root_path = "/home/a001/zhangyan/LitCollector/temp_result"
    
    # 页码转换为start参数
    start = page * int(num)

    # 创建带时间戳的文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = os.path.join(base_root_path, f"serpapi_{timestamp}")
    os.makedirs(results_folder, exist_ok=True)

    # API密钥池 - 如果有多个API密钥，随机选择一个
    api_keys = [
        "8ac6dcab616fd37995cfb90e2c7ee43dfac3b01741d6a20fe5fd9a06c1feb7e0",
        "d7b51d2f001caa5ebd1cffda283e01779b6f1dcb6e6ffa12e4aaba4fee09536d",
        # 添加更多API密钥
    ]
    
    if api_key is None:
        api_key = random.choice(api_keys)
    
    # SerpAPI 搜索参数
    params = {
        "engine": "google_scholar",
        "q": q,
        "as_ylo": as_ylo,
        "as_yhi": as_yhi,
        "hl": "en",
        "lr": "lang_en",
        "as_vis": "0",
        "as_rr": "0",
        "start": start,
        "num": num,
        "api_key": api_key
    }
    
    # 执行搜索
    print(f"---------------- 执行搜索 (页码: {page+1}, start: {start}) ----------------")
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # 检查结果中是否有organic_results
    if "organic_results" not in results or len(results["organic_results"]) == 0:
        print("警告: 没有找到有机结果!")
        # 保存整个响应以便调试
        with open(os.path.join(results_folder, 'full_response.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results_folder, None
    
    organic_results = results["organic_results"]
    print(f"获取到 {len(organic_results)} 条结果")

    # 保存 organic results 到带时间戳的文件夹
    txt_path = os.path.join(results_folder, 'organic_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        json.dump(organic_results, f, indent=2, ensure_ascii=False)

    return results_folder, txt_path


# def get_google_scholar_results(q, as_ylo, as_yhi, num):
#     # 定义根文件夹路径
#     base_root_path = "/home/a001/zhangyan/LitCollector/temp_result"
    
#     # 获取动态start值
#     start = get_start_offset()

#     # 创建带时间戳的文件夹
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     results_folder = os.path.join(base_root_path, f"serpapi_{timestamp}")
#     os.makedirs(results_folder, exist_ok=True)

#     # SerpAPI 搜索参数  # fty的key
#     # ZEZ:d7b51d2f001caa5ebd1cffda283e01779b6f1dcb6e6ffa12e4aaba4fee09536d
#     api_key = "8ac6dcab616fd37995cfb90e2c7ee43dfac3b01741d6a20fe5fd9a06c1feb7e0"
#     params = {
#         "engine": "google_scholar",
#         "q": q,
#         "as_ylo": as_ylo,
#         "as_yhi": as_yhi,
#         "hl": "en",
#         "lr": "lang_en|lang_zh-CN",
#         "as_vis": "0",
#         "as_rr": "0",
#         "start": start,
#         "num": num,
#         "api_key": api_key
#     }
    
#     # 执行搜索
#     print("----------------执行搜索----------------")
#     search = GoogleSearch(params)
#     print("----------------搜索结束----------------")
#     results = search.get_dict()
#     # print("results:",results)
#     organic_results = results["organic_results"]

#     # 保存 organic results 到带时间戳的文件夹
#     txt_path = os.path.join(results_folder, 'organic_results.txt')
#     with open(txt_path, 'w', encoding='utf-8') as f:
#         json.dump(organic_results, f, indent=2, ensure_ascii=False)

#     return results_folder,txt_path