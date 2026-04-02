
from google_search import get_google_scholar_results
from extract_info import process_scholar_results
import time
import os
from datetime import datetime
import pandas as pd

def check_collected_papers():
    """
    检查已收集论文的数量和唯一标题
    返回: (已收集论文数量, 已收集论文标题集合)
    """
    excel_path = "/home/a001/zhangyan/LitCollector/papers_info.xlsx"
    
    if not os.path.exists(excel_path):
        return 0, set()
    
    try:
        df = pd.read_excel(excel_path)
        # 获取当前记录数和标题集合
        current_records = len(df)
        title_set = set(df['title'].str.lower().tolist() if 'title' in df.columns else [])
        return current_records, title_set
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return 0, set()
    

def batch_collect_papers_with_rotating_queries(
    query_list,
    start_year,
    end_year,
    articles_per_batch=20,
    total_articles=300,
    time_interval=300,  # x分钟间隔
    max_pages_per_query=10,  # 每个查询最多获取多少页
    base_root_path="/home/a001/zhangyan/LitCollector"
):
    """
    使用轮换查询和分页批量收集论文
    
    参数:
    query_list: 搜索关键词列表
    start_year: 开始年份
    end_year: 结束年份
    articles_per_batch: 每批次收集的文章数量
    total_articles: 总共要收集的文章数量
    time_interval: 批次之间的时间间隔(秒)
    max_pages_per_query: 每个查询最多获取多少页结果
    base_root_path: 基础保存路径
    """
    # 创建一个日志文件记录每次运行的情况
    log_file = os.path.join(base_root_path, f"collection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(log_file, "w") as f:
        f.write(f"开始收集论文: {datetime.now()}\n")
        f.write(f"查询列表: {query_list}\n")
        f.write(f"年份范围: {start_year}-{end_year}\n")
        f.write(f"计划收集总数: {total_articles}\n")
        f.write(f"每批次数量: {articles_per_batch}\n")
        f.write(f"批次间隔: {time_interval}秒\n\n")
    
    all_collected_folders = []
    current_count, collected_titles = check_collected_papers()
    
    # 记录每个查询当前的页码
    query_page_index = {query: 0 for query in query_list}
    
    # 循环直到收集到足够的论文
    query_index = 0
    
    while current_count < total_articles:
        # 轮换选择查询
        current_query = query_list[query_index]
        current_page = query_page_index[current_query]
        
        # 如果当前查询已经达到最大页数，切换到下一个查询
        if current_page >= max_pages_per_query:
            query_index = (query_index + 1) % len(query_list)
            continue
        
        batch_start_time = datetime.now()
        
        print(f"\n{'='*50}")
        print(f"开始收集 - 查询: {current_query} (页码: {current_page+1})")
        print(f"当前已收集: {current_count}/{total_articles}")
        print(f"时间: {batch_start_time}")
        print(f"{'='*50}\n")
        
        # 记录日志
        with open(log_file, "a") as f:
            f.write(f"查询: {current_query} (页码: {current_page+1}) 开始: {batch_start_time}\n")
        
        try:
            # 调用serpapi收集相关论文信息
            results_folder, txt_path = get_google_scholar_results(
                current_query, 
                start_year, 
                end_year,
                str(articles_per_batch),
                page=current_page
            )
            
            if txt_path:
                print("txt_path:", txt_path)
                
                # 根据收集的信息整理成表并下载论文
                this_time_folder = process_scholar_results(results_folder, txt_path, base_root_path)
                print("results_folder:", this_time_folder)
                
                all_collected_folders.append(this_time_folder)
                
                # 更新当前收集的数量
                new_count, new_titles = check_collected_papers()
                papers_added = new_count - current_count
                current_count = new_count
                collected_titles = new_titles
                
                # 记录日志
                with open(log_file, "a") as f:
                    f.write(f"查询 {current_query} (页码: {current_page+1}) 完成: {datetime.now()}\n")
                    f.write(f"本次新增论文: {papers_added}\n")
                    f.write(f"当前总数: {current_count}/{total_articles}\n")
                    f.write(f"保存路径: {this_time_folder}\n\n")
            else:
                print(f"查询 {current_query} 页码 {current_page+1} 没有返回结果，将尝试下一个查询")
                # 记录日志
                with open(log_file, "a") as f:
                    f.write(f"查询 {current_query} 页码 {current_page+1} 没有返回结果\n\n")
                
            # 更新页码并切换到下一个查询
            query_page_index[current_query] += 1
            query_index = (query_index + 1) % len(query_list)
                
        except Exception as e:
            print(f"查询 {current_query} 页码 {current_page+1} 出错: {str(e)}")
            # 记录错误
            with open(log_file, "a") as f:
                f.write(f"查询 {current_query} 页码 {current_page+1} 出错: {str(e)}\n\n")
            
            # 如果出错，切换到下一个查询
            query_index = (query_index + 1) % len(query_list)
        
        # 等待一段时间再继续
        batch_end_time = datetime.now()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        
        # 计算需要额外等待的时间
        wait_time = max(0, time_interval - batch_duration)
        
        print(f"\n本批次用时: {batch_duration:.1f}秒")
        print(f"等待 {wait_time:.1f}秒后继续下一批次...\n")
        
        # 记录日志
        with open(log_file, "a") as f:
            f.write(f"等待 {wait_time:.1f}秒后继续下一批次\n")
        
        time.sleep(wait_time)
        
        # 检查是否所有查询都已经达到最大页数
        if all(page >= max_pages_per_query for page in query_page_index.values()):
            print("所有查询已经达到最大页数，停止收集")
            break
    
    # 记录总结信息
    with open(log_file, "a") as f:
        f.write(f"\n论文收集完成: {datetime.now()}\n")
        f.write(f"共收集 {current_count} 篇论文\n")
        f.write(f"收集文件夹列表:\n")
        for folder in all_collected_folders:
            f.write(f"- {folder}\n")
    
    print(f"\n{'='*50}")
    print(f"论文收集完成!")
    print(f"共收集 {current_count} 篇论文")
    print(f"执行日志: {log_file}")
    print(f"{'='*50}\n")
    
    return all_collected_folders

if __name__ == "__main__":
    # 定义多个查询关键词，针对CPAC工具、ADHD-200和ds002748数据集的预处理文献
    query_list = [
        # ========== CPAC工具核心文献 ==========
        '"Configurable Pipeline for the Analysis of Connectomes" OR "C-PAC" preprocessing',
        'CPAC fMRI preprocessing pipeline parameters configuration',
        'CPAC neuroimaging workflow resting-state fMRI',
        
        # ========== ADHD-200数据集相关 ==========
        'ADHD-200 fMRI preprocessing pipeline methods',
        'ADHD-200 resting-state fMRI quality control preprocessing',
        'ADHD-200 neuroimaging data analysis preprocessing steps',
        
        # ========== OpenNeuro ds002748/抑郁数据集 ==========
        'OpenNeuro ds002748 depression fMRI preprocessing',
        '"ds002748" resting-state fMRI preprocessing',
        'depression resting-state fMRI preprocessing pipeline',
        
        # ========== 多站点数据预处理（重要） ==========
        'multi-site fMRI preprocessing harmonization ComBat',
        'multi-site neuroimaging data harmonization methods',
        'cross-site fMRI preprocessing standardization',
        
        # ========== 静息态fMRI预处理最佳实践 ==========
        'resting-state fMRI preprocessing best practices',
        'rs-fMRI motion correction slice timing normalization',
        'resting-state fMRI denoising nuisance regression',
        
        # ========== CPAC相关技术细节 ==========
        'fMRI preprocessing motion correction realignment',
        'fMRI spatial normalization registration segmentation',
        'temporal filtering fMRI preprocessing bandpass',
        'nuisance signal regression fMRI preprocessing',
        
        # ========== 质量控制 ==========
        'fMRI quality control preprocessing visual inspection',
        'neuroimaging quality assurance preprocessing',
        
        # ========== 连接组分析（CPAC的核心） ==========
        'functional connectivity preprocessing resting-state',
        'connectome analysis fMRI preprocessing pipeline',
        
        # ========== 通用MRI预处理（保留原有关键词） ==========
        "(MRI OR fMRI) AND preprocessing AND (pipeline OR workflow OR protocol) AND (neuroimaging OR brain)",
        "(MRI OR fMRI) AND preprocessing AND software AND (tools OR methods)",
        "(MRI OR fMRI) AND (preprocessing OR processing) AND (SPM OR FSL OR AFNI OR FreeSurfer)",
    ]
    
    start_year = "2010"  # CPAC是2013年左右发布的，ADHD-200也是2012-2013年
    end_year = "2025"
    total_articles = 300  # 针对特定主题的精准文献
    
    # 使用轮换查询收集论文
    collected_folders = batch_collect_papers_with_rotating_queries(
        query_list=query_list,
        start_year=start_year,
        end_year=end_year,
        articles_per_batch=20,
        total_articles=total_articles,
        time_interval=300,  # 5分钟
        max_pages_per_query=10  # 每个查询最多获取10页结果
    )


# # 使用 Ollama API 优化关键词 ; Databases/Journals: [list of relevant databases/journals]'
# from ollama_api import call_ollama_api_generate
# optimized_keywords = call_ollama_api_generate(
#     f"Optimize these keywords for searching academic papers: {key_words}. "
#     "Respond only with 'Keywords: [list of optimized keywords]."
#     "Do not include additional explanations or self-descriptive text."
# )
# print("optimized_keywords:",optimized_keywords)