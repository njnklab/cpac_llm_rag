import pandas as pd
import requests
import os
import json
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager as fm

def download_pdf(url, folder_path, filename, timeout=300):
    """
    Download PDF file from URL with a timeout
    
    Args:
        url (str): URL of the PDF file
        folder_path (str): Directory to save the file
        filename (str): Name of the file to be saved
        timeout (int): Maximum time to wait for download in seconds (default: 5 minutes)
    """
    start_time = time.time()
    try:
        # 设置流式下载和超时
        response = requests.get(url, stream=True, timeout=timeout)
        
        # 检查响应状态码
        if response.status_code == 200:
            filepath = os.path.join(folder_path, filename)
            
            # 确保目标文件夹存在
            os.makedirs(folder_path, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    # 检查是否超时
                    if time.time() - start_time > timeout:
                        raise requests.exceptions.Timeout("Download took too long")
                    f.write(chunk)
            
            # 计算下载耗时
            download_time = time.time() - start_time
            print(f"Successfully downloaded {filename} in {download_time:.2f} seconds")
            return True
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
    
    except requests.exceptions.Timeout:
        print(f"Error downloading {filename}: Download timed out after {timeout} seconds")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error downloading {filename}: {str(e)}")
    
    return False

def process_scholar_results(results_folder,txt_path,base_root_path):
    
    # 预定义路径
    base_root_path = base_root_path
    papers_info_path = os.path.join(base_root_path, "papers_info.xlsx")
    papers_pdfs_path = os.path.join(base_root_path, "papers_pdfs")
    
    # 确保PDF下载目录存在
    os.makedirs(papers_pdfs_path, exist_ok=True)

    print("-------read txt----------")

    # 加载单次搜索结果
    with open(txt_path, 'r', encoding='utf-8') as f:
        organic_results = json.load(f)

    # 准备数据框
    papers_data = []
    
    for paper in organic_results:
        # 提取基本信息
        paper_info = {
            'Position': paper.get('position'),
            'Title': paper.get('title'),
            'Link': paper.get('link'),
            'Snippet': paper.get('snippet'),
            'Citations': paper.get('inline_links', {}).get('cited_by', {}).get('total'),
            'Year': '',  # 将从 publication_info 提取
            'Journal': ''  # 新增期刊字段
        }
        
        # 提取作者
        pub_info = paper.get('publication_info', {})
        if pub_info:
            paper_info['Authors'] = '; '.join([author['name'] for author in pub_info.get('authors', [])] 
                                            if 'authors' in pub_info else [])
            paper_info['Publication Summary'] = pub_info.get('summary', '')
        
        # 从 summary 中提取年份
        summary = pub_info.get('summary', '')
        year_match = re.search(r'(\d{4})', summary)
        if year_match:
            paper_info['Year'] = year_match.group(1)

        # 提取期刊名称
        # 匹配 "- Journal Name, " 或 "- Journal Name - " 的模式
        journal_match = re.search(r'-\s*(.*?)\s*(?:,|\s-\s)', summary)
        if journal_match:
            journal_name = journal_match.group(1).strip()
            # 移除出版商域名（如果存在）
            journal_name = re.sub(r'\s*-\s*\w+\.[\w\.]+$', '', journal_name)
            paper_info['Journal'] = journal_name

        # 提取 PDF 链接
        pdf_link = None
        if 'resources' in paper:
            for resource in paper['resources']:
                if resource.get('file_format') == 'PDF':
                    pdf_link = resource.get('link')
                    break
        
        paper_info['PDF_Link'] = pdf_link
        
        # 尝试下载 PDF（如果链接存在）
        if pdf_link:
            print("-------download PDF when link exists----------")
            print(pdf_link)
            filename = f"{paper['title']}.pdf"
            success = download_pdf(pdf_link, papers_pdfs_path, filename)
            paper_info['PDF_Downloaded'] = 'Yes' if success else 'No'
            paper_info['PDF_Filename'] = filename if success else ''
        else:
            paper_info['PDF_Downloaded'] = 'No Link'
            paper_info['PDF_Filename'] = ''
        
        papers_data.append(paper_info)
    
    # 创建数据框
    df = pd.DataFrame(papers_data)
    
    # 保存当前结果到时间戳文件夹
    excel_path = os.path.join(results_folder, 'papers_summary.xlsx')
    df.to_excel(excel_path, index=False)

    # 加载或创建主 papers_info.xlsx
    if os.path.exists(papers_info_path):
        master_df = pd.read_excel(papers_info_path)
        # 追加新数据，根据标题去除重复
        master_df = pd.concat([master_df, df], ignore_index=True).drop_duplicates(subset=['Title'])
    else:
        master_df = df

    # 保存更新后的主论文信息
    master_df.to_excel(papers_info_path, index=False)

    print(f"Results processed and saved in folder: {results_folder}")
    print(f"Master papers info updated at: {papers_info_path}")
    print("\nFirst few entries of the summary:")
    print(df.head())


    # ------------------填充IF和分区字段---------------
    match_and_update_paper_info()

    # ---------------分析文献数据并生成可视化图表---------
    papers_info = pd.read_excel('/home/a001/zhangyan/LitCollector/papers_info.xlsx')
    results = analyze_literature_data(papers_info)

    return results_folder

def analyze_literature_data(df,output_path="/home/a001/zhangyan/LitCollector"):
    """
    分析文献数据并生成可视化图表
    
    Parameters:
    df (pandas.DataFrame): 包含文献信息的数据框
    """

    # # 设置中文字体
    # font_path = r"E:\多Agent的文献知识库构建\SimHei.ttf"
    # zh_font = fm.FontProperties(fname=font_path)
    # plt.rcParams['font.family'] = zh_font.get_name()
    # plt.rcParams['axes.unicode_minus'] = False

    # 清理数据
    df = df.copy()
    
    # 基础统计信息
    total_papers = len(df)
    total_citations = df['Citations'].sum()
    avg_citations = df['Citations'].mean()
    
    # 按分区统计论文数量
    quartile_stats = df['Quartile'].value_counts().sort_index()
    
    # 按年份统计论文数量
    year_stats = df['Year'].value_counts().sort_index()
    
    # 计算影响因子统计
    if_stats = df['IF'].describe()
    
    # 打印统计信息
    print("\n=== 文献分析报告 ===")
    print(f"总文献数量: {total_papers}")
    print(f"总引用次数: {total_citations:,.0f}")
    print(f"平均引用次数: {avg_citations:.2f}")
    print("\n影响因子统计:")
    print(if_stats)
    
    # 创建图形
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 分区分布饼图
    plt.subplot(221)
    quartile_stats.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Periodical partition distribution')
    
    # 2. 年份分布柱状图
    plt.subplot(222)
    year_stats.plot(kind='bar')
    plt.title('Document year distribution')
    plt.xlabel('year')
    plt.ylabel('Number of documents')
    plt.xticks(rotation=45)
    
    # 3. 引用次数箱线图（按分区）
    plt.subplot(223)
    sns.boxplot(data=df, x='Quartile', y='Citations')
    plt.title('Distribution of reference citations in each region')
    plt.xlabel('region')
    plt.ylabel('citations')
    
    # 4. 影响因子与引用次数散点图
    plt.subplot(224)
    plt.scatter(df['IF'], df['Citations'], alpha=0.5)
    plt.title('Relationship between IF and number of citations')
    plt.xlabel('IF')
    plt.ylabel('citations')
    plt.tight_layout()

    # 保存图形
    fig.savefig(f"{output_path}/literature_analysis.png", dpi=600, bbox_inches='tight')
    print(f"图表已保存到: {output_path}/literature_analysis.png")

    # plt.show()
    
    # 额外分析：高被引论文
    top_cited = df.nlargest(5, 'Citations')[['Title', 'Citations', 'Year', 'Journal', 'IF', 'Quartile']]
    print("\n引用量最高的5篇论文:")
    print(top_cited)
    
    # 分区质量分析
    quartile_quality = df.groupby('Quartile').agg({
        'Citations': ['count', 'mean', 'sum'],
        'IF': ['count', 'mean']
    }).round(2)
    
    print("\n各分区统计信息:")
    print(quartile_quality)

    return {
        'total_papers': total_papers,
        'total_citations': total_citations,
        'avg_citations': avg_citations,
        'quartile_stats': quartile_stats,
        'year_stats': year_stats,
        'if_stats': if_stats
    }


# ------------------填充IF和分区字段---------------
def match_and_update_paper_info():
    """
    该函数主要用于读取当前目录下的两个Excel文件（papers_info.xlsx和2023JCR.xlsx），
    对papers_info.xlsx进行一系列处理操作，包括检查并添加特定列、清洗Journal列数据，
    然后将其Journal列与2023JCR.xlsx的“名字”列进行匹配，最后将匹配到的记录的相关数据填充到papers_info.xlsx中。
    """
    
    papers_info_df = pd.read_excel("/home/a001/zhangyan/LitCollector/papers_info.xlsx")
    jcr_df = pd.read_excel("/home/a001/zhangyan/LitCollector/2023JCR.xlsx")

    # 检查papers_info.xlsx中是否有“IF”和“Quartile”列，没有则添加
    if "IF" not in papers_info_df.columns:
        papers_info_df["IF"] = None
    if "Quartile" not in papers_info_df.columns:
        papers_info_df["Quartile"] = None

    # 数据清洗Journal列，只保留英文字符
    papers_info_df["Journal"] = papers_info_df["Journal"].astype(str).str.replace(r'[^\w\s]', '', regex=True).str.replace(r'\d', '', regex=True).str.strip()

    # 将Journal列与2023JCR.xlsx的“名字”列做匹配，并填充数据
    for index, row in papers_info_df.iterrows():
        journal_name = row["Journal"]
        match = jcr_df[jcr_df["名字"].str.lower() == journal_name.lower()]
        if not match.empty:
            papers_info_df.at[index, "IF"] = match.iloc[0]["2023最新IF"]
            papers_info_df.at[index, "Quartile"] = match.iloc[0]["分区"]

    # 将更新后的数据写回papers_info.xlsx
    papers_info_df.to_excel("/home/a001/zhangyan/LitCollector/papers_info.xlsx", index=False)


# txt_path = "/home/user/zhangyan/RAG/knowledge_base_based_on_multi-agent/google-search-results-python/test/organic_results.txt"

# # Process results and get DataFrame
# print("-------get DataFrame and download pdf----------")
# df, results_folder = process_scholar_results(txt_path)

# print(f"Results processed and saved in folder: {results_folder}")
# print("\nFirst few entries of the summary:")
# print(df.head())