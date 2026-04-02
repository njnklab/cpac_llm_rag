import requests
import json

def call_ollama_api_generate(prompt, model = "llama3.1:8b-instruct-fp16", url_base="http://192.168.0.205:11434"):
    
    if isinstance(prompt, list):
        # 将列表转换为字符串，去掉两边的方括号，并保留内部结构
        prompt = str(prompt)[1:-1]
        # 替换单引号为双引号，以确保 JSON 格式正确
        prompt = prompt.replace("'", '"')
        
    print(f"Calling Ollama API with prompt(--prompt优化后---): {prompt}")
    
    """调用本地 Ollama API 生成 generate 语句"""
    url = f"{url_base}/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt
    }

    try:
        response = requests.post(url, json=data, stream=True)
        if response.status_code!= 200:
            print(f"Error response content: {response.text}")
            return f"Error: API returned status code {response.status_code}"
        
        res = ""
        
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if 'response' in json_response:
                    res += json_response['response']
                if json_response.get('done', False):
                    break
        # 移除“Tips for Searching”部分及其后的内容
        tips_start = res.find("Tips")
        if tips_start!= -1:
            res = res[:tips_start]
        return res
    except requests.RequestException as e:
        return f"Error connecting to Ollama API: {str(e)}"
    

def process_optimized_keywords(ollama_response):
    # 去除开头不需要的部分
    start = ollama_response.find("**Optimized Keywords:**")
    if start!= -1:
        ollama_response = ollama_response[start:]
    # 提取关键词部分
    keywords_start = ollama_response.find("*")
    if keywords_start!= -1:
        keywords_end = ollama_response.find("**Literature Databases & Journals:**")
        if keywords_end!= -1:
            keywords_section = ollama_response[keywords_start:keywords_end]
            # 进一步处理关键词部分，去除多余的描述
            keywords = []
            for line in keywords_section.splitlines():
                if line.startswith("*") and "For" not in line:
                    keyword = line.replace("*", "").strip()
                    keywords.append(keyword)
            # 合并关键词为一个字符串
            keyword_str = " ".join(keywords)
            # 提取数据库和期刊部分
            databases_start = ollama_response.find("**Literature Databases & Journals:**")
            if databases_start!= -1:
                databases_end = ollama_response.find("**Tips for Searching:**")
                if databases_end!= -1:
                    databases_section = ollama_response[databases_start:databases_end]
                    # 提取数据库和期刊名称
                    databases = []
                    for line in databases_section.splitlines():
                        if line.startswith("*") and "A" in line or "PubMed" in line:
                            database = line.replace("*", "").strip()
                            databases.append(database)
                    # 合并数据库和期刊名称为一个字符串
                    database_str = ", ".join(databases)
                    return f"Keywords: {keyword_str}, Databases/Journals: {database_str}"
    return ollama_response