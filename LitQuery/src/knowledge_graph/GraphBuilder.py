import os
import json
from llm_transformer import create_llm_transformer
from functions import process_all_pdfs, merge_graphs, optimize_graph, visualize_graph_from_json, save_graph_to_json,load_existing_graphs
from functions import read_graph_json

# 获取项目根目录（LitQuery目录）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 config.py 文件
import sys
sys.path.append(os.path.join(PROJECT_ROOT, 'config'))
import config  # 导入配置模块

# 设置 API 密钥
os.environ["OPENAI_BASE_URL"] = "https://www.askapi.chat/v1"
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# 配置LLMGraphTransformer，设置allowed_nodes和allowed_relationships
llm_transformer_filtered = create_llm_transformer()

# PDF 目录 和 处理记录（相对于项目根目录）
pdf_directory = os.path.join(PROJECT_ROOT, "data", "pdf_files")
processed_files_path = os.path.join(PROJECT_ROOT, "data", "processed_files.json")
graph_output_dir = os.path.join(PROJECT_ROOT, "data", "graphs")

# 确保输出目录存在
os.makedirs(pdf_directory, exist_ok=True)
os.makedirs(graph_output_dir, exist_ok=True)

# 加载已处理文件列表
if os.path.exists(processed_files_path) and os.path.getsize(processed_files_path) > 0:
    with open(processed_files_path, 'r') as f:
        processed_files = set(json.load(f))
        print(f"已处理文件列表 {processed_files}")
else:
    processed_files = set()

# 找到新添加的文件
all_files = set(os.listdir(pdf_directory))
new_files = all_files - processed_files

# 如果有新文件，处理并更新记录
if new_files:
    print(f"Processing new files: {new_files}")
    
    # 处理新文件生成知识图谱
    print(f"---------处理新文件生成知识图谱-----------")
    new_graphs = process_all_pdfs(pdf_directory, llm_transformer_filtered, new_files)  

    # 加载现有的图谱数据
    print(f"---------加载现有的图谱数据-----------")
    output_file = os.path.join(graph_output_dir, "merged_graphs.json")
    existing_graphs = load_existing_graphs(output_file)

    # 合并新生成的知识图谱
    print(f"---------合并新生成的知识图谱-----------")
    merged_graphs = merge_graphs(new_graphs, existing_graphs)
    print("merged_graphs:",merged_graphs)

    # 保存合并后的图谱到文件
    save_graph_to_json(merged_graphs, os.path.join(graph_output_dir, "merged_graphs.json"))

    # 优化合并后的知识图谱
    print(f"---------优化合并后的知识图谱-----------")
    optimized_graphs = optimize_graph(merged_graphs)
    print("optimized_graphs:",optimized_graphs)
    
    # 保存优化后的图谱到文件
    save_graph_to_json(optimized_graphs, os.path.join(graph_output_dir, "optimized_graph.json"))

    # 可视化合并后的知识图谱
    print(f"---------可视化合并后的知识图谱-----------")
    visualize_graph_from_json(os.path.join(graph_output_dir, "optimized_graph.json"))

    # 更新处理文件记录
    processed_files.update(new_files)
    with open(processed_files_path, 'w') as f:
        json.dump(list(processed_files), f)
else:
    print("No new files to process.")
