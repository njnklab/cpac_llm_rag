import os
import json
from langchain_community.document_loaders import PyPDFLoader
from fuzzywuzzy import fuzz
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
# 设置全局字体
mpl.rcParams['font.family'] = 'SimHei'
# import matplotlib.font_manager as fm
# font_properties = fm.FontProperties(fname='/home/user/zhangyan/RAG/LitAgent/GraphBuilder/SimHei.ttf')
from pathlib import Path


class Node:
    def __init__(self, id, type, properties):
        self.id = id
        self.type = type
        self.properties = properties
    
    def __hash__(self):
        # 使用 id 和 type 作为哈希依据，确保唯一性
        return hash((self.id, self.type))
    
    def __eq__(self, other):
        # 比较 id 和 type 确定节点是否相等
        if isinstance(other, Node):
            return self.id == other.id and self.type == other.type
        return False
    
    def __repr__(self):
        return f"Node(id={self.id}, type={self.type}, properties={self.properties})"
        
class Relationship:
    def __init__(self, source, target, type, properties):
        self.source = source
        self.target = target
        self.type = type
        self.properties = properties
    
    def __hash__(self):
        # 使用 source 和 target 的 id 以及关系类型作为哈希依据
        return hash((self.source.id, self.target.id, self.type))
    
    def __eq__(self, other):
        # 比较 source 和 target 的 id 以及类型判断关系是否相等
        if isinstance(other, Relationship):
            return (self.source.id == other.source.id and
                    self.target.id == other.target.id and
                    self.type == other.type)
        return False
    
    def __repr__(self):
        return f"Relationship(source={self.source}, target={self.target}, type={self.type}, properties={self.properties})"


def extract_knowledge_graph(pages, llm_transformer):
    # 创建文档对象
    # 使用递归字符文本分割器进行文本拆分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    for page in pages:
        texts.extend(text_splitter.split_text(page.page_content))

    # 将分割的文本合并为一个字符串
    full_text = "\n".join(texts)

    documents = [Document(page_content=full_text)]

    # 转换为图谱文档
    print("-----convert_to_graph_documents---------")
    graph_documents_filtered = llm_transformer.convert_to_graph_documents(documents)

    if graph_documents_filtered:
        nodes = graph_documents_filtered[0].nodes
        relationships = graph_documents_filtered[0].relationships
        print(f"Nodes: {nodes}")
        print(f"Relationships: {relationships}")
    else:
        print("No graph documents were returned.")

    return nodes, relationships

def process_single_pdf(file_path, llm_transformer):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print("----处理---", file_path)
        nodes, relationships = extract_knowledge_graph(pages, llm_transformer)
        return nodes, relationships
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None

def process_all_pdfs(directory, llm_transformer, files_to_process=None):
    """
    处理指定目录中的PDF文件，如果提供了文件列表，则只处理列表中的文件。
    
    参数：
    - directory (str): PDF文件的目录。
    - llm_transformer: 用于知识图谱抽取的LLM transformer。
    - files_to_process (set or list, optional): 要处理的特定文件列表。
    
    返回：
    - dict: 包含所有处理文件的知识图谱。
    """
    all_graphs = {}
    for filename in os.listdir(directory):
        # 如果提供了特定文件列表，则只处理列表中的文件
        if filename.endswith(".pdf") and (files_to_process is None or filename in files_to_process):
            file_path = os.path.join(directory, filename)
            nodes, relationships = process_single_pdf(file_path, llm_transformer)
            if nodes is not None and relationships is not None:
                all_graphs[filename] = {"nodes": nodes, "relationships": relationships}
    
    return all_graphs

def load_existing_graphs(file_path):
    """加载现有的 merged_graphs.json 数据，如果文件不存在则返回空字典。"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def read_graph_json(file_path):
    """
    Read and parse a JSON graph file.
    
    Args:
        file_path (str): Path to the JSON file containing graph data
        
    Returns:
        dict: Parsed JSON data containing nodes and relationships
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        # Convert string path to Path object
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Read and parse JSON file
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Validate basic structure
        if not isinstance(data, dict):
            raise ValueError("JSON data must be an object")
        if 'nodes' not in data or 'relationships' not in data:
            raise ValueError("JSON must contain 'nodes' and 'relationships' keys")
            
        return data
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def merge_graphs(new_graphs, existing_graphs):
    """
    合并新生成的知识图谱到已有的知识图谱。
    参数:
    - new_graphs (dict): 新生成的知识图谱。
    - existing_graphs (dict): 已有的知识图谱。
    
    返回:
    - dict: 合并后的完整知识图谱。
    """
    merged_nodes = existing_graphs.get("nodes", [])
    merged_relationships = existing_graphs.get("relationships", [])

    # 合并新生成的图谱数据
    for graph in new_graphs.values():
        # 添加新节点，避免重复
        for node in graph["nodes"]:
            if node not in merged_nodes:
                merged_nodes.append(node)

        # 添加新关系，避免重复
        for relationship in graph["relationships"]:
            if relationship not in merged_relationships:
                merged_relationships.append(relationship)

    # 更新已有的知识图谱数据
    return {"nodes": merged_nodes, "relationships": merged_relationships}

# def merge_graphs(all_graphs):
#     merged_nodes = []
#     merged_relationships = []

#     # 遍历每个PDF的图谱数据
#     for filename, graph in all_graphs.items():
#         # 将每个文件的节点加入到合并结果中，去掉完全相同的节点
#         for node in graph["nodes"]:
#             if node not in merged_nodes:
#                 merged_nodes.append(node)
        
#         # 将每个文件的关系加入到合并结果中，去掉完全相同的关系
#         for relationship in graph["relationships"]:
#             if relationship not in merged_relationships:
#                 merged_relationships.append(relationship)
    
#     return merged_nodes, merged_relationships


def optimize_graph(merged_graphs, similarity_threshold=80):
    """
    优化合并后的知识图谱，合并相似的节点和关系。
    
    参数:
    - merged_graphs (dict): 合并后的完整知识图谱，包含节点和关系。
    - similarity_threshold (int): 节点ID的相似度阈值，用于合并相似节点。
    
    返回:
    - dict: 包含优化后的节点和关系，格式与 merged_graphs 保持一致。
    """
    # 将节点和关系转换为字典格式
    nodes = [
        {"id": node.id, "type": node.type, "properties": node.properties}
        if hasattr(node, 'id') else node
        for node in merged_graphs["nodes"]
    ]
    
    relationships = [
        {
            "source": rel.source.id if hasattr(rel.source, 'id') else rel.source,
            "target": rel.target.id if hasattr(rel.target, 'id') else rel.target,
            "type": rel.type,
            "properties": rel.properties
        }
        if hasattr(rel, 'source') else rel
        for rel in merged_graphs["relationships"]
    ]

    # 创建一个有向图
    G = nx.DiGraph()

    # 将节点添加到图中
    for node in nodes:
        G.add_node(node["id"], type=node["type"], properties=node.get("properties", {}))

    # 将关系添加到图中
    for rel in relationships:
        source_id = rel["source"]
        target_id = rel["target"]
        # 确保source和target是字符串类型的节点ID
        if isinstance(source_id, dict) and "id" in source_id:
            source_id = source_id["id"]
        if isinstance(target_id, dict) and "id" in target_id:
            target_id = target_id["id"]
            
        G.add_edge(source_id, target_id, type=rel["type"], properties=rel.get("properties", {}))

    # 创建一个映射，用于合并相似的节点
    node_mapping = {}
    for node_id_1 in list(G.nodes):
        for node_id_2 in list(G.nodes):
            if node_id_1 != node_id_2:
                node_1 = G.nodes[node_id_1]
                node_2 = G.nodes[node_id_2]
                if node_1.get('type') == node_2.get('type') and fuzz.ratio(str(node_id_1), str(node_id_2)) >= similarity_threshold:
                    if node_id_2 not in node_mapping and node_id_1 not in node_mapping:
                        node_mapping[node_id_2] = node_id_1

    # 重新生成优化后的节点和关系
    optimized_nodes = []
    seen_nodes = set()  # 用于跟踪已处理的节点

    # 处理节点
    for node_id in G.nodes:
        mapped_id = node_mapping.get(node_id, node_id)
        if mapped_id not in seen_nodes:
            node = G.nodes[node_id]
            # 如果这是一个被映射的节点，合并属性
            if node_id in node_mapping:
                source_node = G.nodes[node_mapping[node_id]]
                merged_properties = {**node.get('properties', {}), **source_node.get('properties', {})}
                optimized_nodes.append({
                    "id": mapped_id,
                    "type": node.get('type', 'unknown'),  # 为缺失的 type 提供默认值
                    "properties": merged_properties
                })
            else:
                optimized_nodes.append({
                    "id": node_id,
                    "type": node.get('type', 'unknown'),  # 为缺失的 type 提供默认值
                    "properties": node.get('properties', {})
                })
            seen_nodes.add(mapped_id)

    # 更新关系
    optimized_relationships = []
    seen_relationships = set()  # 用于跟踪已处理的关系

    for u, v, data in G.edges(data=True):
        source_id = node_mapping.get(u, u)
        target_id = node_mapping.get(v, v)
        
        # 创建一个唯一标识符来检测重复的关系
        rel_identifier = (source_id, target_id, data['type'])
        
        if rel_identifier not in seen_relationships:
            if source_id in {node["id"] for node in optimized_nodes} and \
               target_id in {node["id"] for node in optimized_nodes}:
                optimized_relationships.append({
                    "source": source_id,
                    "target": target_id,
                    "type": data['type'],
                    "properties": data.get('properties', {})
                })
                seen_relationships.add(rel_identifier)

    return {"nodes": optimized_nodes, "relationships": optimized_relationships}

# def optimize_graph(nodes, relationships, similarity_threshold=80):
#     # --------相似度阈值设置为80----------
#     # 创建一个无向图
#     G = nx.DiGraph()  # 使用有向图（如果关系是有向的）

#     # 将节点添加到图中（使用节点的 id 和 type）
#     for node in nodes:
#         G.add_node(node.id, type=node.type, properties=node.properties)

#     # 将关系添加到图中（使用关系的 source 和 target ID）
#     for rel in relationships:
#         G.add_edge(rel.source.id, rel.target.id, type=rel.type, properties=rel.properties)

#     # 创建一个映射，用于合并相似的节点
#     node_mapping = {}
#     for node_id_1 in list(G.nodes):
#         for node_id_2 in list(G.nodes):
#             if node_id_1 != node_id_2:
#                 # 获取节点的相似度
#                 node_1 = G.nodes[node_id_1]
#                 node_2 = G.nodes[node_id_2]
#                 # 检查节点类型和相似度
#                 if node_1['type'] == node_2['type'] and fuzz.ratio(node_id_1, node_id_2) >= similarity_threshold:
#                     if node_id_2 not in node_mapping:  # 确保每个节点只被合并一次
#                         node_mapping[node_id_2] = node_id_1  # 将节点2映射到节点1

#     # 重新生成优化后的节点和关系
#     optimized_nodes = []
#     optimized_relationships = []

#     # 处理节点
#     for node_id in list(G.nodes):
#         if node_id not in node_mapping:
#             # 保留未被合并的节点
#             node = G.nodes[node_id]
#             optimized_nodes.append(Node(id=node_id, type=node['type'], properties=node['properties']))
#         else:
#             # 合并节点的属性可以进一步处理，如合并properties
#             source_node = G.nodes[node_mapping[node_id]]
#             target_node = G.nodes[node_id]
#             merged_properties = {**source_node['properties'], **target_node['properties']}  # 合并属性字典
#             optimized_nodes.append(Node(id=node_mapping[node_id], type=source_node['type'], properties=merged_properties))

#     # 更新关系（边）
#     for u, v, data in list(G.edges(data=True)):
#         source_id = node_mapping.get(u, u)  # 如果节点已合并，使用映射后的ID
#         target_id = node_mapping.get(v, v)  # 如果节点已合并，使用映射后的ID
#         optimized_relationships.append(Relationship(
#             source=Node(id=source_id, type=G.nodes[source_id]['type'], properties=G.nodes[source_id]['properties']),
#             target=Node(id=target_id, type=G.nodes[target_id]['type'], properties=G.nodes[target_id]['properties']),
#             type=data['type'],
#             properties=data['properties']
#         ))

#     return optimized_nodes, optimized_relationships

# # 将图谱数据保存到 JSON 文件
# def save_graph_to_json(graph_data, output_file):
#     # 检查输入的数据类型
#     if isinstance(graph_data, dict) and 'nodes' in graph_data and 'relationships' in graph_data:
#         nodes = graph_data['nodes']
#         relationships = graph_data['relationships']
#     elif isinstance(graph_data, tuple) and len(graph_data) == 2:
#         nodes, relationships = graph_data
#     else:
#         raise ValueError("Invalid graph data format. Expected a dictionary with 'nodes' and 'relationships' keys or a tuple of (nodes, relationships).")

#     # 序列化节点
#     serialized_nodes = [
#         {
#             'id': node.id if hasattr(node, 'id') else node['id'],
#             'type': node.type if hasattr(node, 'type') else node['type'],
#             'properties': node.properties if hasattr(node, 'properties') else node.get('properties', {})
#         } for node in nodes
#     ]
    
#     # 序列化关系
#     serialized_relationships = [
#         {
#             'source': relationship.source.id if hasattr(relationship.source, 'id') else relationship['source'],
#             'target': relationship.target.id if hasattr(relationship.target, 'id') else relationship['target'],
#             'type': relationship.type if hasattr(relationship, 'type') else relationship['type'],
#             'properties': relationship.properties if hasattr(relationship, 'properties') else relationship.get('properties', {})
#         } for relationship in relationships
#     ]
    
#     # 将节点和关系组织为图谱的结构
#     graph = {
#         'nodes': serialized_nodes,
#         'relationships': serialized_relationships
#     }
    
#     # 保存到JSON文件
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(graph, f, indent=4, ensure_ascii=False)
    
#     print(f"Graph has been saved to {output_file}")

def save_graph_to_json(graph_data, output_file):
    """
    通用的图谱数据保存函数，可以处理对象格式和字典格式的数据
    
    参数:
    - graph_data: 可以是包含 'nodes' 和 'relationships' 的字典，或者是 (nodes, relationships) 元组
    - output_file: 输出的 JSON 文件路径
    """
    # 检查并获取节点和关系数据
    if isinstance(graph_data, dict) and 'nodes' in graph_data and 'relationships' in graph_data:
        nodes = graph_data['nodes']
        relationships = graph_data['relationships']
    elif isinstance(graph_data, tuple) and len(graph_data) == 2:
        nodes, relationships = graph_data
    else:
        raise ValueError("Invalid graph data format. Expected a dictionary with 'nodes' and 'relationships' keys or a tuple of (nodes, relationships).")

    # 序列化节点
    def serialize_node(node):
        # 如果已经是字典格式，直接返回
        if isinstance(node, dict):
            return node
        # 如果是Node对象，转换为字典
        return {
            'id': node.id,
            'type': node.type,
            'properties': node.properties
        }

    # 序列化关系
    def serialize_relationship(rel):
        # 如果已经是字典格式，直接返回
        if isinstance(rel, dict):
            return rel
        # 如果是Relationship对象，转换为字典
        return {
            'source': rel.source.id if hasattr(rel.source, 'id') else rel.source,
            'target': rel.target.id if hasattr(rel.target, 'id') else rel.target,
            'type': rel.type,
            'properties': rel.properties
        }

    try:
        # 检查并创建文件夹（如果不存在）
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 序列化所有节点和关系
        serialized_nodes = [serialize_node(node) for node in nodes]
        serialized_relationships = [serialize_relationship(rel) for rel in relationships]
        
        # 构建最终的图谱结构
        graph = {
            'nodes': serialized_nodes,
            'relationships': serialized_relationships
        }
        
        # 保存到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=4, ensure_ascii=False)
        
        print(f"Graph has been saved to {output_file}")
        
    except Exception as e:
        print(f"Error while serializing graph data: {str(e)}")
        # 输出调试信息
        print("\nDebug information:")
        print(f"Nodes type: {type(nodes)}")
        print(f"Relationships type: {type(relationships)}")
        if nodes:
            print(f"First node type: {type(nodes[0])}")
        if relationships:
            print(f"First relationship type: {type(relationships[0])}")
        raise

    return graph


# # 将图谱数据保存到 JSON 文件
# def save_graph_to_json(graph_data, output_file):
#     # 检查输入的数据类型
#     if isinstance(graph_data, dict) and 'nodes' in graph_data and 'relationships' in graph_data:
#         # 如果是包含 "nodes" 和 "relationships" 的字典
#         nodes = graph_data['nodes']
#         relationships = graph_data['relationships']
#     elif isinstance(graph_data, tuple) and len(graph_data) == 2:
#         # 如果是包含节点和关系的元组
#         nodes, relationships = graph_data
#     else:
#         raise ValueError("Invalid graph data format. Expected a dictionary with 'nodes' and 'relationships' keys or a tuple of (nodes, relationships).")

#     # 序列化节点
#     serialized_nodes = [
#         {
#             'id': node.id,
#             'type': node.type,
#             'properties': node.properties
#         } for node in nodes
#     ]
    
#     # 序列化关系
#     serialized_relationships = [
#         {
#             'source': relationship.source.id,
#             'target': relationship.target.id,
#             'type': relationship.type,
#             'properties': relationship.properties
#         } for relationship in relationships
#     ]
    
#     # 将节点和关系组织为图谱的结构
#     graph = {
#         'nodes': serialized_nodes,
#         'relationships': serialized_relationships
#     }
    
#     # 保存到JSON文件
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(graph, f, indent=4, ensure_ascii=False)
    
#     print(f"Graph has been saved to {output_file}")

    
# # 将 all_graphs 保存到文件中，可以考虑使用JSON格式  这个后面弃用
# def save_graphs_to_json(all_graphs, output_file):
#     # 转换节点和关系为可序列化的格式
#     def serialize_graph(graph):
#         # 序列化节点
#         serialized_nodes = [
#             {
#                 'id': node.id,
#                 'type': node.type,
#                 'properties': node.properties
#             } for node in graph['nodes']
#         ]
        
#         # 序列化关系
#         serialized_relationships = [
#             {
#                 'source': relationship.source.id,
#                 'target': relationship.target.id,
#                 'type': relationship.type,
#                 'properties': relationship.properties
#             } for relationship in graph['relationships']
#         ]
        
#         return {
#             'nodes': serialized_nodes,
#             'relationships': serialized_relationships
#         }
    
#     # 序列化所有图谱
#     serialized_graphs = {filename: serialize_graph(graph) for filename, graph in all_graphs.items()}
    
#     # 保存到JSON文件
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(serialized_graphs, f, indent=4, ensure_ascii=False)
    
#     print(f"Graphs have been saved to {output_file}")

def load_graph_from_json(json_file):
    # 从 JSON 文件读取图数据
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取节点和关系
    nodes = data.get("nodes", [])
    relationships = data.get("relationships", [])
    
    return nodes, relationships

def visualize_graph_from_json(json_file, output_file='/home/user/zhangyan/RAG/LitAgent/GraphBuilder/pic/graph_visualization.svg', dpi=600):
    # 从JSON文件加载图数据
    nodes, relationships = load_graph_from_json(json_file)
    
    G = nx.Graph()
    
    # 添加所有节点，即使没有连边的节点也添加进去
    G.add_nodes_from([(node["id"], {"node_type": node.get("type", "default")}) for node in nodes])
    
    
    # 添加关系
    for rel in relationships:
        G.add_edge(rel["source"], rel["target"], relationship_type=rel.get("type", "association"))
    
    # 使用更适合大图的布局
    pos = nx.spring_layout(G, k=0.3, iterations=50)  # 调整k和迭代次数以增加节点之间的间距
    plt.figure(figsize=(30, 20))  # 增大画布尺寸

    # 调整节点大小和颜色
    degrees = dict(G.degree)
    node_sizes = [degrees[node] * 200 if degrees[node] > 0 else 500 for node in G.nodes()]  # 孤立节点大小设为500
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=6, font_weight='bold')
    
    # 绘制边，添加透明度以减少重叠
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, width=1)

    # 显示边的标签（关系类型）
    edge_labels = nx.get_edge_attributes(G, 'relationship_type')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=4)

    # 设置标题和布局
    plt.title("Knowledge Graph Visualization")
    plt.axis('off')
    plt.tight_layout()

    # 保存图像到文件，确保使用SVG格式以支持高分辨率放大
    plt.savefig(output_file, format='SVG', dpi=dpi)
    plt.close()
    print(f"Graph visualization saved to {output_file}")
    return output_file