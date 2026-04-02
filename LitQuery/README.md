# LitQuery - 文献检索RAG系统

LitQuery 是一个面向神经影像预处理领域的文献检索增强生成(RAG)系统，支持多知识库检索、混合搜索策略和知识图谱构建。

## 📁 项目结构

```
LitQuery/
├── README.md                          # 本文档
├── .gitignore                         # Git忽略规则
│
├── config/                            # 配置目录
│   ├── config.py                      # 配置文件（包含API密钥）
│   └── config.example.py              # 配置示例模板
│
├── src/                               # 源代码目录
│   ├── rag/                           # RAG核心模块
│   │   ├── main.py                    # 多知识库查询入口
│   │   ├── multi_db_rag.py            # 多数据库RAG协调器
│   │   ├── chromadb_rag.py            # ChromaDB向量存储实现
│   │   ├── serpapi_rag.py             # SerpAPI网络搜索集成
│   │   └── embedding.py               # 文本向量化模块
│   │
│   ├── collector/                     # 文献收集模块
│   │   ├── LitCollector.py            # 主收集器（轮换查询批量收集）
│   │   ├── google_search.py           # Google Scholar搜索
│   │   ├── extract_info.py            # PDF信息提取
│   │   ├── ollama_api.py              # Ollama API封装
│   │   ├── ollama_client.py           # Ollama客户端
│   │   └── 2023JCR.xlsx               # 期刊分区表
│   │
│   └── knowledge_graph/               # 知识图谱模块（实验性）
│       ├── GraphBuilder.py            # 知识图谱构建主程序
│       ├── llm_transformer.py         # LLM图转换器
│       └── functions.py               # 图操作函数库
│
├── experiments/                       # 实验与评估
│   ├── retrieval_evaluation/          # 检索策略评估
│   │   ├── run_experiment.py          # 主实验脚本
│   │   ├── evaluate_retrieval_quality.py  # 检索质量评估
│   │   ├── test_vector_dbs.py         # 向量数据库测试
│   │   ├── plot_results.py            # 结果可视化
│   │   ├── expand_results.py          # 结果扩展处理
│   │   ├── modify_results.py          # 结果修改处理
│   │   ├── 检索策略对比.md            # 检索策略评估报告
│   │   ├── result/                    # RAGAS评估结果（CSV/JSON）
│   │   │   ├── samples_*.jsonl        # 样本数据
│   │   │   ├── ragas_*.csv            # 每样本评估结果
│   │   │   └── ragas_*.json           # 总体评估结果
│   │   └── plot/                      # 可视化图表（PNG）
│   │
│   └── embedding_evaluation/          # 嵌入模型评估
│       ├── ragas_evaluate.py          # 主评估脚本
│       ├── ragas_evaluate_ds002748.py # ds002748专用评估
│       ├── build_embed_db-minilm.py   # 向量数据库构建
│       ├── ragas本地评估.md           # 嵌入模型评估说明
│       ├── ds002748/                  # ds002748数据集评估结果
│       │   ├── temp/                  # 临时结果（qwen3/nomic/mxbai/minilm）
│       │   └── update_metrics*.py     # 结果更新脚本
│       └── adhd200/                   # ADHD-200数据集评估结果
│           └── temp/                  # 临时结果（qwen3/nomic/mxbai/minilm）
│
├── serpapi_db_qwen3/                  # ChromaDB向量数据库（实验必需）
│   └── chroma.sqlite3                 # 向量数据主文件
│
└── docs/                              # 文档目录（预留）
```

## 🚀 快速开始

### 1. 环境配置

复制配置模板并填入你的API密钥：

```bash
cd config
cp config.example.py config.py
# 编辑 config.py 填入你的API密钥
```

### 2. 安装依赖

```bash
pip install chromadb langchain langchain-community langchain-openai sentence-transformers
pip install PyPDF2 pandas openpyxl
pip install ollama  # 如果使用本地模型
```

### 3. 使用RAG系统

```bash
# 方法1：从项目根目录运行
cd /path/to/LitQuery
python -m src.rag.main

# 方法2：先进入目录再运行
cd src/rag
python main.py
```

系统会读取 `questions.txt` 文件（每行一个问题的文本文件）并执行多知识库查询。

### 4. 文献收集

```bash
# 批量收集文献
cd src/collector
python LitCollector.py
```

## 🔧 核心功能

### RAG检索系统
- **多知识库支持**: 可同时查询多个ChromaDB数据库
- **混合搜索**: 结合向量检索和BM25关键词检索
- **模型灵活切换**: 支持Ollama本地模型和OpenAI API

### 文献收集
- **轮换查询**: 自动轮换多个关键词批量收集文献
- **去重机制**: Excel表格自动去重，避免重复下载
- **JCR分区**: 结合2023JCR表评估期刊质量

### 知识图谱（实验性）
- 从PDF构建神经影像知识图谱
- 支持图谱合并与优化
- 可视化展示

## 📊 实验评估

### 检索策略对比

- 对比三种检索策略：**纯向量检索** vs **BM25关键词检索** vs **混合搜索**
- 使用 **RAGAS** 框架评估检索质量
- 评估指标：
  - `context_utilization` - 上下文利用率
  - `faithfulness` - 忠实度
  - `answer_relevancy` - 答案相关性
  - `cpac_plan_quality` (Rubric-A) - CPAC规划质量
  - `evidence_uncertainty` (Rubric-B) - 证据不确定性
- 结果保存在 `experiments/retrieval_evaluation/result/`

### 嵌入模型评估

- 对比4个嵌入模型：
  - **qwen3-embedding** - 4.7GB，性能最优
  - **mxbai-embed-large** - 669MB，性能中等
  - **nomic-embed-text** - 轻量级
  - **all-minilm** - 45MB，速度最快
- 在 **ds002748**（抑郁症）和 **ADHD-200** 数据集上测试
- 结果保存在 `experiments/embedding_evaluation/ds002748/` 和 `adhd200/`

## 🗂️ 数据库说明

`serpapi_db_qwen3/` 目录包含 **ChromaDB向量数据库**，存储了从SerpAPI收集的文献向量数据。

- **用途**：实验必需，用于RAG检索
- **内容**：文献PDF的嵌入向量和元数据
- **注意**：文件较大（数百MB），Git提交时请确保包含

## ⚙️ 配置说明

主要配置项在 `config/config.py`：
- `OPENAI_API_KEY`: OpenAI API密钥（用于RAG总结）
- `OPENAI_BASE_URL`: OpenAI API代理地址
- `SERPAPI_API_KEY`: SerpAPI密钥（用于文献搜索）

## 📝 使用示例

### 单知识库查询

```python
from src.rag.chromadb_rag import OllamaRAG

rag = OllamaRAG(
    persist_dir="./serpapi_db_qwen3",
    model_name="qwen3:32b",
    embed_model="nomic-embed-text:latest"
)

result = rag.query("fMRI预处理的最佳实践是什么？", n_results=5)
print(result["answer"])
```

### 多知识库查询

```python
from src.rag.multi_db_rag import MultiDBRAG
from src.rag.chromadb_rag import OllamaRAG

multi_rag = MultiDBRAG(model_name="qwen3:32b")

# 添加多个数据库
rag1 = OllamaRAG(
    persist_dir="./serpapi_db_qwen3",
    model_name="qwen3:32b",
    embed_model="nomic-embed-text:latest"
)
multi_rag.add_rag_instance("serpapi_db", rag1)

# 并行查询
result = multi_rag.query(
    question="抑郁症fMRI预处理",
    n_results=5,
    use_hybrid_search=True,
    use_openai=False
)
print(result["answer"])
```

## 📖 相关文档

- `experiments/retrieval_evaluation/检索策略对比.md` - 检索策略评估报告
- `experiments/embedding_evaluation/ragas本地评估.md` - 嵌入模型评估说明

## 🔌 模块导入说明

由于代码结构采用 `src/` 子目录组织，导入模块时需要注意路径：

### 在LitQuery内部导入

```python
# src/rag/chromadb_rag.py 中导入config
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'config'))
import config
```

### 从外部项目导入

```python
# 在 generate_dataset_config.py 中使用LitQuery
import sys
LITQUERY_DIR = '/path/to/LitQuery'

# 添加RAG模块路径
rag_module_dir = os.path.join(LITQUERY_DIR, 'src', 'rag')
if rag_module_dir not in sys.path:
    sys.path.append(rag_module_dir)

# 添加config路径
config_module_dir = os.path.join(LITQUERY_DIR, 'config')
if config_module_dir not in sys.path:
    sys.path.append(config_module_dir)

# 现在可以导入
from multi_db_rag import MultiDBRAG
from chromadb_rag import OllamaRAG
```

## 🤝 集成说明

本系统与 `generate_dataset_config` 模块集成，可为C-PAC配置生成提供文献支持：

```python
# 在 generate_dataset_config.py 中启用RAG
USE_RAG = True
LITQUERY_DIR = '/path/to/LitQuery'
RAG_DB_NAME = 'serpapi_db_qwen3'
```

## ⚠️ 注意事项

1. API密钥已包含在代码中（废弃密钥），如需使用请更换为自己的密钥
2. 向量数据库文件较大，Git提交时请注意
3. 知识图谱功能为实验性，路径需要根据实际情况配置

## 📧 联系方式

如有问题请联系项目负责人。
