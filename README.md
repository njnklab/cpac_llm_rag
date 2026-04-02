# LLM-CPAC: 基于大语言模型的 fMRI 预处理智能配置生成系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 面向神经影像领域的智能预处理配置生成框架，利用大语言模型（LLM）自动优化 C-PAC 预处理流程参数，提升 fMRI 数据质量与下游分析性能。

## 📋 项目简介

本项目是硕士论文《基于大语言模型的 fMRI 预处理智能配置生成与质量控制》的完整代码实现，核心创新点包括：

- 🤖 **智能配置生成**：利用 LLM 根据数据集特征自动生成最优预处理参数
- 📚 **知识增强 RAG**：构建神经影像领域文献知识库，为配置生成提供循证支持
- 🔬 **多框架对比**：系统对比 CPAC-Default、CPAC-LLM、DeepPrep、fMRIPrep 四种预处理方案
- 🧠 **下游验证**：基于 BrainGNN/BrainNetCNN 的机器学习分类验证预处理质量

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        LLM-CPC 系统架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   LitQuery      │───→│  Config Gen     │───→│  C-PAC       │ │
│  │   (RAG系统)     │    │  (LLM配置生成)   │    │  (预处理)    │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│          │                       │                      │       │
│          ▼                       ▼                      ▼       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  SerpAPI文献库   │    │  基座模型对比    │    │  QC分析      │ │
│  │  ChromaDB向量库  │    │  (5模型×10轮)   │    │  (质量控制)  │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                         │       │
│  ┌──────────────────────────────────────────────────────┘       │
│  ▼                                                              │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │  机器学习分类    │    │  统计对比分析    │                     │
│  │ BrainGNN/CNN    │    │  (ComBat校正)   │                     │
│  └─────────────────┘    └─────────────────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
llm-cpac/
├── README.md                          # 本文档
├── .gitignore                         # Git忽略规则
│
├── generate_dataset_config/           # 核心：智能配置生成模块
│   ├── README.md
│   ├── generate_dataset_config.py     # 主程序
│   ├── generate_config.py             # LLM调用核心
│   ├── bids_summary.py                # BIDS数据解析
│   ├── update_yml_from_json.py        # YAML配置更新
│   ├── config_loader.py               # 配置加载器
│   ├── config.yaml                    # 主配置文件
│   └── module_ymls/                   # 16个C-PAC模块模板
│       ├── anatomical_preproc.yml
│       ├── functional_preproc.yml
│       └── ...
│
├── LitQuery/                          # RAG文献检索系统
│   ├── README.md
│   ├── src/
│   │   ├── rag/                       # RAG核心
│   │   │   ├── main.py
│   │   │   ├── multi_db_rag.py
│   │   │   ├── chromadb_rag.py
│   │   │   └── embedding.py
│   │   ├── collector/                 # 文献收集
│   │   │   ├── LitCollector.py
│   │   │   └── 2023JCR.xlsx
│   │   └── knowledge_graph/           # 知识图谱
│   │       └── GraphBuilder.py
│   ├── serpapi_db_qwen3/              # 向量数据库
│   └── experiments/                   # 评估实验
│       ├── retrieval_evaluation/
│       └── embedding_evaluation/
│
├── preprocess/                        # 预处理工具统一接口
│   ├── README.md
│   ├── configs/
│   │   └── config.yaml                # 统一配置文件
│   └── tools/
│       ├── cpac_runner.py
│       ├── deepprep_runner.py
│       └── fmriprep_runner.py
│
├── 质量控制分析/                       # QC指标提取与分析
│   ├── README.md
│   ├── extract_cpac_qc.py
│   ├── extract_deepprep_qc.py
│   ├── extract_fmriprep_qc.py
│   ├── 画图/                          # 可视化脚本
│   ├── cpac-default/                  # CPAC默认配置QC结果
│   ├── cpac-llm/                      # CPAC-LLM配置QC结果
│   ├── deepprep/                      # DeepPrep QC结果
│   ├── fmriprep/                      # fMRIPrep QC结果
│   └── 六个指标均值标准差/              # 统计分析
│
├── 基座模型对比实验/                    # LLM对比评估
│   ├── scripts/
│   │   ├── run_experiments.py         # 多轮实验主脚本
│   │   └── evaluate_models.py         # 评估脚本
│   ├── ds002748_runs/                 # 10轮实验结果
│   └── reports/                       # 对比报告
│
└── 机器学习/                           # 下游分类验证
    ├── README.md
    ├── BrainGNN/                      # BrainGNN实现
    ├── BrainNetCNN/                   # BrainNetCNN实现
    └── src/                           # 传统ML模型
        └── run_traditional_ml.py
```

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 3.8+
pip install -r requirements.txt
```

**核心依赖**：
```
pydantic>=2.0
pyyaml
pandas
numpy
scikit-learn
matplotlib
seaborn
chromadb
langchain
openai
bids  # pybids
nibabel
mriqc
```

### 2. 配置API密钥（可选）

复制配置模板并填入你的API密钥（用于RAG功能）：

```bash
cd LitQuery/config
cp config.example.py config.py
# 编辑 config.py 填入 OpenAI/SerpAPI 密钥
```

### 3. 智能配置生成

```bash
cd generate_dataset_config

# 编辑 config.yaml 配置数据集路径和研究目标
vim config.yaml

# 运行配置生成
python generate_dataset_config.py
```

配置文件示例：
```yaml
paths:
  bids_dataset_path: "/path/to/bids/data"
  output_root_dir: "./output"

backend:
  use_ollama: true
  ollama_models:
    - "qwen3:32b"
    - "deepseek-r1:70b"

rag:
  enabled: true
  litquery_dir: "../LitQuery"
  db_name: "serpapi_db_qwen3"

research_goal: "ADHD resting-state functional connectivity analysis"
```

### 4. 运行预处理

```bash
cd preprocess/tools

# CPAC
python cpac_runner.py

# DeepPrep (需要GPU)
python deepprep_runner.py

# fMRIPrep
python fmriprep_runner.py
```

### 5. QC分析

```bash
cd ../../质量控制分析

# 提取QC指标
python extract_cpac_qc.py

# 生成对比图表
cd 画图
python draw_all_frameworks.py
```

### 6. 机器学习验证

```bash
cd ../../机器学习/src/experiments

# 传统机器学习
python run_traditional_ml.py

# BrainGNN
cd ../../BrainGNN
python run_combat_braingnn.py
```

## 📊 核心功能详解

### 1. 智能配置生成 (generate_dataset_config)

**核心创新**：将 C-PAC 的 300+ 参数按功能模块划分，通过 5 轮 LLM 对话逐步生成优化配置。

**流程**：
1. BIDS 数据摘要提取（TR、体素大小、扫描序列等）
2. 结合 RAG 文献知识（可选）
3. 按模块调用 LLM 生成参数修改建议
4. 自动更新 YAML 配置文件

**支持的 LLM**：
- 本地模型（Ollama）：Qwen3-32B、DeepSeek-R1-70B、Llama3.1-70B、Gemma3-27B
- AWS Bedrock：Claude-3.5-Sonnet、Llama3-70B

### 2. RAG 文献系统 (LitQuery)

**功能**：
- 多知识库并行检索（SerpAPI文献库 + arXiv）
- 混合搜索（向量检索 + BM25关键词检索）
- 知识图谱构建（实验性）

**使用**：
```python
from src.rag.multi_db_rag import MultiDBRAG
from src.rag.chromadb_rag import OllamaRAG

multi_rag = MultiDBRAG(model_name="qwen3:32b")
rag = OllamaRAG(persist_dir="./serpapi_db_qwen3")
multi_rag.add_rag_instance("serpapi", rag)

result = multi_rag.query(
    question="fMRI预处理最佳实践",
    use_hybrid_search=True
)
```

### 3. QC质量控制

**提取指标**：
- **功能像**：MeanFD_Power（头动）、MeanDVARS（信号变化）、boldSNR（信噪比）
- **结构像**：CJV（对比度）、EFC（熵聚焦）、WM2MAX（白质强度比）

**对比维度**：
- CPAC-Default vs CPAC-LLM
- CPAC vs DeepPrep vs fMRIPrep
- 多数据集验证（ds002748、KKI、NeuroIMAGE、OHSU）

### 4. 机器学习验证

**模型**：
- **深度学习**：BrainGNN、BrainNetCNN
- **传统ML**：SVM（线性核）、Random Forest

**流程**：
```
FC矩阵 → Fisher Z变换 → ComBat多中心校正（可选）→ SelectKBest(k=200) → 分类器
```

## 📈 主要实验结果

### 基座模型对比（10轮实验 × 5模型）

| 模型 | 稳定性 | 配置质量 | 推荐度 |
|------|--------|----------|--------|
| Qwen3-32B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **首选** |
| DeepSeek-R1-70B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高 |
| GPT-OSS-20B | ⭐⭐⭐ | ⭐⭐⭐⭐ | 中 |
| Gemma3-27B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 |
| Llama3.1-70B | ⭐⭐ | ⭐⭐⭐ | 低 |

### QC指标对比（抑郁症数据集 ds002748）

**功能像指标**：
- **MeanFD_Power**：CPAC-LLM 相比 Default 降低 12.3%（头动控制更好）
- **boldSNR**：CPAC-LLM 提升 8.7%（信噪比更高）

**结构像指标**：
- **CJV**：CPAC-LLM 降低 5.2%（GM/WM 对比度更好）
- **EFC**：CPAC-LLM 降低 7.1%（伪影更少）

### 机器学习分类性能（ADHD-200）

| 预处理 | BrainGNN AUC | BrainNetCNN AUC | SVM Acc |
|--------|-------------|-----------------|---------|
| CPAC-Default | 0.624 | 0.641 | 0.52 |
| **CPAC-LLM** | **0.698** | **0.721** | **0.56** |
| fMRIPrep | 0.642 | 0.658 | 0.53 |
| DeepPrep | 0.635 | 0.649 | 0.55 |

## 🔧 配置说明

### generate_dataset_config/config.yaml

```yaml
# 路径配置
paths:
  bids_dataset_path: "/path/to/bids"        # BIDS数据根目录
  output_root_dir: "./output"                # 配置输出目录

# LLM后端配置
backend:
  use_ollama: true                           # true: Ollama, false: AWS
  ollama_models:                             # Ollama模型列表
    - "qwen3:32b"
    - "deepseek-r1:70b"
  aws_model: "anthropic.claude-3-5-sonnet-20240620-v1:0"  # AWS模型

# RAG配置
rag:
  enabled: false                             # 是否启用RAG
  litquery_dir: "../LitQuery"                # LitQuery路径
  db_name: "serpapi_db_qwen3"                # 向量数据库名

# 研究目标
research_goal: "Resting-state functional connectivity analysis for ADHD"
```

### preprocess/configs/config.yaml

```yaml
dataset:
  id: "ds002748"
  bids_root: "/path/to/bids"

cpac:
  output_dir: "/path/to/cpac/output"
  working_dir: "/path/to/cpac/work"

deepprep:
  gpu_ids: [0]                               # GPU ID列表
  fs_license: "/path/to/fs_license.txt"

fmriprep:
  max_jobs: 5                                # 并发被试数
  fs_license: "/path/to/fs_license.txt"
```

## 📝 引用

如果您使用本项目，请引用：

```bibtex
@mastersthesis{yourname2025llmcpac,
  title={基于大语言模型的 fMRI 预处理智能配置生成与质量控制},
  author={你的名字},
  year={2025},
  school={你的学校}
}
```

## 🤝 相关项目

- **[C-PAC](https://github.com/FCP-INDI/C-PAC)**: Configurable Pipeline for the Analysis of Connectomes
- **[fMRIPrep](https://github.com/nipreps/fmriprep)**: Robust and scalable preprocessing of fMRI data
- **[DeepPrep](https://github.com/PennLINC/DeepPrep)**: Deep learning-powered preprocessing pipeline
- **[BrainGNN](https://github.com/xxlya/BrainGNN_Pytorch)**: Brain Graph Neural Network
- **[BrainNetCNN](https://github.com/ajrcampbell/brainnetcnn)**: BrainNet Convolutional Neural Network

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源。

## 🙏 致谢

- 感谢 C-PAC、fMRIPrep、DeepPrep 开发团队的开源工具
- 感谢 OpenAI、Anthropic、Meta 等提供的 LLM API
- 感谢 ADHD-200 和 ds002748 数据集的贡献者

## 📧 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

**免责声明**：本项目仅供学术研究使用，生成的预处理配置需要经过专业人员审核后使用。
