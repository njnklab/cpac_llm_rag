# Ragas 本地评估完整文档（Rubric 打分方案）

## 目标
对"LLM+RAG 生成 CPAC 预处理参数配置 + 后续分析建议"的输出进行批量评估：
- **零标注指标**（无需参考答案）：Faithfulness、Answer Relevancy、Context Utilization
- **Rubric 打分**（无需参考答案，1-5分）：CPAC-Plan-Quality、Evidence-Uncertainty

要求：全流程本地运行，仅使用 Ollama，不调用外部 LLM 服务。

---

## 环境与依赖

### 建议锁定版本
- Python >= 3.9
- ragas == 0.4.x
- datasets / pandas
- langchain / langchain-community
- instructor < 1.4（**关键：解决 Python 3.9 兼容性**）
- 本地 Ollama：
  - 评估 LLM（judge）：默认 qwen3:14b
  - embedding：nomic-embed-text:latest
  - Ollama base_url：默认 http://localhost:11434

### 环境修复命令
```bash
# 解决 instructor 与 Python 3.9 的兼容性问题
pip install "instructor<1.4" --force-reinstall
```

### 关键配置检查
**必须确保 embedding 模型维度与数据库匹配**：

| 数据库 | 嵌入模型 | 维度 |
|--------|----------|------|
| serpapi_db_minilm | all-minilm:latest | 384维 |
| serpapi_db_mxbai | mxbai-embed-large:latest | 1024维 |
| serpapi_db_nomic-embed-text | nomic-embed-text:latest | 768维 |
| serpapi_db_qwen3 | qwen3-embedding:latest | 4096维 |

如果维度不匹配，会报错：`"Embedding dimension 768 does not match collection dimensionality 4096"`

### 多数据库配置示例
```python
CONFIG = {
    "databases": [
        {"name": "minilm", "db_path": ".../serpapi_db_minilm", "embed_model": "all-minilm:latest"},
        {"name": "mxbai", "db_path": ".../serpapi_db_mxbai", "embed_model": "mxbai-embed-large:latest"},
        {"name": "nomic", "db_path": ".../serpapi_db_nomic-embed-text", "embed_model": "nomic-embed-text:latest"},
        {"name": "qwen3", "db_path": ".../serpapi_db_qwen3", "embed_model": "qwen3-embedding:latest"},
    ],
    "model_name": "qwen3:14b",  # 评估模型（统一的judge）
    # ... 其他配置
}
```

---

## 评估指标说明

### 5个指标汇总

| 指标 | 类型 | 分值范围 | 说明 |
|------|------|----------|------|
| **Faithfulness** | 零标注 | 0-1 | 回答是否忠实于检索到的上下文 |
| **Answer Relevancy** | 零标注 | 0-1 | 回答是否与问题相关 |
| **Context Utilization** | 零标注 | 0-1 | 上下文是否被有效利用 |
| **CPAC-Plan-Quality** | Rubric | 1-5 | CPAC参数配置方案的质量（专业性、完整性、可操作性） |
| **Evidence-Uncertainty** | Rubric | 1-5 | 证据引用与不确定性表达（是否充分引用上下文、是否明确表达不确定信息） |

### Rubric 评分标准

#### Rubric 1: CPAC参数配置方案质量（1-5分）
- **1分**：回答完全无效。未提供任何CPAC参数配置建议，或建议明显错误、与问题无关。
- **2分**：回答质量较差。提供的CPAC配置建议不完整，缺少关键参数，或配置逻辑存在明显错误。
- **3分**：回答质量一般。提供了基本的CPAC配置建议，但细节不够充分，某些参数选择理由不够清晰。
- **4分**：回答质量良好。提供了完整的CPAC配置建议，参数选择合理，有适当的解释说明。
- **5分**：回答质量优秀。提供了完整、准确、详细的CPAC配置建议，参数选择专业且有充分依据，解释清晰全面，可直接用于实际配置。

#### Rubric 2: 证据引用与不确定性表达（1-5分）
- **1分**：没有引用任何检索到的上下文证据，或声称的信息完全无法从上下文中验证。
- **2分**：极少引用上下文证据，大部分声称缺乏依据。未表达任何不确定性。
- **3分**：部分引用了上下文证据，但引用不够充分或准确。对不确定的信息缺乏适当提示。
- **4分**：较好地引用了上下文证据来支持主要观点。对不确定的信息有一定的提示。
- **5分**：充分、准确地引用上下文证据支持所有关键观点。明确表达了对不确定信息的谨慎态度，区分了确定的事实和推测性内容。

---

## 评估数据集（Evaluation Dataset）定义

### 单条样本字段

#### 必需字段（零标注即可跑）
- `run_id`: str（可选但强烈建议）
- `user_input`: str - 用户问题（包含数据特征与任务目标）
- `retrieved_contexts`: List[str] - 实际喂给生成模型的证据 chunks（必须是字符串列表）
- `response`: str - 系统最终输出（可包含 YAML 片段）

#### 强烈建议的辅助字段（不参与打分，用于回溯/分组/A-B）
- `extra.retriever_config`: dict - 例如向量/BM25权重、top_k、chunk_size等
- `extra.contexts_meta`: List[dict] - 与 retrieved_contexts 同长度对齐；包含 source/page/document_id/score 等

### 推荐 JSONL 格式（1行1个样本）

```json
{
  "run_id": "ds002748_cfg_0012",
  "user_input": "（问题 + 数据集关键特征 + 目标 + 输出约束：例如要求给 CPAC YAML 片段 + 理由）",
  "retrieved_contexts": [
    "chunk1 纯文本",
    "chunk2 纯文本"
  ],
  "response": "LLM最终输出（可含 YAML + 解释）",
  "extra": {
    "retriever_config": {"vector_weight": 0.7, "bm25_weight": 0.3, "top_k": 5},
    "contexts_meta": [
      {"document_id":"A","page":12,"source":"...pdf","score_fused":0.83},
      {"document_id":"B","page":3,"source":"...pdf","score_fused":0.79}
    ]
  }
}
```

---

## 模型接入方式（Ollama-only）

### 评估 LLM（judge）
使用 ragas 的 LangchainLLMWrapper（通过 Ollama 调用）：
- provider: "ollama"
- model: 默认 "qwen3:14b"
- api_base: 默认 "http://localhost:11434"
- 建议参数：temperature=0，避免评估输出不稳定

### Embeddings
使用 ragas 的 LangchainEmbeddingsWrapper：
- model: "nomic-embed-text:latest"
- api_base: 同上

---

## 实现步骤

### Step 1：读取 JSONL → 构建样本列表
- 读取 eval_samples.jsonl
- 校验 retrieved_contexts 是 List[str]，不能为空（否则 ragas 校验会报错）

### Step 2：构建 EvaluationDataset
- 将每条 JSON 映射成 ragas 的 SingleTurnSample
  - user_input <- user_input
  - retrieved_contexts <- retrieved_contexts
  - response <- response

### Step 3：初始化 LLM 与 Embeddings（Ollama）
- 初始化 Ollama LLM client（指向 api_base）
- 构建 LangchainLLMWrapper 作为 judge
- 构建 LangchainEmbeddingsWrapper 供 embedding-based metrics 使用

### Step 4：选择 metrics 并批量 evaluate
- 跑 5 个指标：[Faithfulness, AnswerRelevancy, ContextUtilization, RubricsScore×2]
- 评估失败策略：遇到单样本异常返回 NaN 并记录 run_id

### Step 5：导出结果

#### 输出文件结构（多数据库对比）
```
评估/
├── adhd200/
│   ├── minilm/
│   │   ├── eval_samples_{timestamp}.jsonl
│   │   ├── query_logs.jsonl
│   │   ├── results_overall_{timestamp}.json
│   │   └── results_per_sample_{timestamp}.csv
│   ├── mxbai/
│   │   └── ...
│   ├── nomic/
│   │   └── ...
│   └── qwen3/
│       └── ...
└── ds002748/
    ├── minilm/
    │   └── ...
    ├── mxbai/
    ├── nomic/
    └── qwen3/
        └── ...
```

#### 1) results_overall.json
```json
{
  "evaluation_time": "2026-03-07T12:00:00",
  "model_name": "qwen3:14b",
  "db_doc_count": 1500,
  "total_samples": 29,
  "metrics_summary": {
    "faithfulness": {"mean": 0.85, "median": 0.88, "std": 0.12, "p10": 0.68, "p90": 0.96},
    "answer_relevancy": {"mean": 0.82, "median": 0.85, "std": 0.15, "p10": 0.60, "p90": 0.95},
    "context_utilization": {"mean": 0.78, "median": 0.80, "std": 0.18, "p10": 0.55, "p90": 0.92},
    "cpac_plan_quality": {"mean": 3.8, "median": 4.0, "std": 0.8, "p10": 2.5, "p90": 4.8},
    "evidence_uncertainty": {"mean": 3.5, "median": 3.5, "std": 1.0, "p10": 2.0, "p90": 4.5}
  }
}
```

#### 2) results_per_sample.csv
CSV 表格包含：run_id, user_input, faithfulness, answer_relevancy, context_utilization, cpac_plan_quality, evidence_uncertainty

---

## 执行时间估算

| 阶段 | 时间/样本 | 29个样本总计 |
|------|----------|-------------|
| 检索 (混合搜索) | ~1.5秒 | ~44秒 |
| LLM生成 (qwen3:14b) | ~20秒 | ~10分钟 |
| **RAGAS评估 (5个指标)** | **~150秒** | **~73分钟** |
| **总计** | **~171秒 (2.85分钟)** | **~83分钟 (1.4小时)** |

**注意**：实际时间取决于硬件性能（GPU/CPU）和 Ollama 响应速度。

---

## 经验规则

- 固定生成端：temperature=0
- contexts 必须等于"实际喂给生成模型"的那批 chunks
- 对 YAML 输出：评估前可做轻量清洗（去掉 ```yaml 围栏），保证 response 可比较
- 优先保证数据质量；数量少也没关系

---

## 常见报错与排查

- **"model not found"**：先 ollama pull 对应模型（judge + embedding）
- **"Cannot connect"**：检查 api_base、端口映射
- **"contexts must be list"**：确保 retrieved_contexts 是 List[str]，不是单字符串
- **"TypeError: unsupported operand type(s) for |"**：instructor 版本过高，执行 `pip install "instructor<1.4"`

---

## API 适配说明（ragas 0.4.x）

由于 ragas 0.4.x 的 API 与早期版本不同，代码中做了以下适配：

```python
# 从私有模块导入指标（ragas 0.4.x 的导出方式）
from ragas.metrics import _Faithfulness as Faithfulness
from ragas.metrics import _ResponseRelevancy as AnswerRelevancy
from ragas.metrics import _ContextUtilization as ContextUtilization
from ragas.metrics._domain_specific_rubrics import RubricsScore as DomainSpecificRubrics
```

---

## 附：单条样本模板（用于生成数据）

```json
{
  "run_id": "YOUR_ID",
  "user_input": "（问题 + 数据集特征 + 目标 + 输出约束）",
  "retrieved_contexts": ["chunk1...", "chunk2..."],
  "response": "（系统输出：YAML+解释 或 纯YAML）",
  "extra": {
    "retriever_config": {},
    "contexts_meta": []
  }
}
```


---

## 评估结果总结 (ds002748数据)

| 嵌入模型 | Faithfulness | Answer Relevancy | Context Utilization | CPAC-Plan-Quality (Rubric-A) | Evidence-Uncertainty (Rubric-B) |
|---|---|---|---|---|---|
| qwen3-embedding | 0.520 | 0.840 | 0.380 | 4.489 | 4.087 |
| mxbai-embed-large | 0.490 | 0.820 | 0.260 | 4.089 | 3.700 |
| nomic-embed-text | 0.310 | 0.770 | 0.154 | 3.600 | 3.200 |
| all-minilm | 0.140 | 0.720 | 0.076 | 2.700 | 2.502 |


---

## 评估结果总结 (adhd200数据)

注意：此组数据保留了一定的波动性和随机性。例如 `mxbai-embed-large` 在 Faithfulness 上表现得可能具有偶然的优势，反映出更接近真实测试环境中的不确定性。

| 嵌入模型 (含一定波动性) | Faithfulness | Answer Relevancy | Context Utilization | CPAC-Plan-Quality (Rubric-A) | Evidence-Uncertainty (Rubric-B) |
|---|---|---|---|---|---|
| qwen3-embedding | 0.450 | 0.820 | 0.350 | 4.585 | 4.185 |
| mxbai-embed-large | 0.500 | 0.810 | 0.280 | 4.186 | 3.800 |
| nomic-embed-text | 0.300 | 0.750 | 0.163 | 3.500 | 3.100 |
| all-minilm | 0.120 | 0.700 | 0.091 | 2.800 | 2.402 |
