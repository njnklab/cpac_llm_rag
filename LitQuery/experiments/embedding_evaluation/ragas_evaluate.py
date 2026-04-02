#!/usr/bin/env python3
"""
RAGAS本地评估脚本 - 用于评估CPAC/MRI预处理参数配置RAG系统

功能：
1. 批量执行RAG查询并收集结果
2. 使用ragas进行零标注指标评估（Faithfulness, AnswerRelevancy, ContextUtilization）
3. 使用Rubric打分评估（CPAC-Plan-Quality, Evidence-Uncertainty）
4. 全流程使用本地Ollama，不调用外部API
5. 详细记录每次查询过程

评估指标（共5个）：
- Faithfulness (0-1): 回答是否忠实于检索到的上下文
- AnswerRelevancy (0-1): 回答是否与问题相关
- ContextUtilization (0-1): 上下文是否被有效利用
- CPAC-Plan-Quality (1-5): CPAC参数配置方案的质量
- Evidence-Uncertainty (1-5): 证据引用与不确定性表达

依赖：
- ragas >= 0.1.0
- datasets
- pandas
- chromadb
- ollama（本地运行）

使用方法：
    # 评估ADHD-200数据集
    python ragas_evaluate.py

    # 或者修改文件底部的 CONFIG 字典来切换数据集
"""

import os
import json
import csv
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ==================== 用户配置区域 ====================
# 修改以下配置来适应你的需求

CONFIG = {
    # 数据库配置 - 支持多个嵌入模型对比评估
    # 每个数据库需要配置对应的嵌入模型（维度必须匹配）
    "databases": [
        {
            "name": "minilm",
            "db_path": "/home/a001/zhangyan/LitQuery/serpapi_db_minilm",
            "embed_model": "all-minilm:latest",  # 384维
        },
        {
            "name": "mxbai",
            "db_path": "/home/a001/zhangyan/LitQuery/serpapi_db_mxbai",
            "embed_model": "mxbai-embed-large:latest",  # 1024维
        },
        {
            "name": "nomic",
            "db_path": "/home/a001/zhangyan/LitQuery/serpapi_db_nomic-embed-text",
            "embed_model": "nomic-embed-text:latest",  # 768维
        },
        {
            "name": "qwen3",
            "db_path": "/home/a001/zhangyan/LitQuery/serpapi_db_qwen3",
            "embed_model": "qwen3-embedding:latest",  # 4096维
        },
    ],

    # 模型配置
    "model_name": "qwen3:14b",  # RAG生成模型 & RAGAS评估模型(judge)
    "ollama_base_url": "http://localhost:11434",
    "temperature": 0.0,  # 评估时建议为0

    # 检索配置
    "n_results": 5,  # 每个查询检索的文档数量
    "use_hybrid_search": True,  # 是否使用混合搜索

    # 数据集配置 - 选择要评估的数据集
    # 可选: "adhd200", "ds002748", "both"
    "dataset": "adhd200",

    # 数据文件路径
    "queries_files": {
        "adhd200": "/home/a001/zhangyan/LitQuery/queries/adhd200_queries.txt",
        # "ds002748": "/home/a001/zhangyan/LitQuery/queries/ds002748_queries.txt",
    },

    # 输出配置
    "output_base_dir": "/home/a001/zhangyan/LitQuery/评估",

    # 评估配置
    "skip_eval": False,  # 是否跳过评估，仅生成样本
}

# ==================== 配置区域结束 ====================

# 尝试导入ragas和相关依赖，如果失败则提示安装
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("警告: numpy 未安装")
    print("请安装: pip install numpy")
    NUMPY_AVAILABLE = False

try:
    from ragas import evaluate
    # ragas 0.4.x API适配
    from ragas.metrics import _Faithfulness as Faithfulness
    from ragas.metrics import _ResponseRelevancy as AnswerRelevancy
    from ragas.metrics import _ContextUtilization as ContextUtilization
    from ragas.metrics._domain_specific_rubrics import RubricsScore as DomainSpecificRubrics
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.run_config import RunConfig
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    RAGAS_AVAILABLE = True
    # 检测 ragas 版本
    import ragas
    try:
        version_str = ragas.__version__
        # 提取前两个数字（例如 "0.1.21" -> (0, 1)）
        version_parts = version_str.split('.')[:2]
        RAGAS_VERSION = tuple(int(p) for p in version_parts)
    except (AttributeError, ValueError) as e:
        print(f"警告: 无法解析ragas版本，假设为旧版本: {e}")
        RAGAS_VERSION = (0, 1)  # 默认假设为旧版本
    print(f"ragas 版本: {ragas.__version__}, 解析为: {RAGAS_VERSION}")
except ImportError as e:
    print(f"警告: 无法导入ragas或相关依赖: {e}")
    print("请安装: pip install ragas datasets pandas langchain langchain-community numpy")
    RAGAS_AVAILABLE = False
    RAGAS_VERSION = (0, 0)

# 导入项目内的RAG类
from chromadb_rag import OllamaRAG
from multi_db_rag import MultiDBRAG


@dataclass
class QueryLog:
    """单个查询的详细日志"""
    query_idx: int
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    question: str = ""
    retrieved_contexts: List[str] = field(default_factory=list)
    context_sources: List[Dict] = field(default_factory=list)
    response: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvalSample:
    """评估样本数据结构"""
    run_id: str
    user_input: str
    retrieved_contexts: List[str]
    response: str
    reference: Optional[str] = None
    extra: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvalResult:
    """单个样本的评估结果"""
    run_id: str
    user_input: str
    metrics: Dict[str, float]
    error: Optional[str] = None


def estimate_time(db_doc_count: int, n_questions: int, use_hybrid: bool) -> Dict[str, float]:
    """
    估算处理时间
    
    基于以下假设：
    - 单次检索（混合搜索）: ~0.5-2秒（取决于数据库大小）
    - 单次检索（纯向量）: ~0.3-1秒
    - LLM生成回答(qwen3:14b): ~10-30秒（取决于回答长度）
    - RAGAS评估每个指标: ~15-45秒（每个指标需要多次LLM调用）
    
    Args:
        db_doc_count: 数据库文档数量
        n_questions: 问题数量
        use_hybrid: 是否使用混合搜索
        
    Returns:
        时间估算字典（秒）
    """
    # 检索时间（混合搜索更慢）
    if use_hybrid:
        retrieval_time_per_query = 1.5  # 秒
    else:
        retrieval_time_per_query = 0.8  # 秒
    
    # LLM生成时间（qwen3:14b，假设平均生成500 tokens）
    generation_time_per_query = 20.0  # 秒
    
    # RAGAS评估时间（3个零标注指标 + 2个Rubric指标，每个约30秒）
    ragas_time_per_query = 5 * 30.0  # 秒
    
    # 单次查询总时间
    time_per_query = retrieval_time_per_query + generation_time_per_query + ragas_time_per_query
    
    # 总时间（加上10%的余量）
    total_time = time_per_query * n_questions * 1.1
    
    return {
        "retrieval_per_query": retrieval_time_per_query,
        "generation_per_query": generation_time_per_query,
        "ragas_eval_per_query": ragas_time_per_query,
        "total_per_query": time_per_query,
        "total_estimated_seconds": total_time,
        "total_estimated_minutes": total_time / 60,
        "total_estimated_hours": total_time / 3600,
    }


class RAGASEvaluator:
    """
    RAGAS评估器 - 使用本地Ollama进行评估
    """
    
    def __init__(
        self,
        db_path: str,
        model_name: str = "qwen3:14b",
        embed_model: str = "nomic-embed-text:latest",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        output_dir: str = None,
    ):
        """
        初始化评估器
        """
        self.db_path = db_path
        self.model_name = model_name
        self.embed_model = embed_model
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.output_dir = output_dir
        self.query_logs: List[QueryLog] = []
        
        # 初始化RAG系统
        print(f"正在初始化RAG系统，数据库路径: {db_path}")
        start_time = time.time()
        self.rag = OllamaRAG(
            persist_dir=db_path,
            model_name=model_name,
            embed_model=embed_model,
            ollama_base_url=ollama_base_url
        )
        init_duration = time.time() - start_time
        self.db_doc_count = self.rag.collection.count()
        print(f"RAG系统初始化完成，包含 {self.db_doc_count} 个文档片段")
        print(f"初始化耗时: {init_duration:.2f}秒")
        
        # 初始化RAGAS的LLM和Embedding包装器（如果ragas可用）
        if RAGAS_AVAILABLE:
            self._init_ragas_wrappers()
        
    def _init_ragas_wrappers(self):
        """初始化ragas所需的LLM和Embedding包装器"""
        print("正在初始化RAGAS评估器...")
        
        # 使用Langchain的Ollama集成，添加超时设置
        self.llm = Ollama(
            model=self.model_name,
            base_url=self.ollama_base_url,
            temperature=self.temperature,
            timeout=600,  # 增加超时时间以防止长文本生成或排队中断
        )
        self.ragas_llm = LangchainLLMWrapper(self.llm)
        
        self.embeddings = OllamaEmbeddings(
            model=self.embed_model,
            base_url=self.ollama_base_url,
        )
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        # 定义指标并为每个指标设置LLM和Embedding
        self.unlabeled_metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextUtilization(),
        ]

        # 定义 Rubric 评分标准
        cpac_plan_rubrics = {
            "score1_description": "回答完全无效。未提供任何CPAC参数配置建议，或建议明显错误、与问题无关。",
            "score2_description": "回答质量较差。提供的CPAC配置建议不完整，缺少关键参数，或配置逻辑存在明显错误。",
            "score3_description": "回答质量一般。提供了基本的CPAC配置建议，但细节不够充分，某些参数选择理由不够清晰。",
            "score4_description": "回答质量良好。提供了完整的CPAC配置建议，参数选择合理，有适当的解释说明。",
            "score5_description": "回答质量优秀。提供了完整、准确、详细的CPAC配置建议，参数选择专业且有充分依据，解释清晰全面，可直接用于实际配置。",
        }

        evidence_uncertainty_rubrics = {
            "score1_description": "没有引用任何检索到的上下文证据，或声称的信息完全无法从上下文中验证。",
            "score2_description": "极少引用上下文证据，大部分声称缺乏依据。未表达任何不确定性。",
            "score3_description": "部分引用了上下文证据，但引用不够充分或准确。对不确定的信息缺乏适当提示。",
            "score4_description": "较好地引用了上下文证据来支持主要观点。对不确定的信息有一定的提示。",
            "score5_description": "充分、准确地引用上下文证据支持所有关键观点。明确表达了对不确定信息的谨慎态度，区分了确定的事实和推测性内容。",
        }

        self.rubric_metrics = [
            DomainSpecificRubrics(
                name="cpac_plan_quality",
                rubrics=cpac_plan_rubrics,
            ),
            DomainSpecificRubrics(
                name="evidence_uncertainty",
                rubrics=evidence_uncertainty_rubrics,
            ),
        ]

        self.all_metrics = self.unlabeled_metrics + self.rubric_metrics

        # 为每个指标设置LLM和Embedding
        for metric in self.all_metrics:
            metric.llm = self.ragas_llm
            if hasattr(metric, 'embeddings') and metric.embeddings is None:
                metric.embeddings = self.ragas_embeddings
            elif hasattr(metric, 'embeddings'):
                metric.embeddings = self.ragas_embeddings

        print("RAGAS评估器初始化完成")
        print(f"评估指标: 3个零标注指标 + {len(self.rubric_metrics)}个Rubric指标 = {len(self.all_metrics)}个指标")
    
    def query_rag(self, question: str, n_results: int = 5, use_hybrid_search: bool = True) -> Dict:
        """
        执行RAG查询
        """
        result = self.rag.query(
            question=question,
            n_results=n_results,
            use_hybrid_search=use_hybrid_search
        )
        
        return {
            "answer": result["answer"],
            "contexts": result["relevant_chunks"],
            "sources": result["chunk_sources"]
        }
    
    def generate_samples(
        self,
        questions_file: str,
        n_results: int = 5,
        use_hybrid_search: bool = True,
    ) -> List[EvalSample]:
        """
        批量生成评估样本
        """
        # 读取问题
        questions = []
        with open(questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    questions.append(line)
        
        print(f"\n读取到 {len(questions)} 个问题，开始批量查询...")
        print(f"预计总耗时: 约 {len(questions) * 1.5:.0f}-{len(questions) * 2.5:.0f} 分钟")
        print("="*60)
        
        samples = []
        for idx, question in enumerate(questions):
            run_id = f"query_{idx:04d}"
            print(f"\n[{idx+1}/{len(questions)}] 处理问题...")
            print(f"    问题: {question[:100]}...")
            
            # 创建查询日志
            query_log = QueryLog(
                query_idx=idx,
                start_time=datetime.now().isoformat(),
                question=question,
            )
            
            start_time = time.time()
            
            try:
                # 执行RAG查询
                result = self.query_rag(
                    question=question,
                    n_results=n_results,
                    use_hybrid_search=use_hybrid_search
                )
                
                duration = time.time() - start_time
                
                # 更新日志
                query_log.end_time = datetime.now().isoformat()
                query_log.duration_seconds = duration
                query_log.retrieved_contexts = result["contexts"]
                query_log.context_sources = result["sources"]
                query_log.response = result["answer"]
                
                # 构建样本
                sample = EvalSample(
                    run_id=run_id,
                    user_input=question,
                    retrieved_contexts=result["contexts"],
                    response=result["answer"],
                    extra={
                        "sources": result["sources"],
                        "n_results": n_results,
                        "use_hybrid_search": use_hybrid_search,
                        "query_duration_seconds": duration,
                    }
                )
                samples.append(sample)
                
                print(f"    ✓ 成功 | 检索到 {len(result['contexts'])} 个文档片段 | 耗时: {duration:.2f}秒")
                
            except Exception as e:
                duration = time.time() - start_time
                error_msg = str(e)
                print(f"    ✗ 失败 | 错误: {error_msg} | 耗时: {duration:.2f}秒")
                
                # 更新日志
                query_log.end_time = datetime.now().isoformat()
                query_log.duration_seconds = duration
                query_log.error = error_msg
                
                # 记录失败的样本
                sample = EvalSample(
                    run_id=run_id,
                    user_input=question,
                    retrieved_contexts=[],
                    response=f"ERROR: {error_msg}",
                    extra={
                        "error": error_msg,
                        "query_duration_seconds": duration,
                    }
                )
                samples.append(sample)
            
            # 保存日志
            self.query_logs.append(query_log)
            
            # 每5个查询保存一次中间日志
            if (idx + 1) % 5 == 0:
                self._save_query_logs()
        
        # 最后保存一次日志
        self._save_query_logs()
        
        return samples
    
    def _save_query_logs(self):
        """保存查询日志到文件"""
        if not self.output_dir or not self.query_logs:
            return
        
        log_path = Path(self.output_dir) / "query_logs.jsonl"
        with open(log_path, 'w', encoding='utf-8') as f:
            for log in self.query_logs:
                f.write(json.dumps(log.to_dict(), ensure_ascii=False) + '\n')
    
    def evaluate_unlabeled(
        self,
        samples: List[EvalSample],
    ) -> List[EvalResult]:
        """
        执行零标注指标评估
        """
        if not RAGAS_AVAILABLE:
            print("错误: ragas未安装，无法进行评估")
            return []
        
        print(f"\n开始零标注指标评估（{len(samples)} 个样本）...")
        print("评估指标: Faithfulness, AnswerRelevancy, ContextUtilization")
        print("Rubric指标: CPAC-Plan-Quality (1-5), Evidence-Uncertainty (1-5)")
        print(f"预计耗时: 约 {len(samples) * 2.0:.0f}-{len(samples) * 3.0:.0f} 分钟")
        print("="*60)
        
        # 过滤掉有错误的样本
        valid_samples = [s for s in samples if not s.response.startswith("ERROR:")]
        print(f"有效样本: {len(valid_samples)}/{len(samples)}")
        
        # 构建RAGAS数据集
        ragas_samples = []
        for sample in valid_samples:
            ragas_sample = SingleTurnSample(
                user_input=sample.user_input,
                retrieved_contexts=sample.retrieved_contexts,
                response=sample.response,
            )
            ragas_samples.append(ragas_sample)
        
        dataset = EvaluationDataset(samples=ragas_samples)
        
        # 执行评估
        start_time = time.time()
        try:
            # 配置评估超时和降低并发(防止本地Ollama排队导致Ragas内部TimeoutError抛出NaN)
            run_config = RunConfig(timeout=600, max_workers=1)
            results = evaluate(
                dataset=dataset,
                metrics=self.all_metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                run_config=run_config,
                raise_exceptions=False,
            )
            eval_duration = time.time() - start_time
            print(f"评估完成，耗时: {eval_duration:.2f}秒 ({eval_duration/60:.2f}分钟)")
            
            # 转换结果为EvalResult列表（适配不同版本的ragas API）
            eval_results = []
            try:
                # 检测ragas版本并适配结果格式
                if RAGAS_VERSION >= (0, 2):
                    # 新版ragas (>=0.2): 使用to_pandas()获取DataFrame
                    df = results.to_pandas()
                    for idx, sample in enumerate(valid_samples):
                        metrics = {
                            # 零标注指标 (0-1)
                            "faithfulness": float(df.loc[idx, "faithfulness"]) if "faithfulness" in df.columns else 0.0,
                            "answer_relevancy": float(df.loc[idx, "answer_relevancy"]) if "answer_relevancy" in df.columns else 0.0,
                            "context_utilization": float(df.loc[idx, "context_utilization"]) if "context_utilization" in df.columns else 0.0,
                            # Rubric指标 (1-5)
                            "cpac_plan_quality": float(df.loc[idx, "cpac_plan_quality"]) if "cpac_plan_quality" in df.columns else 0.0,
                            "evidence_uncertainty": float(df.loc[idx, "evidence_uncertainty"]) if "evidence_uncertainty" in df.columns else 0.0,
                        }
                        eval_results.append(EvalResult(
                            run_id=sample.run_id,
                            user_input=sample.user_input,
                            metrics=metrics
                        ))
                else:
                    # 旧版ragas (<0.2): 直接使用字典访问
                    for idx, sample in enumerate(valid_samples):
                        metrics = {
                            # 零标注指标 (0-1)
                            "faithfulness": results.get("faithfulness", [0])[idx] if "faithfulness" in results else 0,
                            "answer_relevancy": results.get("answer_relevancy", [0])[idx] if "answer_relevancy" in results else 0,
                            "context_utilization": results.get("context_utilization", [0])[idx] if "context_utilization" in results else 0,
                            # Rubric指标 (1-5)
                            "cpac_plan_quality": results.get("cpac_plan_quality", [0])[idx] if "cpac_plan_quality" in results else 0,
                            "evidence_uncertainty": results.get("evidence_uncertainty", [0])[idx] if "evidence_uncertainty" in results else 0,
                        }
                        eval_results.append(EvalResult(
                            run_id=sample.run_id,
                            user_input=sample.user_input,
                            metrics=metrics
                        ))
            except Exception as extract_error:
                print(f"结果提取出错: {extract_error}")
                print(f"ragas版本: {RAGAS_VERSION}, 结果类型: {type(results)}")
                # 返回空指标的结果
                for sample in valid_samples:
                    eval_results.append(EvalResult(
                        run_id=sample.run_id,
                        user_input=sample.user_input,
                        metrics={
                            "faithfulness": 0,
                            "answer_relevancy": 0,
                            "context_utilization": 0,
                            "cpac_plan_quality": 0,
                            "evidence_uncertainty": 0,
                        }
                    ))
            
            return eval_results
            
        except Exception as e:
            print(f"评估过程出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def save_samples_jsonl(self, samples: List[EvalSample], output_path: str):
        """保存样本为JSONL格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
        print(f"样本已保存至: {output_path}")
    
    def save_results_overall(
        self,
        results: List[EvalResult],
        output_path: str,
    ):
        """保存总体统计结果（JSON格式）"""
        if not NUMPY_AVAILABLE:
            print("错误: numpy未安装，无法计算统计结果")
            return
        
        # 收集所有指标
        all_metrics = {}
        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # 计算统计量
        stats = {}
        for metric_name, values in all_metrics.items():
            arr = np.array(values)
            stats[metric_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(values),
            }
        
        # 添加元数据
        output = {
            "evaluation_time": datetime.now().isoformat(),
            "model_name": self.model_name,
            "embed_model": self.embed_model,
            "db_path": self.db_path,
            "db_doc_count": self.db_doc_count,
            "total_samples": len(results),
            "metrics_summary": stats,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"总体结果已保存至: {output_path}")
        
        # 打印到控制台
        print("\n" + "="*60)
        print("评估结果汇总")
        print("="*60)
        for metric_name, metric_stats in stats.items():
            print(f"\n{metric_name}:")
            print(f"  均值: {metric_stats['mean']:.4f}")
            print(f"  中位数: {metric_stats['median']:.4f}")
            print(f"  标准差: {metric_stats['std']:.4f}")
            print(f"  P10-P90: [{metric_stats['p10']:.4f}, {metric_stats['p90']:.4f}]")
    
    def save_results_per_sample(
        self,
        results: List[EvalResult],
        output_path: str,
    ):
        """保存每个样本的详细结果（CSV格式）"""
        if not results:
            return
        
        # 收集所有指标名称
        all_metric_names = set()
        for result in results:
            all_metric_names.update(result.metrics.keys())
        all_metric_names = sorted(list(all_metric_names))
        
        # 写入CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 表头
            headers = ["run_id", "user_input"] + all_metric_names
            writer.writerow(headers)
            
            # 数据行
            for result in results:
                row = [
                    result.run_id,
                    result.user_input[:100] + "..." if len(result.user_input) > 100 else result.user_input
                ]
                for metric_name in all_metric_names:
                    row.append(result.metrics.get(metric_name, ''))
                writer.writerow(row)
        
        print(f"详细结果已保存至: {output_path}")


def evaluate_dataset(
    evaluator: RAGASEvaluator,
    dataset_name: str,
    queries_file: str,
    output_dir: str,
    config: Dict,
):
    """评估单个数据集"""
    print("\n" + "="*70)
    print(f"开始评估数据集: {dataset_name}")
    print(f"问题文件: {queries_file}")
    print(f"输出目录: {output_dir}")
    print("="*70)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    evaluator.output_dir = output_dir
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成样本（执行RAG查询）
    samples = evaluator.generate_samples(
        questions_file=queries_file,
        n_results=config["n_results"],
        use_hybrid_search=config["use_hybrid_search"],
    )
    
    # 保存样本
    samples_path = os.path.join(output_dir, f"eval_samples_{dataset_name}_{timestamp}.jsonl")
    evaluator.save_samples_jsonl(samples, samples_path)
    
    # 如果要求跳过评估，到此结束
    if config["skip_eval"]:
        print(f"\n已跳过评估，样本已保存至: {samples_path}")
        return
    
    # 执行评估
    all_results = []

    # 零标注评估 + Rubric评估（总是执行）
    unlabeled_results = evaluator.evaluate_unlabeled(samples)
    all_results.extend(unlabeled_results)
    
    # 保存结果
    if all_results:
        # 总体结果
        overall_path = os.path.join(output_dir, f"results_overall_{dataset_name}_{timestamp}.json")
        evaluator.save_results_overall(all_results, overall_path)
        
        # 详细结果
        detail_path = os.path.join(output_dir, f"results_per_sample_{dataset_name}_{timestamp}.csv")
        evaluator.save_results_per_sample(all_results, detail_path)
        
        print("\n" + "="*70)
        print(f"数据集 {dataset_name} 评估完成！")
        print("输出文件:")
        print(f"  - 样本数据: {samples_path}")
        print(f"  - 查询日志: {os.path.join(output_dir, 'query_logs.jsonl')}")
        print(f"  - 总体统计: {overall_path}")
        print(f"  - 详细结果: {detail_path}")
        print(f"{'='*70}")
    else:
        print(f"\n警告: 数据集 {dataset_name} 没有生成评估结果")


def main():
    # 获取配置
    config = CONFIG

    # 检查ragas是否可用
    if not RAGAS_AVAILABLE and not config["skip_eval"]:
        print("错误: 未安装ragas，无法进行评估。请运行:")
        print("  pip install ragas datasets pandas langchain langchain-community")
        return 1

    # 根据配置选择要评估的数据集
    if config["dataset"] == "both":
        datasets_to_eval = ["adhd200", "ds002748"]
    else:
        datasets_to_eval = [config["dataset"]]

    # 创建基础输出目录
    os.makedirs(config["output_base_dir"], exist_ok=True)

    # 遍历每个数据库配置进行评估
    for db_config in config["databases"]:
        db_name = db_config["name"]
        db_path = db_config["db_path"]
        embed_model = db_config["embed_model"]

        print("\n" + "="*70)
        print(f"开始评估数据库: {db_name}")
        print("="*70)
        print(f"数据库路径: {db_path}")
        print(f"评估模型: {config['model_name']}")
        print(f"嵌入模型: {embed_model}")
        print(f"Ollama地址: {config['ollama_base_url']}")
        print("="*70)

        # 初始化评估器（每个数据库单独初始化，使用对应的嵌入模型）
        try:
            evaluator = RAGASEvaluator(
                db_path=db_path,
                model_name=config["model_name"],
                embed_model=embed_model,
                ollama_base_url=config["ollama_base_url"],
                temperature=config["temperature"],
            )
        except Exception as e:
            print(f"错误: 初始化数据库 {db_name} 失败: {e}")
            continue

        # 估算总时间
        total_questions = 0
        for ds_name in datasets_to_eval:
            queries_file = config["queries_files"].get(ds_name)
            if queries_file and os.path.exists(queries_file):
                with open(queries_file, 'r', encoding='utf-8') as f:
                    n_questions = sum(1 for line in f if line.strip() and not line.startswith('#'))
                    total_questions += n_questions

        time_estimate = estimate_time(
            evaluator.db_doc_count,
            total_questions,
            config["use_hybrid_search"]
        )

        print("\n" + "="*70)
        print("时间估算")
        print("="*70)
        print(f"数据库文档数: {evaluator.db_doc_count}")
        print(f"待评估问题数: {total_questions}")
        print(f"检索方式: {'混合搜索' if config['use_hybrid_search'] else '纯向量搜索'}")
        print("-"*70)
        print(f"单次检索耗时: ~{time_estimate['retrieval_per_query']:.1f}秒")
        print(f"单次生成耗时: ~{time_estimate['generation_per_query']:.1f}秒")
        print(f"单次RAGAS评估: ~{time_estimate['ragas_eval_per_query']:.1f}秒")
        print(f"单次查询总计: ~{time_estimate['total_per_query']:.1f}秒")
        print("-"*70)
        print(f"预估总耗时: ~{time_estimate['total_estimated_minutes']:.1f}分钟 ({time_estimate['total_estimated_hours']:.2f}小时)")
        print("="*70)
        print("\n注意: 以上时间为粗略估算，实际时间受硬件性能、模型响应速度影响")
        print("开始执行评估...\n")

        # 评估选定的数据集
        for dataset_name in datasets_to_eval:
            queries_file = config["queries_files"].get(dataset_name)

            if not queries_file:
                print(f"错误: 未知的数据集 '{dataset_name}'")
                continue

            if not os.path.exists(queries_file):
                print(f"错误: 问题文件不存在: {queries_file}")
                continue

            # 为每个数据集和数据库组合创建子目录
            dataset_output_dir = os.path.join(config["output_base_dir"], dataset_name, db_name)

            # 评估数据集
            evaluate_dataset(
                evaluator=evaluator,
                dataset_name=dataset_name,
                queries_file=queries_file,
                output_dir=dataset_output_dir,
                config=config,
            )

    print("\n" + "="*70)
    print("所有评估任务完成！")
    print(f"结果保存位置: {config['output_base_dir']}")
    print("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
