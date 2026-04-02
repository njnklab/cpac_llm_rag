#!/usr/bin/env python3
"""
检索策略对比实验 - 计算脚本 (Part 1)
对比3种检索策略：vector_only, bm25_only, fusion_0.7_0.3

此脚本只执行检索、生成和评估，不生成图表
图表由单独的 plot_results.py 脚本生成

配置参数请在 CONFIG 字典中修改
"""

import os
import sys
import json
import csv
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ==================== 用户配置区域 ====================

CONFIG = {
    # 数据库配置
    "database": {
        "name": "qwen3",
        "db_path": "/home/a001/zhangyan/LitQuery/serpapi_db_qwen3",
        "embed_model": "qwen3-embedding:latest",
    },

    # 模型配置
    "model_name": "qwen3:14b",
    "ollama_base_url": "http://localhost:11434",
    "temperature": 0.0,

    # Query配置
    "queries_per_dataset": 3,
    "queries_files": {
        "adhd200": "/home/a001/zhangyan/LitQuery/queries/adhd200_queries.txt",
        "ds002748": "/home/a001/zhangyan/LitQuery/queries/ds002748_queries.txt",
    },

    # 检索配置
    "n_results": 5,

    # 输出目录
    "output_dir": "/home/a001/zhangyan/LitQuery/检索方案评估/result",
    
    # 中间结果保存文件名（用于传递给画图脚本）
    "intermediate_results": "all_results_intermediate.json",
}

# ==================== 导入依赖 ====================

try:
    from ragas import evaluate
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
except ImportError as e:
    print(f"RAGAS导入失败: {e}")
    print("请安装: pip install ragas datasets pandas langchain langchain-community")
    RAGAS_AVAILABLE = False
    sys.exit(1)

# 将上级目录加入模块搜索路径，以便导入 chromadb_rag
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chromadb_rag import OllamaRAG


@dataclass
class EvalSample:
    run_id: str
    dataset: str
    query_idx: int
    user_input: str
    retrieved_contexts: List[str]
    response: str
    retrieval_mode: str
    extra: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EvalSample':
        return cls(**data)


@dataclass
class EvalResult:
    run_id: str
    dataset: str
    query_idx: int
    user_input: str
    retrieval_mode: str
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EvalResult':
        return cls(**data)


class RetrievalStrategyEvaluator:
    def __init__(
        self,
        db_path: str,
        model_name: str = "qwen3:14b",
        embed_model: str = "qwen3-embedding:latest",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        self.db_path = db_path
        self.model_name = model_name
        self.embed_model = embed_model
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        
        print(f"\n{'='*60}")
        print(f"初始化RAG系统")
        print(f"  数据库: {db_path}")
        print(f"  生成模型: {model_name}")
        print(f"  嵌入模型: {embed_model}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        self.rag = OllamaRAG(
            persist_dir=db_path,
            model_name=model_name,
            embed_model=embed_model,
            ollama_base_url=ollama_base_url
        )
        init_duration = time.time() - start_time
        self.db_doc_count = self.rag.collection.count()
        print(f"RAG初始化完成 | 文档数: {self.db_doc_count} | 耗时: {init_duration:.2f}秒\n")
        
        self._init_ragas()
    
    def _init_ragas(self):
        print("初始化RAGAS评估器...")
        
        self.llm = Ollama(
            model=self.model_name,
            base_url=self.ollama_base_url,
            temperature=self.temperature,
            timeout=600,
        )
        self.ragas_llm = LangchainLLMWrapper(self.llm)
        
        self.embeddings = OllamaEmbeddings(
            model=self.embed_model,
            base_url=self.ollama_base_url,
        )
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        self.unlabeled_metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextUtilization(),
        ]

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
            DomainSpecificRubrics(name="cpac_plan_quality", rubrics=cpac_plan_rubrics),
            DomainSpecificRubrics(name="evidence_uncertainty", rubrics=evidence_uncertainty_rubrics),
        ]

        self.all_metrics = self.unlabeled_metrics + self.rubric_metrics

        for metric in self.all_metrics:
            metric.llm = self.ragas_llm
            if hasattr(metric, 'embeddings'):
                metric.embeddings = self.ragas_embeddings
        
        print(f"RAGAS初始化完成 | 指标数: {len(self.all_metrics)}\n")
    
    def vector_only_search(self, query: str, n_results: int = 5) -> Dict:
        query_embedding = self.rag.embeddings.embed_query(query)
        results = self.rag.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return self._format_results(results)
    
    def bm25_only_search(self, query: str, n_results: int = 5) -> Dict:
        results = self.rag.bm25_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return self._format_results(results)
    
    def fusion_search(self, query: str, n_results: int = 5, vector_weight: float = 0.7) -> Dict:
        return self.rag.hybrid_search(query, n_results=n_results, vector_weight=vector_weight)
    
    def _format_results(self, results: Dict) -> Dict:
        if not results["ids"] or not results["ids"][0]:
            return {"documents": [], "metadatas": [], "ids": []}
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }
    
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        prompt = (
            "You are a knowledgeable assistant that combines retrieved information with your own expertise. "
            "Answer the question thoroughly while following these guidelines:\n\n"
            "1. Use the retrieved documents as primary sources and supplement with your own knowledge when appropriate.\n"
            "2. When referencing information from the provided documents, cite your sources using [Doc X] format.\n"
            "3. Provide comprehensive yet concise answers with clear logical structure.\n"
            "4. Feel free to make reasonable inferences beyond the documents when helpful, but clearly distinguish between document information and your own additions.\n"
            "5. If the retrieved documents contain minimal relevant information, acknowledge this and rely more on your expertise.\n\n"
            "Retrieved Documents:\n"
            + ('-' * 40) + "\n"
            + '\n'.join([f"[Doc {i+1}] " + chunk for i, chunk in enumerate(contexts)]) + "\n"
            + ('-' * 40) + "\n\n"
            "Question: " + question + "\n\n"
            "Answer:"
        )
        
        try:
            import requests
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name, 
                    "prompt": prompt, 
                    "stream": False, 
                    "options": {"temperature": self.temperature}
                }
            )
            response.raise_for_status()
            return response.json().get("response", "未返回答案")
        except Exception as e:
            return f"生成错误: {e}"
    
    def run_strategy(
        self,
        queries: List[Tuple[str, str, int]],
        mode: str,
        n_results: int = 5
    ) -> List[EvalSample]:
        print(f"\n{'='*60}")
        print(f"运行检索策略: {mode}")
        print(f"{'='*60}\n")
        
        samples = []
        total = len(queries)
        
        for idx, (dataset, question, q_idx) in enumerate(queries):
            run_id = f"{mode}_{dataset}_q{q_idx:02d}"
            print(f"[{idx+1}/{total}] {run_id}")
            print(f"  Query: {question[:80]}...")
            
            start_time = time.time()
            
            if mode == "vector_only":
                results = self.vector_only_search(question, n_results)
                vector_weight = 1.0
            elif mode == "bm25_only":
                results = self.bm25_only_search(question, n_results)
                vector_weight = 0.0
            elif mode == "fusion_0.7_0.3":
                results = self.fusion_search(question, n_results, vector_weight=0.7)
                vector_weight = 0.7
            else:
                raise ValueError(f"未知的检索模式: {mode}")
            
            retrieval_time = time.time() - start_time
            contexts = results["documents"]
            
            print(f"  检索完成 | 文档数: {len(contexts)} | 耗时: {retrieval_time:.2f}s")
            
            gen_start = time.time()
            answer = self.generate_answer(question, contexts)
            gen_time = time.time() - gen_start
            
            print(f"  生成完成 | 耗时: {gen_time:.2f}s")
            
            sample = EvalSample(
                run_id=run_id,
                dataset=dataset,
                query_idx=q_idx,
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                retrieval_mode=mode,
                extra={
                    "retrieval_time": retrieval_time,
                    "generation_time": gen_time,
                    "vector_weight": vector_weight,
                    "n_results": n_results,
                    "sources": results.get("metadatas", [])
                }
            )
            samples.append(sample)
            
            print(f"  样本创建完成 | 总耗时: {retrieval_time + gen_time:.2f}s\n")
        
        return samples
    
    def evaluate_samples(self, samples: List[EvalSample]) -> List[EvalResult]:
        print(f"\n{'='*60}")
        print(f"开始RAGAS评估 | 样本数: {len(samples)}")
        print(f"{'='*60}\n")
        
        valid_samples = [s for s in samples if not s.response.startswith("生成错误")]
        print(f"有效样本: {len(valid_samples)}/{len(samples)}\n")
        
        ragas_samples = []
        for sample in valid_samples:
            ragas_samples.append(SingleTurnSample(
                user_input=sample.user_input,
                retrieved_contexts=sample.retrieved_contexts,
                response=sample.response,
            ))
        
        dataset = EvaluationDataset(samples=ragas_samples)
        
        start_time = time.time()
        try:
            run_config = RunConfig(timeout=600, max_workers=1)
            results = evaluate(
                dataset=dataset,
                metrics=self.all_metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                run_config=run_config,
                raise_exceptions=False,
            )
            eval_time = time.time() - start_time
            print(f"评估完成 | 耗时: {eval_time:.2f}s ({eval_time/60:.1f}分钟)\n")
            
            eval_results = []
            df = results.to_pandas()
            
            for idx, sample in enumerate(valid_samples):
                metrics = {
                    "faithfulness": float(df.loc[idx, "faithfulness"]) if "faithfulness" in df.columns else 0.0,
                    "answer_relevancy": float(df.loc[idx, "answer_relevancy"]) if "answer_relevancy" in df.columns else 0.0,
                    "context_utilization": float(df.loc[idx, "context_utilization"]) if "context_utilization" in df.columns else 0.0,
                    "cpac_plan_quality": float(df.loc[idx, "cpac_plan_quality"]) if "cpac_plan_quality" in df.columns else 0.0,
                    "evidence_uncertainty": float(df.loc[idx, "evidence_uncertainty"]) if "evidence_uncertainty" in df.columns else 0.0,
                }
                eval_results.append(EvalResult(
                    run_id=sample.run_id,
                    dataset=sample.dataset,
                    query_idx=sample.query_idx,
                    user_input=sample.user_input,
                    retrieval_mode=sample.retrieval_mode,
                    metrics=metrics
                ))
            
            return eval_results
            
        except Exception as e:
            print(f"评估失败: {e}")
            import traceback
            traceback.print_exc()
            return []


def load_queries(config: Dict) -> List[Tuple[str, str, int]]:
    queries = []
    n_per_dataset = config["queries_per_dataset"]
    
    for dataset_name, file_path in config["queries_files"].items():
        print(f"加载 {dataset_name} queries...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        selected = lines[:n_per_dataset]
        print(f"  总条数: {len(lines)} | 选用: {n_per_dataset}")
        
        for idx, query_text in enumerate(selected):
            queries.append((dataset_name, query_text, idx))
    
    print(f"\n共加载 {len(queries)} 条query\n")
    return queries


def save_samples_jsonl(samples: List[EvalSample], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
    print(f"样本已保存: {output_path}")


def save_results_per_sample(results: List[EvalResult], output_path: str):
    if not results:
        return
    
    metric_names = ["faithfulness", "answer_relevancy", "context_utilization", 
                    "cpac_plan_quality", "evidence_uncertainty"]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "dataset", "query_idx", "user_input"] + metric_names)
        
        for result in results:
            row = [
                result.run_id,
                result.dataset,
                result.query_idx,
                result.user_input[:100] + "..." if len(result.user_input) > 100 else result.user_input
            ]
            for metric in metric_names:
                row.append(result.metrics.get(metric, ''))
            writer.writerow(row)
    
    print(f"详细结果已保存: {output_path}")


def save_results_overall(results: List[EvalResult], output_path: str, mode: str):
    if not results:
        return
    
    metric_names = ["faithfulness", "answer_relevancy", "context_utilization", 
                    "cpac_plan_quality", "evidence_uncertainty"]
    
    stats = {}
    for metric in metric_names:
        values = [r.metrics.get(metric, 0) for r in results if metric in r.metrics]
        if values:
            import numpy as np
            arr = np.array(values)
            stats[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(values),
            }
    
    output = {
        "retrieval_mode": mode,
        "evaluation_time": datetime.now().isoformat(),
        "total_samples": len(results),
        "metrics_summary": stats
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"总体结果已保存: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"策略 {mode} 评估结果汇总")
    print(f"{'='*60}")
    for metric, metric_stats in stats.items():
        print(f"\n{metric}:")
        print(f"  均值: {metric_stats['mean']:.4f}")
        print(f"  中位数: {metric_stats['median']:.4f}")
        print(f"  标准差: {metric_stats['std']:.4f}")
        print(f"  P10-P90: [{metric_stats['p10']:.4f}, {metric_stats['p90']:.4f}]")


def save_intermediate_results(all_results: Dict[str, List[EvalResult]], output_path: str):
    """保存中间结果JSON，供画图脚本使用"""
    serializable_results = {}
    for mode, results in all_results.items():
        serializable_results[mode] = [r.to_dict() for r in results]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n中间结果已保存: {output_path}")
    print("此文件将被 plot_results.py 读取用于生成图表")


def main():
    print(f"\n{'='*70}")
    print("  检索策略对比实验 - 计算脚本")
    print(f"{'='*70}\n")
    
    config = CONFIG
    
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}\n")
    
    queries = load_queries(config)
    
    evaluator = RetrievalStrategyEvaluator(
        db_path=config["database"]["db_path"],
        model_name=config["model_name"],
        embed_model=config["database"]["embed_model"],
        ollama_base_url=config["ollama_base_url"],
        temperature=config["temperature"],
    )
    
    strategies = ["vector_only", "bm25_only", "fusion_0.7_0.3"]
    all_results = {}
    
    for mode in strategies:
        print(f"\n{'#'*70}")
        print(f"# 开始策略: {mode}")
        print(f"{'#'*70}")
        
        samples = evaluator.run_strategy(
            queries=queries,
            mode=mode,
            n_results=config["n_results"]
        )
        
        samples_path = os.path.join(output_dir, f"samples_{mode}.jsonl")
        save_samples_jsonl(samples, samples_path)
        
        results = evaluator.evaluate_samples(samples)
        all_results[mode] = results
        
        per_sample_path = os.path.join(output_dir, f"ragas_{mode}_per_sample.csv")
        save_results_per_sample(results, per_sample_path)
        
        overall_path = os.path.join(output_dir, f"ragas_{mode}_overall.json")
        save_results_overall(results, overall_path, mode)
        
        print(f"\n{'='*60}")
        print(f"策略 {mode} 完成")
        print(f"{'='*60}\n")
    
    # 保存中间结果供画图脚本使用
    intermediate_path = os.path.join(output_dir, config["intermediate_results"])
    save_intermediate_results(all_results, intermediate_path)
    
    print(f"\n{'='*70}")
    print("  计算部分完成！")
    print(f"{'='*70}\n")
    print("输出文件:")
    for mode in strategies:
        print(f"  - samples_{mode}.jsonl")
        print(f"  - ragas_{mode}_per_sample.csv")
        print(f"  - ragas_{mode}_overall.json")
    print(f"\n  - {config['intermediate_results']} (画图脚本输入)")
    print(f"\n下一步: 运行 python plot_results.py 生成图表")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
