#!/usr/bin/env python3
"""
检索质量评估脚本 - 使用本地Ollama qwen3:14b评估四个嵌入模型的检索结果
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass

sys.path.insert(0, '/home/a001/zhangyan/LitQuery')


# ==================== 配置区域 ====================

CONFIG = {
    # 输入文件路径（上一步的测试结果）
    "input_file": "/home/a001/zhangyan/LitQuery/检索测试/retrieval_test_20260305_003556.json",
    
    # Ollama配置
    "ollama_base_url": "http://localhost:11434",
    "model_name": "qwen3:14b",
    "temperature": 0.3,
    
    # 输出配置
    "output_dir": "/home/a001/zhangyan/LitQuery/检索测试",
}

# ==================== 配置区域结束 ====================


@dataclass
class EvaluationResult:
    """单个数据库的评估结果"""
    db_name: str
    relevance_score: float
    completeness_score: float
    actionability_score: float
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str
    reasoning: str


def call_ollama(prompt: str, model: str = None, temperature: float = None) -> str:
    """调用本地Ollama API"""
    if model is None:
        model = CONFIG["model_name"]
    if temperature is None:
        temperature = CONFIG["temperature"]
    
    url = f"{CONFIG['ollama_base_url']}/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        }
    }
    
    try:
        response = requests.post(url, json=data, timeout=300)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"ERROR: {str(e)}"


def evaluate_single_db(db_name: str, query: str, sources: List[Dict]) -> EvaluationResult:
    """使用Ollama评估单个数据库的检索结果"""
    
    print(f"\n{'='*70}")
    print(f"评估数据库: {db_name}")
    print(f"{'='*70}")
    
    # 构建评估提示
    sources_text = ""
    for i, source in enumerate(sources, 1):
        doc_id = source.get("document_id", "unknown")
        page = source.get("page", "unknown")
        sources_text += f"[{i}] {doc_id} (页 {page})\n"
    
    prompt = f"""你是一位fMRI预处理专家，精通CPAC (Configurable Pipeline for the Analysis of Connectomes) 工具。

**用户查询问题**：
{query}

**检索到的文档列表**：
{sources_text}

请评估这些检索结果对回答用户查询问题的帮助程度。

请从以下四个维度评分（1-10分，10分最高）：

1. **相关性 (Relevance)**：检索到的文档与CPAC anatomical_preproc配置的相关程度
2. **完整性 (Completeness)**：这些文档是否涵盖了brain extraction、bias-field correction、spatial normalization三个关键配置点
3. **可执行性 (Actionability)**：文档内容是否能直接指导CPAC参数配置
4. **总体评分 (Overall)**：综合考虑上述因素

请用JSON格式输出你的评估结果：
{{
    "relevance_score": 评分数字,
    "completeness_score": 评分数字,
    "actionability_score": 评分数字,
    "overall_score": 评分数字,
    "strengths": ["优点1", "优点2"],
    "weaknesses": ["缺点1", "缺点2"],
    "recommendation": "对该数据库使用的建议",
    "reasoning": "简要解释评分理由"
}}

请只输出JSON，不要有任何其他文字。"""

    print("正在调用Ollama进行评估...")
    start_time = time.time()
    response = call_ollama(prompt)
    duration = time.time() - start_time
    print(f"评估完成，耗时: {duration:.2f}秒")
    
    # 解析JSON响应
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(response[start:end])
            else:
                raise ValueError("无法解析JSON")
        except Exception as e:
            print(f"  警告: JSON解析失败，使用默认值。错误: {e}")
            result = {
                "relevance_score": 0,
                "completeness_score": 0,
                "actionability_score": 0,
                "overall_score": 0,
                "strengths": ["解析失败"],
                "weaknesses": [f"JSON解析错误: {str(e)}"],
                "recommendation": "无法评估",
                "reasoning": response[:500]
            }
    
    return EvaluationResult(
        db_name=db_name,
        relevance_score=result.get("relevance_score", 0),
        completeness_score=result.get("completeness_score", 0),
        actionability_score=result.get("actionability_score", 0),
        overall_score=result.get("overall_score", 0),
        strengths=result.get("strengths", []),
        weaknesses=result.get("weaknesses", []),
        recommendation=result.get("recommendation", ""),
        reasoning=result.get("reasoning", "")
    )


def generate_comparison_report(evaluations: List[EvaluationResult], query: str) -> str:
    """生成对比报告"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = []
    report.append("="*80)
    report.append("向量数据库检索质量评估报告")
    report.append("="*80)
    report.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"评估模型: {CONFIG['model_name']}")
    report.append("")
    report.append("测试查询:")
    report.append("-"*80)
    report.append(query)
    report.append("")
    report.append("="*80)
    report.append("")
    
    # 评分对比表
    report.append("评分对比:")
    report.append("-"*80)
    report.append(f"{'数据库':<20} {'相关性':<10} {'完整性':<10} {'可执行性':<10} {'总体':<10}")
    report.append("-"*80)
    
    for ev in evaluations:
        report.append(
            f"{ev.db_name:<20} "
            f"{ev.relevance_score:>8.1f}   "
            f"{ev.completeness_score:>8.1f}   "
            f"{ev.actionability_score:>8.1f}   "
            f"{ev.overall_score:>8.1f}"
        )
    
    report.append("-"*80)
    report.append("")
    
    # 排序并找出最佳
    sorted_evals = sorted(evaluations, key=lambda x: x.overall_score, reverse=True)
    best = sorted_evals[0]
    
    report.append(f"🏆 最佳表现: {best.db_name} (总体评分: {best.overall_score}/10)")
    report.append("")
    
    # 详细评估
    report.append("详细评估:")
    report.append("="*80)
    
    for ev in sorted_evals:
        report.append(f"\n{'='*80}")
        report.append(f"数据库: {ev.db_name}")
        report.append(f"{'='*80}")
        report.append(f"\n评分:")
        report.append(f"  相关性:    {ev.relevance_score}/10")
        report.append(f"  完整性:    {ev.completeness_score}/10")
        report.append(f"  可执行性:  {ev.actionability_score}/10")
        report.append(f"  总体评分:  {ev.overall_score}/10")
        report.append(f"\n优点:")
        for strength in ev.strengths:
            report.append(f"  ✓ {strength}")
        report.append(f"\n缺点:")
        for weakness in ev.weaknesses:
            report.append(f"  ✗ {weakness}")
        report.append(f"\n评估理由:")
        report.append(f"  {ev.reasoning}")
        report.append(f"\n建议:")
        report.append(f"  {ev.recommendation}")
    
    report.append("\n" + "="*80)
    report.append("总结与建议")
    report.append("="*80)
    report.append("")
    report.append("基于Ollama qwen3:14b的评估结果，建议：")
    report.append("")
    
    for i, ev in enumerate(sorted_evals, 1):
        if ev.overall_score >= 7:
            level = "推荐使用"
        elif ev.overall_score >= 5:
            level = "可以使用"
        else:
            level = "不推荐"
        
        report.append(f"{i}. {ev.db_name}: {level} (评分: {ev.overall_score}/10)")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report), timestamp


def main():
    print("\n" + "="*80)
    print("检索质量评估 - 使用Ollama qwen3:14b")
    print("="*80)
    
    # 读取测试结果
    print(f"\n读取测试结果: {CONFIG['input_file']}")
    with open(CONFIG['input_file'], 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    query = test_data['query']
    results = test_data['results']
    
    print(f"查询问题: {query[:80]}...")
    print(f"待评估数据库: {len(results)} 个")
    
    # 评估每个数据库
    evaluations = []
    for result in results:
        db_name = result['db_name']
        sources = result.get('chunk_sources', [])
        
        evaluation = evaluate_single_db(db_name, query, sources)
        evaluations.append(evaluation)
    
    # 生成报告
    print("\n" + "="*80)
    print("生成评估报告...")
    print("="*80)
    
    report_text, timestamp = generate_comparison_report(evaluations, query)
    
    # 保存文本报告
    report_path = os.path.join(CONFIG['output_dir'], f"quality_evaluation_{timestamp}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n文本报告已保存: {report_path}")
    
    # 保存JSON格式结果
    json_result = {
        "evaluation_time": datetime.now().isoformat(),
        "model": CONFIG['model_name'],
        "query": query,
        "evaluations": [
            {
                "db_name": ev.db_name,
                "relevance_score": ev.relevance_score,
                "completeness_score": ev.completeness_score,
                "actionability_score": ev.actionability_score,
                "overall_score": ev.overall_score,
                "strengths": ev.strengths,
                "weaknesses": ev.weaknesses,
                "recommendation": ev.recommendation,
                "reasoning": ev.reasoning,
            }
            for ev in evaluations
        ]
    }
    
    json_path = os.path.join(CONFIG['output_dir'], f"quality_evaluation_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    
    print(f"JSON结果已保存: {json_path}")
    
    # 打印报告到控制台
    print("\n" + report_text)
    
    print(f"\n{'='*80}")
    print("评估完成！")
    print(f"所有结果已保存至: {CONFIG['output_dir']}")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    exit(main())