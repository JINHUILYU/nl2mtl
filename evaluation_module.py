"""
Evaluation and Ablation Study Module
评估和消融实验模块，用于系统性评估歧义检测框架的性能
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ambiguity_detector import AmbiguityDetector, ProcessOrder
from rag_module import RAGModule
import time

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """评估指标"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    consensus_rate: float
    avg_confidence: float
    processing_time: float
    
@dataclass
class AblationResult:
    """消融实验结果"""
    experiment_name: str
    condition: str
    metrics: EvaluationMetrics
    sample_results: List[Dict[str, Any]]

class EvaluationModule:
    """评估模块"""
    
    def __init__(self, output_dir: str = "data/evaluation"):
        """初始化评估模块"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_detector(self, detector: AmbiguityDetector, 
                         test_data: List[Dict[str, Any]]) -> EvaluationMetrics:
        """
        评估检测器性能
        
        Args:
            detector: 歧义检测器
            test_data: 测试数据，格式为 [{"sentence": str, "is_ambiguous": bool, "ground_truth": str}]
            
        Returns:
            评估指标
        """
        predictions = []
        ground_truths = []
        confidences = []
        consensus_reached = 0
        total_time = 0
        
        logger.info(f"开始评估，测试样本数: {len(test_data)}")
        
        for i, sample in enumerate(test_data):
            sentence = sample["sentence"]
            ground_truth = sample["is_ambiguous"]
            
            logger.info(f"评估样本 {i+1}/{len(test_data)}: {sentence[:50]}...")
            
            start_time = time.time()
            result = detector.detect_ambiguity(sentence)
            end_time = time.time()
            
            predictions.append(result.is_ambiguous)
            ground_truths.append(ground_truth)
            confidences.append(result.confidence)
            total_time += (end_time - start_time)
            
            if result.consensus_reached_at:
                consensus_reached += 1
        
        # 计算指标
        accuracy = accuracy_score(ground_truths, predictions)
        precision = precision_score(ground_truths, predictions, zero_division=0)
        recall = recall_score(ground_truths, predictions, zero_division=0)
        f1 = f1_score(ground_truths, predictions, zero_division=0)
        consensus_rate = consensus_reached / len(test_data)
        avg_confidence = np.mean(confidences)
        avg_processing_time = total_time / len(test_data)
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            consensus_rate=consensus_rate,
            avg_confidence=avg_confidence,
            processing_time=avg_processing_time
        )
    
    def run_rag_ablation(self, test_data: List[Dict[str, Any]], 
                        config_path: str = "config/detector_config.json") -> Dict[str, AblationResult]:
        """
        运行RAG消融实验
        
        Args:
            test_data: 测试数据
            config_path: 配置文件路径
            
        Returns:
            消融实验结果
        """
        logger.info("开始RAG消融实验")
        
        results = {}
        
        # 实验1: 不使用RAG
        logger.info("实验1: 不使用RAG")
        detector_no_rag = AmbiguityDetector(config_path)
        detector_no_rag.rag_enabled = False
        
        metrics_no_rag = self.evaluate_detector(detector_no_rag, test_data)
        results["no_rag"] = AblationResult(
            experiment_name="RAG Ablation",
            condition="Without RAG",
            metrics=metrics_no_rag,
            sample_results=[]
        )
        
        # 实验2: 使用RAG
        logger.info("实验2: 使用RAG")
        detector_with_rag = AmbiguityDetector(config_path)
        detector_with_rag.rag_enabled = True
        
        metrics_with_rag = self.evaluate_detector(detector_with_rag, test_data)
        results["with_rag"] = AblationResult(
            experiment_name="RAG Ablation",
            condition="With RAG",
            metrics=metrics_with_rag,
            sample_results=[]
        )
        
        # 保存结果
        self._save_ablation_results("rag_ablation", results)
        
        return results
    
    def run_order_ablation(self, test_data: List[Dict[str, Any]], 
                          config_path: str = "config/detector_config.json") -> Dict[str, AblationResult]:
        """
        运行顺序消融实验（先共识vs先投票）
        
        Args:
            test_data: 测试数据
            config_path: 配置文件路径
            
        Returns:
            消融实验结果
        """
        logger.info("开始顺序消融实验")
        
        results = {}
        
        # 实验1: 先共识后投票
        logger.info("实验1: 先共识后投票")
        detector_consensus_first = AmbiguityDetector(config_path)
        detector_consensus_first.process_order = ProcessOrder.CONSENSUS_FIRST
        
        metrics_consensus_first = self.evaluate_detector(detector_consensus_first, test_data)
        results["consensus_first"] = AblationResult(
            experiment_name="Order Ablation",
            condition="Consensus First",
            metrics=metrics_consensus_first,
            sample_results=[]
        )
        
        # 实验2: 先投票后共识
        logger.info("实验2: 先投票后共识")
        detector_voting_first = AmbiguityDetector(config_path)
        detector_voting_first.process_order = ProcessOrder.VOTING_FIRST
        
        metrics_voting_first = self.evaluate_detector(detector_voting_first, test_data)
        results["voting_first"] = AblationResult(
            experiment_name="Order Ablation",
            condition="Voting First",
            metrics=metrics_voting_first,
            sample_results=[]
        )
        
        # 保存结果
        self._save_ablation_results("order_ablation", results)
        
        return results
    
    def run_agent_diversity_ablation(self, test_data: List[Dict[str, Any]]) -> Dict[str, AblationResult]:
        """
        运行Agent多样性消融实验
        
        Args:
            test_data: 测试数据
            
        Returns:
            消融实验结果
        """
        logger.info("开始Agent多样性消融实验")
        
        results = {}
        
        # 实验1: 单一模型（所有Agent使用相同配置）
        logger.info("实验1: 单一模型配置")
        uniform_config = {
            "agents": [
                {
                    "name": "Agent_A",
                    "role": "logician",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.5,
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_URL"
                },
                {
                    "name": "Agent_B",
                    "role": "logician",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.5,
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_URL"
                },
                {
                    "name": "Agent_C",
                    "role": "logician",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.5,
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_URL"
                }
            ],
            "similarity_threshold": 0.95,
            "rag_enabled": False,
            "process_order": "consensus_first"
        }
        
        # 保存临时配置
        temp_config_path = self.output_dir / "temp_uniform_config.json"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(uniform_config, f, ensure_ascii=False, indent=2)
        
        detector_uniform = AmbiguityDetector(str(temp_config_path))
        metrics_uniform = self.evaluate_detector(detector_uniform, test_data)
        
        results["uniform_agents"] = AblationResult(
            experiment_name="Agent Diversity Ablation",
            condition="Uniform Agents",
            metrics=metrics_uniform,
            sample_results=[]
        )
        
        # 实验2: 多样化Agent（不同角色、模型、参数）
        logger.info("实验2: 多样化Agent配置")
        diverse_config = {
            "agents": [
                {
                    "name": "Agent_A",
                    "role": "logician",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.3,
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_URL"
                },
                {
                    "name": "Agent_B",
                    "role": "creative",
                    "model": "gpt-4o",
                    "temperature": 0.8,
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_URL"
                },
                {
                    "name": "Agent_C",
                    "role": "pragmatic",
                    "model": "deepseek-chat",
                    "temperature": 0.5,
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url_env": "DEEPSEEK_API_URL"
                }
            ],
            "similarity_threshold": 0.95,
            "rag_enabled": False,
            "process_order": "consensus_first"
        }
        
        # 保存临时配置
        temp_diverse_config_path = self.output_dir / "temp_diverse_config.json"
        with open(temp_diverse_config_path, 'w', encoding='utf-8') as f:
            json.dump(diverse_config, f, ensure_ascii=False, indent=2)
        
        detector_diverse = AmbiguityDetector(str(temp_diverse_config_path))
        metrics_diverse = self.evaluate_detector(detector_diverse, test_data)
        
        results["diverse_agents"] = AblationResult(
            experiment_name="Agent Diversity Ablation",
            condition="Diverse Agents",
            metrics=metrics_diverse,
            sample_results=[]
        )
        
        # 清理临时文件
        temp_config_path.unlink(missing_ok=True)
        temp_diverse_config_path.unlink(missing_ok=True)
        
        # 保存结果
        self._save_ablation_results("agent_diversity_ablation", results)
        
        return results
    
    def run_comprehensive_evaluation(self, test_data: List[Dict[str, Any]], 
                                   config_path: str = "config/detector_config.json") -> Dict[str, Any]:
        """
        运行综合评估，包括所有消融实验
        
        Args:
            test_data: 测试数据
            config_path: 配置文件路径
            
        Returns:
            综合评估结果
        """
        logger.info("开始综合评估")
        
        comprehensive_results = {
            "test_data_size": len(test_data),
            "timestamp": pd.Timestamp.now().isoformat(),
            "ablation_studies": {}
        }
        
        # RAG消融实验
        try:
            rag_results = self.run_rag_ablation(test_data, config_path)
            comprehensive_results["ablation_studies"]["rag"] = rag_results
        except Exception as e:
            logger.error(f"RAG消融实验失败: {e}")
            comprehensive_results["ablation_studies"]["rag"] = {"error": str(e)}
        
        # 顺序消融实验
        try:
            order_results = self.run_order_ablation(test_data, config_path)
            comprehensive_results["ablation_studies"]["order"] = order_results
        except Exception as e:
            logger.error(f"顺序消融实验失败: {e}")
            comprehensive_results["ablation_studies"]["order"] = {"error": str(e)}
        
        # Agent多样性消融实验
        try:
            diversity_results = self.run_agent_diversity_ablation(test_data)
            comprehensive_results["ablation_studies"]["agent_diversity"] = diversity_results
        except Exception as e:
            logger.error(f"Agent多样性消融实验失败: {e}")
            comprehensive_results["ablation_studies"]["agent_diversity"] = {"error": str(e)}
        
        # 保存综合结果
        output_file = self.output_dir / "comprehensive_evaluation.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"综合评估完成，结果保存到: {output_file}")
        
        # 生成报告
        self._generate_evaluation_report(comprehensive_results)
        
        return comprehensive_results
    
    def _save_ablation_results(self, experiment_name: str, results: Dict[str, AblationResult]):
        """保存消融实验结果"""
        output_file = self.output_dir / f"{experiment_name}_results.json"
        
        # 转换为可序列化的格式
        serializable_results = {}
        for condition, result in results.items():
            serializable_results[condition] = {
                "experiment_name": result.experiment_name,
                "condition": result.condition,
                "metrics": {
                    "accuracy": result.metrics.accuracy,
                    "precision": result.metrics.precision,
                    "recall": result.metrics.recall,
                    "f1_score": result.metrics.f1_score,
                    "consensus_rate": result.metrics.consensus_rate,
                    "avg_confidence": result.metrics.avg_confidence,
                    "processing_time": result.metrics.processing_time
                },
                "sample_results": result.sample_results
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"消融实验结果已保存: {output_file}")
    
    def _generate_evaluation_report(self, results: Dict[str, Any]):
        """生成评估报告"""
        report_file = self.output_dir / "evaluation_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 歧义检测系统评估报告\n\n")
            f.write(f"**评估时间**: {results['timestamp']}\n")
            f.write(f"**测试样本数**: {results['test_data_size']}\n\n")
            
            # RAG消融实验结果
            if "rag" in results["ablation_studies"] and "error" not in results["ablation_studies"]["rag"]:
                f.write("## RAG消融实验结果\n\n")
                rag_results = results["ablation_studies"]["rag"]
                
                f.write("| 条件 | 准确率 | 精确率 | 召回率 | F1分数 | 共识率 | 平均置信度 | 处理时间(s) |\n")
                f.write("|------|--------|--------|--------|--------|--------|------------|-------------|\n")
                
                for condition, result in rag_results.items():
                    if isinstance(result, dict) and "metrics" in result:
                        metrics = result["metrics"]
                        f.write(f"| {result['condition']} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | "
                               f"{metrics['recall']:.3f} | {metrics['f1_score']:.3f} | {metrics['consensus_rate']:.3f} | "
                               f"{metrics['avg_confidence']:.3f} | {metrics['processing_time']:.3f} |\n")
                
                f.write("\n")
            
            # 顺序消融实验结果
            if "order" in results["ablation_studies"] and "error" not in results["ablation_studies"]["order"]:
                f.write("## 顺序消融实验结果\n\n")
                order_results = results["ablation_studies"]["order"]
                
                f.write("| 条件 | 准确率 | 精确率 | 召回率 | F1分数 | 共识率 | 平均置信度 | 处理时间(s) |\n")
                f.write("|------|--------|--------|--------|--------|--------|------------|-------------|\n")
                
                for condition, result in order_results.items():
                    if isinstance(result, dict) and "metrics" in result:
                        metrics = result["metrics"]
                        f.write(f"| {result['condition']} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | "
                               f"{metrics['recall']:.3f} | {metrics['f1_score']:.3f} | {metrics['consensus_rate']:.3f} | "
                               f"{metrics['avg_confidence']:.3f} | {metrics['processing_time']:.3f} |\n")
                
                f.write("\n")
            
            # Agent多样性消融实验结果
            if "agent_diversity" in results["ablation_studies"] and "error" not in results["ablation_studies"]["agent_diversity"]:
                f.write("## Agent多样性消融实验结果\n\n")
                diversity_results = results["ablation_studies"]["agent_diversity"]
                
                f.write("| 条件 | 准确率 | 精确率 | 召回率 | F1分数 | 共识率 | 平均置信度 | 处理时间(s) |\n")
                f.write("|------|--------|--------|--------|--------|--------|------------|-------------|\n")
                
                for condition, result in diversity_results.items():
                    if isinstance(result, dict) and "metrics" in result:
                        metrics = result["metrics"]
                        f.write(f"| {result['condition']} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | "
                               f"{metrics['recall']:.3f} | {metrics['f1_score']:.3f} | {metrics['consensus_rate']:.3f} | "
                               f"{metrics['avg_confidence']:.3f} | {metrics['processing_time']:.3f} |\n")
                
                f.write("\n")
            
            f.write("## 结论\n\n")
            f.write("基于以上消融实验结果，可以得出以下结论：\n\n")
            f.write("1. **RAG效果**: 比较有无RAG的性能差异\n")
            f.write("2. **处理顺序**: 比较先共识vs先投票的效率和效果\n")
            f.write("3. **Agent多样性**: 比较统一配置vs多样化配置的影响\n\n")
            f.write("详细的数值分析和可视化图表请参考相应的结果文件。\n")
        
        logger.info(f"评估报告已生成: {report_file}")

def create_test_dataset() -> List[Dict[str, Any]]:
    """创建测试数据集"""
    test_data = [
        {
            "sentence": "他用望远镜看到了那个人。",
            "is_ambiguous": True,
            "ground_truth": "句法歧义：介词短语修饰歧义",
            "ambiguity_type": "syntactic"
        },
        {
            "sentence": "银行在河边。",
            "is_ambiguous": True,
            "ground_truth": "词汇歧义：一词多义",
            "ambiguity_type": "lexical"
        },
        {
            "sentence": "今天天气很好。",
            "is_ambiguous": False,
            "ground_truth": "无歧义",
            "ambiguity_type": "none"
        },
        {
            "sentence": "Flying planes can be dangerous.",
            "is_ambiguous": True,
            "ground_truth": "句法歧义：动名词vs形容词+名词",
            "ambiguity_type": "syntactic"
        },
        {
            "sentence": "The chicken is ready to eat.",
            "is_ambiguous": True,
            "ground_truth": "句法歧义：主语和宾语关系",
            "ambiguity_type": "syntactic"
        },
        {
            "sentence": "I love programming.",
            "is_ambiguous": False,
            "ground_truth": "无歧义",
            "ambiguity_type": "none"
        },
        {
            "sentence": "老师的书很有趣。",
            "is_ambiguous": False,
            "ground_truth": "无歧义",
            "ambiguity_type": "none"
        },
        {
            "sentence": "他们在讨论老师的问题。",
            "is_ambiguous": True,
            "ground_truth": "句法歧义：所有格修饰歧义",
            "ambiguity_type": "syntactic"
        }
    ]
    
    return test_data

if __name__ == "__main__":
    # 创建测试数据
    test_data = create_test_dataset()
    
    # 初始化评估模块
    evaluator = EvaluationModule()
    
    # 运行综合评估
    try:
        results = evaluator.run_comprehensive_evaluation(test_data)
        print("综合评估完成！")
        print(f"结果保存在: {evaluator.output_dir}")
    except Exception as e:
        print(f"评估过程出错: {e}")
        logger.error(f"评估失败: {e}")