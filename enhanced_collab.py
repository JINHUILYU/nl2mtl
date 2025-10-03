
"""
Enhanced Collaborative Ambiguity Detection System
增强版多智能体协作歧义检测系统，整合所有改进功能
"""

import json
import copy
import re
import os
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv
import logging

from ambiguity_detector import AmbiguityDetector, DetectionResult, ProcessOrder
from rag_module import RAGModule
from evaluation_module import EvaluationModule

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CollaborativeResult:
    """协作检测结果"""
    sentence: str
    is_ambiguous: bool
    final_interpretation: str
    confidence: float
    consensus_reached_at: Optional[str]
    agent_responses: Dict[str, Any]
    discussion_log: List[str]
    voting_results: Optional[Dict[str, Any]]
    arbitration_result: Optional[Dict[str, Any]]
    processing_time: float
    rag_knowledge_used: List[Dict[str, Any]]

class EnhancedCollaborativeDetector:
    """增强版协作检测器"""
    
    def __init__(self, config_path: str = "config/detector_config.json", 
                 enable_rag: bool = True, enable_evaluation: bool = True):
        """
        初始化增强版协作检测器
        
        Args:
            config_path: 配置文件路径
            enable_rag: 是否启用RAG
            enable_evaluation: 是否启用评估功能
        """
        self.config_path = config_path
        self.detector = AmbiguityDetector(config_path)
        
        # RAG模块
        self.rag_enabled = enable_rag
        if self.rag_enabled:
            try:
                self.rag_module = RAGModule()
                logger.info("RAG模块已启用")
            except Exception as e:
                logger.warning(f"RAG模块初始化失败: {e}")
                self.rag_enabled = False
        
        # 评估模块
        self.evaluation_enabled = enable_evaluation
        if self.evaluation_enabled:
            self.evaluator = EvaluationModule()
            logger.info("评估模块已启用")
        
        # 句子嵌入模型
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # 结果缓存
        self.results_cache = {}
        
    def detect_with_full_pipeline(self, sentence: str, context: str = "") -> CollaborativeResult:
        """
        使用完整流水线进行歧义检测
        
        Args:
            sentence: 待检测句子
            context: 上下文信息
            
        Returns:
            协作检测结果
        """
        start_time = time.time()
        logger.info(f"开始完整流水线检测: {sentence}")
        
        # 检查缓存
        cache_key = f"{sentence}_{context}"
        if cache_key in self.results_cache:
            logger.info("使用缓存结果")
            return self.results_cache[cache_key]
        
        # 第一阶段：独立分析
        agent_responses = self._run_independent_analysis(sentence, context)
        discussion_log = ["完成独立分析阶段"]
        
        # 检查初始一致性
        initial_consensus = self._check_initial_consensus(agent_responses)
        if initial_consensus:
            end_time = time.time()
            result = CollaborativeResult(
                sentence=sentence,
                is_ambiguous=initial_consensus["is_ambiguous"],
                final_interpretation=initial_consensus["interpretation"],
                confidence=initial_consensus["confidence"],
                consensus_reached_at="initial",
                agent_responses=agent_responses,
                discussion_log=discussion_log + ["初始阶段达成共识"],
                voting_results=None,
                arbitration_result=None,
                processing_time=end_time - start_time,
                rag_knowledge_used=initial_consensus.get("rag_knowledge", [])
            )
            self.results_cache[cache_key] = result
            return result
        
        # 第二阶段：多轮讨论
        discussion_results = self._run_multi_round_discussion(sentence, agent_responses)
        discussion_log.extend(discussion_results["log"])
        
        # 检查讨论后一致性
        discussion_consensus = self._check_discussion_consensus(discussion_results["refined_responses"])
        if discussion_consensus:
            end_time = time.time()
            result = CollaborativeResult(
                sentence=sentence,
                is_ambiguous=discussion_consensus["is_ambiguous"],
                final_interpretation=discussion_consensus["interpretation"],
                confidence=discussion_consensus["confidence"],
                consensus_reached_at="discussion",
                agent_responses=discussion_results["refined_responses"],
                discussion_log=discussion_log + ["讨论后达成共识"],
                voting_results=None,
                arbitration_result=None,
                processing_time=end_time - start_time,
                rag_knowledge_used=discussion_consensus.get("rag_knowledge", [])
            )
            self.results_cache[cache_key] = result
            return result
        
        # 第三阶段：投票
        voting_results = self._run_structured_voting(sentence, discussion_results["refined_responses"])
        discussion_log.append("完成投票阶段")
        
        if voting_results["winner"]:
            end_time = time.time()
            result = CollaborativeResult(
                sentence=sentence,
                is_ambiguous=True,  # 需要投票说明存在歧义
                final_interpretation=voting_results["winning_interpretation"],
                confidence=voting_results["confidence"],
                consensus_reached_at="voting",
                agent_responses=discussion_results["refined_responses"],
                discussion_log=discussion_log + ["投票决出获胜者"],
                voting_results=voting_results,
                arbitration_result=None,
                processing_time=end_time - start_time,
                rag_knowledge_used=voting_results.get("rag_knowledge", [])
            )
            self.results_cache[cache_key] = result
            return result
        
        # 第四阶段：专家仲裁
        arbitration_result = self._run_expert_arbitration(sentence, discussion_results["refined_responses"], voting_results)
        discussion_log.append("完成专家仲裁")
        
        end_time = time.time()
        result = CollaborativeResult(
            sentence=sentence,
            is_ambiguous=arbitration_result.get("is_ambiguous", True),
            final_interpretation=arbitration_result.get("final_interpretation", "仲裁失败"),
            confidence=arbitration_result.get("confidence", 0.5),
            consensus_reached_at="arbitration",
            agent_responses=discussion_results["refined_responses"],
            discussion_log=discussion_log + ["仲裁完成"],
            voting_results=voting_results,
            arbitration_result=arbitration_result,
            processing_time=end_time - start_time,
            rag_knowledge_used=arbitration_result.get("rag_knowledge", [])
        )
        
        self.results_cache[cache_key] = result
        return result
    
    def _run_independent_analysis(self, sentence: str, context: str = "") -> Dict[str, Any]:
        """运行独立分析阶段"""
        logger.info("=== 第一阶段：独立分析 ===")
        
        agent_responses = {}
        
        for agent in self.detector.agents:
            # 构建提示词
            prompt = self._build_enhanced_prompt(sentence, context, agent.role.value)
            
            # 如果启用RAG，增强提示词
            if self.rag_enabled:
                try:
                    prompt = self.rag_module.enhance_prompt_with_knowledge(prompt, sentence)
                except Exception as e:
                    logger.warning(f"RAG增强失败: {e}")
            
            # 调用LLM
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            try:
                response = self.detector._call_llm(agent.name, messages)
                structured_answer = self.detector._extract_structured_answer(response)
                
                agent_responses[agent.name] = {
                    "raw_response": response,
                    "structured": structured_answer,
                    "agent_role": agent.role.value,
                    "model": agent.model,
                    "temperature": agent.temperature
                }
                
                logger.info(f"{agent.name} ({agent.role.value}): "
                           f"{len(structured_answer.get('interpretations', []))} 种解释")
                
            except Exception as e:
                logger.error(f"Agent {agent.name} 分析失败: {e}")
                agent_responses[agent.name] = {
                    "error": str(e),
                    "agent_role": agent.role.value
                }
        
        return agent_responses
    
    def _build_enhanced_prompt(self, sentence: str, context: str, role: str) -> str:
        """构建增强版提示词"""
        base_prompt = f"""
请作为{role}专家，深入分析以下句子是否存在歧义。

句子: "{sentence}"
{f"上下文: {context}" if context else ""}

请按照以下结构化格式输出你的分析：

```json
{{
    "is_ambiguous": true/false,
    "confidence": 0.85,
    "interpretations": [
        {{
            "interpretation": "具体解释内容",
            "confidence": 0.8,
            "reasoning": "支持这种解释的理由",
            "linguistic_evidence": "语言学证据"
        }}
    ],
    "ambiguity_type": "词汇歧义/句法歧义/语义歧义/语用歧义/无歧义",
    "ambiguity_source": "歧义来源的具体分析",
    "resolution_strategy": "解决歧义的建议策略",
    "overall_reasoning": "整体分析过程和结论"
}}
```

分析要求：
1. 仔细考虑句子的所有可能理解方式
2. 提供具体的语言学证据支持你的判断
3. 如果存在歧义，明确指出歧义的类型和来源
4. 给出解决歧义的具体建议
5. 根据你的专业角色特点进行分析
"""
        
        # 根据角色添加特定指导
        role_guidance = {
            "logician": """
作为逻辑学家，请特别关注：
- 句子的逻辑结构和形式化表示
- 语法规则的严格应用
- 逻辑关系的一致性
- 形式语义学的分析方法
""",
            "creative": """
作为创意思维专家，请特别关注：
- 多种可能的理解角度
- 隐喻和比喻的使用
- 文化背景和语境因素
- 非常规的解释可能性
""",
            "pragmatic": """
作为实用主义分析师，请特别关注：
- 日常使用中的常见理解
- 交际意图和语用含义
- 现实世界的常识推理
- 实际应用场景的考虑
"""
        }
        
        return base_prompt + role_guidance.get(role, "")
    
    def _check_initial_consensus(self, agent_responses: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检查初始共识"""
        valid_responses = {k: v for k, v in agent_responses.items() 
                          if "structured" in v and "error" not in v}
        
        if len(valid_responses) < 2:
            return None
        
        # 提取所有解释
        all_interpretations = []
        for agent_name, response in valid_responses.items():
            structured = response["structured"]
            for interp in structured.get("interpretations", []):
                all_interpretations.append({
                    "agent": agent_name,
                    "interpretation": interp.get("interpretation", ""),
                    "confidence": interp.get("confidence", 0.5),
                    "is_ambiguous": structured.get("is_ambiguous", False)
                })
        
        if not all_interpretations:
            return None
        
        # 使用语义相似度检查一致性
        threshold = self.detector.config.get("similarity_threshold", 0.95)
        interpretations_text = [item["interpretation"] for item in all_interpretations]
        
        if len(interpretations_text) < 2:
            return None
        
        # 计算相似度矩阵
        embeddings = self.sentence_model.encode(interpretations_text, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(embeddings, embeddings)
        
        # 检查是否所有解释都高度相似
        high_similarity_count = 0
        total_pairs = 0
        
        for i in range(len(interpretations_text)):
            for j in range(i + 1, len(interpretations_text)):
                similarity = float(similarities[i][j])
                total_pairs += 1
                if similarity >= threshold:
                    high_similarity_count += 1
        
        # 如果大部分解释都高度相似，认为达成共识
        if total_pairs > 0 and high_similarity_count / total_pairs >= 0.8:
            # 选择置信度最高的解释
            best_interpretation = max(all_interpretations, key=lambda x: x["confidence"])
            
            return {
                "is_ambiguous": best_interpretation["is_ambiguous"],
                "interpretation": best_interpretation["interpretation"],
                "confidence": best_interpretation["confidence"],
                "consensus_type": "semantic_similarity"
            }
        
        return None
    
    def _run_multi_round_discussion(self, sentence: str, initial_responses: Dict[str, Any]) -> Dict[str, Any]:
        """运行多轮讨论"""
        logger.info("=== 第二阶段：多轮讨论 ===")
        
        max_rounds = self.detector.config.get("max_discussion_rounds", 3)
        refined_responses = copy.deepcopy(initial_responses)
        discussion_log = []
        
        for round_num in range(max_rounds):
            logger.info(f"讨论轮次 {round_num + 1}/{max_rounds}")
            
            # 准备讨论摘要
            discussion_summary = self._prepare_discussion_summary(refined_responses)
            
            # 每个Agent参与讨论
            round_responses = {}
            for agent in self.detector.agents:
                if agent.name not in refined_responses or "error" in refined_responses[agent.name]:
                    continue
                
                discussion_prompt = self._build_discussion_prompt(
                    sentence, refined_responses, discussion_summary, agent.role.value, round_num
                )
                
                messages = [
                    {"role": "system", "content": agent.system_prompt + 
                     "\n你现在需要参与多轮讨论，分析其他专家的观点并完善自己的答案。"},
                    {"role": "user", "content": discussion_prompt}
                ]
                
                try:
                    response = self.detector._call_llm(agent.name, messages)
                    structured_answer = self.detector._extract_structured_answer(response)
                    
                    round_responses[agent.name] = {
                        "raw_response": response,
                        "structured": structured_answer,
                        "agent_role": agent.role.value,
                        "discussion_round": round_num + 1
                    }
                    
                except Exception as e:
                    logger.error(f"Agent {agent.name} 讨论失败: {e}")
                    round_responses[agent.name] = refined_responses[agent.name]
            
            refined_responses.update(round_responses)
            discussion_log.append(f"完成第 {round_num + 1} 轮讨论")
            
            # 检查是否达成共识
            consensus = self._check_discussion_consensus(refined_responses)
            if consensus:
                discussion_log.append(f"在第 {round_num + 1} 轮讨论后达成共识")
                break
        
        return {
            "refined_responses": refined_responses,
            "log": discussion_log
        }
    
    def _prepare_discussion_summary(self, responses: Dict[str, Any]) -> str:
        """准备讨论摘要"""
        summary = "各专家当前的分析结果：\n\n"
        
        for agent_name, response in responses.items():
            if "structured" not in response or "error" in response:
                continue
                
            structured = response["structured"]
            role = response.get("agent_role", "unknown")
            
            summary += f"**{agent_name} ({role})**:\n"
            summary += f"- 是否有歧义: {'是' if structured.get('is_ambiguous', False) else '否'}\n"
            summary += f"- 置信度: {structured.get('confidence', 0.5):.2f}\n"
            summary += f"- 歧义类型: {structured.get('ambiguity_type', '未指定')}\n"
            
            interpretations = structured.get('interpretations', [])
            summary += f"- 解释数量: {len(interpretations)}\n"
            
            for i, interp in enumerate(interpretations[:2], 1):  # 只显示前2个解释
                summary += f"  {i}. {interp.get('interpretation', '')[:100]}...\n"
            
            if structured.get('overall_reasoning'):
                summary += f"- 整体推理: {structured['overall_reasoning'][:150]}...\n"
            
            summary += "\n"
        
        return summary
    
    def _build_discussion_prompt(self, sentence: str, responses: Dict[str, Any], 
                               summary: str, role: str, round_num: int) -> str:
        """构建讨论提示词"""
        return f"""
现在进行第 {round_num + 1} 轮专家讨论。

原句子: "{sentence}"

{summary}

作为{role}专家，请基于其他专家的分析，重新评估并完善你的观点：

1. **分析其他专家观点**: 指出你同意或不同意的地方，并说明理由
2. **识别分歧点**: 明确指出主要的分歧在哪里
3. **提供新证据**: 基于你的专业角度，提供额外的分析证据
4. **完善解释**: 更新或完善你的解释，尝试解决分歧
5. **寻求共识**: 尝试找到各方都能接受的解释

请继续使用JSON格式输出你的更新分析：
```json
{{
    "is_ambiguous": true/false,
    "confidence": 0.85,
    "interpretations": [...],
    "ambiguity_type": "...",
    "discussion_points": "对其他观点的具体分析",
    "consensus_attempt": "尝试达成的共识内容",
    "disagreement_analysis": "对分歧点的分析",
    "new_evidence": "提供的新证据或观点",
    "overall_reasoning": "更新后的整体分析过程"
}}
```
"""
    
    def _check_discussion_consensus(self, responses: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检查讨论后的共识"""
        # 实现与初始共识检查类似的逻辑，但考虑讨论后的细化
        return self._check_initial_consensus(responses)
    
    def _run_structured_voting(self, sentence: str, responses: Dict[str, Any]) -> Dict[str, Any]:
        """运行结构化投票"""
        logger.info("=== 第三阶段：结构化投票 ===")
        
        # 收集所有候选解释
        candidates = self._collect_voting_candidates(responses)
        
        if not candidates:
            return {"winner": None, "error": "没有有效的候选解释"}
        
        # 每个Agent投票
        votes = {}
        vote_details = {}
        
        for agent in self.detector.agents:
            if agent.name not in responses or "error" in responses[agent.name]:
                continue
            
            vote_prompt = self._build_voting_prompt(sentence, candidates)
            
            messages = [
                {"role": "system", "content": agent.system_prompt + 
                 "\n你需要客观评估所有候选解释并投票选择最佳答案。"},
                {"role": "user", "content": vote_prompt}
            ]
            
            try:
                response = self.detector._call_llm(agent.name, messages)
                vote_result = self._extract_vote_result(response)
                
                if vote_result:
                    votes[agent.name] = vote_result["choice"]
                    vote_details[agent.name] = vote_result
                    
            except Exception as e:
                logger.error(f"Agent {agent.name} 投票失败: {e}")
        
        # 统计投票结果
        vote_count = {}
        for vote in votes.values():
            vote_count[vote] = vote_count.get(vote, 0) + 1
        
        # 确定获胜者
        winner = None
        winning_interpretation = None
        confidence = 0.5
        
        if vote_count:
            max_votes = max(vote_count.values())
            winners = [k for k, v in vote_count.items() if v == max_votes]
            
            if len(winners) == 1:
                winner = winners[0]
                winning_interpretation = candidates.get(winner, {}).get("interpretation", "")
                confidence = candidates.get(winner, {}).get("confidence", 0.5)
        
        return {
            "winner": winner,
            "winning_interpretation": winning_interpretation,
            "confidence": confidence,
            "vote_count": vote_count,
            "votes": votes,
            "vote_details": vote_details,
            "candidates": candidates
        }
    
    def _collect_voting_candidates(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """收集投票候选项"""
        candidates = {}
        
        for agent_name, response in responses.items():
            if "structured" not in response or "error" in response:
                continue
            
            structured = response["structured"]
            interpretations = structured.get("interpretations", [])
            
            for i, interp in enumerate(interpretations):
                candidate_id = f"{agent_name}_{i}"
                candidates[candidate_id] = {
                    "interpretation": interp.get("interpretation", ""),
                    "confidence": interp.get("confidence", 0.5),
                    "source_agent": agent_name,
                    "reasoning": interp.get("reasoning", ""),
                    "is_ambiguous": structured.get("is_ambiguous", False)
                }
        
        return candidates
    
    def _build_voting_prompt(self, sentence: str, candidates: Dict[str, Any]) -> str:
        """构建投票提示词"""
        candidates_text = ""
        for candidate_id, candidate in candidates.items():
            candidates_text += f"**选项 {candidate_id}**:\n"
            candidates_text += f"- 解释: {candidate['interpretation']}\n"
            candidates_text += f"- 置信度: {candidate['confidence']:.2f}\n"
            candidates_text += f"- 来源: {candidate['source_agent']}\n"
            candidates_text += f"- 推理: {candidate['reasoning'][:100]}...\n\n"
        
        return f"""
请为以下句子的最佳解释投票：

句子: "{sentence}"

候选解释：
{candidates_text}

请客观评估所有候选解释，选择你认为最准确、最合理的一个，并详细说明理由。

输出格式：
```json
{{
    "choice": "选项ID (如 Agent_A_0)",
    "confidence": 0.9,
    "reason": "详细的选择理由",
    "analysis": {{
        "strengths": "所选解释的优点",
        "weaknesses": "其他解释的不足",
        "evidence": "支持选择的证据"
    }},
    "alternative_ranking": ["次优选择1", "次优选择2"]
}}
```
"""
    
    def _extract_vote_result(self, text: str) -> Optional[Dict[str, Any]]:
        """提取投票结果"""
        # 尝试提取JSON格式
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试提取普通代码块
        code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 文本解析
        choice_match = re.search(r'"choice":\s*"([^"]+)"', text)
        reason_match = re.search(r'"reason":\s*"([^"]+)"', text)
        
        if choice_match:
            return {
                "choice": choice_match.group(1),
                "reason": reason_match.group(1) if reason_match else "未提供理由",
                "confidence": 0.5
            }
        
        return None
    
    def _run_expert_arbitration(self, sentence: str, responses: Dict[str, Any], 
                              voting_results: Dict[str, Any]) -> Dict[str, Any]:
        """运行专家仲裁"""
        logger.info("=== 第四阶段：专家仲裁 ===")
        
        arbitration_prompt = self._build_arbitration_prompt(sentence, responses, voting_results)
        
        # 使用最强的可用模型进行仲裁
        arbitrator_client = None
        for client_name, client in self.detector.clients.items():
            arbitrator_client = client
            break
        
        if not arbitrator_client:
            return {"error": "没有可用的仲裁客户端"}
        
        messages = [
            {"role": "system", "content": """你是一位资深的语言学专家和歧义分析专家，负责对有争议的语言歧义问题做出最终裁决。
            你需要综合考虑所有专家意见、投票结果和语言学证据，做出公正、准确的判断。"""},
            {"role": "user", "content": arbitration_prompt}
        ]
        
        try:
            # 使用第一个可用的Agent配置
            agent = self.detector.agents[0]
            response = arbitrator_client.chat.completions.create(
                model=agent.model,
                messages=messages,
                temperature=0.3  # 仲裁时使用较低的温度
            ).choices[0].message.content.strip()
            
            arbitration_result = self.detector._extract_structured_answer(response)
            
            return {
                "arbitration_response": response,
                "final_decision": arbitration_result,
                "is_ambiguous": arbitration_result.get("is_ambiguous", True),
                "final_interpretation": arbitration_result.get("best_interpretation", "仲裁失败"),
                "confidence": arbitration_result.get("confidence", 0.5),
                "arbitrator": "Expert System"
            }
            
        except Exception as e:
            logger.error(f"专家仲裁失败: {e}")
            return {"error": f"仲裁失败: {str(e)}"}
    
    def _build_arbitration_prompt(self, sentence: str, responses: Dict[str, Any], 
                                voting_results: Dict[str, Any]) -> str:
        """构建仲裁提示词"""
        # 整理专家意见
        expert_opinions = ""
        for agent_name, response in responses.items():
            if "structured" not in response:
                continue
            structured = response["structured"]
            expert_opinions += f"**{agent_name}**: {structured.get('is_ambiguous', False)}, "
            expert_opinions += f"置信度: {structured.get('confidence', 0.5):.2f}\n"
            for i, interp in enumerate(structured.get('interpretations', [])[:2]):
                expert_opinions += f"  解释{i+1}: {interp.get('interpretation', '')}\n"
            expert_opinions += "\n"
        
        # 整理投票结果
        voting_summary = f"投票结果: {voting_results.get('vote_count', {})}\n"
        if voting_results.get('winner'):
            voting_summary += f"获胜选项: {voting_results['winner']}\n"
            voting_summary += f"获胜解释: {voting_results.get('winning_interpretation', '')}\n"
        
        return f"""
作为最终仲裁者，请对以下有争议的语言歧义问题做出裁决：

**待分析句子**: "{sentence}"

**专家意见汇总**:
{expert_opinions}

**投票结果**:
{voting_summary}

请综合考虑所有信息，做出最终判断。请特别注意：
1. 语言学理论的支持
2. 专家意见的权重
3. 投票结果的参考价值
4. 实际语言使用的常见模式

请以JSON格式输出你的最终裁决：
```json
{{
    "is_ambiguous": true/false,
    "best_interpretation": "最佳解释",
    "confidence": 0.9,
    "reasoning": "详细的仲裁理由",
    "linguistic_analysis": "语言学分析",
    "final_judgment": "最终判决说明"
}}
```
"""

    def run_batch_detection(self, input_file: str, output_file: str,
                           enable_detailed_logging: bool = True) -> Dict[str, Any]:
        """
        批量检测功能
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            enable_detailed_logging: 是否启用详细日志
            
        Returns:
            批量检测统计结果
        """
        logger.info(f"开始批量检测: {input_file}")
        
        # 读取输入数据
        df = pd.read_excel(input_file)
        sentences = df["sentence"].tolist() if "sentence" in df.columns else df.iloc[:, 0].tolist()
        
        results = []
        stats = {
            "total_processed": 0,
            "ambiguous_count": 0,
            "consensus_at_initial": 0,
            "consensus_at_discussion": 0,
            "consensus_at_voting": 0,
            "consensus_at_arbitration": 0,
            "avg_processing_time": 0.0,
            "avg_confidence": 0.0
        }
        
        total_time = 0
        total_confidence = 0
        
        for i, sentence in enumerate(sentences):
            logger.info(f"处理第 {i+1}/{len(sentences)} 个句子: {sentence[:50]}...")
            
            try:
                result = self.detect_with_full_pipeline(sentence)
                
                # 统计信息
                stats["total_processed"] += 1
                if result.is_ambiguous:
                    stats["ambiguous_count"] += 1
                
                if result.consensus_reached_at == "initial":
                    stats["consensus_at_initial"] += 1
                elif result.consensus_reached_at == "discussion":
                    stats["consensus_at_discussion"] += 1
                elif result.consensus_reached_at == "voting":
                    stats["consensus_at_voting"] += 1
                elif result.consensus_reached_at == "arbitration":
                    stats["consensus_at_arbitration"] += 1
                
                total_time += result.processing_time
                total_confidence += result.confidence
                
                # 保存结果
                result_dict = {
                    "sentence": result.sentence,
                    "is_ambiguous": result.is_ambiguous,
                    "final_interpretation": result.final_interpretation,
                    "confidence": result.confidence,
                    "consensus_reached_at": result.consensus_reached_at,
                    "processing_time": result.processing_time,
                    "discussion_log": "; ".join(result.discussion_log)
                }
                
                if enable_detailed_logging:
                    result_dict.update({
                        "agent_responses": json.dumps(result.agent_responses, ensure_ascii=False),
                        "voting_results": json.dumps(result.voting_results, ensure_ascii=False) if result.voting_results else None,
                        "arbitration_result": json.dumps(result.arbitration_result, ensure_ascii=False) if result.arbitration_result else None,
                        "rag_knowledge_used": json.dumps(result.rag_knowledge_used, ensure_ascii=False)
                    })
                
                results.append(result_dict)
                
                # 每10个样本保存一次
                if (i + 1) % 10 == 0:
                    pd.DataFrame(results).to_excel(output_file, index=False)
                    logger.info(f"已保存 {len(results)} 个结果")
                
            except Exception as e:
                logger.error(f"处理句子失败: {sentence[:50]}... - {e}")
                results.append({
                    "sentence": sentence,
                    "error": str(e),
                    "is_ambiguous": None,
                    "confidence": 0.0
                })
        
        # 计算最终统计
        if stats["total_processed"] > 0:
            stats["avg_processing_time"] = total_time / stats["total_processed"]
            stats["avg_confidence"] = total_confidence / stats["total_processed"]
        
        # 最终保存
        pd.DataFrame(results).to_excel(output_file, index=False)
        
        # 保存统计信息
        stats_file = output_file.replace(".xlsx", "_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"批量检测完成! 处理了 {stats['total_processed']} 个句子")
        logger.info(f"结果保存到: {output_file}")
        logger.info(f"统计信息保存到: {stats_file}")
        
        return stats

def main():
    """主函数，演示系统使用"""
    # 创建增强版检测器
    detector = EnhancedCollaborativeDetector(
        config_path="config/detector_config.json",
        enable_rag=True,
        enable_evaluation=True
    )
    
    # 测试句子
    test_sentences = [
        "他用望远镜看到了那个人。",
        "银行在河边。",
        "今天天气很好。",
        "Flying planes can be dangerous.",
        "The chicken is ready to eat."
    ]
    
    print("=== 增强版协作歧义检测系统演示 ===\n")
    
    for sentence in test_sentences:
        print(f"句子: {sentence}")
        print("-" * 60)
        
        try:
            result = detector.detect_with_full_pipeline(sentence)
            
            print(f"是否有歧义: {result.is_ambiguous}")
            print(f"最终解释: {result.final_interpretation}")
            print(f"置信度: {result.confidence:.3f}")
            print(f"共识达成阶段: {result.consensus_reached_at}")
            print(f"处理时间: {result.processing_time:.2f}秒")
            print(f"处理日志: {'; '.join(result.discussion_log)}")
            
            if result.rag_knowledge_used:
                print(f"使用的RAG知识: {len(result.rag_knowledge_used)} 条")
            
        except Exception as e:
            print(f"检测失败: {e}")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()