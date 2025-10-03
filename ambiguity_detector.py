"""
Enhanced Ambiguity Detection Framework
基于多智能体协作的歧义检测系统，实现创新的功能驱动检测方法
"""

import json
import re
import copy
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv
import logging
from pathlib import Path
from rag_module import RAGModule

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Agent角色定义"""
    LOGICIAN = "logician"
    CREATIVE = "creative" 
    PRAGMATIC = "pragmatic"

class ProcessOrder(Enum):
    """处理顺序定义"""
    CONSENSUS_FIRST = "consensus_first"  # 先共识讨论，后投票
    VOTING_FIRST = "voting_first"       # 先投票，后共识讨论

@dataclass
class AgentConfig:
    """Agent配置"""
    name: str
    role: AgentRole
    model: str
    temperature: float
    system_prompt: str

@dataclass
class DetectionResult:
    """检测结果"""
    is_ambiguous: bool
    final_answer: str
    confidence: float
    process_log: List[str]
    phase_results: Dict[str, Any]
    consensus_reached_at: Optional[str] = None
    mtl_expressions: Optional[List[str]] = None
    voting_results: Optional[Dict] = None
    expert_arbitration: Optional[str] = None

class AmbiguityDetector:
    """歧义检测器主类"""
    
    def __init__(self, config_path: str = "config/detector_config.json"):
        """初始化检测器"""
        self.config = self._load_config(config_path)
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.clients = self._initialize_clients()
        self.agents = self._initialize_agents()
        self.rag_enabled = self.config.get("rag_enabled", False)
        self.process_order = ProcessOrder(self.config.get("process_order", "consensus_first"))
        
        # 初始化RAG模块
        if self.rag_enabled:
            self.rag_module = RAGModule()
        else:
            self.rag_module = None
            
        # 加载基础prompt
        self.base_prompt = self._load_base_prompt()
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 未找到，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
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
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.8,
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_URL"
                },
                {
                    "name": "Agent_C",
                    "role": "pragmatic", 
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.5,
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_URL"
                }
            ],
            "similarity_threshold": 0.95,
            "rag_enabled": False,
            "process_order": "consensus_first",
            "max_discussion_rounds": 3,
            "confidence_threshold": 0.8
        }
    
    def _initialize_clients(self) -> Dict[str, OpenAI]:
        """初始化API客户端"""
        load_dotenv()
        clients = {}
        
        for agent_config in self.config["agents"]:
            api_key = os.getenv(agent_config["api_key_env"])
            base_url = os.getenv(agent_config["base_url_env"])
            
            if api_key:
                clients[agent_config["name"]] = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
            else:
                logger.warning(f"API密钥未找到: {agent_config['api_key_env']}")
                
        return clients
    
    def _load_base_prompt(self) -> str:
        """加载基础prompt模板"""
        try:
            with open("base_prompt.txt", 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("基础prompt文件未找到，使用默认prompt")
            return """请分析以下句子并生成对应的MTL(Metric Temporal Logic)表达式。
            
            句子: [INPUT TEXT]
            
            请提供详细的分析过程和最终的MTL表达式。"""
    
    def _initialize_agents(self) -> List[AgentConfig]:
        """初始化Agent配置"""
        agents = []
        role_prompts = {
            AgentRole.LOGICIAN: """You are a rigorous logician who analyzes problems with strict logical reasoning. 
            You focus on formal structure, consistency, and precise interpretation of language.""",
            
            AgentRole.CREATIVE: """You are a creative thinker who considers multiple interpretations and possibilities.
            You explore alternative meanings, metaphorical uses, and unconventional perspectives.""",
            
            AgentRole.PRAGMATIC: """You are a pragmatic analyst who focuses on common-sense interpretations and practical applications.
            You consider real-world context and typical usage patterns."""
        }
        
        for agent_config in self.config["agents"]:
            role = AgentRole(agent_config["role"])
            agents.append(AgentConfig(
                name=agent_config["name"],
                role=role,
                model=agent_config["model"],
                temperature=agent_config["temperature"],
                system_prompt=role_prompts[role]
            ))
            
        return agents
    
    def _call_llm(self, agent_name: str, messages: List[Dict]) -> str:
        """调用LLM API"""
        if agent_name not in self.clients:
            raise ValueError(f"客户端未找到: {agent_name}")
            
        client = self.clients[agent_name]
        agent = next(a for a in self.agents if a.name == agent_name)
        
        try:
            response = client.chat.completions.create(
                model=agent.model,
                messages=messages,  # type: ignore
                temperature=agent.temperature
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            logger.error(f"LLM调用失败 {agent_name}: {e}")
            return ""
    
    def _extract_structured_answer(self, text: str) -> Dict:
        """提取结构化答案"""
        # 尝试提取JSON格式的答案
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试提取普通代码块中的JSON
        code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 如果无法提取JSON，返回文本解析结果
        return self._parse_text_answer(text)
    
    def _parse_text_answer(self, text: str) -> Dict:
        """解析文本答案为结构化格式"""
        # 尝试提取MTL表达式
        mtl_expressions = []
        
        # 查找代码块中的MTL表达式
        code_blocks = re.findall(r'```(?:ltl|mtl)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        for block in code_blocks:
            if block.strip():
                mtl_expressions.append({
                    "expression": block.strip(),
                    "interpretation": "从代码块提取的MTL表达式",
                    "confidence": 0.7,
                    "reasoning": "从响应的代码块中提取"
                })
        
        # 如果没有找到代码块，尝试其他模式
        if not mtl_expressions:
            # 查找可能的MTL表达式模式
            mtl_patterns = [
                r'(?:MTL|LTL)(?:\s*表达式)?[：:]\s*([^\n]+)',
                r'(?:表达式|Expression)[：:]\s*([^\n]+)',
                r'(?:Formula|公式)[：:]\s*([^\n]+)'
            ]
            
            for pattern in mtl_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if match.strip() and len(match.strip()) > 3:
                        mtl_expressions.append({
                            "expression": match.strip(),
                            "interpretation": "基于模式匹配提取的表达式",
                            "confidence": 0.5,
                            "reasoning": "通过正则表达式模式匹配提取"
                        })
        
        # 如果仍然没有找到，使用默认处理
        if not mtl_expressions:
            # 查找编号的解释
            numbered_pattern = r'(\d+\.?\s*)(.*?)(?=\d+\.|\n\n|$)'
            matches = re.findall(numbered_pattern, text, re.DOTALL)
            
            for match in matches:
                interpretation = match[1].strip()
                if interpretation and len(interpretation) > 10:
                    mtl_expressions.append({
                        "expression": interpretation[:50] + "...",  # 截断作为表达式
                        "interpretation": interpretation,
                        "confidence": 0.3,
                        "reasoning": "从编号列表提取"
                    })
        
        # 确保至少有一个表达式
        if not mtl_expressions:
            mtl_expressions = [{
                "expression": text[:100].replace('\n', ' ').strip(),
                "interpretation": text[:200],
                "confidence": 0.2,
                "reasoning": "默认提取"
            }]
        
        return {
            "mtl_expressions": mtl_expressions,
            "is_ambiguous": len(mtl_expressions) > 1,
            "ambiguity_analysis": text
        }
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        embeddings = self.sentence_model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return float(similarity.item())
    
    def _check_answer_consistency(self, answers: Dict[str, Dict]) -> Tuple[bool, List[str]]:
        """检查答案一致性"""
        if len(answers) < 2:
            return True, []
        
        # 提取所有解释
        all_interpretations = []
        for agent_name, answer in answers.items():
            for interp in answer.get("interpretations", []):
                all_interpretations.append((agent_name, interp["interpretation"]))
        
        # 计算相似度矩阵
        threshold = self.config.get("similarity_threshold", 0.95)
        inconsistencies = []
        
        for i, (agent1, interp1) in enumerate(all_interpretations):
            for j, (agent2, interp2) in enumerate(all_interpretations[i+1:], i+1):
                similarity = self._calculate_semantic_similarity(interp1, interp2)
                if similarity < threshold:
                    inconsistencies.append(f"{agent1} vs {agent2}: {similarity:.3f}")
        
        is_consistent = len(inconsistencies) == 0
        return is_consistent, inconsistencies
    
    def _generate_initial_answers(self, sentence: str, context: str = "") -> Dict[str, Dict]:
        """第一阶段：生成初始答案"""
        logger.info("=== 第一阶段：独立解释生成 ===")
        
        answers = {}
        
        for agent in self.agents:
            prompt = self._build_initial_prompt(sentence, context, agent.role)
            
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self._call_llm(agent.name, messages)
            structured_answer = self._extract_structured_answer(response)
            
            answers[agent.name] = {
                "raw_response": response,
                "structured": structured_answer,
                "agent_role": agent.role.value
            }
            
            mtl_count = len(structured_answer.get('mtl_expressions', []))
            logger.info(f"{agent.name} ({agent.role.value}): {mtl_count} 种MTL表达式")
        
        return answers
    
    def _build_initial_prompt(self, sentence: str, context: str, role: AgentRole) -> str:
        """构建初始提示词"""
        # 使用base_prompt作为模板，替换INPUT TEXT
        prompt = self.base_prompt.replace("[INPUT TEXT]", sentence)
        
        # 如果有上下文信息，添加到prompt中
        if context:
            context_section = f"\n\n**Additional Context:**\n{context}\n"
            prompt = prompt.replace("<Question>", f"**Additional Context:**\n{context}\n\n<Question>")
        
        # 添加歧义检测的特殊要求
        ambiguity_instruction = f"""

**Ambiguity Detection Task:**
请同时分析该句子是否存在歧义，如果存在歧义，请提供所有可能的MTL表达式解释。

请以如下JSON格式输出结果：
```json
{{
    "is_ambiguous": true/false,
    "mtl_expressions": [
        {{
            "expression": "MTL表达式1",
            "interpretation": "对应的自然语言解释",
            "confidence": 0.8,
            "reasoning": "选择这个表达式的原因"
        }}
    ],
    "ambiguity_analysis": "歧义分析过程"
}}
```
        """
        
        # 根据角色添加特定指导
        role_guidance = {
            AgentRole.LOGICIAN: "\n**Role Guidance:** 请特别关注逻辑结构和形式化分析，确保MTL表达式的严谨性和完整性。",
            AgentRole.CREATIVE: "\n**Role Guidance:** 请发挥创造性思维，考虑各种可能的时间逻辑理解方式，包括隐含的时序关系。",
            AgentRole.PRAGMATIC: "\n**Role Guidance:** 请基于实际应用场景和常识，提供最符合实际使用的MTL表达式。"
        }
        
        return prompt + ambiguity_instruction + role_guidance[role]
    
    def detect_ambiguity(self, sentence: str, context: str = "") -> DetectionResult:
        """主检测流程 - 实现完整的五阶段检测"""
        logger.info(f"开始检测句子歧义: {sentence}")
        
        process_log = []
        phase_results = {}
        
        try:
            # 准备阶段：RAG增强（如果启用）
            if self.rag_enabled and self.rag_module:
                rag_context = self.rag_module.retrieve_relevant_knowledge(sentence)
                if isinstance(rag_context, list):
                    rag_text = "\n".join([str(item) for item in rag_context])
                else:
                    rag_text = str(rag_context)
                context = f"{context}\n\nRAG Retrieved Context:\n{rag_text}" if context else rag_text
                process_log.append("RAG模块增强完成")
            
            # 第一阶段：独立解释生成
            initial_answers = self._generate_initial_answers(sentence, context)
            phase_results["phase_1_initial_answers"] = initial_answers
            process_log.append("第一阶段：独立解释生成完成")
            
            # 检查初始一致性
            consistency_result = self._analyze_initial_consistency(initial_answers)
            
            if consistency_result["status"] == "consensus":
                # 情况1：所有Agent生成一种且语义相同的解释
                return DetectionResult(
                    is_ambiguous=False,
                    final_answer=consistency_result["consensus_answer"],
                    confidence=0.95,
                    process_log=process_log + ["第一阶段达成完全共识"],
                    phase_results=phase_results,
                    consensus_reached_at="phase_1",
                    mtl_expressions=consistency_result.get("mtl_expressions", [])
                )
            
            elif consistency_result["status"] == "complete_disagreement":
                # 情况2：三个Agent结果均不相同
                return DetectionResult(
                    is_ambiguous=True,
                    final_answer="检测到高度歧义，需要用户干预",
                    confidence=0.85,
                    process_log=process_log + ["三个Agent完全不同意，需要人工干预"],
                    phase_results=phase_results,
                    mtl_expressions=consistency_result.get("all_expressions", [])
                )
            
            # 情况3：存在部分共识，进入后续阶段
            process_log.append(f"检测到部分分歧，进入多阶段处理")
            
            # 根据配置决定处理顺序
            if self.process_order == ProcessOrder.CONSENSUS_FIRST:
                return self._consensus_first_process(sentence, initial_answers, process_log, phase_results)
            else:
                return self._voting_first_process(sentence, initial_answers, process_log, phase_results)
                
        except Exception as e:
            logger.error(f"检测过程出错: {e}")
            return DetectionResult(
                is_ambiguous=True,
                final_answer=f"检测过程出错: {str(e)}",
                confidence=0.0,
                process_log=process_log + [f"错误: {str(e)}"],
                phase_results=phase_results
            )

    def _analyze_initial_consistency(self, initial_answers: Dict[str, Dict]) -> Dict[str, Any]:
        """分析初始答案的一致性"""
        # 提取所有MTL表达式
        all_expressions = []
        agent_expressions = {}
        
        for agent_name, answer in initial_answers.items():
            structured = answer["structured"]
            expressions = structured.get("mtl_expressions", [])
            agent_expressions[agent_name] = expressions
            all_expressions.extend([expr["expression"] for expr in expressions])
        
        # 如果所有agent都只生成了一个表达式且相似度高
        if all(len(expressions) == 1 for expressions in agent_expressions.values()):
            expressions_list = [expressions[0]["expression"] for expressions in agent_expressions.values()]
            if len(expressions_list) == 3:
                sim1 = self._calculate_semantic_similarity(expressions_list[0], expressions_list[1])
                sim2 = self._calculate_semantic_similarity(expressions_list[0], expressions_list[2])
                sim3 = self._calculate_semantic_similarity(expressions_list[1], expressions_list[2])
                
                threshold = self.config.get("similarity_threshold", 0.95)
                if sim1 >= threshold and sim2 >= threshold and sim3 >= threshold:
                    return {
                        "status": "consensus",
                        "consensus_answer": expressions_list[0],
                        "mtl_expressions": [expressions_list[0]]
                    }
        
        # 检查是否完全不同
        unique_expressions = set(all_expressions)
        if len(unique_expressions) == len(all_expressions) and len(all_expressions) >= 3:
            return {
                "status": "complete_disagreement",
                "all_expressions": list(unique_expressions)
            }
        
        # 部分分歧
        return {
            "status": "partial_disagreement",
            "agent_expressions": agent_expressions,
            "all_expressions": all_expressions
        }
    
    def _consensus_first_process(self, sentence: str, initial_answers: Dict, 
                               process_log: List[str], phase_results: Dict) -> DetectionResult:
        """先共识讨论，后投票的处理流程"""
        # 第二阶段：多轮共识讨论
        consensus_result = self._conduct_consensus_discussion(sentence, initial_answers)
        phase_results["phase_2_consensus"] = consensus_result
        process_log.append("第二阶段：共识讨论完成")
        
        if consensus_result["consensus_reached"]:
            return DetectionResult(
                is_ambiguous=consensus_result.get("is_ambiguous", True),
                final_answer=consensus_result["final_answer"],
                confidence=consensus_result.get("confidence", 0.8),
                process_log=process_log + ["共识讨论阶段达成一致"],
                phase_results=phase_results,
                consensus_reached_at="phase_2",
                mtl_expressions=consensus_result.get("mtl_expressions", [])
            )
        
        # 第三阶段：投票仲裁
        voting_result = self._conduct_voting(sentence, initial_answers, consensus_result)
        phase_results["phase_3_voting"] = voting_result
        process_log.append("第三阶段：投票仲裁完成")
        
        if voting_result["winner"]:
            return DetectionResult(
                is_ambiguous=True,
                final_answer=voting_result["winner_answer"],
                confidence=voting_result.get("confidence", 0.7),
                process_log=process_log + [f"投票获胜者: {voting_result['winner']}"],
                phase_results=phase_results,
                voting_results=voting_result,
                mtl_expressions=voting_result.get("mtl_expressions", [])
            )
        
        # 第四阶段：专家仲裁
        return self._expert_arbitration(sentence, initial_answers, consensus_result, 
                                      voting_result, process_log, phase_results)
    
    def _voting_first_process(self, sentence: str, initial_answers: Dict,
                            process_log: List[str], phase_results: Dict) -> DetectionResult:
        """先投票，后共识讨论的处理流程"""
        # 第二阶段：投票仲裁
        voting_result = self._conduct_voting(sentence, initial_answers)
        phase_results["phase_2_voting"] = voting_result
        process_log.append("第二阶段：投票仲裁完成")
        
        if voting_result["winner"]:
            return DetectionResult(
                is_ambiguous=True,
                final_answer=voting_result["winner_answer"],
                confidence=voting_result.get("confidence", 0.7),
                process_log=process_log + [f"投票获胜者: {voting_result['winner']}"],
                phase_results=phase_results,
                voting_results=voting_result,
                mtl_expressions=voting_result.get("mtl_expressions", [])
            )
        
        # 第三阶段：共识讨论
        consensus_result = self._conduct_consensus_discussion(sentence, initial_answers)
        phase_results["phase_3_consensus"] = consensus_result  
        process_log.append("第三阶段：共识讨论完成")
        
        if consensus_result["consensus_reached"]:
            return DetectionResult(
                is_ambiguous=consensus_result.get("is_ambiguous", True),
                final_answer=consensus_result["final_answer"],
                confidence=consensus_result.get("confidence", 0.8),
                process_log=process_log + ["共识讨论阶段达成一致"],
                phase_results=phase_results,
                consensus_reached_at="phase_3",
                mtl_expressions=consensus_result.get("mtl_expressions", [])
            )
        
        # 第四阶段：专家仲裁
        return self._expert_arbitration(sentence, initial_answers, consensus_result,
                                      voting_result, process_log, phase_results)
    
    def _conduct_consensus_discussion(self, sentence: str, initial_answers: Dict) -> Dict:
        """进行多轮共识讨论"""
        logger.info("开始多轮共识讨论")
        
        max_rounds = self.config.get("max_discussion_rounds", 3)
        discussion_history = []
        
        # 构建讨论prompt
        discussion_prompt = self._build_discussion_prompt(sentence, initial_answers)
        
        for round_num in range(max_rounds):
            logger.info(f"讨论轮次 {round_num + 1}")
            round_responses = {}
            
            for agent in self.agents:
                messages = [
                    {"role": "system", "content": agent.system_prompt},
                    {"role": "user", "content": discussion_prompt}
                ]
                
                if discussion_history:
                    # 添加之前的讨论历史
                    history_text = "\n".join(discussion_history)
                    messages.append({"role": "user", "content": f"之前的讨论:\n{history_text}\n\n请基于讨论历史继续分析："})
                
                response = self._call_llm(agent.name, messages)
                round_responses[agent.name] = response
            
            # 分析本轮是否达成共识
            consensus_check = self._check_discussion_consensus(round_responses)
            if consensus_check["consensus_reached"]:
                return consensus_check
            
            # 记录讨论历史
            discussion_history.append(f"第{round_num + 1}轮讨论结果:\n" + 
                                    "\n".join([f"{name}: {resp[:200]}..." for name, resp in round_responses.items()]))
        
        # 未达成共识
        return {"consensus_reached": False, "discussion_history": discussion_history}
    
    def _conduct_voting(self, sentence: str, initial_answers: Dict, 
                       consensus_result: Optional[Dict] = None) -> Dict:
        """进行投票仲裁"""
        logger.info("开始投票仲裁")
        
        # 收集所有候选答案
        candidates = {}
        for agent_name, answer in initial_answers.items():
            structured = answer["structured"]
            for i, expr in enumerate(structured.get("mtl_expressions", [])):
                candidate_id = f"{agent_name}_expr_{i}"
                candidates[candidate_id] = expr["expression"]
        
        # 构建投票prompt
        voting_prompt = self._build_voting_prompt(sentence, candidates)
        
        # 收集投票
        votes = {}
        for agent in self.agents:
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": voting_prompt}
            ]
            
            response = self._call_llm(agent.name, messages)
            vote = self._extract_vote(response, candidates)
            votes[agent.name] = vote
        
        # 统计投票结果
        vote_counts = {}
        for vote in votes.values():
            if vote:
                vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        # 确定获胜者
        if vote_counts:
            max_votes = max(vote_counts.values())
            winners = [candidate for candidate, count in vote_counts.items() if count == max_votes]
            
            if len(winners) == 1 and max_votes >= 2:
                return {
                    "winner": winners[0],
                    "winner_answer": candidates[winners[0]],
                    "vote_counts": vote_counts,
                    "votes": votes,
                    "confidence": max_votes / len(self.agents),
                    "mtl_expressions": [candidates[winners[0]]]
                }
        
        return {"winner": None, "vote_counts": vote_counts, "votes": votes}
    
    def _expert_arbitration(self, sentence: str, initial_answers: Dict,
                          consensus_result: Dict, voting_result: Dict,
                          process_log: List[str], phase_results: Dict) -> DetectionResult:
        """专家仲裁（第四阶段）"""
        logger.info("进入专家仲裁阶段")
        
        # 这里可以调用更高级的模型或人工标注
        # 暂时使用最佳启发式方法
        
        # 收集所有信息
        arbitration_prompt = self._build_arbitration_prompt(
            sentence, initial_answers, consensus_result, voting_result
        )
        
        # 使用最强的模型进行仲裁（假设第一个agent使用最强模型）
        expert_agent = self.agents[0]  
        messages = [
            {"role": "system", "content": "你是一个专业的语言学专家和MTL逻辑专家，负责对复杂的歧义进行最终裁决。"},
            {"role": "user", "content": arbitration_prompt}
        ]
        
        expert_response = self._call_llm(expert_agent.name, messages)
        expert_answer = self._extract_structured_answer(expert_response)
        
        phase_results["phase_4_expert"] = {
            "expert_response": expert_response,
            "expert_analysis": expert_answer
        }
        
        return DetectionResult(
            is_ambiguous=expert_answer.get("is_ambiguous", True),
            final_answer=expert_response,
            confidence=0.6,  # 专家仲裁置信度相对较低
            process_log=process_log + ["专家仲裁完成"],
            phase_results=phase_results,
            expert_arbitration=expert_response,
            mtl_expressions=expert_answer.get("mtl_expressions", [])
        )
    
    def _build_discussion_prompt(self, sentence: str, initial_answers: Dict) -> str:
        """构建讨论提示词"""
        prompt = f"""
请参与多智能体讨论，分析以下句子的MTL表达式歧义问题。

原句子: "{sentence}"

各Agent的初始分析:
"""
        for agent_name, answer in initial_answers.items():
            structured = answer["structured"]
            prompt += f"\n{agent_name}的分析:\n{json.dumps(structured, ensure_ascii=False, indent=2)}\n"
        
        prompt += """
请仔细分析所有解释的合理性，通过讨论尝试达成一个最终的、最合理的MTL表达式。

要求：
1. 评估每个MTL表达式的正确性和合理性
2. 指出可能存在的问题或歧义
3. 提出你认为最准确的MTL表达式
4. 说明你的推理过程

请以JSON格式回复：
```json
{
    "preferred_expression": "你认为最好的MTL表达式",
    "reasoning": "详细的推理过程",
    "agreement_level": 0.8,
    "concerns": "对其他表达式的担忧"
}
```
"""
        return prompt
    
    def _build_voting_prompt(self, sentence: str, candidates: Dict[str, str]) -> str:
        """构建投票提示词"""
        prompt = f"""
请对以下MTL表达式候选项进行投票，选择最能准确表示句子含义的表达式。

原句子: "{sentence}"

候选MTL表达式:
"""
        for candidate_id, expression in candidates.items():
            prompt += f"{candidate_id}: {expression}\n"
        
        prompt += """
请仔细分析每个候选表达式，并投票选择最佳答案。

请回复格式：
```json
{
    "vote": "candidate_id",
    "reasoning": "投票理由"
}
```
"""
        return prompt
    
    def _build_arbitration_prompt(self, sentence: str, initial_answers: Dict,
                                consensus_result: Dict, voting_result: Dict) -> str:
        """构建专家仲裁提示词"""
        prompt = f"""
作为MTL专家，请对以下复杂歧义案例进行最终裁决。

原句子: "{sentence}"

处理过程总结:
1. 初始分析结果: {len(initial_answers)}个不同观点
2. 共识讨论结果: {'达成共识' if consensus_result.get('consensus_reached') else '未达成共识'}  
3. 投票结果: {'有获胜者' if voting_result.get('winner') else '无明确获胜者'}

详细信息:
{json.dumps({'initial': initial_answers, 'consensus': consensus_result, 'voting': voting_result}, ensure_ascii=False, indent=2)}

请提供最终的专家判断:
1. 该句子是否确实存在歧义？
2. 最准确的MTL表达式是什么？
3. 为什么选择这个表达式？

请以详细的分析回复。
"""
        return prompt
    
    def _check_discussion_consensus(self, round_responses: Dict[str, str]) -> Dict:
        """检查讨论轮次是否达成共识"""
        # 提取每个agent的首选表达式
        preferences = {}
        for agent_name, response in round_responses.items():
            try:
                structured = self._extract_structured_answer(response)
                if "preferred_expression" in structured:
                    preferences[agent_name] = structured["preferred_expression"]
            except:
                continue
        
        if len(preferences) >= 2:
            expressions = list(preferences.values())
            # 计算表达式相似度
            similarities = []
            for i in range(len(expressions)):
                for j in range(i+1, len(expressions)):
                    sim = self._calculate_semantic_similarity(expressions[i], expressions[j])
                    similarities.append(sim)
            
            if similarities and sum(similarities) / len(similarities) > 0.9:
                return {
                    "consensus_reached": True,
                    "final_answer": expressions[0],
                    "confidence": 0.8,
                    "mtl_expressions": [expressions[0]]
                }
        
        return {"consensus_reached": False}
    
    def _extract_vote(self, response: str, candidates: Dict[str, str]) -> Optional[str]:
        """从回复中提取投票"""
        try:
            structured = self._extract_structured_answer(response)
            vote = structured.get("vote")
            if vote in candidates:
                return vote
        except:
            pass
        
        # 尝试文本匹配
        for candidate_id in candidates.keys():
            if candidate_id in response:
                return candidate_id
        
        return None

def run_batch_detection(input_file: str, output_file: str, config_path: Optional[str] = None):
    """批量检测功能"""
    detector = AmbiguityDetector(config_path) if config_path else AmbiguityDetector()
    
    # 读取输入数据
    df = pd.read_excel(input_file)
    sentences = df["sentence"].tolist() if "sentence" in df.columns else df.iloc[:, 0].tolist()
    
    results = []
    
    for i, sentence in enumerate(sentences):
        logger.info(f"处理第 {i+1}/{len(sentences)} 个句子")
        
        result = detector.detect_ambiguity(sentence)
        
        results.append({
            "sentence": sentence,
            "is_ambiguous": result.is_ambiguous,
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "consensus_reached_at": result.consensus_reached_at,
            "process_log": "; ".join(result.process_log)
        })
        
        # 每10个样本保存一次
        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_excel(output_file, index=False)
            logger.info(f"已保存 {len(results)} 个结果")
    
    # 最终保存
    pd.DataFrame(results).to_excel(output_file, index=False)
    logger.info(f"批量检测完成，结果保存到: {output_file}")

if __name__ == "__main__":
    # 示例用法
    detector = AmbiguityDetector()
    
    # 测试句子
    test_sentences = [
        "他用望远镜看到了那个人。",
        "银行在河边。",
        "今天天气很好。",
        "Flying planes can be dangerous."
    ]
    
    for sentence in test_sentences:
        print(f"\n{'='*50}")
        print(f"测试句子: {sentence}")
        print('='*50)
        
        result = detector.detect_ambiguity(sentence)
        
        print(f"是否有歧义: {result.is_ambiguous}")
        print(f"最终答案: {result.final_answer}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"处理日志: {'; '.join(result.process_log)}")