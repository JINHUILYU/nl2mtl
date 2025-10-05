"""
功能流程：
1. 输入自然语言句子
2. 与examples中的句子计算相似度，提取最相似的5个作为示例
3. 四阶段处理：
   - 第一阶段：三个Agent独立分析
   - 第二阶段：Agent相互讨论（最多5轮）
   - 第三阶段：Agent投票
   - 第四阶段：仲裁Agent选择最佳结果
4. 统计全程token消耗
"""

import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv
import logging
import os
import re
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HumanDecision(Enum):
    """人工决策选项"""
    CONTINUE = "continue"
    TERMINATE = "terminate"

@dataclass
class TokenUsage:
    """Token使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class StageResult:
    """阶段结果"""
    stage_name: str
    agent_responses: Dict[str, str]
    token_usage: TokenUsage
    processing_time: float
    human_decision: Optional[HumanDecision] = None
    candidate_details: Optional[Dict[str, Dict[str, str]]] = None

@dataclass
class ProcessResult:
    """处理结果"""
    input_sentence: str
    final_mtl_expression: Optional[str]
    total_token_usage: TokenUsage
    total_processing_time: float
    stage_results: List[StageResult]
    termination_reason: str

class SimplifiedNL2MTL:
    """简化版NL2MTL处理器"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """初始化处理器"""
        self.config = self._load_config(config_path)
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.clients = self._initialize_clients()
        self.agents = self._initialize_agents()
        self.examples_data = self._load_examples()
        self.base_prompt = self._load_base_prompt()
        self.total_token_usage = TokenUsage()
        
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
                    "model": "deepseek-chat",
                    "temperature": 0.7,
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url_env": "DEEPSEEK_API_URL"
                },
                {
                    "name": "Agent_B", 
                    "role": "creative",
                    "model": "deepseek-chat",
                    "temperature": 0.7,
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url_env": "DEEPSEEK_API_URL"
                },
                {
                    "name": "Agent_C",
                    "role": "pragmatic", 
                    "model": "deepseek-chat",
                    "temperature": 0.7,
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url_env": "DEEPSEEK_API_URL"
                }
            ]
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
    
    def _initialize_agents(self) -> List[Dict]:
        """初始化Agent配置"""
        agents = []
        role_prompts = {
            "logician": "你是一个严谨的逻辑学家，用严格的逻辑推理分析问题。",
            "creative": "你是一个创造性思考者，考虑多种解释和可能性。",
            "pragmatic": "你是一个实用主义分析师，专注于常识性解释。"
        }
        
        for agent_config in self.config["agents"]:
            agents.append({
                "name": agent_config["name"],
                "role": agent_config["role"],
                "model": agent_config["model"],
                "temperature": agent_config["temperature"],
                "system_prompt": role_prompts.get(agent_config["role"], "你是一个有用的助手。")
            })
            
        return agents
    
    def _load_examples(self) -> pd.DataFrame:
        """加载示例数据"""
        try:
            return pd.read_excel("data/input/examples.xlsx")
        except FileNotFoundError:
            logger.error("示例文件未找到: data/input/examples.xlsx")
            return pd.DataFrame(columns=["Natural language traffic rule", "Answer"])
    
    def _load_base_prompt(self) -> str:
        """加载基础prompt模板"""
        try:
            with open("config/base_prompt.txt", 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("基础prompt文件未找到，使用默认prompt")
            return """请分析以下句子并生成对应的MTL表达式。

句子: [INPUT TEXT]

示例:
[EXAMPLE]

请提供详细的分析过程和最终的MTL表达式。"""
    
    def _calculate_similarity(self, sentence: str, examples: List[str]) -> List[float]:
        """计算句子与示例的相似度"""
        try:
            # 编码输入句子和所有示例
            sentence_embedding = self.sentence_model.encode([sentence], convert_to_tensor=True)
            example_embeddings = self.sentence_model.encode(examples, convert_to_tensor=True)
            
            # 计算余弦相似度
            similarities = util.pytorch_cos_sim(sentence_embedding, example_embeddings)[0]
            return similarities.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return [0.0] * len(examples)
    
    def _get_top_examples(self, sentence: str, top_k: int = 5) -> str:
        """获取最相似的top-k个示例"""
        if self.examples_data.empty:
            return "无可用示例"
        
        examples = self.examples_data["Natural language traffic rule"].tolist()
        answers = self.examples_data["Answer"].tolist()
        
        similarities = self._calculate_similarity(sentence, examples)
        
        # 获取top-k索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 构建示例文本
        example_text = ""
        for i, idx in enumerate(top_indices, 1):
            example_text += f"示例{i}:\n"
            example_text += f"Input Text: {examples[idx]}\n"
            example_text += f"Answer: {answers[idx]}\n\n"
        
        return example_text
    
    def _call_llm(self, agent_name: str, messages: List[Dict]) -> Tuple[str, TokenUsage]:
        """调用LLM并追踪token使用"""
        if agent_name not in self.clients:
            raise ValueError(f"客户端未找到: {agent_name}")
            
        client = self.clients[agent_name]
        agent = next(a for a in self.agents if a["name"] == agent_name)
        
        try:
            response = client.chat.completions.create(
                model=agent["model"],
                messages=messages,  # type: ignore
                temperature=agent["temperature"]
            )
            
            # 追踪token使用
            token_usage = TokenUsage()
            if hasattr(response, 'usage') and response.usage:
                token_usage.prompt_tokens = response.usage.prompt_tokens
                token_usage.completion_tokens = response.usage.completion_tokens
                token_usage.total_tokens = response.usage.total_tokens
            
            # 累计到总使用量
            self.total_token_usage.prompt_tokens += token_usage.prompt_tokens
            self.total_token_usage.completion_tokens += token_usage.completion_tokens
            self.total_token_usage.total_tokens += token_usage.total_tokens
            
            content = response.choices[0].message.content
            return content.strip() if content else "", token_usage
            
        except Exception as e:
            logger.error(f"LLM调用失败 {agent_name}: {e}")
            return "", TokenUsage()
    
    def _extract_mtl_expression(self, response: str) -> str:
        """从回答中提取MTL表达式"""
        # 尝试提取```包围的MTL表达式
        mtl_match = re.search(r'```(.*?)```', response, re.DOTALL)
        if mtl_match:
            return mtl_match.group(1).strip()
        
        # 尝试提取MTL translation: 后面的内容
        mtl_match = re.search(r'MTL translation:\s*(.+)', response, re.IGNORECASE)
        if mtl_match:
            return mtl_match.group(1).strip()
        
        # 尝试提取包含G(、F_、P_等MTL操作符的行
        lines = response.split('\n')
        for line in lines:
            if any(op in line for op in ['G(', 'F_[', 'P_[', 'U_[', 'X(']):
                return line.strip()
        
        return "未找到MTL表达式"

    def _extract_vote_info(self, response: str) -> Dict[str, str]:
        """从投票回答中提取投票信息"""
        vote_info = {"vote": "未识别", "reason": "未提供理由"}
        
        # 尝试提取投票选择
        vote_patterns = [
            r'我投票给[：:]?\s*([^，,\n]+)',
            r'投票[：:]?\s*([^，,\n]+)',
            r'选择[：:]?\s*([^，,\n]+)',
            r'候选\d+',
        ]
        
        for pattern in vote_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                vote_info["vote"] = match.group(1).strip() if match.groups() else match.group(0).strip()
                break
        
        # 尝试提取投票理由
        reason_patterns = [
            r'理由[：:]?\s*([^。\n]+)',
            r'因为\s*([^。\n]+)',
            r'原因[：:]?\s*([^。\n]+)',
        ]
        
        for pattern in reason_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                vote_info["reason"] = match.group(1).strip()
                break
        
        return vote_info

    def _display_voting_details(self, agent_responses: Dict[str, str]) -> None:
        """显示投票详细信息"""
        print("\n🗳️  投票详情:")
        print("-" * 50)
        
        vote_summary = {}
        for agent_name, response in agent_responses.items():
            vote_info = self._extract_vote_info(response)
            vote_choice = vote_info["vote"]
            vote_reason = vote_info["reason"]
            
            print(f"\n【{agent_name}】:")
            print(f"🎯 投票选择: {vote_choice}")
            print(f"💭 投票理由: {vote_reason}")
            
            # 统计投票
            if vote_choice in vote_summary:
                vote_summary[vote_choice] += 1
            else:
                vote_summary[vote_choice] = 1
            
            print("-" * 30)
        
        # 显示投票统计
        if vote_summary:
            print(f"\n📊 投票统计:")
            for choice, count in sorted(vote_summary.items(), key=lambda x: x[1], reverse=True):
                print(f"   {choice}: {count} 票")
    
    def _display_discussion_summary(self, agent_responses: Dict[str, str]) -> None:
        """显示讨论阶段摘要"""
        print("\n💬 讨论结果摘要:")
        print("-" * 50)
        
        mtl_expressions = []
        for agent_name, response in agent_responses.items():
            mtl_expr = self._extract_mtl_expression(response)
            mtl_expressions.append(mtl_expr)
            
            print(f"\n【{agent_name}】:")
            print(f"🎯 最终MTL表达式: {mtl_expr}")
            
            # 显示讨论要点（前200字符）
            print(f"💭 讨论要点:")
            if len(response) > 200:
                print(f"{response[:200]}...")
            else:
                print(response)
            print("-" * 30)
        
        # 检查是否达成共识
        unique_expressions = set(expr for expr in mtl_expressions if expr != "未找到MTL表达式")
        if len(unique_expressions) == 1 and unique_expressions:
            print(f"\n✅ 讨论达成共识: {list(unique_expressions)[0]}")
        elif len(unique_expressions) > 1:
            print(f"\n⚠️  仍存在分歧，共有 {len(unique_expressions)} 种不同观点")
        else:
            print(f"\n❌ 未能提取到有效的MTL表达式")
    
    def _request_human_decision(self, stage_name: str, results: Dict[str, Any],
                               agent_responses: Optional[Dict[str, str]] = None,
                               stage_result: Optional[StageResult] = None) -> HumanDecision:
        """请求人工决策"""
        print(f"\n{'='*60}")
        print(f"🤔 人工决策请求 - {stage_name}")
        print(f"{'='*60}")
        
        # 显示Agent的具体回答
        if agent_responses:
            if "投票" in stage_name:
                # 投票阶段先显示候选项详情，再显示投票详情
                if stage_result and hasattr(stage_result, 'candidate_details') and stage_result.candidate_details:
                    print("\n📋 投票候选项:")
                    print("-" * 50)
                    for cand_id, details in stage_result.candidate_details.items():
                        print(f"\n【{cand_id}】:")
                        print(f"🏷️  来源: {details['source']}")
                        print(f"🎯 MTL表达式: {details['mtl_expression']}")
                        print(f"📝 分析摘要: {details['summary']}")
                        print("-" * 30)
                
                # 然后显示投票详情
                self._display_voting_details(agent_responses)
            elif "讨论" in stage_name:
                # 讨论阶段显示讨论摘要
                self._display_discussion_summary(agent_responses)
            else:
                # 其他阶段显示常规信息
                print("🤖 各Agent的回答:")
                print("-" * 50)
                for agent_name, response in agent_responses.items():
                    print(f"\n【{agent_name}】:")
                    
                    # 提取并显示MTL表达式
                    mtl_expr = self._extract_mtl_expression(response)
                    print(f"🎯 MTL表达式: {mtl_expr}")
                    
                    # 显示回答摘要（前300字符）
                    print(f"📝 分析过程:")
                    if len(response) > 300:
                        print(f"{response[:300]}...")
                    else:
                        print(response)
                    
                    print(f"📏 完整回答长度: {len(response)} 字符")
                    print("-" * 30)
        
        # 显示结果摘要
        print("\n📊 阶段统计信息:")
        for key, value in results.items():
            if key != "agent_responses":  # 避免重复显示
                print(f"   {key}: {value}")
        
        print(f"\n💡 操作选项:")
        print(f"1. 继续到下一阶段 (continue)")
        print(f"2. 终止处理 (terminate)")
        print(f"3. 查看完整回答 (view)")
        if "投票" in stage_name and stage_result and hasattr(stage_result, 'candidate_details'):
            print(f"4. 查看候选项详情 (candidates)")
        
        max_choice = 4 if ("投票" in stage_name and stage_result and hasattr(stage_result, 'candidate_details')) else 3
        
        while True:
            try:
                choice = input(f"\n请输入选择 (1-{max_choice}): ").strip()
                
                if choice == "1":
                    return HumanDecision.CONTINUE
                elif choice == "2":
                    return HumanDecision.TERMINATE
                elif choice == "3":
                    # 显示完整回答
                    if agent_responses:
                        print(f"\n{'='*60}")
                        print("📖 完整回答详情")
                        print(f"{'='*60}")
                        for agent_name, response in agent_responses.items():
                            print(f"\n【{agent_name} - 完整回答】:")
                            print(response)
                            print("-" * 50)
                        print("\n返回选择菜单...")
                        continue
                    else:
                        print("没有可显示的回答")
                        continue
                elif choice == "4" and "投票" in stage_name and stage_result and hasattr(stage_result, 'candidate_details'):
                    # 显示候选项详情（仅投票阶段）
                    self._show_voting_candidates(stage_result)
                    continue
                else:
                    print(f"无效选择，请输入 1-{max_choice}!")
                    continue
                    
            except KeyboardInterrupt:
                print("\n用户中断，默认终止处理")
                return HumanDecision.TERMINATE
            except Exception as e:
                print(f"输入错误: {e}")
                continue

    def _show_voting_candidates(self, stage_result: StageResult) -> None:
        """显示投票候选项详情"""
        print(f"\n{'='*60}")
        print("🗳️  候选项详情")
        print(f"{'='*60}")
        
        # 从stage_result中获取候选项详情
        if hasattr(stage_result, 'candidate_details') and stage_result.candidate_details:
            candidate_details = stage_result.candidate_details
            print("\n📋 所有候选项详细信息:")
            
            for cand_id, details in candidate_details.items():
                print(f"\n【{cand_id}】:")
                print(f"🏷️  来源: {details['source']}")
                print(f"🎯 MTL表达式: {details['mtl_expression']}")
                print(f"📝 分析摘要:")
                print(f"   {details['summary']}")
                print("-" * 50)
                
                # 询问是否查看完整回答
                view_full = input(f"是否查看{cand_id}的完整分析？(y/n): ").strip().lower()
                if view_full in ['y', 'yes', '是']:
                    print(f"\n📖 【{cand_id}】完整分析:")
                    print(details['full_response'])
                    print("-" * 50)
        else:
            print("\n⚠️  候选项详情信息未找到")
            print("这可能是因为投票阶段还未执行或数据结构发生变化")
        
        print("\n返回选择菜单...")
    
    def _stage_1_independent_analysis(self, sentence: str, examples: str) -> StageResult:
        """第一阶段：独立分析"""
        start_time = time.time()
        logger.info("=== 第一阶段：独立分析 ===")
        
        # 构建prompt
        prompt = self.base_prompt.replace("[INPUT TEXT]", sentence)
        prompt = prompt.replace("[EXAMPLE]", examples)
        
        agent_responses = {}
        stage_token_usage = TokenUsage()
        
        for agent in self.agents:
            messages = [
                {"role": "system", "content": agent["system_prompt"]},
                {"role": "user", "content": prompt}
            ]
            
            try:
                response, token_usage = self._call_llm(agent["name"], messages)
                agent_responses[agent["name"]] = response
                
                # 累计token使用
                stage_token_usage.prompt_tokens += token_usage.prompt_tokens
                stage_token_usage.completion_tokens += token_usage.completion_tokens
                stage_token_usage.total_tokens += token_usage.total_tokens
                
                logger.info(f"{agent['name']} ({agent['role']}) 分析完成")
                
            except Exception as e:
                logger.error(f"Agent {agent['name']} 分析失败: {e}")
                agent_responses[agent["name"]] = f"分析失败: {str(e)}"
        
        processing_time = time.time() - start_time
        
        return StageResult(
            stage_name="第一阶段：独立分析",
            agent_responses=agent_responses,
            token_usage=stage_token_usage,
            processing_time=processing_time
        )
    
    def _stage_2_discussion(self, sentence: str, stage1_result: StageResult) -> StageResult:
        """第二阶段：讨论（最多5轮，每轮后人工决策）"""
        start_time = time.time()
        logger.info("=== 第二阶段：讨论 ===")
        
        max_rounds = 5
        discussion_history = []
        stage_token_usage = TokenUsage()
        final_responses = {}
        
        # 构建讨论prompt
        discussion_prompt = f"""
请参与多智能体讨论，分析以下句子的MTL表达式。

原句子: "{sentence}"

各Agent的初始分析:
"""
        for agent_name, response in stage1_result.agent_responses.items():
            discussion_prompt += f"\n{agent_name}: {response}\n"
        
        discussion_prompt += """
请仔细分析所有解释的合理性，通过讨论尝试达成一个最终的、最合理的MTL表达式。
请提供你认为最准确的MTL表达式和推理过程。
"""
        
        for round_num in range(max_rounds):
            logger.info(f"讨论轮次 {round_num + 1}")
            round_responses = {}
            round_token_usage = TokenUsage()
            
            for agent in self.agents:
                messages = [
                    {"role": "system", "content": agent["system_prompt"]},
                    {"role": "user", "content": discussion_prompt}
                ]
                
                if discussion_history:
                    history_text = "\n".join(discussion_history)
                    messages.append({"role": "user", "content": f"之前的讨论:\n{history_text}\n\n请基于讨论历史继续分析："})
                
                try:
                    response, token_usage = self._call_llm(agent["name"], messages)
                    round_responses[agent["name"]] = response
                    
                    # 累计token使用
                    round_token_usage.prompt_tokens += token_usage.prompt_tokens
                    round_token_usage.completion_tokens += token_usage.completion_tokens
                    round_token_usage.total_tokens += token_usage.total_tokens
                    
                except Exception as e:
                    logger.error(f"Agent {agent['name']} 讨论失败: {e}")
                    round_responses[agent["name"]] = f"讨论失败: {str(e)}"
            
            # 累计到总token使用
            stage_token_usage.prompt_tokens += round_token_usage.prompt_tokens
            stage_token_usage.completion_tokens += round_token_usage.completion_tokens
            stage_token_usage.total_tokens += round_token_usage.total_tokens
            
            # 显示本轮讨论结果并请求人工决策
            print(f"\n{'='*60}")
            print(f"📝 第{round_num + 1}轮讨论结果")
            print(f"{'='*60}")
            
            # 显示每个Agent的结论
            self._display_discussion_summary(round_responses)
            
            # 记录讨论历史
            discussion_history.append(f"第{round_num + 1}轮讨论:\n" +
                                    "\n".join([f"{name}: {resp[:200]}..." for name, resp in round_responses.items()]))
            
            final_responses = round_responses
            
            # 如果不是最后一轮，询问是否继续
            if round_num < max_rounds - 1:
                decision = self._request_human_decision(f"第{round_num + 1}轮讨论完成", {
                    "当前轮次": f"{round_num + 1}/{max_rounds}",
                    "本轮处理时间": f"{(time.time() - start_time):.2f}秒",
                    "本轮Token使用": round_token_usage.total_tokens,
                    "累计Token使用": stage_token_usage.total_tokens
                }, round_responses)
                
                if decision == HumanDecision.TERMINATE:
                    logger.info(f"人工决策：在第{round_num + 1}轮后终止讨论")
                    break
                else:
                    logger.info(f"人工决策：继续第{round_num + 2}轮讨论")
            else:
                logger.info("已完成最大讨论轮次")
        
        processing_time = time.time() - start_time
        
        return StageResult(
            stage_name="第二阶段：讨论",
            agent_responses=final_responses,
            token_usage=stage_token_usage,
            processing_time=processing_time
        )
    
    def _stage_3_voting(self, sentence: str, stage1_result: StageResult, stage2_result: StageResult) -> StageResult:
        """第三阶段：投票"""
        start_time = time.time()
        logger.info("=== 第三阶段：投票 ===")
        
        # 收集所有候选答案（包含完整信息）
        candidates = {}
        candidate_details = {}
        candidate_id = 1
        
        # 从第一阶段收集候选答案
        for agent_name, response in stage1_result.agent_responses.items():
            cand_key = f"候选{candidate_id}"
            mtl_expr = self._extract_mtl_expression(response)
            candidates[cand_key] = f"{agent_name}的独立分析: {mtl_expr}"
            candidate_details[cand_key] = {
                "source": f"第一阶段 - {agent_name}",
                "mtl_expression": mtl_expr,
                "full_response": response,
                "summary": response[:300] + "..." if len(response) > 300 else response
            }
            candidate_id += 1
        
        # 从第二阶段收集候选答案
        for agent_name, response in stage2_result.agent_responses.items():
            cand_key = f"候选{candidate_id}"
            mtl_expr = self._extract_mtl_expression(response)
            candidates[cand_key] = f"{agent_name}的讨论结果: {mtl_expr}"
            candidate_details[cand_key] = {
                "source": f"第二阶段 - {agent_name}",
                "mtl_expression": mtl_expr,
                "full_response": response,
                "summary": response[:300] + "..." if len(response) > 300 else response
            }
            candidate_id += 1
        
        # 构建投票prompt
        voting_prompt = f"""
请对以下MTL表达式候选项进行投票，选择最能准确表示句子含义的表达式。

原句子: "{sentence}"

候选答案:
"""
        for cand_id, cand_text in candidates.items():
            voting_prompt += f"{cand_id}: {cand_text}\n"
        
        voting_prompt += """
请仔细分析每个候选表达式，并投票选择最佳答案。
请回复格式：我投票给：[候选ID]，理由：[投票理由]
"""
        
        agent_responses = {}
        stage_token_usage = TokenUsage()
        
        for agent in self.agents:
            messages = [
                {"role": "system", "content": agent["system_prompt"]},
                {"role": "user", "content": voting_prompt}
            ]
            
            try:
                response, token_usage = self._call_llm(agent["name"], messages)
                agent_responses[agent["name"]] = response
                
                # 累计token使用
                stage_token_usage.prompt_tokens += token_usage.prompt_tokens
                stage_token_usage.completion_tokens += token_usage.completion_tokens
                stage_token_usage.total_tokens += token_usage.total_tokens
                
                logger.info(f"{agent['name']} 投票完成")
                
            except Exception as e:
                logger.error(f"Agent {agent['name']} 投票失败: {e}")
                agent_responses[agent["name"]] = f"投票失败: {str(e)}"
        
        processing_time = time.time() - start_time
        
        # 创建包含候选项详情的结果
        result = StageResult(
            stage_name="第三阶段：投票",
            agent_responses=agent_responses,
            token_usage=stage_token_usage,
            processing_time=processing_time
        )
        
        # 将候选项详情添加到结果中（通过扩展属性）
        result.candidate_details = candidate_details
        
        return result
    
    def _stage_4_arbitration(self, sentence: str, stage1_result: StageResult, 
                           stage2_result: StageResult, stage3_result: StageResult) -> StageResult:
        """第四阶段：仲裁"""
        start_time = time.time()
        logger.info("=== 第四阶段：仲裁 ===")
        
        # 构建仲裁prompt
        arbitration_prompt = f"""
作为MTL专家，请对以下复杂案例进行最终裁决。

原句子: "{sentence}"

处理过程总结:
第一阶段 - 独立分析结果:
"""
        for agent_name, response in stage1_result.agent_responses.items():
            arbitration_prompt += f"{agent_name}: {response[:300]}...\n"
        
        arbitration_prompt += "\n第二阶段 - 讨论结果:\n"
        for agent_name, response in stage2_result.agent_responses.items():
            arbitration_prompt += f"{agent_name}: {response[:300]}...\n"
        
        arbitration_prompt += "\n第三阶段 - 投票结果:\n"
        for agent_name, response in stage3_result.agent_responses.items():
            arbitration_prompt += f"{agent_name}: {response[:300]}...\n"
        
        arbitration_prompt += """

请提供最终的专家判断:
1. 该句子的最准确的MTL表达式是什么？
2. 为什么选择这个表达式？

请提供详细的分析和最终的MTL表达式。
"""
        
        # 使用第一个agent作为仲裁者
        arbitrator = self.agents[0]
        messages = [
            {"role": "system", "content": "你是一个专业的MTL逻辑专家，负责对复杂的案例进行最终裁决。"},
            {"role": "user", "content": arbitration_prompt}
        ]
        
        stage_token_usage = TokenUsage()
        
        try:
            response, token_usage = self._call_llm(arbitrator["name"], messages)
            agent_responses = {f"仲裁者_{arbitrator['name']}": response}
            
            stage_token_usage = token_usage
            logger.info("仲裁完成")
            
        except Exception as e:
            logger.error(f"仲裁失败: {e}")
            agent_responses = {"仲裁者": f"仲裁失败: {str(e)}"}
        
        processing_time = time.time() - start_time
        
        return StageResult(
            stage_name="第四阶段：仲裁",
            agent_responses=agent_responses,
            token_usage=stage_token_usage,
            processing_time=processing_time
        )
    
    def process(self, sentence: str) -> ProcessResult:
        """完整处理流程"""
        start_time = time.time()
        logger.info(f"开始处理句子: {sentence}")
        
        # 重置token统计
        self.total_token_usage = TokenUsage()
        stage_results = []
        final_mtl_expression = None
        termination_reason = "完成所有阶段"
        
        try:
            # 获取相似示例
            examples = self._get_top_examples(sentence)
            logger.info("获取相似示例完成")
            
            # 第一阶段：独立分析
            stage1_result = self._stage_1_independent_analysis(sentence, examples)
            stage_results.append(stage1_result)
            
            # 人工决策：是否继续到第二阶段
            decision = self._request_human_decision("第一阶段完成", {
                "成功分析的Agent数": len([r for r in stage1_result.agent_responses.values() if "失败" not in r]),
                "处理时间": f"{stage1_result.processing_time:.2f}秒",
                "Token使用": stage1_result.token_usage.total_tokens
            }, stage1_result.agent_responses)
            stage1_result.human_decision = decision
            
            if decision == HumanDecision.TERMINATE:
                termination_reason = "第一阶段后人工终止"
                # 尝试从第一阶段提取MTL表达式
                for response in stage1_result.agent_responses.values():
                    mtl_match = re.search(r'```(.*?)```', response, re.DOTALL)
                    if mtl_match:
                        final_mtl_expression = mtl_match.group(1).strip()
                        break
                return self._create_result(sentence, start_time, stage_results, final_mtl_expression, termination_reason)
            
            # 第二阶段：讨论
            stage2_result = self._stage_2_discussion(sentence, stage1_result)
            stage_results.append(stage2_result)
            
            # 人工决策：是否继续到第三阶段
            decision = self._request_human_decision("第二阶段完成", {
                "讨论轮次": "5轮",
                "处理时间": f"{stage2_result.processing_time:.2f}秒",
                "Token使用": stage2_result.token_usage.total_tokens
            }, stage2_result.agent_responses)
            stage2_result.human_decision = decision
            
            if decision == HumanDecision.TERMINATE:
                termination_reason = "第二阶段后人工终止"
                # 尝试从第二阶段提取MTL表达式
                for response in stage2_result.agent_responses.values():
                    mtl_match = re.search(r'```(.*?)```', response, re.DOTALL)
                    if mtl_match:
                        final_mtl_expression = mtl_match.group(1).strip()
                        break
                return self._create_result(sentence, start_time, stage_results, final_mtl_expression, termination_reason)
            
            # 第三阶段：投票
            stage3_result = self._stage_3_voting(sentence, stage1_result, stage2_result)
            stage_results.append(stage3_result)
            
            # 人工决策：是否继续到第四阶段
            decision = self._request_human_decision("第三阶段完成", {
                "投票完成": "所有Agent已投票",
                "处理时间": f"{stage3_result.processing_time:.2f}秒",
                "Token使用": stage3_result.token_usage.total_tokens
            }, stage3_result.agent_responses, stage3_result)
            stage3_result.human_decision = decision
            
            if decision == HumanDecision.TERMINATE:
                termination_reason = "第三阶段后人工终止"
                # 尝试从投票结果提取MTL表达式
                for response in stage3_result.agent_responses.values():
                    mtl_match = re.search(r'```(.*?)```', response, re.DOTALL)
                    if mtl_match:
                        final_mtl_expression = mtl_match.group(1).strip()
                        break
                return self._create_result(sentence, start_time, stage_results, final_mtl_expression, termination_reason)
            
            # 第四阶段：仲裁
            stage4_result = self._stage_4_arbitration(sentence, stage1_result, stage2_result, stage3_result)
            stage_results.append(stage4_result)
            
            # 人工决策：最终确认
            decision = self._request_human_decision("第四阶段完成（最终）", {
                "仲裁完成": "专家仲裁已完成",
                "处理时间": f"{stage4_result.processing_time:.2f}秒",
                "Token使用": stage4_result.token_usage.total_tokens
            }, stage4_result.agent_responses)
            stage4_result.human_decision = decision
            
            # 从仲裁结果提取最终MTL表达式
            for response in stage4_result.agent_responses.values():
                mtl_match = re.search(r'```(.*?)```', response, re.DOTALL)
                if mtl_match:
                    final_mtl_expression = mtl_match.group(1).strip()
                    break
            
            if decision == HumanDecision.TERMINATE:
                termination_reason = "第四阶段后人工确认终止"
            else:
                termination_reason = "完成所有阶段"
            
        except Exception as e:
            logger.error(f"处理过程出错: {e}")
            termination_reason = f"系统错误: {str(e)}"
        
        return self._create_result(sentence, start_time, stage_results, final_mtl_expression, termination_reason)
    
    def _create_result(self, sentence: str, start_time: float, stage_results: List[StageResult],
                      final_mtl_expression: Optional[str], termination_reason: str) -> ProcessResult:
        """创建处理结果"""
        total_processing_time = time.time() - start_time
        
        return ProcessResult(
            input_sentence=sentence,
            final_mtl_expression=final_mtl_expression,
            total_token_usage=self.total_token_usage,
            total_processing_time=total_processing_time,
            stage_results=stage_results,
            termination_reason=termination_reason
        )
    
    def save_result(self, result: ProcessResult, output_file: str):
        """保存处理结果"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 收集模型配置信息
        model_info = {
            "agents": [
                {
                    "name": agent["name"],
                    "role": agent["role"],
                    "model": agent["model"],
                    "temperature": agent["temperature"]
                }
                for agent in self.agents
            ],
            "total_agents": len(self.agents),
            "sentence_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        save_data = {
            "input_sentence": result.input_sentence,
            "final_mtl_expression": result.final_mtl_expression,
            "termination_reason": result.termination_reason,
            "total_processing_time": result.total_processing_time,
            "total_token_usage": asdict(result.total_token_usage),
            "model_info": model_info,
            "stage_results": [
                {
                    "stage_name": stage.stage_name,
                    "processing_time": stage.processing_time,
                    "token_usage": asdict(stage.token_usage),
                    "human_decision": stage.human_decision.value if stage.human_decision else None,
                    "agent_responses": stage.agent_responses,
                    "candidate_details": getattr(stage, 'candidate_details', None)
                }
                for stage in result.stage_results
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_file}")

def main():
    """主函数演示"""
    print("=== NL2MTL ===\n")
    
    # 创建处理器
    processor = SimplifiedNL2MTL()
    
    # 测试句子
    test_sentence = "After receiving the signal, the system must respond within 10 seconds."
    
    print(f"处理句子: {test_sentence}")
    print("-" * 60)
    try:
        # 执行处理
        result = processor.process(test_sentence)
        
        # 显示结果
        print(f"\n最终MTL表达式: {result.final_mtl_expression}")
        print(f"终止原因: {result.termination_reason}")
        print(f"总处理时间: {result.total_processing_time:.2f}秒")
        print(f"总Token使用: {result.total_token_usage.total_tokens}")
        
        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"data/output/simplified_{timestamp}.json"
        processor.save_result(result, output_file)
        
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    main()
        