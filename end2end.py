"""
End-to-End NL2MTL Processing System
Simplified version with configuration-based prompt construction, optional RAG, and single agent processing.

功能流程：
1. 加载配置文件
2. 根据配置构造prompt（可选RAG）
3. 发送给单个agent处理
4. 提取结果并统计token消耗
5. 保存结果
"""

import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv
import logging
import os
import re
import copy

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """Token使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class ProcessResult:
    """处理结果"""
    input_sentence: str
    final_mtl_expression: Optional[str]
    agent_response: str
    token_usage: TokenUsage
    processing_time: float
    rag_enabled: bool
    examples_used: Optional[str] = None

class End2EndNL2MTL:
    """端到端NL2MTL处理器"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """初始化处理器"""
        self.config = self._load_config(config_path)
        self.client = self._initialize_client()
        self.sentence_model = None
        self.examples_data = None
        self.base_prompt = self._load_base_prompt()
        
        # 获取RAG配置
        self.rag_enabled = self.config.get("rag_enabled", True)
        logger.info(f"RAG启用状态: {self.rag_enabled}")
        
        # 如果启用RAG，初始化相关组件
        if self.rag_enabled:
            self._initialize_rag_components()
        
        # 获取agent配置
        self.agent_config = self.config["agents"][0]  # 使用第一个agent
        logger.info(f"使用Agent: {self.agent_config['name']} ({self.agent_config['model']})")
        
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
                }
            ],
            "rag_enabled": True
        }
    
    def _initialize_client(self) -> OpenAI:
        """初始化API客户端"""
        load_dotenv()
        
        agent_config = self.config["agents"][0]  # 使用第一个agent
        api_key = os.getenv(agent_config["api_key_env"])
        base_url = os.getenv(agent_config["base_url_env"])
        
        if not api_key:
            raise ValueError(f"API密钥未找到: {agent_config['api_key_env']}")
            
        return OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def _initialize_rag_components(self):
        """初始化RAG相关组件"""
        try:
            self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.examples_data = pd.read_excel("data/input/examples.xlsx")
            logger.info("RAG组件初始化成功")
        except Exception as e:
            logger.error(f"RAG组件初始化失败: {e}")
            self.rag_enabled = False
    
    def _load_base_prompt(self) -> str:
        """加载基础prompt模板"""
        try:
            with open("config/base_prompt.txt", 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("基础prompt文件未找到，使用默认prompt")
            return """请分析以下自然语言句子并生成对应的MTL表达式。

句子: [INPUT TEXT]

[EXAMPLE]

请提供详细的分析过程和最终的MTL表达式。

IMPORTANT: Format your response as follows:
1. First provide your detailed reasoning and analysis
2. End with: Repeat the final answer of the original question: ```
[your_final_answer]
```
3. Make sure your final answer is enclosed in triple backticks with newlines before and after
4. For MTL formulas, use standard notation with correct symbols
5. Your final answer between triple backticks must be the exact formula with no additional text"""
    
    def _calculate_similarity(self, sentence: str, examples: List[str]) -> List[float]:
        """计算句子与示例的相似度"""
        if self.sentence_model is None:
            logger.error("句子模型未初始化")
            return [0.0] * len(examples)
            
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
        # 如果RAG未启用，返回空示例
        if not self.rag_enabled:
            logger.info("RAG未启用，跳过示例检索")
            return ""
        
        if self.examples_data is None or self.examples_data.empty:
            logger.warning("示例数据未加载或为空")
            return ""
        
        try:
            # 获取示例数据的列名（兼容不同的列名）
            input_col = None
            output_col = None
            
            for col in self.examples_data.columns:
                if any(keyword in col.lower() for keyword in ['input', 'text', 'natural', 'rule']):
                    input_col = col
                elif any(keyword in col.lower() for keyword in ['answer', 'output', 'mtl', 'formula']):
                    output_col = col
            
            if input_col is None or output_col is None:
                logger.error(f"无法识别示例数据的列名: {list(self.examples_data.columns)}")
                return ""
            
            examples = self.examples_data[input_col].tolist()
            answers = self.examples_data[output_col].tolist()
            
            similarities = self._calculate_similarity(sentence, examples)
            
            # 获取top-k索引
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # 构建示例文本
            example_text = ""
            for i, idx in enumerate(top_indices, 1):
                example_text += f"**<Example {i}>**\n"
                example_text += f"**Input Text**: {examples[idx]}\n"
                example_text += f"**Analysis Process**: {answers[idx]}\n\n"
            
            logger.info(f"检索到 {len(top_indices)} 个相似示例")
            return example_text
            
        except Exception as e:
            logger.error(f"示例检索失败: {e}")
            return ""
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> tuple[str, TokenUsage]:
        """调用LLM并追踪token使用"""
        try:
            response = self.client.chat.completions.create(
                model=self.agent_config["model"],
                messages=messages,  # type: ignore
                temperature=self.agent_config["temperature"]
            )
            
            # 追踪token使用
            token_usage = TokenUsage()
            if hasattr(response, 'usage') and response.usage:
                token_usage.prompt_tokens = response.usage.prompt_tokens
                token_usage.completion_tokens = response.usage.completion_tokens
                token_usage.total_tokens = response.usage.total_tokens
            
            content = response.choices[0].message.content
            return content.strip() if content else "", token_usage
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return f"LLM调用失败: {str(e)}", TokenUsage()
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """从回答中提取最终答案"""
        # 匹配 "Repeat the final answer" 或类似提示后的三重反引号内容
        match = re.search(r"(?:Repeat the final answer|Final answer|MTL Formula).*?:?\s*```(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 匹配简单的三重反引号格式
        match = re.search(r"```(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 尝试提取包含MTL操作符的行
        lines = text.split('\n')
        for line in lines:
            if any(op in line for op in ['G(', 'F_[', 'P_[', 'U_[', 'X(', 'G[', 'F[', 'U[', 'P[']):
                return line.strip()

        return None
    
    def process_single(self, sentence: str) -> ProcessResult:
        """处理单个句子"""
        start_time = time.time()
        logger.info(f"开始处理句子: {sentence}")
        
        try:
            # 获取相似示例（如果启用RAG）
            examples = self._get_top_examples(sentence) if self.rag_enabled else ""
            
            # 构建完整的prompt
            prompt = self.base_prompt.replace("[INPUT TEXT]", sentence)
            if self.rag_enabled and examples:
                prompt = prompt.replace("[EXAMPLE]", examples)
            else:
                prompt = prompt.replace("[EXAMPLE]", "")
            
            # 构建消息
            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert in natural language to Linear Temporal Logic (LTL) conversion. Please solve the following problem independently and precisely."
                },
                {"role": "user", "content": prompt}
            ]
            
            # 调用LLM
            response, token_usage = self._call_llm(messages)
            
            # 提取最终答案
            final_answer = self._extract_final_answer(response)
            
            processing_time = time.time() - start_time
            
            logger.info(f"处理完成，用时: {processing_time:.2f}秒，Token使用: {token_usage.total_tokens}")
            
            return ProcessResult(
                input_sentence=sentence,
                final_mtl_expression=final_answer,
                agent_response=response,
                token_usage=token_usage,
                processing_time=processing_time,
                rag_enabled=self.rag_enabled,
                examples_used=examples if self.rag_enabled else None
            )
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            processing_time = time.time() - start_time
            return ProcessResult(
                input_sentence=sentence,
                final_mtl_expression=None,
                agent_response=f"处理失败: {str(e)}",
                token_usage=TokenUsage(),
                processing_time=processing_time,
                rag_enabled=self.rag_enabled
            )
    
    def process_batch(self, sentences: List[str], output_file: str) -> List[ProcessResult]:
        """批量处理句子"""
        logger.info(f"开始批量处理 {len(sentences)} 个句子")
        
        results = []
        total_tokens = 0
        total_time = 0
        
        for i, sentence in enumerate(sentences, 1):
            logger.info(f"处理第 {i}/{len(sentences)} 个句子")
            
            result = self.process_single(sentence)
            results.append(result)
            
            total_tokens += result.token_usage.total_tokens
            total_time += result.processing_time
            
            # 每5个样本保存一次中间结果
            if i % 5 == 0 or i == len(sentences):
                self.save_batch_results(results, output_file)
                logger.info(f"已保存 {len(results)} 个结果到 {output_file}")
        
        logger.info(f"批量处理完成！总Token使用: {total_tokens}，总用时: {total_time:.2f}秒")
        return results
    
    def save_result(self, result: ProcessResult, output_file: str):
        """保存单个结果"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "input_sentence": result.input_sentence,
            "final_mtl_expression": result.final_mtl_expression,
            "agent_response": result.agent_response,
            "token_usage": asdict(result.token_usage),
            "processing_time": result.processing_time,
            "rag_enabled": result.rag_enabled,
            "examples_used": result.examples_used,
            "agent_config": {
                "name": self.agent_config["name"],
                "model": self.agent_config["model"],
                "temperature": self.agent_config["temperature"]
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_file}")
    
    def save_batch_results(self, results: List[ProcessResult], output_file: str):
        """保存批量结果"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为DataFrame格式便于查看
        batch_data = []
        total_tokens = 0
        total_time = 0
        
        for result in results:
            batch_data.append({
                "Input_Sentence": result.input_sentence,
                "Final_MTL_Expression": result.final_mtl_expression,
                "Agent_Response": result.agent_response,
                "Processing_Time": result.processing_time,
                "Token_Usage": result.token_usage.total_tokens,
                "RAG_Enabled": result.rag_enabled
            })
            total_tokens += result.token_usage.total_tokens
            total_time += result.processing_time
        
        # 保存为Excel文件
        df = pd.DataFrame(batch_data)
        df.to_excel(output_file, index=False)
        
        # 同时保存详细的JSON结果
        json_file = output_file.replace('.xlsx', '_detailed.json')
        detailed_data = {
            "summary": {
                "total_sentences": len(results),
                "total_tokens": total_tokens,
                "total_time": total_time,
                "average_time_per_sentence": total_time / len(results) if results else 0,
                "agent_config": {
                    "name": self.agent_config["name"],
                    "model": self.agent_config["model"],
                    "temperature": self.agent_config["temperature"]
                },
                "rag_enabled": self.rag_enabled,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": [
                {
                    "input_sentence": result.input_sentence,
                    "final_mtl_expression": result.final_mtl_expression,
                    "agent_response": result.agent_response,
                    "token_usage": asdict(result.token_usage),
                    "processing_time": result.processing_time,
                    "rag_enabled": result.rag_enabled,
                    "examples_used": result.examples_used
                }
                for result in results
            ]
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"批量结果已保存到: {output_file} 和 {json_file}")

def main():
    """主函数演示"""
    print("=== End-to-End NL2MTL Processing ===\n")
    
    # 创建处理器
    processor = End2EndNL2MTL()
    
    # 设置处理模式
    batch_mode = True  # 设置为True进行批量处理，False进行单句处理
    
    if not batch_mode:
        # 单句处理示例
        test_sentence = "Globally, if a is true, then b will be true in the next step."
        print(f"处理句子: {test_sentence}")
        print("-" * 60)
        
        result = processor.process_single(test_sentence)
        
        print(f"\n🤖 Agent回答:")
        print(result.agent_response)
        print(f"\n📝 提取的MTL表达式: {result.final_mtl_expression}")
        print(f"⏱️  处理时间: {result.processing_time:.2f}秒")
        print(f"🔢 Token使用: {result.token_usage.total_tokens}")
        print(f"🔍 RAG启用: {result.rag_enabled}")
        
        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"data/output/end2end/single_{timestamp}.json"
        processor.save_result(result, output_file)
        
    else:
        # 批量处理示例
        print("开始批量处理数据集...")
        
        try:
            # 读取数据集
            dataset_df = pd.read_excel("data/input/nl2spec-dataset.xlsx")
            sentences = dataset_df["NL"].tolist()
            
            # 限制处理数量（测试用）
            # sentences = sentences[:10]  # 只处理前10个
            
            print(f"共加载 {len(sentences)} 个句子")
            
            # 批量处理
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"data/output/end2end/batch_{timestamp}.xlsx"
            
            results = processor.process_batch(sentences, output_file)
            
            # 显示统计信息
            total_tokens = sum(r.token_usage.total_tokens for r in results)
            total_time = sum(r.processing_time for r in results)
            successful_results = len([r for r in results if r.final_mtl_expression is not None])
            
            print(f"\n✅ 批量处理完成!")
            print(f"📊 处理统计:")
            print(f"   - 总句子数: {len(results)}")
            print(f"   - 成功提取: {successful_results}")
            print(f"   - 成功率: {successful_results/len(results)*100:.1f}%")
            print(f"   - 总Token使用: {total_tokens}")
            print(f"   - 总处理时间: {total_time:.2f}秒")
            print(f"   - 平均每句用时: {total_time/len(results):.2f}秒")
            print(f"   - 结果保存至: {output_file}")
            
        except FileNotFoundError as e:
            print(f"❌ 数据集文件未找到: {e}")
            print("请确保 data/input/nl2spec-dataset.xlsx 文件存在")
        except Exception as e:
            print(f"❌ 批量处理失败: {e}")

if __name__ == "__main__":
    main()