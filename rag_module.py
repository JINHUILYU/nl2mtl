"""
RAG Module for Knowledge-Enhanced Ambiguity Detection
检索增强生成模块，为歧义检测提供外部知识支持
"""

import json
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RAGModule:
    """检索增强生成模块"""
    
    def __init__(self, knowledge_base_path: str = "data/knowledge_base", 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化RAG模块
        
        Args:
            knowledge_base_path: 知识库路径
            model_name: 句子嵌入模型名称
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.model = SentenceTransformer(model_name)
        self.knowledge_base = self._load_knowledge_base()
        self.embeddings = self._compute_embeddings()
        
    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        """加载知识库"""
        knowledge_base = []
        
        # 加载不同类型的知识源
        knowledge_sources = [
            "ambiguity_examples.json",
            "linguistic_patterns.json", 
            "domain_knowledge.json"
        ]
        
        for source in knowledge_sources:
            source_path = self.knowledge_base_path / source
            if source_path.exists():
                try:
                    with open(source_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            knowledge_base.extend(data)
                        else:
                            knowledge_base.append(data)
                    logger.info(f"已加载知识源: {source}")
                except Exception as e:
                    logger.warning(f"加载知识源失败 {source}: {e}")
            else:
                logger.warning(f"知识源不存在: {source}")
        
        # 如果没有找到知识库文件，创建默认知识库
        if not knowledge_base:
            knowledge_base = self._create_default_knowledge_base()
            
        return knowledge_base
    
    def _create_default_knowledge_base(self) -> List[Dict[str, Any]]:
        """创建默认知识库"""
        default_knowledge = [
            {
                "id": "ambiguity_001",
                "type": "syntactic_ambiguity",
                "example": "他用望远镜看到了那个人",
                "interpretations": [
                    "他使用望远镜看到了那个人",
                    "他看到了那个拿着望远镜的人"
                ],
                "pattern": "介词短语修饰歧义",
                "keywords": ["用", "看到", "工具", "修饰"]
            },
            {
                "id": "ambiguity_002", 
                "type": "lexical_ambiguity",
                "example": "银行在河边",
                "interpretations": [
                    "金融机构在河边",
                    "河岸在河边"
                ],
                "pattern": "一词多义",
                "keywords": ["银行", "河边", "多义词"]
            },
            {
                "id": "ambiguity_003",
                "type": "syntactic_ambiguity", 
                "example": "Flying planes can be dangerous",
                "interpretations": [
                    "Flying planes (gerund) can be dangerous - 驾驶飞机可能很危险",
                    "Flying planes (adjective + noun) can be dangerous - 飞行的飞机可能很危险"
                ],
                "pattern": "动名词与形容词+名词歧义",
                "keywords": ["flying", "planes", "gerund", "adjective"]
            },
            {
                "id": "pattern_001",
                "type": "linguistic_pattern",
                "pattern": "介词短语修饰歧义",
                "description": "介词短语可以修饰不同的成分，导致句法歧义",
                "indicators": ["用", "在", "从", "向", "with", "in", "from", "to"],
                "resolution_strategy": "明确修饰关系，考虑语义合理性"
            },
            {
                "id": "pattern_002",
                "type": "linguistic_pattern", 
                "pattern": "一词多义",
                "description": "同一个词在不同语境下有不同含义",
                "indicators": ["多义词", "同形异义", "语境依赖"],
                "resolution_strategy": "结合上下文确定具体含义"
            }
        ]
        
        # 确保知识库目录存在
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        
        # 保存默认知识库
        with open(self.knowledge_base_path / "default_knowledge.json", 'w', encoding='utf-8') as f:
            json.dump(default_knowledge, f, ensure_ascii=False, indent=2)
            
        logger.info("已创建默认知识库")
        return default_knowledge
    
    def _compute_embeddings(self) -> np.ndarray:
        """计算知识库条目的嵌入向量"""
        if not self.knowledge_base:
            return np.array([])
            
        # 为每个知识库条目创建文本表示
        texts = []
        for item in self.knowledge_base:
            text_parts = []
            
            # 添加示例文本
            if "example" in item:
                text_parts.append(item["example"])
                
            # 添加解释
            if "interpretations" in item:
                text_parts.extend(item["interpretations"])
                
            # 添加模式描述
            if "description" in item:
                text_parts.append(item["description"])
                
            # 添加关键词
            if "keywords" in item:
                text_parts.extend(item["keywords"])
                
            combined_text = " ".join(text_parts)
            texts.append(combined_text)
        
        # 计算嵌入向量
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().numpy()
    
    def retrieve_relevant_knowledge(self, query: str, top_k: int = 5, 
                                  similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        检索相关知识
        
        Args:
            query: 查询文本
            top_k: 返回最相关的k个结果
            similarity_threshold: 相似度阈值
            
        Returns:
            相关知识列表
        """
        if not self.knowledge_base or len(self.embeddings) == 0:
            return []
            
        # 计算查询的嵌入向量
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        
        # 计算相似度
        similarities = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        
        # 获取最相似的条目
        top_indices = similarities.argsort(descending=True)[:top_k]
        
        relevant_knowledge = []
        for idx in top_indices:
            similarity_score = float(similarities[idx])
            if similarity_score >= similarity_threshold:
                knowledge_item = self.knowledge_base[idx].copy()
                knowledge_item["similarity_score"] = similarity_score
                relevant_knowledge.append(knowledge_item)
        
        return relevant_knowledge
    
    def enhance_prompt_with_knowledge(self, original_prompt: str, sentence: str) -> str:
        """
        使用检索到的知识增强提示词
        
        Args:
            original_prompt: 原始提示词
            sentence: 待分析的句子
            
        Returns:
            增强后的提示词
        """
        # 检索相关知识
        relevant_knowledge = self.retrieve_relevant_knowledge(sentence)
        
        if not relevant_knowledge:
            return original_prompt
            
        # 构建知识增强部分
        knowledge_section = "\n\n**相关知识参考:**\n"
        
        for i, knowledge in enumerate(relevant_knowledge, 1):
            knowledge_section += f"\n{i}. **{knowledge.get('type', '未知类型')}**\n"
            
            if "example" in knowledge:
                knowledge_section += f"   示例: {knowledge['example']}\n"
                
            if "interpretations" in knowledge:
                knowledge_section += f"   可能解释:\n"
                for interp in knowledge['interpretations']:
                    knowledge_section += f"   - {interp}\n"
                    
            if "pattern" in knowledge:
                knowledge_section += f"   模式: {knowledge['pattern']}\n"
                
            if "resolution_strategy" in knowledge:
                knowledge_section += f"   解决策略: {knowledge['resolution_strategy']}\n"
                
            knowledge_section += f"   相似度: {knowledge['similarity_score']:.3f}\n"
        
        # 将知识插入到原始提示词中
        enhanced_prompt = original_prompt + knowledge_section
        enhanced_prompt += "\n\n请参考以上相关知识进行分析，但不要完全依赖，要结合具体句子进行独立判断。"
        
        return enhanced_prompt
    
    def add_knowledge_item(self, knowledge_item: Dict[str, Any]) -> bool:
        """
        添加新的知识条目
        
        Args:
            knowledge_item: 知识条目
            
        Returns:
            是否添加成功
        """
        try:
            # 验证知识条目格式
            required_fields = ["id", "type"]
            for field in required_fields:
                if field not in knowledge_item:
                    logger.error(f"知识条目缺少必需字段: {field}")
                    return False
            
            # 检查ID是否已存在
            existing_ids = [item.get("id") for item in self.knowledge_base]
            if knowledge_item["id"] in existing_ids:
                logger.warning(f"知识条目ID已存在: {knowledge_item['id']}")
                return False
            
            # 添加到知识库
            self.knowledge_base.append(knowledge_item)
            
            # 重新计算嵌入向量
            self.embeddings = self._compute_embeddings()
            
            # 保存到文件
            self._save_knowledge_base()
            
            logger.info(f"成功添加知识条目: {knowledge_item['id']}")
            return True
            
        except Exception as e:
            logger.error(f"添加知识条目失败: {e}")
            return False
    
    def _save_knowledge_base(self):
        """保存知识库到文件"""
        try:
            output_file = self.knowledge_base_path / "updated_knowledge.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
            logger.info(f"知识库已保存到: {output_file}")
        except Exception as e:
            logger.error(f"保存知识库失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        if not self.knowledge_base:
            return {"total_items": 0}
            
        stats = {
            "total_items": len(self.knowledge_base),
            "types": {},
            "has_examples": 0,
            "has_patterns": 0
        }
        
        for item in self.knowledge_base:
            # 统计类型
            item_type = item.get("type", "unknown")
            stats["types"][item_type] = stats["types"].get(item_type, 0) + 1
            
            # 统计特征
            if "example" in item:
                stats["has_examples"] += 1
            if "pattern" in item:
                stats["has_patterns"] += 1
        
        return stats

def create_sample_knowledge_base():
    """创建示例知识库文件"""
    knowledge_base_path = Path("data/knowledge_base")
    knowledge_base_path.mkdir(parents=True, exist_ok=True)
    
    # 歧义示例
    ambiguity_examples = [
        {
            "id": "amb_001",
            "type": "syntactic_ambiguity",
            "example": "他用望远镜看到了那个人",
            "interpretations": [
                "他使用望远镜看到了那个人（工具用法）",
                "他看到了那个拿着望远镜的人（修饰用法）"
            ],
            "pattern": "介词短语修饰歧义",
            "keywords": ["用", "看到", "工具", "修饰", "介词短语"],
            "difficulty": "medium",
            "language": "zh"
        },
        {
            "id": "amb_002",
            "type": "lexical_ambiguity", 
            "example": "银行在河边",
            "interpretations": [
                "金融机构在河边",
                "河岸在河边"
            ],
            "pattern": "一词多义",
            "keywords": ["银行", "河边", "多义词", "同形异义"],
            "difficulty": "easy",
            "language": "zh"
        },
        {
            "id": "amb_003",
            "type": "syntactic_ambiguity",
            "example": "Flying planes can be dangerous",
            "interpretations": [
                "Flying planes (动名词) can be dangerous - 驾驶飞机可能很危险",
                "Flying planes (形容词+名词) can be dangerous - 飞行的飞机可能很危险"
            ],
            "pattern": "动名词与形容词+名词歧义",
            "keywords": ["flying", "planes", "gerund", "adjective", "syntactic"],
            "difficulty": "hard",
            "language": "en"
        }
    ]
    
    # 语言学模式
    linguistic_patterns = [
        {
            "id": "pattern_001",
            "type": "linguistic_pattern",
            "pattern": "介词短语修饰歧义",
            "description": "介词短语可以修饰句子中的不同成分，导致句法歧义",
            "indicators": ["用", "在", "从", "向", "with", "in", "from", "to"],
            "resolution_strategy": "明确修饰关系，考虑语义合理性和上下文",
            "examples": ["他用望远镜看到了那个人", "I saw the man with the telescope"],
            "frequency": "high"
        },
        {
            "id": "pattern_002", 
            "type": "linguistic_pattern",
            "pattern": "一词多义",
            "description": "同一个词在不同语境下有不同含义",
            "indicators": ["多义词", "同形异义", "语境依赖"],
            "resolution_strategy": "结合上下文和领域知识确定具体含义",
            "examples": ["银行在河边", "The bank is by the river"],
            "frequency": "very_high"
        }
    ]
    
    # 领域知识
    domain_knowledge = [
        {
            "id": "domain_001",
            "type": "domain_knowledge",
            "domain": "finance",
            "terms": {
                "银行": ["金融机构", "储蓄机构", "贷款机构"],
                "利率": ["interest rate", "借贷成本"],
                "投资": ["investment", "资金配置"]
            },
            "common_ambiguities": ["银行（金融机构 vs 河岸）"]
        },
        {
            "id": "domain_002",
            "type": "domain_knowledge", 
            "domain": "technology",
            "terms": {
                "网络": ["network", "internet", "connection"],
                "云": ["cloud computing", "cloud storage"],
                "平台": ["platform", "system", "framework"]
            },
            "common_ambiguities": ["云（天空中的云 vs 云计算）"]
        }
    ]
    
    # 保存文件
    with open(knowledge_base_path / "ambiguity_examples.json", 'w', encoding='utf-8') as f:
        json.dump(ambiguity_examples, f, ensure_ascii=False, indent=2)
        
    with open(knowledge_base_path / "linguistic_patterns.json", 'w', encoding='utf-8') as f:
        json.dump(linguistic_patterns, f, ensure_ascii=False, indent=2)
        
    with open(knowledge_base_path / "domain_knowledge.json", 'w', encoding='utf-8') as f:
        json.dump(domain_knowledge, f, ensure_ascii=False, indent=2)
    
    print(f"示例知识库已创建在: {knowledge_base_path}")

if __name__ == "__main__":
    # 创建示例知识库
    create_sample_knowledge_base()
    
    # 测试RAG模块
    rag = RAGModule()
    
    # 测试检索
    test_queries = [
        "他用望远镜看到了那个人",
        "银行在河边",
        "Flying planes can be dangerous"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 50)
        
        relevant_knowledge = rag.retrieve_relevant_knowledge(query, top_k=3)
        
        for i, knowledge in enumerate(relevant_knowledge, 1):
            print(f"{i}. 类型: {knowledge.get('type')}")
            print(f"   相似度: {knowledge.get('similarity_score', 0):.3f}")
            if "example" in knowledge:
                print(f"   示例: {knowledge['example']}")
            print()
    
    # 显示统计信息
    print("\n知识库统计:")
    print(json.dumps(rag.get_statistics(), ensure_ascii=False, indent=2))