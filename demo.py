"""
Enhanced Ambiguity Detection System Demo
增强版歧义检测系统演示脚本
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

try:
    from enhanced_collab import EnhancedCollaborativeDetector
    from rag_module import RAGModule, create_sample_knowledge_base
    from evaluation_module import EvaluationModule, create_test_dataset
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有依赖已正确安装: pip install -r requirements.txt")
    sys.exit(1)

def setup_demo_environment():
    """设置演示环境"""
    print("🔧 设置演示环境...")
    
    # 创建必要的目录
    directories = [
        "data/knowledge_base",
        "data/input", 
        "data/output",
        "data/evaluation",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # 创建示例知识库
    try:
        create_sample_knowledge_base()
        print("✅ 示例知识库创建成功")
    except Exception as e:
        print(f"⚠️ 创建知识库失败: {e}")
    
    # 检查环境变量
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️ 警告: 未找到API密钥，请配置 .env 文件")
        print("   复制 .env.example 为 .env 并填入你的API密钥")
        return False
    
    return True

def demo_basic_detection():
    """演示基础检测功能"""
    print("\n" + "="*60)
    print("📝 基础歧义检测演示")
    print("="*60)
    
    # 测试句子
    test_sentences = [
        {
            "sentence": "他用望远镜看到了那个人。",
            "description": "经典句法歧义：介词短语修饰歧义"
        },
        {
            "sentence": "银行在河边。", 
            "description": "词汇歧义：一词多义（金融机构 vs 河岸）"
        },
        {
            "sentence": "今天天气很好。",
            "description": "无歧义句子"
        },
        {
            "sentence": "Flying planes can be dangerous.",
            "description": "英语句法歧义：动名词 vs 形容词+名词"
        }
    ]
    
    try:
        # 创建检测器（简化配置，不依赖外部API）
        detector = EnhancedCollaborativeDetector(
            enable_rag=True,
            enable_evaluation=False
        )
        
        for i, test_case in enumerate(test_sentences, 1):
            sentence = test_case["sentence"]
            description = test_case["description"]
            
            print(f"\n{i}. 句子: {sentence}")
            print(f"   类型: {description}")
            print("-" * 50)
            
            try:
                result = detector.detect_with_full_pipeline(sentence)
                
                print(f"✅ 检测结果:")
                print(f"   是否有歧义: {'是' if result.is_ambiguous else '否'}")
                print(f"   最终解释: {result.final_interpretation}")
                print(f"   置信度: {result.confidence:.3f}")
                print(f"   共识达成阶段: {result.consensus_reached_at or '未达成'}")
                print(f"   处理时间: {result.processing_time:.2f}秒")
                
                if result.rag_knowledge_used:
                    print(f"   使用RAG知识: {len(result.rag_knowledge_used)} 条")
                
            except Exception as e:
                print(f"❌ 检测失败: {e}")
                
    except Exception as e:
        print(f"❌ 初始化检测器失败: {e}")
        print("   这可能是由于缺少API密钥或网络问题")

def demo_rag_functionality():
    """演示RAG功能"""
    print("\n" + "="*60)
    print("🧠 RAG知识检索演示")
    print("="*60)
    
    try:
        rag = RAGModule()
        
        test_queries = [
            "他用望远镜看到了那个人",
            "银行在河边",
            "Flying planes can be dangerous"
        ]
        
        for query in test_queries:
            print(f"\n🔍 查询: {query}")
            print("-" * 40)
            
            relevant_knowledge = rag.retrieve_relevant_knowledge(query, top_k=3)
            
            if relevant_knowledge:
                for i, knowledge in enumerate(relevant_knowledge, 1):
                    print(f"{i}. 类型: {knowledge.get('type', '未知')}")
                    print(f"   相似度: {knowledge.get('similarity_score', 0):.3f}")
                    if 'example' in knowledge:
                        print(f"   示例: {knowledge['example']}")
                    if 'pattern' in knowledge:
                        print(f"   模式: {knowledge['pattern']}")
                    print()
            else:
                print("   未找到相关知识")
        
        # 显示知识库统计
        stats = rag.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   总条目数: {stats['total_items']}")
        print(f"   类型分布: {stats['types']}")
        
    except Exception as e:
        print(f"❌ RAG演示失败: {e}")

def demo_evaluation_system():
    """演示评估系统"""
    print("\n" + "="*60)
    print("📊 评估系统演示")
    print("="*60)
    
    try:
        # 创建测试数据集
        test_data = create_test_dataset()
        print(f"✅ 创建测试数据集: {len(test_data)} 个样本")
        
        # 显示测试数据统计
        ambiguous_count = sum(1 for item in test_data if item["is_ambiguous"])
        print(f"   歧义句子: {ambiguous_count} 个")
        print(f"   非歧义句子: {len(test_data) - ambiguous_count} 个")
        
        # 显示样本
        print(f"\n📝 测试样本预览:")
        for i, sample in enumerate(test_data[:3], 1):
            print(f"{i}. {sample['sentence']}")
            print(f"   歧义: {'是' if sample['is_ambiguous'] else '否'}")
            print(f"   类型: {sample['ambiguity_type']}")
        
        if len(test_data) > 3:
            print(f"   ... 还有 {len(test_data) - 3} 个样本")
        
        # 注意：实际评估需要API密钥，这里只演示数据准备
        print(f"\n💡 提示: 完整评估需要配置API密钥")
        print(f"   配置后可运行: python -c \"from evaluation_module import *; evaluator = EvaluationModule(); evaluator.run_comprehensive_evaluation(create_test_dataset())\"")
        
    except Exception as e:
        print(f"❌ 评估演示失败: {e}")

def demo_batch_processing():
    """演示批量处理功能"""
    print("\n" + "="*60)
    print("⚡ 批量处理演示")
    print("="*60)
    
    # 创建示例输入文件
    sample_data = [
        {"sentence": "他用望远镜看到了那个人。"},
        {"sentence": "银行在河边。"},
        {"sentence": "今天天气很好。"},
        {"sentence": "老师的书很有趣。"},
        {"sentence": "他们在讨论老师的问题。"}
    ]
    
    input_file = "data/input/demo_sentences.xlsx"
    output_file = "data/output/demo_results.xlsx"
    
    try:
        # 创建输入文件
        df = pd.DataFrame(sample_data)
        df.to_excel(input_file, index=False)
        print(f"✅ 创建示例输入文件: {input_file}")
        print(f"   包含 {len(sample_data)} 个句子")
        
        # 显示输入数据
        print(f"\n📝 输入数据预览:")
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"{i}. {row['sentence']}")
        
        print(f"\n💡 批量处理命令:")
        print(f"   python -c \"from enhanced_collab import *; detector = EnhancedCollaborativeDetector(); detector.run_batch_detection('{input_file}', '{output_file}')\"")
        
    except Exception as e:
        print(f"❌ 批量处理演示失败: {e}")

def main():
    """主演示函数"""
    print("🚀 Enhanced Ambiguity Detection System Demo")
    print("   增强版歧义检测系统演示")
    print("="*60)
    
    # 设置环境
    if not setup_demo_environment():
        print("\n⚠️ 环境设置不完整，某些功能可能无法正常工作")
    
    # 运行各个演示
    try:
        demo_rag_functionality()
        demo_evaluation_system() 
        demo_batch_processing()
        
        # 基础检测演示（需要API密钥）
        if os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY"):
            demo_basic_detection()
        else:
            print(f"\n⚠️ 跳过基础检测演示（需要API密钥）")
            print(f"   配置API密钥后可体验完整功能")
        
    except KeyboardInterrupt:
        print(f"\n\n👋 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程出错: {e}")
    
    print(f"\n" + "="*60)
    print(f"✅ 演示完成！")
    print(f"📚 更多信息请查看 README.md")
    print(f"🔧 配置说明请查看 config/detector_config.json")
    print(f"📁 结果文件保存在 data/output/ 目录")
    print(f"="*60)

if __name__ == "__main__":
    main()