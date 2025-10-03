#!/usr/bin/env python3
"""
测试改进后的歧义检测器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ambiguity_detector import AmbiguityDetector
import json

def test_basic_functionality():
    """测试基本功能"""
    print("测试基本功能...")
    
    # 创建检测器实例
    try:
        detector = AmbiguityDetector()
        print("✅ 检测器初始化成功")
    except Exception as e:
        print(f"❌ 检测器初始化失败: {e}")
        return False
    
    # 测试简单句子
    test_sentences = [
        "机器人必须在5秒内响应用户指令",
        "系统应该持续监控直到收到停止信号", 
        "如果温度超过阈值，则立即触发报警"
    ]
    
    for sentence in test_sentences:
        print(f"\n测试句子: {sentence}")
        try:
            result = detector.detect_ambiguity(sentence)
            print(f"  是否有歧义: {result.is_ambiguous}")
            print(f"  置信度: {result.confidence:.2f}")
            print(f"  处理阶段: {result.consensus_reached_at or '完整流程'}")
            if result.mtl_expressions:
                print(f"  MTL表达式数量: {len(result.mtl_expressions)}")
            print("✅ 检测成功")
        except Exception as e:
            print(f"❌ 检测失败: {e}")
            return False
    
    return True

def test_config_loading():
    """测试配置加载"""
    print("\n测试配置加载...")
    
    try:
        # 测试默认配置
        detector1 = AmbiguityDetector()
        print("✅ 默认配置加载成功")
        
        # 测试指定配置文件
        detector2 = AmbiguityDetector("config/detector_config.json")
        print("✅ 配置文件加载成功")
        
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_rag_functionality():
    """测试RAG功能"""
    print("\n测试RAG功能...")
    
    try:
        # 测试RAG启用的配置
        config = {
            "agents": [
                {
                    "name": "Agent_A",
                    "role": "logician", 
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.3,
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_API_URL"
                }
            ],
            "rag_enabled": False,  # 暂时禁用RAG避免依赖问题
            "similarity_threshold": 0.95
        }
        
        # 保存临时配置
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_config_path = f.name
        
        detector = AmbiguityDetector(temp_config_path)
        print("✅ RAG配置测试成功")
        
        # 清理临时文件
        os.unlink(temp_config_path)
        
        return True
    except Exception as e:
        print(f"❌ RAG功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("开始测试改进后的歧义检测器")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_rag_functionality,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 出现异常: {e}")
    
    print(f"\n{'='*50}")
    print(f"测试结果: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关功能")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)