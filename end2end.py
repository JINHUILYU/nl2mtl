from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv
import copy
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re


def load_env():
    """自动加载环境变量"""
    env_path = Path(__file__).resolve().parent / ".env"
    if not load_dotenv(env_path):
        raise FileNotFoundError("❌ cannot find .env file")


def create_client() -> OpenAI:
    """创建 OpenAI 客户端"""
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_API_URL")
        # api_key=os.getenv("OPENAI_API_KEY"),
        # base_url=os.getenv("OPENAI_API_URL"),
    )


def call_openai_chat(client: OpenAI, messages: list) -> str:
    """调用 OpenAI 聊天模型"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        # model="deepseek-reasoner",
        # model="gpt-3.5-turbo",
        # model="gpt-4o",
        # model="o3",
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def extract_final_answer(text):
    """
    Extract the final answer from the response, supporting various formats and multi-line extraction
    """
    # Match "Repeat the final answer" or similar prompts followed by content between triple backticks
    match = re.search(r"(?:Repeat the final answer|Final answer|LTL Formula).*?:?\s*```(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Match simple triple backtick format
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def nl2ltl(prompt: str, client: OpenAI) -> str:
    complete_result = ""
    history = []
    messages = [
        {"role": "system", "content": (
            "You are an expert in natural language to Linear Temporal Logic (LTL) conversion. "
            "Please solve the following problem independently and precisely.\n\n"
            "IMPORTANT: Format your response as follows:\n"
            "1. First provide your detailed reasoning and analysis\n"
            "2. End with: Repeat the final answer of the original question: ```\n[your_final_answer]\n```\n"
            "3. Make sure your final answer is enclosed in triple backticks with newlines before and after\n"
            "4. For LTL formulas, use standard notation with correct symbols\n"
            "5. Your final answer between triple backticks must be the exact formula with no additional text"
        )},
        {"role": "user", "content": prompt}
    ]

    reply = call_openai_chat(client, messages)
    history.append({"role": "assistant", "content": reply})

    # Extract the final answer
    final_answer = extract_final_answer(reply)

    print(f"\n🤖 Agent's answer: \n{reply}")
    complete_result += f"\n🤖 Agent's answer: \n{reply}\n"

    if final_answer:
        print(f"\n📝 Extracted final answer: \n{final_answer}")
        complete_result += f"\n📝 Extracted final answer: \n{final_answer}\n"
    else:
        print("\n⚠️ Could not extract a clear final answer")
        complete_result += "\n⚠️ Could not extract a clear final answer\n"

    return complete_result


def end2end(question: str) -> str:
    load_env()
    client = create_client()
    return nl2ltl(question, client)


# 示例调用
if __name__ == "__main__":
    batch = True  # Set to True for batch processing, False for single question
    if not batch:
        # open the base prompt file
        with open("config/prompts/base_prompt.txt") as f:
            # read the base prompt content
            base_prompt = f.read()  # 任务描述 + 输出格式 + 知识库 + 样例 + 执行要求 + 问题
        Question = "Globally, if a is true, then b will be true in the next step."
        base_prompt = base_prompt.replace("[INPUT TEXT]", Question)
        # 计算相似度，构成完整的 Prompt
        # 读取 examples.xlsx 中的示例，与输入的问题进行相似度计算
        # 读取examples.xlsx文件中的示例数据
        examples_df = pd.read_excel("data/examples.xlsx")

        # 加载模型
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # 生成句子嵌入
        example_texts = examples_df["Input Text"].tolist()
        all_texts = [Question] + example_texts
        embeddings = model.encode(all_texts, convert_to_tensor=True)

        # 计算相似度
        cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1:])[0]

        # 获取前5个最相似的索引
        top_indices = cosine_scores.argsort(descending=True)[:5].tolist()

        # 构建示例模板
        examples_template = ""
        for idx in top_indices:
            input_text = examples_df.iloc[idx]["Input Text"]
            analysis_process = examples_df.iloc[idx]["Analysis Process"]

            examples_template += f"**<Example>**\n"
            examples_template += f"**Input Text**: {input_text}\n"
            examples_template += f"**Analysis Process**:\n{analysis_process}\n\n"

        # 将示例模板添加到base_prompt中
        base_prompt = base_prompt.replace("[Examples]", examples_template)
        print(base_prompt)
        result = end2end(base_prompt)
        print("\n✅ The final output result:\n", result)
    else:
        # 批量处理测试集
        print("\n开始批量处理数据集...")

        # 读取待处理的数据集
        dataset_df = pd.read_excel("data/input/nl2spec-dataset.xlsx")  # nl2spec dataset
        requirements = dataset_df["NL"].tolist()  # NL

        # dataset_df = pd.read_excel("data/input/temp.xlsx")  # dataset
        # requirements = dataset_df["Requirement"].tolist()  # Requirement

        # 创建结果列表
        results = []
        result_file = "data/output/end2end/end2end-deepseek-v3-nl2spec-result-checked-new.xlsx"

        # 读取base_prompt模板
        with open("config/prompts/base_prompt.txt") as f:
            base_prompt = f.read()
        # 批量处理每个需求
        for i, req in enumerate(requirements):
            print(f"\n处理需求 {i + 1}/{len(requirements)}: {req[:50]}...")

            # 构建完整的提示词
            current_prompt = copy.deepcopy(base_prompt)  # 替换为深拷贝，避免修改原始模板
            current_prompt = current_prompt.replace("[INPUT TEXT]", req)

            examples_df = pd.read_excel("data/examples.xlsx")
            # 加载模型
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            # 生成句子嵌入
            example_texts = examples_df["Input Text"].tolist()
            all_texts = [req] + example_texts
            embeddings = model.encode(all_texts, convert_to_tensor=True)

            # 计算相似度
            cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1:])[0]

            # 获取前5个最相似的索引
            top_indices = cosine_scores.argsort(descending=True)[:5].tolist()

            examples_template = ""
            for idx in top_indices:
                input_text = examples_df.iloc[idx]["Input Text"]
                analysis_process = examples_df.iloc[idx]["Analysis Process"]
                examples_template += f"**<Example>**\n"
                examples_template += f"**Input Text**: {input_text}\n"
                examples_template += f"**Analysis Process**:\n{analysis_process}\n\n"

            current_prompt = current_prompt.replace("[Examples]", examples_template)

            result = end2end(current_prompt)

            # 将结果添加到列表
            results.append({"Requirement": req, "Result": result})

            # 每5个样本保存一次中间结果（防止运行中断导致全部丢失）
            if (i + 1) % 5 == 0 or i == len(requirements) - 1:
                pd.DataFrame(results).to_excel(result_file, index=False)
                print(f"已保存 {len(results)} 个结果到 {result_file}")

        print(f"\n✅ 批量处理完成! 已处理 {len(results)} 个需求，结果已保存到 {result_file}")