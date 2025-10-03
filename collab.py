import copy
import re
from openai import OpenAI, base_url
from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv
import pandas as pd
import checker

# ===== 初始化 =====
def load_env():
    load_dotenv()

def create_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_URL"))
    # return OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_API_URL"))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ===== 基础函数 =====
def extract_final_answer(text):
    """
    从回复中提取最终答案，支持多种格式和跨行提取
    """
    # 匹配 "Repeat the final answer" 或类似提示后跟着的三个反引号之间的内容
    match = re.search(r"(?:Repeat the final answer|Final answer|LTL Formula).*?:?\s*```(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 匹配简单的三个反引号格式
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None

def call_openai_chat(client, messages):
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        # model="gpt-4o",
        # model="o3",
        # model="deepseek-chat",
        # model="deepseek-reasoner",
        messages=messages,
        temperature=0.7
    ).choices[0].message.content.strip()

# ===== 多模型生成阶段 =====
def generate_initial_answers(prompt, client):
    agents = {}
    answers = {}
    extracted_answers = {}

    print("\n📝 Collecting initial answers from agents...")

    for name in ["A", "B", "C"]:
        messages = [
            {"role": "system", "content": (
                f"You are Agent {name}, an expert in natural language to Linear Temporal Logic (LTL) conversion. "
                f"Please solve the following problem independently and precisely.\n\n"
                f"IMPORTANT: Format your response as follows:\n"
                f"1. First provide your detailed reasoning and analysis\n"
                f"2. End with: Repeat the final answer of the original question: ```\n[your_final_answer]\n```\n"
                f"3. Make sure your final answer is enclosed in triple backticks with newlines before and after\n"
                f"4. For LTL formulas, use standard notation with correct symbols\n"
                f"5. Your final answer between triple backticks must be the exact formula with no additional text"
            )},
            {"role": "user", "content": prompt}
        ]

        reply = call_openai_chat(client, messages)
        agents[name] = reply

        # 提取最终答案
        final_answer = extract_final_answer(reply)
        if final_answer:
            extracted_answers[name] = final_answer

        answers[name] = final_answer

        print(f"\nAgent {name}'s answer: {final_answer if final_answer else '[No clear answer extracted]'}")

    # 记录提取出的答案数量
    extraction_success = sum(1 for a in answers.values() if a is not None)
    print(f"\nSuccessfully extracted {extraction_success}/{len(agents)} answers")

    return agents, answers, extracted_answers

# ===== 信心评分阶段 =====
def score_confidence(prompt, answer, client):
    score_prompt = f"""Estimate your confidence (0 to 1) in the following answer's correctness:
Question: {prompt}
Answer:
{answer}
"""
    messages = [
        {"role": "system", "content": "You are a careful evaluator. Output only: Confidence: 0.x"},
        {"role": "user", "content": score_prompt}
    ]
    reply = call_openai_chat(client, messages)
    match = re.search(r"Confidence:\s*(0\.\d+)", reply)
    return float(match.group(1)) if match else 0.5

# ===== 修正轮阶段 =====
def run_refinement_round(prompt, client, raw_answers, extracted_answers):
    """
    Refinement round: Each Agent improves their answer after reviewing other Agents' responses
    """
    refined = {}
    refined_full_responses = {}
    print("\n🔄 Starting refinement round...")

    for name in raw_answers:
        # Build display information for other Agents' answers
        others_info = []
        for other_name in raw_answers:
            if other_name != name:
                other_answer = extracted_answers.get(other_name, "[No clear answer extracted]")
                others_info.append(f"Agent {other_name}'s answer:\n```\n{other_answer}\n```")

        others_display = "\n\n".join(others_info)

        review_prompt = f"""As Agent {name}, please improve your answer based on the responses from other Agents.

Original question:
{prompt}

Your initial answer:
{raw_answers[name]}

Other Agents' answers:
{others_display}

Please analyze the similarities and differences between other answers and your answer, and improve your response based on this information. If you believe your answer is correct, please explain why.

Please format your improved answer as follows:
1. Analysis of other answers
2. Improvement process
3. End with "Repeat the final answer of the original question: ```" and close with "```" to provide your final answer
"""

        messages = [
            {"role": "system", "content": (
                f"You are Agent {name}, an expert in converting natural language to Linear Temporal Logic (LTL). "
                f"Please carefully analyze other Agents' answers, identify their strengths and weaknesses, and then propose an improved answer. "
                f"Ensure your final answer is the most precise LTL formula using standard notation."
            )},
            {"role": "user", "content": review_prompt}
        ]

        reply = call_openai_chat(client, messages)
        refined_full_responses[name] = reply

        # Extract the improved final answer
        final_answer = extract_final_answer(reply)
        if final_answer:
            refined[name] = final_answer
            print(f"\nAgent {name}'s improved answer: {final_answer}")
        else:
            # If extraction fails, use the original answer
            refined[name] = extracted_answers.get(name, "Could not extract answer")
            print(f"\nAgent {name} - Could not extract improved answer, using original answer")

    return refined, refined_full_responses

# ===== 投票阶段 =====
def run_voting(prompt, client, refined_answers, refined_full_responses):
    """
    Voting phase: Each Agent evaluates all answers and votes for the best one
    """
    votes = {}
    vote_reasons = {}
    print("\n🗳️ Starting voting process...")

    # Prepare a summary of candidate answers for voting
    candidates_summary = ""
    for name, answer in refined_answers.items():
        candidates_summary += f"Option {name}: ```\n{answer}\n```\n\n"

    # Each Agent casts their vote
    for name in refined_answers:
        vote_prompt = f"""As Agent {name}, you have analyzed the original question and provided your answer. Now you need to select the most accurate answer from all candidate answers.

Original question:
{prompt}

Candidate answers:
{candidates_summary}

Your task is to objectively evaluate all candidate answers (including your own) and vote for the one you believe is most accurate.
Please format your voting result as follows:

Analysis:
[Analyze the strengths and weaknesses of each candidate answer]

My vote goes to: [A/B/C]
Reason: [Brief explanation for your choice]
"""

        messages = [
            {"role": "system", "content": f"You are Agent {name}, an LTL expert. Please evaluate all candidate answers objectively, even if the best answer is not your own."},
            {"role": "user", "content": vote_prompt}
        ]

        vote_reply = call_openai_chat(client, messages)
        print(f"\nAgent {name}'s vote:\n{vote_reply}")

        # Extract voting result
        vote_match = re.search(r"My vote goes to:\s*(?:\*\*?)?([ABC])(?:\*\*?)?", vote_reply)
        if vote_match:
            vote = vote_match.group(1)
            votes[name] = vote

            # Extract voting reason
            reason_match = re.search(r"Reason:\s*(.*?)$", vote_reply, re.DOTALL)
            vote_reasons[name] = reason_match.group(1).strip() if reason_match else "No reason provided"

            print(f"Agent {name} voted for option {vote}")
        else:
            print(f"⚠️ Could not determine Agent {name}'s vote")

    # Tally the votes
    vote_count = {}
    for vote in votes.values():
        vote_count[vote] = vote_count.get(vote, 0) + 1

    print("\n📊 Voting results:")
    for option, count in vote_count.items():
        print(f"Option {option}: {count} vote(s)")

    # Determine if there's a winner
    if vote_count:
        max_votes = max(vote_count.values())
        winners = [option for option, count in vote_count.items() if count == max_votes]

        if len(winners) == 1:
            winner = winners[0]
            return winner, refined_answers[winner], vote_count, votes, vote_reasons

    # If there's no clear winner or voting is invalid, return None
    return None, None, vote_count, votes, vote_reasons

# ===== 仲裁阶段 =====
def run_arbitration(prompt, client, refined_answers, vote_results):
    """
    Arbitration phase: When voting results in a tie, an arbitrator determines the final answer
    """
    print("\n👨‍⚖️ Starting arbitration process...")

    # Prepare a summary of candidate answers
    candidates_info = []
    vote_count, votes, vote_reasons = vote_results

    for name, answer in refined_answers.items():
        votes_for_this = sum(1 for v in votes.values() if v == name)
        candidate_info = f"Option {name}:\nAnswer: ```\n{answer}\n```\nVotes received: {votes_for_this}"

        # Add voting reasons
        for voter, vote in votes.items():
            if vote == name:
                candidate_info += f"\nAgent {voter}'s reason for voting: {vote_reasons.get(voter, 'No reason provided')}"

        candidates_info.append(candidate_info)

    arbitration_prompt = f"""As an impartial arbitrator, your task is to select the best answer from multiple candidate answers. These answers have inconsistencies or tied votes, and you need to make the final decision.

Original question:
{prompt}

Candidate answers and voting results:
{"".join(candidates_info)}

Please carefully analyze each candidate answer, considering its correctness, completeness, and precision, then select the best answer.

Please format your decision as follows:
The best answer is: [A/B/C]
Reason: [Detailed explanation for your choice]
"""

    messages = [
        {"role": "system", "content": "You are an impartial arbitrator responsible for selecting the best answer from multiple expert responses, especially in cases of ties or disputes."},
        {"role": "user", "content": arbitration_prompt}
    ]

    arbitration_reply = call_openai_chat(client, messages)
    print(f"\n👨‍⚖️ Arbitration result:\n{arbitration_reply}")

    # Extract arbitration result
    match = re.search(r"The best answer is:\s*([ABC])", arbitration_reply)
    if match:
        winner = match.group(1)
        print(f"Arbitrator selected option {winner}")
        return winner, refined_answers[winner], arbitration_reply

    # If unable to extract a result, return None
    print("⚠️ Unable to determine a winner from the arbitration result")
    return None, None, arbitration_reply

# ===== 主流程 =====
def collaborative_decision(prompt):
    """
    Agent-based collaborative decision-making process
    """
    load_env()
    client = create_client()
    result_summary = ""

    # 1. Initial answers phase
    print("\n=== Phase 1: Initial Answer Generation ===")
    raw_outputs, answers, extracted_answers = generate_initial_answers(prompt, client)

    result_summary += "\n=== Phase 1: Initial Answer Generation ===\n"
    for name, answer in extracted_answers.items():
        result_summary += f"Agent {name}'s answer: {answer}\n"

    # Check if initial consensus is reached
    if (len(extracted_answers) >= 2 and len(set(extracted_answers.values())) == 1) \
           or (len(extracted_answers) == 3 and all(checker.spot_spider(list(extracted_answers.values())[0], list(extracted_answers.values())[i]) for i in range(1, 3))):
        consensus_answer = list(extracted_answers.values())[0]
        print(f"\n✅ Strong consensus reached! All Agents gave the same answer:\n```\n{consensus_answer}\n```")
        result_summary += f"\n✅ Strong consensus reached! Final answer:\n```\n{consensus_answer}\n```"
        return result_summary

    # 2. Refinement phase
    print("\n=== Phase 2: Answer Refinement ===")
    refined_answers, refined_full_responses = run_refinement_round(prompt, client, raw_outputs, extracted_answers)

    result_summary += "\n=== Phase 2: Answer Refinement ===\n"
    for name, answer in refined_answers.items():
        result_summary += f"Agent {name}'s refined answer: {answer}\n"

    # Check if consensus is reached after refinement
    if (len(refined_answers) >= 2 and len(set(refined_answers.values())) == 1) \
            or (len(refined_answers) == 3 and all(checker.spot_spider(list(refined_answers.values())[0], list(refined_answers.values())[i]) for i in range(1, 3))):
        consensus_answer = list(refined_answers.values())[0]
        print(f"\n✅ Consensus reached after refinement! Final answer:\n```\n{consensus_answer}\n```")
        result_summary += f"\n✅ Consensus reached after refinement! Final answer:\n```\n{consensus_answer}\n```"
        return result_summary

    # 3. Voting phase
    print("\n=== Phase 3: Voting Selection ===")
    winner, winning_answer, vote_count, votes, vote_reasons = run_voting(prompt, client, refined_answers, refined_full_responses)

    result_summary += "\n=== Phase 3: Voting Selection ===\n"
    if vote_count:
        for option, count in vote_count.items():
            result_summary += f"Option {option}: {count} vote(s)\n"

    if winner:
        print(f"\n🏆 Voting result: Agent {winner}'s answer wins!\nFinal answer:\n```\n{winning_answer}\n```")
        result_summary += f"\n🏆 Voting result: Agent {winner}'s answer wins!\nFinal answer:\n```\n{winning_answer}\n```"
        return result_summary

    # 4. Arbitration phase
    print("\n=== Phase 4: Arbitration Decision ===")
    vote_results = (vote_count, votes, vote_reasons)
    arb_winner, arb_answer, arbitration_details = run_arbitration(prompt, client, refined_answers, vote_results)

    result_summary += "\n=== Phase 4: Arbitration Decision ===\n"
    if arb_winner:
        print(f"\n⚖️ Arbitration result: Agent {arb_winner}'s answer was selected\nFinal answer:\n```\n{arb_answer}\n```")
        result_summary += f"Arbitrator selected Agent {arb_winner}'s answer\n\n⚖️ Final answer:\n```\n{arb_answer}\n```"
    else:
        # If arbitration fails, use the answer with highest confidence
        confidence_scores = {}
        for name, ans in refined_answers.items():
            confidence_scores[name] = score_confidence(prompt, ans, client)

        best_agent = max(confidence_scores.items(), key=lambda x: x[1])[0]
        best_answer = refined_answers[best_agent]

        print(f"\n⚠️ Arbitration failed, using answer with highest confidence\nAgent {best_agent}'s answer (confidence: {confidence_scores[best_agent]:.2f}):\n```\n{best_answer}\n```")
        result_summary += f"\n⚠️ Arbitration failed, using answer with highest confidence\nAgent {best_agent} (confidence: {confidence_scores[best_agent]:.2f}):\n```\n{best_answer}\n```"

    return result_summary

# ===== 示例调用 =====
if __name__ == "__main__":
    batch = True
    if not batch:
        Question = "Globally, if a is true, then b will be true in the next step."
        # open the base prompt file
        with open("config/prompts/base_prompt.txt") as f:
            # read the base prompt content
            base_prompt = f.read()  # 任务描述 + 输出格式 + 知识库 + 样例 + 执行要求 + 问题
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
        result = collaborative_decision(base_prompt)
        print(result)
    else:
        print("\n开始批量处理数据集...")

        # 读取待处理的数据集
        dataset_df = pd.read_excel("data/input/nl2spec-dataset.xlsx")  # nl2spec dataset
        requirements = dataset_df["Requirement"].tolist()  # NL

        # dataset_df = pd.read_excel("data/input/dataset.xlsx")  # dataset
        # requirements = dataset_df["Requirement"].tolist()  # Requirement

        # 创建结果列表
        results = []
        result_file = "data/output/collaborative/llm_collab-gpt-3.5-turbo-nl2spec-result-checked-new.xlsx"

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

            result = collaborative_decision(current_prompt)
            print(result)

            # 将结果添加到列表
            results.append({"Requirement": req, "Result": result})

            # 每5个样本保存一次中间结果（防止运行中断导致全部丢失）
            if (i + 1) % 5 == 0 or i == len(requirements) - 1:
                pd.DataFrame(results).to_excel(result_file, index=False)
                print(f"已保存 {len(results)} 个结果到 {result_file}")

        print(f"\n✅ 批量处理完成! 已处理 {len(results)} 个需求，结果已保存到 {result_file}")