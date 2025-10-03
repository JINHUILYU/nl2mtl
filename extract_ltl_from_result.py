import pandas as pd
import re

def extract_last_triple_backtick(text):
    # 使用正则表达式提取所有 ```包裹的内容
    matches = re.findall(r"```(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else ""

def process_excel(input_file, output_file):
    # 读取 Excel 文件
    df = pd.read_excel(input_file, engine='openpyxl')

    # 如果没有 'Result' 列，则报错
    if 'Result' not in df.columns:
        raise ValueError("Excel中未找到 'Result' 列")

    # 如果没有 'LTL' 列，则添加
    if 'LTL' not in df.columns:
        df['LTL'] = ""

    # 遍历每行提取并写入 LTL
    df['LTL'] = df['Result'].astype(str).apply(extract_last_triple_backtick)

    # 保存结果到输出文件
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"处理完成，已保存至：{output_file}")

if __name__ == "__main__":
    input_path = "data/output/collaborative/llm_collab-gpt-3.5-turbo-nl2spec-result-checked-new.xlsx"  # 输入文件名
    output_path = "data/output/collaborative/llm_collab-gpt-3.5-turbo-nl2spec-result-checked-new.xlsx" # 输出文件名，可与输入相同覆盖原文件
    process_excel(input_path, output_path)