import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import os
import traceback


def spot_spider(first_formula: str, second_formula: str, max_retries=3):
    """
    使用 Selenium 爬取 Spot LTL 比较结果，添加重试机制
    """
    for attempt in range(max_retries):
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')

            driver = webdriver.Chrome(options=options)
            driver.get("https://spot.lre.epita.fr/app/")

            # 点击 Compare 标签
            compare_tab = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Compare']]"))
            )
            compare_tab.click()

            inputs = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "input[type='text']"))
            )
            inputs[0].clear()
            inputs[0].send_keys(first_formula)
            inputs[1].clear()
            inputs[1].send_keys(second_formula + Keys.RETURN)

            # 等待结果段落
            result_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "/html/body/div/div/div/div[4]/p"))
            )
            result_text = result_element.text.strip()
            print(f"比较结果：{result_text}")
            driver.quit()
            return result_text
            
        except Exception as e:
            print(f"尝试 {attempt+1}/{max_retries} 失败: {e}")
            traceback.print_exc()
            if driver:
                try:
                    driver.quit()
                except:
                    pass
            
            # 最后一次尝试失败后返回错误信息
            if attempt == max_retries - 1:
                return "Error comparing formulas"
                
            # 等待几秒后重试
            time.sleep(3)


def process_excel(file_path: str, output_path: str, save_interval=5):
    """
    处理Excel文件，添加断点续传和定期保存功能
    """
    df = pd.read_excel(file_path)
    
    # 检查是否已有进度，可以继续处理
    start_row = 0
    if os.path.exists(output_path):
        existing_df = pd.read_excel(output_path)
        # 找到最后一个已处理的行
        if 'Spot Check Result' in existing_df.columns:
            processed_rows = existing_df['Spot Check Result'].notna()
            if processed_rows.any():
                start_row = processed_rows.sum()
                df = existing_df.copy()
                print(f"继续从第 {start_row+1} 行开始处理...")

    for idx, row in df.iloc[start_row:].iterrows():
        ltl = str(row['LTL'])
        ref = str(row['nl2spec dataset answer'])
        
        if ref == "Correct":
            print(f"跳过行 {idx+1}，因为答案标记为 'Correct'")
            break
            
        print(f"Processing row {idx+1}: {ltl} vs {ref}")
        
        result = spot_spider(ltl, ref)
        df.at[idx, 'Spot Check Result'] = result
        df.at[idx, 'Correct'] = 'YES' if "equivalent" in result else 'NO'
        
        # 定期保存结果，防止中途崩溃导致数据丢失
        if (idx - start_row + 1) % save_interval == 0:
            df.to_excel(output_path, index=False)
            print(f"✅ 已保存中间结果到：{output_path} (行 {idx+1})")
    
    # 最终保存
    df.to_excel(output_path, index=False)
    print(f"\n✅ 已保存完整结果到：{output_path}")


if __name__ == "__main__":
    process_excel("data/output/end2end/end2end-deepseek-v3-nl2spec-result-checked-new.xlsx",
                  "data/output/end2end-deepseek-v3-nl2spec-result-checked-new_check.xlsx")