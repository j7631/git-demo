import pandas as pd
import requests
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
import os
import torch

# 读取Excel文件
# df = pd.read_excel('/opt/tiger/trl/consistency/data/data_v1.xlsx')
df = pd.read_csv('/opt/tiger/trl/consistency/git-demo/data/final2model.csv')
# 显示DataFrame的前几行
print("Original DataFrame:")
print(df.shape)

# 检查是否有需要的列
required_columns = ['subject', 'relation', 'object', 'q1', 'q2', 'q3']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' is missing in the Excel file")

# 初始化模型和处理器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf").to(device)
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

base_path = '/opt/tiger/trl/consistency/data/images'

# 定义一个函数来生成答案
def generate_answer(image_path, question):
    image = Image.open(image_path)
    inputs = processor(text=question, images=image, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, max_new_tokens=75)
    result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return result

# 定义保存路径和保存频率
output_path = 'result_total.csv'
save_frequency = 20  # 每处理20行保存一次

# 创建一个新的 DataFrame 来存储答案
results_df = pd.DataFrame(columns=df.columns.tolist() + ['question1_answer', 'question2_answer', 'question3_answer'])

# 遍历每一行生成答案
for index, row in df.iterrows():
    try:
        s = row['subject']
        r = row['relation']
        o = row['object']
        folder_name = f"{str(index) + '_' + s.replace(' ', '_')}"
        image_folder_path = os.path.join(base_path, folder_name)

        image_path = os.path.join(image_folder_path, "0.jpg")

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        print(index)
        # 从DataFrame中获取问题
        question1 = row['q1']
        question2 = row['q2']
        question3 = row['q3']

        # Step 1: 生成问题1的答案
        answer1 = generate_answer('/opt/tiger/trl/consistency/data/black.jpg', f"Please answer the question directly in a few words without repeating the question.[INST] <image>\n{question1} [/INST]")
        # Step 2: 生成问题2的答案
        answer2 = generate_answer(image_path, f"Please answer the question directly in a few words without repeating the question.[INST] <image>\n{question2} [/INST]")
        # Step 3: 生成问题3的答案
        answer3 = generate_answer(image_path, f"Please answer the question directly in a few words without repeating the question.[INST] <image>\n{question3} [/INST]")

        # 将答案添加到 DataFrame 中
        results_df = results_df.append({
            **row,
            'question1_answer': answer1,
            'question2_answer': answer2,
            'question3_answer': answer3
        }, ignore_index=True)

        # 定期保存中间结果到 CSV 文件
        if (index + 1) % save_frequency == 0:
            results_df.to_csv(output_path, index=False)

    except Exception as e:
        print(f"Error processing row {index}: {e}")
        continue

# 打印最终结果
print("\nFinal DataFrame with Answers:")
print(results_df.head())

# 保存最终结果到 CSV 文件
results_df.to_csv(output_path, index=False)