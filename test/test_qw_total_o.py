import pandas as pd
import requests
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
import os
import torch
import json
import re
import random
from PIL import Image
import requests
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# 读取CSV文件
df = pd.read_csv('/opt/tiger/trl/consistency/git-demo/data/final2model.csv')
df = df[:3]
print("Original DataFrame:")
print(df.shape)

# 检查是否有需要的列
required_columns = ['subject', 'relation', 'object', 'q1', 'q2', 'q3']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' is missing in the CSV file")

# 初始化模型和处理器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct").to(device)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

base_path = '/opt/tiger/trl/consistency/git-demo/data/images_'

# 定义一个函数来生成答案
def generate_answer(image_path, question):
    image = Image.open(image_path)
    inputs = processor(text=question, images=image, padding=True, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_new_tokens=150)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    result = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return result

# 从JSON文件读取alias字典
alias_file_path = '/opt/tiger/trl/consistency/git-demo/data/cleaned.ent2mq_wiki_alias.cleaned.json'
with open(alias_file_path, 'r', encoding='utf-8') as f:
    alias_dict = json.load(f)

def match_with_alias(answer, expected, alias_dict):
    if check_answer_in_output(answer, expected):
        return True
    
    if expected in alias_dict:
        for alias in alias_dict[expected]:
            if check_answer_in_output(answer, alias):
                return True
    return False

def check_answer_in_output(answer, expected):
    pattern = re.compile(re.escape(str(expected)), re.IGNORECASE)
    return bool(pattern.search(str(answer)))

def check_answer(row, alias_dict, question_col, answer_col, expected):
    answer = row[answer_col]
    return match_with_alias(answer, expected, alias_dict)

def get_correct_image_path(image_folder_path, question, expected_answer):
    for file_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, file_name)
        if os.path.isfile(image_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            answer = generate_answer(image_path, question)
            if match_with_alias(answer, expected_answer, alias_dict):
                return image_path
    return None

def deal_question(question):
    return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
# 定义保存路径和保存频率
output_path = 'result_total_qw_o.csv'
save_frequency = 20

# 创建一个新的 DataFrame 来存储答案
results_df = pd.DataFrame(columns=df.columns.tolist() + ['question1_answer', 'question2_answer', 'question3_answer'])

# 遍历每一行生成答案
for index, row in df.iterrows():
    try:
        s = row['subject']
        r = row['relation']
        o = row['object']
        print(s)
        print(r)
        print(o)
        first_index = row['first_index']
        folder_name = f"{str(first_index) + '_' + s.replace(' ', '_')}"
        image_folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(image_folder_path):
            print(f"Folder not found: {image_folder_path}")
            continue
        
        question1 = row['q1']
        question2 = row['q2']
        question3 = row['q3']
        
        
        correct_image_path = get_correct_image_path(image_folder_path, deal_question(f"Please answer the question directly in a few words without repeating the question.[INST] <image>\n{question2} [/INST]"), s)
        
        if correct_image_path:
            answer2 = generate_answer(correct_image_path, deal_question(f"Please answer the question directly in a few words without repeating the question.[INST] <image>\n{question2} [/INST]"))
            answer3 = generate_answer(correct_image_path, deal_question(f"Please answer the question directly in a few words without repeating the question.[INST] <image>\n{question3} [/INST]"))
            answer4 = generate_answer(correct_image_path, deal_question(f"Please answer the question directly in a few words without repeating the question.[INST] <image>\n {question2}And answer the following question step by step. {question3} [/INST]"))
            answer5 = generate_answer(correct_image_path, deal_question(f"Please answer the question directly in a few words without repeating the question.[INST] <image>\n{question3} \n Let's think step by step.[/INST]"))
            answer6 = generate_answer(correct_image_path, deal_question(f"Please answer the question directly in a few words without repeating the question.\n Let's think step by step.[INST] <image>\n{question3} [/INST]"))

        else:
            random_image_path = os.path.join(image_folder_path, random.choice(os.listdir(image_folder_path)))
            answer2 = generate_answer(random_image_path, deal_question(f"Please answer the question directly in a few words without repeating the question.[INST] <image>\n{question2} [/INST]"))
            answer3 = generate_answer(random_image_path, deal_question(f"Please answer the question directly in a few words without repeating the question.[INST] <image>\n{question3} [/INST]"))
            answer4 = generate_answer(random_image_path, deal_question(f"Please answer the question directly in a few words without repeating the question.[INST] <image>\n {question2}And answer the following question step by step. {question3} [/INST]"))
            answer5 = generate_answer(random_image_path, deal_question(f"Please answer the question directly in a few words without repeating the question.[INST] <image>\n{question3} \n Let's think step by step.[/INST]"))
            answer6 = generate_answer(random_image_path, deal_question(f"Please answer the question directly in a few words without repeating the question.\n Let's think step by step.[INST] <image>\n{question3} [/INST]"))

        answer1 = generate_answer('/opt/tiger/trl/consistency/git-demo/data/black.jpg', deal_question(f"Please answer the question directly in a few words without repeating the question.[INST] <image>\n{question1} [/INST]"))
        
        results_df = results_df.append({
            **row,
            'question1_answer': answer1,
            'question2_answer': answer2,
            'question3_answer': answer3,
            'question4_answer': answer4,
            'question5_answer': answer5,
            'question6_answer': answer6,
            'path': correct_image_path

        }, ignore_index=True)
        
        if (index + 1) % save_frequency == 0:
            results_df.to_csv(output_path, index=False)
    
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        continue

print("\nFinal DataFrame with Answers:")
print(results_df.head())

results_df.to_csv(output_path, index=False)

# 读取和检查模型输出
# df = pd.read_excel('/opt/tiger/trl/consistency/result/result_v1.xlsx')
# df = pd.read_csv('/opt/tiger/trl/consistency/git-demo/data/final2model.csv')


# df['问题1_答案是否正确'] = df.apply(lambda row: check_answer(row, alias_dict, 'q1', 'question1_answer', row['object']), axis=1)
# df['问题2_识别是否正确'] = df.apply(lambda row: check_answer(row, alias_dict, 'q2', 'question2_answer', row['subject']), axis=1)
# df['问题3_回答是否正确'] = df.apply(lambda row: check_answer(row, alias_dict, 'q3', 'question3_answer', row['object']), axis=1)

# df.to_excel('/opt/tiger/trl/consistency/result/result_v1_checked.xlsx', index=False)