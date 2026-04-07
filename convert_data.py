# -*- coding: utf-8 -*-
import json

# 读取原始数据
with open('data/origin/python_data.json', 'r', encoding='utf-8') as f:
    python_data = json.load(f)

with open('data/origin/java_data.json', 'r', encoding='utf-8') as f:
    java_data = json.load(f)

print(f"Python: {len(python_data)}, Java: {len(java_data)}")

# 标签映射
label_map = {
    'Logical Error': 'logical-error',
    'Memory Error': 'memory-error',
    'Invalid Condition': 'invalid-condition',
    'Omission': 'omission',
}

def convert(item, lang):
    labels = item.get('labels', [])
    if isinstance(labels, str):
        labels = [labels]
    
    # 标准化标签
    new_labels = []
    for l in labels:
        l = l.strip()
        new_labels.append(label_map.get(l, l))
    
    err = item.get('error_code_snippet', '')
    corr = item.get('correct_code_snippet', '')
    
    if not err or not corr:
        return None
    
    prompt = f"""Task: Fix the {lang} code error.
Error type: {new_labels}
Error code:
{err}

Provide the correct code:"""
    
    return {
        'prompt': prompt,
        'error_code': err,
        'correct_code': corr,
        'labels': new_labels,
        'language': lang,
    }

# 转换
all_data = []
for item in python_data:
    r = convert(item, 'python')
    if r:
        all_data.append(r)

for item in java_data:
    r = convert(item, 'java')
    if r:
        all_data.append(r)

print(f"Total converted: {len(all_data)}")

# 保存
with open('data/origin/ppo_train.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print("Saved to data/origin/ppo_train.json")
print(f"First sample: {all_data[0]}")

