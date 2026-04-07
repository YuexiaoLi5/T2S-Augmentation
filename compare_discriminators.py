"""
比较训练前后的一致性和流畅性判别器性能
"""

import os
import json
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from datasets import Dataset

# 禁用wandb
os.environ["WANDB_DISABLED"] = "true"

# 配置
MODEL_NAME = "microsoft/codebert-base"
CODEBERT_MODEL = "microsoft/codebert-base"
QWEN_MODEL = "D:/HF_Models/Qwen/Qwen2.5-Coder-3B-Instruct"
MAX_LENGTH = 512

# 一致性标签
LABEL_NAMES = [
    'logical_error',
    'memory_error', 
    'omission',
    'invalid_condition',
    'infinite_loop_error',
]

def parse_label_vector(labels_str):
    """将标签字符串转换为多标签向量"""
    labels_str = labels_str.strip().strip("'").strip('"')
    
    if ',' in labels_str:
        parts = [p.strip().strip("'").strip('"') for p in labels_str.split(',')]
    else:
        parts = [labels_str.strip().strip("'").strip('"')]
    
    normalized = []
    for p in parts:
        lp = p.lower()
        if 'logical' in lp:
            normalized.append('logical_error')
        elif 'memory' in lp:
            normalized.append('memory_error')
        elif 'omission' in lp:
            normalized.append('omission')
        elif 'invalid' in lp or 'condition' in lp:
            normalized.append('invalid_condition')
        elif 'infinite' in lp or 'loop' in lp:
            normalized.append('infinite_loop_error')
    
    label_vector = [0] * len(LABEL_NAMES)
    for label in normalized:
        if label in LABEL_NAMES:
            label_vector[LABEL_NAMES.index(label)] = 1
    
    return label_vector


def load_test_data():
    """加载测试数据"""
    with open("data/origin/test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"测试样本数: {len(test_data)}")
    return test_data


def create_fluency_samples(test_data):
    """创建流利度测试样本"""
    samples = []
    noise_methods = [
        lambda c: '\n'.join(c.split('\n')[:-1]) if len(c.split('\n')) > 3 else c,
        lambda c: c.replace(';', '').replace('{', '').replace('}', ''),
    ]
    
    for item in test_data:
        code = item['code']
        
        # 正样本（原始代码）
        samples.append({
            'text': code,
            'label': 1
        })
        
        # 负样本
        for method in noise_methods:
            noisy_code = method(code)
            if noisy_code != code:
                samples.append({
                    'text': noisy_code,
                    'label': 0
                })
    
    return samples


def create_alignment_samples(test_data):
    """创建一致性测试样本"""
    samples = []
    
    for item in test_data:
        code = item['code']
        text = item['text']
        
        if "Label:" in text:
            label_start = text.find("Label:") + 6
            label_end = text.find("\n", label_start)
            if label_end != -1:
                label_str = text[label_start:label_end].strip()
                label_vector = parse_label_vector(label_str)
                
                samples.append({
                    'text': code,
                    'labels': [float(x) for x in label_vector]
                })
    
    return samples


def preprocess_data(samples, tokenizer, is_multi_label=False):
    """预处理数据"""
    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH
        )
    
    dataset = Dataset.from_list(samples)
    dataset = dataset.map(preprocess_function, batched=True)
    return dataset


def evaluate_fluency_model(model, tokenizer, test_dataset):
    """评估流利度模型"""
    model.eval()
    
    true_labels = []
    predicted_labels = []
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            
            inputs = {
                'input_ids': torch.tensor([sample['input_ids']]),
                'attention_mask': torch.tensor([sample['attention_mask']])
            }
            
            outputs = model(**inputs)
            predicted = torch.argmax(outputs.logits, dim=-1).item()
            
            true_labels.append(sample['label'])
            predicted_labels.append(predicted)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='binary')
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def evaluate_alignment_model(model, tokenizer, test_dataset):
    """评估一致性模型"""
    model.eval()
    
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            
            inputs = tokenizer(
                sample['text'],
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            probabilities = torch.sigmoid(logits)
            predicted = (probabilities > 0.5).int().squeeze().tolist()
            
            if isinstance(predicted, int):
                predicted = [predicted]
            
            true_labels.append(sample['labels'])
            predicted_labels.append(predicted)
    
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    # 计算每个标签的F1
    f1_scores = []
    for i in range(len(LABEL_NAMES)):
        if np.sum(true_labels[:, i]) > 0:
            f1 = f1_score(true_labels[:, i], predicted_labels[:, i], average='binary')
            f1_scores.append(f1)
        else:
            f1_scores.append(0.0)
    
    macro_f1 = np.mean(f1_scores)
    micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'per_label_f1': dict(zip(LABEL_NAMES, f1_scores)),
    }


def run_comparison():
    """运行比较"""
    print("=" * 80)
    print("判别器性能比较：训练前 vs 训练后")
    print("=" * 80)
    
    # 1. 加载测试数据
    print("\n1. 加载测试数据...")
    test_data = load_test_data()
    
    # 2. 加载tokenizer
    print("\n2. 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 准备流利度测试数据
    print("\n3. 准备流利度测试数据...")
    fluency_samples = create_fluency_samples(test_data)
    fluency_dataset = preprocess_data(fluency_samples, tokenizer)
    print(f"流利度测试样本数: {len(fluency_samples)}")
    
    # 4. 准备一致性测试数据
    print("\n4. 准备一致性测试数据...")
    alignment_samples = create_alignment_samples(test_data)
    alignment_dataset = preprocess_data(alignment_samples, tokenizer)
    print(f"一致性测试样本数: {len(alignment_samples)}")
    
    results = {}
    
    # ========== 流利度判别器比较 ==========
    print("\n" + "=" * 60)
    print("流利度判别器比较")
    print("=" * 60)
    
    # 未训练模型
    print("\n评估未训练模型...")
    untrained_fluency = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, problem_type="single_label_classification"
    )
    untrained_fluency_results = evaluate_fluency_model(untrained_fluency, tokenizer, fluency_dataset)
    
    # 训练后模型
    print("评估训练后模型...")
    try:
        trained_fluency = AutoModelForSequenceClassification.from_pretrained(
            "output/fluency_discriminator_codebert/checkpoint-200",
            num_labels=2, 
            problem_type="single_label_classification"
        )
        trained_fluency_results = evaluate_fluency_model(trained_fluency, tokenizer, fluency_dataset)
    except Exception as e:
        print(f"加载训练后模型失败: {e}")
        trained_fluency_results = untrained_fluency_results
    
    print("\n流利度判别器结果:")
    print(f"{'模型':<20} | {'准确率':<8} | {'F1分数':<8} | {'精确率':<8} | {'召回率':<8}")
    print("-" * 60)
    print(f"{'未训练':<20} | {untrained_fluency_results['accuracy']:.4f}  | {untrained_fluency_results['f1']:.4f}  | {untrained_fluency_results['precision']:.4f}  | {untrained_fluency_results['recall']:.4f}")
    print(f"{'训练后':<20} | {trained_fluency_results['accuracy']:.4f}  | {trained_fluency_results['f1']:.4f}  | {trained_fluency_results['precision']:.4f}  | {trained_fluency_results['recall']:.4f}")
    
    results['fluency'] = {
        'untrained': untrained_fluency_results,
        'trained': trained_fluency_results,
    }
    
    # ========== 一致性判别器比较 ==========
    print("\n" + "=" * 60)
    print("一致性判别器比较")
    print("=" * 60)
    
    # 未训练模型
    print("\n评估未训练模型...")
    untrained_alignment = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(LABEL_NAMES),
        problem_type="multi_label_classification"
    )
    untrained_alignment_results = evaluate_alignment_model(untrained_alignment, tokenizer, alignment_dataset)
    
    # 训练后模型
    print("评估训练后模型...")
    try:
        trained_alignment = AutoModelForSequenceClassification.from_pretrained(
            "output/alignment_discriminator_codebert/checkpoint-330",
            num_labels=len(LABEL_NAMES),
            problem_type="multi_label_classification"
        )
        trained_alignment_results = evaluate_alignment_model(trained_alignment, tokenizer, alignment_dataset)
    except Exception as e:
        print(f"加载训练后模型失败: {e}")
        trained_alignment_results = untrained_alignment_results
    
    print("\n一致性判别器结果:")
    print(f"{'模型':<20} | {'Macro F1':<10} | {'Micro F1':<10}")
    print("-" * 50)
    print(f"{'未训练':<20} | {untrained_alignment_results['macro_f1']:.4f}     | {untrained_alignment_results['micro_f1']:.4f}")
    print(f"{'训练后':<20} | {trained_alignment_results['macro_f1']:.4f}     | {trained_alignment_results['micro_f1']:.4f}")
    
    print("\n各标签F1分数:")
    print("-" * 40)
    for label in LABEL_NAMES:
        untrained_f1 = untrained_alignment_results['per_label_f1'].get(label, 0)
        trained_f1 = trained_alignment_results['per_label_f1'].get(label, 0)
        print(f"{label:<20} | 未训练: {untrained_f1:.4f} | 训练后: {trained_f1:.4f}")
    
    results['alignment'] = {
        'untrained': untrained_alignment_results,
        'trained': trained_alignment_results,
    }
    
    # 保存结果
    with open("output/discriminator_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: output/discriminator_comparison_results.json")
    
    # 总结提升
    print("\n" + "=" * 80)
    print("性能提升总结")
    print("=" * 80)
    
    flu_f1_improvement = trained_fluency_results['f1'] - untrained_fluency_results['f1']
    align_f1_improvement = trained_alignment_results['macro_f1'] - untrained_alignment_results['macro_f1']
    
    print(f"\n流利度判别器 F1提升: {flu_f1_improvement:+.4f}")
    print(f"一致性判别器 Macro F1提升: {align_f1_improvement:+.4f}")
    
    return results


if __name__ == "__main__":
    run_comparison()
