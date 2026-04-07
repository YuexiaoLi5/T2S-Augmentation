"""
一致性判别器训练脚本 - 修复版v2
1. 移除unknown标签
2. 修复F1计算逻辑：使用样本级匹配度
"""

import os
import json
import random
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# 禁用wandb
os.environ["WANDB_DISABLED"] = "true"

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# 配置
MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 15
WARMUP_STEPS = 200
OUTPUT_DIR = "output/alignment_discriminator_fixed_v2"
DATA_DIR = "data/origin"
EVAL_STEPS = 50

# 多标签错误类型（移除unknown）
LABEL_NAMES = [
    'logical_error',
    'memory_error', 
    'omission',
    'invalid_condition',
    'infinite_loop_error',
    # 'unknown',  # 移除unknown标签
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
        # 移除unknown处理
    
    # 创建5维向量（移除unknown）
    label_vector = [0] * len(LABEL_NAMES)
    for label in normalized:
        if label in LABEL_NAMES:
            label_vector[LABEL_NAMES.index(label)] = 1
    
    return label_vector

def load_and_process_data():
    """加载并处理训练和验证数据"""
    
    # 加载训练数据
    with open(os.path.join(DATA_DIR, "train.json"), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 加载验证数据
    with open(os.path.join(DATA_DIR, "test.json"), 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    print(f"训练样本数: {len(train_data)}")
    print(f"验证样本数: {len(eval_data)}")
    
    # 处理训练数据
    train_samples = []
    
    for item in train_data:
        code = item['code']
        text = item['text']
        
        # 从text字段提取标签
        if "Label:" in text:
            label_start = text.find("Label:") + 6
            label_end = text.find("\n", label_start)
            if label_end != -1:
                label_str = text[label_start:label_end].strip()
                label_vector = parse_label_vector(label_str)
                
                # 创建样本（标签转换为浮点数）
                train_samples.append({
                    'text': code,
                    'labels': [float(x) for x in label_vector]
                })
    
    # 处理验证数据
    eval_samples = []
    
    for item in eval_data:
        code = item['code']
        text = item['text']
        
        if "Label:" in text:
            label_start = text.find("Label:") + 6
            label_end = text.find("\n", label_start)
            if label_end != -1:
                label_str = text[label_start:label_end].strip()
                label_vector = parse_label_vector(label_str)
                
                eval_samples.append({
                    'text': code,
                    'labels': [float(x) for x in label_vector]
                })
    
    print(f"训练集有效样本数: {len(train_samples)}")
    print(f"验证集有效样本数: {len(eval_samples)}")
    
    # 显示标签分布
    if train_samples:
        label_counts = [0] * len(LABEL_NAMES)
        for sample in train_samples:
            for i, val in enumerate(sample['labels']):
                if val == 1:
                    label_counts[i] += 1
        
        print("\n训练集标签分布:")
        for i, count in enumerate(label_counts):
            print(f"  {LABEL_NAMES[i]}: {count}")
    
    return Dataset.from_list(train_samples), Dataset.from_list(eval_samples)

def sample_level_f1(true_labels, predicted_probs, threshold=0.5):
    """计算样本级F1分数（正确的逻辑）"""
    
    # 应用阈值
    predicted_labels = (predicted_probs > threshold).astype(int)
    
    # 样本级匹配度计算
    sample_f1_scores = []
    
    for i in range(len(true_labels)):
        true_vec = true_labels[i]
        pred_vec = predicted_labels[i]
        
        # 计算TP, FP, FN
        tp = np.sum((true_vec == 1) & (pred_vec == 1))
        fp = np.sum((true_vec == 0) & (pred_vec == 1))
        fn = np.sum((true_vec == 1) & (pred_vec == 0))
        
        # 计算样本级F1
        if tp + fp + fn == 0:
            sample_f1 = 1.0  # 完美匹配
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if precision + recall > 0:
                sample_f1 = 2 * precision * recall / (precision + recall)
            else:
                sample_f1 = 0
        
        sample_f1_scores.append(sample_f1)
    
    return np.mean(sample_f1_scores)

def compute_metrics(eval_pred):
    """计算多标签分类的评估指标（修复版）"""
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    predictions, labels = eval_pred
    
    # 应用sigmoid获取概率
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    
    # 计算样本级F1（正确的指标）
    sample_f1 = sample_level_f1(labels, predictions)
    
    # 计算宏平均指标（作为参考）
    predicted_labels = (predictions > 0.5).astype(int)
    
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for i in range(len(LABEL_NAMES)):
        if len(labels) > 0 and np.sum(labels[:, i]) > 0:  # 确保标签存在
            f1 = f1_score(labels[:, i], predicted_labels[:, i], average='binary', zero_division=0)
            precision = precision_score(labels[:, i], predicted_labels[:, i], average='binary', zero_division=0)
            recall = recall_score(labels[:, i], predicted_labels[:, i], average='binary', zero_division=0)
            
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
        else:
            f1_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
    
    # 计算宏平均指标
    macro_f1 = np.mean(f1_scores)
    macro_precision = np.mean(precision_scores)
    macro_recall = np.mean(recall_scores)
    
    # 计算微平均指标
    micro_f1 = f1_score(labels, predicted_labels, average='micro', zero_division=0)
    micro_precision = precision_score(labels, predicted_labels, average='micro', zero_division=0)
    micro_recall = recall_score(labels, predicted_labels, average='micro', zero_division=0)
    
    return {
        'sample_f1': sample_f1,  # 主要指标：样本级匹配度
        'macro_f1': macro_f1,     # 参考指标
        'micro_f1': micro_f1,     # 参考指标
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'f1_logical_error': f1_scores[0],
        'f1_memory_error': f1_scores[1],
        'f1_omission': f1_scores[2],
        'f1_invalid_condition': f1_scores[3],
        'f1_infinite_loop_error': f1_scores[4],
    }

def main():
    """主训练函数"""
    
    print("=" * 70)
    print("一致性判别器训练 - 修复版v2")
    print("=" * 70)
    print("修复内容:")
    print("1. 移除unknown标签（所有样本都是0）")
    print("2. 使用样本级F1作为主要指标")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    train_dataset, eval_dataset = load_and_process_data()
    
    if len(train_dataset) == 0:
        print("错误: 没有有效的训练数据")
        return
    
    # 2. 加载模型和tokenizer
    print("\n2. 加载CodeBERT模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),  # 现在只有5个标签
        problem_type="multi_label_classification"
    )
    
    # 3. 数据预处理
    print("\n3. 数据预处理...")
    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH
        )
    
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    
    # 4. 训练参数（使用sample_f1作为主要指标）
    print("\n4. 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        logging_steps=25,
        warmup_steps=WARMUP_STEPS,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_sample_f1",  # 使用样本级F1
        greater_is_better=True,
        report_to=[],
        dataloader_pin_memory=False,
        gradient_accumulation_steps=2,
        lr_scheduler_type="cosine",
    )
    
    # 5. 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 6. 创建Trainer
    print("\n5. 创建Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 7. 开始训练
    print("\n6. 开始训练...")
    print(f"训练配置:")
    print(f"  - 模型: {MODEL_NAME}")
    print(f"  - 标签数量: {len(LABEL_NAMES)} (移除unknown)")
    print(f"  - 主要指标: sample_f1 (样本级匹配度)")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - 学习率: {LEARNING_RATE}")
    
    trainer.train()
    
    # 8. 保存最终模型
    print("\n7. 保存最终模型...")
    trainer.save_model()
    
    # 9. 最终评估
    print("\n8. 最终评估...")
    eval_results = trainer.evaluate()
    print("最终评估结果:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("🎉 修复版一致性判别器训练完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()