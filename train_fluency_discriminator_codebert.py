"""
流利度判别器训练脚本 - 使用CodeBERT，添加验证步骤
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
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 100
OUTPUT_DIR = "output/fluency_discriminator_codebert"
DATA_DIR = "data/origin"
EVAL_STEPS = 100  # 每100步验证一次

# ============== 数据增强函数 ==============
def add_noise_methods():
    """多样化的负样本扰动方法"""
    
    def truncate_lines(code):
        lines = code.split('\n')
        if len(lines) > 3:
            return '\n'.join(lines[:random.randint(1, len(lines)-2)])
        return code
    
    def remove_random_lines(code):
        lines = code.split('\n')
        if len(lines) > 3:
            num_to_remove = random.randint(1, min(3, len(lines)-2))
            indices_to_remove = random.sample(range(len(lines)), num_to_remove)
            return '\n'.join([line for i, line in enumerate(lines) if i not in indices_to_remove])
        return code
    
    def shuffle_lines(code):
        lines = code.split('\n')
        if len(lines) > 3:
            # 保持第一行和最后一行不变，只打乱中间部分
            first_line = lines[0]
            last_line = lines[-1]
            middle_lines = lines[1:-1]
            random.shuffle(middle_lines)
            return '\n'.join([first_line] + middle_lines + [last_line])
        return code
    
    def add_syntax_errors(code):
        # 添加简单的语法错误
        errors = [
            lambda c: c.replace(';', ''),  # 删除分号
            lambda c: c.replace('{', ''),   # 删除大括号
            lambda c: c.replace('}', ''),
            lambda c: c.replace('(', ''),  # 删除括号
            lambda c: c.replace(')', ''),
        ]
        error_func = random.choice(errors)
        return error_func(code)
    
    return [truncate_lines, remove_random_lines, shuffle_lines, add_syntax_errors]

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
    noise_methods = add_noise_methods()
    
    for item in train_data:
        code = item['code']
        
        # 正样本（原始代码）
        train_samples.append({
            'text': code,
            'label': 1  # 流利
        })
        
        # 负样本（添加噪声）
        for method in noise_methods:
            noisy_code = method(code)
            if noisy_code != code:  # 确保有变化
                train_samples.append({
                    'text': noisy_code,
                    'label': 0  # 不流利
                })
    
    # 处理验证数据
    eval_samples = []
    for item in eval_data:
        code = item['code']
        
        # 正样本
        eval_samples.append({
            'text': code,
            'label': 1
        })
        
        # 负样本
        for method in noise_methods[:2]:  # 验证集使用较少的噪声方法
            noisy_code = method(code)
            if noisy_code != code:
                eval_samples.append({
                    'text': noisy_code,
                    'label': 0
                })
    
    print(f"训练集总样本数: {len(train_samples)}")
    print(f"验证集总样本数: {len(eval_samples)}")
    
    return Dataset.from_list(train_samples), Dataset.from_list(eval_samples)

def main():
    """主训练函数"""
    
    print("=" * 60)
    print("流利度判别器训练 - CodeBERT")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    train_dataset, eval_dataset = load_and_process_data()
    
    # 2. 加载模型和tokenizer
    print("\n2. 加载CodeBERT模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # 二分类：流利/不流利
        problem_type="single_label_classification"
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
    
    # 4. 训练参数
    print("\n4. 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="steps",  # 按步骤验证
        eval_steps=EVAL_STEPS,        # 每100步验证一次
        save_steps=EVAL_STEPS,
        logging_steps=50,
        warmup_steps=WARMUP_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],  # 禁用wandb
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
    )
    
    # 7. 开始训练
    print("\n6. 开始训练...")
    print(f"训练配置:")
    print(f"  - 模型: {MODEL_NAME}")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - 训练轮数: {NUM_EPOCHS}")
    print(f"  - 验证频率: 每 {EVAL_STEPS} 步")
    print(f"  - 输出目录: {OUTPUT_DIR}")
    
    trainer.train()
    
    # 8. 保存最终模型
    print("\n7. 保存最终模型...")
    trainer.save_model()
    
    print("\n" + "=" * 60)
    print("🎉 流利度判别器训练完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()