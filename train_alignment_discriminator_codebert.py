"""
一致性判别器训练脚本 - 使用CodeBERT，添加验证步骤
"""

import os
import json
import random
import re
import torch
import torch.nn as nn
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
from transformers.trainer_callback import TrainerCallback

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
OUTPUT_DIR = "output/alignment_discriminator_codebert"
DATA_DIR = "data/origin"
EVAL_STEPS = 50  # 每50步验证一次
PATIENCE = 3  # 早停耐心值: 如果连续3次验证F1没提升就停止

# 多标签错误类型
LABEL_NAMES = [
    'logical_error',
    'memory_error',
    'omission',
    'invalid_condition',
    'infinite_loop_error',
    'unknown',
]

# 类别权重 (根据数据分布计算)
CLASS_WEIGHTS = None


def compute_class_weights(train_samples):
    """根据训练数据分布计算类别权重"""
    num_classes = len(LABEL_NAMES)
    total_samples = len(train_samples)

    # 统计每个类别的正样本数
    label_counts = [0] * num_classes
    for sample in train_samples:
        for i, val in enumerate(sample['labels']):
            if val == 1:
                label_counts[i] += 1

    # 计算权重: 使用 inverse frequency
    # 避免除以0，加一个平滑项
    weights = []
    for count in label_counts:
        if count > 0:
            # 权重 = 总样本数 / (类别数 * 该类别样本数)
            weight = total_samples / (num_classes * count)
        else:
            weight = 1.0
        weights.append(weight)

    # 归一化权重
    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes

    print("\n类别权重:")
    for i, w in enumerate(weights):
        print(f"  {LABEL_NAMES[i]}: {w:.4f}")

        return torch.tensor(weights, dtype=torch.float32)


class WeightedBCEWithLogitsLoss(nn.Module):
    """加权二元交叉熵损失 for 多标签分类"""

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, logits, labels):
        # 使用BCEWithLogitsLoss
        loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.weights)
        loss = loss_fn(logits, labels)
        return loss.mean()


class EarlyStoppingCallback(TrainerCallback):
    """早停回调: 如果验证F1连续N次没有提升则停止训练"""

    def __init__(self, patience=3, metric_name="eval_macro_f1", greater_is_better=True):
        self.patience = patience
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = None
        self.no_improvement_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        current_metric = metrics.get(self.metric_name)
        if current_metric is None:
            return

        if self.best_metric is None:
            self.best_metric = current_metric
            return

        # 检查是否有提升
        if self.greater_is_better:
            has_improvement = current_metric > self.best_metric
        else:
            has_improvement = current_metric < self.best_metric

        if has_improvement:
            self.best_metric = current_metric
            self.no_improvement_count = 0
            print(f"\n✓ {self.metric_name} 提升至 {current_metric:.4f}")
        else:
            self.no_improvement_count += 1
            print(f"\n✗ {self.metric_name} 未提升 ({self.no_improvement_count}/{self.patience}): {current_metric:.4f} vs best {self.best_metric:.4f}")

            if self.no_improvement_count >= self.patience:
                print(f"\n早停触发! 连续{self.patience}次验证没有提升")
                control.should_training_stop = True

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
        else:
            normalized.append('unknown')
    
    # 创建6维向量
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

def compute_metrics(eval_pred):
    """计算多标签分类的评估指标"""
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    predictions, labels = eval_pred
    
    # 应用sigmoid并设置阈值
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    predictions = (predictions > 0.5).astype(int)
    
    # 计算每个标签的F1分数
    f1_scores = []
    for i in range(len(LABEL_NAMES)):
        if len(labels) > 0 and np.sum(labels[:, i]) > 0:  # 确保标签存在
            f1 = f1_score(labels[:, i], predictions[:, i], average='binary')
            f1_scores.append(f1)
        else:
            f1_scores.append(0.0)
    
    # 计算宏平均F1
    macro_f1 = np.mean(f1_scores)

    return {
        'macro_f1': macro_f1,
        'f1_logical_error': f1_scores[0],
        'f1_memory_error': f1_scores[1],
        'f1_omission': f1_scores[2],
        'f1_invalid_condition': f1_scores[3],
        'f1_infinite_loop_error': f1_scores[4],
        'f1_unknown': f1_scores[5],
    }


class WeightedLossTrainer(Trainer):
    """自定义Trainer，支持加权损失"""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # 移动权重到正确的设备
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
        else:
            weights = None

        # 使用加权BCE损失
        if weights is not None:
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            loss_fct = nn.BCEWithLogitsLoss()

        # 转换labels为float
        labels = labels.float()

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def main():
    """主训练函数"""

    print("=" * 60)
    print("一致性判别器训练 - CodeBERT (加权损失 + 早停)")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载数据...")
    train_dataset, eval_dataset = load_and_process_data()

    if len(train_dataset) == 0:
        print("错误: 没有有效的训练数据")
        return

    # 获取训练样本用于计算类别权重
    train_samples = train_dataset.to_list()

    # 2. 加载模型和tokenizer
    print("\n2. 加载CodeBERT模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),  # 多标签分类
        problem_type="multi_label_classification"
    )

    # 设置模型pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # 3. 计算类别权重
    print("\n3. 计算类别权重...")
    class_weights = compute_class_weights(train_samples)

    # 4. 数据预处理
    print("\n4. 数据预处理...")
    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH
        )

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    # 5. 训练参数
    print("\n5. 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="steps",  # 按步骤验证
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        logging_steps=50,
        warmup_steps=WARMUP_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to=[],  # 禁用wandb
    )

    # 6. 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 7. 创建Trainer (带加权损失)
    print("\n6. 创建Trainer...")

    # 使用自定义Trainer以支持加权损失
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # 8. 添加早停回调
    early_stopping_callback = EarlyStoppingCallback(
        patience=PATIENCE,
        metric_name="eval_macro_f1",
        greater_is_better=True
    )
    trainer.add_callback(early_stopping_callback)

    # 9. 开始训练
    print("\n7. 开始训练...")
    print(f"训练配置:")
    print(f"  - 模型: {MODEL_NAME}")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - 训练轮数: {NUM_EPOCHS}")
    print(f"  - 验证频率: 每 {EVAL_STEPS} 步")
    print(f"  - 早停耐心: {PATIENCE}")
    print(f"  - 输出目录: {OUTPUT_DIR}")
    print(f"  - 标签数量: {len(LABEL_NAMES)}")

    trainer.train()

    # 10. 保存最终模型
    print("\n8. 保存最终模型...")
    trainer.save_model()

    # 11. 最终评估
    print("\n9. 最终评估...")
    eval_results = trainer.evaluate()
    print("最终评估结果:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()