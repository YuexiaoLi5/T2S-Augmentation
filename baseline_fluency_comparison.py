"""
Baseline流程2：比较训练前后的流利度判别器性能
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

# 禁用wandb
os.environ["WANDB_DISABLED"] = "true"

# 配置
MODEL_NAME = "microsoft/codebert-base"
TRAINED_MODEL_PATH = "output/fluency_discriminator_codebert/checkpoint-200"
OPTIMIZED_MODEL_PATH = "output/fluency_discriminator_optimized"
MAX_LENGTH = 512

def load_test_data():
    """加载测试数据"""
    
    # 加载验证数据
    with open("data/origin/test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"测试样本数: {len(test_data)}")
    
    # 处理测试数据
    samples = []
    
    for item in test_data:
        code = item['code']
        
        # 正样本（原始代码）
        samples.append({
            'text': code,
            'label': 1  # 流利
        })
        
        # 负样本（添加简单噪声）
        # 1. 删除随机行
        lines = code.split('\n')
        if len(lines) > 3:
            noisy_code = '\n'.join(lines[:-1])  # 删除最后一行
            samples.append({
                'text': noisy_code,
                'label': 0  # 不流利
            })
        
        # 2. 删除分号和大括号
        noisy_code2 = code.replace(';', '').replace('{', '').replace('}', '')
        if noisy_code2 != code:
            samples.append({
                'text': noisy_code2,
                'label': 0
            })
    
    print(f"测试集总样本数: {len(samples)}")
    
    # 检查类别平衡
    pos_count = sum(1 for sample in samples if sample['label'] == 1)
    neg_count = len(samples) - pos_count
    
    print(f"测试集 - 正样本: {pos_count}, 负样本: {neg_count}")
    
    return samples

def preprocess_data(samples, tokenizer):
    """预处理数据"""
    
    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH
        )
    
    from datasets import Dataset
    dataset = Dataset.from_list(samples)
    dataset = dataset.map(preprocess_function, batched=True)
    
    return dataset

def evaluate_model(model, tokenizer, test_dataset):
    """评估模型性能"""
    
    print("\n开始评估...")
    
    model.eval()
    
    true_labels = []
    predicted_labels = []
    predicted_probs = []
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            if i % 20 == 0:
                print(f"评估进度: {i}/{len(test_dataset)}")
            
            # 获取样本
            sample = test_dataset[i]
            
            # 准备输入
            inputs = {
                'input_ids': torch.tensor([sample['input_ids']]),
                'attention_mask': torch.tensor([sample['attention_mask']])
            }
            
            # 预测
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 获取预测结果
            probabilities = torch.softmax(logits, dim=-1)
            predicted = torch.argmax(logits, dim=-1).item()
            
            true_labels.append(sample['label'])
            predicted_labels.append(predicted)
            predicted_probs.append(probabilities[0].tolist())
    
    # 计算指标
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='binary')
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'predicted_probs': predicted_probs
    }

def compare_models():
    """比较不同模型的性能"""
    
    print("=" * 80)
    print("Baseline流程2：流利度判别器性能比较")
    print("=" * 80)
    
    # 1. 加载测试数据
    print("\n1. 加载测试数据...")
    test_samples = load_test_data()
    
    if len(test_samples) == 0:
        print("错误: 没有有效的测试数据")
        return
    
    # 限制样本数量以加快测试
    if len(test_samples) > 100:
        test_samples = test_samples[:100]
        print(f"限制测试样本数为: {len(test_samples)}")
    
    # 2. 加载tokenizer
    print("\n2. 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 预处理数据
    print("\n3. 预处理数据...")
    test_dataset = preprocess_data(test_samples, tokenizer)
    
    # 4. 评估未训练模型（baseline）
    print("\n4. 评估未训练模型（Baseline）...")
    baseline_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification"
    )
    baseline_results = evaluate_model(baseline_model, tokenizer, test_dataset)
    
    # 5. 评估已训练模型
    print("\n5. 评估已训练模型...")
    try:
        trained_model = AutoModelForSequenceClassification.from_pretrained(
            TRAINED_MODEL_PATH,
            num_labels=2,
            problem_type="single_label_classification"
        )
        trained_results = evaluate_model(trained_model, tokenizer, test_dataset)
        trained_available = True
    except Exception as e:
        print(f"❌ 已训练模型加载失败: {e}")
        trained_results = baseline_results.copy()
        trained_available = False
    
    # 6. 评估优化模型
    print("\n6. 评估优化模型...")
    try:
        optimized_model = AutoModelForSequenceClassification.from_pretrained(
            OPTIMIZED_MODEL_PATH,
            num_labels=2,
            problem_type="single_label_classification"
        )
        optimized_results = evaluate_model(optimized_model, tokenizer, test_dataset)
        optimized_available = True
    except Exception as e:
        print(f"❌ 优化模型加载失败: {e}")
        optimized_results = baseline_results.copy()
        optimized_available = False
    
    # 7. 显示比较结果
    print("\n" + "=" * 80)
    print("Baseline 2 性能比较结果")
    print("=" * 80)
    
    print("\n模型性能对比:")
    print("-" * 60)
    print(f"{'模型':<20} | {'准确率':<8} | {'F1分数':<8} | {'精确率':<8} | {'召回率':<8}")
    print("-" * 60)
    
    print(f"{'未训练模型':<20} | {baseline_results['accuracy']:.4f}  | {baseline_results['f1']:.4f}  | {baseline_results['precision']:.4f}  | {baseline_results['recall']:.4f}")
    
    if trained_available:
        print(f"{'已训练模型':<20} | {trained_results['accuracy']:.4f}  | {trained_results['f1']:.4f}  | {trained_results['precision']:.4f}  | {trained_results['recall']:.4f}")
    
    if optimized_available:
        print(f"{'优化模型':<20} | {optimized_results['accuracy']:.4f}  | {optimized_results['f1']:.4f}  | {optimized_results['precision']:.4f}  | {optimized_results['recall']:.4f}")
    
    print("-" * 60)
    
    # 8. 计算性能提升
    print("\n性能提升分析:")
    if trained_available:
        accuracy_improvement = trained_results['accuracy'] - baseline_results['accuracy']
        f1_improvement = trained_results['f1'] - baseline_results['f1']
        print(f"已训练模型 vs 未训练模型:")
        print(f"  准确率提升: {accuracy_improvement:+.4f}")
        print(f"  F1分数提升: {f1_improvement:+.4f}")
    
    if optimized_available:
        accuracy_improvement = optimized_results['accuracy'] - baseline_results['accuracy']
        f1_improvement = optimized_results['f1'] - baseline_results['f1']
        print(f"优化模型 vs 未训练模型:")
        print(f"  准确率提升: {accuracy_improvement:+.4f}")
        print(f"  F1分数提升: {f1_improvement:+.4f}")
    
    # 9. 保存结果
    results = {
        'baseline_type': 'fluency_discriminator_comparison',
        'test_samples': len(test_samples),
        'models': {
            'baseline': {
                'available': True,
                'metrics': baseline_results
            },
            'trained': {
                'available': trained_available,
                'metrics': trained_results if trained_available else None
            },
            'optimized': {
                'available': optimized_available,
                'metrics': optimized_results if optimized_available else None
            }
        }
    }
    
    with open("output/baseline2_fluency_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: output/baseline2_fluency_comparison_results.json")
    
    # 10. 显示一些预测示例
    print("\n预测示例（前3个样本）:")
    for i in range(min(3, len(test_samples))):
        print(f"\n样本 {i+1}:")
        print(f"  代码片段: {test_samples[i]['text'][:80]}...")
        print(f"  真实标签: {'流利' if test_samples[i]['label'] == 1 else '不流利'}")
        
        baseline_pred = baseline_results['predicted_labels'][i]
        baseline_prob = baseline_results['predicted_probs'][i][baseline_pred]
        print(f"  未训练模型预测: {'流利' if baseline_pred == 1 else '不流利'} (置信度: {baseline_prob:.4f})")
        
        if trained_available:
            trained_pred = trained_results['predicted_labels'][i]
            trained_prob = trained_results['predicted_probs'][i][trained_pred]
            print(f"  已训练模型预测: {'流利' if trained_pred == 1 else '不流利'} (置信度: {trained_prob:.4f})")
        
        if optimized_available:
            optimized_pred = optimized_results['predicted_labels'][i]
            optimized_prob = optimized_results['predicted_probs'][i][optimized_pred]
            print(f"  优化模型预测: {'流利' if optimized_pred == 1 else '不流利'} (置信度: {optimized_prob:.4f})")
    
    print("\n" + "=" * 80)
    print("🎉 Baseline 2 流程完成！")
    print("=" * 80)

if __name__ == "__main__":
    compare_models()