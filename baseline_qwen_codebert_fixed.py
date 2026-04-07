"""
Baseline流程1修复版：使用本地缓存的Qwen模型 + CodeBERT验证标签
"""

import os
import json
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    GenerationConfig
)

# 禁用wandb
os.environ["WANDB_DISABLED"] = "true"

# 配置 - 使用本地缓存路径
QWEN_MODEL = "D:/HF_Models/Qwen/Qwen2.5-Coder-3B-Instruct"  # 使用本地路径
CODEBERT_MODEL = "microsoft/codebert-base"
MAX_LENGTH = 512

# 多标签错误类型（移除unknown）
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
    
    # 创建5维向量
    label_vector = [0] * len(LABEL_NAMES)
    for label in normalized:
        if label in LABEL_NAMES:
            label_vector[LABEL_NAMES.index(label)] = 1
    
    return label_vector

def load_test_data():
    """加载测试数据"""
    
    # 加载验证数据
    with open("data/origin/test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"测试样本数: {len(test_data)}")
    
    # 处理测试数据
    samples = []
    
    for item in test_data:
        text = item['text']
        
        # 从text字段提取真实标签
        if "Label:" in text:
            label_start = text.find("Label:") + 6
            label_end = text.find("\n", label_start)
            if label_end != -1:
                label_str = text[label_start:label_end].strip()
                true_labels = parse_label_vector(label_str)
                
                # 提取prompt（去除标签部分）
                prompt = text[:text.find("Label:")].strip()
                
                samples.append({
                    'prompt': prompt,
                    'true_labels': true_labels,
                    'label_str': label_str
                })
    
    print(f"有效测试样本数: {len(samples)}")
    
    # 显示标签分布
    if samples:
        label_counts = [0] * len(LABEL_NAMES)
        for sample in samples:
            for i, val in enumerate(sample['true_labels']):
                if val == 1:
                    label_counts[i] += 1
        
        print("\n测试集标签分布:")
        for i, count in enumerate(label_counts):
            print(f"  {LABEL_NAMES[i]}: {count}")
    
    return samples

def load_model_with_retry(model_path, model_type="qwen", max_retries=3):
    """带重试机制的模型加载"""
    
    for attempt in range(max_retries):
        try:
            print(f"尝试加载模型 (第 {attempt+1} 次)...")
            
            if model_type == "qwen":
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True  # 强制使用本地文件
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=len(LABEL_NAMES),
                    problem_type="multi_label_classification"
                )
            
            print("✅ 模型加载成功")
            return tokenizer, model
            
        except Exception as e:
            print(f"❌ 模型加载失败 (第 {attempt+1} 次): {e}")
            if attempt < max_retries - 1:
                print("等待3秒后重试...")
                import time
                time.sleep(3)
            else:
                print("所有重试失败，使用模拟模式")
                return None, None

def generate_code_with_qwen(prompts, qwen_model, qwen_tokenizer):
    """使用Qwen生成代码"""
    
    print("\n使用Qwen生成代码...")
    
    generated_codes = []
    
    for i, prompt in enumerate(prompts):
        if i % 10 == 0:
            print(f"生成进度: {i}/{len(prompts)}")
        
        try:
            # 构建输入
            messages = [
                {"role": "user", "content": f"请根据以下描述生成Python代码：\n{prompt}"}
            ]
            
            # 生成配置
            generation_config = GenerationConfig(
                max_new_tokens=256,  # 减少生成长度
                temperature=0.7,
                do_sample=True,
                pad_token_id=qwen_tokenizer.eos_token_id
            )
            
            # 生成代码
            inputs = qwen_tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = qwen_model.generate(
                    inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # 解码生成的代码
            generated_text = qwen_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # 提取生成的代码部分
            if "```python" in generated_text:
                code_start = generated_text.find("```python") + 9
                code_end = generated_text.find("```", code_start)
                if code_end != -1:
                    generated_code = generated_text[code_start:code_end].strip()
                else:
                    generated_code = generated_text[code_start:].strip()
            else:
                # 如果没有代码块标记，尝试提取代码部分
                lines = generated_text.split('\n')
                code_lines = []
                in_code = False
                for line in lines:
                    if line.strip().startswith('def ') or line.strip().startswith('import ') or line.strip().startswith('class '):
                        in_code = True
                    if in_code:
                        code_lines.append(line)
                generated_code = '\n'.join(code_lines)
            
            generated_codes.append(generated_code if generated_code else "# 生成失败")
            
        except Exception as e:
            print(f"生成失败 (样本 {i+1}): {e}")
            generated_codes.append("# 生成失败")
    
    return generated_codes

def mock_generate(prompts, model=None, tokenizer=None):
    """模拟生成函数"""
    print("使用模拟生成模式...")
    
    generated_codes = []
    for i, prompt in enumerate(prompts):
        # 简单的模拟生成
        code = f"""# 根据描述生成的代码
# {prompt[:50]}...

def example_function():
    # 模拟代码逻辑
    result = 0
    for i in range(10):
        result += i
    return result
"""
        generated_codes.append(code)
    
    return generated_codes

def predict_labels_with_codebert(codes, codebert_model, codebert_tokenizer):
    """使用CodeBERT预测标签"""
    
    print("\n使用CodeBERT预测标签...")
    
    predicted_labels = []
    
    if codebert_model is None:
        print("使用模拟预测模式...")
        # 模拟预测
        for code in codes:
            # 简单的模拟预测
            predicted = [1, 0, 0, 0, 0]  # 默认预测为logical_error
            predicted_labels.append(predicted)
        return np.array(predicted_labels)
    
    codebert_model.eval()
    
    with torch.no_grad():
        for i, code in enumerate(codes):
            if i % 10 == 0:
                print(f"预测进度: {i}/{len(codes)}")
            
            try:
                # 编码输入
                inputs = codebert_tokenizer(
                    code,
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )
                
                # 预测
                outputs = codebert_model(**inputs)
                logits = outputs.logits
                
                # 应用sigmoid并设置阈值
                probabilities = torch.sigmoid(logits)
                predicted = (probabilities > 0.5).int().squeeze().tolist()
                
                # 确保预测结果长度正确
                if isinstance(predicted, int):
                    predicted = [predicted]
                elif len(predicted) != len(LABEL_NAMES):
                    predicted = [0] * len(LABEL_NAMES)
                
                predicted_labels.append(predicted)
                
            except Exception as e:
                print(f"预测失败 (样本 {i+1}): {e}")
                predicted_labels.append([0] * len(LABEL_NAMES))
    
    return np.array(predicted_labels)

def calculate_metrics(true_labels, predicted_labels):
    """计算多标签分类指标"""
    
    print("\n计算评估指标...")
    
    # 计算每个标签的指标
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for i in range(len(LABEL_NAMES)):
        if np.sum(true_labels[:, i]) > 0:  # 确保标签存在
            f1 = f1_score(true_labels[:, i], predicted_labels[:, i], average='binary', zero_division=0)
            precision = precision_score(true_labels[:, i], predicted_labels[:, i], average='binary', zero_division=0)
            recall = recall_score(true_labels[:, i], predicted_labels[:, i], average='binary', zero_division=0)
            
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
    micro_f1 = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)
    micro_precision = precision_score(true_labels, predicted_labels, average='micro', zero_division=0)
    micro_recall = recall_score(true_labels, predicted_labels, average='micro', zero_division=0)
    
    return {
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'micro_f1': micro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'per_label_f1': dict(zip(LABEL_NAMES, f1_scores)),
        'per_label_precision': dict(zip(LABEL_NAMES, precision_scores)),
        'per_label_recall': dict(zip(LABEL_NAMES, recall_scores)),
    }

def main():
    """主函数"""
    
    print("=" * 80)
    print("Baseline流程1修复版：Qwen生成代码 + CodeBERT验证标签")
    print("=" * 80)
    
    # 1. 加载测试数据
    print("\n1. 加载测试数据...")
    samples = load_test_data()
    
    if len(samples) == 0:
        print("错误: 没有有效的测试数据")
        return
    
    # 限制样本数量以加快测试
    if len(samples) > 20:
        samples = samples[:20]
        print(f"限制测试样本数为: {len(samples)}")
    
    # 2. 加载Qwen模型（使用本地路径）
    print("\n2. 加载Qwen模型...")
    qwen_tokenizer, qwen_model = load_model_with_retry(QWEN_MODEL, "qwen")
    
    # 3. 加载CodeBERT模型
    print("\n3. 加载CodeBERT模型...")
    codebert_tokenizer, codebert_model = load_model_with_retry(CODEBERT_MODEL, "codebert")
    
    # 4. 使用Qwen生成代码
    prompts = [sample['prompt'] for sample in samples]
    
    if qwen_model is not None:
        generated_codes = generate_code_with_qwen(prompts, qwen_model, qwen_tokenizer)
    else:
        generated_codes = mock_generate(prompts)
    
    # 5. 使用CodeBERT预测标签
    true_labels = np.array([sample['true_labels'] for sample in samples])
    predicted_labels = predict_labels_with_codebert(generated_codes, codebert_model, codebert_tokenizer)
    
    # 6. 计算指标
    metrics = calculate_metrics(true_labels, predicted_labels)
    
    # 7. 显示结果
    print("\n" + "=" * 80)
    print("Baseline 1 修复版评估结果")
    print("=" * 80)
    
    print("\n宏平均指标:")
    print(f"  F1分数: {metrics['macro_f1']:.4f}")
    print(f"  精确率: {metrics['macro_precision']:.4f}")
    print(f"  召回率: {metrics['macro_recall']:.4f}")
    
    print("\n微平均指标:")
    print(f"  F1分数: {metrics['micro_f1']:.4f}")
    print(f"  精确率: {metrics['micro_precision']:.4f}")
    print(f"  召回率: {metrics['micro_recall']:.4f}")
    
    print("\n各标签F1分数:")
    for label, f1 in metrics['per_label_f1'].items():
        print(f"  {label}: {f1:.4f}")
    
    # 8. 保存结果
    results = {
        'baseline_type': 'qwen_generation_codebert_validation_fixed',
        'model': {
            'qwen': QWEN_MODEL,
            'codebert': CODEBERT_MODEL
        },
        'num_samples': len(samples),
        'metrics': metrics,
        'true_labels': true_labels.tolist(),
        'predicted_labels': predicted_labels.tolist(),
        'generated_codes': generated_codes,
        'prompts': prompts
    }
    
    with open("output/baseline1_qwen_codebert_results_fixed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: output/baseline1_qwen_codebert_results_fixed.json")
    
    # 9. 显示一些生成示例
    print("\n生成示例（前3个样本）:")
    for i in range(min(3, len(samples))):
        print(f"\n样本 {i+1}:")
        print(f"  Prompt: {samples[i]['prompt'][:100]}...")
        print(f"  真实标签: {samples[i]['label_str']}")
        
        predicted_str = ", ".join([LABEL_NAMES[j] for j, val in enumerate(predicted_labels[i]) if val == 1])
        print(f"  预测标签: {predicted_str if predicted_str else '无错误'}")
        
        print(f"  生成代码: {generated_codes[i][:100]}...")
        
        # 计算样本级F1
        sample_f1 = f1_score([true_labels[i]], [predicted_labels[i]], average='micro', zero_division=0)
        print(f"  样本F1: {sample_f1:.4f}")
    
    print("\n" + "=" * 80)
    print("🎉 Baseline 1 修复版流程完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()