"""
使用 TRL (Transformer Reinforcement Learning) 进行 PPO 训练的脚本
"""
import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig
from trl import GRPOTrainer, GRPOConfig
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
from sklearn.metrics import f1_score
import numpy as np

# ============== 验证回调类 ==============
class ValidationCallback:
    def __init__(self, validation_data, fluency_model, alignment_model, tokenizer):
        self.validation_data = validation_data
        self.fluency_model = fluency_model
        self.alignment_model = alignment_model
        self.tokenizer = tokenizer
        self.validation_results = []
    
    def light_validation(self, model, step):
        """轻量级实时验证（10个样本）"""
        print(f"\n[验证] 步骤 {step}: 开始轻量级验证")
        
        # 随机选择10个验证样本
        sample_size = min(10, len(self.validation_data))
        samples = self.validation_data[:sample_size]
        
        # 生成代码修复
        generated_texts = []
        for sample in samples:
            prompt = sample['prompt']
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        # 计算判别器评分
        fluency_scores = []
        alignment_scores = []
        
        for text in generated_texts:
            # 流利度评分
            fluency_inputs = self.fluency_model.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                fluency_output = self.fluency_model.model(**fluency_inputs)
                fluency_score = torch.softmax(fluency_output.logits, dim=-1)[0, 1].item()
            fluency_scores.append(fluency_score)
            
            # 一致性评分（简化版）
            alignment_score = 0.5  # 默认值
            alignment_scores.append(alignment_score)
        
        avg_fluency = np.mean(fluency_scores)
        avg_alignment = np.mean(alignment_scores)
        
        print(f"[验证] 平均流利度: {avg_fluency:.4f}, 平均一致性: {avg_alignment:.4f}")
        
        return {
            'step': step,
            'avg_fluency': avg_fluency,
            'avg_alignment': avg_alignment,
            'validation_type': 'light'
        }
    
    def full_validation(self, model, step):
        """完整离线验证（全部验证集）"""
        print(f"\n[验证] 步骤 {step}: 开始完整验证")
        
        # 这里可以添加更复杂的验证逻辑
        # 包括F1值计算、代码质量评估等
        
        return {
            'step': step,
            'validation_type': 'full',
            'message': '完整验证待实现'
        }
    
    def on_step_end(self, step, model):
        """步骤结束回调"""
        if step % 10 == 0:
            result = self.light_validation(model, step)
            self.validation_results.append(result)
        
        if step % 50 == 0:
            result = self.full_validation(model, step)
            self.validation_results.append(result)

# ============== 配置 ==============
# 禁用 wandb
os.environ["WANDB_DISABLED"] = "true"

# 启用 HuggingFace 在线下载
os.environ["HF_HUB_OFFLINE"] = "0"

# 模型路径 - 使用HuggingFace在线模型
ACTOR_MODEL_PATH = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # 1.5B模型
FLUENCY_MODEL_PATH = "output/fluency_discriminator_hf/final"  # 流利度判别器
ALIGNMENT_MODEL_PATH = "output/consistency_discriminator/model/run1"  # 一致性判别器

# 训练参数
OUTPUT_DIR = "output/ppo_trl_3epoch"
MAX_STEPS = 200  # 3个epoch
BATCH_SIZE = 1   # 批次大小
ACTOR_LR = 1e-5
MAX_GEN_LENGTH = 256  # 保持较大的生成长度
MAX_PROMPT_LENGTH = 4096  # 增大以适应 Label + AST + Original Code

# ============== 数据加载 ==============
def load_data(data_path):
    """加载训练数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_dataset(data):
    """准备 TRL 需要的格式"""
    # TRL 需要 prompt 和 response 字段
    processed = []
    for item in data:
        text = item.get('text', '')  # S-表达式 (Label + AST)
        code = item.get('code', '')    # 原始buggy code
        
        # 提取 Label 和 AST
        label = ""
        ast_text = text
        if "Label:" in text:
            start = text.find("Label:") + len("Label:")
            end = text.find("AST:")
            if end != -1:
                label = text[start:end].strip()
                ast_text = text
        
        # 构建完整 prompt: Label + AST + Original Code
        prompt = f"{ast_text}\n\nOriginal Code:\n{code}"
        
        # 限制prompt长度，避免tokenization问题
        if len(prompt) > 8000:
            prompt = prompt[:8000]  # 截断过长的prompt
        
        # 确保prompt和code不为空
        if not prompt or not code:
            continue  # 跳过空样本
        
        processed.append({
            'prompt': prompt,
            'response': code,
        })
    return Dataset.from_list(processed)

# ============== Reward 函数 ==============
class RewardEvaluator:
    """奖励评估器 - 整合流利度和一致性判别器"""
    
    def __init__(self, fluency_path, alignment_path, device='cuda'):
        self.device = device
        
        # 加载流利度判别器 (二分类: 流利/不流利)
        print(f"加载流利度判别器: {fluency_path}")
        self.fluency_model = AutoModelForSequenceClassification.from_pretrained(
            fluency_path,
            num_labels=2,
            torch_dtype=torch.float16,
            device_map=device
        )
        self.fluency_tokenizer = AutoTokenizer.from_pretrained(fluency_path)
        
        # 加载一致性判别器 (多标签分类)
        print(f"加载一致性判别器: {alignment_path}")
        self.alignment_model = AutoModelForSequenceClassification.from_pretrained(
            alignment_path,
            num_labels=6,
            problem_type="multi_label_classification",
            torch_dtype=torch.float16,
            device_map=device
        )
        self.alignment_tokenizer = AutoTokenizer.from_pretrained(alignment_path)
        
        # 错误标签列表 (与训练时一致)
        self.label_list = [
            'variable-misuse', 'type-error', 'logical-error', 
            'syntax-error', 'omission', 'wrong-order'
        ]
        
    def compute_reward(self, texts, labels=None):
        """
        计算奖励
        
        Args:
            texts: 生成的代码列表
            labels: 目标错误标签列表 (可选)
            
        Returns:
            rewards: 奖励分数列表
        """
        rewards = []
        
        # 1. 流利度奖励 (0-1)
        fluency_inputs = self.fluency_tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            fluency_logits = self.fluency_model(**fluency_inputs).logits
            # 取第二类的概率 (流利的概率)
            fluency_probs = torch.softmax(fluency_logits, dim=-1)
            fluency_scores = fluency_probs[:, 1].cpu().numpy()
        
        # 2. 一致性奖励 (如果提供了标签)
        if labels is not None:
            alignment_scores = []
            for text, label in zip(texts, labels):
                # 解析标签
                label = label.lower().replace(' ', '-') if label else ''
                
                alignment_inputs = self.alignment_tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    alignment_logits = self.alignment_model(**alignment_inputs).logits
                    alignment_probs = torch.sigmoid(alignment_logits).cpu().numpy()[0]
                
                # 计算与目标标签的匹配度
                target_labels = [l.strip().lower().replace('-', ' ').replace('_', '-') for l in label.split(',')]
                match_score = 0
                for i, l in enumerate(self.label_list):
                    l_normalized = l.replace('-', ' ').replace('_', '-')
                    if any(tl in l_normalized or l_normalized in tl for tl in target_labels):
                        match_score += alignment_probs[i]
                alignment_scores.append(match_score / max(len(target_labels), 1))
            
            alignment_scores = torch.tensor(alignment_scores, dtype=torch.float32).cpu().numpy()
        else:
            alignment_scores = torch.ones(len(texts)).numpy()
        
        # 3. 综合奖励
        for flu, align in zip(fluency_scores, alignment_scores):
            # 流利度权重 0.4，一致性权重 0.6
            reward = flu * 0.4 + align * 0.6
            rewards.append(reward)
            
        return rewards


def create_value_head(model):
    """为模型添加 value head (用于 Critic)"""
    # 获取隐藏层大小
    hidden_size = model.config.hidden_size
    
    # 添加 value head
    class ValueHeadModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.value_head = torch.nn.Linear(hidden_size, 1)
            # 添加必要的属性，用于 TRL
            self.base_model_prefix = 'model'  # Qwen 模型使用 'model' 作为前缀
            self.model = base_model  # 添加 model 属性，与 base_model_prefix 对应
            # 复制必要的属性
            self.config = base_model.config
            self.dtype = base_model.dtype
            self.device = base_model.device
        
        def forward(self, input_ids, attention_mask=None, **kwargs):
            outputs = self.base_model(input_ids, attention_mask=attention_mask, **kwargs)
            # 使用最后一层隐藏状态
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
            # 取最后一个 token 的表示
            value = self.value_head(hidden_states[:, -1, :])
            return value
    
    return ValueHeadModel(model)


class RewardModelForPPO(torch.nn.Module):
    """包装 RewardEvaluator 用于 TRL PPOTrainer"""
    
    def __init__(self, reward_evaluator):
        super().__init__()
        self.reward_evaluator = reward_evaluator
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        返回奖励分数
        TRL 会调用这个函数来获取奖励
        """
        # 解码生成的内容
        texts = []
        for ids in input_ids:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)
        
        # 计算奖励
        rewards = self.reward_evaluator.compute_reward(texts)
        return torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)

def train_grpo():
    """使用 TRL 的 GRPOTrainer 进行训练 (不需要 Critic)"""
    
    print("=" * 50)
    print("开始 GRPO 训练 (使用 TRL 框架)")
    print("=" * 50)
    
    # 加载数据
    print("\n加载数据...")
    train_data = load_data("data/origin/train.json")
    dataset = prepare_dataset(train_data)
    print(f"训练样本数: {len(dataset)}")
    
    # 加载验证数据
    print("\n加载验证数据...")
    try:
        validation_data = load_data("data/origin/test.json")
        print(f"验证样本数: {len(validation_data)}")
    except:
        print("警告: 验证集不存在，使用训练集前10个样本作为验证")
        validation_data = train_data[:10]
    
    # 加载 Actor 模型
    print("\n加载 Actor 模型...")
    tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
    # 确保tokenizer配置正确
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    
    # 使用安全的LoRA配置（避免数值不稳定）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,  # 减小r值，避免数值不稳定
        lora_alpha=8,  # 减小lora_alpha
        lora_dropout=0.3,  # 增加dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 减少target modules
        bias="none",
    )
    
    # 使用4-bit量化加载模型
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        ACTOR_MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 确保模型在单精度下运行
    model = model.float()
    
    # 加载奖励模型
    print("\n加载奖励模型...")
    reward_evaluator = RewardEvaluator(FLUENCY_MODEL_PATH, ALIGNMENT_MODEL_PATH)
    
    # 初始化验证回调
    print("\n初始化验证回调...")
    validation_callback = ValidationCallback(
        validation_data=validation_data,
        fluency_model=reward_evaluator.fluency_model,
        alignment_model=reward_evaluator.alignment_model,
        tokenizer=tokenizer
    )
    
    # 定义 reward 函数
    def reward_func(prompts, completions, completion_ids=None, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            text = prompt + completion
            # 从 kwargs 中获取 label
            label = kwargs.get('label', [''])[0] if 'label' in kwargs else ''
            reward = reward_evaluator.compute_reward([text], [label])[0]
            rewards.append(reward)
        return rewards
    
    # GRPO 配置
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=ACTOR_LR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,  # 有效batch size = 2 * 2 = 4
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=50,
        # 生成数量4
        num_generations=4,
        # 使用bf16
        bf16=True,
        fp16=False,
        # 禁用 wandb
        report_to=[],
    )
    
    # 初始化 GRPOTrainer
    print("\n初始化 GRPOTrainer...")
    grpo_trainer = GRPOTrainer(
        args=grpo_config,
        model=model,
        train_dataset=dataset,
        reward_funcs=reward_func,  # 使用 reward_funcs 而不是 reward_function
        processing_class=tokenizer,  # 使用 processing_class 而不是 tokenizer
    )
    
    # 开始训练（GRPOConfig 的 max_steps=200 会由 train() 内部跑完，save_steps=50 会自动保存 checkpoint）
    print("\n开始训练...")
    grpo_trainer.train()

    # 保存最终模型
    print("\n保存最终模型...")
    grpo_trainer.save_model()
    
    # 输出验证结果
    print("\n验证结果汇总:")
    for result in validation_callback.validation_results:
        print(f"步骤 {result['step']}: {result}")
    
    print("\n" + "=" * 50)
    print("训练完成!")
    print("=" * 50)


if __name__ == "__main__":
    train_grpo()

