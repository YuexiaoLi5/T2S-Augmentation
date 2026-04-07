"""
PPO 继续训练脚本 - 使用新的 alignment_discriminator_codebert_v2
- 从 checkpoint-200 继续训练
- 早停机制
"""
import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn as nn

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_OFFLINE"] = "0"

ACTOR_MODEL_PATH = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
FLUENCY_MODEL_PATH = "output/fluency_discriminator_codebert"
ALIGNMENT_MODEL_PATH = "output/alignment_discriminator_codebert_v2"
PPO_BASE_PATH = "output/ppo_trl_3epoch/checkpoint-200"
OUTPUT_DIR = "output/ppo_trl_v2_continued"
MAX_STEPS = 100
PATIENCE = 10
BATCH_SIZE = 2
ACTOR_LR = 5e-6

LABEL_NAMES = ['logical_error', 'memory_error', 'omission', 'invalid_condition', 'infinite_loop_error', 'unknown']


def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_dataset(data):
    processed = []
    for item in data:
        text = item.get('text', '')
        code = item.get('code', '')
        if "AST:" in text:
            ast_start = text.find("AST:")
            ast_text = text[ast_start:]
        else:
            ast_text = text[:2000]
        if len(ast_text) > 1500:
            ast_text = ast_text[:1500]
        prompt = f"{ast_text}\n\nOriginal Code:\n{code[:1000]}"
        if prompt and code:
            processed.append({'prompt': prompt, 'response': code})
    return Dataset.from_list(processed)


class RewardEvaluator:
    def __init__(self, fluency_path, alignment_path, device='cuda'):
        self.device = device
        print(f"加载流利度判别器: {fluency_path}")
        try:
            self.fluency_model = AutoModelForSequenceClassification.from_pretrained(
                fluency_path, num_labels=2, torch_dtype=torch.float16, device_map=device
            )
            self.fluency_tokenizer = AutoTokenizer.from_pretrained(fluency_path)
        except Exception as e:
            print(f"流利度判别器加载失败: {e}")
            self.fluency_model = None
            self.fluency_tokenizer = None

        print(f"加载一致性判别器: {alignment_path}")
        self.alignment_model = AutoModelForSequenceClassification.from_pretrained(
            alignment_path,
            num_labels=len(LABEL_NAMES),
            problem_type="multi_label_classification",
            torch_dtype=torch.float16,
            device_map=device
        )
        self.alignment_tokenizer = AutoTokenizer.from_pretrained(alignment_path)

    def compute_reward(self, texts):
        rewards = []
        for text in texts:
            if self.fluency_model is not None:
                inputs = self.fluency_tokenizer(text, return_tensors="pt", max_length=512,
                    truncation=True, padding=True).to(self.device)
                with torch.no_grad():
                    logits = self.fluency_model(**inputs).logits
                    fluency_score = torch.softmax(logits, dim=-1)[0, 1].item()
            else:
                fluency_score = 0.5

            align_inputs = self.alignment_tokenizer(text, return_tensors="pt", max_length=512,
                truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                align_logits = self.alignment_model(**align_inputs).logits
                align_probs = torch.sigmoid(align_logits).cpu().numpy()[0]
                alignment_score = align_probs.mean()

            reward = fluency_score * 0.4 + alignment_score * 0.6
            rewards.append(reward)
        return rewards


class EarlyStoppingCallback:
    def __init__(self, patience=10, min_improvement=0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_reward = None
        self.no_improvement_count = 0

    def check(self, avg_reward, step):
        if self.best_reward is None:
            self.best_reward = avg_reward
            print(f"[早停] 初始奖励: {avg_reward:.4f}")
            return False

        improvement = avg_reward - self.best_reward
        if improvement > self.min_improvement:
            self.best_reward = avg_reward
            self.no_improvement_count = 0
            print(f"[早停] 奖励提升: {avg_reward:.4f} (+{improvement:.4f})")
            return False
        else:
            self.no_improvement_count += 1
            print(f"[早停] 奖励未提升 ({self.no_improvement_count}/{self.patience}): {avg_reward:.4f}")
            if self.no_improvement_count >= self.patience:
                print(f"[早停] 触发早停!")
                return True
        return False


def main():
    print("=" * 60)
    print("PPO 继续训练 - 使用新的 alignment_discriminator_codebert_v2")
    print("=" * 60)

    print("\n加载数据...")
    train_data = load_data("data/origin/train.json")
    val_data = load_data("data/origin/test.json")
    dataset = prepare_dataset(train_data)
    val_dataset = prepare_dataset(val_data[:50])
    print(f"训练样本数: {len(dataset)}, 验证样本数: {len(val_dataset)}")

    print("\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n加载已有模型: {PPO_BASE_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        ACTOR_MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, PPO_BASE_PATH)
    model.eval()

    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,}")

    print("\n加载奖励模型...")
    reward_evaluator = RewardEvaluator(FLUENCY_MODEL_PATH, ALIGNMENT_MODEL_PATH)

    early_stopping = EarlyStoppingCallback(patience=PATIENCE, min_improvement=0.005)

    def reward_func(prompts, completions, **kwargs):
        texts = [p + c for p, c in zip(prompts, completions)]
        return reward_evaluator.compute_reward(texts)

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=ACTOR_LR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        logging_steps=5,
        save_steps=20,
        num_generations=2,
        bf16=True,
        report_to=[],
    )

    print("\n初始化 GRPOTrainer...")
    grpo_trainer = GRPOTrainer(
        args=grpo_config,
        model=model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        processing_class=tokenizer,
    )

    print("\n开始训练...")
    training_history = []

    for step in range(MAX_STEPS):
        grpo_trainer.train()

        if (step + 1) % 5 == 0:
            print(f"\n--- 验证步骤 {step + 1} ---")
            val_rewards = []
            for i in range(0, len(val_dataset), BATCH_SIZE):
                batch = val_dataset[i:i+BATCH_SIZE]
                prompts = [item['prompt'] for item in batch]
                inputs = tokenizer(prompts, return_tensors="pt", max_length=512,
                                   truncation=True, padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=100,
                        temperature=0.7, do_sample=True, pad_token_id=tokenizer.pad_token_id)
                generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                completions = [g[len(p):] for g, p in zip(generations, prompts)]
                rewards = reward_func(prompts, completions)
                val_rewards.extend(rewards)

            avg_val_reward = np.mean(val_rewards)
            print(f"验证集平均奖励: {avg_val_reward:.4f}")
            training_history.append({'step': step + 1, 'val_reward': avg_val_reward})

            if early_stopping.check(avg_val_reward, step + 1):
                print(f"\n早停触发，保存模型...")
                grpo_trainer.save_model(f"{OUTPUT_DIR}/checkpoint-early-stop-{step+1}")
                break

    print("\n保存最终模型...")
    grpo_trainer.save_model(f"{OUTPUT_DIR}/final")
    with open(f"{OUTPUT_DIR}/training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"模型保存在: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
