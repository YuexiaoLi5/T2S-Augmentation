"""
PPO Continue Training with new alignment discriminator v2
"""
import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import GRPOTrainer, GRPOConfig
from peft import PeftModel

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_OFFLINE"] = "0"

ACTOR_MODEL_PATH = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
FLUENCY_MODEL_PATH = "output/fluency_discriminator_codebert/checkpoint-200"
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
        prompt = ast_text + "\n\nOriginal Code:\n" + code[:1000]
        if prompt and code:
            processed.append({'prompt': prompt, 'response': code})
    return Dataset.from_list(processed)


class RewardEvaluator:
    def __init__(self, fluency_path, alignment_path, device='cuda'):
        self.device = device
        print("Loading fluency discriminator:", fluency_path)
        try:
            self.fluency_model = AutoModelForSequenceClassification.from_pretrained(
                fluency_path, num_labels=2, torch_dtype=torch.float16, device_map=device
            )
            self.fluency_tokenizer = AutoTokenizer.from_pretrained(fluency_path)
        except Exception as e:
            print("Fluency model load failed:", e)
            self.fluency_model = None
            self.fluency_tokenizer = None

        print("Loading alignment discriminator:", alignment_path)
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
    def __init__(self, patience=10, min_improvement=0.005, val_dataset=None, tokenizer=None, model=None, reward_evaluator=None, batch_size=2):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_reward = None
        self.no_improve_count = 0
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.reward_evaluator = reward_evaluator
        self.batch_size = batch_size

    def validate(self):
        if self.val_dataset is None:
            return 0.5

        val_rewards = []
        for i in range(0, len(self.val_dataset), self.batch_size):
            batch = self.val_dataset[i:i+self.batch_size]
            prompts = [item['prompt'] for item in batch]
            inputs = self.tokenizer(prompts, return_tensors="pt", max_length=512,
                                   truncation=True, padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=100,
                    temperature=0.7, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
            generations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            completions = [g[len(p):] for g, p in zip(generations, prompts)]
            texts = [p + c for p, c in zip(prompts, completions)]
            rewards = self.reward_evaluator.compute_reward(texts)
            val_rewards.extend(rewards)

        return np.mean(val_rewards) if val_rewards else 0.5

    def on_step_end(self, step, state=None):
        if step > 0 and step % 5 == 0:
            print(f"\n--- Validation step {step} ---")
            avg_reward = self.validate()
            print(f"Val avg reward: {avg_reward:.4f}")

            if self.best_reward is None:
                self.best_reward = avg_reward
                print(f"[EarlyStop] Initial reward: {avg_reward:.4f}")
                return False

            improvement = avg_reward - self.best_reward
            if improvement > self.min_improvement:
                self.best_reward = avg_reward
                self.no_improve_count = 0
                print(f"[EarlyStop] Reward improved: {avg_reward:.4f} (+{improvement:.4f})")
                return False
            else:
                self.no_improve_count += 1
                print(f"[EarlyStop] No improvement ({self.no_improve_count}/{self.patience}): {avg_reward:.4f}")
                if self.no_improve_count >= self.patience:
                    print("[EarlyStop] Triggered early stopping!")
                    return True
        return False


def main():
    print("=" * 60)
    print("PPO Continue Training with new alignment_discriminator_codebert_v2")
    print("=" * 60)

    print("\nLoading data...")
    train_data = load_data("data/origin/train.json")
    val_data = load_data("data/origin/test.json")
    dataset = prepare_dataset(train_data)
    val_dataset = prepare_dataset(val_data[:50])
    print(f"Train samples: {len(dataset)}, Val samples: {len(val_dataset)}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading model from: {PPO_BASE_PATH}")
    # 先加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        ACTOR_MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True
    )
    # 加载 PEFT adapter，强制 is_trainable=True 覆盖 inference_mode
    model = PeftModel.from_pretrained(base_model, PPO_BASE_PATH, is_trainable=True)
    model.to("cuda")

    model.print_trainable_parameters()

    print("\nLoading reward models...")
    reward_evaluator = RewardEvaluator(FLUENCY_MODEL_PATH, ALIGNMENT_MODEL_PATH)

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

    print("\nInitializing GRPOTrainer...")
    grpo_trainer = GRPOTrainer(
        args=grpo_config,
        model=model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    grpo_trainer.train()

    print("\nSaving final model...")
    grpo_trainer.save_model(f"{OUTPUT_DIR}/final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
