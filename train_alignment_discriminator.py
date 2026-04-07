
import os
import time
import random
import math
import argparse
import torch
import difflib
import pytorch_lightning as pl 
pl.seed_everything(42)

import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification
from torch.utils.data import DataLoader

from utils import load_json, tgenerate_batch, generate_batch
from utils.triplet import make_triplets_seq


# 多标签错误类型名称（顺序固定）
LABEL_NAMES = [
    'logical_error',
    'memory_error',
    'omission',
    'invalid_condition',
    'infinite_loop_error',
    'unknown',
]


def parse_label_vector(labels_str: str):
    """
    将原始的标签字符串转换为 6 维多标签向量（0/1）
    例如：
    "Logical Error,Memory Error,Omission, Invalid Condition"
    -> [1, 1, 1, 1, 0, 0]
    """
    labels_str = labels_str.strip().strip("'").strip('"')

    # 拆分为多个 label 片段
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
        elif 'invalid' in lp:
            normalized.append('invalid_condition')
        elif 'infinite' in lp:
            normalized.append('infinite_loop_error')
        else:
            normalized.append('unknown')

    vec = []
    for name in LABEL_NAMES:
        vec.append(1 if name in normalized else 0)
    return vec



class DataModule(pl.LightningDataModule):
    def __init__(self,
                 model_name_or_path: str='',
                 max_seq_length: int = -1,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 data_dir: str = '',
                 seed: int = 42):

        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length     = max_seq_length
        self.train_batch_size   = train_batch_size
        self.eval_batch_size    = eval_batch_size
        self.data_dir           = data_dir
        self.seed               = seed

        # 这里的 model_name_or_path 建议传入 "microsoft/codebert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
    
    def load_dataset(self):
        """直接从 train.json 构建多标签分类数据
        - 输入：标签字符串 + 代码
        - 输出：多标签二进制向量
        """
        train_file_name = os.path.join(self.data_dir, 'train.json')
        all_examples = load_json(train_file_name)

        examples = []
        for ex in all_examples:
            if 'text' in ex and 'code' in ex:
                text = ex['text']
                code = ex['code']
                if 'Label:' in text:
                    label_start = text.find('Label:') + 6
                    label_end = text.find('\nAST:')
                    if label_end != -1:
                        labels_text = text[label_start:label_end].strip()
                        examples.append({
                            'labels_str': labels_text,  # 标签字符串
                            'sentence': code,            # 代码
            })

        # 8:1:1 划分
        n = len(examples)
        n_train = int(0.8 * n)
        n_dev = int(0.9 * n)

        self.raw_datasets = {
            'train': examples[:n_train],
            'dev'  : examples[n_train:n_dev],
            'test' : examples[n_dev:],
        }

        print('--------- data statistic ---------')
        print('train:', len(self.raw_datasets['train']))
        print('dev:',   len(self.raw_datasets['dev']))
        print('test:',  len(self.raw_datasets['test']))
        print()

    def get_dataloader(self, mode, batch_size, shuffle):
        dataloader = DataLoader(
            dataset=self.raw_datasets[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            prefetch_factor=8,
            num_workers=1,
            collate_fn=MultiLabelDataCollator(
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
            )
        )
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader('dev', self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader('test', self.eval_batch_size, shuffle=False)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--model_name_or_path", type=str, required=True)
        parser.add_argument("--max_seq_length", type=int, default=512)
        parser.add_argument("--train_batch_size", type=int, default=4)
        parser.add_argument("--eval_batch_size", type=int, default=4)
        parser.add_argument("--data_dir", type=str, required=True)
        parser.add_argument("--seed", type=int, default=42)
        return parser



def tok(tokenizer, text, max_seq_length):
    kwargs = {
        'text': text,
        'return_tensors': 'pt'
    }

    if max_seq_length in (-1, 'longest'):
        kwargs['padding'] = True

    else:
        kwargs['padding'] = True
        kwargs['truncation'] = True
        kwargs['max_length'] = max_seq_length

    batch_encodings = tokenizer(**kwargs)
    return batch_encodings




class MultiLabelDataCollator:
    """多标签分类的数据整理器"""

    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # 使用全局 LABEL_NAMES
        self.label_names = LABEL_NAMES

    def tok(self, text):
        return tok(self.tokenizer, text, self.max_seq_length)

    def __call__(self, examples):
        texts = []
        labels = []

        for example in examples:
            # 这里只用代码作为输入；如果需要可以拼接 labels_str
            text = example['sentence']
            texts.append(text)

            # 将标签字符串转换为多标签二进制向量
            label_vector = parse_label_vector(example['labels_str'])
            labels.append(label_vector)

        # Tokenize
        encodings = self.tok(texts)
        labels = torch.tensor(labels, dtype=torch.float32)

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }

class DataCollator:

    def __init__(self, tokenizer, max_seq_length, mode):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode

    def tok(self, text):
        return tok(self.tokenizer, text, self.max_seq_length)

    def __call__(self, examples):
        conditions = []
        real_samples = []
        fake_samples = []
        fake_type_ids  = []

        for example in examples:
            triplets_seq = example['triplets_seq']
            real_sample  = triplets_seq + ' ; ' + example['sentence']
            real_samples.append(real_sample)

            negatives = [
                (i, negative) 
                for i, negative in enumerate(example['triplets_seq_beam']) 
                if negative != triplets_seq
            ]

            if self.mode == 'train':

                if len(negatives) > 0:
                    i, negative = random.choice(negatives)
                    fake_sample = negative + ' ; ' + example['sentence']
                    fake_samples.append(fake_sample)
                    fake_type_ids.append(i)

            else:
                for i, negative in negatives:
                    fake_sample = negative + ' ; ' + example['sentence']
                    fake_samples.append(fake_sample)
                    fake_type_ids.append(i)

        real_batch_encodings = self.tok(real_samples)
        fake_batch_encodings = self.tok(fake_samples)

        real_input_ids = real_batch_encodings['input_ids']
        fake_input_ids = fake_batch_encodings['input_ids']
        real_attention_mask = real_batch_encodings['attention_mask']
        fake_attention_mask = fake_batch_encodings['attention_mask']

        fake_type_ids  = torch.tensor(fake_type_ids)

        return {
            'real_input_ids': real_input_ids,
            'fake_input_ids': fake_input_ids,
            'real_attention_mask': real_attention_mask,
            'fake_attention_mask': fake_attention_mask,
            'fake_type_ids': fake_type_ids
        }



class LightningModule(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        # 使用CodeBERT作为多标签分类骨干网络
        # 建议 model_name_or_path 传入 "microsoft/codebert-base"
        print(f"## 初始化多标签CodeBERT判别器: {self.hparams.model_name_or_path}")
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.hparams.model_name_or_path,
            num_labels=len(LABEL_NAMES),
            problem_type="multi_label_classification",
            torch_dtype=torch.float32,
        )

        # 冻结前 6 层 encoder
        freeze_layers = 6
        print(f"## 冻结前 {freeze_layers} 层 encoder")
        for layer in self.model.roberta.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        # 确保模型有pad_token_id，避免 batch_size>1 报错
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.data_module.tokenizer.pad_token_id
        
        # 确保模型在训练模式下
        self.model.train()
        
        # 初始化指标追踪
        self.current_val_metric = {}
        self.best_val_metric = {}
        self.test_metric = {}
    
    def _make_model_dir(self):
        # 使用ModelCheckpoint保存的目录路径
        return os.path.join(self.hparams.output_dir, 'model', self.hparams.output_sub_dir)

    @pl.utilities.rank_zero_only
    def save_model(self):
        dir_name = self._make_model_dir()
        print(f'## save model to {dir_name}')
        os.makedirs(dir_name, exist_ok=True)
        self.model.config.time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        self.model.save_pretrained(dir_name)
        self.data_module.tokenizer.save_pretrained(dir_name)
        print(f'[OK] 模型已保存到: {dir_name}')

    def on_train_end(self):
        """训练结束时手动保存模型"""
        print('== 训练完成，开始保存模型...')
        self.save_model()

    def load_model(self):
        dir_name = self._make_model_dir()
        print(f'## load model to {dir_name}')
        # 使用绝对路径和正确的本地加载方式
        abs_path = os.path.abspath(dir_name)
        print(f'## absolute path: {abs_path}')

        # 检查模型文件是否存在
        config_path = os.path.join(abs_path, 'config.json')
        model_path = os.path.join(abs_path, 'model.safetensors')

        if not os.path.exists(config_path):
            print(f'[WARNING] 警告: 配置文件不存在 {config_path}，跳过加载测试')
            return

        try:
            # 从保存的本地目录重新加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(
                abs_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                local_files_only=True
            )
            print('[OK] 模型加载成功')
        except Exception as e:
            print(f'[ERROR] 模型加载失败: {e}')
            print('[WARNING] 训练仍然成功，模型已保存到磁盘')

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), 
            eps=self.hparams.adam_epsilon, 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]

    def decode(self, ids):
        return self.data_module.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def forward(self, **batch):
        """多标签分类前向传播"""
        labels = batch['labels']

        # 模型前向传播
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=labels  # 传递标签，自动计算BCEWithLogitsLoss
        )

        logits = outputs.logits  # (batch_size, num_labels)

        # 计算预测概率
        probs = torch.sigmoid(logits)

        # 计算预测结果（阈值=0.5）
        predictions = (probs > 0.5).float()

        # 计算样本级准确率（所有标签都预测正确）
        sample_accuracy = (predictions == labels).all(dim=1).float().mean()

        # 计算标签级准确率（每个标签的预测准确率平均）
        label_accuracy = (predictions == labels).float().mean()

        # 详细的batch级日志（只在rank 0打印，避免刷屏）
        if self.global_step % 50 == 0:
            pos_per_label = predictions.sum(dim=0).detach().cpu().numpy().tolist()
            true_per_label = labels.sum(dim=0).detach().cpu().numpy().tolist()
            print(f"[step {self.global_step}] batch_sample_acc={sample_accuracy.item():.4f}, batch_label_acc={label_accuracy.item():.4f}")
            print(f"[step {self.global_step}] 预测为1的数量: {pos_per_label}")
            print(f"[step {self.global_step}] 真实为1的数量: {true_per_label}")

        return {
            'loss': outputs.loss,
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
            'labels': labels,
            'sample_accuracy': sample_accuracy,
            'label_accuracy': label_accuracy,
        }
    
    def training_step(self, batch, batch_idx):
        output = self(**batch)

        loss = output['loss']
        sample_accuracy = output['sample_accuracy']
        label_accuracy = output['label_accuracy']

        # 记录训练指标
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_sample_acc', sample_accuracy, prog_bar=True)
        self.log('train_label_acc', label_accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)

        result = {
            'loss': output['loss'].item(),
            'sample_accuracy': output['sample_accuracy'].item(),
            'label_accuracy': output['label_accuracy'].item(),
        }

        # 保存输出用于epoch结束时的聚合
        if not hasattr(self, 'validation_step_outputs'):
            self.validation_step_outputs = []
        self.validation_step_outputs.append(result)

        return result
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        output = self(**batch)
        
        result = {
            'loss': output['loss'].item(),
            'sample_accuracy': output['sample_accuracy'].item(),
            'label_accuracy': output['label_accuracy'].item(),
        }
        
        # 保存输出用于epoch结束时的聚合
        if not hasattr(self, 'test_step_outputs'):
            self.test_step_outputs = []
        self.test_step_outputs.append(result)
        
        return result

    @staticmethod
    def eval_epoch_end(outputs):
        loss = sum([output['loss'] for output in outputs]) / len(outputs)
        sample_accuracy = sum([output['sample_accuracy'] for output in outputs]) / len(outputs)
        label_accuracy = sum([output['label_accuracy'] for output in outputs]) / len(outputs)
        
        metric = {
            'loss': loss,
            'sample_accuracy': sample_accuracy,
            'label_accuracy': label_accuracy,
            'monitor': sample_accuracy,  # 使用样本准确率作为监控指标
        }
        return metric

    def on_validation_epoch_end(self):
        """验证epoch结束时调用，设置current_val_metric并更新best_val_metric"""
        if hasattr(self, 'validation_step_outputs') and len(self.validation_step_outputs) > 0:
            metric = self.eval_epoch_end(self.validation_step_outputs)
            self.current_val_metric = metric
            
            # 更新最佳指标（基于monitor值）
            if not self.best_val_metric or metric['monitor'] > self.best_val_metric.get('monitor', -float('inf')):
                self.best_val_metric = metric.copy()
            
            # 记录到tensorboard
            self.log('val_loss', metric['loss'])
            self.log('val_sample_acc', metric['sample_accuracy'])
            self.log('val_label_acc', metric['label_accuracy'])
            
            # 清空输出列表
            self.validation_step_outputs = []
    
    def on_test_epoch_end(self):
        """测试epoch结束时调用，设置test_metric"""
        if hasattr(self, 'test_step_outputs') and len(self.test_step_outputs) > 0:
            metric = self.eval_epoch_end(self.test_step_outputs)
            self.test_metric = metric
            
            # 记录到tensorboard
            self.log('test_loss', metric['loss'])
            self.log('test_sample_acc', metric['sample_accuracy'])
            self.log('test_label_acc', metric['label_accuracy'])
            
            # 清空输出列表
            self.test_step_outputs = []

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0., type=float)

        parser.add_argument("--output_dir", type=str)
        parser.add_argument("--output_sub_dir", type=str)
        parser.add_argument("--do_train", action='store_true')

        return parser



class LoggingCallback(pl.Callback):
    
    def print_dict(self, prefix, dic):
            print(prefix + ' | '.join([f'{k}: {v:.4f}' for k,v in dic.items()]))

    def on_validation_end(self, trainer, pl_module):
        print()
        self.print_dict('[current] ', pl_module.current_val_metric)
        self.print_dict('[best]    ', pl_module.best_val_metric)
        print()

    def on_test_end(self, trainer, pl_module):
        print()
        self.print_dict('[test]', pl_module.test_metric)
        print()



def main():
    parser = argparse.ArgumentParser()
    # 手动添加Trainer相关参数（替代已弃用的add_argparse_args）
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1)  # 只训练1个epoch
    parser.add_argument("--val_check_interval", type=int, default=1)  # 每1步验证一次（相当于每个epoch验证）

    parser = LightningModule.add_model_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    if args.learning_rate >= 1:
        args.learning_rate /= 1e5

    # 自动根据 batch_size 调整学习率（线性缩放规则）
    # 基准：batch_size=4, lr=1e-5
    base_batch_size = 4
    base_lr = 1e-5
    if args.train_batch_size != base_batch_size:
        # 使用线性缩放规则
        scale_factor = args.train_batch_size / base_batch_size
        adjusted_lr = args.learning_rate * scale_factor
        print(f'[自动调整] batch_size: {args.train_batch_size} (基准: {base_batch_size})')
        print(f'[自动调整] 学习率: {args.learning_rate} -> {adjusted_lr:.2e} (缩放因子: {scale_factor:.2f}x)')
        args.learning_rate = adjusted_lr

    data_module = DataModule(
        model_name_or_path=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        data_dir=args.data_dir,
        seed=args.seed,
    )
    data_module.load_dataset()

    model = LightningModule(args, data_module)

    # 自动调整 val_check_interval：如果超过训练批次数，则调整为训练批次数的一半
    if args.do_train:
        train_dataloader = data_module.train_dataloader()
        num_train_batches = len(train_dataloader)
        if args.val_check_interval > num_train_batches:
            adjusted_interval = max(1, num_train_batches // 2)
            print(f'Warning: val_check_interval ({args.val_check_interval}) > num_train_batches ({num_train_batches})')
            print(f'Automatically adjusting val_check_interval to {adjusted_interval}')
            args.val_check_interval = adjusted_interval

    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

    # 确保输出目录存在
    model_dir = os.path.join(args.output_dir, 'model', args.output_sub_dir)
    os.makedirs(model_dir, exist_ok=True)

    # 添加CSV logger来保存训练日志
    csv_logger = CSVLogger(
        save_dir=args.output_dir,
        name='logs',
        version=args.output_sub_dir
    )

    # 添加TensorBoard logger来生成loss曲线图
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='tensorboard_logs',
        version=args.output_sub_dir
    )

    # 添加ModelCheckpoint回调来保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='best-model-{epoch:02d}-{val_label_acc:.2f}',
        monitor='val_label_acc',
        mode='max',
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        every_n_epochs=1,
    )

    logging_callback = LoggingCallback()
    kwargs = {
        'callbacks': [logging_callback, checkpoint_callback],
        'logger': [csv_logger, tb_logger],  # 同时使用CSV和TensorBoard logger
        'enable_checkpointing': True,  # 启用checkpointing
        'num_sanity_val_steps': 5 if args.do_train else 0,
        'gradient_clip_val': 1.0,  # 梯度裁剪，防止梯度爆炸
        'accumulate_grad_batches': 1,  # 如果需要可以增加梯度累积来减小有效 batch_size
    }
    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        max_epochs=args.max_epochs,
        val_check_interval=args.val_check_interval,
        precision='16-mixed' if args.gpus > 0 else '32',  # 使用混合精度训练以节省内存
        **kwargs
    )

    if args.do_train:
        trainer.fit(model, datamodule=data_module)
        model.load_model()

        # 保存完整的transformers模型格式（用于PPO训练）
        print(f"Saving complete model to {model_dir}")
        model.model.config.time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        model.model.save_pretrained(model_dir)
        data_module.tokenizer.save_pretrained(model_dir)

        trainer.test(model, datamodule=data_module)

    else:
        model.load_model()
        trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()