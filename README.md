# T2S-Augmentation

Controllable Buggy Code Generation via Cold-Start Reinforcement Learning

## Project Overview

This project implements a controllable buggy code generation framework using cold-start PPO (Proximal Policy Optimization) reinforcement learning. The framework generates realistic buggy code with specified error types for data augmentation in program repair tasks.

## Directory Structure

```
T2S-Augmentation/
├── data/                           # Dataset directory
│   ├── origin/                     # Original dataset
│   │   ├── train.json              # Training data (528 samples)
│   │   └── test.json               # Test data (133 samples)
│   └── augmented/                   # Augmented/generated data
│
├── output/                         # Trained models and checkpoints
│   ├── alignment_discriminator_v2/  # Alignment discriminator
│   ├── fluency_discriminator_codebert/  # Fluency discriminator
│   ├── ppo_trl_v2_continued/        # PPO-trained generator
│   └── *.pt, *.safetensors          # Model weights
│
├── thesis/                         # Thesis/latex files
│   ├── latex_ntu/                   # LaTeX thesis
│   │   ├── chapter-1/ - chapter-6/   # Thesis chapters
│   │   ├── c-front-matter/          # Front matter
│   │   ├── c-back-matter/           # Back matter, references
│   │   ├── assets/
│   │   │   ├── figures/             # Experiment figures
│   │   │   └── *.png                # Images
│   │   └── main.tex                 # Main thesis file
│   └── generate_training_figures.py # Script to generate figures
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── metric.py                   # Evaluation metrics
│   ├── triplet.py                  # Triplet utilities
│   └── trlx_utils.py               # TRLX utilities
│
├── wandb/                         # Weights & Biases logs
│
└── *.py                           # Main training scripts
```

## Core Training Scripts

### Generators
| File | Description |
|------|-------------|
| `ppo_tuning_trl.py` | Main PPO training script with cold-start RL |
| `ppo_continue.py` | Continue training from checkpoint |
| `ppo_continue_with_new_discriminator.py` | Continue with updated discriminators |

### Discriminators
| File | Description |
|------|-------------|
| `train_alignment_discriminator_codebert.py` | Train alignment discriminator |
| `train_alignment_discriminator_v2.py` | Alignment discriminator v2 |
| `train_fluency_discriminator_codebert.py` | Train fluency discriminator |

### Evaluation
| File | Description |
|------|-------------|
| `baseline_qwen_codebert_fixed.py` | Baseline comparison |
| `baseline_fluency_comparison.py` | Fluency comparison baselines |
| `compare_discriminators.py` | Compare discriminator performance |

### Utilities
| File | Description |
|------|-------------|
| `download_model_hf.py` | Download models from HuggingFace |
| `convert_data.py` | Data format conversion |

## Dataset Format

Each sample in `train.json` / `test.json` contains:

```json
{
  "code": "# source code with bugs",
  "text": "Label: [error_type]\nAST: (abstract syntax tree)"
}
```

### Error Types
- **Logical Error**: Bugs in algorithmic logic
- **Memory Error**: Incorrect memory operations
- **Omission**: Missing code elements
- **Invalid Condition**: Incorrect conditional expressions
- **Infinite Loop Error**: Non-terminating loops

## Usage

### 1. Train Alignment Discriminator
```bash
python train_alignment_discriminator_codebert.py
```

### 2. Train Fluency Discriminator
```bash
python train_fluency_discriminator_codebert.py
```

### 3. Train Generator with Cold-Start PPO
```bash
python ppo_tuning_trl.py
```

### 4. Generate Training Figures
```bash
cd thesis && python generate_training_figures.py
```

## Key Results

### Code Error Classification (Multi-label)
| Metric | Base Model | Our Method | Improvement |
|--------|------------|------------|-------------|
| Micro F1 | 60.6% | 62.5% | +1.89 pp |
| Macro F1 | - | 0.65 | - |

### Generation Quality
| Metric | Score |
|--------|-------|
| Fluency | 0.89 |
| Consistency | 0.84 |
| CodeBERT Accuracy | 68.7% |

### Ablation Study
- **Without AST Conditioning**: Macro-F1 drops to 0.51 (-14 pp)
- **Without Consistency Discriminator**: Macro-F1 drops to 0.54 (-11 pp)
- **Cold-start RL** outperforms supervised fine-tuning alone

## License

MIT License
