# Training Scripts Guide

## Overview

This directory contains training scripts for the DeepSceneLoc multi-model ensemble system. All scripts support dry-run testing, CPU fallback, and Kaggle T4x2 optimization.

## Scripts

### 🎯 `kaggle_training_suite.py` - Unified Runner
**Best for**: Running everything with one command

```bash
# Smoke test (verify all paths work)
python kaggle_training_suite.py --dry-run

# Train specific model
python kaggle_training_suite.py --model efficientnet
python kaggle_training_suite.py --model vit
python kaggle_training_suite.py --model resnet

# Full pipeline (all 3 models + ensemble)
python kaggle_training_suite.py --full-pipeline

# Fast eval (test transform fix on existing checkpoints)
python kaggle_training_suite.py --eval-only
```

**Features:**
- Progress tracking
- JSON summaries
- Error handling
- Time estimates

---

### 🔧 Individual Training Scripts

#### `run_training_efficientnet_b0.py`
**Recommended first model** - Fast training, good accuracy baseline

```bash
# Full training (Kaggle T4x2: 2-3 hours)
python run_training_efficientnet_b0.py --batch 128 --epochs 45 --full-finetune --swa

# Memory-constrained
python run_training_efficientnet_b0.py --batch 64 --epochs 45 --full-finetune --swa

# Smoke test
python run_training_efficientnet_b0.py --dry-run --allow-cpu
```

**Key features:**
- EfficientNet-B0 backbone (timm pretrained)
- Layer-wise learning rate decay (LLRD)
- Stochastic weight averaging (SWA)
- Full fine-tuning
- Aspect-preserving transforms (Resize(256) → CenterCrop(224))

**Expected**: 86-87% test accuracy

---

#### `run_training_vit_b16.py`
**Best single model** - Highest accuracy, slower training

```bash
# Full training (Kaggle T4x2: 3-4 hours)
python run_training_vit_b16.py --batch 96 --epochs 45 --full-finetune --swa

# Memory-constrained
python run_training_vit_b16.py --batch 48 --epochs 45 --full-finetune --swa

# Smoke test
python run_training_vit_b16.py --dry-run --allow-cpu
```

**Key features:**
- Vision Transformer B/16 (timm pretrained)
- LLRD optimized for transformer layers
- SWA
- Full fine-tuning
- Attention-based feature extraction

**Expected**: 87-88% test accuracy

---

#### `run_training_resnet50.py`
**Classic baseline** - Fast, reliable, no SWA

```bash
# Full training (Kaggle T4x2: 2-3 hours)
python run_training_resnet50.py --batch 128 --epochs 40 --full-finetune

# Memory-constrained
python run_training_resnet50.py --batch 64 --epochs 40 --full-finetune

# Smoke test
python run_training_resnet50.py --dry-run --allow-cpu
```

**Key features:**
- ResNet50 backbone (torchvision pretrained)
- LLRD
- Full fine-tuning
- No SWA (for diversity in ensemble)

**Expected**: 84-85% test accuracy

---

#### `run_ensemble_eval.py`
**Ensemble evaluation** - Combines trained models

```bash
# Full ensemble (val-weighted + calibrated)
python run_ensemble_eval.py \
    --ckpts models/checkpoints/efficientnet/EfficientNet-B0_best.pth \
            models/checkpoints/vit/ViT-B_16_best.pth \
            models/checkpoints/resnet/best_model.pth \
    --data data/processed \
    --weight-by-val-acc \
    --calibrate

# Fast eval (test transform fix on existing checkpoint)
python run_ensemble_eval.py \
    --ckpts models/checkpoints/efficientnet/EfficientNet-B0_best.pth \
    --data data/processed \
    --eval-only

# Smoke test
python run_ensemble_eval.py \
    --ckpts models/checkpoints/efficientnet/EfficientNet-B0_best.pth \
    --data data/processed \
    --dry-run --allow-cpu
```

**Key features:**
- Val-accuracy weighted averaging
- Per-model temperature calibration
- Test time augmentation (TTA)
- Expected calibration error (ECE) metrics

**Expected**: 88-89% test accuracy

---

## Common Arguments

### All Training Scripts
| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/processed` | Dataset directory |
| `--batch` | Varies | Batch size (GPU dependent) |
| `--epochs` | Varies | Training epochs |
| `--lr` | Auto | Peak learning rate |
| `--full-finetune` | - | Fine-tune all layers (vs frozen backbone) |
| `--swa` | - | Enable Stochastic Weight Averaging |
| `--dry-run` | - | Smoke test without actual training |
| `--allow-cpu` | - | Allow CPU execution (debug only) |
| `--workers` | 4 | DataLoader workers |

### Ensemble Script
| Flag | Description |
|------|-------------|
| `--ckpts` | Checkpoint paths (required) |
| `--weight-by-val-acc` | Weight models by validation accuracy |
| `--calibrate` | Apply temperature scaling |
| `--eval-only` | Fast eval on existing checkpoints |

---

## Training Workflow

### 1. Local Development (Smoke Testing)
```bash
# Verify all scripts work before Kaggle commit
python kaggle_training_suite.py --dry-run
```

### 2. Kaggle Training
```bash
# Option A: Full auto
python kaggle_training_suite.py --full-pipeline

# Option B: Individual control
python run_training_efficientnet_b0.py --batch 128 --epochs 45 --full-finetune --swa
python run_training_vit_b16.py --batch 96 --epochs 45 --full-finetune --swa
python run_training_resnet50.py --batch 128 --epochs 40 --full-finetune
python run_ensemble_eval.py --ckpts [...] --weight-by-val-acc --calibrate
```

### 3. Results Collection
Checkpoints saved to:
```
models/checkpoints/
├── efficientnet/
│   └── EfficientNet-B0_best.pth
├── vit/
│   └── ViT-B_16_best.pth
└── resnet/
    └── best_model.pth
```

Evaluation results saved to:
```
results/
├── ensemble_YYYYMMDD_HHMMSS/
│   └── ensemble_evaluation.json
└── training_summary_YYYYMMDD_HHMMSS.json
```

---

## Technical Details

### Transform Fix
**Critical for accuracy boost (+1-2%)**

Old (distorts images):
```python
transforms.Resize((224, 224))  # Squashes/stretches
```

New (aspect-preserving):
```python
transforms.Resize(256),         # Resize shorter side
transforms.CenterCrop(224),     # Crop center square
```

### Layer-wise Learning Rate Decay (LLRD)
Early layers (generic features): Lower LR  
Late layers (task-specific): Higher LR

```python
# Example: EfficientNet
backbone_lr = base_lr * 0.1      # Early layers
classifier_lr = base_lr          # Task head
```

### Stochastic Weight Averaging (SWA)
Averages model weights from last N epochs:
- Smoother loss landscape
- Better generalization
- Free +0.5-1% accuracy

Activated automatically when `--swa` flag is used.

### Temperature Calibration
Scales model confidence without changing predictions:
```python
calibrated_probs = softmax(logits / temperature)
```
- Improves Expected Calibration Error (ECE)
- Better uncertainty estimates
- Applied per-model in ensemble

---

## Memory & Performance

### GPU Memory Usage (Batch Size Guidelines)

**T4 (16GB):**
- EfficientNet-B0: 128
- ViT-B/16: 96
- ResNet50: 128

**P100 (16GB):**
- Similar to T4

**V100 (32GB):**
- Can double all batch sizes

**Memory issues?** Reduce `--batch` by half.

### Training Time Estimates (Kaggle T4x2)

| Model | Batch | Epochs | Time |
|-------|-------|--------|------|
| EfficientNet-B0 | 128 | 45 | 2-3h |
| ViT-B/16 | 96 | 45 | 3-4h |
| ResNet50 | 128 | 40 | 2-3h |
| **Total** | - | - | **8-10h** |

Ensemble eval: ~5 minutes

---

## Troubleshooting

### Import Errors
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
```

### Checkpoint Not Found
```bash
# Verify paths
ls -la models/checkpoints/efficientnet/
ls -la models/checkpoints/vit/
ls -la models/checkpoints/resnet/
```

### Out of Memory
1. Reduce `--batch` size
2. Reduce `--workers` (e.g., `--workers 2`)
3. Close other processes

### CUDA Errors
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU for debugging
python script.py --allow-cpu --dry-run
```

### Transform Verification
```python
from src.preprocessing.transforms import get_val_transforms
tfm = get_val_transforms(224)
print(tfm)
# Should show: Resize(256), CenterCrop(224)
```

---

## Expected Results

### Individual Models

| Model | Architecture | Val Acc | Test Acc | SWA | LLRD |
|-------|--------------|---------|----------|-----|------|
| EfficientNet-B0 | CNN (compound scaling) | 87% | 86% | ✅ | ✅ |
| ViT-B/16 | Transformer | 88% | 87% | ✅ | ✅ |
| ResNet50 | CNN (residual) | 85% | 84% | ❌ | ✅ |

### Ensemble

| Method | Val Acc | Test Acc | ECE |
|--------|---------|----------|-----|
| Uniform average | 88% | 87% | 0.03 |
| Val-weighted | 89% | 88% | 0.02 |
| **Val-weighted + Calibrated** | **89%** | **88%** | **0.015** |

**Transform fix alone**: +1-2% on any existing checkpoint

---

## Kaggle-Specific Notes

### Path Handling
```python
# Local
DATA_DIR = "data/processed"

# Kaggle (if dataset is external input)
DATA_DIR = "/kaggle/input/deepsceneloc-data/data/processed"
```

### Saving Outputs
```python
# Kaggle auto-saves /kaggle/working/
import shutil
shutil.copytree("models/checkpoints", "/kaggle/working/checkpoints")
shutil.copytree("results", "/kaggle/working/results")
```

### GPU Quota
- Kaggle T4: 30 hours/week
- Full pipeline: ~10 hours
- Leave buffer for experimentation

---

## Advanced Usage

### Custom Hyperparameters
```bash
# Different learning rate
python run_training_efficientnet_b0.py --lr 3e-4 --epochs 50

# Different weight decay
python run_training_vit_b16.py --weight-decay 0.01

# Gradient accumulation (simulate larger batch)
python run_training_resnet50.py --batch 32 --grad-accum 4  # Effective batch: 128
```

### Subset Training (Fast Debugging)
```bash
# Train on 10% of data
python run_training_efficientnet_b0.py --data-fraction 0.1 --epochs 5
```

### Resume Training
```bash
# Resume from checkpoint
python run_training_vit_b16.py --resume models/checkpoints/vit/ViT-B_16_epoch_20.pth
```

---

## References

- **Main guide**: `QUICK_START.md`
- **Kaggle notebook**: `KAGGLE_TRAINING_NOTEBOOK.md`
- **Methodology**: `docs/TRAINING_METHODOLOGY.md`
- **Performance tips**: `docs/PERFORMANCE_OPTIMIZATION.md`

---

## Questions?

Check:
1. `QUICK_START.md` for commands
2. `SUMMARY_NEXT_STEPS.md` for workflow
3. `docs/TRAINING_METHODOLOGY.md` for theory
4. Project logs in `docs/16_WEEK_PROGRESS_TRACKING.md`
