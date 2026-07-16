# DeepSceneLoc - Quick Start Guide

## 🎯 Priority Tasks

### 1. **Fastest Proof** (5 minutes)
Test transform fix on existing EfficientNet checkpoint:
```bash
python run_ensemble_eval.py \
    --ckpts models/checkpoints/efficientnet/EfficientNet-B0_best.pth \
    --data data/processed \
    --eval-only
```
Expected: +1-2% accuracy boost from aspect-preserving transforms.

### 2. **Smoke Test** (30 seconds)
Verify all training paths work before GPU commit:
```bash
python kaggle_training_suite.py --dry-run
```
Or individual scripts:
```bash
python run_training_efficientnet_b0.py --dry-run --allow-cpu
python run_training_vit_b16.py --dry-run --allow-cpu
python run_training_resnet50.py --dry-run --allow-cpu
```

### 3. **Full Training Pipeline** (8-10 hours on Kaggle T4x2)
```bash
# Option A: One command for everything
python kaggle_training_suite.py --full-pipeline

# Option B: Individual models (recommended for control)
python run_training_efficientnet_b0.py --batch 128 --epochs 45 --full-finetune --swa
python run_training_vit_b16.py --batch 96 --epochs 45 --full-finetune --swa
python run_training_resnet50.py --batch 128 --epochs 40 --full-finetune

# Then ensemble
python run_ensemble_eval.py \
    --ckpts models/checkpoints/efficientnet/EfficientNet-B0_best.pth \
            models/checkpoints/vit/ViT-B_16_best.pth \
            models/checkpoints/resnet/best_model.pth \
    --data data/processed \
    --weight-by-val-acc \
    --calibrate
```

---

## 📋 Training Commands Reference

### EfficientNet-B0 (Recommended First)
```bash
# Full training with SWA
python run_training_efficientnet_b0.py --batch 128 --epochs 45 --full-finetune --swa

# Reduced memory
python run_training_efficientnet_b0.py --batch 64 --epochs 45 --full-finetune --swa

# Dry run test
python run_training_efficientnet_b0.py --dry-run --allow-cpu
```

### Vision Transformer (ViT-B/16)
```bash
# Full training with SWA
python run_training_vit_b16.py --batch 96 --epochs 45 --full-finetune --swa

# Reduced memory
python run_training_vit_b16.py --batch 48 --epochs 45 --full-finetune --swa

# Dry run test
python run_training_vit_b16.py --dry-run --allow-cpu
```

### ResNet50
```bash
# Full training
python run_training_resnet50.py --batch 128 --epochs 40 --full-finetune

# Reduced memory
python run_training_resnet50.py --batch 64 --epochs 40 --full-finetune

# Dry run test
python run_training_resnet50.py --dry-run --allow-cpu
```

### Ensemble Evaluation
```bash
# Full evaluation with calibration
python run_ensemble_eval.py \
    --ckpts models/checkpoints/efficientnet/EfficientNet-B0_best.pth \
            models/checkpoints/vit/ViT-B_16_best.pth \
            models/checkpoints/resnet/best_model.pth \
    --data data/processed \
    --weight-by-val-acc \
    --calibrate

# Eval only (no training)
python run_ensemble_eval.py \
    --ckpts models/checkpoints/efficientnet/EfficientNet-B0_best.pth \
    --data data/processed \
    --eval-only

# Dry run test
python run_ensemble_eval.py \
    --ckpts models/checkpoints/efficientnet/EfficientNet-B0_best.pth \
    --data data/processed \
    --dry-run --allow-cpu
```

---

## 🚀 Kaggle Workflow

### Upload to Kaggle
1. Create new notebook (GPU: T4 x2)
2. Upload repo as dataset OR clone from GitHub:
   ```bash
   !git clone https://github.com/YOUR_USER/DeepSceneLoc.git
   cd DeepSceneLoc
   ```
3. Install dependencies (if needed):
   ```bash
   !pip install timm torchmetrics pyyaml
   ```

### Run Training
```python
# Verify environment
!python kaggle_training_suite.py --dry-run

# Train all models
!python kaggle_training_suite.py --full-pipeline
```

### Save Results
```python
# Kaggle auto-saves /kaggle/working/
!cp -r models/checkpoints /kaggle/working/
!cp -r results /kaggle/working/
```

---

## 🔧 Troubleshooting

### Import Errors
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
```

### Out of Memory
- Reduce `--batch` size
- Use gradient accumulation (if implemented)
- Close other processes

### Checkpoint Not Found
```bash
# Check paths
ls -la models/checkpoints/efficientnet/
ls -la models/checkpoints/vit/
ls -la models/checkpoints/resnet/
```

### Verify Transforms
```python
from src.preprocessing.transforms import get_val_transforms
tfm = get_val_transforms(224)
print(tfm)
# Should show: Resize(256) → CenterCrop(224)
# NOT: Resize(224)
```

---

## 📊 Expected Results

| Component | Val Acc | Test Acc | Time (T4x2) |
|-----------|---------|----------|-------------|
| EfficientNet-B0 + SWA | 87% | 86% | 2-3h |
| ViT-B/16 + SWA | 88% | 87% | 3-4h |
| ResNet50 | 85% | 84% | 2-3h |
| **Ensemble** | **89%** | **88%** | 5min |

Transform fix: +1-2% on existing checkpoints

---

## 🗂️ File Structure

```
DeepSceneLoc/
├── run_training_efficientnet_b0.py  # EfficientNet trainer
├── run_training_vit_b16.py          # ViT trainer
├── run_training_resnet50.py         # ResNet trainer
├── run_ensemble_eval.py             # Ensemble evaluator
├── kaggle_training_suite.py         # Unified runner
├── src/
│   ├── models/                       # Model definitions
│   ├── preprocessing/                # Data pipelines
│   └── evaluation/                   # Metrics & ensemble
├── data/processed/                   # Dataset (train/val/test)
└── models/checkpoints/               # Saved models
    ├── efficientnet/
    ├── vit/
    └── resnet/
```

---

## 📝 Notes

1. **Transform fix is critical**: Aspect-preserving resize → center crop
2. **SWA** (Stochastic Weight Averaging) adds ~0.5-1% accuracy for free
3. **Full fine-tuning** > frozen backbone for this dataset size
4. **Ensemble** typically beats best single model by 1-2%
5. **Calibration** improves confidence scores (ECE metric)

---

## ⏰ Time Estimates

- Smoke test: 30 seconds
- Eval-only: 5 minutes
- Single model training: 2-4 hours
- Full pipeline (3 models + ensemble): 8-10 hours
- Kaggle GPU quota: 30 hours/week (T4)

---

## 🎓 Academic Context

This is the Semester 2 implementation for:
- **DeepSceneLoc**: Visual Scene Understanding for Image-Based Location Estimation
- **Goal**: 88%+ test accuracy via multi-model ensemble
- **Key techniques**: Transform fix, SWA, LLRD, temperature calibration

---

## 📚 References

- Training methodology: `docs/TRAINING_METHODOLOGY.md`
- Performance optimization: `docs/PERFORMANCE_OPTIMIZATION.md`
- Full pipeline: `KAGGLE_TRAINING_NOTEBOOK.md`
- Repo cleanup: `REPO_CLEANUP_PLAN.md`
