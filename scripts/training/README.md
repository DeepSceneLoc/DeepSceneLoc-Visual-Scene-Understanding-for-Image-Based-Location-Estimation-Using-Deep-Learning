# Training Scripts

This directory contains all model training scripts for the DeepSceneLoc ensemble system.

## Files

### `run_training_efficientnet_b0.py`
Train EfficientNet-B0 with SWA and LLRD

**Usage:**
```bash
python run_training_efficientnet_b0.py --batch 128 --epochs 45 --full-finetune --swa
```

### `run_training_vit_b16.py`
Train Vision Transformer (ViT-B/16) with SWA and LLRD

**Usage:**
```bash
python run_training_vit_b16.py --batch 96 --epochs 45 --full-finetune --swa
```

### `run_training_resnet50.py`
Train ResNet50 with LLRD

**Usage:**
```bash
python run_training_resnet50.py --batch 128 --epochs 40 --full-finetune
```

## Documentation

For detailed training guide, see: `../../docs/guides/TRAINING_README.md`

For quick command reference, see: `../../docs/guides/QUICK_START.md`

## Note on Imports

These scripts expect the project root to be in the Python path. They automatically handle this with:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

## Kaggle Usage

When using on Kaggle, run from project root:
```bash
python scripts/training/run_training_efficientnet_b0.py --batch 128 --epochs 45 --full-finetune --swa
```

Or navigate here and run directly:
```bash
cd scripts/training
python run_training_efficientnet_b0.py --batch 128 --epochs 45 --full-finetune --swa
```
