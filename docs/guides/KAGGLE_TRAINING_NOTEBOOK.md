# Kaggle Training Notebook Template

## Setup Instructions

### 1. Kaggle Environment
- **GPU**: T4 x2 (or P100)
- **Persistence**: Enable "Save version" to keep checkpoints
- **Internet**: On (for downloading pretrained weights)

### 2. Upload Repository
```bash
# In Kaggle notebook cell
!git clone https://github.com/YOUR_USERNAME/DeepSceneLoc.git
cd DeepSceneLoc
```

Or upload as a Kaggle Dataset:
1. Zip your repo (exclude `.git`, `.venv`, `models/`)
2. Upload to Kaggle Datasets
3. Add as data source to notebook

---

## Notebook Cells

### Cell 1: Environment Check
```python
import torch
import sys
from pathlib import Path

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Verify paths
print("\nVerifying paths...")
required = ["src/", "data/", "run_training_efficientnet_b0.py"]
for p in required:
    exists = Path(p).exists()
    print(f"  {'✅' if exists else '❌'} {p}")
```

### Cell 2: Install Dependencies
```python
# If needed, install missing packages
!pip install timm torchmetrics scikit-learn pyyaml tqdm --quiet
```

### Cell 3: Smoke Test (Dry Run)
```python
# Verify all training scripts work before committing GPU hours
!python kaggle_training_suite.py --dry-run
```

### Cell 4A: Quick Validation Test (Fastest)
**Use this first** to verify the transform fix on existing EfficientNet checkpoint:
```python
# Assumes you have models/checkpoints/efficientnet/EfficientNet-B0_best.pth
!python run_ensemble_eval.py \
    --ckpts models/checkpoints/efficientnet/EfficientNet-B0_best.pth \
    --data data/processed \
    --eval-only \
    --allow-cpu  # Remove if GPU available
```
Expected: ~1-2% accuracy boost from transform fix alone.

### Cell 4B: Train EfficientNet-B0 (Recommended First)
```python
# ~2-3 hours on T4x2
!python run_training_efficientnet_b0.py \
    --batch 128 \
    --epochs 45 \
    --full-finetune \
    --swa
```

### Cell 5: Train ViT-B/16
```python
# ~3-4 hours on T4x2
!python run_training_vit_b16.py \
    --batch 96 \
    --epochs 45 \
    --full-finetune \
    --swa
```

### Cell 6: Train ResNet50
```python
# ~2-3 hours on T4x2
!python run_training_resnet50.py \
    --batch 128 \
    --epochs 40 \
    --full-finetune
```

### Cell 7: Ensemble Evaluation
```python
# Combine all three models
!python run_ensemble_eval.py \
    --ckpts models/checkpoints/efficientnet/EfficientNet-B0_best.pth \
            models/checkpoints/vit/ViT-B_16_best.pth \
            models/checkpoints/resnet/best_model.pth \
    --data data/processed \
    --weight-by-val-acc \
    --calibrate
```

### Cell 8: Download Checkpoints
```python
# Save trained models for later use
from google.colab import files  # If using Colab
import shutil

# Create archive
!zip -r trained_models.zip models/checkpoints/

# Download (Colab)
# files.download('trained_models.zip')

# Or copy to Kaggle Output (auto-saved)
shutil.copy('trained_models.zip', '/kaggle/working/')
print("✅ Models saved to Kaggle output")
```

---

## Alternative: Full Pipeline (One Command)
```python
# Run everything sequentially
!python kaggle_training_suite.py --full-pipeline
```

---

## Troubleshooting

### Out of Memory
```python
# Reduce batch sizes
!python run_training_efficientnet_b0.py --batch 64 --epochs 45 --full-finetune --swa
!python run_training_vit_b16.py --batch 48 --epochs 45 --full-finetune --swa
```

### Checkpoint Not Found
```python
# Check paths
!ls -la models/checkpoints/efficientnet/
!ls -la models/checkpoints/vit/
!ls -la models/checkpoints/resnet/
```

### Import Errors
```python
# Ensure project root is on path
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
```

### Transform Verification
```python
# Test that transforms are aspect-preserving
from src.preprocessing.transforms import get_val_transforms
from PIL import Image
import torchvision.transforms as T

tfm = get_val_transforms(image_size=224)
print("Transform chain:")
for t in tfm.transforms:
    print(f"  - {t}")

# Should see: Resize(256), CenterCrop(224), not Resize(224)
```

---

## Expected Results

| Model | Val Acc | Test Acc | Training Time (T4x2) |
|-------|---------|----------|---------------------|
| EfficientNet-B0 + SWA | ~87% | ~86% | 2-3 hours |
| ViT-B/16 + SWA | ~88% | ~87% | 3-4 hours |
| ResNet50 | ~85% | ~84% | 2-3 hours |
| **Ensemble (weighted + calibrated)** | **~89%** | **~88%** | 5 minutes |

Transform fix alone (eval-only): +1-2% on existing checkpoints

---

## Tips

1. **Start with smoke test** (`--dry-run`) to catch import/path errors
2. **Run eval-only first** to verify transform fix on existing checkpoint
3. **Train EfficientNet first** - fastest, good baseline
4. **Enable Kaggle persistence** to save checkpoints between sessions
5. **Monitor GPU memory** with `nvidia-smi` in another cell
6. **Use `--allow-cpu`** only for debugging, not actual training

---

## Kaggle-Specific Notes

### Path Differences
```python
# Local: data/processed/
# Kaggle: /kaggle/input/your-dataset/data/processed/

# Update in training scripts if dataset is external input
DATA_DIR = Path("/kaggle/input/deepsceneloc-data/data/processed")
```

### Saving Outputs
```python
# Kaggle auto-saves /kaggle/working/
# Move important files there
import shutil
shutil.copytree("models/checkpoints", "/kaggle/working/checkpoints")
shutil.copytree("results", "/kaggle/working/results")
```

### Time Limits
- Kaggle GPU: 30 hours/week (T4)
- Plan accordingly: All 3 models + ensemble ≈ 8-10 hours total
