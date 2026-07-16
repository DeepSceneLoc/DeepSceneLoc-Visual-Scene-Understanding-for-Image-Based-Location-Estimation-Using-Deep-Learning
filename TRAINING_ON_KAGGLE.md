# Training DeepSceneLoc Models on Kaggle

## Quick Start

### Step 1: Setup Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Create new notebook: "DeepSceneLoc Training"
3. Settings:
   - **GPU:** T4 x2 (or P100)
   - **Internet:** ON
   - **Persistence:** ON (to save checkpoints)

### Step 2: Clone Repository

```python
# Cell 1: Clone your repo
!git clone -b dev-krishan https://github.com/DeepSceneLoc/DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning.git
%cd DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning
```

### Step 3: Verify Environment

```python
# Cell 2: Check GPU and Python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Verify structure
!ls -la scripts/training/
```

### Step 4: Install Dependencies

```python
# Cell 3: Install if needed
!pip install timm torchmetrics scikit-learn pyyaml tqdm --quiet
```

### Step 5: Add Dataset

**Dataset:** [Places2 MIT Dataset](https://www.kaggle.com/datasets/nickj26/places2-mit-dataset)

1. Click **"+ Add data"** in your notebook
2. Search for: **"nickj26/places2-mit-dataset"**
3. Click **"Add"**

Or use the direct link: https://www.kaggle.com/datasets/nickj26/places2-mit-dataset

```python
# Cell 4: Check data (dataset will be in /kaggle/input/)
!ls -la /kaggle/input/places2-mit-dataset/

# Your code will access it as:
# /kaggle/input/places2-mit-dataset/
```

**Note:** You'll need to map the Places2 categories to our 5 categories (Coastal, Forest, Mountain, Rural, Urban) or use a pre-filtered subset.

### Step 6: Prepare Data Structure

The Places2 dataset needs to be organized into our 5 categories:

```python
# Cell 5: Organize data
# If you have a pre-filtered dataset with only your 5 categories:
!ls -la /kaggle/input/places2-mit-dataset/data_256/

# Or if you need to filter from full Places2:
# Run your data preparation script
!python scripts/prepare_places2_data.py \
    --input /kaggle/input/places2-mit-dataset/ \
    --output data/processed

# Verify structure
!ls -la data/processed/train/
# Should show: Coastal/ Forest/ Mountain/ Rural/ Urban/
```

### Step 7: Train Models

**Option A: Train All (Recommended)**

```python
# Cell 5: Train EfficientNet-B0
!python scripts/training/run_training_efficientnet_b0.py \
    --batch 128 \
    --epochs 45 \
    --full-finetune \
    --swa

# Cell 6: Train ViT-B/16
!python scripts/training/run_training_vit_b16.py \
    --batch 96 \
    --epochs 45 \
    --full-finetune \
    --swa

# Cell 7: Train ResNet50
!python scripts/training/run_training_resnet50.py \
    --batch 128 \
    --epochs 40 \
    --full-finetune
```

**Option B: Test One First**

```python
# Quick test with EfficientNet (fastest)
!python scripts/training/run_training_efficientnet_b0.py \
    --batch 64 \
    --epochs 5 \
    --full-finetune
```

### Step 7: Verify Training

```python
# Cell 8: Check checkpoints
!ls -la models/checkpoints/efficientnet/
!ls -la models/checkpoints/vit/
!ls -la models/checkpoints/resnet/

# Should see best_model.pth files
```

### Step 8: Run Ensemble Evaluation

```python
# Cell 9: Test ensemble
!python scripts/training/run_ensemble_eval.py \
    --ckpts models/checkpoints/efficientnet/EfficientNet-B0_best.pth \
            models/checkpoints/vit/ViT-B_16_best.pth \
            models/checkpoints/resnet/best_model.pth \
    --data data/processed \
    --weight-by-val-acc \
    --calibrate
```

### Step 9: Download Checkpoints

```python
# Cell 10: Create archive
!zip -r trained_models.zip models/checkpoints/

# Download in Kaggle (right-click → Download)
# Or copy to output
import shutil
shutil.copy('trained_models.zip', '/kaggle/working/')
print("✅ Models saved to Kaggle output")
```

## Expected Timeline (T4 x2)

| Model | Epochs | Batch | Time |
|-------|--------|-------|------|
| EfficientNet-B0 + SWA | 45 | 128 | 2-3h |
| ViT-B/16 + SWA | 45 | 96 | 3-4h |
| ResNet50 | 40 | 128 | 2-3h |
| **Total** | - | - | **~8-10h** |

## Expected Results

| Model | Val Acc | Test Acc |
|-------|---------|----------|
| EfficientNet-B0 | 87% | 86% |
| ViT-B/16 | 88% | 87% |
| ResNet50 | 85% | 84% |
| **Ensemble** | **89%** | **88%** |

## After Training

### 1. Download Checkpoints

Download `trained_models.zip` from Kaggle output

### 2. Extract Locally

```bash
# Unzip in your project root
unzip trained_models.zip
```

You should have:
```
models/checkpoints/
├── efficientnet/
│   └── EfficientNet-B0_best.pth
├── vit/
│   └── ViT-B_16_best.pth
└── resnet/
    └── best_model.pth
```

### 3. Update Backend

Stop the mock backend:
```bash
# Kill mock server (port 5000)
taskkill /F /PID <PID>
```

Start real backend:
```bash
python webapp/backend_api.py
```

It will automatically load your trained models!

### 4. Test Frontend

```bash
# Start frontend
cd frontend
npm run dev

# Open http://localhost:3000
# Upload image → See REAL predictions!
```

## Troubleshooting on Kaggle

### Out of Memory

Reduce batch sizes:
```python
# EfficientNet
!python scripts/training/run_training_efficientnet_b0.py --batch 64 ...

# ViT
!python scripts/training/run_training_vit_b16.py --batch 48 ...
```

### Data Not Found

Check paths:
```python
!ls -la /kaggle/input/
!ls -la data/
```

Update data path in training scripts if needed.

### Import Errors

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
```

### Session Timeout

Save checkpoints frequently:
- Kaggle auto-saves to `/kaggle/working/`
- Enable persistence in settings
- Can resume training from checkpoint

## Alternative: Use Pre-uploaded Dataset

If your dataset is already on Kaggle:

```python
# Link to Kaggle dataset
import os
os.symlink('/kaggle/input/your-dataset/', 'data/processed')
```

## Full Notebook Template

I've created a complete Jupyter notebook in:
```
notebooks/kaggle_train_t4x2.ipynb
```

You can:
1. Upload this to Kaggle
2. Run all cells
3. Download trained models

## Commands Summary

```bash
# Clone
git clone -b dev-krishan https://github.com/DeepSceneLoc/DeepSceneLoc...

# Train EfficientNet
python scripts/training/run_training_efficientnet_b0.py --batch 128 --epochs 45 --full-finetune --swa

# Train ViT
python scripts/training/run_training_vit_b16.py --batch 96 --epochs 45 --full-finetune --swa

# Train ResNet
python scripts/training/run_training_resnet50.py --batch 128 --epochs 40 --full-finetune

# Ensemble
python scripts/training/run_ensemble_eval.py --ckpts [...] --weight-by-val-acc --calibrate

# Download
zip -r trained_models.zip models/checkpoints/
```

## Next Steps After Training

1. ✅ Download `trained_models.zip`
2. ✅ Extract to `models/checkpoints/`
3. ✅ Run real backend: `python webapp/backend_api.py`
4. ✅ Test frontend with real predictions
5. ✅ Deploy to production

---

**Ready to train!** 🚀

Go to Kaggle, create a notebook, and follow Step 1-9 above.

When you have the trained models, come back and I'll help you integrate them!
