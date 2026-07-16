# Kaggle Training - Quick Start

## 🚀 In 5 Steps

### 1. Create Kaggle Notebook
- Go to: https://www.kaggle.com/code
- Click: **"New Notebook"**
- Settings:
  - GPU: **T4 x2**
  - Internet: **ON**
  - Persistence: **ON**

### 2. Add Dataset
Click **"+ Add data"** → Search: **"nickj26/places2-mit-dataset"**

Or direct link: https://www.kaggle.com/datasets/nickj26/places2-mit-dataset

### 3. Clone Repository

```python
# Cell 1
!git clone -b dev-krishan https://github.com/DeepSceneLoc/DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning.git
%cd DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning

# Verify
!ls scripts/training/
```

### 4. Prepare Dataset

```python
# Cell 2 - Check dataset
!ls /kaggle/input/places2-mit-dataset/

# If you have pre-filtered data with 5 categories:
DATA = "/kaggle/input/places2-mit-dataset/filtered"

# Otherwise, you'll need to filter Places2 categories
# See DATASET_SETUP.md for filtering code
```

### 5. Train Models

```python
# Cell 3 - EfficientNet (~2-3 hours)
!python scripts/training/run_training_efficientnet_b0.py \
    --data {DATA} \
    --batch 128 \
    --epochs 45 \
    --full-finetune \
    --swa

# Cell 4 - ViT (~3-4 hours)
!python scripts/training/run_training_vit_b16.py \
    --data {DATA} \
    --batch 96 \
    --epochs 45 \
    --full-finetune \
    --swa

# Cell 5 - ResNet (~2-3 hours)
!python scripts/training/run_training_resnet50.py \
    --data {DATA} \
    --batch 128 \
    --epochs 40 \
    --full-finetune

# Cell 6 - Download checkpoints
!zip -r trained_models.zip models/checkpoints/
import shutil
shutil.copy('trained_models.zip', '/kaggle/working/')
print("✓ Download from Kaggle Output tab")
```

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Setup (Steps 1-4) | 5 minutes |
| EfficientNet | 2-3 hours |
| ViT | 3-4 hours |
| ResNet | 2-3 hours |
| **Total** | **~8-10 hours** |

## 📊 Expected Results

| Model | Test Accuracy |
|-------|---------------|
| EfficientNet-B0 | 86% |
| ViT-B/16 | 87% |
| ResNet50 | 84% |
| **Ensemble** | **88%** |

## 📁 What You Get

After training, download `trained_models.zip` containing:

```
models/checkpoints/
├── efficientnet/
│   └── EfficientNet-B0_best.pth
├── vit/
│   └── ViT-B_16_best.pth
└── resnet/
    └── best_model.pth
```

## 🔌 Use with Frontend

After downloading:

```bash
# 1. Extract locally
unzip trained_models.zip

# 2. Stop mock backend
# Kill process on port 5000

# 3. Start real backend
python webapp/backend_api.py

# 4. Start frontend
cd frontend
npm run dev

# 5. Open http://localhost:3000
# Upload image → See REAL predictions!
```

## 📚 Full Guides

- **Complete training guide:** `TRAINING_ON_KAGGLE.md`
- **Dataset setup:** `DATASET_SETUP.md`
- **Command reference:** `docs/guides/QUICK_START.md`
- **Jupyter notebook:** `notebooks/kaggle_train_t4x2.ipynb`

## ⚠️ Important

**Dataset:** https://www.kaggle.com/datasets/nickj26/places2-mit-dataset

**Branch:** `dev-krishan` (not `Second_phase`)

**Categories:** Coastal, Forest, Mountain, Rural, Urban

---

**Ready to train!** Go to Kaggle and follow steps 1-5 above. 🚀
