# Preprocessing Methodology
## DeepSceneLoc — Image Preprocessing & Data Pipeline

**Author:** Anuj Kondawar (Preprocessing & Pipeline Lead)  
**Week 7 Deliverable** — Content notes for final documentation  
**Last Updated:** March 9, 2026

---

## 1. Overview

The preprocessing pipeline transforms raw image files from the Places365 dataset into
normalised, augmented tensors that are consumed by the PyTorch training loop.  The pipeline
is split into two layers:

| Layer | File | Purpose |
|-------|------|---------|
| **Transforms** | `src/preprocessing/transforms.py` | Declarative transform definitions and augmentation policy |
| **Pipeline** | `src/preprocessing/pipeline.py` | Dataset class, DataLoader factory, validation & benchmark helpers |

---

## 2. Resize & Crop Strategy

All models in this project (ResNet-50, EfficientNet-B0, ViT) use an **input resolution of 224 × 224**
to match their ImageNet pre-training configuration.

| Split | Resize | Crop |
|-------|--------|------|
| Train | 257 × 257 (×1.15) | `RandomCrop(224)` |
| Val / Test | 224 × 224 | — (no crop) |

**Why resize-before-crop during training?**  
Resizing to a slightly larger intermediate size before a random crop introduces spatial
variation (different sub-windows of the scene) without distorting the aspect ratio.  This
is the recommended practice from the *CS231n transfer-learning notes* reviewed in Week 2.

---

## 3. Augmentation Policy

Applied **only during training**.  Validation and test splits use a deterministic
resize-and-normalise transform to ensure reproducible metrics.

| # | Augmentation | Parameters | Rationale |
|---|-------------|------------|-----------|
| 1 | `Resize → RandomCrop` | 257 → 224 | Spatial variation |
| 2 | `RandomHorizontalFlip` | p = 0.5 | Mirror-image real-world scenes |
| 3 | `RandomRotation` | ±15° | Camera tilt, non-level horizon |
| 4 | `ColorJitter` | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1 | Lighting & weather variation |
| 5 | `RandomAffine` | translate=(0.1,0.1), scale=(0.9,1.1) | Perspective shift, zoom variation |
| 6 | `ToTensor` | — | PIL → float32 tensor in [0,1] |
| 7 | `Normalize` | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] | Align with ImageNet pre-training |

### Why these augmentations?

- **Horizontal flip** — Scenes classified as "Coastal" or "Urban" are visually symmetric;
  a mirrored image belongs to the same category.
- **Rotation** — Handheld or dashcam images often have slight tilts.
- **Color jitter** — Coastal images under overcast vs. sunny conditions have very different
  colour distributions; jitter forces the model to rely on structural cues.
- **Random affine** — Simulates minor perspective differences between photographers
  standing at similar spots.

---

## 4. Normalization

ImageNet mean and standard deviation are applied as the last pixel-level step:

```
mean = [0.485, 0.456, 0.406]   # R, G, B channel means
std  = [0.229, 0.224, 0.225]   # R, G, B channel std-devs
```

All three backbone models were pre-trained on ImageNet using these statistics.  Applying the
same normalization ensures the pre-trained feature representations remain valid during
fine-tuning.  After normalization, pixel values are distributed roughly in the range **[-2.1, 2.6]**.

---

## 5. Dataset Class — `DeepSceneLocDataset`

```python
from src.preprocessing.pipeline import DeepSceneLocDataset

dataset = DeepSceneLocDataset(
    root_dir="data/processed",
    transform=get_transforms("train"),
    split="train",           # "train" | "val" | "test"
)
```

**Directory contract expected:**

```
data/processed/
    train/
        Coastal/   *.jpg / *.png
        Forest/
        Mountain/
        Rural/
        Urban/
    val/
        ...
    test/
        ...
```

**Key methods:**

| Method | Returns |
|--------|---------|
| `__len__()` | Total number of samples in the split |
| `__getitem__(idx)` | `(image_tensor, label_int)` tuple |
| `get_class_counts()` | `Dict[class_name, count]` |

---

## 6. DataLoader Factory — `create_dataloaders`

```python
from src.preprocessing.pipeline import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir="data/processed",
    batch_size=32,
    num_workers=4,           # tune to your CPU core count
    prefetch_factor=2,       # batches pre-fetched per worker
    persistent_workers=True, # keep workers alive between epochs
)
```

### Performance Optimisations (Week 5)

| Setting | Value | Effect |
|---------|-------|--------|
| `pin_memory=True` | Always on | Faster host → GPU transfer (page-locked memory) |
| `persistent_workers=True` | When `num_workers > 0` | Avoids process-spawn overhead per epoch (~0.5–2 s/epoch saved) |
| `prefetch_factor=2` | When `num_workers > 0` | Each worker pre-fetches 2 batches ahead |
| `drop_last=True` | Train loader only | Keeps batch size constant; avoids last-batch shape surprises |

**Recommended `num_workers`:** `min(4, os.cpu_count())`.  On Google Colab T4 (2 vCPUs)
use `num_workers=2`.

---

## 7. Class Imbalance Handling — `get_class_weights`

The Places365 → 5-category mapping produces an imbalanced distribution (Urban and Forest
classes dominate).  The weighted cross-entropy loss re-balances training:

```python
from src.preprocessing.pipeline import get_class_weights

weights = get_class_weights(train_dataset)
criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
```

Weight formula:

```
w_c = N_total / (K × N_c)
```

where `N_total` = total samples, `K` = number of classes, `N_c` = samples in class `c`.

---

## 8. Preprocessing Validation — `validate_preprocessing`

Runs automated sanity checks on any data split:

```python
from src.preprocessing.pipeline import validate_preprocessing

ok = validate_preprocessing(data_dir="data/processed", split="test")
```

**Checks performed:**

| Check | Expected Value |
|-------|---------------|
| Tensor shape | `(3, 224, 224)` |
| Tensor dtype | `torch.float32` |
| Finite values | No `NaN` or `Inf` |
| Normalised range | Roughly `[-4.0, +4.0]` |
| Label range | `[0, num_classes)` |

---

## 9. DataLoader Benchmark — `benchmark_dataloader`

Measures loading throughput to confirm the pipeline is not a training bottleneck:

```python
from src.preprocessing.pipeline import benchmark_dataloader

stats = benchmark_dataloader(
    data_dir="data/processed",
    batch_size=32,
    num_workers=4,
    num_batches=20,
)
# stats: {'images_per_second': ..., 'ms_per_batch': ...}
```

**Target throughput:** ≥ 500 images/s on a standard research workstation / Colab T4 to
avoid GPU starvation (A100/T4 training step ≈ 50 ms for a 32-image batch).

---

## 10. Key Preprocessing Insights

1. **Random crop beats centre crop** for scene classification.  In natural scene images,
   discriminating features (e.g., water horizon for Coastal, building density for Urban) are
   spread across the frame; a random crop forces the model to localise them robustly.

2. **Color jitter range matters.**  A `hue` shift > 0.2 was found in the literature
   (RepVGG / data-augmentation ablations) to degrade performance on scene datasets because
   it can turn a Green-Forest image towards Brown-Mountain.  Chosen value: `hue=0.1`.

3. **Normalization is mandatory for ImageNet transfer learning.**  Omitting it causes the
   initial feature norms to be ≈ 2× larger, which doubles the gradient magnitude in early
   fine-tuning epochs and destabilizes training.

4. **`drop_last=True` in the train loader** prevents variable-size last-batch issues that
   can corrupt GroupNorm / BatchNorm statistics at the end of an epoch.

5. **`persistent_workers`** had the largest single impact on per-epoch speed in the
   benchmark: eliminating process re-spawning saves ≈ 1–3 seconds per epoch.

---

## 11. File Reference

| File | Lines | Description |
|------|-------|-------------|
| `src/preprocessing/transforms.py` | ~240 | Transform builder functions, denormalize helper, augmentation policy dict |
| `src/preprocessing/pipeline.py` | ~400 | Dataset class, DataLoader factory, class-weight helper, validate & benchmark |
| `src/preprocessing/__init__.py` | ~50 | Public API exports |

---

*Prepared by Anuj Kondawar — Week 7 content contribution for the Semester 1 Final Report.*
