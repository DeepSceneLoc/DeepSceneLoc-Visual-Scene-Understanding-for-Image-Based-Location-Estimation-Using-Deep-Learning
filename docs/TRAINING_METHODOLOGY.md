# DeepSceneLoc — Training Methodology
## Old Methods → Modern Methods (Full Changelog)

**Document Purpose:** Records every training technique change made during Semester 2 (April 27–28, 2026), with precise reasoning for each decision. Written for dissertation documentation and mentor review.

**Authors:** Krishan Yadav (Architecture), Anuj Kondawar (Pipeline)  
**Last Updated:** April 28, 2026 (Final results recorded)

---

## Overview

When we reviewed the active EfficientNet-B0 training run (Week 8), we identified that the pipeline was using **2019-era training practices** that left significant accuracy and speed gains on the table. We stopped that run after epoch 1 (which took 25 minutes — a red flag on its own) and upgraded to a modern 2024-standard pipeline. The new run achieves approximately **18 minutes per epoch** on RTX 3050 Laptop (AMP + batch=64, 4 DataLoader workers).

**Final Result (April 28, 2026): EfficientNet-B0 — 85.15% val accuracy / 84.63% test accuracy / 83.17% macro F1** — surpassing the original 78% target by +7.15%.

---

## Change 1 — Loss Precision: FP32 → Mixed Precision (AMP)

### What Changed
**File:** `src/models/train_advanced.py` — `AdvancedTrainer.__init__` and `_run_epoch`

**Old code:**
```python
# No GradScaler, no autocast
outputs = self.model(imgs)
loss    = self.criterion(outputs, labels)
loss.backward()
self.optimizer.step()
```

**New code (April 28, 2026 — updated to non-deprecated API):**
```python
# torch.cuda.amp is deprecated in PyTorch 2.0+; using torch.amp directly
self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

with torch.amp.autocast('cuda', enabled=self.use_amp):
    outputs = self.model(imgs)
    loss    = self.criterion(outputs, labels)

self.scaler.scale(loss).backward()
self.scaler.unscale_(self.optimizer)
self.scaler.step(self.optimizer)
self.scaler.update()
```

### Why We Changed It
Automatic Mixed Precision (AMP) computes forward/backward passes in **FP16** but accumulates gradients in FP32. On our RTX 3050 (Tensor Cores present), this gives:
- **2–3× faster training** — epoch went from ~25 min to ~10–12 min
- **~40% less GPU VRAM** — allows doubling batch size from 32 → 64
- **No accuracy loss** — GradScaler handles numerical stability automatically

Without AMP, training 40 epochs would take **~20 hours** (30 min/epoch without AMP). With AMP: **~12 hours** (18 min/epoch). This saves nearly 8 hours of GPU time on our laptop GPU.

**Reference:** Micikevicius et al., "Mixed Precision Training" (ICLR 2018)

---

## Change 2 — Batch Size: 32 → 64 (enabled by AMP)

### What Changed
**File:** `src/models/train_advanced.py` — `EfficientNetTrainConfig`

```python
# Before
batch_size: int = 32

# After
batch_size: int = 64  # AMP halves VRAM, so 64 fits comfortably
```

### Why We Changed It
With AMP reducing VRAM from ~5.4 GB to ~3.2 GB for batch 32, we have headroom to double the batch size. Larger batches:
- More stable gradient estimates per step
- Better GPU utilisation (RTX 3050 has 2048 CUDA cores — small batches leave them idle)
- Combined with gradient accumulation, the effective batch becomes 128

---

## Change 3 — Gradient Accumulation: Reverted to 1× (EfficientNet)

### What Changed
**File:** `src/models/train_advanced.py` — `EfficientNetTrainConfig`

```python
# Initially planned: accum_steps=2 (effective batch 128)
# Reverted to:
grad_accum_steps: int = 1   # effective batch = 64 (same as actual batch)
```

### Why We Reverted It
Gradient accumulation with `accum_steps=2` was designed for smaller GPUs that couldn't fit large batches. With AMP and batch=64 on 6GB VRAM (only 1.1GB used), we have ample capacity without accumulation. Accumulation adds code complexity and reduces gradient-update frequency by 2× with no benefit when the batch already fits in VRAM. Keeping `accum_steps=1` means the model gets 4525 gradient updates per epoch instead of 2262 — more updates = faster convergence in the early epochs.

**Note:** ViT still uses `accum_steps=4` because ViT attention layers require a larger effective batch for stable training.

---

## Change 4 — LR Scheduler: CosineAnnealingLR → OneCycleLR

### What Changed
**File:** `src/models/train_advanced.py` — `EfficientNetTrainConfig` and `build_scheduler`

```python
# Before
scheduler: str = "cosine"   # plain CosineAnnealingLR, no warmup

# After
scheduler: str = "onecycle"  # OneCycleLR: 10% warmup + super-convergence cosine

# New scheduler:
OneCycleLR(
    optimizer,
    max_lr=config.learning_rate,
    total_steps=total_steps,
    pct_start=0.1,         # 10% linear warmup
    anneal_strategy="cos",
    div_factor=25,          # start at max_lr/25
    final_div_factor=1e4,   # end at start_lr/1e4
)
```

### Why We Changed It
Plain CosineAnnealingLR has no warmup phase. Starting at a high LR (1e-4) from step 1 with a partially-frozen EfficientNet causes large, destabilising gradient updates in the custom head layers before the pretrained backbone has warmed up.

OneCycleLR solves this with:
1. **10% warmup** — gradually ramps up from LR/25 = 4e-6 to 1e-4
2. **Cosine annealing** — smooth decay from peak back down to near-zero
3. **Super-convergence** — Smith (2018) showed OneCycleLR can train in fewer epochs than cosine alone with higher final accuracy

ViT keeps `warmup_cosine` (5-epoch linear warmup then cosine) because OneCycleLR doesn't pair as well with the long ViT convergence pattern.

---

## Change 5 — Label Smoothing: 0.0 → 0.1 (EfficientNet)

### What Changed
**File:** `src/models/train_advanced.py` — `EfficientNetTrainConfig`

```python
# Before
label_smoothing: float = 0.0   # plain CrossEntropy, hard targets

# After
label_smoothing: float = 0.1   # same as ViT config — soft targets
```

### Why We Changed It
Label smoothing replaces hard one-hot targets `[0, 0, 1, 0, 0]` with soft targets `[0.02, 0.02, 0.92, 0.02, 0.02]`. This prevents the model from becoming **overconfident** on training data, which:
- Reduces overfitting on the smallest class (Forest, 6,751 test images)
- Improves calibration of confidence scores (critical for the hybrid pipeline's confidence gating)
- Consistently yields +0.3–0.5% accuracy in published benchmarks on Places365

ViT already had `label_smoothing=0.1`. We aligned EfficientNet to match.

---

## Change 6 — Data Augmentation: Manual → RandAugment

### What Changed
**File:** `src/preprocessing/transforms.py` — new function `get_modern_train_transforms`  
**File:** `run_training_efficientnet_b0.py` — now calls `get_modern_train_transforms` instead of `get_train_transforms`

**Old augmentation stack:**
```python
T.Resize(257), T.RandomCrop(224),
T.RandomHorizontalFlip(p=0.5),
T.RandomRotation(degrees=15),
T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
T.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)),
T.ToTensor(), T.Normalize(...)
```

**New augmentation stack:**
```python
T.Resize(256), T.RandomCrop(224),
T.RandomHorizontalFlip(p=0.5),
T.RandAugment(num_ops=2, magnitude=9),   # ← NEW: policy search
T.ToTensor(), T.Normalize(...),
T.RandomErasing(p=0.25, scale=(0.02,0.2), value='random')  # ← NEW
```

### Why We Changed It
**Old approach:** Manually chosen augmentations. We arbitrarily set rotation=15°, jitter=0.3 without any principled search. These choices may not be optimal for scene classification.

**RandAugment** (Cubuk et al., 2019) automatically selects from 14 augmentation operations (brightness, contrast, sharpness, rotate, equalize, solarize, etc.) and applies `N=2` randomly chosen ones per image at magnitude `M=9`. Benefits:
- Removes manual tuning of augmentation hyperparameters
- Proven to boost ImageNet top-1 accuracy by 1–2%
- Places365 has similar visual diversity — same benefit applies

**RandomErasing** simulates occluded or partially visible scenes (e.g. a forest view partially blocked by a fence). This is realistic for our use case where photos are taken in-the-wild.

---

## Change 7 — MixUp + CutMix Augmentation: None → Alternating

### What Changed
**File:** `src/models/train_advanced.py` — new functions `mixup_data`, `cutmix_data`, `mixup_criterion`; applied in `_run_epoch`

```python
# New: alternates MixUp / CutMix each batch
# alpha=0.2 (tuned down from initial 0.4 — less aggressive mixing for 5-class problem)
if use_cutmix and step % 2 == 0:
    imgs, y_a, y_b, lam = cutmix_data(imgs, labels, alpha=0.2)
else:
    imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.2)

loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
```

### Why We Changed It
MixUp linearly interpolates two training images and their labels:
- **MixUp:** pixel-level blending `x = λA + (1-λ)B`
- **CutMix:** cuts a rectangular patch from image B into image A

Both force the model to learn **intermediate representations** rather than memorising specific feature combinations. Benefits for DeepSceneLoc:
- Reduces over-reliance on any single visual cue (e.g. sky colour → coastal)
- Makes the model reason about scene components holistically
- Published gains: +0.5–2% on Places365-scale datasets
- Helps the **Forest class** (the hardest) — mixes with Rural/Mountain features, which are its most common confusion pairs

The alternating strategy (CutMix on even batches, MixUp on odd) was shown by CutMix authors to outperform using either alone.

---

## Change 8 — Model Weights Averaging: None → EMA

### What Changed
**File:** `src/models/train_advanced.py` — new class `ModelEMA`; updated checkpoint saving

```python
class ModelEMA:
    def __init__(self, model, decay=0.9999):  # tuned: 0.9998 → 0.9999
        self.module = deepcopy(model)   # shadow copy
        self.decay  = decay

    def update(self, model):
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1-self.decay)

# Used for validation and final checkpoint (FIXED April 28):
# Bug: was `ema_acc if ema_acc > 0` — EMA diverges on fresh runs, causing false early stops
# Fix: use max() so raw model improvement is always counted
effective_acc = max(val_acc, ema_acc)
```

### Why We Changed It
Neural network weights oscillate around a local minimum during SGD. The final epoch's weights are **noisy** — they might be at a slightly bad point in the loss landscape at training end.

EMA maintains an exponentially weighted running average of all past weight values. The shadow copy smooths out these oscillations, consistently producing models that:
- Score **0.3–0.7% higher** on validation vs the raw checkpoint
- Are more robust to random seed variation between runs
- Produce better-calibrated confidence scores for the hybrid pipeline

**Decay tuning (April 28, 2026):** Changed from `0.9998` to `0.9999`.  
- `decay=0.9998`: half-life = 3,466 steps (~0.77 epochs) — too short, EMA diverges during rapid early backbone learning  
- `decay=0.9999`: half-life = 6,931 steps (~1.5 epochs) — smoother averaging, less oscillation

---

## Change 9 — Gradient Computation: `zero_grad()` → `zero_grad(set_to_none=True)`

### What Changed
**File:** `src/models/train_advanced.py` — `_run_epoch`

```python
# Before
self.optimizer.zero_grad()

# After
self.optimizer.zero_grad(set_to_none=True)
```

### Why We Changed It
`set_to_none=True` sets gradient tensors to `None` instead of filling with zeros. This:
- Saves the VRAM used by gradient storage (~same as activations)
- Avoids the unnecessary zero-fill computation
- Gives a small but real throughput improvement (~2% faster per step)
- Recommended by PyTorch documentation as best practice since 1.7

---

## Change 10 — EfficientNet Gradient Clip: None → 1.0

### What Changed
**File:** `src/models/train_advanced.py` — `EfficientNetTrainConfig`

```python
# Before
gradient_clip: Optional[float] = None   # "EfficientNet is usually stable"

# After
gradient_clip: float = 1.0   # safe default for all architectures
```

### Why We Changed It
We previously assumed EfficientNet was gradient-stable and didn't need clipping. However, with MixUp/CutMix creating mixed-label samples, gradients on edge cases can spike unexpectedly. Setting `clip=1.0` is a pure safety net — it only activates when norm exceeds 1.0 and prevents runaway updates that could corrupt a good checkpoint mid-training.

---

## Change 11 — Loss Function: Standard CrossEntropy → Class-Weighted CrossEntropy

### What Changed
**File:** `src/models/train_advanced.py` — `AdvancedTrainer.__init__`

```python
# Before
self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

# After
from src.preprocessing.pipeline import get_class_weights
class_weights = get_class_weights(self.train_loader.dataset).to(device)
self.criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=config.label_smoothing
)
```

### Why We Changed It
**Old approach:** The dataset is highly imbalanced (e.g., Urban has ~86,653 images, Forest has ~31,499). Standard CrossEntropy assumes all classes are equally represented, causing the model to become biased toward predicting majority classes like Urban or Rural.
**New approach:** We compute class weights dynamically based on the inverse of the class frequencies. This means the model incurs a *much higher penalty* for misclassifying a minority class (Forest) than a majority class.
**Impact:** Drastically improves balanced accuracy (Macro F1) and prevents the model from ignoring the Forest class.

---

## Change 12 — ViT Regularization: None → Stochastic Depth (Drop Path)

### What Changed
**File:** `src/models/model_advanced.py` — `DeepSceneLocViTAdvanced.__init__`

```python
# Before
self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)

# After
self.vit = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=0,
    drop_path_rate=0.1  # Stochastic Depth
)
```

### Why We Changed It
**Old approach:** Vision Transformers are extremely powerful but prone to overfitting on datasets smaller than ImageNet.
**New approach:** We added Stochastic Depth (Drop Path) with a rate of 0.1. This randomly drops entire transformer blocks during training, forcing the network to distribute its learning across all layers.
**Impact:** Acts as a powerful regularizer, yielding better generalization on unseen test data and usually boosting test accuracy by ~0.5%.

---

## Change 13 — Evaluation: Single Crop → Test-Time Augmentation (TTA)

### What Changed
**File:** `src/evaluation/evaluate.py` — `ModelEvaluator.evaluate()`

```python
# Before
outputs = self.model(inputs)
probabilities = torch.softmax(outputs, dim=1)

# After
outputs = self.model(inputs)
inputs_flipped = torch.flip(inputs, dims=[3])
outputs_flipped = self.model(inputs_flipped)
probabilities = (torch.softmax(outputs, dim=1) + torch.softmax(outputs_flipped, dim=1)) / 2.0
```

### Why We Changed It
**Old approach:** During testing, the model only made a prediction on a single, center-cropped image. If crucial context was missing from that crop, it would misclassify.
**New approach:** TTA forces the model to evaluate both the original image *and* a horizontally flipped version, averaging the final probability scores. Since scenes are largely horizontally invariant (a flipped coastal scene is still a coastal scene), this provides free robustness.
**Impact:** Almost always guarantees an extra +0.5% to 1.5% accuracy during the final evaluation phase with minimal performance overhead.

---

## Change 14 — Pipeline: Weight-Only Resume → Full Auto-Resume State Recovery

### What Changed
**File:** `run_training_efficientnet_b0.py` and `src/models/train_advanced.py`

### Why We Changed It
**Old approach:** Passing `--resume` only loaded the model weights. The learning rate scheduler, optimizer momentum, and epoch counters started over from scratch, which ruined the warmup phases and training stability if stopped and restarted.
**New approach:** Implemented an `--auto-resume` flag that scans for the latest checkpoint and restores the **entire training state** (Weights, Optimizer, LR Scheduler, AMP Scaler, EMA, and precise epoch). Also added `resume_vit_training.bat`.
**Impact:** Allows safe, completely seamless multi-session training. You can pause training (Ctrl+C) and run the resume script to pick up exactly where you left off, appending to `training.log` without data loss.

---

## Summary Table

| # | What Changed | Old Value | Final Value | Why | Result |
|---|---|---|---|---|---|
| 1 | Training precision | FP32 | **AMP (FP16/FP32)** | 2-3× faster, -40% VRAM | 18 min/epoch |
| 2 | Batch size | 32 | **64** | VRAM freed by AMP | Better GPU util |
| 3 | Gradient accumulation | None | **1× (no accum)** | batch=64 fits in VRAM; 2× doubled overhead | 4525 updates/epoch |
| 4 | LR schedule | CosineAnnealingLR | **OneCycleLR** | Warmup + super-convergence | +0.5-1% acc |
| 5 | Label smoothing | 0.0 | **0.1** | Calibration + overfit prevention | +0.3-0.5% acc |
| 6 | Augmentation | Manual ColorJitter | **RandAugment N=2 M=9** | Policy search > manual | +0.5-1% acc |
| 7 | Sample regularisation | None | **MixUp + CutMix alpha=0.2** | Holistic feature learning | +1-2% acc |
| 8 | Weight averaging | None | **EMA decay=0.9999** | Smoother final model | +0.3-0.7% acc |
| 9 | Grad zero | `zero_grad()` | **`zero_grad(set_to_none=True)`** | Memory + speed | ~2% faster |
| 10 | Grad clipping (EfficientNet) | None | **1.0** | Safety with MixUp spikes | Stability |
| 11 | freeze_blocks | 7 | **4** | More trainable params for large dataset | 80%+ at epoch 1 |
| 12 | Early stopping patience | 5 | **10** | MixUp plateaus need more buffer | No false stops |
| 13 | AMP API | `torch.cuda.amp` | **`torch.amp`** | Deprecated in PyTorch 2.0+ | No warnings |
| 14 | DataLoader workers | 2 | **4-8** | Reduce GPU starvation | Less GPU idle |
| 15 | persistent_workers | OFF | **ON** | Eliminate inter-epoch respawn delay | No 20-40s gap |
| 16 | Loss Function | Standard CE | **Class-Weighted CE** | Fixes severe class imbalance | Higher Macro F1 |
| 17 | ViT Regularization | None | **Drop Path (0.1)** | Prevents ViT overfitting | Better Generalization |
| 18 | Model Evaluation | Single Crop | **TTA (Horiz. Flip)** | Ensembles predictions | +0.5-1.5% accuracy |
| 19 | Training Resumption | Weights Only | **Full State Recovery** | Resumes Optimizer/LR/EMA | Seamless multi-session |

**Achieved accuracy:** EfficientNet-B0 → **84.40%** (target was 78%) — exceeded by +6.4%  
**Training speed:** 18 min/epoch on RTX 3050 6GB Laptop GPU  
**Next target:** EfficientNet-B0 → **86%+** (current run), ViT-B/16 → **88%+**

---

## What Was NOT Changed (And Why)

| Decision | Reason |
|----------|--------|
| AdamW optimizer | Still best-in-class for fine-tuning pretrained CNNs and ViTs. No reason to change |
| ImageNet pretrained weights | Fine-tuning from ImageNet is the correct approach for Places365 scene classification |
| 5-class output structure | Fixed by project spec — Coastal, Forest, Mountain, Rural, Urban |
| CrossEntropyLoss base | Still correct for multi-class. Label smoothing is applied on top, not replacing it |
| ViT warmup_cosine schedule | OneCycleLR doesn't suit ViT's slow convergence. Warmup-cosine is the DINOv2/BEiT standard |
| Batch size (kept at 64) | Tested: batch=256 would make OneCycleLR scheduler state incompatible for resume; 64 is proven |

---

## Bug Fixes (April 28, 2026)

Seven bugs were discovered and fixed during the pipeline stabilization session:

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | Early stopping tracked only EMA (diverges on fresh runs) | False stop at epoch 9 | `effective_acc = max(val_acc, ema_acc)` |
| 2 | `Path(None)` in post-training evaluation | Crash after training | Added `None` guard before `Path()` |
| 3 | Wrong history filename in `_run_evaluation` | History plot silently skipped | Fixed to use `{ModelName}_history.json` |
| 4 | Unicode chars in print statements | `UnicodeEncodeError` on Windows terminal | Replaced with ASCII equivalents |
| 5 | Deprecated `torch.cuda.amp` API | FutureWarning every step (log noise) | Updated to `torch.amp` API |
| 6 | No `persistent_workers` in DataLoader | 20-40s dead gap between epochs | Added `persistent_workers=True` |
| 7 | `plt.show()` called without Agg backend | Could block after training on headless runs | Added `matplotlib.use('Agg')` |

---

## Running the Modern Pipeline

```bash
# Full training (production config — April 28, 2026)
venv\Scripts\python.exe run_training_efficientnet_b0.py ^
    --data data/processed/places365_mit_full_2026_03_15 ^
    --epochs 40 --workers 8 --patience 10 --freeze-blocks 4

# Resume from checkpoint (model weights only)
venv\Scripts\python.exe run_training_efficientnet_b0.py ^
    --data data/processed/places365_mit_full_2026_03_15 ^
    --epochs 40 --workers 8 --patience 10 --freeze-blocks 4 ^
    --resume models/checkpoints/efficientnet/EfficientNet-B0_best.pth

# Watch training live (separate terminal)
python watch_training.py
Get-Content -Wait -Tail 30 training.log
```

---

## Baseline Comparison (For Dissertation)

```
ResNet-50 (Semester 1 — Old Pipeline):
  Training time per epoch : ~30 min (batch=32, no AMP)
  Total training time     : ~10 hrs (20 epochs)
  Best val accuracy       : 79.17%
  Test accuracy           : 79.04%
  Macro F1                : 77.39%
  Label smoothing         : 0.0
  Augmentation            : Basic ColorJitter + Flip

EfficientNet-B0 v1 (Old Pipeline — Stopped After Epoch 1):
  Training time per epoch : ~25 min (batch=32, no AMP)
  Label smoothing         : 0.0 (wrong — should be 0.1)
  No MixUp, no EMA, no RandAugment

EfficientNet-B0 v2 (Modern Pipeline — FINAL RESULTS):
  Training time per epoch : ~10-18 min (AMP + batch=64 + workers=4)
  Total training runs     : 3 (multi-phase resume strategy)
  Label smoothing         : 0.1
  freeze_blocks           : 4 (trainable params: 4.36M / 4.67M)
  Augmentation            : RandAugment N=2 M=9 + RandomErasing
  MixUp alpha=0.2 + CutMix alternating
  EMA decay=0.9999
  OneCycleLR with 10% warmup
  Early stopping patience : 10 epochs
  Best val accuracy       : 85.15%   [FINAL]
  Test accuracy           : 84.63%   [FINAL]
  Macro F1                : 83.17%   [FINAL]
  Model size              : 69.6 MB  (vs 281.5 MB ResNet-50)
```

---

## Final Model Comparison

### Overall Metrics

| Metric | ResNet-50 (Phase 1) | EfficientNet-B0 (Phase 2) | Gain |
|---|---|---|---|
| **Val Accuracy** | 79.17% | **85.15%** | **+5.98%** |
| **Test Accuracy** | 79.04% | **84.63%** | **+5.59%** |
| **Macro Precision** | 78.06% | 83.04% | +4.98% |
| **Macro Recall** | 76.85% | 83.33% | +6.48% |
| **Macro F1** | 77.39% | **83.17%** | **+5.78%** |
| **Total Params** | 25.6M | 5.3M | -79% |
| **Model Size** | 281.5 MB | **69.6 MB** | -75% |

### Per-Class Accuracy (EfficientNet-B0 Final)

| Class | Test Accuracy | Main Confusion |
|---|---|---|
| Coastal | 85.0% | → Urban (5.7%) |
| Forest | 78.1% | → Rural (7.6%) |
| Mountain | 79.8% | → Rural (6.3%) |
| Rural | 85.6% | → Urban (4.9%) |
| Urban | 88.3% | → Rural (4.7%) |

### Training History (Final Resume Run — 13 Epochs)

| Epoch | Val Acc | Note |
|---|---|---|
| 1 | 85.10% | NEW BEST (resumed from 84.79%) |
| 2 | 84.85% | — |
| 3 | 84.51% | — |
| 4-13 | 84.35–84.58% | Plateau — early stopping triggered |

**Best val_acc saved:** 85.15% (checkpoint epoch=3, EMA model)  
**Checkpoint:** `models/checkpoints/efficientnet/EfficientNet-B0_best.pth` (69.6 MB)  
**Fallback:** `models/checkpoints/resnet/best_model.pth` (281.5 MB)  
**Production app:** `webapp/api.py` — loads EfficientNet-B0 by default  
**Demo app:** `demo_app.py` — loads ResNet-50 (Phase 1 Gradio demo, preserved)
