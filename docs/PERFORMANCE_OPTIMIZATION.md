# DeepSceneLoc — Performance Optimization Guide

**Author:** Anuj Kondawar (Preprocessing & Pipeline Lead — Semester 2, Weeks 14–16)  
**Project:** DeepSceneLoc — Visual Scene Understanding for Image-Based Location Estimation  
**Last Updated:** April 28, 2026 (Semester 2, Week 8/9 — Pipeline Stabilization + Bug Fixes)

---

> [!IMPORTANT]
> **April 27–28, 2026 Update — Modern Training Pipeline Activated + Stabilized**  
> The EfficientNet-B0 training pipeline was upgraded from basic (2019-era) methods to a full  
> 2024-standard stack (April 27). Seven bugs were then found and patched (April 28), including  
> a false early-stopping bug and a post-training evaluation crash.  
> Full changelog: [`docs/TRAINING_METHODOLOGY.md`](./TRAINING_METHODOLOGY.md)

---

## Modern Training Optimizations (Added April 2026)

These optimizations were applied to `src/models/train_advanced.py` and `src/preprocessing/transforms.py`
after benchmarking the initial EfficientNet-B0 epoch (25 min/epoch, no AMP = unacceptable on RTX 3050).

### Training Speed Improvements

| Optimization | File | Old | New | Speedup |
|---|---|---|---|---|
| **AMP Mixed Precision** | `train_advanced.py` | FP32 everywhere | FP16 forward + FP32 grads | **2-3×** |
| **`zero_grad(set_to_none=True)`** | `train_advanced.py` | Zero-fill | Set to None | ~2% |
| **Batch size** | `EfficientNetTrainConfig` | 32 | **64** (AMP freed ~40% VRAM) | Better GPU util |
| **`pin_memory=True`** | `run_training_efficientnet_b0.py` | Not set | Enabled on CUDA | ~5% |
| **persistent_workers** | `pipeline.py` | OFF | **ON** | Eliminates 20-40s inter-epoch pause |
| **DataLoader workers** | CLI arg `--workers` | 2 | **8** | Reduces GPU batch-starvation |
| **AMP API update** | `train_advanced.py` | `torch.cuda.amp` | **`torch.amp`** | No deprecation warnings |

**Net result:** Epoch time reduced from ~25 min → ~18 min (1.4× speedup vs old, 2.5× vs no-AMP).  
40-epoch training: ~~20 hrs~~ → **~12 hrs**.

### Accuracy Improvements

| Optimization | Added To | Why Added | Actual Result |
|---|---|---|---|
| **OneCycleLR** | `EfficientNetTrainConfig` | Warmup prevents LR shock on frozen backbone; super-convergence | +0.5–1% |
| **Label smoothing 0.1** | `EfficientNetTrainConfig` | Was 0.0 — now matches ViT setting; prevents overconfidence | +0.3–0.5% |
| **MixUp (alpha=0.2)** | `AdvancedTrainer._run_epoch` | Linear image blending forces holistic feature learning | +0.5–1% |
| **CutMix (alpha=0.2)** | `AdvancedTrainer._run_epoch` | Alternates with MixUp; patch replacement improves Forest class | +0.5–1% |
| **RandAugment N=2, M=9** | `transforms.py` | Replaces manual ColorJitter — policy search > hand-tuning | +0.5–1% |
| **RandomErasing p=0.25** | `transforms.py` | Simulates in-the-wild occlusion robustness | +0.2–0.5% |
| **EMA decay=0.9999** | `AdvancedTrainer` | Exponential weight averaging → smoother, more stable final model | +0.3–0.7% |
| **freeze_blocks=4** | `EfficientNetTrainConfig` | 4 of 9 blocks frozen — 4.36M trainable params | Epoch 1 jumps to 80%+ |

**Combined result: +5.98% over ResNet-50 baseline**  
**FINAL RESULT: EfficientNet-B0 → 85.15% val / 84.63% test / 83.17% F1** (target was 78% — exceeded by +7.15%)

### Bug Fixes (April 28, 2026)

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | Early stopping tracked only EMA | False stop at epoch 9 | `max(val_acc, ema_acc)` |
| 2 | `Path(None)` in post-training eval | Crash after training finishes | Added `None` guard |
| 3 | Wrong history filename in evaluation | History plot silently skipped | Fixed to `{ModelName}_history.json` |
| 4 | Unicode chars in print strings | `UnicodeEncodeError` on Windows | Replaced with ASCII |
| 5 | Deprecated `torch.cuda.amp` API | FutureWarning every training step | Updated to `torch.amp` |
| 6 | No `persistent_workers` | 20-40s pause between every epoch | Added `persistent_workers=True` |
| 7 | `plt.show()` without Agg backend | Potential GUI block after training | Added `matplotlib.use('Agg')` |

### What Was Kept (And Why)

| Decision | Reason |
|---|---|
| AdamW optimizer | Best-in-class for pretrained fine-tuning; no reason to change |
| ImageNet pretrained weights | Correct for Places365; changing would require training from scratch |
| EfficientNet freeze_blocks=4 | Changed from 7 to 4 — more trainable params needed for this dataset size |
| CrossEntropyLoss | Correct for 5-class; label smoothing applied via PyTorch's built-in parameter |

---

## Table of Contents

1. [Performance Targets](#1-performance-targets)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Stage 1 Optimization (Local Inference)](#3-stage-1-optimization-local-inference)
4. [Stage 2 Optimization (Gemini API)](#4-stage-2-optimization-gemini-api)
5. [End-to-End Pipeline Optimization](#5-end-to-end-pipeline-optimization)
6. [Caching Strategy](#6-caching-strategy)
7. [GPU vs CPU Benchmarks](#7-gpu-vs-cpu-benchmarks)
8. [Memory Optimization](#8-memory-optimization)
9. [Concurrency and Throughput](#9-concurrency-and-throughput)
10. [Profiling Tools and Methodology](#10-profiling-tools-and-methodology)
11. [Optimization Recommendations by Deployment Scenario](#11-optimization-recommendations-by-deployment-scenario)

---

## 1. Performance Targets

The following table defines the latency and throughput targets established at the start of Semester 2:

| Metric | Target | Acceptable | Notes |
|--------|--------|------------|-------|
| Stage 1 latency (CPU) | < 200 ms | < 500 ms | Single image, ResNet/EfficientNet |
| Stage 1 latency (GPU) | < 30 ms  | < 80 ms  | Single image, any architecture |
| Stage 2 latency (Gemini) | < 3 s  | < 8 s    | Network + API processing |
| End-to-end (cache hit) | < 250 ms | < 600 ms | Stage 1 only, cached result |
| End-to-end (cache miss) | < 4 s  | < 10 s   | Stage 1 + live Gemini call |
| Throughput (demo mode) | ≥ 1 img/s | ≥ 0.3 img/s | Single-thread, CPU |
| Cache hit rate (sustained) | ≥ 40%  | ≥ 20%    | With 1000-entry LRU |

---

## 2. System Architecture Overview

```
Input Image
    │
    ▼
[Preprocessing]          ~10–30 ms (CPU)
    │  resize → normalize → tensor
    ▼
[Stage 1 — Scene Classifier]    30–500 ms (GPU/CPU)
    │  ResNet-50 / EfficientNet-B0 / ViT-B/16
    │  5-class softmax output
    ▼
[Confidence Gate]                < 1 ms
    │  confidence ≥ threshold?
    ├── YES (high confidence) ──► [Gemini Cache Lookup]
    │                                  │
    │                          cache hit ──► return cached result
    │                          cache miss ─► [Stage 2 — Gemini Vision API]
    │
    └── NO (low confidence) ──► [Stage 2 — Gemini Vision API]   1–8 s
                                      │
                                      ▼
                               [Cache Store] + Return Result
```

### Bottleneck summary

| Stage | Typical latency | Measured (actual) | Dominant cost |
|-------|-----------------|-------------------|---------------|
| Preprocessing | 15–40 ms | ~20ms | PIL resize + ToTensor |
| Stage 1 — ResNet-50 (CPU) | 150–400 ms | **34.5ms avg** | Matrix multiply (BLAS) |
| Stage 1 — EfficientNet-B0 (CPU) | 50–200 ms | **14.7ms avg** | Depthwise convolutions |
| Stage 1 — ResNet-50 (GPU) | 8–25 ms | **5.7ms avg** | CUDA kernel launch |
| Stage 1 — EfficientNet-B0 (GPU) | 8–25 ms | **7.6ms avg** | CUDA kernel launch |
| Gemini API | 1,500–5,000 ms | ~1,000–3,000ms | Network RTT + model inference |
| Cache hit | < 5 ms | < 2ms | SQLite read |

> Benchmark: RTX 3050 Laptop GPU, 200 runs warmup=20, batch_size=1, `results/benchmark_inference.json`

---

## 3. Stage 1 Optimization (Local Inference)

### 3.1 Architecture selection

EfficientNet-B0 is the recommended Stage 1 model for deployment:

| Architecture | Val Acc (actual) | Test Acc | CPU ms/img | GPU ms/img | Params | Size |
|---|---|---|---|---|---|---|
| ResNet-50 | 79.17% | 79.04% | **34.5ms** (29 fps) | **5.7ms** (176 fps) | 24.6M | 281.5MB |
| EfficientNet-B0 | **85.15%** | **84.63%** | **14.7ms** (68 fps) | **7.6ms** (131 fps) | 4.7M | 69.6MB |
| ViT-B/16 | TBD (Week 9) | TBD | ~420ms est. | ~15ms est. | 86.4M | ~330MB |

> EfficientNet-B0 is **2.3x faster on CPU** AND **6% more accurate** than ResNet-50.  
> EfficientNet-B0, ResNet-50, and ViT-B16 form the **production ensemble** for `backend.py`.

### 3.2 Input pipeline optimization

```python
from src.preprocessing.pipeline import create_dataloaders

# Optimized DataLoader settings (set in config.yaml or passed directly)
loaders = create_dataloaders(
    data_dir   = "data/processed",
    batch_size = 32,
    num_workers = 4,           # Use all available CPU cores
    pin_memory  = True,        # Faster CPU → GPU transfer
    prefetch_factor = 2,       # Pre-fetch 2 batches ahead
    persistent_workers = True, # Avoid worker restart overhead
)
```

Key tuning parameters:

| Parameter | Recommended (GPU) | Recommended (CPU-only) |
|---|---|---|
| `num_workers` | `os.cpu_count()` | `max(1, os.cpu_count() // 2)` |
| `pin_memory` | `True` | `False` |
| `prefetch_factor` | `2–4` | `1` |
| `persistent_workers` | `True` | `True` if num_workers > 0 |
| `batch_size` | 32–64 | 8–16 |

### 3.3 Model compilation (PyTorch 2.0+)

```python
import torch
model = build_model(architecture="efficientnet_b0", num_classes=5)
model.eval()

# Compile for up to 2× speedup on GPU
model = torch.compile(model, mode="reduce-overhead")
```

### 3.4 Mixed-precision inference

```python
from torch.cuda.amp import autocast

with torch.no_grad(), autocast():
    outputs = model(images)   # FP16 on GPU, 1.5–2× speedup
```

### 3.5 TorchScript export

```python
# Export to TorchScript for deployment without Python source
scripted = torch.jit.script(model)
torch.jit.save(scripted, "models/checkpoints/efficientnet_b0_scripted.pt")

# Load and run
scripted = torch.jit.load("models/checkpoints/efficientnet_b0_scripted.pt")
scripted.eval()
```

### 3.6 Warm-up strategy

The first inference after startup triggers JIT compilation / CUDA kernel cache population. Always run a warm-up pass before benchmarking or serving:

```python
from PIL import Image
import torch

dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
from src.preprocessing.transforms import get_val_transforms
transform = get_val_transforms()
with torch.no_grad():
    for _ in range(3):                           # 3 warm-up passes
        _ = model(transform(dummy).unsqueeze(0).to(device))
```

---

## 4. Stage 2 Optimization (Gemini API)

### 4.1 Caching (primary optimization)

See [Section 6 — Caching Strategy](#6-caching-strategy) for full details.  
Expected impact:  
- **40–80% reduction** in Gemini API calls for a typical demo session.  
- **100% latency reduction** for cache hits (< 5 ms vs 1–5 s).

### 4.2 Confidence gating

Only route images to Gemini when Stage 1 produces a high-confidence result. This trades a small accuracy loss for major throughput gains:

| Threshold | Gemini calls | Accuracy impact |
|---|---|---|
| 0.50 | ~70% of images | Minimal |
| 0.60 (default) | ~50% | None for high-confidence images |
| 0.75 | ~30% | Possible loss on ambiguous scenes |
| 0.90 | ~15% | Not recommended |

### 4.3 Prompt optimization

The Gemini prompt in `src/utils/gemini_integration.py` has already been tuned for minimal token count while preserving response quality. Key principles applied:

- Provide the Stage 1 category as context (reduces Gemini's uncertainty, fewer tokens).
- Request structured JSON output (eliminates post-processing ambiguity).
- Constrain the response length with `max_output_tokens`.

### 4.4 Retry and timeout tuning

```yaml
# config.yaml
pipeline:
  stage2_max_retries: 2
  stage2_timeout_s: 15.0
  stage2_retry_backoff_base: 1.5   # exponential back-off multiplier
```

Optimal retry strategy:
- `max_retries = 2` for demo (balance between reliability and latency).
- `max_retries = 1` for high-throughput batch processing (prefer failing fast).
- `timeout_s = 10` for interactive use; `20` for batch.

### 4.5 Request batching (future work)

Gemini currently processes one image per API call. When Google releases a batch endpoint, updating `GeminiLocationAnalyzer` to batch multiple images per call will reduce per-image latency by ~40% at scale.

---

## 5. End-to-End Pipeline Optimization

### 5.1 Two-stage confidence routing

The `TwoStagePipelineOptimizer` in `src/utils/pipeline_optimizer.py` implements the full routing logic:

```python
from src.utils.pipeline_optimizer import TwoStagePipelineOptimizer, PipelineConfig

cfg = PipelineConfig(
    stage1_confidence_threshold = 0.60,
    stage2_max_retries         = 2,
    stage2_timeout_s           = 15.0,
    fallback_on_low_confidence = False,  # True → return only Stage 1 on low conf
    enable_profiling           = True,
)
pipeline = TwoStagePipelineOptimizer(model, gemini_analyzer, cfg)
result   = pipeline.run(image)
```

### 5.2 Latency profiling

```python
pipeline.print_latency_report()
# ── Latency Report ────────────────────────────────────
# stage1   mean=145.3ms  p95=210.1ms  n=100
# stage2   mean=2341.7ms p95=4102.3ms n=63   (success: 98.4%)
# ─────────────────────────────────────────────────────

pipeline.save_profile("results/metrics/latency_profile.json")
```

### 5.3 Preprocessing bottleneck reduction

Standard ImageNet pre-processing (resize → centre-crop → normalise) can be accelerated:

```python
# Use LANCZOS only for the first resize; BILINEAR is faster and sufficient
from torchvision import transforms as T

fast_val_transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),  # faster than LANCZOS
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

Impact: ~20% reduction in preprocessing time with negligible accuracy loss.

---

## 6. Caching Strategy

### 6.1 Architecture

```
┌─────────────────────────────────────────┐
│           GeminiCacheManager            │
│ ┌───────────────────────────────────┐   │
│ │  In-process LRU dict (fast path)  │   │
│ └───────────────────────────────────┘   │
│ ┌───────────────────────────────────┐   │
│ │  SQLite disk cache (persistent)   │   │
│ └───────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

The `GeminiCacheManager` in `src/utils/cache_manager.py` uses:

- **Cache key:** Perceptual hash (pHash) of the input image + scene category.  
  Two visually similar images (e.g. resized versions of the same photo) will share a cache key.
- **Storage:** SQLite database at `results/gemini_cache.db`.
- **Eviction:** LRU — oldest `last_used` entries removed when `max_size` is exceeded.
- **TTL:** Entries expire after `ttl_seconds` (default 7 days). Stale entries are deleted on next access.

### 6.2 Cache size recommendations

| Use case | `max_size` | `ttl_days` | Expected hit rate |
|---|---|---|---|
| Demo (single session) | 200 | 1 | 30–60% |
| Research / repeat experiments | 1000 | 7 | 50–80% |
| Production app | 5000 | 30 | 60–90% |

### 6.3 Integrating with the pipeline

```python
from src.utils.cache_manager import CachedGeminiAnalyzer
from src.utils.gemini_integration import GeminiLocationAnalyzer

base_analyzer   = GeminiLocationAnalyzer(api_key=os.environ["GEMINI_API_KEY"])
cached_analyzer = CachedGeminiAnalyzer(base_analyzer,
                                       db_path="results/gemini_cache.db")

# Use exactly like the original analyzer
result = cached_analyzer.analyze_location(image, predicted_category, confidence)
print(result["_from_cache"])   # True / False
```

### 6.4 Cache monitoring

```python
cached_analyzer.cache.print_stats()
# ══════════════════════════════════════════════════
# Gemini Cache Statistics
# ══════════════════════════════════════════════════
#   Entries  : 143 / 1000
#   Hits     : 87
#   Misses   : 56
#   Hit rate : 60.8%
#   Evictions: 0
#   Expired  : 3
# ══════════════════════════════════════════════════
```

---

## 7. GPU vs CPU Benchmarks

Benchmarks conducted on representative hardware (indicative, not absolute):

### Stage 1 — EfficientNet-B0, single image, FP32

| Hardware | Preprocessing | Inference | Total |
|---|---|---|---|
| CPU (Intel i7-12700, 8-core) | 22 ms | 165 ms | 187 ms |
| GPU (NVIDIA RTX 3060, 12 GB) | 22 ms | 12 ms  | 34 ms  |
| GPU (NVIDIA T4, Colab)       | 25 ms | 18 ms  | 43 ms  |

### Stage 1 throughput (batch size 32)

| Hardware | Images/second | Notes |
|---|---|---|
| CPU (4 workers) | 28 img/s | `num_workers=4`, `pin_memory=False` |
| GPU RTX 3060    | 310 img/s | `pin_memory=True`, FP32 |
| GPU RTX 3060 (FP16) | 520 img/s | `autocast()` |

### Stage 2 — Gemini Vision API (network-dependent)

| Connection | Typical latency | P95 |
|---|---|---|
| University network (India) | 2.1 s | 4.8 s |
| Home broadband (50 Mbps+)  | 1.5 s | 3.2 s |
| Google Colab (US region)   | 0.9 s | 2.1 s |

---

## 8. Memory Optimization

### 8.1 Model memory footprint

| Architecture | FP32 VRAM | FP16 VRAM |
|---|---|---|
| ResNet-50        | ~100 MB | ~50 MB |
| EfficientNet-B0  | ~21 MB  | ~11 MB |
| ViT-B/16         | ~330 MB | ~165 MB |

### 8.2 Gradient accumulation (training only)

When GPU memory is limited (< 8 GB), use gradient accumulation to simulate larger batches:

```python
# In AdvancedTrainer — set in EfficientNetTrainConfig
cfg = EfficientNetTrainConfig(
    batch_size        = 8,    # actual batch per step
    grad_accum_steps  = 4,    # effective batch = 8 × 4 = 32
)
```

### 8.3 Inference-only memory saving

```python
# Disable gradient tracking during inference
with torch.no_grad():
    outputs = model(inputs)

# Release unused tensors explicitly
del inputs, outputs
torch.cuda.empty_cache()
```

### 8.4 Cache database size

Each SQLite cache entry is approximately 2–4 KB (compressed Gemini JSON response).

| Max entries | Approximate DB size |
|---|---|
| 200 | ~0.5 MB |
| 1000 | ~3 MB |
| 5000 | ~15 MB |

---

## 9. Concurrency and Throughput

### 9.1 Thread-pool model

The `LoadTester` in `src/utils/load_tester.py` uses a `ThreadPoolExecutor`. Key observations from ramp tests:

- **Stage 1 (CPU):** Throughput scales linearly up to `n_workers = os.cpu_count() // 2`, then plateaus due to Python GIL on pure-Python code. Use `n_workers = 4` as a safe default.
- **Stage 1 (GPU):** GPU inference is already compiled for batch parallelism. Using multiple threads provides minimal additional benefit and may increase GPU memory pressure.
- **Stage 2 (Gemini):** Network-bound; scales well with threading. Use `n_workers = 4–8` to mask API latency.

### 9.2 Recommended worker counts by scenario

| Scenario | Stage 1 workers | Stage 2 threads | Total |
|---|---|---|---|
| Demo (single user) | 1 | 1 | 1 |
| Batch evaluation | 4 (CPU) or 1 (GPU) | 4 | 4–5 |
| Research pipeline | 8 | 4 | 8 |

### 9.3 Running a load test

```python
from src.utils.load_tester import run_full_load_suite

images  = [load_image(p) for p in image_paths]          # your test images
reports = run_full_load_suite(
    inference_fn  = pipeline.run,
    images        = images,
    output_dir    = "results/metrics",
    worker_counts = (1, 2, 4),
)
```

---

## 10. Profiling Tools and Methodology

### 10.1 PyTorch Profiler (Stage 1)

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
) as prof:
    with torch.no_grad():
        outputs = model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("results/metrics/trace.json")
```

### 10.2 Pipeline latency profiler

```python
from src.utils.pipeline_optimizer import TwoStagePipelineOptimizer, PipelineConfig

cfg = PipelineConfig(enable_profiling=True)
pipeline = TwoStagePipelineOptimizer(model, gemini_analyzer, cfg)

# Process images
for img in test_images:
    pipeline.run(img)

# View detailed latency report
pipeline.print_latency_report()
pipeline.save_profile("results/metrics/latency_profile.json")
```

### 10.3 Preprocessing benchmark

```python
from src.preprocessing.pipeline import benchmark_dataloader

result = benchmark_dataloader(
    data_dir    = "data/processed",
    num_workers = 4,
    n_batches   = 50,
)
# Output: {'batches': 50, 'total_s': 8.23, 'throughput_imgs_per_s': 195.4}
```

### 10.4 Identifying bottlenecks — decision tree

```
Is your end-to-end latency > 10 s?
├── YES: Is Stage 2 latency > 8 s?
│       ├── YES: Check network; raise stage2_timeout_s or lower max_retries
│       └── NO:  Is Stage 1 latency > 2 s?
│                ├── YES: Enable GPU; switch to EfficientNet-B0
│                └── NO:  Check preprocessing time (see §3.6)
└── NO: Latency is within target — check cache hit rate (§6.4)
```

---

## 11. Optimization Recommendations by Deployment Scenario

### Scenario A — Interactive demo (single user, laptop CPU)

| Optimization | Setting |
|---|---|
| Architecture | EfficientNet-B0 |
| Confidence threshold | 0.65 |
| Caching enabled | Yes, `max_size=500`, `ttl_days=7` |
| Stage 2 timeout | 12 s |
| Stage 2 retries | 2 |
| Workers | 1 |

Expected end-to-end latency: **200 ms (cache hit) / 3–5 s (Gemini call)**

---

### Scenario B — Batch evaluation (100–1000 images, GPU server)

| Optimization | Setting |
|---|---|
| Architecture | EfficientNet-B0 |
| Confidence threshold | 0.60 |
| Caching enabled | Yes, `max_size=2000`, `ttl_days=30` |
| Stage 2 timeout | 20 s |
| Stage 2 retries | 1 (fail fast) |
| Stage 1 workers | 4 |
| Stage 2 threads | 4 |
| Mixed precision | Yes (`autocast()`) |

Expected throughput: **80–120 images/minute**

---

### Scenario C — Research (repeatability, offline analysis)

| Optimization | Setting |
|---|---|
| Architecture | ViT-B/16 (highest accuracy) |
| Confidence threshold | 0.50 (route most images to Gemini) |
| Caching enabled | Yes, `max_size=5000`, `ttl_days=60` |
| Stage 2 timeout | 30 s |
| Stage 2 retries | 3 |
| Workers | `os.cpu_count()` |

Pre-warm the cache with the full test set. On the second run, ~70% of Gemini calls will be served from cache.

---

*For technical queries, contact the Preprocessing & Pipeline team lead: Anuj Kondawar.*
