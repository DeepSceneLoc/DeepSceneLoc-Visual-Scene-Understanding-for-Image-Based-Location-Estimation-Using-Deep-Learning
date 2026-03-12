# DeepSceneLoc — Deployment Guide

**Author:** Anuj Kondawar (Preprocessing & Pipeline Lead — Semester 2, Weeks 14–16)  
**Project:** DeepSceneLoc — Visual Scene Understanding for Image-Based Location Estimation  
**Last Updated:** Semester 2, Week 16

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Local Deployment (Python)](#2-local-deployment-python)
3. [Google Colab Deployment](#3-google-colab-deployment)
4. [Gradio Public Sharing](#4-gradio-public-sharing)
5. [Environment Variables](#5-environment-variables)
6. [Configuration Overview](#6-configuration-overview)
7. [Model Checkpoints](#7-model-checkpoints)
8. [Docker Deployment (Optional)](#8-docker-deployment-optional)
9. [Troubleshooting](#9-troubleshooting)
10. [Performance Tuning at Deployment](#10-performance-tuning-at-deployment)

---

## 1. Prerequisites

### Hardware requirements

| Mode           | CPU             | RAM   | GPU (optional)           |
|----------------|-----------------|-------|--------------------------|
| Demo / inference | Any modern CPU | 4 GB  | CUDA-capable (≥ 4 GB VRAM) |
| Training         | 8-core+         | 16 GB | ≥ 8 GB VRAM recommended |

### Software requirements

- Python 3.9 – 3.11 (3.10 recommended)
- `pip` ≥ 23.0
- Git

### Gemini API key

The Stage 2 (exact location) pipeline calls the Google Gemini Vision API. You must obtain a free key from [Google AI Studio](https://aistudio.google.com/app/apikey) before running the application.

---

## 2. Local Deployment (Python)

### Step 1 — Clone the repository

```bash
git clone <repository-url>
cd "DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning"
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Set the Gemini API key

```bash
# Windows (Command Prompt)
set GEMINI_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:GEMINI_API_KEY = "your_api_key_here"

# macOS / Linux
export GEMINI_API_KEY="your_api_key_here"
```

Alternatively, create a `.env` file in the project root:

```ini
GEMINI_API_KEY=your_api_key_here
```

> **Security note:** Never commit `.env` or any file containing your API key to version control. The `.gitignore` already excludes `.env`.

### Step 5 — Download a model checkpoint

Place a trained `.pth` checkpoint in `models/checkpoints/`. If you have no checkpoint yet, the demo will run with random weights (predictions will be meaningless but the UI will load).

```
models/
  checkpoints/
    efficientnet_b0_best.pth    ← recommended
    resnet50_best.pth
    vit_best.pth
```

### Step 6 — Launch the demo

```bash
python demo_app.py
```

The Gradio interface will open automatically in your default browser at `http://127.0.0.1:7860`.

---

## 3. Google Colab Deployment

### Quick-start notebook

```python
# Cell 1 — Clone & install
!git clone <repository-url>
%cd "DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning"
!pip install -r requirements.txt -q

# Cell 2 — Mount Drive (optional, for persistent checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3 — Set API key (Colab Secrets recommended)
import os
from google.colab import userdata
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")

# Cell 4 — Copy checkpoint from Drive
!cp /content/drive/MyDrive/deepsceneloc/efficientnet_b0_best.pth models/checkpoints/

# Cell 5 — Launch demo with public URL
import subprocess
proc = subprocess.Popen(["python", "demo_app.py", "--share"])
```

### Notes for Colab

- Use **Colab Pro / Pro+** for GPU access; switch to a T4 or A100 runtime.
- Free-tier GPU sessions time out after ~90 minutes of inactivity.
- The Gemini API is called over the internet from Colab cells; ensure your key has sufficient quota.

---

## 4. Gradio Public Sharing

To expose the demo on a publicly shareable URL (through Gradio's tunnelling):

```python
# In demo_app.py — the share flag is already configurable via config.yaml
```

Or launch directly:

```bash
python demo_app.py --share
```

A URL of the form `https://xxxxxxxxxxxx.gradio.live` will be printed. This link is valid for **72 hours** from launch.

> **Privacy note:** Uploaded images are processed locally (Stage 1) and sent to Google's Gemini API (Stage 2). Do not upload images containing personal or sensitive location data.

---

## 5. Environment Variables

| Variable           | Required | Default | Description |
|--------------------|----------|---------|-------------|
| `GEMINI_API_KEY`   | **Yes**  | —       | Google AI Studio API key for Stage 2 location analysis |
| `DEEPSCENELOC_DEVICE` | No    | `auto`  | Force compute device: `cpu`, `cuda`, `mps`, or `auto` |
| `DEEPSCENELOC_LOG_LEVEL` | No | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `DEEPSCENELOC_CACHE_DB` | No  | `results/gemini_cache.db` | Path to the Gemini response SQLite cache |
| `DEEPSCENELOC_MAX_WORKERS` | No | `1` | Thread-pool size for concurrent inference (demo mode) |

### Loading `.env` automatically

```python
# Add to src/__init__.py or demo_app.py top-level
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass   # python-dotenv optional; set env manually
```

---

## 6. Configuration Overview

All tuneable parameters live in `config.yaml`:

```yaml
# config.yaml (key sections)
model:
  architecture: efficientnet_b0   # resnet50 | efficientnet_b0 | vit
  checkpoint: models/checkpoints/efficientnet_b0_best.pth
  num_classes: 5
  device: auto

pipeline:
  stage1_confidence_threshold: 0.60
  stage2_max_retries: 2
  stage2_timeout_s: 15.0
  enable_caching: true
  cache_db: results/gemini_cache.db
  cache_ttl_days: 7

app:
  share: false
  port: 7860
  max_file_size_mb: 10
```

Load it in code:

```python
import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
arch = cfg["model"]["architecture"]
```

---

## 7. Model Checkpoints

### Checkpoint naming convention

```
models/checkpoints/<architecture>_<epoch>_<val_acc>.pth
models/checkpoints/<architecture>_best.pth   ← symlink / copy of the best run
```

### Loading a checkpoint manually

```python
import torch
from src.models.model import build_model

model = build_model(architecture="efficientnet_b0", num_classes=5)
ckpt  = torch.load("models/checkpoints/efficientnet_b0_best.pth",
                   map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

### Checkpoint contents

Each `.pth` file saved by `AdvancedTrainer` contains:

```python
{
    "epoch":               int,
    "model_state_dict":    dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "val_acc":             float,
    "val_loss":            float,
    "config":              dict,   # training hyperparameters
}
```

### Resuming training

```python
from src.models.train_advanced import AdvancedTrainer, EfficientNetTrainConfig

cfg     = EfficientNetTrainConfig()
trainer = AdvancedTrainer(model, train_loader, val_loader, cfg)
trainer.train(resume_from="models/checkpoints/efficientnet_b0_epoch_10.pth")
```

---

## 8. Docker Deployment (Optional)

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GEMINI_API_KEY=""
ENV DEEPSCENELOC_DEVICE=cpu

EXPOSE 7860

CMD ["python", "demo_app.py"]
```

### Build and run

```bash
docker build -t deepsceneloc .
docker run -p 7860:7860 \
  -e GEMINI_API_KEY="your_api_key_here" \
  -v $(pwd)/models:/app/models \
  deepsceneloc
```

Navigate to `http://localhost:7860`.

### Notes

- For GPU support inside Docker, use `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` as base and install the NVIDIA Container Toolkit on the host.
- Mount `models/` from the host to avoid bundling large checkpoint files in the image.

---

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run from the project root directory or add the root to `PYTHONPATH`:

```bash
# Linux / macOS
export PYTHONPATH=$(pwd)

# Windows PowerShell
$env:PYTHONPATH = (Get-Location).Path
```

---

### `GEMINI_API_KEY not set` / `google.api_core.exceptions.PermissionDenied`

- Verify the key is set: `echo $GEMINI_API_KEY` (Linux) or `echo %GEMINI_API_KEY%` (Windows CMD).
- Ensure the key has the **Gemini API** enabled in Google Cloud Console.
- Check daily quota limits; free-tier keys have 60 requests/minute.

---

### Gradio fails to launch / port already in use

```bash
# Change port
python demo_app.py --port 7861
```

Or update `config.yaml`:

```yaml
app:
  port: 7861
```

---

### CUDA out of memory

Reduce the batch size in `config.yaml` to `1` for inference-only use, or force CPU mode:

```bash
DEEPSCENELOC_DEVICE=cpu python demo_app.py
```

---

### Stage 2 returns empty / timeout errors

- Check internet connectivity (Gemini API is cloud-based).
- Increase `stage2_timeout_s` in `config.yaml`.
- The pipeline falls back to Stage 1 category-only result when Gemini is unreachable.

---

### Slow first-run inference

The first inference after startup triggers model warm-up (JIT compilation on GPU). Subsequent inferences will be significantly faster. You can pre-warm the model:

```python
import torch
from PIL import Image
dummy = Image.new("RGB", (224, 224))
with torch.no_grad():
    _ = model(transform(dummy).unsqueeze(0).to(device))
```

---

## 10. Performance Tuning at Deployment

| Optimisation | Impact | How to enable |
|---|---|---|
| Enable Gemini response cache | Eliminates repeated API calls (up to 100% latency reduction for cached images) | `pipeline.enable_caching: true` in `config.yaml` |
| Increase cache TTL | More cache hits across sessions | `pipeline.cache_ttl_days: 30` |
| Use EfficientNet-B0 over ResNet-50 | ~15% faster Stage 1, similar accuracy | `model.architecture: efficientnet_b0` |
| Run on GPU | 5–20× Stage 1 speedup | Set `DEEPSCENELOC_DEVICE=cuda` |
| Reduce Gemini retries | Faster failure recovery | `pipeline.stage2_max_retries: 1` |
| Lower Stage 1 confidence threshold | More images routed to Gemini (better accuracy, higher cost) | `pipeline.stage1_confidence_threshold: 0.50` |
| Raise Stage 1 confidence threshold | Fewer Gemini calls (lower cost, faster) | `pipeline.stage1_confidence_threshold: 0.80` |

For a detailed analysis of all optimisation strategies, see [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md).
