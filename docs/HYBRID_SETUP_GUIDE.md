# DeepSceneLoc — Hybrid System Setup Guide
## Two-Stage AI Architecture: Scene Classification + Exact Location Detection

**Author:** Anuj Kondawar (Pipeline & Optimization Lead)  
**Week:** 12 (April 7–13, 2026) — advance delivery  
**Last Updated:** April 28, 2026

---

## System Architecture

The DeepSceneLoc hybrid system uses two AI stages to identify the exact place on Earth from any image:

```
Input Image
    │
    ▼
┌─────────────────────────────────┐
│  STAGE 1: Scene Classification  │
│  Model: EfficientNet-B0         │
│  Accuracy: 85.15% (val)         │
│  Speed: 7.6ms/image (GPU)       │
│  Output: scene_type + confidence│
└─────────────────────────────────┘
    │  scene_type (e.g. "Urban")
    │  confidence (e.g. 0.87)
    ▼
┌─────────────────────────────────┐
│  STAGE 2: Location Detection    │
│  Model: Google Gemini AI        │
│  Condition: confidence > 0.5    │
│  Output: landmark, city,        │
│          country, lat/lon       │
└─────────────────────────────────┘
    │
    ▼
Combined Output JSON
```

### Confidence Gating
Stage 2 (Gemini) is only called when Stage 1 confidence > 50%. This prevents wasting API calls on ambiguous inputs and is implemented via `TwoStagePipelineOptimizer` in `src/utils/pipeline_optimizer.py`.

---

## Prerequisites

### Hardware
- GPU: NVIDIA (CUDA) recommended, CPU supported
- RAM: 4GB minimum, 8GB recommended
- Storage: 500MB (models + cache)

### Software
```bash
# Core requirements
venv\Scripts\pip.exe install -r requirements.txt

# Hybrid-specific
venv\Scripts\pip.exe install google-generativeai>=0.3.0
```

### API Key
Get a free Gemini API key at: https://aistudio.google.com/app/apikey  
Free tier: 15 requests/minute, 1M tokens/day (sufficient for demos)

---

## Quick Setup

### Step 1: Verify Stage 1 Model
```bash
venv\Scripts\python.exe -c "
import torch
from pathlib import Path
ckpt = Path('models/checkpoints/efficientnet/EfficientNet-B0_best.pth')
d = torch.load(str(ckpt), map_location='cpu', weights_only=False)
print(f'Model ready: epoch={d[\"epoch\"]} val_acc={d[\"val_acc\"]*100:.2f}%')
"
```
Expected output: `Model ready: epoch=3 val_acc=85.15%`

### Step 2: Set Gemini API Key
```powershell
# Windows PowerShell (temporary — current session only)
$env:GEMINI_API_KEY = "your-api-key-here"

# Permanent (add to system environment variables)
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your-key", "User")
```

### Step 3: Test Gemini Connection
```bash
venv\Scripts\python.exe -c "
import os
from src.utils.gemini_integration import GeminiLocationAnalyzer
key = os.getenv('GEMINI_API_KEY', '')
if not key:
    print('ERROR: GEMINI_API_KEY not set')
else:
    analyzer = GeminiLocationAnalyzer(api_key=key)
    print('Gemini connection: OK')
    print(f'Model: {analyzer.model_name}')
"
```

### Step 4: Run the Hybrid Demo
```bash
venv\Scripts\python.exe demo_app_hybrid.py
```
**Access:** http://127.0.0.1:7861

---

## Production API (FastAPI)

The production webapp at `webapp/api.py` includes the full two-stage pipeline.

### Start Server
```bash
venv\Scripts\python.exe -m uvicorn webapp.api:app --host 0.0.0.0 --port 8000 --reload
```

### Two-Stage Endpoint
```bash
# Full analysis (Stage 1 + Stage 2)
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@image.jpg"
```

### Response Format
```json
{
  "stage1": {
    "top_class": "Urban",
    "confidence": 0.87,
    "emoji": "🏙️",
    "probabilities": {"Coastal": 0.03, "Forest": 0.02, "Mountain": 0.01, "Rural": 0.07, "Urban": 0.87},
    "latency_ms": 7.6
  },
  "stage2": {
    "exact_location": "Times Square, Manhattan, New York",
    "country": "United States",
    "city": "New York",
    "region": "New York State",
    "latitude": 40.758,
    "longitude": -73.9855,
    "landmarks": ["TKTS Booth", "One Times Square", "Broadway"],
    "confidence": "high",
    "latency_ms": 1240.0
  },
  "total_latency_ms": 1247.6
}
```

---

## Pipeline Optimizer (src/utils/pipeline_optimizer.py)

The `TwoStagePipelineOptimizer` class handles:
- **Confidence gating**: Skip Gemini if Stage 1 < 50% confidence
- **Retry logic**: Up to 3 retries on Gemini timeout
- **Latency profiling**: Per-request timing breakdown
- **Error fallback**: Return Stage 1 result if Gemini fails

```python
from src.utils.pipeline_optimizer import TwoStagePipelineOptimizer

optimizer = TwoStagePipelineOptimizer(
    stage1_threshold=0.5,    # Min confidence to trigger Stage 2
    max_retries=3,
    timeout_s=10.0
)
```

---

## Cache Manager (src/utils/cache_manager.py)

Caches Gemini API responses to avoid duplicate API calls:
- **Storage:** SQLite database at `results/gemini_cache.db`
- **Deduplication:** pHash-based image similarity (avoids re-querying same scene)
- **TTL:** 30 days (configurable)
- **LRU eviction:** Max 10,000 entries

```python
from src.utils.gemini_integration import GeminiLocationAnalyzer
from src.utils.cache_manager import CachedGeminiAnalyzer

base = GeminiLocationAnalyzer(api_key=key)
cached = CachedGeminiAnalyzer(base, db_path="results/gemini_cache.db")
result = cached.analyze_location(image, predicted_category="Urban", confidence=0.87)
```

**Cost estimate:** With caching, ~200 unique queries/day stays in Gemini free tier (15 RPM × 60 min × 24h = 21,600 RPM capacity, 200 unique is negligible).

---

## Design Decisions & Pivots

### Why EfficientNet-B0 instead of ResNet-50 for Stage 1?
**Originally planned:** ResNet-50 as Stage 1 (79.17% val acc)  
**Actual:** EfficientNet-B0 (85.15% val acc)  
**Reason:** After completing Semester 2 training, EfficientNet-B0 achieved +5.98% better accuracy at 75% smaller model size (69.6MB vs 281.5MB). The production webapp was updated April 28, 2026.

### Why Gemini for Stage 2 instead of a custom model?
**Originally considered:** Training a custom geolocation model (PlaNet-style)  
**Actual:** Google Gemini AI via API  
**Reason:** Training a geolocation model requires tens of millions of geotagged images (PlaNet used 91M). Our 289,652 image dataset is insufficient for city/landmark-level localization. Gemini provides state-of-the-art visual understanding out of the box with zero training cost.

### Why confidence gating?
**Originally planned:** Always call Gemini  
**Actual:** Only call when Stage 1 > 50% confidence  
**Reason:** If the scene is ambiguous (e.g., nondescript indoor), Gemini hallucination risk is high. Gating on confidence prevents low-quality location guesses and saves API quota.

### Why SQLite cache?
**Originally planned:** No caching (direct API calls)  
**Actual:** SQLite LRU cache with pHash dedup  
**Reason:** Repeat demo images (same test images run multiple times) would exhaust API quota quickly. Cache hit rate in demos is typically 80%+, reducing API calls by 5×.

---

## Load Testing (src/utils/load_tester.py)

Test system stability under concurrent requests:
```bash
venv\Scripts\python.exe -c "
from src.utils.load_tester import LoadTester
tester = LoadTester(base_url='http://localhost:8000')
results = tester.run_load_test(n_requests=50, concurrency=5, image_path='data/sample.jpg')
print(results.summary())
"
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `GEMINI_API_KEY not set` | Environment variable missing | Set `$env:GEMINI_API_KEY` |
| `google.api_core.exceptions.ResourceExhausted` | Rate limit hit | Wait 1 minute or reduce request rate |
| `Stage 2 skipped` | Stage 1 confidence < 50% | Use clearer/higher-quality images |
| `Cache DB locked` | Multiple processes accessing | Stop other instances |
| Slow Stage 2 response | Network latency | Expected 1-3s for Gemini; normal |

---

## Performance Summary

| Stage | Latency (avg) | Can Run Without? |
|---|---|---|
| Stage 1 (EfficientNet-B0 on GPU) | 7.6ms | N/A — required |
| Stage 1 (EfficientNet-B0 on CPU) | 14.7ms | N/A — required |
| Stage 2 (Gemini API) | ~1,000–3,000ms | Yes — system degrades gracefully |
| Total (GPU + Gemini) | ~1,010–3,010ms | — |

---

**See also:** `DEMO_GUIDE.md` for running instructions | `docs/TRAINING_METHODOLOGY.md` for model details | `docs/DEPLOYMENT_GUIDE.md` for production deployment
