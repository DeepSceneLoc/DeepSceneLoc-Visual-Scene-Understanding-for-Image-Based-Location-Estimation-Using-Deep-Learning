# DeepSceneLoc — Demo Guide
## Running All Three Demo Modes

**Last Updated:** April 28, 2026  
**Phase 1 Model:** ResNet-50 — 79.17% val accuracy  
**Phase 2 Model:** EfficientNet-B0 — 85.15% val accuracy (production)  
**Dataset:** MIT Places365 (5 classes: Coastal, Forest, Mountain, Rural, Urban)

---

## Quick Start

| Demo | Model | Interface | Port | Purpose |
|---|---|---|---|---|
| `demo_app.py` | ResNet-50 | Gradio | 7860 | Phase 1 Semester demo |
| `webapp/api.py` | EfficientNet-B0 | FastAPI + HTML | 8000 | Phase 2 Production |
| `demo_app_hybrid.py` | EfficientNet-B0 + Gemini AI | Gradio | 7861 | Two-stage location detection |

---

## Demo 1: Phase 1 — Gradio Demo (ResNet-50)

**Original Phase 1 demo for Semester 1 review. Preserved as reference.**

```bash
# Install dependency
venv\Scripts\pip.exe install gradio

# Run
venv\Scripts\python.exe demo_app.py
```

**Access:** http://127.0.0.1:7860  
**Model:** ResNet-50 (281.5 MB) — `models/checkpoints/resnet/best_model.pth`  
**Accuracy:** 79.17% validation

**What it shows:**
- 5 location category probabilities (bar chart)
- Confidence score for top prediction
- Model architecture info (ResNet-50, 24.6M params)

**Originally Planned:** This was the Semester 1 final demo.  
**Why Preserved:** Kept as Phase 1 baseline for comparison with Phase 2.

---

## Demo 2: Phase 2 — Production Webapp (EfficientNet-B0) ← PRIMARY

**The main Phase 2 production application. Use this for mentor demos.**

```bash
# Run the FastAPI server
venv\Scripts\python.exe -m uvicorn webapp.api:app --host 0.0.0.0 --port 8000 --reload
```

**Access:** http://localhost:8000  
**Model:** EfficientNet-B0 (69.6 MB) — `models/checkpoints/efficientnet/EfficientNet-B0_best.pth`  
**Accuracy:** 85.15% validation / 84.63% test / 83.17% macro F1

**Features:**
- Premium dark-mode HTML/CSS/JS interface
- Drag-and-drop image upload
- Real-time prediction with confidence bars
- Per-class probability visualization
- Model info (architecture, accuracy, inference speed)
- REST API for integration (`POST /api/predict`)

**API Endpoints:**
```
GET  /              → Frontend
GET  /api/status    → Model + GPU health
GET  /api/models    → Available checkpoints
POST /api/predict   → Scene classification (returns JSON)
POST /api/analyze   → Scene + Gemini location (requires GEMINI_API_KEY)
```

**Quick API test:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@your_image.jpg"
```

**Originally Planned:** FastAPI + ResNet-50  
**Actual:** FastAPI + EfficientNet-B0 (swapped April 28, 2026 after training completed)  
**Reason:** EfficientNet-B0 achieved 85.15% vs ResNet-50's 79.17% — 5.6% improvement.

---

## Demo 3: Hybrid AI Demo (EfficientNet-B0 + Gemini)

**Two-stage system: scene classification → exact location detection.**

```bash
# Set Gemini API key (required for Stage 2)
$env:GEMINI_API_KEY = "your-api-key-here"

# Install dependency
venv\Scripts\pip.exe install google-generativeai

# Run
venv\Scripts\python.exe demo_app_hybrid.py
```

**Access:** http://127.0.0.1:7861  

**What it does:**
1. **Stage 1:** EfficientNet-B0 classifies scene (Coastal / Forest / Mountain / Rural / Urban)
2. **Stage 2:** Gemini AI identifies exact location, landmarks, GPS coordinates, country, city

**Without API Key:** Stage 1 only (scene classification still works)  
**With API Key:** Full two-stage output including exact location

**Output example:**
```
Scene: Urban (87% confidence)
Exact Place: Times Square, New York
Country: United States
Coordinates: 40.7580°N, 73.9855°W
Landmarks: TKTS Booth, One Times Square, Broadway
```

---

## Inference Speed Reference

| Model | GPU (avg) | CPU (avg) | Throughput (GPU) | Params | Size |
|---|---|---|---|---|---|
| ResNet-50 | 5.7ms | 34.5ms | 176 fps | 24.6M | 281.5 MB |
| EfficientNet-B0 | 7.6ms | 14.7ms | 131 fps | 4.7M | 69.6 MB |

> EfficientNet-B0 is 2.3× faster on CPU despite higher accuracy.  
> ResNet-50 is faster on GPU due to simpler architecture.

---

## Model Checkpoint Locations

```
models/
├── checkpoints/
│   ├── efficientnet/
│   │   └── EfficientNet-B0_best.pth   # Primary (69.6 MB, 85.15% val)
│   └── resnet/
│       └── best_model.pth             # Fallback (281.5 MB, 79.17% val)
```

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `No module named gradio` | Gradio not installed | `venv\Scripts\pip.exe install gradio` |
| `No module named fastapi` | FastAPI not installed | `venv\Scripts\pip.exe install fastapi uvicorn` |
| Checkpoint not found | File moved | Check `models/checkpoints/` structure above |
| CUDA out of memory | Batch too large | Reduce batch size or use CPU |
| `GEMINI_API_KEY` missing | Not set | Set env variable or use Stage 1 only |
| Port already in use | Another process | Change port: `--port 8001` |

---

## Running All Three Simultaneously

```bash
# Terminal 1: Production webapp (port 8000)
venv\Scripts\python.exe -m uvicorn webapp.api:app --port 8000

# Terminal 2: Phase 1 demo (port 7860)
venv\Scripts\python.exe demo_app.py

# Terminal 3: Hybrid demo (port 7861)
$env:GEMINI_API_KEY="your-key"
venv\Scripts\python.exe demo_app_hybrid.py
```
