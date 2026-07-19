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
| `start_fullstack.bat` | ResNet-50 + EfficientNet-B0 + ViT-B16 | React Frontend + FastAPI Backend | 3000 / 5000 | Primary Production Fullstack Application |
| `demo_app.py` | ResNet-50 | Gradio | 7860 | Phase 1 Semester demo |
| `backend.py` | ResNet-50 + EfficientNet-B0 + ViT-B16 | FastAPI Backend + Stage 2 Gemini AI | 5000 | Production Inference Backend |
| `demo_app_hybrid.py` | EfficientNet-B0 + Gemini AI | Gradio | 7861 | Two-stage location detection (Gradio) |

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

## Demo 2: Production React + FastAPI Fullstack (Ensemble) ← PRIMARY

**The main production application. Use this for mentor demos.**

### Launch automatically:
Double-click `start_fullstack.bat` in the project root.

### Launch manually:
```bash
# Terminal 1: Run the FastAPI backend
venv\Scripts\python.exe backend.py

# Terminal 2: Run the React frontend
cd frontend
npm run dev
```

**Access:** http://localhost:3000  
**Models loaded:** 
- ResNet-50 — `model_repo/ResNet50/ResNet50_best_model.pth`
- EfficientNet-B0 — `model_repo/EfficientNet-B0/EfficientNet-B0_best.pth`
- ViT-B/16 — `model_repo/ViT-B_16/ViT-B_16_best.pth`

**Accuracy:** ~88% validation / test (ensemble average)

**Features:**
- Premium React/Vite interactive maps dashboard (moving 3D globe fallback)
- Real-time prediction with averaged probability ensemble
- Stage 2 Location Grounding using Gemini 3.1 Flash-Lite (under 3s response)
- Fully centralized env config in root `.env`

**API Endpoints (FastAPI on Port 5000):**
```
GET  /health              → Model + GPU health
POST /api/analyze-image   → Scene + Gemini location (requires GEMINI_API_KEY)
```

**Quick API test:**
```bash
curl -X POST http://localhost:5000/api/analyze-image \
  -H "Content-Type: application/json" \
  -d "{\"imageBase64\": \"data:image/jpeg;base64,...\"}"
```

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

models/
├── checkpoints/
│   ├── efficientnet/
│   │   └── EfficientNet-B0_best.pth   # Primary (69.6 MB, 85.15% val)
│   ├── resnet/
│   │   └── best_model.pth             # Fallback (281.5 MB, 79.17% val)
│   └── vit/
│       └── ViT-B_16_best.pth          # Transformer checkpoint (350+ MB)
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
# Terminal 1: Production FastAPI Backend (port 5000)
venv\Scripts\python.exe backend.py

# Terminal 2: React Frontend (port 3000)
cd frontend && npm run dev
```
