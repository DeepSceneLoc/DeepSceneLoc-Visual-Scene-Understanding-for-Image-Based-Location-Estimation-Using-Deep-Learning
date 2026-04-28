"""
DeepSceneLoc — Production FastAPI Backend
Week 8-12: Real Application Server

Endpoints:
  POST /api/predict           — Stage 1: scene classification
  POST /api/analyze           — Stage 1 + Stage 2 (Gemini) full pipeline
  GET  /api/status            — Model + system health
  GET  /api/models            — Available checkpoints
  GET  /                      — Serve frontend

Run:
    venv/Scripts/python.exe -m uvicorn webapp.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import os
import sys
import time
import base64
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ── Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model import create_model
from src.preprocessing.transforms import get_val_transforms

# Try advanced models
try:
    from src.models.model_advanced import create_advanced_model
    _ADV_OK = True
except ImportError:
    _ADV_OK = False

# Try Gemini
try:
    from src.utils.gemini_integration import GeminiLocationAnalyzer
    from src.utils.cache_manager import CachedGeminiAnalyzer
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

CLASS_NAMES = ["Coastal", "Forest", "Mountain", "Rural", "Urban"]
CLASS_EMOJI = {"Coastal": "🏖️", "Forest": "🌲", "Mountain": "⛰️", "Rural": "🌾", "Urban": "🏙️"}
CLASS_DESC  = {
    "Coastal":  "Beaches, shorelines, cliffs, ocean and sea views",
    "Forest":   "Dense woodland, jungle, tree canopy, national parks",
    "Mountain": "High altitude peaks, rocky terrain, alpine scenery",
    "Rural":    "Farmland, countryside, open fields and villages",
    "Urban":    "City streets, buildings, skylines and infrastructure",
}

CHECKPOINT_PRIORITY = [
    # Prefer fully trained EfficientNet when it reaches >50% val_acc
    # For now, ResNet-50 (79.17%) is the production model
    ("resnet50",        "model_repo/best_model.pth"),
    ("resnet50",        "models/checkpoints/best_model.pth"),
    ("efficientnet_b0", "models/checkpoints/efficientnet/EfficientNet-B0_best.pth"),
]

# ─────────────────────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_val_transforms()

_model        = None
_model_info   = {}
_gemini       = None

def _find_best_checkpoint():
    for arch, rel_path in CHECKPOINT_PRIORITY:
        p = ROOT / rel_path
        if p.exists():
            return arch, str(p)
    return None, None

def _load_model():
    global _model, _model_info
    arch, ckpt_path = _find_best_checkpoint()

    if arch is None:
        _model_info = {"status": "no_checkpoint", "arch": "none", "val_acc": None}
        return

    try:
        if arch == "efficientnet_b0" and _ADV_OK:
            m = create_advanced_model("efficientnet_b0", num_classes=5, pretrained=False)
        else:
            m = create_model("resnet50", num_classes=5, pretrained=False)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        key  = "model_state" if "model_state" in ckpt else "model_state_dict"
        m.load_state_dict(ckpt[key])
        m.to(device).eval()

        _model = m
        _model_info = {
            "status":   "loaded",
            "arch":     arch,
            "path":     ckpt_path,
            "val_acc":  ckpt.get("val_acc") or ckpt.get("best_val_acc"),
            "epoch":    ckpt.get("epoch"),
            "device":   str(device),
            "params":   sum(p.numel() for p in m.parameters()),
        }
        print(f"[OK] Model loaded: {arch} from {ckpt_path}")
    except Exception as e:
        _model_info = {"status": "error", "error": str(e)}
        print(f"[ERR] Model load failed: {e}")

def _load_gemini():
    global _gemini
    if not _GEMINI_OK:
        return
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return
    try:
        base   = GeminiLocationAnalyzer(api_key=api_key)
        _gemini = CachedGeminiAnalyzer(base, db_path=str(ROOT / "results/gemini_cache.db"))
        print("[OK] Gemini AI ready with cache")
    except Exception as e:
        print(f"[WARN] Gemini init failed: {e}")

_load_model()
_load_gemini()

# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="DeepSceneLoc API",
    description="Visual Scene Understanding + Exact Location Detection",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─────────────────────────────────────────────────────────────
# Helper: image → prediction
# ─────────────────────────────────────────────────────────────

def _predict(pil_image: Image.Image) -> dict:
    if _model is None:
        # Return uniform mock predictions
        probs = [0.2] * 5
        return {
            "probabilities": {n: probs[i] for i, n in enumerate(CLASS_NAMES)},
            "top_class": "Urban",
            "confidence": 0.2,
            "mock": True,
        }

    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = _model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    top_idx = int(np.argmax(probs))
    return {
        "probabilities": {n: float(probs[i]) for i, n in enumerate(CLASS_NAMES)},
        "top_class":    CLASS_NAMES[top_idx],
        "confidence":   float(probs[top_idx]),
        "mock":         False,
    }

def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = Path(__file__).parent / "templates" / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>DeepSceneLoc API running. Frontend not built yet.</h1>")

@app.get("/api/status")
async def get_status():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        }
    return {
        "model":  _model_info,
        "gemini": {
            "available": _gemini is not None,
            "library":   _GEMINI_OK,
        },
        "device": str(device),
        "gpu":    gpu_info,
        "classes": CLASS_NAMES,
    }

@app.get("/api/models")
async def list_models():
    models = []
    for arch, rel_path in CHECKPOINT_PRIORITY:
        p = ROOT / rel_path
        if p.exists():
            try:
                ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
                models.append({
                    "arch":    arch,
                    "path":    rel_path,
                    "epoch":   ckpt.get("epoch"),
                    "val_acc": ckpt.get("val_acc") or ckpt.get("best_val_acc"),
                    "size_mb": round(p.stat().st_size / 1e6, 1),
                })
            except Exception:
                pass
    return {"models": models}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Stage 1 only — scene classification."""
    t0 = time.perf_counter()
    try:
        data = await file.read()
        image = _bytes_to_pil(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    result = _predict(image)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    top  = result["top_class"]
    return {
        "top_class":    top,
        "confidence":   result["confidence"],
        "emoji":        CLASS_EMOJI.get(top, ""),
        "description":  CLASS_DESC.get(top, ""),
        "probabilities": result["probabilities"],
        "latency_ms":   latency_ms,
        "model":        _model_info.get("arch", "mock"),
        "mock":         result.get("mock", False),
    }

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Full two-stage: scene classification + Gemini location detection."""
    t0 = time.perf_counter()
    try:
        data = await file.read()
        image = _bytes_to_pil(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Stage 1
    stage1 = _predict(image)
    t1 = time.perf_counter()

    # Stage 2
    stage2 = None
    if _gemini is not None:
        try:
            stage2 = _gemini.analyze_location(
                image=image,
                predicted_category=stage1["top_class"],
                confidence=stage1["confidence"],
            )
        except Exception as e:
            stage2 = {"error": str(e), "exact_location": "Gemini error"}
    else:
        stage2 = {
            "exact_location": "Gemini not configured — set GEMINI_API_KEY",
            "confidence": "none",
            "description": f"Scene: {stage1['top_class']} ({stage1['confidence']:.1%})",
            "latitude": None, "longitude": None,
            "country": "—", "city": "—", "landmarks": [],
        }

    t2 = time.perf_counter()
    top = stage1["top_class"]

    return {
        "stage1": {
            "top_class":    top,
            "confidence":   stage1["confidence"],
            "emoji":        CLASS_EMOJI.get(top, ""),
            "description":  CLASS_DESC.get(top, ""),
            "probabilities": stage1["probabilities"],
            "latency_ms":   round((t1 - t0) * 1000, 1),
        },
        "stage2": {
            "exact_location": stage2.get("exact_location", "Unknown"),
            "country":     stage2.get("country", "—"),
            "city":        stage2.get("city", "—"),
            "region":      stage2.get("region", "—"),
            "latitude":    stage2.get("latitude"),
            "longitude":   stage2.get("longitude"),
            "landmarks":   stage2.get("landmarks", []),
            "confidence":  stage2.get("confidence", "none"),
            "description": stage2.get("description", ""),
            "from_cache":  stage2.get("_from_cache", False),
            "latency_ms":  round((t2 - t1) * 1000, 1),
        },
        "total_latency_ms": round((t2 - t0) * 1000, 1),
        "model": _model_info.get("arch", "mock"),
    }
