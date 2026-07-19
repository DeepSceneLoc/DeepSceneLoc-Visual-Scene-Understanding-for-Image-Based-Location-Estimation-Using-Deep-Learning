"""
DeepSceneLoc — Production FastAPI Backend
Exposes POST /api/analyze-image to return sceneCategory and confidence (percentage).
Loads ResNet-50, EfficientNet-B0, and ViT-B16, averages their probabilities.
Optionally performs Stage 2 Gemini/OpenRouter location prediction internally,
returning the full coordinate and location report if keys are set.
"""

from __future__ import annotations

import io
import os
import sys
import base64
import time
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to sys.path so we can import src
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model import create_model
from src.models.model_advanced import create_advanced_model
from src.preprocessing.transforms import get_val_transforms

# Try Gemini/OpenRouter Stage 2 analyzer
try:
    from src.utils.gemini_integration import GeminiLocationAnalyzer
    from src.utils.cache_manager import CachedGeminiAnalyzer
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False

# App Configuration
app = FastAPI(
    title="DeepSceneLoc Core Inference Backend",
    description="Loads PyTorch Ensemble (ResNet-50, EfficientNet-B0, ViT-B16) to classify scenes and resolve locations.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
CLASS_NAMES = ["Coastal", "Forest", "Mountain", "Rural", "Urban"]

# Model checkpoint paths
CHECKPOINTS = {
    "resnet50": "model_repo/ResNet50/ResNet50_best_model.pth",
    "efficientnet_b0": "model_repo/EfficientNet-B0/EfficientNet-B0_best.pth",
    "vit_b16": "model_repo/ViT-B_16/ViT-B_16_best.pth"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_val_transforms()

# Global variables for loaded models and analyzer
loaded_models = {}
gemini_analyzer = None

def load_all_models():
    """Load the models on startup or on demand."""
    print("=" * 60)
    print(f"Loading DeepSceneLoc Ensemble Models on: {device}")
    print("=" * 60)
    
    # 1. ResNet-50 baseline
    resnet_path = ROOT / CHECKPOINTS["resnet50"]
    if resnet_path.exists():
        try:
            print(f"Loading ResNet-50 from {resnet_path}...")
            model = create_model("resnet50", num_classes=5, pretrained=False)
            ckpt = torch.load(str(resnet_path), map_location=device, weights_only=False)
            key = next((k for k in ["ema_state", "model_state", "model_state_dict"] if k in ckpt), None)
            if key:
                model.load_state_dict(ckpt[key])
            else:
                model.load_state_dict(ckpt)
            model.to(device).eval()
            loaded_models["resnet50"] = model
            print("[OK] ResNet-50 loaded successfully.")
        except Exception as e:
            print(f"[ERR] Failed to load ResNet-50: {e}")
    else:
        print(f"[WARN] ResNet-50 checkpoint not found at {resnet_path}")

    # 2. EfficientNet-B0
    effnet_path = ROOT / CHECKPOINTS["efficientnet_b0"]
    if effnet_path.exists():
        try:
            print(f"Loading EfficientNet-B0 from {effnet_path}...")
            model = create_advanced_model("efficientnet_b0", num_classes=5, pretrained=False)
            ckpt = torch.load(str(effnet_path), map_location=device, weights_only=False)
            key = next((k for k in ["ema_state", "model_state", "model_state_dict"] if k in ckpt), None)
            if key:
                model.load_state_dict(ckpt[key])
            else:
                model.load_state_dict(ckpt)
            model.to(device).eval()
            loaded_models["efficientnet_b0"] = model
            print("[OK] EfficientNet-B0 loaded successfully.")
        except Exception as e:
            print(f"[ERR] Failed to load EfficientNet-B0: {e}")
    else:
        print(f"[WARN] EfficientNet-B0 checkpoint not found at {effnet_path}")

    # 3. ViT-B16
    vit_path = ROOT / CHECKPOINTS["vit_b16"]
    if vit_path.exists():
        try:
            print(f"Loading ViT-B16 from {vit_path}...")
            model = create_advanced_model("vit_b16", num_classes=5, pretrained=False)
            ckpt = torch.load(str(vit_path), map_location=device, weights_only=False)
            key = next((k for k in ["ema_state", "model_state", "model_state_dict"] if k in ckpt), None)
            if key:
                model.load_state_dict(ckpt[key])
            else:
                model.load_state_dict(ckpt)
            model.to(device).eval()
            loaded_models["vit_b16"] = model
            print("[OK] ViT-B16 loaded successfully.")
        except Exception as e:
            print(f"[ERR] Failed to load ViT-B16: {e}")
    else:
        print(f"[WARN] ViT-B16 checkpoint not found at {vit_path}")
        
    print("=" * 60)
    print(f"Ensemble setup complete. Loaded {len(loaded_models)}/3 models.")
    print("=" * 60)


def load_gemini_analyzer():
    """Initialize Stage 2 location analyzer if API key is set."""
    global gemini_analyzer
    if not _GEMINI_OK:
        print("[WARN] Gemini/OpenRouter SDK packages not fully imported.")
        return

    # Check for keys in .env or system environment
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    
    if not openrouter_key and (not gemini_key or gemini_key == "your_gemini_api_key_here"):
        print("[INFO] Stage 2 Location Grounding disabled: No valid API keys in .env")
        return

    try:
        base = GeminiLocationAnalyzer()
        # Wrap with cache manager for sqlite caching of duplicates
        cache_db = str(ROOT / "results/gemini_cache.db")
        Path(cache_db).parent.mkdir(parents=True, exist_ok=True)
        gemini_analyzer = CachedGeminiAnalyzer(base, db_path=cache_db)
        print(f"[OK] Stage 2 location analyzer ready. Mode: {base.mode}")
    except Exception as e:
        print(f"[WARN] Stage 2 initialization failed: {e}")


@app.on_event("startup")
def startup_event():
    load_all_models()
    load_gemini_analyzer()


# Input schema matching frontend JSON payload
class AnalysisRequest(BaseModel):
    imageBase64: str
    presetId: Optional[str] = None


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "available_models": list(loaded_models.keys()),
        "gemini_active": gemini_analyzer is not None,
        "device": str(device),
        "categories": CLASS_NAMES
    }


def predict_ensemble(image: Image.Image) -> dict:
    """Run validation transform, evaluate ensemble models with horizontal flip TTA."""
    if not loaded_models:
        raise HTTPException(
            status_code=500,
            detail="No PyTorch checkpoints loaded on Python backend server."
        )

    # 1. Apply validation preprocessing
    tensor = transform(image).unsqueeze(0).to(device)
    # 2. Horizontal Flip for Test-Time Augmentation (TTA)
    tensor_flipped = torch.flip(tensor, dims=[3])

    all_probs = []
    TEMPERATURE = 0.6  # Consistent scaling

    with torch.no_grad():
        for name, model in loaded_models.items():
            # Original and flipped predictions
            logits = model(tensor)
            logits_flip = model(tensor_flipped)
            logits_avg = (logits + logits_flip) / 2.0
            
            # Apply softmax with temperature
            probs = F.softmax(logits_avg / TEMPERATURE, dim=1).squeeze().cpu().numpy()
            all_probs.append(probs)

    # Uniform averaging of probabilities across loaded ensemble models
    avg_probs = np.mean(all_probs, axis=0)
    top_idx = int(np.argmax(avg_probs))
    
    # Confidence in percentage: 0 to 100
    confidence_pct = round(float(avg_probs[top_idx]) * 100, 2)

    return {
        "sceneCategory": CLASS_NAMES[top_idx],
        "confidence": confidence_pct
    }


@app.post("/api/analyze-image")
async def analyze_image(request: AnalysisRequest):
    t0 = time.perf_counter()
    
    base64_data = request.imageBase64
    
    # Clean up the base64 prefix if the frontend sent it (e.g. data:image/jpeg;base64,...)
    if ";base64," in base64_data:
        base64_data = base64_data.split(";base64,")[1]

    try:
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 or open image: {e}")

    try:
        # Run PyTorch classification
        result = predict_ensemble(image)
        
        # Prepare basic response fields
        response_data = {
            "sceneCategory": result["sceneCategory"],
            "confidence": result["confidence"],
            "landmarkName": "Generic / Unknown",
            "city": "Unknown",
            "country": "Unknown",
            "latitude": 0.0,
            "longitude": 0.0,
            "reasoning": "DeepSceneLoc PyTorch classification successful. Stage 2 location grounding was skipped or not configured.",
            "elevation": "N/A",
            "bestSeason": "N/A",
            "geologicalAge": "N/A",
            "aiConfidence": 0.0
        }

        # Run Stage 2 Location Grounding if enabled
        if gemini_analyzer is not None:
            try:
                print(f"[Stage 2] Submitting image to GeminiLocationAnalyzer with hint: {result['sceneCategory']}...")
                stage2 = gemini_analyzer.analyze_location(
                    image=image,
                    predicted_category=result["sceneCategory"],
                    confidence=result["confidence"] / 100.0  # Decimals as hint
                )
                
                if "error" not in stage2:
                    response_data.update({
                        "landmarkName": stage2.get("exact_location", "Unknown"),
                        "city": stage2.get("city", "Unknown"),
                        "country": stage2.get("country", "Unknown"),
                        "latitude": float(stage2["latitude"]) if stage2.get("latitude") is not None else 0.0,
                        "longitude": float(stage2["longitude"]) if stage2.get("longitude") is not None else 0.0,
                        "reasoning": stage2.get("description", ""),
                        "elevation": stage2.get("additional_info", {}).get("elevation", "N/A") if isinstance(stage2.get("additional_info"), dict) else "N/A",
                        "bestSeason": stage2.get("additional_info", {}).get("best_season", "All Year") if isinstance(stage2.get("additional_info"), dict) else "All Year",
                        "geologicalAge": stage2.get("additional_info", {}).get("geological_age", "Recent Holocene") if isinstance(stage2.get("additional_info"), dict) else "Recent Holocene",
                        "aiConfidence": 95.5 if stage2.get("confidence") == "high" else (85.0 if stage2.get("confidence") == "medium" else 65.0)
                    })
                    print(f"[Stage 2 OK] Resolved: {response_data['landmarkName']} ({response_data['latitude']}, {response_data['longitude']})")
                else:
                    print(f"[Stage 2 ERROR] Analyzer returned an error: {stage2.get('error')}")
                    response_data["reasoning"] = f"Stage 2 error: {stage2.get('error')}"
            except Exception as gem_err:
                print(f"[Stage 2 EXCEPTION] Failed to run location grounding: {gem_err}")
                response_data["reasoning"] = f"Stage 2 exception: {gem_err}"

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        print(f"[API] Pipeline complete: {response_data['sceneCategory']} / {response_data['landmarkName']} in {latency_ms}ms")
        
        return {
            "success": True,
            "data": response_data
        }
    except Exception as e:
        print(f"[API ERROR] Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=5000, reload=False)
