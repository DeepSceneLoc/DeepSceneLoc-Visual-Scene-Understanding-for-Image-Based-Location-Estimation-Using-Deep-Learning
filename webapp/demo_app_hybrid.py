"""
DeepSceneLoc — Hybrid Demo Application (Week 12)
Two-Stage AI System: Scene Classification + Exact Location Detection

Authors:
    Krishan Yadav  (System Integration Lead)
    Jensi Paneliya (Demo & UX Lead)
    Anuj Kondawar  (Pipeline & Optimization Lead)
    Aditi Sah      (Evaluation & Testing Lead)

Architecture:
    Stage 1: Scene Classifier (EfficientNet-B0 / ResNet-50 / ViT)
              → Identifies: Urban | Rural | Coastal | Mountain | Forest
    Stage 2: Gemini Vision API
              → Identifies: Exact landmark, GPS coordinates, city/country

Usage:
    python demo_app_hybrid.py

Optional env vars:
    GEMINI_API_KEY  — required for Stage 2 exact location detection
    MODEL_PATH      — path to trained PyTorch checkpoint
    MODEL_ARCH      — model architecture (efficientnet_b0 | resnet50 | vit_b16)

Live demo at: http://localhost:7860
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Model imports
try:
    from src.models.model_advanced import create_advanced_model
    _ADVANCED_OK = True
except ImportError:
    _ADVANCED_OK = False

try:
    from src.models.model import create_model as create_baseline_model
    _BASELINE_OK = True
except ImportError:
    _BASELINE_OK = False

# ── Preprocessing
try:
    from src.preprocessing.transforms import get_val_transforms
    _TRANSFORMS_OK = True
except ImportError:
    _TRANSFORMS_OK = False

# ── Gemini integration
try:
    from src.utils.gemini_integration import GeminiLocationAnalyzer
    from src.utils.cache_manager import CachedGeminiAnalyzer
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False

# ── Gradio
try:
    import gradio as gr
    _GRADIO_OK = True
except ImportError:
    _GRADIO_OK = False
    print("[ERROR] Gradio not installed. Run: pip install gradio")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

CLASS_NAMES = ["Coastal", "Forest", "Mountain", "Rural", "Urban"]

# Category description for display
CLASS_DESCRIPTIONS = {
    "Coastal":  "🏖️ Coastal — Beaches, shorelines, cliffs, ocean/sea views",
    "Forest":   "🌲 Forest  — Dense woodland, jungle, tree canopy",
    "Mountain": "⛰️ Mountain — High peaks, rocky terrain, alpine scenery",
    "Rural":    "🌾 Rural   — Farmland, countryside, open fields, villages",
    "Urban":    "🏙️ Urban   — City streets, buildings, skylines, infrastructure",
}

# ── Default paths
# Primary: model_repo (recommended), Fallback: local checkpoints
MODEL_SEARCH_PATHS = {
    "efficientnet_b0": [
        "model_repo/EfficientNet-B0/EfficientNet-B0_best.pth",
        "models/checkpoints/efficientnet/EfficientNet-B0_best.pth",
    ],
    "vit_b16": [
        "model_repo/ViT-B_16/ViT-B_16_best.pth",
        "models/checkpoints/vit/ViT-B_16_epoch030.pth",
    ],
    "resnet50": [
        "model_repo/ResNet50/ResNet50_best_model.pth",
        "models/checkpoints/resnet/best_model.pth",
    ],
}
DEFAULT_ARCH = os.getenv("MODEL_ARCH", "efficientnet_b0")
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", MODEL_SEARCH_PATHS.get(DEFAULT_ARCH, ["models/checkpoints/efficientnet/EfficientNet-B0_best.pth"])[0])
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ─────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────

def _load_model(
    arch: str = DEFAULT_ARCH,
    model_path: Optional[str] = DEFAULT_MODEL_PATH,
    device: str = "cpu",
) -> Optional[torch.nn.Module]:
    """Load the scene classifier from disk. Returns None on failure."""
    try:
        if arch in ("efficientnet_b0", "vit_b16") and _ADVANCED_OK:
            model = create_advanced_model(arch, num_classes=5, pretrained=False)
        elif _BASELINE_OK:
            model = create_baseline_model("resnet50", num_classes=5, pretrained=False)
        else:
            print("[WARN] No model factories available.")
            return None

        # Try model_path first, then search paths for the architecture
        search_paths = [model_path] if model_path else []
        if arch in MODEL_SEARCH_PATHS:
            search_paths.extend(MODEL_SEARCH_PATHS[arch])
        
        loaded = False
        for p_str in search_paths:
            p = Path(p_str)
            if p.exists():
                try:
                    ckpt = torch.load(str(p), map_location=device, weights_only=False)
                    # Support multiple key formats
                    if "ema_state" in ckpt:
                        key = "ema_state"
                    elif "model_state" in ckpt:
                        key = "model_state"
                    else:
                        key = "model_state_dict"
                    model.load_state_dict(ckpt[key])
                    print(f"[OK]   Model loaded from {p}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load from {p}: {e}, trying next...")
                    continue
        
        if not loaded:
            print(f"[WARN] No checkpoint found in search paths. Using random weights (demo mode).")

        model.to(device).eval()
        return model

    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return None


def _get_transforms():
    if _TRANSFORMS_OK:
        return get_val_transforms()
    from torchvision import transforms as T
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────
# Global state (loaded once at startup)
# ─────────────────────────────────────────────────────────────

device    = "cuda" if torch.cuda.is_available() else "cpu"
transform = _get_transforms()
model     = _load_model(device=device)

# Gemini
_gemini_analyzer = None
if GEMINI_API_KEY and _GEMINI_OK:
    try:
        _base     = GeminiLocationAnalyzer(api_key=GEMINI_API_KEY)
        _gemini_analyzer = CachedGeminiAnalyzer(
            _base,
            db_path="results/gemini_cache.db",
        )
        print("[OK]   Gemini AI initialized with SQLite cache.")
    except Exception as e:
        print(f"[WARN] Gemini init failed: {e}")


# ─────────────────────────────────────────────────────────────
# Stage 1: Scene Classification
# ─────────────────────────────────────────────────────────────

def predict_scene(image: Image.Image) -> Tuple[str, float, dict]:
    """
    Run Stage 1 scene classification.

    Returns
    -------
    top_class : str
    top_conf  : float  (0–1)
    conf_dict : dict   {class_name: float}
    """
    if model is None:
        # Demo mode: return mock predictions
        probs = np.random.dirichlet([3, 1, 1, 1, 1])
        conf_dict = {n: float(p) for n, p in zip(CLASS_NAMES, probs)}
        top_class = max(conf_dict, key=conf_dict.get)
        return top_class, conf_dict[top_class], conf_dict

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    conf_dict = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
    top_idx   = int(np.argmax(probs))
    return CLASS_NAMES[top_idx], float(probs[top_idx]), conf_dict


# ─────────────────────────────────────────────────────────────
# Stage 2: Gemini Location Detection
# ─────────────────────────────────────────────────────────────

def get_exact_location(
    image: Image.Image,
    category: str,
    confidence: float,
) -> dict:
    """
    Run Stage 2 — Gemini Vision API for exact location identification.
    Falls back gracefully if Gemini is unavailable.
    """
    if _gemini_analyzer is None:
        return {
            "exact_location": "Gemini API not configured",
            "confidence": "none",
            "description": (
                "Set GEMINI_API_KEY environment variable to enable exact location detection.\n"
                f"Stage 1 result: {category} ({confidence:.1%} confidence)"
            ),
            "country": "—",
            "city": "—",
            "latitude": None,
            "longitude": None,
            "landmarks": [],
        }

    try:
        result = _gemini_analyzer.analyze_location(
            image=image,
            predicted_category=category,
            confidence=confidence,
        )
        return result
    except Exception as e:
        return {
            "exact_location": "Error",
            "confidence": "none",
            "description": f"Gemini API error: {str(e)}",
            "country": "—",
            "city": "—",
            "latitude": None,
            "longitude": None,
            "landmarks": [],
        }


# ─────────────────────────────────────────────────────────────
# Main inference function (called by Gradio)
# ─────────────────────────────────────────────────────────────

def analyze_image(
    image: Image.Image,
    run_gemini: bool = True,
) -> Tuple[str, str, str]:
    """
    Full two-stage analysis pipeline.

    Returns
    -------
    stage1_result : str   (formatted HTML)
    stage2_result : str   (formatted HTML)
    map_url       : str   (Google Maps URL or empty string)
    """
    if image is None:
        return (
            "⚠️ Please upload an image to analyze.",
            "—",
            "",
        )

    t0 = time.perf_counter()

    # ── Stage 1
    top_class, top_conf, conf_dict = predict_scene(image)
    t1 = time.perf_counter()

    stage1_html = _format_stage1(top_class, top_conf, conf_dict, latency_ms=(t1 - t0) * 1000)

    # ── Stage 2
    if run_gemini:
        gemini_result = get_exact_location(image, top_class, top_conf)
    else:
        gemini_result = {
            "exact_location": "Gemini skipped (checkbox unchecked)",
            "confidence": "none",
            "description": f"Scene classified as: **{top_class}** ({top_conf:.1%})",
            "country": "—", "city": "—",
            "latitude": None, "longitude": None,
            "landmarks": [],
        }
    t2 = time.perf_counter()

    stage2_html = _format_stage2(gemini_result, latency_ms=(t2 - t1) * 1000)

    # ── Map URL
    lat = gemini_result.get("latitude")
    lon = gemini_result.get("longitude")
    map_url = f"https://www.google.com/maps?q={lat},{lon}" if lat and lon else ""

    return stage1_html, stage2_html, map_url


# ─────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────

def _format_stage1(
    top_class: str, top_conf: float, conf_dict: dict, latency_ms: float
) -> str:
    conf_bar = _confidence_bars(conf_dict, top_class)
    desc     = CLASS_DESCRIPTIONS.get(top_class, top_class)
    status   = "HIGH ✓" if top_conf >= 0.65 else ("MEDIUM ~" if top_conf >= 0.40 else "LOW ⚠️")

    return (
        f"### 🔍 Stage 1: Scene Classification\n\n"
        f"**Predicted Category:** {desc}\n\n"
        f"**Confidence:** {top_conf:.1%} ({status})\n\n"
        f"**Inference time:** {latency_ms:.0f} ms\n\n"
        f"---\n\n"
        f"**All class probabilities:**\n\n{conf_bar}"
    )


def _format_stage2(result: dict, latency_ms: float) -> str:
    location   = result.get("exact_location", "Unknown")
    confidence = result.get("confidence", "none")
    country    = result.get("country", "—")
    city       = result.get("city", "—")
    region     = result.get("region", "—")
    landmark   = ", ".join(result.get("landmarks", [])) or "None detected"
    description = result.get("description", "")
    lat        = result.get("latitude")
    lon        = result.get("longitude")
    from_cache = result.get("_from_cache", False)

    cache_tag = " *(cached)*" if from_cache else ""
    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")

    coords = f"{lat:.4f}°N, {lon:.4f}°E" if lat and lon else "Unknown"
    maps_link = f"[View on Google Maps](https://www.google.com/maps?q={lat},{lon})" if lat and lon else ""

    return (
        f"### 🌍 Stage 2: Exact Location (Gemini AI){cache_tag}\n\n"
        f"**Location:** {location}\n\n"
        f"**Confidence:** {conf_icon} {confidence.upper()}\n\n"
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| Country | {country} |\n"
        f"| City / Region | {city} / {region} |\n"
        f"| Coordinates | {coords} |\n"
        f"| Landmarks | {landmark} |\n"
        f"| Latency | {latency_ms:.0f} ms |\n\n"
        f"{maps_link}\n\n"
        f"---\n\n"
        f"**Gemini Analysis:**\n\n{description}"
    )


def _confidence_bars(conf_dict: dict, top_class: str) -> str:
    lines = []
    for name, prob in sorted(conf_dict.items(), key=lambda x: -x[1]):
        bar_len = int(prob * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        marker  = " ← top" if name == top_class else ""
        lines.append(f"`{name:<12}` [{bar}] {prob:.1%}{marker}")
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Example images (bundled for demo)
# ─────────────────────────────────────────────────────────────

EXAMPLE_IMAGES_DIR = ROOT / "data" / "examples"

def _find_examples() -> list:
    if not EXAMPLE_IMAGES_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [
        str(p) for p in EXAMPLE_IMAGES_DIR.iterdir()
        if p.suffix.lower() in exts
    ][:6]


# ─────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    model_status = (
        "✅ Model loaded from checkpoint" if (model is not None and Path(DEFAULT_MODEL_PATH).exists())
        else "⚠️ Demo mode (random weights — upload your checkpoint)"
    )
    provider = getattr(_gemini_analyzer.analyzer if hasattr(_gemini_analyzer, 'analyzer') else _gemini_analyzer, 'mode', 'AI')
    gemini_status = (
        f"✅ Gemini ready ({provider.title()} via OpenRouter)" if provider == "openrouter"
        else ("✅ Gemini ready (Native SDK)" if _gemini_analyzer is not None
        else "⚠️ Gemini not configured (set OPENROUTER_API_KEY or GEMINI_API_KEY)")
    )

    with gr.Blocks(
        title="DeepSceneLoc — Hybrid AI Location System",
        theme=gr.themes.Soft(primary_hue="indigo"),
    ) as demo:

        # Header
        gr.Markdown("""
# 🌏 DeepSceneLoc — Hybrid AI Location System
### Visual Scene Understanding + Exact Place Identification

**Two-Stage Pipeline:**
1. 🧠 **Stage 1 — Scene Classifier** (EfficientNet-B0 / ResNet-50 / ViT): identifies scene category
2. 🌍 **Stage 2 — Gemini Vision AI**: identifies the exact place on Earth

---
""")

        # System status
        with gr.Row():
            gr.Markdown(f"**Stage 1 Model:** {model_status}")
            gr.Markdown(f"**Stage 2 Gemini:** {gemini_status}")

        gr.Markdown("---")

        # Main layout
        with gr.Row():
            # Left — input
            with gr.Column(scale=1):
                img_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    sources=["upload", "clipboard", "webcam"],
                    height=360,
                )
                run_gemini_toggle = gr.Checkbox(
                    label="Enable Stage 2 (Gemini Exact Location)",
                    value=True,
                    info="Uncheck to run Scene Classification only (faster, no API key needed)",
                )
                analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                clear_btn   = gr.Button("🗑️ Clear", variant="secondary")

                examples = _find_examples()
                if examples:
                    gr.Examples(examples=examples, inputs=img_input, label="Example Images")

            # Right — output
            with gr.Column(scale=1):
                stage1_out = gr.Markdown(label="Stage 1 — Scene Classification")
                stage2_out = gr.Markdown(label="Stage 2 — Exact Location (Gemini)")
                map_link   = gr.Markdown(label="Map Link")

        gr.Markdown("---")

        # Technical details accordion
        with gr.Accordion("ℹ️ Technical Details", open=False):
            gr.Markdown(f"""
**Architecture:** Two-stage hybrid system (DeepSceneLoc Semester 2, Week 12)

| Component | Details |
|-----------|---------|
| Stage 1 Model | EfficientNet-B0 (5.3M params, 224×224 input) |
| Stage 1 Task  | 5-class scene classification (Coastal / Forest / Mountain / Rural / Urban) |
| Stage 2 Model | Google Gemini 1.5-Flash (Vision API) |
| Stage 2 Task  | Exact landmark + GPS coordinates + city/country |
| Caching       | SQLite LRU cache (perceptual hash deduplication, 7-day TTL) |
| Confidence gate | Gemini called when Stage 1 confidence ≥ 60% |

**Training:** Places365 dataset → mapped to 5 categories → ResNet-50 / EfficientNet-B0 / ViT-B/16

**Team:** Krishan Yadav · Anuj Kondawar · Aditi Sah · Jensi Paneliya
""")

        # Events
        def start_analysis():
            """Disable UI components during analysis"""
            return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

        def run_analysis(img, use_gemini):
            """Core analysis logic with UI unlocking at the end"""
            if img is None:
                return (
                    "⚠️ Please upload an image to analyze.", "—", "", 
                    gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
                )
            
            s1, s2, maps = analyze_image(img, run_gemini=use_gemini)
            map_md = f"🗺️ [Open in Google Maps]({maps})" if maps else ""
            
            # Return results and re-enable UI
            return (
                s1, s2, map_md, 
                gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
            )

        def reset_ui():
            """Reset all fields and enable inputs"""
            return (
                None, "", "", "", 
                gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
            )

        # Event chain for analysis: Lock -> Process -> Unlock
        analyze_btn.click(
            fn=start_analysis,
            outputs=[img_input, analyze_btn, clear_btn]
        ).then(
            fn=run_analysis,
            inputs=[img_input, run_gemini_toggle],
            outputs=[stage1_out, stage2_out, map_link, img_input, analyze_btn, clear_btn],
            show_progress=True,
        )

        clear_btn.click(
            fn=reset_ui,
            outputs=[img_input, stage1_out, stage2_out, map_link, img_input, analyze_btn, clear_btn],
        )

    return demo


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  DeepSceneLoc — Hybrid Demo Application")
    print("  Two-Stage AI: Scene Classification + Exact Location")
    print("=" * 65)
    print(f"  Device   : {device}")
    print(f"  Model    : {DEFAULT_ARCH}")
    print(f"  Gemini   : {'enabled' if _gemini_analyzer else 'disabled (set GEMINI_API_KEY)'}")
    print("=" * 65 + "\n")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
