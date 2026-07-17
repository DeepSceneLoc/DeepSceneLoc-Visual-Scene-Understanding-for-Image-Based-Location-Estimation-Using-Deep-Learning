# New Backend Architecture Plan

This document outlines the planned future architecture for the Python backend, aligning perfectly with the React frontend's expectations (as defined in `frontend/src/ArchitectureGraph.tsx`).

## Execution Flow Split
The execution is split between Python and Node.js:

### 1. Python Backend (Port 5000)
We will create a new, clean Python server (using FastAPI or Flask, running on port 5000) that exposes `/api/analyze-image`. It will execute the PyTorch pipeline:
- **STAGE 1:** Feed the image through **ResNet-50** for coarse scene classification.
- **STAGE 2 (Vision):** Feed the image through **EfficientNet-B0** (for high-resolution feature extraction) and **ViT-B16** (for attention maps).
- **Fusion:** The Python backend will combine the logits/probabilities from these three models to determine the final, highly accurate `sceneCategory` and `confidence`.
- **Response:** The Python backend will return `{"success": true, "data": {"sceneCategory": "...", "confidence": ...}}` to the Node.js server.

### 2. Node.js Frontend (process.env.PORT || 3000)
- Receives the `sceneCategory` constraint from the Python backend.
- **STAGE 2 (LLM):** Executes the Gemini AI prompt, passing it the image and the PyTorch-derived `sceneCategory`.
- Formats the final geographic coordinates and returns them to the React UI.
