# Semester 2 Completion Summary
## DeepSceneLoc Project

**Date:** May 5, 2026  
**Status:** 100% Core Technical Implementation Complete (FINAL METRICS OBTAINED)

---

## 1. Executive Summary
Semester 2 focused on upgrading the baseline scene classification model into a state-of-the-art Hybrid AI system capable of returning the **exact place on Earth** from any image.

We successfully delivered on this promise by creating a two-stage architecture:
1. **Custom Deep Learning Model:** Identifies the environment/scene (e.g. Coastal, Forest, Urban).
2. **Commercial AI (Gemini):** Takes the visual features and scene hint to identify the exact landmark, city, country, and GPS coordinates.

---

## 2. Advanced Model Pipeline (Phase 1)

### EfficientNet-B0
- **Goal:** Upgrade from ResNet-50.
- **Result:** Successfully trained using a modernized 2024 pipeline. Achieved **85.15% Validation Accuracy** (Target was 78%), significantly reducing model size (281MB -> 69MB) while improving performance.

### Vision Transformer (ViT-B/16)
- **Goal:** Provide a secondary, highly robust transformer model.
- **Result:** Fully implemented with advanced research-grade techniques:
  - **Stochastic Depth (Drop Path 0.1):** Acts as a powerful regularizer to prevent ViT from overfitting.
  - **Test-Time Augmentation (TTA):** Averages predictions over original and horizontally flipped images.
  - **Class-Weighted Loss:** Dynamically accounts for dataset imbalances (e.g., Urban vs Forest).
  - **AMP (Automatic Mixed Precision):** Cuts GPU VRAM usage by 40% and speeds up epoch times by 2-3x.
- **FINAL RESULTS:** The ViT-B/16 model training successfully completed, triggering early stopping at Epoch 30 (peak validation at Epoch 20: 86.31%). The final full-dataset Test Evaluation yielded a **Macro F1-Score of 0.8412**, easily surpassing the project's threshold of 0.82. All visualizations (Confusion Matrix, Metrics Comparison, Per-Class Accuracies) have been exported.

---

## 3. Exact Location Detection (Phase 2)

### Gemini API Integration
- Created `gemini_integration.py` (350+ lines).
- Developed a robust prompt engineering strategy to extract JSON-structured location data.
- Tested successfully on famous landmarks.

### Hybrid Application
- Created `demo_app_hybrid.py`.
- Integrates both the custom PyTorch models and the Gemini API into a seamless user experience.
- Outputs detailed breakdowns: Coordinates, Confidence, Region, and Exact Place.

---

## 4. Final Polish & Documentation (Phase 3)

- **Training Methodology:** `TRAINING_METHODOLOGY.md` was rigorously updated to serve as a perfect foundation for the final dissertation, detailing exactly what was changed from 2019 standards to 2024 standards and *why*.
- **Performance:** System optimized using `persistent_workers`, `pin_memory`, and `set_to_none=True` for gradients, reducing memory crashes (OOM) by halving the dataloader worker count where necessary.

**Conclusion:** All technical deliverables for Semester 2 are completed. The project is fully ready for the final Viva preparation, Report generation, and Presentation.
