# Results Tracking and Metrics Log

**Project Purpose:** Get the exact place on Earth from any image

**Project Structure:** 16 weeks total (Semester 1: Weeks 1-7 scene classification + Semester 2: Weeks 8-16 exact location detection)

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/></svg> Overview

This document serves as a central log for all project results, metrics, and findings throughout the 16-week project timeline.

**Project Phases:**
- **Weeks 1-7 (Semester 1):** Scene classification foundation (Urban, Rural, Coastal, Mountain, Forest)
- **Weeks 8-10 (Semester 2):** Infrastructure setup and hybrid system development
- **Weeks 11-13 (CRITICAL):** Gemini AI integration for exact location detection
- **Weeks 14-16 (Semester 2):** Final testing, documentation, and presentation

**Current Status (as of February 27, 2026):**
- Weeks 1-4: <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Complete (Sign-offs obtained)
- Week 5: <svg width="12" height="12" fill="orange"><circle cx="6" cy="6" r="5"/></svg> In Progress (Requesting sign-off)
- Weeks 6-16: Pending

---

## Results Tracking Template

This document serves as a central log for all project results, metrics, and findings throughout both semesters.

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M1 2.5A1.5 1.5 0 0 1 2.5 1h3A1.5 1.5 0 0 1 7 2.5v3A1.5 1.5 0 0 1 5.5 7h-3A1.5 1.5 0 0 1 1 5.5v-3z"/></svg> Semester 1 Results (Weeks 1-7)

**Purpose:** Build scene classification foundation for exact location detection

**Contribution:** 30% of project (scene categorization as Stage 1 of hybrid system)

### Week 5-6: Baseline Model (ResNet-50)

#### Model Configuration
- Model: ResNet-50 (Pretrained on ImageNet)
- Input Size: 224x224
- Output Classes: 5 (Urban, Rural, Coastal, Mountain, Forest)
- Frozen Layers: Layer 1, 2, 3
- Fine-tuned Layers: Layer 4, 5, Classification head
- Training Epochs: 40
- Batch Size: 32

#### Training Metrics (To be filled)
```
Epoch 1:    Training Loss: ___  Validation Accuracy: ___
Epoch 10:   Training Loss: ___  Validation Accuracy: ___
Epoch 20:   Training Loss: ___  Validation Accuracy: ___
Epoch 30:   Training Loss: ___  Validation Accuracy: ___
Epoch 40:   Training Loss: ___  Validation Accuracy: ___

Final Training Loss: ___
Final Validation Accuracy: ___
```

#### Test Set Performance (To be filled)
```
Test Accuracy: ___

Per-Class Accuracy:
- Urban:    ___
- Rural:    ___
- Coastal:  ___
- Mountain: ___
- Forest:   ___

Per-Class Precision:
- Urban:    ___
- Rural:    ___
- Coastal:  ___
- Mountain: ___
- Forest:   ___

Per-Class Recall:
- Urban:    ___
- Rural:    ___
- Coastal:  ___
- Mountain: ___
- Forest:   ___

Per-Class F1-Score:
- Urban:    ___
- Rural:    ___
- Coastal:  ___
- Mountain: ___
- Forest:   ___
```

#### Training Statistics
```
Training Time: ___ hours ___ minutes
Inference Time per Image: ___ ms
Model Size on Disk: ___ MB
Total Parameters: 25.5M
Trainable Parameters: ___ M
```

#### Confusion Matrix
```
                Urban   Rural   Coastal Mountain Forest
Urban           ___     ___     ___     ___      ___
Rural           ___     ___     ___     ___      ___
Coastal         ___     ___     ___     ___      ___
Mountain        ___     ___     ___     ___      ___
Forest          ___     ___     ___     ___      ___
```

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><circle cx="8" cy="8" r="6" fill="orange"/></svg> Semester 2 Results (Weeks 8-16)

**Purpose:** Get the exact place on Earth from any image

**Contribution:** 70% of project (hybrid two-stage system for exact location)

**System Architecture:**
- **Stage 1:** Scene Classification (ResNet-50 from Semester 1)
- **Stage 2:** Exact Location Detection (Gemini AI integration Weeks 11-13)

**Key Enhancement (Weeks 11-13):**
Integration of Google Gemini AI to provide:
- Specific landmarks identification
- City and country information
- GPS coordinates estimation
- Confidence levels for location predictions

---

### Weeks 8-10: Advanced Models & Hybrid System Development

#### Week 1-2: EfficientNet-B0 (Optional Enhancement)

##### Model Configuration
- Model: EfficientNet-B0 (Pretrained on ImageNet)
- Input Size: 224x224
- Output Classes: 5
- Total Parameters: 5.3M
- Frozen Blocks: 1-4
- Fine-tuned Blocks: 5-7 + Classification head
- Training Epochs: 40
- Batch Size: 32

##### Training Metrics (To be filled)
```
Epoch 1:    Training Loss: ___  Validation Accuracy: ___
Epoch 10:   Training Loss: ___  Validation Accuracy: ___
Epoch 20:   Training Loss: ___  Validation Accuracy: ___
Epoch 30:   Training Loss: ___  Validation Accuracy: ___
Epoch 40:   Training Loss: ___  Validation Accuracy: ___

Final Training Loss: ___
Final Validation Accuracy: ___
```

##### Test Set Performance (To be filled)
```
Test Accuracy: ___

Per-Class Metrics:
Urban:      Accuracy=___, Precision=___, Recall=___, F1=___
Rural:      Accuracy=___, Precision=___, Recall=___, F1=___
Coastal:    Accuracy=___, Precision=___, Recall=___, F1=___
Mountain:   Accuracy=___, Precision=___, Recall=___, F1=___
Forest:     Accuracy=___, Precision=___, Recall=___, F1=___
```

---

#### Week 1-2: Vision Transformer (ViT-Base) (Optional Enhancement)

##### Model Configuration
- Model: Vision Transformer Base (Pretrained on ImageNet-21k)
- Input Size: 224x224
- Patch Size: 16x16
- Output Classes: 5
- Total Parameters: 86M
- Frozen Components: Patch embedding, Blocks 1-10
- Fine-tuned Components: Blocks 11-12, Classification head
- Training Epochs: 40
- Batch Size: 32

##### Training Metrics (To be filled)
```
Epoch 1:    Training Loss: ___  Validation Accuracy: ___
Epoch 10:   Training Loss: ___  Validation Accuracy: ___
Epoch 20:   Training Loss: ___  Validation Accuracy: ___
Epoch 30:   Training Loss: ___  Validation Accuracy: ___
Epoch 40:   Training Loss: ___  Validation Accuracy: ___

Final Training Loss: ___
Final Validation Accuracy: ___
```

##### Test Set Performance (To be filled)
```
Test Accuracy: ___

Per-Class Metrics:
Urban:      Accuracy=___, Precision=___, Recall=___, F1=___
Rural:      Accuracy=___, Precision=___, Recall=___, F1=___
Coastal:    Accuracy=___, Precision=___, Recall=___, F1=___
Mountain:   Accuracy=___, Precision=___, Recall=___, F1=___
Forest:     Accuracy=___, Precision=___, Recall=___, F1=___
```

---

### Weeks 11-13: AI Integration Results (CRITICAL PHASE)

**System:** Hybrid Two-Stage Architecture
- **Stage 1:** Scene Classification (ResNet-50)
- **Stage 2:** Exact Location Detection (Gemini AI)

#### Gemini AI Integration Metrics (To be filled)

##### Location Detection Accuracy
```
Famous Landmarks:
  - Correct Location: ____%
  - Correct City: ____%
  - Correct Country: ____%
  - Average Confidence: ___

Generic Scenes:
  - Reasonable Region: ____%
  - Confidence Level: Low/Medium/High
  - Fallback to Scene Category: ____%

Overall Location Accuracy:
  - Exact Location (within 1km): ____%
  - City-level Accuracy: ____%
  - Country-level Accuracy: ____%
  - Region-level Accuracy: ____%
```

##### Hybrid System Performance
```
Stage 1 (Scene Classification) Time: ___ ms
Stage 2 (Gemini AI) Time: ___ ms
Total Pipeline Time: ___ ms

API Usage Statistics:
  - Total API Calls: ___
  - Successful Responses: ____%
  - API Response Time (avg): ___ ms
  - Cost per Image: $___
  - Total Cost: $___
```

##### Example Results
```
Test Case 1: Eiffel Tower
  Stage 1: Urban (85% confidence)
  Stage 2: Eiffel Tower, Paris, France
  GPS: 48.8584°N, 2.2945°E
  Result: <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Correct

Test Case 2: Generic Beach
  Stage 1: Coastal (92% confidence)
  Stage 2: Coastal region (confidence: MEDIUM)
  Result: Falls back to scene category

Test Case 3: Taj Mahal
  Stage 1: Urban (78% confidence)
  Stage 2: Taj Mahal, Agra, India
  GPS: 27.1751°N, 78.0421°E
  Result: <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Correct
```

---

### Weeks 14-15: Final Testing & Comparative Analysis

### Phase 2: Comparative Analysis

#### Model Comparison Summary (To be filled)

| Metric | ResNet-50 | EfficientNet | ViT |
|--------|-----------|--------------|-----|
| Test Accuracy | ___% | ___% | ___% |
| Training Time/Epoch | ___ min | ___ min | ___ min |
| Inference Time | ___ ms | ___ ms | ___ ms |
| Parameters | 25.5M | 5.3M | 86M |
| Model Size | ___ MB | ___ MB | ___ MB |
| Top-1 Accuracy | ___% | ___% | ___% |
| Top-3 Accuracy | ___% | ___% | ___% |
| Top-5 Accuracy | ___% | ___% | ___% |
| Overfitting Gap | ___% | ___% | ___% |

#### Winner Analysis
```
Best Overall Accuracy: _______________ Model
Best Speed: _______________ Model
Best Efficiency: _______________ Model (Parameters vs Accuracy)
Recommended for Production: _______________ Model

Justification:
_________________________________________________________________
_________________________________________________________________
```

---

### Week 15: Embedding Analysis

#### Embedding Quality Metrics (To be filled)

##### PCA Analysis
```
Explained Variance (2 components): ____%
Silhouette Score: ___
Within-class Variance: ___
Between-class Variance: ___
Separability Index: ___
```

##### t-SNE Analysis
```
Perplexity Used: 30
Iterations: 1000
Final KL Divergence: ___
Visual Clustering Quality: Good / Moderate / Poor
```

##### Similarity Metrics (To be filled)
```
Average In-Class Similarity: ___
Average Cross-Class Similarity: ___
Ratio (In/Cross): ___
Silhouette Coefficient: ___
Davies-Bouldin Index: ___
```

---

### Week 15: Top-K Analysis

#### Top-K Accuracy Results (To be filled)
```
Top-1 Accuracy: ____%
Top-3 Accuracy: ____%
Top-5 Accuracy: ____%

Per-Category Top-K:
Urban:      Top-1=___%, Top-3=___%, Top-5=___%
Rural:      Top-1=___%, Top-3=___%, Top-5=___%
Coastal:    Top-1=___%, Top-3=___%, Top-5=___%
Mountain:   Top-1=___%, Top-3=___%, Top-5=___%
Forest:     Top-1=___%, Top-3=___%, Top-5=___%
```

#### Confidence Analysis (To be filled)
```
Average Model Confidence: ___
Confidence When Correct: ___
Confidence When Incorrect: ___
Expected Calibration Error: ___
Model Well-Calibrated: Yes / No
```

---

### Week 15: Error Analysis

#### Error Rate Summary (To be filled)
```
Total Test Samples: ____
Total Errors: ____
Error Rate: ____%

Top 5 Misclassification Patterns:
1. _______________ classified as _______________ (___%)
2. _______________ classified as _______________ (___%)
3. _______________ classified as _______________ (___%)
4. _______________ classified as _______________ (___%)
5. _______________ classified as _______________ (___%)
```

#### Per-Category Error Analysis (To be filled)
```
Urban Errors:
  - Most confused with: _______________
  - Error rate: ____%
  - Common characteristics: _____________________________

Rural Errors:
  - Most confused with: _______________
  - Error rate: ____%
  - Common characteristics: _____________________________

Coastal Errors:
  - Most confused with: _______________
  - Error rate: ____%
  - Common characteristics: _____________________________

Mountain Errors:
  - Most confused with: _______________
  - Error rate: ____%
  - Common characteristics: _____________________________

Forest Errors:
  - Most confused with: _______________
  - Error rate: ____%
  - Common characteristics: _____________________________
```

---

### Week 15: Optimization Results

#### Hyperparameter Tuning Results (To be filled)
```
ResNet-50 Optimization:
  Initial Accuracy: ____%
  Final Accuracy: ____%
  Improvement: ____%
  Best Configuration: _____________________________

EfficientNet-B0 Optimization:
  Initial Accuracy: ____%
  Final Accuracy: ____%
  Improvement: ____%
  Best Configuration: _____________________________

Vision Transformer Optimization:
  Initial Accuracy: ____%
  Final Accuracy: ____%
  Improvement: ____%
  Best Configuration: _____________________________
```

---

### Week 16: Final Results & Presentation

#### Final Performance Metrics (To be filled)

```
FINAL TEST SET RESULTS:

=== STAGE 1: SCENE CLASSIFICATION ===
Overall Accuracy: ____%

Per-Class Performance:
Category    Accuracy  Precision  Recall   F1-Score  Support
Urban       ____%     ____%      ____%    ____      ____
Rural       ____%     ____%      ____%    ____      ____
Coastal     ____%     ____%      ____%    ____      ____
Mountain    ____%     ____%      ____%    ____      ____
Forest      ____%     ____%      ____%    ____      ____

Weighted Avg ____%     ____%      ____%    ____      ____
Macro Avg   ____%     ____%      ____%    ____      ____

Top-K Accuracy:
Top-1: ____%
Top-3: ____%
Top-5: ____%

Model Statistics:
Best Model: _______________
Total Training Time: ___ hours
Total Inference Time (Test Set): ___ seconds
Average Inference Time: ___ ms per image

=== STAGE 2: EXACT LOCATION DETECTION ===
Location Accuracy (Famous Landmarks): ____%
City-Level Accuracy: ____%
Country-Level Accuracy: ____%
Average Gemini Response Time: ___ ms
Total API Calls: ___
Total API Cost: $___

=== HYBRID SYSTEM PERFORMANCE ===
End-to-End Accuracy: ____%
Total Pipeline Time: ___ ms per image
System Reliability: ____%
User Satisfaction Score: ___ / 10
```

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M2 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2H2z"/></svg> Visualization Artifacts

### To Be Generated and Saved:

Semester 1 (Weeks 1-7):
- [ ] Training/Validation accuracy curve
- [ ] Training/Validation loss curve
- [ ] Confusion matrix heatmap
- [ ] Per-class accuracy bar chart

Semester 2 (Weeks 8-16):
- [ ] Hybrid system architecture diagram
- [ ] Two-stage pipeline visualization
- [ ] Gemini API integration flow
- [ ] Location detection accuracy by landmark type
- [ ] Model accuracy comparison (3 models)
- [ ] Training speed comparison
- [ ] Model efficiency scatter plot (parameters vs accuracy)
- [ ] Inference speed comparison
- [ ] Learning curves overlay (all 3 models)
- [ ] PCA embedding visualization
- [ ] t-SNE embedding visualization
- [ ] Similarity heatmap (category-to-category)
- [ ] Top-K accuracy comparison
- [ ] Confidence score distribution
- [ ] Calibration curve
- [ ] Confusion matrix (best model)
- [ ] Per-class F1-score comparison
- [ ] Error gallery (worst predictions)

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M2 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2H2z"/></svg> Key Observations and Notes

### Semester 1 Observations (Weeks 1-7):
```
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________
```

### Semester 2 Observations (Weeks 8-16):
```
Weeks 8-10: Hybrid system development and infrastructure setup
Weeks 11-13: Gemini AI integration - exact location detection capability
Weeks 14-16: Final testing, optimization, and presentation preparation

Key Achievement: Successfully combined scene classification with AI-powered 
location detection to get the exact place on Earth from any image.
_________________________________________________________________
```

### AI Integration Notes (Weeks 11-13):
```
Gemini API Integration:
- Model: gemini-1.5-flash
- Purpose: Exact location detection from scene images
- Input: Image + Scene category from Stage 1
- Output: Landmark name, city, country, GPS coordinates, confidence level

Challenges:
_________________________________________________________________

Solutions:
_________________________________________________________________
```

### Unexpected Results:
```
_________________________________________________________________
_________________________________________________________________
```

### Future Improvements:
```
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________
```

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/></svg> Reproducibility Information

### Environment Details
- Python Version: 3.8 / 3.9 / 3.10
- PyTorch Version: 1.x / 2.x
- CUDA Version: 11.x / 12.x
- GPU Used: _______________
- Google Colab Runtime: Yes / No

### Random Seeds
- PyTorch Seed: 42
- NumPy Seed: 42
- Python Random Seed: 42

### API Configuration (Weeks 11-13)
- Gemini Model: gemini-1.5-flash
- API Key: Environment variable (GEMINI_API_KEY)
- Rate Limiting: Applied
- Retry Logic: Implemented

### Dataset Version
- Places365 Subset Version: _____
- Dataset Download Date: _____
- Total Images Used: _____
- Train/Val/Test Split: 70/15/15

---

## References to Result Documents
- Training logs: results/training_logs/
- Evaluation metrics: results/metrics/
- Visualizations: results/visualizations/
- Models: models/

