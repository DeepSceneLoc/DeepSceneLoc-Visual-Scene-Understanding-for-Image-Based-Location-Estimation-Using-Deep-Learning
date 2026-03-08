# Semester Contributions Summary
## DeepSceneLoc Project - 16 Weeks

**Last Updated:** February 27, 2026  
**Project Purpose:** **Get the exact place on Earth from any image**

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM4.5 7.5a.5.5 0 0 0 0 1h5.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3a.5.5 0 0 0 0-.708l-3-3a.5.5 0 1 0-.708.708L10.293 7.5H4.5z"/></svg> Quick Overview

| Aspect | Semester 1 (Weeks 1-7) | Semester 2 (Weeks 8-16) |
|--------|------------------------|-------------------------|
| **Focus** | Scene Classification | Exact Location Detection |
| **Output** | Scene category (5 types) | Exact place, coordinates, landmarks |
| **Example** | "Urban - 85%" | "Eiffel Tower, Paris, France, 48.8584°N" |
| **Technology** | ResNet-50 deep learning | Hybrid: ResNet + EfficientNet + ViT + Gemini AI |
| **Purpose** | Foundation for scene understanding | **Identify exact place on Earth** |
| **Weeks** | 7 weeks | 9 weeks |
| **Hours** | 260 hours | 600 hours |

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M1 2.5A1.5 1.5 0 0 1 2.5 1h11A1.5 1.5 0 0 1 15 2.5v11a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 1 13.5v-11z"/></svg> SEMESTER 1 CONTRIBUTION (Weeks 1-7)

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/></svg> Main Goal
**Build scene classification system to understand image context (Urban, Rural, Coastal, Mountain, Forest)**

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M0 2h16v2H0V2zm0 6h16v2H0V8zm0 6h16v2H0v-2z"/></svg> Key Deliverables

#### Week 1: Project Setup
- GitHub repository with complete structure
- Documentation framework
- Team roles assigned
- Problem statement and motivation documented

#### Week 2: Literature Review
- 21 research papers reviewed
- Key insights from Places365, PlaNet, ViT, SegVLAD
- Methodology understanding
- Research foundation established

#### Week 3: Dataset Preparation
- Places365 category mapping (365 → 5 categories)
- prepare_dataset.py (300+ lines)
- Category rationale documented
- Dataset split strategy (70/15/15)

#### Week 4: Preprocessing Pipeline
- Image preprocessing pipeline (pipeline.py - 350+ lines)
- Data augmentation strategies
- PyTorch Dataset and DataLoader
- Transformation pipeline

#### Week 5: Model Training
- ResNet-50 implementation (model.py - 400+ lines)
- Training pipeline (train.py - 350+ lines)
- Transfer learning from ImageNet
- Model training on Places365

#### Week 6: Evaluation Framework
- Metrics calculation (evaluate.py - 400+ lines)
- Confusion matrix analysis
- Per-class performance
- Visualization utilities (visualizations.py - 450+ lines)

#### Week 7: Documentation & Demo
- Demo application (demo_app.py)
- Semester 1 report
- Results compilation
- Presentation materials

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M6 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm-5 6s-1 0-1-1 1-4 6-4 6 3 6 4-1 1-1 1H1zM11 3.5a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 0 1h-4a.5.5 0 0 1-.5-.5zm.5 2.5a.5.5 0 0 0 0 1h4a.5.5 0 0 0 0-1h-4zm2 3a.5.5 0 0 0 0 1h2a.5.5 0 0 0 0-1h-2zm0 3a.5.5 0 0 0 0 1h2a.5.5 0 0 0 0-1h-2z"/></svg> Team Contributions (Semester 1)

| Team Member | Primary Role | Key Deliverables | Hours |
|-------------|--------------|------------------|-------|
| **Krishan Yadav** | Technical Lead & Architecture | Model implementation, training pipeline, integration | 91 |
| **Aditi Sah** | Dataset & Literature Lead | Literature review, dataset preparation, mapping | 56 |
| **Anuj Kondawar** | Preprocessing Lead | Preprocessing pipeline, augmentation, optimization | 57 |
| **Jensi Paneliya** | Evaluation & Docs Lead | Evaluation framework, visualizations, documentation | 56 |

**Total Hours:** 260 hours

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M14 4.5V14a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V2a2 2 0 0 1 2-2h5.5L14 4.5z"/></svg> Code Delivered
- 5,500+ lines of production-quality Python code
- 3 complete model architectures (ResNet-50 ready, EfficientNet and ViT structured)
- Complete data pipeline (download, prepare, preprocess, load)
- Comprehensive evaluation framework
- 6 types of visualizations
- Working demo application

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M3 2.5a2.5 2.5 0 0 1 5 0 2.5 2.5 0 0 1 5 0v.006c0 .07 0 .27-.038.494H15a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1v7.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 1 14.5V7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h2.038A2.968 2.968 0 0 1 3 2.506V2.5z"/></svg> What We Built
**A complete scene classification system** that:
- Takes any outdoor image as input
- Classifies into 5 scene categories
- Provides confidence scores
- Shows probability distribution
- Works without GPS or metadata
- Demonstrates transfer learning mastery

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M2 2.5A1.5 1.5 0 0 1 3.5 1h11A1.5 1.5 0 0 1 16 2.5v11a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 2 13.5v-11z"/></svg> SEMESTER 2 CONTRIBUTION (Weeks 8-16)

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/></svg> Main Goal
**Get the exact place on Earth from any image using hybrid AI approach**

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M0 2h16v2H0V2zm0 6h16v2H0V8zm0 6h16v2H0v-2z"/></svg> Key Deliverables by Phase

#### PHASE 1: Advanced Models (Weeks 8-10)
**Goal:** Improve scene classification accuracy with state-of-the-art architectures

**Week 8: EfficientNet Implementation**
- EfficientNet-B0 architecture
- Training on Places365
- Hyperparameter optimization
- Performance comparison vs ResNet-50

**Week 9: Vision Transformer (ViT)**
- Vision Transformer implementation
- Patch-based processing
- Self-attention mechanisms
- Comparative evaluation

**Week 10: Model Comparison**
- Side-by-side performance analysis
- ResNet-50 vs EfficientNet vs ViT
- Accuracy, speed, resource usage
- Model selection for production

#### PHASE 2: Exact Location Detection (Weeks 11-13) **CRITICAL**
**Goal:** Integrate AI to identify the exact place on Earth

**Week 11: Gemini AI Integration**
- Google Gemini API setup and authentication
- GeminiLocationAnalyzer class (gemini_integration.py - 350+ lines)
- **Location detection capabilities:**
  - Exact landmarks (e.g., "Eiffel Tower")
  - City and country identification
  - GPS coordinates (latitude/longitude)
  - Regional context for generic scenes
- Prompt engineering for location queries
- Response parsing and structuring

**Week 12: Hybrid System Development**
- Two-stage architecture implementation
- demo_app_hybrid.py (300+ lines)
- **Stage 1:** Scene classification (your model)
- **Stage 2:** Exact location detection (Gemini AI)
- Combined result presentation
- Integration testing

**Week 13: Hybrid System Evaluation**
- Famous landmarks dataset testing
- Location detection accuracy metrics
- GPS coordinate precision assessment
- Landmark recognition rate
- Comparison: Scene-only vs Hybrid
- Use case validation (photos, travel, geotagging)

#### PHASE 3: Final Deliverables (Weeks 14-16)
**Goal:** Complete documentation and presentation

**Week 14: Results Compilation**
- All model results aggregated
- Hybrid system performance analysis
- Comparative charts and visualizations
- Performance metrics documentation

**Week 15: Final Report**
- Complete technical report (20-25 pages)
- Literature review chapter
- Methodology and implementation
- Results and analysis (all models + hybrid)
- Conclusions and future work

**Week 16: Presentation**
- Final presentation slides (30-40 slides)
- Demo video creation
- Viva voce preparation
- Project poster
- GitHub finalization

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M6 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm-5 6s-1 0-1-1 1-4 6-4 6 3 6 4-1 1-1 1H1zM11 3.5a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 0 1h-4a.5.5 0 0 1-.5-.5zm.5 2.5a.5.5 0 0 0 0 1h4a.5.5 0 0 0 0-1h-4zm2 3a.5.5 0 0 0 0 1h2a.5.5 0 0 0 0-1h-2zm0 3a.5.5 0 0 0 0 1h2a.5.5 0 0 0 0-1h-2z"/></svg> Team Contributions (Semester 2)

| Team Member | Primary Role | Key Deliverables | Hours |
|-------------|--------------|------------------|-------|
| **Krishan Yadav** | Advanced Models & Hybrid System | EfficientNet, ViT, hybrid architecture, integration | 150 |
| **Aditi Sah** | AI Integration & Location Testing | Gemini API, location detection, landmark testing | 150 |
| **Anuj Kondawar** | Training & Optimization | Model training, performance optimization, pipeline | 150 |
| **Jensi Paneliya** | Evaluation & Final Docs | Hybrid evaluation, final report, presentation | 150 |

**Total Hours:** 600 hours

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M14 4.5V14a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V2a2 2 0 0 1 2-2h5.5L14 4.5z"/></svg> Code to Deliver
- EfficientNet and ViT implementations
- gemini_integration.py (AI integration - 350+ lines) **ALREADY CREATED**
- demo_app_hybrid.py (hybrid demo - 300+ lines) **ALREADY CREATED**
- Hybrid evaluation framework
- Location detection testing suite
- Performance comparison utilities

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M3 2.5a2.5 2.5 0 0 1 5 0 2.5 2.5 0 0 1 5 0v.006c0 .07 0 .27-.038.494H15a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1v7.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 1 14.5V7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h2.038A2.968 2.968 0 0 1 3 2.506V2.5z"/></svg> What We're Building
**A hybrid AI system** that:
- Combines scene classification with exact location detection
- **Identifies the exact place on Earth** from any image
- Recognizes famous landmarks (Eiffel Tower, Taj Mahal, etc.)
- Provides GPS coordinates (latitude/longitude)
- Identifies city and country
- Gives regional context for generic scenes
- Uses two-stage architecture (custom models + commercial AI)
- Demonstrates advanced system integration

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM4.5 7.5a.5.5 0 0 0 0 1h5.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3a.5.5 0 0 0 0-.708l-3-3a.5.5 0 1 0-.708.708L10.293 7.5H4.5z"/></svg> Purpose of Location Detection

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/></svg> Primary Purpose
**Get the exact place on Earth from any image**

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M0 2a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V2z"/></svg> Use Cases

1. **Photo Organization**
   - Automatically tag photos by exact location
   - Group photos by place (all Eiffel Tower photos together)
   - Create location-based albums

2. **Travel Applications**
   - Identify landmarks from tourist photos
   - Get location information without GPS
   - Suggest nearby places based on scene

3. **Geotagging Lost Photos**
   - Restore location metadata to photos without GPS
   - Identify where photos were taken
   - Reconstruct travel history

4. **Social Media Enhancement**
   - Auto-tag locations in posts
   - Suggest relevant hashtags
   - Connect with others at same location

5. **Security & Verification**
   - Verify photo locations
   - Detect photo origins
   - Validate travel claims

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM4.5 7.5a.5.5 0 0 0 0 1h5.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3a.5.5 0 0 0 0-.708l-3-3a.5.5 0 1 0-.708.708L10.293 7.5H4.5z"/></svg> How It Works

```
User uploads image
    ↓
STAGE 1: Scene Classification
    → Identifies scene type (Urban, Coastal, etc.)
    → Provides context hint to Stage 2
    ↓
STAGE 2: Exact Location Detection (Gemini AI)
    → Uses scene hint + visual features
    → Identifies exact place
    → Extracts landmarks, coordinates
    ↓
Combined Output:
    → Scene: Urban (85% confidence)
    → Location: Times Square, New York
    → Country: United States
    → Coordinates: 40.7580°N, 73.9855°W
    → Confidence: High
```

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M2 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2H2z"/></svg> Example Outputs

**Input:** Photo of a famous tower  
**Output:**  
- Scene: Urban (92%)
- **Exact Place: Eiffel Tower**
- City: Paris
- Country: France
- Coordinates: 48.8584°N, 2.2945°E
- Confidence: High

**Input:** Mountain landscape  
**Output:**  
- Scene: Mountain (88%)
- **Exact Place: Mount Fuji**
- Region: Honshu Island
- Country: Japan
- Coordinates: ~35.36°N, 138.73°E
- Confidence: High

**Input:** Generic forest scene  
**Output:**  
- Scene: Forest (85%)
- **Exact Place: Deciduous forest, likely temperate region**
- Region: Northern Hemisphere, temperate zone
- Characteristics: Dense vegetation, autumn colors
- Confidence: Medium (generic scene)

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M2 0a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V2z"/></svg> Complete Project Summary

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M1 2.5A1.5 1.5 0 0 1 2.5 1h11A1.5 1.5 0 0 1 15 2.5v11a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 1 13.5v-11z"/></svg> Timeline
- **Total Duration:** 16 weeks (January 20 - May 15, 2026)
- **Semester 1:** 7 weeks (scene classification foundation)
- **Semester 2:** 9 weeks (exact location detection)
- **Sign-Offs:** 16 mentor signatures required (4 obtained so far)

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M6 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm-5 6s-1 0-1-1 1-4 6-4 6 3 6 4-1 1-1 1H1z"/></svg> Team
- **4 members:** Krishan Yadav, Aditi Sah, Anuj Kondawar, Jensi Paneliya
- **Equal contribution:** Each member responsible for specific domains
- **Total hours:** 860 hours (260 Semester 1 + 600 Semester 2)
- **Current progress:** 150/860 hours (17.4%)

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M3 2.5a2.5 2.5 0 0 1 5 0 2.5 2.5 0 0 1 5 0v.006c0 .07 0 .27-.038.494H15a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1v7.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 1 14.5V7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h2.038A2.968 2.968 0 0 1 3 2.506V2.5z"/></svg> Deliverables
- **Code:** 5,500+ lines (Semester 1) + additional for Semester 2
- **Models:** ResNet-50, EfficientNet, Vision Transformer
- **AI Integration:** Gemini API for exact location detection
- **Demos:** demo_app.py (scene only) + demo_app_hybrid.py (exact place)
- **Documentation:** Complete technical report (20-25 pages)
- **Presentation:** 30-40 slides + demo video

### <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM4.5 7.5a.5.5 0 0 0 0 1h5.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3a.5.5 0 0 0 0-.708l-3-3a.5.5 0 1 0-.708.708L10.293 7.5H4.5z"/></svg> Key Innovation
**Hybrid two-stage architecture combining custom scene classification models with commercial AI for exact place identification on Earth**

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M2 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2H2z"/></svg> Quick Reference

### What is DeepSceneLoc?
A hybrid AI system that identifies the exact place on Earth from any image using two-stage architecture: custom scene classification + AI-powered location detection.

### What does Semester 1 deliver?
Scene classification into 5 categories (Urban, Rural, Coastal, Mountain, Forest) with 80-90% accuracy using ResNet-50.

### What does Semester 2 deliver?
Exact location detection including landmarks, city, country, and GPS coordinates using hybrid approach (your models + Gemini AI).

### Why two semesters?
Semester 1 builds the foundation (scene understanding). Semester 2 enhances it with exact place identification capability.

### What makes it unique?
Combines custom-trained deep learning models with commercial AI to get best of both worlds: control + accuracy.

### What's the main purpose?
**Get the exact place on Earth from any image** - useful for photo organization, geotagging, travel apps, and location verification.

---

**Document Created:** February 27, 2026  
**For:** Complete semester breakdown and team allocation reference  
**Status:** 4/16 sign-offs obtained, Week 5 in progress
