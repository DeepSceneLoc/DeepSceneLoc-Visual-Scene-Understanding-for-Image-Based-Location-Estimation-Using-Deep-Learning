# DeepSceneLoc - Week 5 Sign-off Presentation
## February 27, 2026

**Project:** DeepSceneLoc - Visual Scene Understanding for Image-Based Location Estimation

**Team:**
- Krishan Yadav (Technical Lead)
- Aditi Sah (Data Lead)
- Anuj Kondawar (Preprocessing Lead)
- Jensi Paneliya (Documentation Lead)

**Meeting Purpose:** Week 5 Sign-off Request

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/></svg> Slide 1: Project Overview

**Speaker: Krishan Yadav**

### What We're Building
**Get the exact place on Earth from any image**

### Two-Stage Hybrid System

**Stage 1 (Semester 1 - Current):**
- Scene classification into 5 categories
- Urban, Rural, Coastal, Mountain, Forest
- Custom-trained ResNet-50 model

**Stage 2 (Semester 2 - Future):**
- Exact location detection with AI integration
- Specific landmarks, cities, countries
- GPS coordinates estimation

### Project Duration
**16 weeks total:**
- Semester 1: Weeks 1-7 (Scene classification foundation)
- Semester 2: Weeks 8-16 (Exact location detection with AI)

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.5-13v1.5a.5.5 0 0 1-1 0V3h-3a.5.5 0 0 1 0-1h7a.5.5 0 0 1 0 1h-3z"/><path d="M8 4.5a.5.5 0 0 1 .5.5v3.5H11a.5.5 0 0 1 0 1H8a.5.5 0 0 1-.5-.5V5a.5.5 0 0 1 .5-.5z"/></svg> Slide 2: Progress Status

**Speaker: Aditi Sah**

### Current Status: 4/16 Sign-offs Obtained (25%)

| Week | Milestone | Status | Sign-off Date |
|------|-----------|--------|---------------|
| Week 1 | Project Setup & Scope Freeze | <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Complete | January 26, 2026 |
| Week 2 | Literature Review | <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Complete | February 2, 2026 |
| Week 3 | Dataset Preparation | <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Complete | February 9, 2026 |
| Week 4 | Preprocessing Pipeline | <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Complete | February 16, 2026 |
| **Week 5** | **Baseline Model (ResNet-50)** | <svg width="12" height="12" fill="orange"><circle cx="6" cy="6" r="5"/></svg> **Requesting Today** | **February 27, 2026** |
| Week 6 | Model Evaluation | Pending | Target: March 6, 2026 |
| Week 7 | Documentation & Review | Pending | Target: March 13, 2026 |

### Today's Request
**Requesting Week 5 sign-off for Baseline Model infrastructure completion**

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M2 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2H2z"/></svg> Slide 3: Week 5 Deliverables

**Speaker: Anuj Kondawar**

### What We've Completed This Week

#### 1. Model Architecture Implementation
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> ResNet-50 pretrained on ImageNet loaded
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Custom classification head (2048 → 512 → 5 classes)
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Transfer learning configuration complete

#### 2. Training Infrastructure
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Complete training pipeline (Trainer class)
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Adam optimizer (lr=0.001)
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> StepLR scheduler (step=7, gamma=0.1)
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> CrossEntropyLoss configured

#### 3. System Components
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Checkpointing system (best model, latest model, final model)
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Early stopping logic
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Training history logging (JSON)
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Google Colab environment setup

#### 4. Code Deliverables
- **Files Created:** `src/models/model.py` (400+ lines), `src/models/train.py` (350+ lines)
- **Total Code:** 750+ lines of production-ready training infrastructure
- **Documentation:** Complete docstrings and comments

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M1 2.5A1.5 1.5 0 0 1 2.5 1h3A1.5 1.5 0 0 1 7 2.5v3A1.5 1.5 0 0 1 5.5 7h-3A1.5 1.5 0 0 1 1 5.5v-3z"/></svg> Slide 4: Live Demo

**Speaker: Krishan Yadav**

### Interactive Scene Classification Demo

**Demo Application:** `demo_app.py`

**Access:** http://localhost:7860

### What the Demo Shows

1. **Image Upload Interface**
   - Drag and drop or click to upload
   - Supports JPG, PNG, JPEG formats

2. **Real-time Classification**
   - Model processes image immediately
   - Predicts one of 5 scene categories
   - Shows confidence percentages

3. **Output Display**
   - Primary category with confidence
   - Top-3 predictions with scores
   - Visual scene type identification

### Technical Details
- **Model:** ResNet-50 (pretrained backbone - demo mode)
- **Device:** CPU (for portability)
- **Categories:** Coastal, Forest, Mountain, Rural, Urban
- **Interface:** Gradio web UI

**Note:** Full training will be completed in Week 5-6 on Google Colab GPU

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M2 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2H2z"/></svg> Slide 5: Documentation & Repository

**Speaker: Jensi Paneliya**

### Comprehensive Documentation Created

#### Core Documentation (7 files created today)
1. **LICENSE** - Proprietary license (team-only use)
2. **16_WEEK_PROGRESS_TRACKING.md** - Complete timeline with sign-off tracking (1000+ lines)
3. **SEMESTER_CONTRIBUTIONS_SUMMARY.md** - Detailed semester breakdown (900+ lines)
4. **REPLANNING_SUMMARY.md** - Meeting preparation guide (700+ lines)
5. **SEMESTER_1_COMPLETION_SUMMARY.md** - Achievements documentation
6. **PROJECT_SCOPE_UPDATE.md** - Purpose and scope clarification
7. **DEMO_GUIDE.md** - Demo instructions

#### Updated Documentation (8 files)
- README.md - Complete project overview with hybrid system
- PROJECT_STATUS_AND_MILESTONES.md - 16-week structure
- PROJECT_OVERVIEW.md - Exact location purpose
- TEAM_ALLOCATION.md - All 16 weeks allocated
- Semester planning files (1 & 2)
- RESULTS_LOG.md - AI integration tracking

### GitHub Repository
**URL:** https://github.com/DeepSceneLoc/DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning

**Latest Commit:** Today (February 27, 2026)
- 31 files changed
- 6,133 lines added
- All documentation with SVG icons (no emojis - institutional standard)

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM4.5 7.5a.5.5 0 0 0 0 1h5.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3a.5.5 0 0 0 0-.708l-3-3a.5.5 0 1 0-.708.708L10.293 7.5H4.5z"/></svg> Slide 6: Technical Achievements

**Speaker: Krishan Yadav**

### What Makes This Week Significant

#### 1. Complete ML Pipeline
- Data → Preprocessing → Model → Training → Checkpoints
- Production-ready, modular architecture
- Reproducible with fixed seeds (42)

#### 2. Transfer Learning Implementation
- Successfully integrated ImageNet pretrained weights
- Custom head for 5-class scene classification
- Optimized for Places365 outdoor scenes

#### 3. Training Infrastructure
- Checkpoint management (best/latest/final)
- Early stopping to prevent overfitting
- Learning rate scheduling
- Training history logging

#### 4. Scalability
- Easy to extend to other models (EfficientNet, ViT planned for Semester 2)
- Configurable hyperparameters
- Google Colab compatible

#### 5. Quality Assurance
- Complete docstrings
- Type hints where applicable
- Modular design for testing
- Git version control

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M1 2.5A1.5 1.5 0 0 1 2.5 1h3A1.5 1.5 0 0 1 7 2.5v3A1.5 1.5 0 0 1 5.5 7h-3A1.5 1.5 0 0 1 1 5.5v-3z"/></svg> Slide 7: Team Contributions (Week 5)

**Speaker: Aditi Sah**

### Individual Contributions This Week

#### Krishan Yadav (Technical Lead) - 35 hours
- ResNet-50 model architecture implementation
- Training pipeline development
- Demo application creation
- GitHub repository management
- Code review and integration

#### Aditi Sah (Data Lead) - 21.5 hours
- Dataset validation for training
- Data loader optimization
- Training data verification
- Performance metric planning
- Documentation of data flow

#### Anuj Kondawar (Preprocessing Lead) - 22 hours
- Training pipeline integration with preprocessing
- Augmentation strategy validation
- Batch processing optimization
- Memory management
- System performance testing

#### Jensi Paneliya (Documentation Lead) - 21.5 hours
- 16-week progress tracking document
- Semester contributions summary
- Meeting preparation materials
- Repository documentation update
- Presentation materials preparation

**Total Team Hours This Week:** 100 hours
**Cumulative Project Hours:** 360/860 hours (42% of total allocation)

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.5-13v1.5a.5.5 0 0 1-1 0V3h-3a.5.5 0 0 1 0-1h7a.5.5 0 0 1 0 1h-3z"/><path d="M8 4.5a.5.5 0 0 1 .5.5v3.5H11a.5.5 0 0 1 0 1H8a.5.5 0 0 1-.5-.5V5a.5.5 0 0 1 .5-.5z"/></svg> Slide 8: Next Steps (Week 6-7)

**Speaker: Anuj Kondawar**

### Immediate Next Steps

#### Week 6: Model Training & Evaluation (Target: March 6)
- Train ResNet-50 on Google Colab GPU (20-40 epochs)
- Monitor training curves (loss & accuracy)
- Validate on validation set
- Save best model checkpoint
- **Deliverable:** Trained model ready for evaluation

#### Week 7: Documentation & Semester 1 Completion (Target: March 13)
- Comprehensive evaluation metrics
- Confusion matrix analysis
- Per-class performance visualization
- Error analysis
- Final Semester 1 report
- **Deliverable:** Complete Semester 1 documentation package

### Semester 1 Timeline Remaining
- **Week 5:** Infrastructure complete (today's sign-off)
- **Week 6:** Training execution (1 week)
- **Week 7:** Final evaluation and documentation (1 week)
- **Total:** 3 weeks to Semester 1 completion

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><circle cx="8" cy="8" r="6" fill="orange"/></svg> Slide 9: Semester 2 Preview

**Speaker: Jensi Paneliya**

### Looking Ahead: Exact Location Detection

#### Semester 2 Structure (Weeks 8-16)

**Weeks 8-10: Advanced Models & Infrastructure**
- EfficientNet-B0 and Vision Transformer implementation
- Comparative analysis setup
- Hybrid system architecture design

**Weeks 11-13: AI Integration (CRITICAL PHASE)**
- **Google Gemini AI integration**
- Exact location detection capability
- Landmark identification
- City, country, GPS coordinates
- Two-stage pipeline: Scene + Location

**Weeks 14-16: Final Testing & Presentation**
- End-to-end system testing
- Performance optimization
- Final documentation
- Project presentation preparation

### Ultimate Goal
**Input:** Any image
**Output:** 
- Stage 1: Scene category (Urban/Rural/Coastal/Mountain/Forest)
- Stage 2: Exact place (e.g., "Eiffel Tower, Paris, France, 48.8584°N, 2.2945°E")

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/></svg> Slide 10: Summary & Sign-off Request

**Speaker: Krishan Yadav**

### Week 5 Summary

#### What We've Achieved
1. <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Complete baseline model architecture (ResNet-50 with custom head)
2. <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Full training infrastructure with checkpointing
3. <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Interactive demo application (running live today)
4. <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> 750+ lines of production-ready code
5. <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Comprehensive documentation (15+ files updated/created)
6. <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Everything pushed to GitHub repository

#### Team Performance
- 100 team hours invested this week
- All deliverables completed on schedule
- Quality code with documentation
- Ready for training phase

#### Risk Mitigation
- Demo proves infrastructure works
- Clear path to Week 6 training
- No blocking issues identified

### Request for Sign-off

**We respectfully request Week 5 sign-off approval for:**
- Baseline Model Infrastructure Completion
- Date: February 27, 2026
- Status: All deliverables complete and demonstrated

**Next Meeting:** Week 6 sign-off (Target: March 6, 2026)
- Will demonstrate trained model performance
- Show training curves and initial metrics

---

## Questions & Discussion

**Team ready to answer questions on:**
- Technical architecture decisions
- Dataset and preprocessing pipeline
- Training strategy and hyperparameters
- Documentation and progress tracking
- Timeline and resource allocation
- Semester 2 planning

---

**Thank you for your time!**

**Team DeepSceneLoc**
- Krishan Yadav, Aditi Sah, Anuj Kondawar, Jensi Paneliya

**Project Repository:** https://github.com/DeepSceneLoc/DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning

**Demo:** http://localhost:7860 (Running now)
