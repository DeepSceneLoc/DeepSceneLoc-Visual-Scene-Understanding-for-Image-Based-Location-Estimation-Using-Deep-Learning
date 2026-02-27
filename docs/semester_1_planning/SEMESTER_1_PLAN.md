# Semester 1 Planning - DeepSceneLoc
## Weeks 1-7 Execution Plan
## Purpose: Build scene classification foundation for exact location detection

### <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/></svg> Overview

This semester focuses on project foundation, setup, dataset preparation, and establishing a baseline scene classification model. This lays the groundwork for Semester 2's exact location detection enhancement.

**Context:** Part of 16-week project structure
- Semester 1 (Weeks 1-7): Scene classification (5 categories)
- Semester 2 (Weeks 8-16): Exact location detection with AI integration

**All work uses only free and open resources.**

---

## Week 1: Project Setup & Scope Freeze
**Target: 30% project initialization**

### Tasks

#### 1. Finalize Project Name
- Project Name: DeepSceneLoc
- Full Title: Visual Scene Understanding for Image-Based Location Estimation Using Deep Learning
- Status: CONFIRMED

#### 2. Create GitHub Repository
- Repository Setup
  - Create private or public repository
  - Name: DeepSceneLoc
  - Description: Visual scene understanding for location estimation using CNN and Vision Transformers
  - Initialize with README
  - Add .gitignore for Python projects

#### 3. Add README and Proprietary License
- Create comprehensive README.md
- Add LICENSE file (proprietary)
- Document project motivation and problem statement
- Add quick start guide

#### 4. Freeze Scope
**Semantic Location Categories:**
- Urban (cities, downtown, streets)
- Rural (farmland, countryside, villages)
- Coastal (beaches, seaside, harbors)
- Mountain (peaks, slopes, highlands)
- Forest (wooded areas, dense vegetation)

**Out of Scope:**
- GPS-based solutions
- EXIF data usage
- Metadata extraction
- Real-time location tracking

### Deliverables
- [ ] GitHub repository created and configured
- [ ] README.md with project overview
- [ ] Proprietary LICENSE file added
- [ ] Problem statement documented
- [ ] Project motivation documented
- [ ] Scope definition finalized

### Success Metrics
- Repository publicly visible (if chosen)
- Clear project structure established
- Scope document signed off

---

## Week 2: Literature Review
**Target: Research foundation established**

### Tasks

#### 1. Study Open-Access Research Papers
- Places: A 10 Million Image Database for Scene Recognition
  - Focus: Large-scale scene dataset
  - Key Insights: Scene category definitions, dataset construction

- PlaNet: Photo Geolocation with Convolutional Neural Networks
  - Focus: Location estimation using CNNs
  - Key Insights: Architecture design, loss functions

- An Image is Worth 16x16 Words (Vision Transformer)
  - Focus: Transformer-based image understanding
  - Key Insights: Self-attention mechanisms, patch-based processing

#### 2. Study Free Learning Resources
- CS231n (Stanford - Free)
  - Convolutional Neural Networks
  - Transfer Learning concepts
  - Fine-tuning strategies

- PyTorch Official Tutorials
  - Image classification
  - Transfer learning
  - Custom dataset handling

- Google Research Blogs
  - Latest vision model advances
  - Practical implementation guides

#### 3. Note-Taking Strategy
- Create literature review notes (2-3 pages)
- Organize by topic:
  - Scene Recognition
  - CNN Architectures
  - Vision Transformers
  - Transfer Learning
  - Location Estimation

#### 4. Reference Management
- Compile complete reference list
- Format: APA style
- Include URLs and access dates

### Deliverables
- [ ] Literature review notes (2-3 pages)
- [ ] Complete reference list (APA formatted)
- [ ] Key insights documented
- [ ] Architecture design decisions justified

### Success Metrics
- 5+ quality papers reviewed
- Clear understanding of SOTA approaches
- Documented design rationale

---

## Week 3: Dataset Preparation
**Target: Training data ready**

### Tasks

#### 1. Download Places365 Dataset
- Dataset: Places365 (Outdoor Subset)
- Free download from official source
- Storage: Organize locally or cloud storage

#### 2. Scene Category Mapping
Map Places365 categories to project categories:

| Location Category | Places365 Categories |
|------------------|----------------------|
| Urban | street, plaza, building_exterior, market, subway_station, urban_street |
| Rural | farm, farmland, field, village, countryside |
| Coastal | beach, seashore, harbor, pier, lighthouse |
| Mountain | mountain, peak, slope, cliff, snowy_mountain |
| Forest | forest, woods, tree_house, botanical_garden |

#### 3. Dataset Split
- Training Set: 70%
- Validation Set: 15%
- Test Set: 15%

Dataset Statistics Template:
```
Total Images: XXXX
Training: XXXX images
Validation: XXXX images
Test: XXXX images

Per Category Distribution:
- Urban: XXXX images
- Rural: XXXX images
- Coastal: XXXX images
- Mountain: XXXX images
- Forest: XXXX images

Class Balance: YES/NO
```

#### 4. Data Organization
- Create directory structure
- Label mapping CSV file
- Create splits (train/val/test) with file lists

### Deliverables
- [ ] Places365 dataset downloaded
- [ ] Label mapping table created
- [ ] Train/val/test splits created
- [ ] Dataset statistics documented
- [ ] Class distribution verified
- [ ] Data organization complete

### Success Metrics
- Balanced dataset across categories
- Clear label mapping
- All splits ready for training

---

## Week 4: Image Preprocessing Pipeline
**Target: Preprocessing code ready**

### Tasks

#### 1. Implement Preprocessing Functions

**Image Resizing**
- Target size: 224x224 pixels (standard for ImageNet models)
- Method: Aspect ratio preserving with padding
- Library: PIL/OpenCV

**Normalization**
- ImageNet normalization: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
- Per-image normalization
- Optional: Z-score normalization

**Data Augmentation**
- Random horizontal flip (50%)
- Random rotation (up to 15 degrees)
- Random brightness/contrast adjustment
- Random color jitter
- Optional: Random crop

#### 2. Implementation Details
- Create preprocessing.py module
- Implement PyTorch transforms
- Create both train and validation pipelines
- Train: includes augmentation
- Validation/Test: minimal transformations

#### 3. Verification
- Test on sample images
- Visualize augmented samples
- Check normalization correctness
- Verify no data leakage

#### 4. Performance Optimization
- CPU-friendly operations
- Batch preprocessing for efficiency
- Memory-efficient implementations

### Deliverables
- [ ] preprocessing.py created
- [ ] Resize function implemented
- [ ] Normalization function implemented
- [ ] Data augmentation implemented
- [ ] Sample augmented images saved
- [ ] Preprocessing verified on test data

### Success Metrics
- All augmentation operations working
- Sample images visually correct
- No preprocessing errors
- CPU-friendly execution

---

## Week 5: Baseline Model Development
**Target: First trained model**

### Tasks

#### 1. Load Pretrained ResNet-50
- Use PyTorch torchvision.models.resnet50
- Load ImageNet pretrained weights (free)
- No training from scratch

#### 2. Model Adaptation
- Freeze backbone layers (first 3 blocks)
- Replace final fully connected layer
  - Remove: 1000-class ImageNet head
  - Add: 5-class custom head (Urban, Rural, Coastal, Mountain, Forest)
- Add softmax activation for classification

#### 3. Training Setup
- Loss Function: CrossEntropyLoss
- Optimizer: Adam or SGD with momentum
- Learning Rate: 0.001 (start with default)
- Batch Size: 32 (adjust for memory)
- Epochs: 30-50 (monitor for convergence)

#### 4. Training on Google Colab
- Setup Google Colab notebook
- Mount Google Drive for datasets
- Use free GPU (T4 or K80)
- Implement learning rate scheduling
- Save model checkpoints

#### 5. Monitoring and Logging
- Track training loss
- Track validation accuracy
- Save training curves
- Log hyperparameters

### Deliverables
- [ ] ResNet-50 model loaded
- [ ] Custom head implemented
- [ ] Training code ready
- [ ] Trained baseline model saved
- [ ] Training logs created
- [ ] Training curves plotted

### Success Metrics
- Model training without errors
- Validation accuracy: 60-70% (baseline expectation)
- No overfitting issues
- Model saved successfully

---

## Week 6: Baseline Model Evaluation
**Target: Performance metrics established**

### Tasks

#### 1. Test Set Evaluation
- Evaluate on held-out test set
- Calculate overall accuracy
- Document inference time

#### 2. Performance Metrics
- Accuracy: (TP + TN) / (Total)
- Precision: TP / (TP + FP) per class
- Recall: TP / (TP + FN) per class
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
- Per-class metrics

#### 3. Confusion Matrix Analysis
- Generate confusion matrix
- Identify misclassification patterns
- Visualize as heatmap
- Analyze which categories are confused most

#### 4. Error Analysis
- Document major error patterns
- Identify hardest samples
- Note edge cases
- Record observations for improvement

#### 5. Visualization
- Create confusion matrix plot
- Plot per-class accuracy bars
- Generate error distribution charts
- Save all visualizations

### Deliverables
- [ ] Test set evaluation completed
- [ ] Accuracy metrics calculated
- [ ] Confusion matrix generated
- [ ] Per-class metrics documented
- [ ] Confusion matrix plots created
- [ ] Error analysis notes documented
- [ ] All results saved

### Success Metrics
- Clear performance metrics established
- Confusion matrix shows interpretable patterns
- Error analysis guides future improvements
- Baseline established for comparison

---

## Week 7: Documentation & Review Preparation
**Target: Semester review ready**

### Tasks

#### 1. System Architecture Documentation
- Create architecture diagram
- Document data flow:
  - Input: Raw images
  - Processing: Preprocessing pipeline
  - Model: ResNet-50 classifier
  - Output: Location category predictions

#### 2. Flow of Execution
Document:
- Data loading process
- Preprocessing steps
- Model forward pass
- Loss calculation
- Backpropagation
- Evaluation process

#### 3. Baseline Results Summary
- Training accuracy progression
- Validation accuracy progression
- Test set accuracy
- Per-class accuracy breakdown
- Training time and resources used

#### 4. Create Review Presentation
- Create PowerPoint presentation
- Include sections:
  - Project overview
  - Problem statement
  - Methodology
  - Dataset description
  - Model architecture
  - Results
  - Challenges and solutions
  - Next steps

#### 5. Update GitHub README
- Add project description
- Add dataset information
- Add model details
- Add usage instructions
- Add results summary
- Add references

### Deliverables
- [ ] System architecture diagram created
- [ ] Execution flow documentation completed
- [ ] Baseline results summary documented
- [ ] PowerPoint presentation created
- [ ] README.md updated
- [ ] All documentation pushed to GitHub

### Success Metrics
- Clear architecture visualization
- Complete execution documentation
- Professional presentation ready
- Clean GitHub repository

---

## Semester 1 Completion Summary

### Completed Components (30% of total project)
| Component | Status | Evidence |
|-----------|--------|----------|
| Project Setup | Complete | GitHub repo, README, LICENSE |
| Literature Review | Complete | Notes and references document |
| Dataset Preparation | Complete | 70/15/15 split ready |
| Preprocessing Pipeline | Complete | preprocessing.py module |
| Baseline CNN Model | Complete | Trained ResNet-50 |
| Model Evaluation | Complete | Metrics and confusion matrix |
| Documentation | Complete | Architecture and results docs |

### Key Deliverables Checklist
- [ ] GitHub repository live and documented
- [ ] Problem statement finalized
- [ ] Literature review (2-3 pages)
- [ ] Reference list (APA formatted)
- [ ] Dataset description and statistics
- [ ] Label mapping table
- [ ] Preprocessing code and samples
- [ ] Trained baseline model
- [ ] Training logs and curves
- [ ] Evaluation metrics and plots
- [ ] System architecture diagram
- [ ] Execution flow documentation
- [ ] Semester review presentation (PPT)
- [ ] Updated README with results

### Performance Benchmarks Achieved
- Baseline Model Accuracy: 60-70%
- Training Stability: Convergent
- Dataset Size: 5000+ labeled images
- Model Parameters: 25.5M (ResNet-50)

### Semester Grade Expectation
- Foundation: 30% Complete
- Ready for Advanced Development: Yes
- Review Ready: Yes

---

## Notes and Observations
- Focus on solid foundation for next semester
- All work uses free resources
- No GPU limitations for training
- Dataset adequately sized for transfer learning
- Preprocessing pipeline extensible for future augmentations
