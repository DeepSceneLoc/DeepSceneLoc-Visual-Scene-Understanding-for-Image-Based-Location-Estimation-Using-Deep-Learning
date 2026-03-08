# DeepSceneLoc - Semester 1 Review Presentation

**DeepSceneLoc: Visual Scene Understanding for Image-Based Location Estimation Using Deep Learning**

Project Team: Krishan Yadav, Aditi Sah, Anuj Kondawar, Jensi Paneliya  
Date: March 2026  
Status: Semester 1 Complete (30%)

---

## Slide 1: Title Slide

### DeepSceneLoc
**Visual Scene Understanding for Image-Based Location Estimation Using Deep Learning**

**Team Members:**
- Krishan Yadav (Technical Lead)
- Aditi Sah (Data Lead)
- Anuj Kondawar (Preprocessing Lead)
- Jensi Paneliya (Documentation Lead)

**Semester 1 Review**  
March 2026

---

## Slide 2: Problem Statement

### Challenge
**Can we estimate the location category of an image from visual scene understanding alone?**

### Key Constraints
- ❌ No GPS coordinates
- ❌ No EXIF metadata
- ❌ No textual information
- ✅ Only visual scene features

### Why This Matters
- Privacy-preserving location estimation
- Works on historical images without metadata
- Semantic understanding vs precise coordinates
- Applications in image organization, content moderation, recommendations

---

## Slide 3: Project Objectives

### Primary Goal
Classify images into 5 semantic location categories using deep learning

### Location Categories
1. **Urban** - Cities, streets, buildings
2. **Rural** - Farmland, countryside, villages
3. **Coastal** - Beaches, harbors, seaside
4. **Mountain** - Peaks, highlands, terrain
5. **Forest** - Wooded areas, dense vegetation

### Approach
- Transfer learning from ImageNet pretrained models
- Fine-tune on Places365 outdoor subset
- Classification-based (not coordinate regression)

---

## Slide 4: Methodology

### Architecture Overview
```
Input Image (RGB)
    ↓
Preprocessing Pipeline
- Resize to 224×224
- Data augmentation (training)
- Normalization (ImageNet stats)
    ↓
ResNet-50 Backbone
- Pretrained on ImageNet
- Feature extractor (2048 dims)
    ↓
Custom Classification Head
- 2048 → 512 (ReLU, Dropout)
- 512 → 5 (Location classes)
    ↓
Softmax → Predictions
```

### Key Design Decisions
- **Model:** ResNet-50 (proven architecture, 25.5M parameters)
- **Dataset:** Places365 outdoor subset mapped to 5 categories
- **Split:** 70% train, 15% val, 15% test
- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropy
- **Scheduler:** StepLR (step=7, gamma=0.1)

---

## Slide 5: Dataset

### Places365 Dataset
- **Source:** MIT CSAIL, Zhou et al.
- **Original Size:** 1.8M images, 365 scene categories
- **Our Subset:** Outdoor categories only
- **Mapping:** 365 → 5 location categories

### Category Mapping Examples

**Urban (19 Places365 categories)**
- street, downtown, building_facade, skyscraper, plaza, shopping_mall, etc.

**Rural (16 categories)**
- field, farm, barn, pasture, cornfield, wheat_field, village, etc.

**Coastal (18 categories)**
- beach, coast, ocean, harbor, pier, lighthouse, marina, etc.

**Mountain (18 categories)**
- mountain, canyon, valley, cliff, summit, ski_slope, etc.

**Forest (17 categories)**
- forest, rainforest, bamboo_forest, woodland, jungle, swamp, etc.

### Data Split
- Training: 70%
- Validation: 15%
- Test: 15%

---

## Slide 6: Preprocessing Pipeline

### Image Transformations

**Training Augmentation:**
```python
- Resize to 224×224
- Random Horizontal Flip (p=0.5)
- Random Rotation (±15 degrees)
- Color Jitter:
  - Brightness: ±20%
  - Contrast: ±20%
  - Saturation: ±20%
  - Hue: ±10%
- Random Affine (translate ±10%, scale 0.9-1.1)
- Normalize (ImageNet mean & std)
```

**Validation/Test:**
```python
- Resize to 224×224
- Normalize only (no augmentation)
```

### Why This Pipeline?
- Augmentation improves generalization
- Prevents overfitting on small variations
- ImageNet normalization for transfer learning
- Maintains spatial information (no crops)

---

## Slide 7: Model Architecture

### ResNet-50 Baseline

**Backbone:**
- Pretrained on ImageNet (1000 classes)
- Residual blocks with skip connections
- 25.5M total parameters
- Feature extraction: 2048-dimensional embeddings

**Custom Head:**
```
Layer 1: Linear(2048 → 512)
       → ReLU
       → Dropout(0.3)
Layer 2: Linear(512 → 5)
       → Softmax (prediction)
```

**Training Configuration:**
- Batch Size: 32
- Epochs: 20
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Scheduler: StepLR (reduce by 10× every 7 epochs)
- Hardware: Google Colab Free GPU (Tesla T4)

---

## Slide 8: Training Process

### Training Strategy

**Week 5 Execution:**
1. Load pretrained ResNet-50 from torchvision
2. Replace final FC layer with custom head
3. Fine-tune all layers (no freezing)
4. Monitor train/val loss and accuracy
5. Save best model based on validation accuracy
6. Early stopping with patience=5

### Checkpointing
- Save every 5 epochs
- Save best model automatically
- Store: model weights, optimizer state, scheduler state, history

### Resource Usage
- Platform: Google Colab Free Tier
- GPU: Tesla T4 (12 hours max session)
- Training Time: ~2-3 hours (20 epochs)
- Storage: ~100MB per checkpoint

---

## Slide 9: Results - Overall Performance

### Test Set Evaluation
*(To be filled with actual results)*

| Metric | Score |
|--------|-------|
| Overall Accuracy | XX.XX% |
| Macro Precision | X.XXX |
| Macro Recall | X.XXX |
| Macro F1-Score | X.XXX |
| Total Test Samples | XXXXX |

### Training Convergence
- Training Loss: X.XX → X.XX (trend)
- Validation Loss: X.XX → X.XX (trend)
- Training Accuracy: XX% → XX%
- Validation Accuracy: XX% → XX%
- Best Epoch: XX/20

---

## Slide 10: Results - Per-Class Performance

### Per-Category Accuracy
*(To be filled with actual results)*

| Category | Accuracy | Precision | Recall | F1-Score | Support |
|----------|----------|-----------|--------|----------|---------|
| Urban | XX.XX% | X.XX | X.XX | X.XX | XXXX |
| Rural | XX.XX% | X.XX | X.XX | X.XX | XXXX |
| Coastal | XX.XX% | X.XX | X.XX | X.XX | XXXX |
| Mountain | XX.XX% | X.XX | X.XX | X.XX | XXXX |
| Forest | XX.XX% | X.XX | X.XX | X.XX | XXXX |

### Key Observations
- Best performing category: [TBD]
- Most challenging category: [TBD]
- Most common confusion: [TBD] ↔ [TBD]

---

## Slide 11: Confusion Matrix

### Confusion Analysis
*(Include confusion matrix heatmap visualization here)*

**Top Confusions:**
1. [Category A] → [Category B]: XX samples (XX%)
2. [Category C] → [Category D]: XX samples (XX%)
3. [Category E] → [Category F]: XX samples (XX%)

**Why These Confusions?**
- Visual similarity between categories
- Ambiguous scenes (e.g., coastal mountains, urban forests)
- Dataset biases or label noise
- Lighting/weather variations

---

## Slide 12: Visualizations

### Training Curves
*(Include training/validation loss and accuracy plots)*

### Per-Class Performance
*(Include bar chart of per-class accuracy)*

### Metrics Comparison
*(Include grouped bar chart: precision, recall, F1 per class)*

---

## Slide 13: Error Analysis

### Challenging Cases
*(Identify patterns in misclassifications)*

**Example Error Categories:**
1. **Ambiguous Scenes**
   - Coastal mountains (mountain vs coastal)
   - Urban parks (urban vs forest)
   - Rural coastline (rural vs coastal)

2. **Lighting/Weather**
   - Foggy mountains → forest
   - Night urban → unrecognizable

3. **Unusual Perspectives**
   - Aerial views harder to classify
   - Close-up textures lose context

### Insights for Improvement
- Need more diverse training examples
- Consider multi-label for ambiguous scenes
- Attention mechanisms to focus on key features

---

## Slide 14: Challenges Faced

### Technical Challenges
1. **Dataset Size**
   - 1.8M images → storage and processing constraints
   - Solution: Focused on outdoor subset, used efficient dataloaders

2. **Class Imbalance**
   - Some categories more represented than others
   - Solution: Monitored per-class metrics, considered class weights

3. **Colab Session Limits**
   - 12-hour max runtime
   - Solution: Checkpointing every 5 epochs, Colab Pro not required

4. **Computation Time**
   - Long training times with large batches
   - Solution: Optimized batch size (32), used pin_memory

### Organizational Challenges
- Coordinating 4 team members across different tasks
- Ensuring code compatibility and documentation
- Weekly progress tracking and reviews

---

## Slide 15: Key Learnings

### Technical Learnings
- **Transfer Learning:** Pretrained models drastically reduce training time
- **Data Augmentation:** Critical for generalization with limited data
- **Hyperparameter Tuning:** Learning rate and scheduler significantly impact convergence
- **Evaluation:** Multiple metrics (not just accuracy) reveal model behavior

### Teamwork Learnings
- Clear role division improves efficiency
- Documentation crucial for collaboration
- Weekly syncs keep everyone aligned
- Git/GitHub essential for version control

### Domain Knowledge
- Scene recognition differs from object recognition
- Context matters more than individual objects
- Semantic location vs precise coordinates trade-offs
- Privacy implications of location estimation

---

## Slide 16: Week-by-Week Progress

### Semester 1 Timeline (7 Weeks)

**Week 1:** Project Setup & Scope Freeze ✅
- GitHub repository created
- Documentation structure established
- Scope defined: 5 categories, no GPS

**Week 2:** Literature Review ✅
- Studied 7 core papers (Places365, PlaNet, ViT, etc.)
- Documented key insights and methodologies

**Week 3:** Dataset Preparation ✅
- Mapped Places365 (365 → 5 categories)
- Created train/val/test splits (70/15/15)
- Verified data integrity

**Week 4:** Preprocessing Pipeline ✅
- Implemented custom Dataset class
- Built augmentation pipeline
- Created efficient DataLoaders

**Week 5:** Model Training ✅
- Implemented ResNet-50 baseline
- Trained for 20 epochs
- Saved best model checkpoint

**Week 6:** Evaluation & Analysis ✅
- Calculated all metrics
- Generated confusion matrix
- Analyzed error patterns

**Week 7:** Documentation & Review ✅
- Updated README and documentation
- Created visualizations
- Prepared this presentation

---

## Slide 17: Code Structure

### Repository Organization
```
DeepSceneLoc/
├── src/                     # Modular source code
│   ├── data/                # Dataset preparation (Aditi)
│   ├── preprocessing/       # Preprocessing pipeline (Anuj)
│   ├── models/              # Model training (Krishan)
│   ├── evaluation/          # Evaluation (Krishan)
│   └── utils/               # Visualizations (Team)
├── notebooks/               # Jupyter notebooks
├── docs/                    # Comprehensive documentation
├── config.yaml              # Centralized configuration
└── requirements.txt         # Dependency management
```

### Key Features
- **Modular Design:** Each component reusable and testable
- **Configuration-Driven:** Easy to experiment with hyperparameters
- **Documented:** Docstrings, comments, README files
- **Version Controlled:** Git with proper .gitignore
- **Reproducible:** Fixed random seeds, checkpointing

---

## Slide 18: Semester 2 Plan (Preview)

### Advanced Models (70% Completion)

**Phase 1: Multi-Model Training**
- EfficientNet-B0 (lightweight, 5.3M params)
- Vision Transformer (attention-based, 86M params)
- Compare with ResNet-50 baseline

**Phase 2: Comparative Study**
- Accuracy comparison across architectures
- Inference time analysis
- Model size and efficiency trade-offs

**Phase 3: Embeddings Analysis**
- Extract feature embeddings
- Visualize with t-SNE and PCA
- Analyze learned representations

**Phase 4: Top-K Predictions**
- Implement Top-3/Top-5 accuracy
- Confidence calibration
- Handle ambiguous cases

**Phase 5: Advanced Error Analysis**
- Failure case deep-dive
- Feature ablation studies
- Saliency maps (Grad-CAM)

**Phase 6: Optimization**
- Model compression (pruning, quantization)
- Deployment preparation
- Interactive demo (Gradio/Streamlit)

**Phase 7-9: Final Documentation**
- Comprehensive final report
- Research paper draft
- Viva preparation

---

## Slide 19: Potential Applications

### Real-World Use Cases

1. **Photo Organization**
   - Automatically categorize vacation photos
   - Create location-based albums without GPS

2. **Content Moderation**
   - Identify location context for safety
   - Detect inappropriate location-based content

3. **Historical Image Analysis**
   - Categorize archival photos without metadata
   - Study urbanization patterns over time

4. **Search & Recommendation**
   - "Show me more coastal images"
   - Location-aware content suggestions

5. **Privacy-Preserving Location Services**
   - Coarse location without exact coordinates
   - Share "I'm at the beach" not "lat/lon"

---

## Slide 20: Limitations & Future Work

### Current Limitations

1. **Semantic vs Precise Location**
   - Only 5 broad categories (not city-level)
   - Cannot distinguish similar locations

2. **Ambiguous Scenes**
   - Some images genuinely fit multiple categories
   - Single-label classification may be too restrictive

3. **Dataset Bias**
   - Limited to Places365 outdoor categories
   - May not generalize to all geographies

4. **Computational Resources**
   - Limited by Colab free tier
   - Cannot experiment extensively

### Future Enhancements

**Semester 2:**
- Advanced models (EfficientNet, ViT)
- Multi-label classification for ambiguous scenes
- Hierarchical predictions (e.g., Urban → City → Downtown)
- Ensemble methods

**Beyond Project:**
- Incorporate complementary signals (time of day, season)
- Fine-grained location subcategories
- Cross-dataset evaluation (generalization)
- Mobile deployment (TFLite, ONNX)
- Real-time inference optimization

---

## Slide 21: Research Contributions

### Alignment with Literature

**Building Upon:**
1. **Places365** (Zhou et al.) - Scene recognition dataset
2. **PlaNet** (Weyand et al.) - CNN-based geolocation
3. **ViT** (Dosovitskiy et al.) - Attention mechanisms
4. **Transfer Learning** - ImageNet pretraining

### Our Contribution
- Simplified 5-category location taxonomy
- Privacy-preserving semantic location estimation
- Comprehensive comparison across architectures (Sem 2)
- Open-source implementation with documentation

---

## Slide 22: Reproducibility

### Ensuring Reproducible Results

**Fixed Random Seeds:**
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

**Documented Environment:**
- requirements.txt with exact versions
- config.yaml for all hyperparameters
- GPU specifications documented

**Version Control:**
- All code on GitHub
- Tagged releases for milestones
- Commit history preserved

**Checkpointing:**
- Model weights saved
- Training history logged
- Full checkpoint (optimizer, scheduler)

**Instructions:**
- Detailed README with usage examples
- Jupyter notebooks with step-by-step demos
- Documentation for each module

---

## Slide 23: Team Contributions

### Role-Based Development

**Krishan Yadav (Technical Lead) - 35%**
- Model architecture design
- Training pipeline implementation
- Evaluation framework
- Technical coordination

**Aditi Sah (Data Lead) - 21.5%**
- Dataset preparation and mapping
- Data integrity verification
- Category distribution analysis
- Test data curation

**Anuj Kondawar (Preprocessing Lead) - 22%**
- Preprocessing pipeline development
- Data augmentation design
- DataLoader optimization
- Image quality assurance

**Jensi Paneliya (Documentation Lead) - 21.5%**
- Project documentation
- Report writing
- Presentation preparation
- Literature review compilation

### Collaborative Effort
- Weekly sync meetings (Fridays 5 PM)
- Code reviews via GitHub PRs
- Knowledge sharing sessions
- Pair programming for complex tasks

---

## Slide 24: Resources & References

### Key Papers
1. Zhou et al., "Places: A 10 Million Image Database for Scene Recognition"
2. Weyand et al., "PlaNet - Photo Geolocation with CNNs"
3. Dosovitskiy et al., "An Image is Worth 16x16 Words: ViT"

### Tools & Frameworks
- **PyTorch 2.0+** - Deep learning framework
- **torchvision** - Pretrained models
- **Google Colab** - Free GPU
- **scikit-learn** - Metrics
- **matplotlib/seaborn** - Visualizations

### Learning Resources
- CS231n (Stanford) - CNN course
- PyTorch official tutorials
- Places365 dataset website
- Research papers (21 total in literature review)

### Code Repository
- GitHub: DeepSceneLoc/DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning
- Full source code, documentation, notebooks
- MIT Places365 dataset (external) link provided

---

## Slide 25: Next Steps

### Immediate (After Review)
1. Incorporate feedback from review
2. Document lessons learned
3. Archive Semester 1 deliverables
4. Prepare Semester 2 environment

### Semester 2 Kickoff
1. **Week 1-2:** Implement EfficientNet-B0
2. **Week 3-4:** Implement Vision Transformer
3. **Week 5-6:** Comparative study and analysis
4. **Week 7-8:** Embeddings visualization
5. **Week 9-10:** Error analysis and optimization
6. **Week 11-12:** Final report and viva prep

### Milestones to Track
- Advanced model accuracies exceed baseline
- Embeddings show clear clustering
- Final report draft complete by Week 10
- Demo ready for viva

---

## Slide 26: Q&A

### Questions?

**Thank you for your attention!**

---

### Contact Information
- **Repository:** https://github.com/DeepSceneLoc/DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning
- **Documentation:** See `/docs` folder
- **Team:** Krishan Yadav, Aditi Sah, Anuj Kondawar, Jensi Paneliya

---

*This presentation covers the Semester 1 (30%) completion of DeepSceneLoc project. Detailed results, code, documentation, and visualizations available in the GitHub repository.*
