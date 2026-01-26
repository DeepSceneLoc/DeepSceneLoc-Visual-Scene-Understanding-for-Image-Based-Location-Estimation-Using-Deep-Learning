# Team Task Allocation - Semester 1
## DeepSceneLoc Project

**Team Size:** 4 members
**Semester Duration:** 7 weeks
**Project Coordinator:** Krishan Yadav

---

## Team Members and Primary Roles

### 1. Krishan Yadav (Coordinator & Technical Lead)
**Role:** Project Coordination, Architecture & Integration, Model Development

**Core Responsibilities:**
- Overall project timeline and coordination
- Technical architecture decisions and system design
- Model training pipeline design and optimization
- Integration of all components
- GitHub repository structure and management
- Final quality assurance and testing
- Weekly coordination meetings
- Troubleshooting technical issues

**Technical Ownership:**
- Model architecture and training framework
- Optimization and hyperparameter tuning
- Model evaluation and interpretation

---

### 2. Aditi Sah (Data & Literature Lead)
**Role:** Dataset Management, Literature Review, Research

**Core Responsibilities:**
- Lead literature review and research compilation
- Dataset acquisition and preparation
- Data organization and validation
- Category mapping and labeling
- Dataset documentation and analysis
- Research paper synthesis and summary
- Data quality assurance
- Create dataset validation scripts

**Technical Ownership:**
- Complete dataset pipeline (download → organize → validate)
- Literature review compilation and critical analysis
- Dataset statistics and distribution analysis

---

### 3. Anuj Kondawar (Preprocessing & Pipeline Lead)
**Role:** Data Preprocessing, Pipeline Development, Implementation

**Core Responsibilities:**
- Lead preprocessing pipeline design and implementation
- Image augmentation strategy and coding
- Data transformation and normalization
- Preprocessing testing and validation
- Optimization of data loading pipeline
- Documentation of preprocessing approach
- Create preprocessing modules and utilities
- Performance optimization

**Technical Ownership:**
- Complete image preprocessing pipeline
- Data augmentation implementation
- Performance benchmarking and optimization

---

### 4. Jensi Paneliya (Evaluation & Analysis Lead)
**Role:** Model Evaluation, Metrics & Analysis, Visualization

**Core Responsibilities:**
- Design and implement comprehensive evaluation framework
- Calculate all performance metrics
- Create visualizations and plots
- Perform error analysis and interpretation
- Generate evaluation reports
- Confusion matrix analysis
- Per-class performance breakdown
- Documentation of findings

**Technical Ownership:**
- Complete evaluation pipeline
- Performance metrics computation
- Error analysis and visualization
- Results interpretation and reporting

---

## Weekly Task Breakdown

### Week 1: Project Setup & Scope Freeze
**Coordinator:** Krishan Yadav
**All members contribute to project planning**

#### Krishan Yadav (Coordination)
- Create and configure GitHub repository
- Setup project structure and directories
- Coordinate team roles and responsibilities
- Setup communication channels
- Create initial project documentation
- **Deliverable:** GitHub repo initialized and ready

#### Aditi Sah (Research Foundation)
- Compile preliminary literature review outline
- Identify key papers to read
- Create research planning document
- Research data sources (Places365 availability)
- **Deliverable:** Literature review plan and data source research

#### Anuj Kondawar (Technical Planning)
- Plan preprocessing pipeline architecture
- Research image processing best practices
- Document preprocessing strategy
- Plan data augmentation approach
- **Deliverable:** Preprocessing design document

#### Jensi Paneliya (Evaluation Planning)
- Plan evaluation metrics framework
- Research appropriate metrics for location classification
- Document evaluation methodology
- Plan visualization approach
- **Deliverable:** Evaluation plan document

**Week 1 Deliverables:**
- GitHub repo live with structure
- Problem statement (all contribute)
- Project motivation (all contribute)
- Scope freeze document
- Research, preprocessing, and evaluation plans

---

### Week 2: Literature Review & Research
**Lead:** Aditi Sah
**Support:** All team members study assigned papers

#### Krishan Yadav
- Oversee overall literature review quality
- Compile final reference list
- Create integrated literature summary
- Document key insights for architecture decisions
- **Contribution:** 8-10 hours

#### Aditi Sah (Lead - 12 hours)
- Study "Places: A 10 Million Image Database" paper
- Study "SegVLAD" paper
- Study additional VPR papers
- Compile comprehensive dataset and scene understanding section
- Create dataset methodology document
- Synthesize key insights
- Create APA formatted references

#### Anuj Kondawar (12 hours)
- Study "Vision Transformer" paper
- Study "RepVGG" paper
- Study image preprocessing and augmentation papers
- Compile preprocessing methodology insights
- Document architecture implications for preprocessing
- Create technical notes

#### Jensi Paneliya (12 hours)
- Study "PlaNet" paper
- Study "Patch-NetVLAD" paper
- Study evaluation and metrics papers
- Compile model architecture and evaluation insights
- Document training and evaluation methodology
- Create technical notes

**Week 2 Deliverables:**
- Complete literature review document (4-5 pages)
- Individual paper summaries (one per person)
- IEEE formatted reference list
- Key insights document
- Architecture and methodology notes

---

### Week 3: Dataset Preparation
**Lead:** Aditi Sah
**Support:** Anuj Kondawar (splits), Jensi Paneliya (analysis)

#### Krishan Yadav (Support - 8 hours)
- Verify dataset structure and organization
- Quality assurance of preparation process
- Troubleshoot any data issues
- Approve final dataset structure

#### Aditi Sah (Lead - 14 hours)
- Download Places365 dataset (outdoor subset)
- Map 365 categories to 5 location types:
  - Urban, Rural, Coastal, Mountain, Forest
- Create label mapping (CSV)
- Organize directory structure
- Verify data integrity
- Document dataset methodology
- Create dataset statistics document

#### Anuj Kondawar (Support - 6 hours)
- Create train/validation/test splits (70/15/15)
- Generate split file lists
- Verify split balance across categories
- Create utility scripts for data loading

#### Jensi Paneliya (Support - 8 hours)
- Calculate comprehensive dataset statistics
- Generate distribution analysis
- Create visualization plots
- Identify class imbalances
- Generate statistics report

**Week 3 Deliverables:**
- Places365 dataset organized and ready
- Label mapping table
- Train/val/test split files
- Dataset statistics and analysis report
- Data loading utility scripts
- Dataset verification report

---

### Week 4: Preprocessing Pipeline Implementation
**Lead:** Anuj Kondawar
**Support:** Jensi Paneliya (testing), Aditi Sah (verification)

#### Krishan Yadav (Oversight - 8 hours)
- Review preprocessing architecture
- Ensure code quality and efficiency
- Integration planning for training
- Optimization recommendations

#### Aditi Sah (Support - 6 hours)
- Verify preprocessing on diverse image types
- Test label preservation
- Quality check on preprocessed samples

#### Anuj Kondawar (Lead - 16 hours)
- Implement image resizing (224×224)
- Implement normalization (ImageNet standard)
- Implement data augmentation:
  - Random horizontal flip
  - Random rotation
  - Color jitter
  - Brightness/contrast adjustment
  - Random crop
- Create preprocessing module (preprocessing.py)
- Create transforms module (transforms.py)
- Test with sample images
- Optimize data loading performance
- Document preprocessing pipeline

#### Jensi Paneliya (Support - 6 hours)
- Test preprocessing with batches
- Verify compatibility with model input
- Performance benchmark
- Verify output quality

**Week 4 Deliverables:**
- preprocessing.py module
- transforms.py with all augmentations
- Sample augmented images (20+ examples)
- Preprocessing documentation
- Performance report
- Batch testing results

---

### Week 5: Baseline Model (ResNet-50)
**Lead:** Krishan Yadav
**All members support execution**

#### Krishan Yadav (Lead - 20 hours)
- Design training architecture
- Load ResNet-50 pretrained model
- Implement custom classification head (2048 → 512 → 5)
- Configure optimizer and learning rate scheduling
- Configure loss function (CrossEntropyLoss)
- Implement checkpoint saving
- Implement early stopping logic
- Create training validation script
- Execute training on Google Colab
- Monitor training convergence and metrics
- Generate training curves
- Save training logs
- Troubleshoot any training issues
- Save final model checkpoint

#### Aditi Sah (Support - 6 hours)
- Final data preparation for training
- Verify data loading pipeline
- Quality check on training batches

#### Anuj Kondawar (Support - 6 hours)
- Finalize preprocessing for training
- Optimize data pipeline
- Test augmentation with model input

#### Jensi Paneliya (Support - 8 hours)
- Assist with training monitoring
- Help log and track metrics
- Support model checkpointing
- Assist with training documentation

**Week 5 Deliverables:**
- Trained ResNet-50 checkpoint
- Training logs and metrics
- Training curves (accuracy, loss)
- Model configuration file
- Training summary document

---

### Week 6: Model Evaluation & Analysis
**Lead:** Krishan Yadav (Technical Analysis)
**Support:** Aditi Sah, Anuj Kondawar (data/preprocessing verification)

#### Krishan Yadav (Lead - 20 hours)
- Design overall evaluation framework
- Evaluate ResNet-50 on test set
- Calculate metrics:
  - Overall accuracy
  - Per-class accuracy
  - Precision, recall, F1-score
- Generate confusion matrix
- Perform error analysis:
  - Identify misclassification patterns
  - Document challenging cases
  - Analyze error types
- Create visualizations:
  - Confusion matrix heatmap
  - Per-class accuracy bars
  - Error distribution
- Quality assurance of evaluation
- Final analysis and interpretation

#### Aditi Sah (Support - 6 hours)
- Prepare test dataset
- Verify test data integrity
- Assist in interpretation

#### Anuj Kondawar (Support - 6 hours)
- Verify test preprocessing correct
- Data pipeline validation

#### Jensi Paneliya (Support - 8 hours)
- Assist with metric calculations
- Help create evaluation visualizations
- Support error analysis documentation
- Generate evaluation report sections

**Week 6 Deliverables:**
- Evaluation metrics table
- Confusion matrix (numerical + visualization)
- Per-class breakdown report
- Error analysis document
- Visualization plots (5+ visualizations)
- Comprehensive evaluation report

---

### Week 7: Documentation & Presentation
**Lead:** Jensi Paneliya (Documentation & Report)
**Support:** Krishan Yadav (Technical Review), Aditi Sah, Anuj Kondawar (Content)

#### Krishan Yadav (Technical Review - 8 hours)
- Review all documentation for technical accuracy
- Create system architecture diagram
- Final quality review of all materials
- Oversee GitHub documentation structure
- Approve final deliverables

#### Aditi Sah (Content Support - 8 hours)
- Prepare dataset methodology content/notes
- Provide key dataset insights and findings
- Create dataset visualization data
- Review dataset sections

#### Anuj Kondawar (Content Support - 3 hours)
- Prepare preprocessing methodology content/notes
- Provide key preprocessing insights and results
- Create preprocessing visualization data
- Review preprocessing sections

#### Jensi Paneliya (Documentation Lead - 10 hours)
- **Lead complete documentation writing:**
  - Write introduction and problem statement
  - Write literature review synthesis
  - Integrate dataset and preprocessing sections
  - Write model and evaluation results section
  - Write conclusions and future work
- **Create comprehensive final report (10-12 pages)**
- **Create PowerPoint presentation (15-18 slides):**
  - Project overview slides
  - Problem statement and motivation
  - Literature review summary
  - Dataset, preprocessing, and model slides
  - Results and performance metrics
  - Analysis and findings
  - Conclusions and next steps
- **Update comprehensive README.md**
- **Organize all GitHub documentation**
- **Create presentation speaker notes**
- **Coordinate final review with team**

**Week 7 Deliverables:**
- Complete technical report (12-15 pages)
- System architecture diagram
- Execution flow documentation
- PowerPoint presentation (15-18 slides)
- Updated comprehensive README.md
- Speaker notes for presentation
- All materials organized on GitHub
- Final project summary document

---

## Workload Distribution Summary

## Workload Distribution Summary

### Distribution Overview

All team members contribute to the project with distinct leadership areas:

- **Krishan Yadav:** Technical Architecture & Coordination (25%)
- **Jensi Paneliya:** Documentation & Presentation (25%)
- **Aditi Sah:** Literature Review & Dataset Management (25%)
- **Anuj Kondawar:** Preprocessing & Pipeline Development (25%)

**Total Team Effort:** 260 hours (equally distributed across all members)

---

### Detailed Weekly Breakdown

The following weekly breakdown provides granular task allocation showing how 260 hours are distributed across 7 weeks:

| Team Member | Week 1 | Week 2 | Week 3 | Week 4 | Week 5 | Week 6 | Week 7 | Total | % |
|---|---|---|---|---|---|---|---|---|---|
| Krishan Yadav | 12 | 8 | 8 | 8 | 20 | 20 | 15 | 91 | 35 |
| Aditi Sah | 8 | 12 | 14 | 6 | 6 | 6 | 4 | 56 | 21.5 |
| Anuj Kondawar | 8 | 12 | 6 | 16 | 6 | 6 | 3 | 57 | 22 |
| Jensi Paneliya | 6 | 12 | 6 | 6 | 8 | 8 | 10 | 56 | 21.5 |

**Note:** Weekly distribution varies based on task complexity and phase requirements. Each person's total contribution reflects their domain leadership responsibilities.

---

## Responsibility Matrix by Area

### Literature Review & Research
- Aditi Sah: 40%
- Krishan Yadav: 25%
- Anuj Kondawar: 20%
- Jensi Paneliya: 15%

### Dataset Management & Preparation
- Aditi Sah: 45%
- Jensi Paneliya: 20%
- Anuj Kondawar: 15%
- Krishan Yadav: 20%

### Image Preprocessing & Pipeline
- Anuj Kondawar: 50%
- Jensi Paneliya: 20%
- Aditi Sah: 15%
- Krishan Yadav: 15%

### Model Training & Architecture
- Krishan Yadav: 60%
- Anuj Kondawar: 20%
- Jensi Paneliya: 12%
- Aditi Sah: 8%

### Evaluation & Analysis
- Krishan Yadav: 55%
- Jensi Paneliya: 20%
- Anuj Kondawar: 15%
- Aditi Sah: 10%

### Documentation & Presentation
- Jensi Paneliya: 40%
- Krishan Yadav: 35%
- Aditi Sah: 15%
- Anuj Kondawar: 10%

---

## Key Differences in This Allocation

## Key Responsibilities

**Krishan Yadav** - Technical Architecture & Coordination
- Overall project architecture and system design
- Technical decision-making and troubleshooting
- Model training and evaluation framework
- Quality assurance and code review
- Integration of all components

**Jensi Paneliya** - Documentation & Presentation
- Comprehensive report writing
- Presentation creation and speaker notes
- Synthesizing work from all domains
- Professional final deliverables
- Project communication strategy

**Aditi Sah** - Literature Review & Dataset
- Research compilation and analysis
- Dataset acquisition and organization
- Data quality assurance and validation
- Academic foundation and credibility
- Data methodology documentation

**Anuj Kondawar** - Preprocessing & Pipeline Development
- Data pipeline architecture
- Image preprocessing and augmentation
- Data transformation and optimization
- Infrastructure and utilities development
- Performance optimization

---

## Communication and Meetings

### Weekly Sync Meetings
- **Frequency:** Once per week (Friday recommended)
- **Duration:** 45 minutes
- **Format:** 
  - Progress updates (10 min each person)
  - Blockers and solutions (15 min)
  - Next week planning (5 min)

### Task Handoffs and Integration
- End of week: Domain lead completes deliverables
- Beginning of next week: Handoff and integration
- Krishan Yadav ensures compatibility and quality

### Communication Channels
- Primary: Team messaging/chat
- Secondary: Weekly sync meetings
- Tertiary: GitHub issues and pull requests

---

## Success Criteria

### Team Success
- All Week 7 deliverables completed
- Baseline model accuracy: 60-70%
- Professional presentation and report
- Clean, well-documented codebase
- Strong foundation for Semester 2
- Effective team collaboration

---

## Notes

- Equal participation across all team members
- Each person leads distinct domain expertise
- Success requires genuine collaboration and communication
- Clear task ownership and delivery expectations
- All domains equally important to project success

