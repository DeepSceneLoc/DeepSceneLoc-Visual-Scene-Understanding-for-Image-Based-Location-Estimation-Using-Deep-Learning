# Weekly Progress Tracking - Semester 1
## DeepSceneLoc Project

**Project Period:** 7 weeks (January 20 - March 7, 2026)
**Update Frequency:** Every Friday by 6:00 PM

---

## Week 1: Project Setup & Scope Freeze
**Duration:** January 20-26, 2026
**Status:** [  ] Not Started | [  ] In Progress | [x] Complete

### Deliverables Checklist

#### Krishan Yadav (12 hours planned)
- [x] Create GitHub repository
- [x] Setup repository structure
- [x] Create and configure README.md
- [x] Add LICENSE file
- [x] Write problem statement (1-2 pages)
- [x] Write project motivation
- [x] Coordinate team assignments
- [x] Setup communication channels

**Hours Completed:** 12 / 12 | **Blockers:** None

#### Aditi Sah (8 hours planned)
- [x] Review problem statement
- [x] Provide feedback on scope
- [x] Document literature review plan
- [x] Prepare for Week 2

**Hours Completed:** 8 / 8 | **Blockers:** None

#### Anuj Kondawar (8 hours planned)
- [x] Review scope requirements
- [x] Plan preprocessing pipeline architecture
- [x] Document data requirements
- [x] Prepare technical planning

**Hours Completed:** 8 / 8 | **Blockers:** None

#### Jensi Paneliya (6 hours planned)
- [x] Review scope and requirements
- [x] Plan model and evaluation approach
- [x] Document training strategy
- [x] Prepare documentation plan

**Hours Completed:** 6 / 6 | **Blockers:** None

---

## Week 2: Literature Review & Research
**Duration:** January 27 - February 2, 2026
**Status:** [  ] Not Started | [  ] In Progress | [x] Complete

### Deliverables Checklist

#### Krishan Yadav (8 hours planned)
- [x] Oversee literature review process
- [x] Compile reference list
- [x] Create integrated literature summary
- [x] Review quality of submissions

**Hours Completed:** 8 / 8 | **Blockers:** None

#### Aditi Sah (12 hours planned)
- [x] Download Places paper
- [x] Read and analyze Places paper
- [x] Download SegVLAD paper
- [x] Read and analyze SegVLAD paper
- [x] Write paper summary (dataset insights)
- [x] Document key findings

**Hours Completed:** 12 / 12 | **Blockers:** None

#### Anuj Kondawar (12 hours planned)
- [x] Download Vision Transformer paper
- [x] Read and analyze ViT paper
- [x] Download RepVGG paper
- [x] Read and analyze RepVGG paper
- [x] Write paper summary (preprocessing insights)
- [x] Document key findings

**Hours Completed:** 12 / 12 | **Blockers:** None

#### Jensi Paneliya (12 hours planned)
- [x] Download PlaNet paper
- [x] Read and analyze PlaNet paper
- [x] Download Patch-NetVLAD paper
- [x] Read and analyze Patch-NetVLAD paper
- [x] Write paper summary (model insights)
- [x] Document key findings

**Hours Completed:** 12 / 12 | **Blockers:** None

---

## Week 3: Dataset Preparation
**Duration:** February 3-9, 2026
**Status:** [  ] Not Started | [  ] In Progress | [x] Complete

### Deliverables Checklist

#### Krishan Yadav (8 hours planned)
- [x] Verify dataset structure
- [x] Quality assurance of preparation
- [x] Approve final organization
- [x] Troubleshoot any issues

**Hours Completed:** 8 / 8 | **Blockers:** None

#### Aditi Sah (14 hours planned)
- [x] Begin Places365 download
- [x] Complete dataset download
- [x] Map 365 categories to 5 location types
- [x] Create label mapping CSV
- [x] Organize directory structure
- [x] Verify data integrity
- [x] Generate dataset statistics

**Hours Completed:** 14 / 14 | **Blockers:** None

#### Anuj Kondawar (6 hours planned)
- [x] Create train/validation/test splits (70/15/15)
- [x] Generate train.txt file
- [x] Generate val.txt file
- [x] Generate test.txt file
- [x] Verify split balance across categories

**Hours Completed:** 6 / 6 | **Blockers:** None

#### Jensi Paneliya (6 hours planned)
- [x] Analyze dataset distribution
- [x] Create class distribution visualization
- [x] Identify class imbalances
- [x] Generate statistics report
- [x] Document findings

**Hours Completed:** 6 / 6 | **Blockers:** None

---

## Week 4: Image Preprocessing Pipeline
**Duration:** February 10-16, 2026
**Status:** [  ] Not Started | [  ] In Progress | [x] Complete

### Deliverables Checklist

#### Krishan Yadav (8 hours planned)
- [x] Review preprocessing architecture
- [x] Provide code quality feedback
- [x] Verify optimization opportunities
- [x] Approve final pipeline

**Hours Completed:** 8 / 8 | **Blockers:** None

#### Aditi Sah (6 hours planned)
- [x] Verify preprocessing on diverse image types
- [x] Test label preservation
- [x] Quality check on preprocessed samples
- [x] Provide feedback to Anuj

**Hours Completed:** 6 / 6 | **Blockers:** None

#### Anuj Kondawar (16 hours planned)
- [x] Implement image resizing (224x224)
- [x] Implement ImageNet normalization
- [x] Implement random horizontal flip
- [x] Implement random rotation
- [x] Implement color jitter
- [x] Implement brightness/contrast adjustment
- [x] Implement random crop
- [x] Create preprocessing.py module
- [x] Create transforms.py module
- [x] Test with sample images
- [x] Create sample augmented image outputs
- [x] Optimize data loading performance
- [x] Document preprocessing pipeline

**Hours Completed:** 16 / 16 | **Blockers:** None

#### Jensi Paneliya (6 hours planned)
- [x] Test preprocessing with batches
- [x] Verify model input compatibility
- [x] Performance benchmark
- [x] Verify output quality
- [x] Provide feedback

**Hours Completed:** 6 / 6 | **Blockers:** None

---

## Week 5: Baseline Model Training (ResNet-50)
**Duration:** February 17-23, 2026
**Status:** [  ] Not Started | [  ] In Progress | [x] Complete

### Deliverables Checklist

#### Krishan Yadav (20 hours planned)
- [x] Design training architecture
- [x] Load ResNet-50 pretrained model
- [x] Implement custom classification head (2048 → 512 → 5)
- [x] Configure Adam optimizer
- [x] Configure learning rate scheduling
- [x] Configure CrossEntropyLoss
- [x] Implement checkpoint saving
- [x] Implement early stopping logic
- [x] Create training validation script
- [x] Setup Google Colab environment
- [x] Execute training
- [x] Monitor training convergence
- [x] Generate training curves
- [x] Save training logs
- [x] Troubleshoot any training issues
- [x] Save final model checkpoint

**Hours Completed:** 20 / 20 | **Blockers:** None

#### Aditi Sah (6 hours planned)
- [x] Final data verification
- [x] Verify data loading pipeline
- [x] Quality check on training batches
- [x] Ensure proper data formatting

**Hours Completed:** 6 / 6 | **Blockers:** None

#### Anuj Kondawar (6 hours planned)
- [x] Finalize preprocessing for training
- [x] Optimize data loading speed
- [x] Test augmentation with model input
- [x] Verify pipeline stability

**Hours Completed:** 6 / 6 | **Blockers:** Dataset download (27GB) pending, blocking actual training run

#### Jensi Paneliya (8 hours planned)
- [x] Assist with training monitoring
- [x] Help track metrics
- [x] Support model checkpointing
- [x] Assist with training documentation

**Hours Completed:** 8 / 8 | **Blockers:** None

---

## Week 6: Model Evaluation & Analysis
**Duration:** February 24 - March 2, 2026
**Status:** [  ] Not Started | [  ] In Progress | [x] Complete

### Deliverables Checklist

#### Krishan Yadav (20 hours planned)
- [x] Design evaluation framework
- [x] Load trained ResNet-50 model
- [x] Evaluate on test set
- [x] Calculate overall accuracy
- [x] Calculate per-class accuracy
- [x] Calculate precision scores
- [x] Calculate recall scores
- [x] Calculate F1-scores
- [x] Generate confusion matrix (numerical)
- [x] Generate confusion matrix (heatmap visualization)
- [x] Identify misclassification patterns
- [x] Document challenging cases
- [x] Analyze error types
- [x] Create per-class accuracy bar chart
- [x] Create error distribution plot
- [x] Generate comprehensive analysis report

**Hours Completed:** 20 / 20 | **Blockers:** None

#### Aditi Sah (6 hours planned)
- [x] Prepare test dataset
- [x] Verify test data integrity
- [x] Assist in interpretation
- [x] Review results

**Hours Completed:** 6 / 6 | **Blockers:** None

#### Anuj Kondawar (6 hours planned)
- [x] Verify test preprocessing
- [x] Validate data pipeline for evaluation
- [x] Ensure consistency

**Hours Completed:** 6 / 6 | **Blockers:** Trained model checkpoint pending (depends on Week 5 training)

#### Jensi Paneliya (8 hours planned)
- [x] Assist with metric calculations
- [x] Help create visualizations
- [x] Support error analysis documentation
- [x] Generate evaluation report sections

**Hours Completed:** 8 / 8 | **Blockers:** None

---

## Week 7: Documentation & Presentation
**Duration:** March 3-7, 2026
**Status:** [  ] Not Started | [  ] In Progress | [x] Complete

### Deliverables Checklist

#### Krishan Yadav (15 hours planned)
- [x] Review all documentation for accuracy
- [x] Create system architecture diagram
- [x] Create execution flow documentation
- [x] Final quality review of all materials
- [x] Oversee GitHub documentation structure
- [x] Approve final deliverables
- [x] Write technical sections and appendices

**Hours Completed:** 15 / 15 | **Blockers:** None

#### Aditi Sah (4 hours planned)
- [x] Prepare dataset methodology content
- [x] Provide key dataset insights
- [x] Create dataset visualization data
- [x] Review dataset sections

**Hours Completed:** 4 / 4 | **Blockers:** None

#### Anuj Kondawar (3 hours planned)
- [x] Prepare preprocessing methodology content
- [x] Provide key preprocessing insights
- [x] Create preprocessing visualization data
- [x] Review preprocessing sections

**Hours Completed:** 3 / 3 | **Blockers:** None

#### Jensi Paneliya (10 hours planned)
- [x] Write introduction and problem statement
- [x] Write literature review synthesis
- [x] Integrate dataset section
- [x] Integrate preprocessing section
- [x] Write model and evaluation results section
- [x] Write conclusions and future work
- [x] Create comprehensive final report (10-12 pages)
- [x] Create PowerPoint presentation (15-18 slides)
- [x] Write project overview slides
- [x] Write problem statement and motivation
- [x] Write literature review summary
- [x] Write dataset, preprocessing, model slides
- [x] Write results and performance metrics slides
- [x] Write analysis and findings slides
- [x] Write conclusions and next steps
- [x] Update comprehensive README.md
- [x] Organize all GitHub documentation
- [x] Create presentation speaker notes
- [x] Coordinate final review with team

**Hours Completed:** 10 / 10 | **Blockers:** None

---

## Week 8: Training Continuation
**Duration:** March 8-14, 2026
**Status:** [x] In Progress

### Deliverables Checklist

#### Krishan Yadav (10 hours planned)
- [x] Fix optimizer state device mismatch issue.
- [x] Resume training from checkpoint.
- [x] Monitor live training progress.
- [ ] Analyze training metrics.

**Hours Completed:** 6/10 | **Blockers:** None

#### Aditi Sah (6 hours planned)
- [x] Verify training data integrity.
- [x] Assist in monitoring training progress.
- [ ] Document training updates.

**Hours Completed:** 4/6 | **Blockers:** None

#### Anuj Kondawar (6 hours planned)
- [x] Verify preprocessing pipeline stability.
- [x] Ensure data augmentation consistency.
- [ ] Support training monitoring.

**Hours Completed:** 4/6 | **Blockers:** None

#### Jensi Paneliya (6 hours planned)
- [x] Update documentation with training progress.
- [ ] Assist in analyzing training metrics.
- [ ] Prepare updated visualizations.

**Hours Completed:** 2/6 | **Blockers:** None

---

## Week 9: Places365 Training Completion
**Duration:** March 15-21, 2026
**Status:** [x] Complete

### Deliverables Checklist

#### Krishan Yadav (12 hours planned)
- [x] Finalize training on Places365 dataset.
- [x] Save best model checkpoint.
- [x] Generate training history plots.
- [x] Document training details.

**Hours Completed:** 12/12 | **Blockers:** None

#### Aditi Sah (8 hours planned)
- [x] Verify training data integrity.
- [x] Assist in generating evaluation metrics.
- [x] Review training history plots.
- [x] Document evaluation results.

**Hours Completed:** 8/8 | **Blockers:** None

#### Anuj Kondawar (6 hours planned)
- [x] Verify preprocessing pipeline consistency.
- [x] Ensure data augmentation quality.
- [x] Support training monitoring.
- [x] Document preprocessing updates.

**Hours Completed:** 6/6 | **Blockers:** None

#### Jensi Paneliya (6 hours planned)
- [x] Create visualizations for evaluation metrics.
- [x] Update documentation with training progress.
- [x] Assist in preparing final report.

**Hours Completed:** 6/6 | **Blockers:** None

---

## Quick Reference - Weekly Status

| Week | Focus Area | Lead | Status |
|---|---|---|---|
| 1 | Setup & Planning | Krishan | [x] Complete |
| 2 | Literature Review | All | [x] Complete |
| 3 | Dataset Preparation | Aditi | [x] Complete |
| 4 | Preprocessing Pipeline | Anuj | [x] Complete |
| 5 | Model Training | Krishan | [x] Complete |
| 6 | Evaluation & Analysis | Krishan | [x] Complete |
| 7 | Documentation | Jensi | [x] Complete |
| 8 | Training Continuation | All | [x] In Progress |
| 9 | Places365 Training Completion | All | [x] Complete |

---

## Team Hours Summary

**Cumulative Progress:**

- Krishan Yadav: 91 / 91 hours (100%)
- Aditi Sah: 56 / 56 hours (100%)
- Anuj Kondawar: 57 / 57 hours (100%)
- Jensi Paneliya: 56 / 56 hours (100%)

**Total Effort:** 260 / 260 hours (100%)

---

## Critical Path & Dependencies

```
Week 1: Setup (Krishan)
    ↓
Week 2: Literature (All)
    ↓
Week 3: Dataset (Aditi) ← CRITICAL: Unblocks Weeks 4-6
    ↓
Week 4: Preprocessing (Anuj) ← Enables training
    ↓
Week 5: Training (Krishan) ← Enables evaluation
    ↓
Week 6: Evaluation (Krishan) ← Enables documentation
    ↓
Week 7: Documentation (Jensi) ← Final deliverables
```

---

## Issues & Blockers Log

| Week | Team Member | Issue | Status | Resolution |
|---|---|---|---|---|
| | | | [  ] Open | |
| | | | [  ] In Progress | |
| | | | [  ] Resolved | |

---

## Notes

- Check off items as they are completed
- Update hours completed every Friday by 6:00 PM
- Report blockers immediately (don't wait for Friday)
- Weekly sync meeting: Friday 5:00 PM
- Escalate critical path risks to Krishan immediately



