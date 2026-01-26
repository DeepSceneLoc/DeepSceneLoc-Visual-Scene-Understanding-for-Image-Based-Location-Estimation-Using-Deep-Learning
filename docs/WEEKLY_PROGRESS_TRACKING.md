# Weekly Progress Tracking - Semester 1
## DeepSceneLoc Project

**Project Period:** 7 weeks (January 20 - March 7, 2026)
**Update Frequency:** Every Friday by 6:00 PM

---

## Week 1: Project Setup & Scope Freeze
**Duration:** January 20-26, 2026
**Status:** [  ] Not Started | [  ] In Progress | [  ] Complete

### Deliverables Checklist

#### Krishan Yadav (12 hours planned)
- [ ] Create GitHub repository
- [ ] Setup repository structure
- [ ] Create and configure README.md
- [ ] Add LICENSE file
- [ ] Write problem statement (1-2 pages)
- [ ] Write project motivation
- [ ] Coordinate team assignments
- [ ] Setup communication channels

**Hours Completed:** __ / 12 | **Blockers:** ___________

#### Aditi Sah (8 hours planned)
- [ ] Review problem statement
- [ ] Provide feedback on scope
- [ ] Document literature review plan
- [ ] Prepare for Week 2

**Hours Completed:** __ / 8 | **Blockers:** ___________

#### Anuj Kondawar (8 hours planned)
- [ ] Review scope requirements
- [ ] Plan preprocessing pipeline architecture
- [ ] Document data requirements
- [ ] Prepare technical planning

**Hours Completed:** __ / 8 | **Blockers:** ___________

#### Jensi Paneliya (6 hours planned)
- [ ] Review scope and requirements
- [ ] Plan model and evaluation approach
- [ ] Document training strategy
- [ ] Prepare documentation plan

**Hours Completed:** __ / 6 | **Blockers:** ___________

---

## Week 2: Literature Review & Research
**Duration:** January 27 - February 2, 2026
**Status:** [  ] Not Started | [  ] In Progress | [  ] Complete

### Deliverables Checklist

#### Krishan Yadav (8 hours planned)
- [ ] Oversee literature review process
- [ ] Compile reference list
- [ ] Create integrated literature summary
- [ ] Review quality of submissions

**Hours Completed:** __ / 8 | **Blockers:** ___________

#### Aditi Sah (12 hours planned)
- [ ] Download Places paper
- [ ] Read and analyze Places paper
- [ ] Download SegVLAD paper
- [ ] Read and analyze SegVLAD paper
- [ ] Write paper summary (dataset insights)
- [ ] Document key findings

**Hours Completed:** __ / 12 | **Blockers:** ___________

#### Anuj Kondawar (12 hours planned)
- [ ] Download Vision Transformer paper
- [ ] Read and analyze ViT paper
- [ ] Download RepVGG paper
- [ ] Read and analyze RepVGG paper
- [ ] Write paper summary (preprocessing insights)
- [ ] Document key findings

**Hours Completed:** __ / 12 | **Blockers:** ___________

#### Jensi Paneliya (12 hours planned)
- [ ] Download PlaNet paper
- [ ] Read and analyze PlaNet paper
- [ ] Download Patch-NetVLAD paper
- [ ] Read and analyze Patch-NetVLAD paper
- [ ] Write paper summary (model insights)
- [ ] Document key findings

**Hours Completed:** __ / 12 | **Blockers:** ___________

---

## Week 3: Dataset Preparation
**Duration:** February 3-9, 2026
**Status:** [  ] Not Started | [  ] In Progress | [  ] Complete

### Deliverables Checklist

#### Krishan Yadav (8 hours planned)
- [ ] Verify dataset structure
- [ ] Quality assurance of preparation
- [ ] Approve final organization
- [ ] Troubleshoot any issues

**Hours Completed:** __ / 8 | **Blockers:** ___________

#### Aditi Sah (14 hours planned)
- [ ] Begin Places365 download
- [ ] Complete dataset download
- [ ] Map 365 categories to 5 location types
- [ ] Create label mapping CSV
- [ ] Organize directory structure
- [ ] Verify data integrity
- [ ] Generate dataset statistics

**Hours Completed:** __ / 14 | **Blockers:** ___________

#### Anuj Kondawar (6 hours planned)
- [ ] Create train/validation/test splits (70/15/15)
- [ ] Generate train.txt file
- [ ] Generate val.txt file
- [ ] Generate test.txt file
- [ ] Verify split balance across categories

**Hours Completed:** __ / 6 | **Blockers:** ___________

#### Jensi Paneliya (6 hours planned)
- [ ] Analyze dataset distribution
- [ ] Create class distribution visualization
- [ ] Identify class imbalances
- [ ] Generate statistics report
- [ ] Document findings

**Hours Completed:** __ / 6 | **Blockers:** ___________

---

## Week 4: Image Preprocessing Pipeline
**Duration:** February 10-16, 2026
**Status:** [  ] Not Started | [  ] In Progress | [  ] Complete

### Deliverables Checklist

#### Krishan Yadav (8 hours planned)
- [ ] Review preprocessing architecture
- [ ] Provide code quality feedback
- [ ] Verify optimization opportunities
- [ ] Approve final pipeline

**Hours Completed:** __ / 8 | **Blockers:** ___________

#### Aditi Sah (6 hours planned)
- [ ] Verify preprocessing on diverse image types
- [ ] Test label preservation
- [ ] Quality check on preprocessed samples
- [ ] Provide feedback to Anuj

**Hours Completed:** __ / 6 | **Blockers:** ___________

#### Anuj Kondawar (16 hours planned)
- [ ] Implement image resizing (224x224)
- [ ] Implement ImageNet normalization
- [ ] Implement random horizontal flip
- [ ] Implement random rotation
- [ ] Implement color jitter
- [ ] Implement brightness/contrast adjustment
- [ ] Implement random crop
- [ ] Create preprocessing.py module
- [ ] Create transforms.py module
- [ ] Test with sample images
- [ ] Create sample augmented image outputs
- [ ] Optimize data loading performance
- [ ] Document preprocessing pipeline

**Hours Completed:** __ / 16 | **Blockers:** ___________

#### Jensi Paneliya (6 hours planned)
- [ ] Test preprocessing with batches
- [ ] Verify model input compatibility
- [ ] Performance benchmark
- [ ] Verify output quality
- [ ] Provide feedback

**Hours Completed:** __ / 6 | **Blockers:** ___________

---

## Week 5: Baseline Model Training (ResNet-50)
**Duration:** February 17-23, 2026
**Status:** [  ] Not Started | [  ] In Progress | [  ] Complete

### Deliverables Checklist

#### Krishan Yadav (20 hours planned)
- [ ] Design training architecture
- [ ] Load ResNet-50 pretrained model
- [ ] Implement custom classification head (2048 → 512 → 5)
- [ ] Configure Adam optimizer
- [ ] Configure learning rate scheduling
- [ ] Configure CrossEntropyLoss
- [ ] Implement checkpoint saving
- [ ] Implement early stopping logic
- [ ] Create training validation script
- [ ] Setup Google Colab environment
- [ ] Execute training
- [ ] Monitor training convergence
- [ ] Generate training curves
- [ ] Save training logs
- [ ] Troubleshoot any training issues
- [ ] Save final model checkpoint

**Hours Completed:** __ / 20 | **Blockers:** ___________

#### Aditi Sah (6 hours planned)
- [ ] Final data verification
- [ ] Verify data loading pipeline
- [ ] Quality check on training batches
- [ ] Ensure proper data formatting

**Hours Completed:** __ / 6 | **Blockers:** ___________

#### Anuj Kondawar (6 hours planned)
- [ ] Finalize preprocessing for training
- [ ] Optimize data loading speed
- [ ] Test augmentation with model input
- [ ] Verify pipeline stability

**Hours Completed:** __ / 6 | **Blockers:** ___________

#### Jensi Paneliya (8 hours planned)
- [ ] Assist with training monitoring
- [ ] Help track metrics
- [ ] Support model checkpointing
- [ ] Assist with training documentation

**Hours Completed:** __ / 8 | **Blockers:** ___________

---

## Week 6: Model Evaluation & Analysis
**Duration:** February 24 - March 2, 2026
**Status:** [  ] Not Started | [  ] In Progress | [  ] Complete

### Deliverables Checklist

#### Krishan Yadav (20 hours planned)
- [ ] Design evaluation framework
- [ ] Load trained ResNet-50 model
- [ ] Evaluate on test set
- [ ] Calculate overall accuracy
- [ ] Calculate per-class accuracy
- [ ] Calculate precision scores
- [ ] Calculate recall scores
- [ ] Calculate F1-scores
- [ ] Generate confusion matrix (numerical)
- [ ] Generate confusion matrix (heatmap visualization)
- [ ] Identify misclassification patterns
- [ ] Document challenging cases
- [ ] Analyze error types
- [ ] Create per-class accuracy bar chart
- [ ] Create error distribution plot
- [ ] Generate comprehensive analysis report

**Hours Completed:** __ / 20 | **Blockers:** ___________

#### Aditi Sah (6 hours planned)
- [ ] Prepare test dataset
- [ ] Verify test data integrity
- [ ] Assist in interpretation
- [ ] Review results

**Hours Completed:** __ / 6 | **Blockers:** ___________

#### Anuj Kondawar (6 hours planned)
- [ ] Verify test preprocessing
- [ ] Validate data pipeline for evaluation
- [ ] Ensure consistency

**Hours Completed:** __ / 6 | **Blockers:** ___________

#### Jensi Paneliya (8 hours planned)
- [ ] Assist with metric calculations
- [ ] Help create visualizations
- [ ] Support error analysis documentation
- [ ] Generate evaluation report sections

**Hours Completed:** __ / 8 | **Blockers:** ___________

---

## Week 7: Documentation & Presentation
**Duration:** March 3-7, 2026
**Status:** [  ] Not Started | [  ] In Progress | [  ] Complete

### Deliverables Checklist

#### Krishan Yadav (15 hours planned)
- [ ] Review all documentation for accuracy
- [ ] Create system architecture diagram
- [ ] Create execution flow documentation
- [ ] Final quality review of all materials
- [ ] Oversee GitHub documentation structure
- [ ] Approve final deliverables
- [ ] Write technical sections and appendices

**Hours Completed:** __ / 15 | **Blockers:** ___________

#### Aditi Sah (4 hours planned)
- [ ] Prepare dataset methodology content
- [ ] Provide key dataset insights
- [ ] Create dataset visualization data
- [ ] Review dataset sections

**Hours Completed:** __ / 4 | **Blockers:** ___________

#### Anuj Kondawar (3 hours planned)
- [ ] Prepare preprocessing methodology content
- [ ] Provide key preprocessing insights
- [ ] Create preprocessing visualization data
- [ ] Review preprocessing sections

**Hours Completed:** __ / 3 | **Blockers:** ___________

#### Jensi Paneliya (10 hours planned)
- [ ] Write introduction and problem statement
- [ ] Write literature review synthesis
- [ ] Integrate dataset section
- [ ] Integrate preprocessing section
- [ ] Write model and evaluation results section
- [ ] Write conclusions and future work
- [ ] Create comprehensive final report (10-12 pages)
- [ ] Create PowerPoint presentation (15-18 slides)
- [ ] Write project overview slides
- [ ] Write problem statement and motivation
- [ ] Write literature review summary
- [ ] Write dataset, preprocessing, model slides
- [ ] Write results and performance metrics slides
- [ ] Write analysis and findings slides
- [ ] Write conclusions and next steps
- [ ] Update comprehensive README.md
- [ ] Organize all GitHub documentation
- [ ] Create presentation speaker notes
- [ ] Coordinate final review with team

**Hours Completed:** __ / 10 | **Blockers:** ___________

---

## Quick Reference - Weekly Status

| Week | Focus Area | Lead | Status |
|---|---|---|---|
| 1 | Setup & Planning | Krishan | [  ] Complete |
| 2 | Literature Review | All | [  ] Complete |
| 3 | Dataset Preparation | Aditi | [  ] Complete |
| 4 | Preprocessing Pipeline | Anuj | [  ] Complete |
| 5 | Model Training | Krishan | [  ] Complete |
| 6 | Evaluation & Analysis | Krishan | [  ] Complete |
| 7 | Documentation | Jensi | [  ] Complete |

---

## Team Hours Summary

**Cumulative Progress:**

- Krishan Yadav: __ / 91 hours (__%)
- Aditi Sah: __ / 56 hours (__%)
- Anuj Kondawar: __ / 57 hours (__%)
- Jensi Paneliya: __ / 56 hours (__%)

**Total Effort:** __ / 260 hours (__%)

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



