# Repository Cleanup Plan

## Current Issues
- Training scripts scattered in root (should be in `scripts/training/`)
- Untracked new files need to be added to git
- Root directory cluttered with various file types

## Proposed Structure

```
DeepSceneLoc/
├── src/                    # Core library code
│   ├── models/
│   ├── preprocessing/
│   └── evaluation/
├── scripts/               # Executable scripts
│   ├── training/         # NEW: Training runners
│   │   ├── run_training_efficientnet_b0.py
│   │   ├── run_training_vit_b16.py
│   │   ├── run_training_resnet50.py
│   │   └── run_ensemble_eval.py
│   ├── dataset/          # Dataset management
│   │   ├── download_dataset.py
│   │   └── split_dataset.py
│   └── phase2_full_pipeline.py
├── notebooks/            # NEW: Kaggle/Jupyter notebooks
├── webapp/               # Web interface
├── frontend/             # Frontend (if separate)
├── data/                 # Data directory
├── results/              # Training results
├── docs/                 # Documentation
├── tests/                # Tests
├── Minor_Report/         # Project reports
├── models/               # NEW: Model checkpoints (gitignored)
│   └── checkpoints/
│       ├── efficientnet/
│       ├── vit/
│       └── resnet/
├── requirements.txt
├── config.yaml
├── README.md
└── LICENSE
```

## Cleanup Steps (Safe Order)

### Phase 1: Create New Structure
1. Create `scripts/training/` directory
2. Create `notebooks/` directory  
3. Ensure `models/checkpoints/` exists in .gitignore

### Phase 2: Move Training Scripts
4. Move `run_training_*.py` → `scripts/training/`
5. Move `run_ensemble_eval.py` → `scripts/training/`
6. Update imports in moved files (relative paths)

### Phase 3: Consolidate Demo Apps
7. Keep `demo_app.py` in root (main entry point)
8. Consider moving `demo_app_hybrid.py` to `webapp/` or archive

### Phase 4: Clean Root
9. Move `app.py` → `webapp/` or consolidate with demo_app
10. Move `CSE_64(Minor_Project_Report).pdf` → `Minor_Report/`

### Phase 5: Git Housekeeping
11. Stage untracked files
12. Commit cleanup with clear message
13. Update README with new structure

## Import Fix Strategy

When moving `scripts/training/run_*.py`, add to top of each file:
```python
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

## Risks
- **Import breakage**: Fixed by path manipulation above
- **Kaggle paths**: Use relative paths from kaggle working dir
- **CI/CD**: Update any automated scripts

## Deferred (Low Priority)
- Merge `frontend/` and `webapp/` (needs analysis)
- Archive old training scripts
- Reorganize `docs/` subdirectories
