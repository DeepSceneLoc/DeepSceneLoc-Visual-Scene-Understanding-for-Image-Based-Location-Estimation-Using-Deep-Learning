# DeepSceneLoc
## Visual Scene Understanding for Image-Based Location Estimation Using Deep Learning

<img width="1732" height="1127" alt="Dark" src="https://github.com/user-attachments/assets/091191af-ec5a-494f-b310-7d52fa4073be" />


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

DeepSceneLoc is a two-stage hybrid system for location estimation from images:

**Stage 1 (Semester 1):** Scene classification into 5 broad categories (Urban, Rural, Coastal, Mountain, Forest) using deep learning models trained on visual scene understanding.

**Stage 2 (Semester 2):** Exact location detection using AI integration to identify the specific place, landmark, city, country, and GPS coordinates from the image.

**Purpose:** Get the exact place on Earth from any image, combining custom-trained scene classifiers with AI-powered location recognition, without relying on GPS or EXIF metadata.

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM4.5 7.5a.5.5 0 0 0 0 1h5.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3a.5.5 0 0 0 0-.708l-3-3a.5.5 0 1 0-.708.708L10.293 7.5H4.5z"/></svg> Project Overview

**Primary Objective:** Identify the exact place on Earth from any image using a hybrid AI approach.

**Semester 1 Contribution:** Scene classification system
- Transfer learning from ImageNet pretrained models
- Fine-tuned on Places365 outdoor subset mapped to 5 categories
- Provides scene context (Urban, Rural, Coastal, Mountain, Forest)
- Model: ResNet-50 baseline

**Semester 2 Contribution:** Exact location detection
- Advanced models: EfficientNet-B0, Vision Transformer (ViT)
- AI integration: Gemini API for landmark and location identification
- Hybrid system: Combines scene classification + exact place detection
- Output: Specific landmark, city, country, GPS coordinates

**Scene Categories (Stage 1):**
1. **Urban** - Cities, streets, buildings, urban infrastructure
2. **Rural** - Farmland, countryside, villages, agricultural areas
3. **Coastal** - Beaches, harbors, seaside, coastal landscapes
4. **Mountain** - Peaks, highlands, mountain terrain
5. **Forest** - Wooded areas, dense vegetation, jungle

**Location Detection (Stage 2):**
- Exact landmarks (e.g., "Eiffel Tower, Paris")
- City and country identification
- GPS coordinates (latitude/longitude)
- Regional context for generic scenes

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM5.5 5.5a.5.5 0 0 0-1 0v5.793L2.854 9.646a.5.5 0 1 0-.708.708l3 3a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L6.5 11.293V5.5zm3 .5a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 0 1h-4a.5.5 0 0 1-.5-.5z"/></svg> Key Features

- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> **Exact Location Detection** - Identifies specific places, landmarks, cities, countries
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> **Hybrid Two-Stage System** - Scene classification + AI-powered location recognition
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> **No GPS Required** - Pure visual understanding without metadata
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> **Privacy-Preserving** - No EXIF data extraction needed
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> **Transfer Learning** - Leverages ImageNet pretrained weights
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> **Multiple Architectures** - ResNet-50, EfficientNet, ViT
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> **Free Resources** - Runs on Google Colab free GPU + Gemini API free tier
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> **Comprehensive Evaluation** - Accuracy, precision, recall, F1, confusion analysis
- <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> **Rich Visualizations** - Confusion matrices, performance charts, training curves

---

## 📁 Project Structure

```
DeepSceneLoc/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # Source code
│   ├── data/                   # Dataset preparation
│   │   └── prepare_dataset.py # Places365 mapping & splitting
│   ├── preprocessing/          # Data preprocessing
│   │   └── pipeline.py         # Transformations & dataloaders
│   ├── models/                 # Model architectures
│   │   ├── model.py            # ResNet-50, EfficientNet, ViT
│   │   └── train.py            # Training pipeline
│   ├── evaluation/             # Evaluation scripts
│   │   └── evaluate.py         # Metrics calculation
│   └── utils/                  # Utilities
│       └── visualizations.py   # Plotting functions
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training_demo.ipynb
│   ├── 03_evaluation_and_visualization.ipynb
│   └── 04_interactive_demo.ipynb
│
├── docs/                       # Documentation
│   ├── PROJECT_OVERVIEW.md
│   ├── TEAM_ALLOCATION.md
│   ├── WEEKLY_PROGRESS_TRACKING.md
│   ├── semester_1_planning/
│   ├── semester_2_planning/
│   └── literature_review/
│
├── data/                       # Data directory (not in git)
│   ├── raw/                    # Raw downloaded data
│   └── processed/              # Processed train/val/test splits
│
├── models/                     # Saved models (not in git)
│   ├── checkpoints/            # Training checkpoints
│   └── best_model.pth          # Best model
│
├── results/                    # Results and outputs
│   ├── visualizations/         # Generated plots
│   └── metrics/                # Evaluation metrics
│
└── logs/                       # Training logs
    └── training_history.json
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 10GB+ disk space for dataset

### Setup

```bash
# Clone repository
git clone https://github.com/DeepSceneLoc/DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning.git
cd DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.5-13v1.5a.5.5 0 0 1-1 0V3h-3a.5.5 0 0 1 0-1h7a.5.5 0 0 1 0 1h-3z"/><path d="M8 4.5a.5.5 0 0 1 .5.5v3.5H11a.5.5 0 0 1 0 1H8a.5.5 0 0 1-.5-.5V5a.5.5 0 0 1 .5-.5z"/></svg> Dataset

**Source:** Places365 (outdoor subset)
- **Original Size:** 1.8M images, 365 categories
- **Mapped to:** 5 location categories
- **Split:** 70% train, 15% validation, 15% test

### Dataset Preparation

```python
from src.data.prepare_dataset import Places365Mapper, DatasetSplitter

# Map Places365 categories to 5 locations
mapper = Places365Mapper()
mapper.save_mapping("data/category_mapping.json")

# Split dataset
splitter = DatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
splitter.split_dataset(
    data_dir="data/raw",
    output_dir="data/processed"
)
```

---

## 🏋️ Training

### Quick Start

```python
from src.models.train import setup_training

# Configuration
config = {
    'data_dir': 'data/processed',
    'model_name': 'resnet50',
    'num_classes': 5,
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'pretrained': True
}

# Setup and train
trainer = setup_training(config)
trainer.train(num_epochs=20)
```

### Training on Google Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository and install
!git clone https://github.com/DeepSceneLoc/DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning.git
%cd DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning
!pip install -r requirements.txt

# Run training (automatically uses GPU if available)
# Use notebooks/02_model_training_demo.ipynb
```

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M0 0h1v15h15v1H0V0Zm10 3.5a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-1 0V4.9l-3.613 4.417a.5.5 0 0 1-.74.037L7.06 6.767l-3.656 5.027a.5.5 0 0 1-.808-.588l4-5.5a.5.5 0 0 1 .758-.06l2.609 2.61L13.445 4H10.5a.5.5 0 0 1-.5-.5z"/></svg> Evaluation

### Evaluate Trained Model

```python
from src.evaluation.evaluate import load_model_and_evaluate
from src.preprocessing.pipeline import create_dataloaders

# Load test data
_, _, test_loader = create_dataloaders(
    data_dir='data/processed',
    batch_size=64
)

# Evaluate
metrics = load_model_and_evaluate(
    model_path='models/checkpoints/best_model.pth',
    model_name='resnet50',
    test_loader=test_loader,
    class_names=['Coastal', 'Forest', 'Mountain', 'Rural', 'Urban']
)
```

### Generate Visualizations

```python
from src.utils.visualizations import create_all_visualizations

create_all_visualizations(
    metrics=metrics,
    history_path='logs/training_history.json',
    output_dir='results/visualizations'
)
```

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.5-13v1.5a.5.5 0 0 1-1 0V3h-3a.5.5 0 0 1 0-1h7a.5.5 0 0 1 0 1h-3z"/><path d="M8 4.5a.5.5 0 0 1 .5.5v3.5H11a.5.5 0 0 1 0 1H8a.5.5 0 0 1-.5-.5V5a.5.5 0 0 1 .5-.5z"/></svg> Results (Semester 1 Baseline)

### ResNet-50 Performance

| Metric | Score |
|--------|-------|
| Overall Accuracy | TBD |
| Macro Precision | TBD |
| Macro Recall | TBD |
| Macro F1-Score | TBD |

**Per-Class Results:**

| Category | Accuracy | Precision | Recall | F1-Score |
|----------|----------|-----------|--------|----------|
| Urban | TBD | TBD | TBD | TBD |
| Rural | TBD | TBD | TBD | TBD |
| Coastal | TBD | TBD | TBD | TBD |
| Mountain | TBD | TBD | TBD | TBD |
| Forest | TBD | TBD | TBD | TBD |

*Results will be updated after Week 6 evaluation.*

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M1 2.5A1.5 1.5 0 0 1 2.5 1h3A1.5 1.5 0 0 1 7 2.5v3A1.5 1.5 0 0 1 5.5 7h-3A1.5 1.5 0 0 1 1 5.5v-3z"/></svg> Usage Examples

### Predict Single Image

```python
import torch
from PIL import Image
from torchvision import transforms
from src.models.model import create_model

# Load model
model = create_model('resnet50', num_classes=5, pretrained=False)
checkpoint = torch.load('models/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('path/to/image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

categories = ['Coastal', 'Forest', 'Mountain', 'Rural', 'Urban']
print(f"Predicted: {categories[predicted_class]}")
print(f"Confidence: {probabilities[0][predicted_class].item():.2%}")
```

---

## 📚 Documentation

Detailed documentation available in `/docs`:
- [Project Overview](docs/PROJECT_OVERVIEW.md)
- [Semester 1 Plan](docs/semester_1_planning/SEMESTER_1_PLAN.md)
- [Semester 2 Plan](docs/semester_2_planning/SEMESTER_2_PLAN.md)
- [Literature Review](docs/literature_review/RESEARCH_PAPERS_AND_REFERENCES.md)
- [Technical Specifications](docs/technical_specifications/ARCHITECTURE_AND_SPECS.md)
- [Team Allocation](docs/TEAM_ALLOCATION.md)
- [Progress Tracking](docs/WEEKLY_PROGRESS_TRACKING.md)

---

## 👥 Team

**DeepSceneLoc Development Team**
- Krishan Yadav - Technical Lead, Model Training & Evaluation
- Aditi Sah - Data Lead, Dataset Preparation & Analysis
- Anuj Kondawar - Preprocessing Lead, Pipeline Development
- Jensi Paneliya - Documentation Lead, Project Documentation

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M1 2.5A1.5 1.5 0 0 1 2.5 1h3A1.5 1.5 0 0 1 7 2.5v3A1.5 1.5 0 0 1 5.5 7h-3A1.5 1.5 0 0 1 1 5.5v-3zM2.5 2a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3zm6.5.5A1.5 1.5 0 0 1 10.5 1h3A1.5 1.5 0 0 1 15 2.5v3A1.5 1.5 0 0 1 13.5 7h-3A1.5 1.5 0 0 1 9 5.5v-3zm1.5-.5a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3zM1 10.5A1.5 1.5 0 0 1 2.5 9h3A1.5 1.5 0 0 1 7 10.5v3A1.5 1.5 0 0 1 5.5 15h-3A1.5 1.5 0 0 1 1 13.5v-3zm1.5-.5a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3zm6.5.5A1.5 1.5 0 0 1 10.5 9h3a1.5 1.5 0 0 1 1.5 1.5v3a1.5 1.5 0 0 1-1.5 1.5h-3A1.5 1.5 0 0 1 9 13.5v-3zm1.5-.5a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3z"/></svg> Roadmap

### <svg width="12" height="12" fill="green"><path d="M12 0L4 8L0 4L1.5 2.5L4 5L10.5 -1.5L12 0Z"/></svg> Semester 1 (30% Completion)
- [x] Week 1: Project Setup & Scope Freeze
- [x] Week 2: Literature Review
- [x] Week 3: Dataset Preparation
- [x] Week 4: Preprocessing Pipeline
- [x] Week 5: Baseline Model Training (ResNet-50)
- [x] Week 6: Model Evaluation & Analysis
- [x] Week 7: Documentation & Review Preparation

### <svg width="12" height="12" fill="orange"><circle cx="6" cy="6" r="5"/></svg> Semester 2 (70% Completion)
- [ ] Advanced model implementation (EfficientNet, ViT)
- [ ] Comparative study of architectures
- [ ] Embeddings visualization (t-SNE, PCA)
- [ ] Top-K predictions analysis
- [ ] Comprehensive error analysis
- [ ] Model optimization & compression
- [ ] Final report & presentation
- [ ] Viva preparation

---

## 🔬 Research Foundation

This project builds upon:
- **Places365** (Zhou et al.) - Scene recognition dataset
- **PlaNet** (Weyand et al.) - CNN-based geolocation
- **Vision Transformer** (Dosovitskiy et al.) - Attention mechanisms
- **Transfer Learning** best practices from ImageNet

See [Literature Review](docs/literature_review/RESEARCH_PAPERS_AND_REFERENCES.md) for complete references.

---

## 📜 License

Proprietary License - Academic Project

This project is developed for academic purposes. All rights reserved.

---

## 🤝 Contributing

This is an academic project developed by a dedicated team. For questions or collaboration inquiries, please contact the team.

---

## 📧 Contact

- **Repository:** https://github.com/DeepSceneLoc/DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning
- **Issues:** Use GitHub Issues for bug reports or feature requests

---

## <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/></svg> Acknowledgments

- **Places365 Team** for the comprehensive scene dataset
- **PyTorch Community** for excellent documentation and tutorials
- **Google Colab** for free GPU resources
- **Research Community** for foundational work in scene recognition

---

**Last Updated:** February 26, 2026 | **Version:** 1.0.0 (Semester 1 Complete)
