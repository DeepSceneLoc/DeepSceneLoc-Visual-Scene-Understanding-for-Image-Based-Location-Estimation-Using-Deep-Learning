# DFD Level 0 - Context Diagram
## DeepSceneLoc System Overview

```mermaid
graph TB
    User[👤 User/Developer]
    System[ DeepSceneLoc System<br/>Image-Based Location Estimation]
    Dataset[(Data Places365 Dataset<br/>50GB)]
    GeminiAPI[Cloud Google Gemini API<br/>Semester 2]
    GitHub[📁 GitHub Repository<br/>Version Control]
    
    User -->|Upload Image| System
    System -->|Scene Category + Location| User
    
    Dataset -->|Training Images| System
    System -->|Query: Image + Scene Context| GeminiAPI
    GeminiAPI -->|Location Details: Landmark, GPS| System
    
    System -->|Code, Models, Docs| GitHub
    GitHub -->|Pull Latest Version| System
    
    style System fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style GeminiAPI fill:#FFA500,stroke:#CC8400,stroke-width:2px,stroke-dasharray: 5 5
    style Dataset fill:#90EE90,stroke:#228B22,stroke-width:2px
    style GitHub fill:#333,stroke:#000,stroke-width:2px,color:#fff
    style User fill:#FFE4B5,stroke:#8B4513,stroke-width:2px
```

## System Inputs
- **User Images:** JPG, PNG, JPEG formats
- **Training Dataset:** Places365 outdoor scenes (5 categories)
- **Pretrained Weights:** ImageNet ResNet-50
- **Configuration:** YAML files, hyperparameters

## System Outputs
- **Stage 1 (Semester 1):** Scene category (Coastal/Forest/Mountain/Rural/Urban) + confidence scores
- **Stage 2 (Semester 2):** Exact location (landmark name, city, country, GPS coordinates)
- **Development Outputs:** Trained models, evaluation metrics, checkpoints

## External Entities
1. **User/Developer:** Provides images, receives predictions, monitors training
2. **Places365 Dataset:** Source of training/validation/test images
3. **Google Gemini API:** AI service for exact location detection (Semester 2)
4. **GitHub Repository:** Code versioning and collaboration

## Data Flows
- **Bidirectional:** User ↔ System (image upload, results display)
- **Unidirectional:** Dataset → System (training data)
- **Unidirectional:** System → Gemini API → System (location queries)
- **Bidirectional:** System ↔ GitHub (version control)
