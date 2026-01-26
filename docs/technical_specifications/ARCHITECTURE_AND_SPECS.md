# Technical Specifications and Architecture

## System Architecture Overview

### Data Flow Pipeline
```
Raw Images (jpg/png)
         |
         v
    Image Resizing (224x224)
         |
         v
    Normalization
         |
         v
    Data Augmentation (train only)
         |
         v
    Model Input
         |
    +----+----+----+
    |    |    |    |
    v    v    v    v
   ResNet EfficientNet ViT
   50    B0         Base
    |    |    |    |
    +----+----+----+
         |
    Feature Embedding
         |
    Classification Head
         |
    Softmax Output
         |
    Probability Scores
         |
    Location Prediction
```

---

## Model Architectures

### 1. ResNet-50 (Baseline Model)

#### Architecture Details
- Total Layers: 50
- Total Parameters: 25.5 million
- Input Size: 224x224x3
- Output: 5 classes
- Backbone: 4 residual blocks
- Skip Connections: Present in all blocks

#### Layer Breakdown
```
Layer 1 (64 channels):   7x7 conv, stride=2
Layer 2 (256 channels):  3 residual blocks
Layer 3 (512 channels):  4 residual blocks
Layer 4 (1024 channels): 6 residual blocks
Layer 5 (2048 channels): 3 residual blocks
Global Average Pool:     7x7
Classification Head:     2048 -> 5
```

#### Residual Block Structure
```
Input
  |
  +-->Conv(1x1, in_channels -> out_channels/4)
  |   |
  |   v
  |   BatchNorm
  |   |
  |   v
  |   ReLU
  |   |
  +-->Conv(3x3, out_channels/4 -> out_channels/4)
  |   |
  |   v
  |   BatchNorm
  |   |
  |   v
  |   ReLU
  |   |
  +-->Conv(1x1, out_channels/4 -> out_channels)
      |
      v
      BatchNorm
      |
      v
    + (Residual Connection)
      |
      v
      ReLU
```

#### Configuration for Fine-tuning
- Freeze: Layers 1, 2, 3 (first 30 layers approximately)
- Fine-tune: Layer 4, 5, Classification head
- Dropout: 0.5 in classification head
- Batch Normalization: Keep frozen from pretrained

#### Key Advantages
- Well-studied architecture
- Excellent ImageNet transfer learning
- Good gradient flow due to skip connections
- Efficient computation

---

### 2. EfficientNet-B0 (Advanced Model)

#### Architecture Details
- Total Parameters: 5.3 million
- Input Size: 224x224x3
- Output: 5 classes
- Scaling: Compound scaling (width, depth, resolution)
- Efficiency Coefficient: Baseline (B0)

#### Layer Breakdown
```
Stem (3x32):             3x3 conv, stride=2
Mobile Inverted Block 1: 3x16 (1 layer, expansion=1)
Mobile Inverted Block 2: 3x24 (2 layers, expansion=6)
Mobile Inverted Block 3: 5x40 (2 layers, expansion=6)
Mobile Inverted Block 4: 3x80 (3 layers, expansion=6)
Mobile Inverted Block 5: 5x112 (3 layers, expansion=6)
Mobile Inverted Block 6: 5x192 (4 layers, expansion=6)
Mobile Inverted Block 7: 3x320 (1 layer, expansion=6)
Head (1280):             1x1 conv
Classification Head:     1280 -> 5
```

#### Mobile Inverted Bottleneck Block
```
Input (in_channels)
  |
  v
DepthwiseConv(kernel, expansion*in_channels)
  |
  v
BatchNorm + ReLU6
  |
  v
PointwiseConv(expansion*in_channels -> out_channels)
  |
  v
BatchNorm (no activation)
  |
  +-->Residual Connection (if in_channels == out_channels)
  |
  v
Output
```

#### Configuration for Fine-tuning
- Freeze: MobileInverted blocks 1-4 (first 14 layers)
- Fine-tune: Blocks 5-7, head layers
- Dropout: 0.2 in classification head (smaller model)
- Batch Normalization: Keep frozen

#### Key Advantages
- 5x fewer parameters than ResNet-50
- Better accuracy-efficiency trade-off
- Faster inference (ideal for deployment)
- Compound scaling principle
- Lower memory footprint

---

### 3. Vision Transformer (ViT-Base)

#### Architecture Details
- Total Parameters: 86 million (base model)
- Input Size: 224x224x3
- Patch Size: 16x16
- Number of Patches: 196 (14x14 grid)
- Embedding Dimension: 768
- Number of Attention Heads: 12
- Number of Transformer Blocks: 12
- Hidden Dimension (MLP): 3072

#### Patch Embedding Process
```
Input Image (224x224x3)
         |
         v
    Divide into Patches (16x16)
    = 196 patches
         |
         v
    Flatten each patch
    (16*16*3 = 768 dimensions)
         |
         v
    Linear Projection
    768 -> 768 (embedding)
         |
         v
    Add positional encoding (197 positions)
    (1 class token + 196 patch tokens)
         |
         v
    Transformer Input
```

#### Transformer Block Structure
```
Input (197x768)
  |
  v
LayerNorm
  |
  v
MultiHeadAttention(768 dim, 12 heads)
  |
  +-->Residual Connection
  |
  v
LayerNorm
  |
  v
MLP (768 -> 3072 -> 768)
  |
  +-->Residual Connection
  |
  v
Output (197x768)
```

#### Classification Head
```
Output of Transformer (197x768)
         |
    Extract class token
    [CLS] token (1x768)
         |
         v
    LayerNorm
         |
         v
    Linear (768 -> 5)
         |
         v
    Classification Output
```

#### Configuration for Fine-tuning
- Freeze: Patch embedding, transformer blocks 1-10
- Fine-tune: Blocks 11-12, classification head
- Dropout: 0.1 (transformers need regularization)
- Stochastic depth: 0.1 (for robustness)
- Learning rate: Lower than CNN (0.00005)

#### Attention Mechanism
- Scaled dot-product attention
- Query (Q), Key (K), Value (V) projections
- Multiple heads for different feature spaces
- Self-attention: Each patch attends to all patches

#### Key Advantages
- Captures long-range dependencies
- Global receptive field from start
- Excellent transfer learning on large datasets
- Interpretable attention weights
- Modern state-of-the-art approach

---

## Custom Classification Head Design

### Head Architecture (All Models)
```
Features from backbone (2048/1280/768)
         |
         v
    BatchNorm1d
         |
         v
    Dropout(p=0.3 for ResNet, 0.2 for EfficientNet, 0.1 for ViT)
         |
         v
    Linear(in_features -> 512)
         |
         v
    ReLU
         |
         v
    BatchNorm1d
         |
         v
    Dropout(p=0.3)
         |
         v
    Linear(512 -> 256)
         |
         v
    ReLU
         |
         v
    BatchNorm1d
         |
         v
    Dropout(p=0.3)
         |
         v
    Linear(256 -> 5)
         |
         v
    Softmax (for prediction)
```

### Hyperparameters for Head
- Hidden dimensions: 512 -> 256
- Activation: ReLU
- Regularization: Dropout + BatchNorm
- Output activation: Softmax (for probabilities)

---

## Training Specifications

### Loss Function
- CrossEntropyLoss (combines LogSoftmax + NLLLoss)
- Formula: L = -log(exp(score_true) / sum(exp(scores)))
- Handles class imbalance with weight parameter if needed

### Optimizer Configuration

#### ResNet-50 and EfficientNet
- Optimizer: Adam
- Initial Learning Rate: 0.001
- Beta1: 0.9, Beta2: 0.999
- Epsilon: 1e-8
- Weight Decay: 0.0001

#### Vision Transformer
- Optimizer: AdamW
- Initial Learning Rate: 0.00005 (smaller than CNN)
- Beta1: 0.9, Beta2: 0.999
- Epsilon: 1e-8
- Weight Decay: 0.05 (higher for transformers)

### Learning Rate Scheduling
```
Warmup Phase (5 epochs):
    LR increases linearly from 0 to initial_lr

Main Training Phase (35 epochs):
    LR = initial_lr * 0.5 * (1 + cos(pi * epoch / epochs))
    (Cosine annealing)
```

### Batch Configuration
- Training Batch Size: 32
- Validation Batch Size: 64 (no gradient computation)
- Gradient Accumulation: Steps=1 (no accumulation)
- Number of Workers: 4 (for data loading)

### Training Stopping Criteria
- Maximum Epochs: 40
- Early Stopping: Yes
  - Metric: Validation accuracy
  - Patience: 5 epochs
  - Min delta: 0.001

---

## Data Preprocessing Pipeline

### Image Resizing
- Target Size: 224x224
- Aspect Ratio Preservation: Yes
- Method: Resize then pad with center crop
- Interpolation: Bilinear

### Normalization
```
pixel_normalized = (pixel - mean) / std

ImageNet Normalization:
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

### Data Augmentation (Training Only)

#### Applied Transformations
1. Random Horizontal Flip: 50% probability
2. Random Rotation: -15 to +15 degrees
3. ColorJitter:
   - Brightness: 0.2
   - Contrast: 0.2
   - Saturation: 0.2
   - Hue: 0.1
4. Random Perspective: 0.3 probability
5. Random Erasing: 0.2 probability (0.5 area)

#### Validation/Test Pipeline
- No augmentation
- Only resize and normalize
- Deterministic transformations

---

## Model Output and Inference

### Output Format
```
Input Image -> Model -> Output Logits
                       [logit_1, logit_2, logit_3, logit_4, logit_5]
                       |
                       v
                    Softmax
                       |
    [prob_urban, prob_rural, prob_coastal, prob_mountain, prob_forest]
                       |
                       v
                   Top-K Selection
                       |
    Top-3: [(class_1, score_1), (class_2, score_2), (class_3, score_3)]
```

### Inference Process
1. Load image and preprocess
2. Forward through model (no gradient computation)
3. Get output logits
4. Apply softmax for probabilities
5. Get top-K predictions
6. Return predictions with confidence scores

### Inference Speed Targets
- ResNet-50: 30-40ms per image
- EfficientNet-B0: 15-25ms per image
- Vision Transformer: 50-70ms per image
- Batch inference: 10-50ms per image (larger batches)

---

## Evaluation Metrics

### Classification Metrics
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision_i = TP_i / (TP_i + FP_i)

Recall_i = TP_i / (TP_i + FN_i)

F1_i = 2 * (Precision_i * Recall_i) / (Precision_i + Recall_i)

Weighted F1 = sum(F1_i * support_i) / total_samples

Macro F1 = mean(F1_i) for all classes
```

### Confusion Matrix
- 5x5 matrix: True classes vs Predicted classes
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications
- Normalized version: Percentages by true class

### Top-K Accuracy
```
Top-1 Accuracy = samples where true label in top-1 / total

Top-3 Accuracy = samples where true label in top-3 / total

Top-5 Accuracy = samples where true label in top-5 / total
```

### Confidence Metrics
```
Average Confidence = mean(max_probability) for all samples

Confidence for Correct = mean(probability) when prediction correct

Confidence for Incorrect = mean(probability) when prediction incorrect

Expected Calibration Error = mean(|accuracy - confidence|) per bin
```

---

## Repository Structure

```
DeepSceneLoc/
├── README.md
├── LICENSE
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── semester_1_planning/
│   │   └── SEMESTER_1_PLAN.md
│   ├── semester_2_planning/
│   │   └── SEMESTER_2_PLAN.md
│   ├── literature_review/
│   │   └── RESEARCH_PAPERS_AND_REFERENCES.md
│   ├── technical_specifications/
│   │   └── ARCHITECTURE_AND_SPECS.md
│   └── results_tracking/
│       ├── RESULTS_LOG.md
│       ├── METRICS_FINAL.md
│       └── visualizations/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet50.py
│   │   ├── efficientnet.py
│   │   └── vit.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── loader.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── transforms.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── notebooks/
│   ├── training_example.ipynb
│   ├── inference_example.ipynb
│   └── visualization_example.ipynb
├── models/
│   ├── resnet50_final.pth
│   ├── efficientnet_final.pth
│   └── vit_final.pth
├── data/
│   ├── places365_subset/
│   ├── train_split.txt
│   ├── val_split.txt
│   └── test_split.txt
└── results/
    ├── training_logs/
    ├── metrics/
    ├── visualizations/
    └── confusion_matrices/
```

---

## Performance Specifications

### Computational Requirements
- Minimum GPU: 4GB (for training)
- Recommended GPU: 8GB+ (T4, V100)
- CPU RAM: 8GB minimum
- Storage: 50GB (for dataset + models)
- Network: 10GB+ for dataset download

### Training Time Estimates
- ResNet-50: 2-3 hours (40 epochs)
- EfficientNet-B0: 2-3 hours (40 epochs)
- Vision Transformer: 3-4 hours (40 epochs)
- All three models: 8-10 hours total

### Inference Performance
- Batch Size 1: 30-70ms per image
- Batch Size 32: 5-15ms per image (10-30 images/sec)
- CPU inference: 200-500ms per image (not recommended for production)

