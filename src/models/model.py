"""
Model Definitions
Defines ResNet-50 baseline and other architectures
Week 5 - Model Training (Krishan Yadav - Lead)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B0_Weights,
)
from typing import Optional


class DeepSceneLocResNet50(nn.Module):
    """
    ResNet-50 baseline model for location classification
    Pre-trained on ImageNet, fine-tuned for 5 location categories
    """
    
    def __init__(self, num_classes=5, pretrained=True, freeze_layers=0):
        """
        Args:
            num_classes: Number of output classes (default: 5)
            pretrained: Whether to use ImageNet pre-trained weights
            freeze_layers: Number of initial layers to freeze (0 = fine-tune all)
        """
        super(DeepSceneLocResNet50, self).__init__()
        
        # Load pretrained ResNet-50
        _weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=_weights)
        
        # Freeze layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features  # 2048 for ResNet-50
        
        # Replace the final fully connected layer
        # Architecture: 2048 -> 512 -> 5 (as per project specs)
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
    
    def _freeze_layers(self, num_layers):
        """Freeze the first num_layers of the backbone"""
        layers = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        ]
        
        for i, layer in enumerate(layers[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


class DeepSceneLocEfficientNet(nn.Module):
    """
    EfficientNet-B0 model for location classification
    (Semester 2 - Advanced models)
    """
    
    def __init__(self, num_classes=5, pretrained=True):
        super(DeepSceneLocEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet-B0
        _weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=_weights)
        
        # Get number of features
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)


class DeepSceneLocViT(nn.Module):
    """
    Vision Transformer (ViT) model for location classification
    (Semester 2 - Advanced models)
    """
    
    def __init__(self, num_classes=5, pretrained=True):
        super(DeepSceneLocViT, self).__init__()
        
        # Load pretrained ViT (requires timm library)
        try:
            import timm
            self.backbone = timm.create_model(
                'vit_base_patch16_224',
                pretrained=pretrained,
                num_classes=num_classes
            )
        except ImportError:
            print("Warning: timm not installed. ViT model requires: pip install timm")
            # Fallback to simple linear classifier
            self.backbone = nn.Linear(224*224*3, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)


def create_model(
    model_name: str = 'resnet50',
    num_classes: int = 5,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_name: Name of model ('resnet50', 'efficientnet_b0', 'vit')
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights
        **kwargs: Additional model-specific arguments
        
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'resnet50':
        model = DeepSceneLocResNet50(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_name == 'efficientnet_b0':
        model = DeepSceneLocEfficientNet(
            num_classes=num_classes,
            pretrained=pretrained
        )
    elif model_name == 'vit':
        model = DeepSceneLocViT(
            num_classes=num_classes,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def test_models():
    """Test model creation"""
    print("="*60)
    print("Testing Model Architectures")
    print("="*60)
    
    # Test ResNet-50
    print("\n1. Testing ResNet-50...")
    model = create_model('resnet50', num_classes=5, pretrained=False)
    print(f"   Total parameters: {model.get_total_params():,}")
    print(f"   Trainable parameters: {model.get_trainable_params():,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"    ResNet-50 working correctly")
    
    # Test EfficientNet
    print("\n2. Testing EfficientNet-B0...")
    try:
        model = create_model('efficientnet_b0', num_classes=5, pretrained=False)
        out = model(x)
        print(f"   Output shape: {out.shape}")
        print(f"    EfficientNet-B0 working correctly")
    except Exception as e:
        print(f"   Note: {e}")
    
    print("\n Model definitions ready!")


if __name__ == "__main__":
    test_models()
