"""
Advanced Model Definitions -- EfficientNet-B0 & Vision Transformer (ViT-B/16)
DeepSceneLoc -- Semester 2, Weeks 8-9

Authors:
    Krishan Yadav  (Model Architecture Lead)
    Anuj Kondawar  (Preprocessing & Pipeline Lead -- architecture integration)

This module provides:
  - DeepSceneLocEfficientNetAdvanced  -- EfficientNet-B0 with configurable freeze blocks,
                                        proper 5-class head (1280 -> 512 -> 5), BN + Dropout
  - DeepSceneLocViTAdvanced           -- ViT-B/16 (timm) with freezeable transformer blocks
                                        and a dedicated classification head
  - create_advanced_model()           -- factory function (mirrors create_model() API)
  - model_summary()                   -- prints parameter counts + layer sizes

Design notes:
  - EfficientNet freezes the first `freeze_blocks` MBConv blocks (of 9); default = 7.
  - ViT freezes the patch embedding + first `freeze_encoder_blocks` transformer blocks; default = 10.
  - Both models expose get_trainable_params() / get_total_params() helpers.
  - All models return raw logits (no softmax) -- consistent with CrossEntropyLoss.
  - Mixed-precision (torch.cuda.amp) compatible -- no explicit fp32 casts needed in forward().
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)
from typing import Optional


# -------------------------------------------------------------
# EfficientNet-B0
# -------------------------------------------------------------

class DeepSceneLocEfficientNetAdvanced(nn.Module):
    """
    EfficientNet-B0 fine-tuned for 5-class scene/location classification.

    Architecture:
        EfficientNet-B0 backbone (ImageNet pretrained)
            +- Feature extractor: 9 MBConv blocks -> Global avg-pool -> 1280-d vector
            +- Custom head:  1280 -> BN -> Dropout(0.3) -> 512 -> ReLU -> Dropout(0.2) -> 5

    Freezing strategy (default: freeze_blocks=7):
        blocks 0-6 frozen  (low-level + mid-level features)
        blocks 7-8 trainable (high-level scene features)
        head       trainable

    Success target (Semester 2 plan): val_acc >= 75 %
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_blocks: int = 7,
        dropout_head: float = 0.3,
        dropout_head2: float = 0.2,
    ):
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # EfficientNet-B0 has 9 MBConv blocks in backbone.features[1:-1]
        # backbone.features index map:
        #   [0] = stem conv
        #   [1..8] = MBConv blocks 0..7
        #   [9] = top conv (1x1 expanding to 1280)
        #   backbone.avgpool -> backbone.classifier -> (not used)

        self.features = backbone.features  # Sequential of Conv + MBConvs + TopConv
        self.avgpool  = backbone.avgpool   # AdaptiveAvgPool2d -> (B, 1280, 1, 1)

        # Freeze the first `freeze_blocks` MBConv stages
        # features[0] = stem, features[1..8] = MBConvs, features[9] = top-conv
        self._freeze_efficientnet(freeze_blocks)

        # Custom classification head
        in_features = 1280  # EfficientNet-B0 top conv output channels
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_head),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_head2),
            nn.Linear(512, num_classes),
        )

        self.num_classes = num_classes

    def _freeze_efficientnet(self, freeze_blocks: int) -> None:
        """
        Freeze stem + first `freeze_blocks` MBConv stages.
        features[0]     = stem conv  -> always frozen when freeze_blocks > 0
        features[1..8]  = MBConv blocks 0..7
        features[9]     = top-conv  -> always trainable
        """
        if freeze_blocks <= 0:
            return
        # Freeze stem
        for param in self.features[0].parameters():
            param.requires_grad = False
        # Freeze MBConv blocks up to freeze_blocks
        for i in range(1, min(freeze_blocks + 1, 9)):  # features[1..8] = blocks 0..7
            for param in self.features[i].parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, 1280, 7, 7)
        x = self.avgpool(x)           # (B, 1280, 1, 1)
        x = x.flatten(1)              # (B, 1280)
        x = self.head(x)              # (B, num_classes)
        return x

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def unfreeze_all(self) -> None:
        """Unfreeze entire network (for progressive training)."""
        for param in self.parameters():
            param.requires_grad = True

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}"


# -------------------------------------------------------------
# Vision Transformer (ViT-B/16)
# -------------------------------------------------------------

class DeepSceneLocViTAdvanced(nn.Module):
    """
    Vision Transformer ViT-B/16 fine-tuned for 5-class scene classification.

    Uses `timm` library -- must be installed: ``pip install timm``.

    Architecture:
        ViT-B/16 (12-layer transformer, 768-d hidden, 12 heads, 16x16 patches)
            Feature dim: 768-d CLS token
        Custom head: 768 -> LayerNorm -> Dropout(0.2) -> 5

    Freezing strategy (default: freeze_encoder_blocks=10):
        patch embedding frozen
        blocks[0..9] frozen  (first 10 of 12 transformer blocks)
        blocks[10..11] trainable
        classification head trainable

    Success target (Semester 2 plan): val_acc >= 78 %
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_encoder_blocks: int = 10,
        dropout_head: float = 0.2,
    ):
        super().__init__()

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for ViT. Install with: pip install timm"
            ) from exc

        # Load ViT-B/16 WITHOUT timm's built-in head so we can add our own
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,          # Remove the ImageNet head -> output is (B, 768)
        )

        # Freeze patch embedding + positional embedding
        self._freeze_embedding()

        # Freeze the first N transformer blocks
        self._freeze_blocks(freeze_encoder_blocks)

        # Custom head (768 -> 5)
        hidden_dim = self.vit.num_features  # 768 for ViT-B/16
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_dim, num_classes),
        )

        self.num_classes = num_classes

    def _freeze_embedding(self) -> None:
        """Freeze patch embed and positional embedding."""
        for param in self.vit.patch_embed.parameters():
            param.requires_grad = False
        if hasattr(self.vit, "pos_embed") and self.vit.pos_embed is not None:
            self.vit.pos_embed.requires_grad = False
        if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
            self.vit.cls_token.requires_grad = False

    def _freeze_blocks(self, n: int) -> None:
        """Freeze the first n transformer blocks."""
        blocks = list(self.vit.blocks)
        for block in blocks[:n]:
            for param in block.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm ViT with num_classes=0 returns the CLS token: (B, 768)
        features = self.vit(x)
        return self.head(features)

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def unfreeze_all(self) -> None:
        """Unfreeze entire network (for full fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}"


# -------------------------------------------------------------
# Factory function
# -------------------------------------------------------------

def create_advanced_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 5,
    pretrained: bool = True,
    freeze_blocks: int = -1,   # -1 -> use architecture defaults
    **kwargs,
) -> nn.Module:
    """
    Factory function for advanced model creation.

    Args:
        model_name:   ``'efficientnet_b0'`` or ``'vit_b16'``
        num_classes:  Number of output classes (default: 5)
        pretrained:   Whether to load ImageNet pretrained weights
        freeze_blocks: Number of blocks to freeze (-1 uses architecture default)
        **kwargs:     Extra keyword arguments passed to the model constructor

    Returns:
        An ``nn.Module`` subclass with proper 5-class head.

    Example::

        model = create_advanced_model("efficientnet_b0", num_classes=5, freeze_blocks=7)
        model = create_advanced_model("vit_b16", num_classes=5, freeze_blocks=10)
    """
    name = model_name.lower().replace("-", "_")

    if name in ("efficientnet_b0", "efficientnet"):
        fb = freeze_blocks if freeze_blocks >= 0 else 7
        return DeepSceneLocEfficientNetAdvanced(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_blocks=fb,
            **kwargs,
        )

    if name in ("vit_b16", "vit", "vit_base"):
        fb = freeze_blocks if freeze_blocks >= 0 else 10
        return DeepSceneLocViTAdvanced(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_encoder_blocks=fb,
            **kwargs,
        )

    raise ValueError(
        f"Unknown model '{model_name}'. Choose 'efficientnet_b0' or 'vit_b16'."
    )


# -------------------------------------------------------------
# Utility: model summary
# -------------------------------------------------------------

def model_summary(model: nn.Module, input_shape=(1, 3, 224, 224)) -> None:
    """
    Print a concise model summary: parameter counts and a single forward-pass check.

    Args:
        model:       Any nn.Module
        input_shape: Tuple (B, C, H, W) for the dummy forward pass
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print("=" * 60)
    print(f"  Model : {model.__class__.__name__}")
    print(f"  Total params     : {total:>12,}")
    print(f"  Trainable params : {trainable:>12,}")
    print(f"  Frozen params    : {frozen:>12,}")
    print("-" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dev = model.to(device).eval()
    dummy = torch.randn(*input_shape).to(device)

    with torch.no_grad():
        out = model_dev(dummy)

    print(f"  Input  shape : {tuple(dummy.shape)}")
    print(f"  Output shape : {tuple(out.shape)}")
    print("=" * 60)


# -------------------------------------------------------------
# Smoke test
# -------------------------------------------------------------

def _smoke_test():
    print("\n[Smoke test] EfficientNet-B0 (no pretrained)...")
    eff = create_advanced_model("efficientnet_b0", num_classes=5, pretrained=False)
    model_summary(eff)

    print("\n[Smoke test] ViT-B/16 (no pretrained)...")
    try:
        vit = create_advanced_model("vit_b16", num_classes=5, pretrained=False)
        model_summary(vit)
    except ImportError as e:
        print(f"  [SKIP] timm not installed: {e}")

    print("\n  All checks passed.\n")


if __name__ == "__main__":
    _smoke_test()
