"""
Multi-Model Ensemble Inference + Evaluation
DeepSceneLoc -- Semester 2 (T4x2 accuracy/robustness upgrade)

Averages calibrated softmax probabilities across several trained checkpoints
(ResNet-50 / EfficientNet-B0 / ViT-B16, or any mix) with multi-view TTA per
model. Ensembling independently-trained backbones reduces variance and
corrects each model's individual mistakes -- typically +1-2% over the best
single member, and more robust on out-of-distribution ("any data") images.

Design:
  - Architecture is inferred from each checkpoint's saved ``config.model_name``
    (advanced ckpts) or falls back to ResNet-50 for legacy ``model_state_dict``
    checkpoints that lack a config.
  - Optional per-model weighting by validation accuracy.
  - Optional temperature-scaling calibration fit per model on the val set.
  - Reuses ``ModelEvaluator._calculate_metrics`` so the output JSON schema
    matches single-model runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm

from src.models.model import create_model
from src.models.model_advanced import create_advanced_model
from src.evaluation.evaluate import ModelEvaluator
from src.evaluation.calibration import collect_logits, fit_temperature


# Map saved config.model_name -> factory + arch key
_ADVANCED_NAMES = {
    "EfficientNet-B0": ("advanced", "efficientnet_b0"),
    "ViT-B/16":        ("advanced", "vit_b16"),
}


def _infer_arch(ckpt: dict) -> str:
    """Return an arch key ('resnet50' | 'efficientnet_b0' | 'vit_b16')."""
    cfg = ckpt.get("config", {})
    name = cfg.get("model_name") if isinstance(cfg, dict) else None
    if name in _ADVANCED_NAMES:
        return _ADVANCED_NAMES[name][1]
    # Legacy ResNet checkpoints: model_state_dict, no config
    return "resnet50"


def build_model_from_checkpoint(
    ckpt_path: str,
    num_classes: int = 5,
    device: str = "cuda",
) -> nn.Module:
    """Load a checkpoint, infer its architecture, and return a ready model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = _infer_arch(ckpt)

    if arch == "resnet50":
        model = create_model("resnet50", num_classes=num_classes, pretrained=False)
    else:
        model = create_advanced_model(arch, num_classes=num_classes, pretrained=False)

    # State dict lives under different keys across trainers
    state = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"No model weights found in {ckpt_path}")
    # Strip any stray DataParallel prefix for safety
    state = { k.replace("module.", "", 1): v for k, v in state.items() }
    model.load_state_dict(state)
    model.to(device).eval()
    return model


class EnsembleEvaluator:
    """
    Evaluate an ensemble of models on a test set with per-model TTA,
    optional val-accuracy weighting, and optional temperature calibration.
    """

    def __init__(
        self,
        ckpt_paths: List[str],
        class_names: List[str],
        device: str = "cuda",
        weights: Optional[List[float]] = None,
    ):
        self.ckpt_paths = ckpt_paths
        self.class_names = class_names
        self.device = device
        self.num_classes = len(class_names)

        self.models = [
            build_model_from_checkpoint(p, self.num_classes, device)
            for p in ckpt_paths
        ]
        # Per-model weight (default uniform). Normalized at use time.
        if weights is None:
            weights = [1.0] * len(self.models)
        self.weights = weights
        self.temperatures = [1.0] * len(self.models)  # filled by calibrate()

        self.all_predictions: List[int] = []
        self.all_labels: List[int] = []
        self.all_probabilities: List[np.ndarray] = []

    # ----------------------------------------------------------
    def calibrate(self, val_loader, use_amp: bool = False):
        """Fit a temperature per model on the val set (in place)."""
        print("Calibrating ensemble members on validation set ...")
        for i, model in enumerate(self.models):
            logits, labels = collect_logits(model, val_loader, self.device, use_amp)
            T = fit_temperature(logits, labels)
            self.temperatures[i] = T
            print(f"  model[{i}] ({Path(self.ckpt_paths[i]).name}): T={T:.3f}")

    def set_weights_from_val_acc(self, val_loader):
        """Weight each model by its standalone val accuracy."""
        accs = []
        for i, model in enumerate(self.models):
            logits, labels = collect_logits(model, val_loader, self.device)
            acc = (logits.argmax(1) == labels).float().mean().item()
            accs.append(acc)
            print(f"  model[{i}] val_acc={acc:.4f}")
        self.weights = accs

    # ----------------------------------------------------------
    @torch.no_grad()
    def _tta_probs(self, model: nn.Module, inputs: torch.Tensor, T: float) -> torch.Tensor:
        """Original + h-flip TTA, temperature-scaled softmax."""
        logits = model(inputs) / T
        probs = torch.softmax(logits, dim=1)
        flipped = torch.flip(inputs, dims=[3])
        probs = probs + torch.softmax(model(flipped) / T, dim=1)
        return probs / 2.0

    @torch.no_grad()
    def evaluate(self, test_loader) -> Dict:
        """Run ensemble inference and compute the standard metric dict."""
        w = np.asarray(self.weights, dtype=np.float64)
        w = w / w.sum()  # normalize

        print("=" * 60)
        print(f"Ensemble Evaluation ({len(self.models)} models)")
        print("=" * 60)

        for inputs, labels in tqdm(test_loader, desc="Ensemble eval"):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            batch_probs = None
            for i, model in enumerate(self.models):
                p = self._tta_probs(model, inputs, self.temperatures[i])
                p = p * float(w[i])
                batch_probs = p if batch_probs is None else batch_probs + p

            _, predicted = torch.max(batch_probs, 1)
            self.all_predictions.extend(predicted.cpu().numpy())
            self.all_labels.extend(labels.cpu().numpy())
            self.all_probabilities.extend(batch_probs.cpu().numpy())

        # Reuse ModelEvaluator's metric computation + printing for schema parity
        proxy = ModelEvaluator.__new__(ModelEvaluator)
        proxy.class_names = self.class_names
        proxy.all_predictions = np.array(self.all_predictions)
        proxy.all_labels = np.array(self.all_labels)
        proxy.all_probabilities = np.array(self.all_probabilities)
        metrics = proxy._calculate_metrics()
        proxy._print_results(metrics)
        return metrics
