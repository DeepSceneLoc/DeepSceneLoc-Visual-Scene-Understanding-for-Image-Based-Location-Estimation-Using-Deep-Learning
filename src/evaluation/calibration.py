"""
Temperature-Scaling Calibration + ECE
DeepSceneLoc -- Semester 2 (T4x2 accuracy/robustness upgrade)

Post-hoc probability calibration. A single scalar temperature ``T`` is fit on
the validation set by minimizing NLL (Guo et al. 2017, "On Calibration of
Modern Neural Networks"). At inference, divide logits by ``T`` before softmax:
this leaves the argmax (accuracy) unchanged but makes the predicted
probabilities trustworthy -- important for "any data" behavior where an
over-confident wrong prediction on an out-of-distribution image is worse than
a hesitant one.

Usage::

    from src.evaluation.calibration import fit_temperature, expected_calibration_error
    T = fit_temperature(val_logits, val_labels)     # scalar tensor
    probs = torch.softmax(logits / T, dim=1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader,
    device: str = "cuda",
    use_amp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a loader and return (logits, labels) on CPU."""
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(imgs)
        all_logits.append(out.float().cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.01,
) -> float:
    """
    Fit a single temperature scalar T > 0 that minimizes NLL on (logits, labels).

    Uses LBFGS (the standard optimizer for this 1-parameter problem).
    Returns the fitted temperature as a Python float.
    """
    logits = logits.detach().float()
    labels = labels.detach().long()

    log_T = torch.zeros(1, requires_grad=True)  # optimize log(T) to keep T > 0
    nll = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def _closure():
        optimizer.zero_grad()
        loss = nll(logits / log_T.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(log_T.exp().item())


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE): weighted gap between confidence and
    accuracy across ``n_bins`` confidence bins. Lower is better (0 = perfectly
    calibrated).
    """
    probs = probs.detach().float()
    labels = labels.detach().long()
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(lo) & confidences.le(hi)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc_bin = accuracies[in_bin].float().mean()
            conf_bin = confidences[in_bin].mean()
            ece += (acc_bin - conf_bin).abs() * prop
    return float(ece.item())


def calibrate_and_report(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    n_bins: int = 15,
) -> dict:
    """
    Fit T on the val set and report ECE before/after. Returns a dict with the
    fitted temperature and the two ECE values.
    """
    pre_probs = torch.softmax(val_logits, dim=1)
    pre_ece = expected_calibration_error(pre_probs, val_labels, n_bins)

    T = fit_temperature(val_logits, val_labels)

    post_probs = torch.softmax(val_logits / T, dim=1)
    post_ece = expected_calibration_error(post_probs, val_labels, n_bins)

    return {
        "temperature": T,
        "ece_before": pre_ece,
        "ece_after": post_ece,
    }
