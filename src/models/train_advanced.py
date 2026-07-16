"""
Advanced Model Training -- EfficientNet-B0 & Vision Transformer (ViT)
DeepSceneLoc -- Semester 2, Weeks 8-9

Author: Anuj Kondawar (Preprocessing & Pipeline Lead -- training support)
       Krishan Yadav (Model Architecture Lead)

This module provides:
  - Training-configuration dataclasses for EfficientNet-B0 and ViT
  - Hyperparameter search utilities
  - Training-run orchestration with checkpoint management
  - Cosine-annealing and linear-warmup LR schedules
  - Gradient clipping support (required for ViT stability)
"""

import json
import time
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR, OneCycleLR
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# AMP scaler -- gracefully disabled on CPU
_AMP_AVAILABLE = torch.cuda.is_available()


# -------------------------------------------------------------
# Training Configuration Dataclasses  (Anuj -- Week 8 task)
# -------------------------------------------------------------

@dataclass
class EfficientNetTrainConfig:
    """
    Hyperparameter configuration for EfficientNet-B0 fine-tuning.
    Updated (April 2026) with modern training techniques:
      - AMP (Automatic Mixed Precision)       -> 2-3x faster, -40% VRAM
      - MixUp / CutMix augmentation           -> +1-2% accuracy
      - Label smoothing 0.1                   -> +0.5% accuracy
      - EMA weights                           -> +0.3-0.5% accuracy
      - RandAugment                           -> stronger augmentation

    Tuning notes (April 27, 2026):
      - freeze_blocks reduced 7->4: more backbone layers fine-tuned for faster
        convergence and higher accuracy. ResNet-50 used freeze=0. Unfreeze
        last 5 MBConv blocks gives much better early-epoch accuracy.
      - mixup_alpha reduced 0.4->0.2: aggressive alpha=0.4 slows convergence
        badly in early epochs (ep1 val=20% = random). 0.2 is the
        standard for smaller datasets (<500K images).
      - grad_accum reduced 2->1: effective batch=64 is sufficient; accum=2
        was giving marginal benefit but slowing convergence.
      - lr raised 1e-4->2e-4: more backbone unfrozen needs slightly higher LR.
    """
    # Model
    model_name: str = "EfficientNet-B0"
    num_classes: int = 5
    freeze_blocks: int = 4           # Changed 7->4: unfreeze last 5 MBConv blocks

    # Optimizer
    optimizer: str = "AdamW"
    learning_rate: float = 2e-4      # Changed 1e-4->2e-4: more layers unfrozen
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)

    # Schedule
    scheduler: str = "onecycle"      # OneCycleLR: 10% warmup + cosine decay
    warmup_epochs: int = 3
    min_lr: float = 1e-6

    # Training loop
    epochs: int = 40
    batch_size: int = 64             # AMP allows 64 comfortably on 6GB RTX 3050
    grad_accum_steps: int = 1        # Changed 2->1: batch=64 is sufficient
    early_stopping_patience: int = 10 # Raised 7->10: MixUp/CutMix plateaus need more buffer
    gradient_clip: float = 1.0

    # Loss
    loss: str = "cross_entropy"
    label_smoothing: float = 0.1

    # Modern augmentation
    use_mixup: bool = True
    use_cutmix: bool = True
    mixup_alpha: float = 0.2         # Changed 0.4->0.2: less aggressive, faster convergence
    use_randaugment: bool = True
    randaugment_n: int = 2
    randaugment_m: int = 9
    use_random_erasing: bool = True

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999       # Tuned: 0.9998->0.9999. Half-life=6931 steps (~1.5 epochs).
                                    # Smoother averaging, less divergence during rapid backbone learning.

    # Precision
    use_amp: bool = True
    compile_model: bool = False

    # Full fine-tune helpers (April 2026 -- T4x2 upgrade)
    llrd_decay: Optional[float] = None   # layer-wise LR decay (e.g. 0.75); None = off
    use_swa: bool = False                # Stochastic Weight Averaging
    swa_start_frac: float = 0.75         # begin SWA after this fraction of epochs
    swa_lr: float = 1e-5                 # constant LR during SWA phase

    # Data
    image_size: int = 224
    augment_train: bool = True
    num_workers: int = 4

    # Paths
    checkpoint_dir: str = "models/checkpoints/efficientnet"
    log_dir: str = "logs/efficientnet"

    # Success targets
    target_val_accuracy: float = 0.80  # Raised: 4 unfrozen blocks should beat ResNet-50


@dataclass
class ViTTrainConfig:
    """
    Hyperparameter configuration for Vision Transformer (ViT-B/16) fine-tuning.
    Updated with modern 2024 techniques: AMP, MixUp, EMA, OneCycleLR.
    """
    # Model
    model_name: str = "ViT-B/16"
    num_classes: int = 5
    freeze_encoder_blocks: int = 10

    # Optimizer -- use AdamW with decoupled weight decay
    optimizer: str = "AdamW"
    learning_rate: float = 5e-5
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.999)

    # Schedule: warmup is MANDATORY for ViT stability
    scheduler: str = "warmup_cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-7

    # Training loop
    epochs: int = 40
    batch_size: int = 32            # ViT uses more VRAM -- keep at 32
    grad_accum_steps: int = 4       # effective batch = 32 x 4 = 128
    early_stopping_patience: int = 7
    gradient_clip: float = 1.0      # MANDATORY for ViT

    # Loss
    loss: str = "cross_entropy"
    label_smoothing: float = 0.1

    # Modern augmentation
    use_mixup: bool = True
    use_cutmix: bool = True
    mixup_alpha: float = 0.2        # lower alpha for ViT
    use_randaugment: bool = True
    randaugment_n: int = 2
    randaugment_m: int = 9
    use_random_erasing: bool = True

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999       # higher decay -- ViT benefits more from EMA

    # Precision
    use_amp: bool = True
    compile_model: bool = False

    # Full fine-tune helpers (April 2026 -- T4x2 upgrade)
    llrd_decay: Optional[float] = None   # layer-wise LR decay (e.g. 0.65 for ViT); None = off
    use_swa: bool = False
    swa_start_frac: float = 0.75
    swa_lr: float = 1e-6

    # Data
    image_size: int = 224
    augment_train: bool = True
    num_workers: int = 4

    # Paths
    checkpoint_dir: str = "models/checkpoints/vit"
    log_dir: str = "logs/vit"

    # Success targets
    target_val_accuracy: float = 0.82   # raised -- DINOv2-style training


# -------------------------------------------------------------
# LR Schedule Builders
# -------------------------------------------------------------

def build_scheduler(
    optimizer: optim.Optimizer,
    config,
    steps_per_epoch: int,
) -> optim.lr_scheduler._LRScheduler:
    """
    Build LR scheduler. Supports:
      - ``'onecycle'``      -- OneCycleLR (best for fixed-epoch training)
      - ``'cosine'``        -- CosineAnnealingLR
      - ``'warmup_cosine'`` -- Linear warm-up then cosine (required for ViT)
      - ``'none'``          -- Constant LR
    """
    name         = config.scheduler.lower()
    total_epochs = config.epochs
    accum        = getattr(config, "grad_accum_steps", 1)
    total_steps  = (steps_per_epoch // accum) * total_epochs

    if name == "onecycle":
        # Preserve per-group LRs (LLRD): OneCycleLR accepts a list of max_lr,
        # one per param group. Without this, a scalar max_lr collapses every
        # group to the same peak and defeats layer-wise decay.
        group_lrs = [g["lr"] for g in optimizer.param_groups]
        max_lr = group_lrs if len(group_lrs) > 1 else config.learning_rate
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.1,           # 10% warmup
            anneal_strategy="cos",
            div_factor=25,           # start lr = max_lr / 25
            final_div_factor=1e4,    # end lr = start_lr / 1e4
        )

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=config.min_lr,
        )

    if name == "warmup_cosine":
        warmup = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - config.warmup_epochs,
            eta_min=config.min_lr,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.warmup_epochs],
        )

    # 'none'
    return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=config.learning_rate)


# -------------------------------------------------------------
# Layer-wise LR Decay (LLRD) param groups
# -------------------------------------------------------------

def _ordered_stages(model: nn.Module) -> List[Tuple[str, List[nn.Parameter]]]:
    """
    Return backbone stages ordered SHALLOW -> DEEP, each paired with its
    parameters, plus the classification head last. Works for both advanced
    wrappers (DeepSceneLocEfficientNetAdvanced / DeepSceneLocViTAdvanced).

    Shallow layers (generic edges/textures) should learn slowest; the head
    fastest. LLRD assigns lr * decay^(depth_from_head).
    """
    stages: List[Tuple[str, List[nn.Parameter]]] = []

    if hasattr(model, "features"):          # EfficientNet: features[0..9] + head
        for i, block in enumerate(model.features):
            stages.append((f"features.{i}", list(block.parameters())))
        if hasattr(model, "head"):
            stages.append(("head", list(model.head.parameters())))

    elif hasattr(model, "vit"):             # ViT: patch_embed + blocks[..] + norm + head
        embed_params: List[nn.Parameter] = list(model.vit.patch_embed.parameters())
        for attr in ("cls_token", "pos_embed"):
            p = getattr(model.vit, attr, None)
            if isinstance(p, nn.Parameter):
                embed_params.append(p)
        stages.append(("patch_embed", embed_params))
        for i, block in enumerate(model.vit.blocks):
            stages.append((f"vit.blocks.{i}", list(block.parameters())))
        if hasattr(model.vit, "norm"):
            stages.append(("vit.norm", list(model.vit.norm.parameters())))
        if hasattr(model, "head"):
            stages.append(("head", list(model.head.parameters())))

    else:                                   # fallback: single group of all params
        stages.append(("all", list(model.parameters())))

    return stages


def build_llrd_param_groups(
    model: nn.Module,
    base_lr: float,
    decay: float,
    weight_decay: float = 0.0,
) -> List[Dict]:
    """
    Build optimizer param groups with layer-wise LR decay.

    The head gets ``base_lr``; each stage one step shallower gets multiplied
    by ``decay`` (0 < decay < 1). Only trainable params are included, so this
    composes with any freezing. Norm/bias params keep their group's LR but
    get no weight decay (standard practice).
    """
    stages = _ordered_stages(model)
    n = len(stages)
    groups: List[Dict] = []

    for idx, (name, params) in enumerate(stages):
        # depth from head: head (last) -> 0, shallowest -> n-1
        depth_from_head = (n - 1) - idx
        lr = base_lr * (decay ** depth_from_head)

        decay_p = [p for p in params if p.requires_grad and p.ndim >= 2]
        nodecay_p = [p for p in params if p.requires_grad and p.ndim < 2]

        if decay_p:
            groups.append({"params": decay_p, "lr": lr, "weight_decay": weight_decay,
                           "name": f"{name}_decay"})
        if nodecay_p:
            groups.append({"params": nodecay_p, "lr": lr, "weight_decay": 0.0,
                           "name": f"{name}_nodecay"})

    return groups

class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Maintains a shadow copy of parameters for more stable evaluation.

    Usage::
        ema = ModelEMA(model, decay=0.9998)
        # After each optimizer.step():
        ema.update(model)
        # For evaluation, use ema.module instead of model
    """
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(
            self.module.parameters(), model.parameters()
        ):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.module.state_dict()


# -------------------------------------------------------------
# MixUp / CutMix
# -------------------------------------------------------------

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation. Returns mixed_x, y_a, y_b, lam."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix augmentation. Returns mixed_x, y_a, y_b, lam."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    batch_size, _, H, W = x.size()
    idx = torch.randperm(batch_size, device=x.device)

    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed, y, y[idx], lam


def mixup_criterion(
    criterion: nn.Module,
    outputs: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """MixUp/CutMix-compatible loss: lam * L(pred, y_a) + (1-lam) * L(pred, y_b)."""
    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)



# -------------------------------------------------------------
# Advanced Trainer
# -------------------------------------------------------------

class AdvancedTrainer:
    """
    Training orchestrator for EfficientNet-B0 and ViT models.

    Extends the Semester 1 ``Trainer`` with:
      - Gradient clipping
      - Label smoothing in the loss
      - Warmup-cosine scheduling
      - Per-epoch JSON logging
      - Resumable checkpoints (saves optimizer + scheduler state)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = "cuda",
        resume_checkpoint: dict = None,
        multi_gpu: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Paths
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.log_dir  = Path(config.log_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Loss
        try:
            from src.preprocessing.pipeline import get_class_weights
            class_weights = get_class_weights(self.train_loader.dataset).to(device)
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=config.label_smoothing
            )
            print(f"  Loss   : CrossEntropy (Class weights: {class_weights.cpu().numpy().round(2)})")
        except Exception as e:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=config.label_smoothing
            )
            print(f"  Loss   : CrossEntropy (No class weights: {e})")

        # Optimizer -- layer-wise LR decay (LLRD) when configured, else a
        # single group over all trainable params.
        llrd_decay = getattr(config, "llrd_decay", None)
        if llrd_decay:
            param_groups = build_llrd_param_groups(
                model, base_lr=config.learning_rate,
                decay=llrd_decay, weight_decay=config.weight_decay,
            )
            self.optimizer = optim.AdamW(param_groups, betas=config.betas)
            print(f"  LLRD   : ON (decay={llrd_decay}, {len(param_groups)} param groups)")
        else:
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=config.betas,
            )

        # Gradient accumulation
        self.accum_steps = getattr(config, "grad_accum_steps", 1)

        # Scheduler
        self.scheduler = build_scheduler(
            self.optimizer, config, len(train_loader)
        )
        self._is_onecycle = config.scheduler.lower() == "onecycle"

        # AMP scaler -- use torch.amp API (cuda.amp deprecated in PyTorch 2.0+)
        self.use_amp = getattr(config, "use_amp", False) and _AMP_AVAILABLE
        self.scaler  = torch.amp.GradScaler('cuda', enabled=self.use_amp)


        # EMA -- built from the un-wrapped model (deepcopy must stay a plain module)
        self.use_ema = getattr(config, "use_ema", False)
        self.ema     = ModelEMA(self.model, decay=getattr(config, "ema_decay", 0.9998)) \
                       if self.use_ema else None

        # MixUp / CutMix flags
        self.use_mixup  = getattr(config, "use_mixup",  False)
        self.use_cutmix = getattr(config, "use_cutmix", False)
        self.mixup_alpha = getattr(config, "mixup_alpha", 0.4)

        # Multi-GPU -- wrap AFTER EMA/optimizer are bound to the raw module.
        # DataParallel forwards .parameters()/.state_dict() to the wrapped
        # module (with a "module." prefix on state_dict keys), so checkpoint
        # saving must unwrap via .module.
        self._raw_model = self.model  # keep handle to the un-wrapped module
        self.use_dp = (
            multi_gpu
            and torch.device(device).type == "cuda"
            and torch.cuda.device_count() > 1
        )
        if self.use_dp:
            print(f"  [DataParallel] Training across {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # SWA (Stochastic Weight Averaging) -- averages weights over the tail
        # of training for flatter minima / better generalization. Built on the
        # RAW module (DP shares the same param tensors) so the averaged copy
        # stays a plain module and saves arch-native.
        self.use_swa = getattr(config, "use_swa", False)
        if self.use_swa:
            from torch.optim.swa_utils import AveragedModel, SWALR
            self.swa_model = AveragedModel(self._raw_model)
            self.swa_start = max(1, int(config.epochs * getattr(config, "swa_start_frac", 0.75)))
            self.swa_scheduler = SWALR(
                self.optimizer, swa_lr=getattr(config, "swa_lr", 1e-5)
            )
            self.swa_n = 0
            print(f"  SWA    : ON (start epoch={self.swa_start}, swa_lr={getattr(config, 'swa_lr', 1e-5)})")
        else:
            self.swa_model = None
            self.swa_start = None
        self._swa_active = False

        # torch.compile (PyTorch 2.0+)
        if getattr(config, "compile_model", False):
            try:
                self.model = torch.compile(self.model)
                print("  [torch.compile] Model compiled for faster inference")
            except Exception as e:
                print(f"  [torch.compile] Skipped: {e}")

        # History
        self.history: Dict[str, List] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "ema_val_acc": [],
            "lr":         [],
        }
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.start_epoch = 1

        if resume_checkpoint is not None:
            if "optimizer_state" in resume_checkpoint:
                self.optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            if "scheduler_state" in resume_checkpoint:
                self.scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
            if "scaler_state" in resume_checkpoint and self.use_amp:
                self.scaler.load_state_dict(resume_checkpoint["scaler_state"])
            if "ema_state" in resume_checkpoint and self.use_ema:
                self.ema.module.load_state_dict(resume_checkpoint["ema_state"])
            
            self.start_epoch = resume_checkpoint.get("epoch", 0) + 1
            self.best_val_acc = resume_checkpoint.get("val_acc", 0.0)
            print(f"  [Resume] Full training state restored. Starting at epoch {self.start_epoch}.")

        print(f"  AMP    : {'ON' if self.use_amp else 'OFF (CPU)'}")
        print(f"  EMA    : {'ON  (decay=' + str(getattr(config, 'ema_decay', '--')) + ')' if self.use_ema else 'OFF'}")
        print(f"  MixUp  : {'ON' if self.use_mixup else 'OFF'}  CutMix: {'ON' if self.use_cutmix else 'OFF'}")
        print(f"  GradAccum: {self.accum_steps}x  (effective batch={config.batch_size * self.accum_steps})")

    # ----------------------------------------------------------
    def _run_epoch(
        self, loader: DataLoader, epoch: int, train: bool
    ) -> Tuple[float, float]:
        """One epoch with AMP, MixUp/CutMix, gradient accumulation."""
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        use_mix = train and (self.use_mixup or self.use_cutmix)

        desc = f"Ep {epoch}/{self.config.epochs} [{ 'Train' if train else 'Val' }]"
        self.optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

        with torch.set_grad_enabled(train):
            for step, (imgs, labels) in enumerate(tqdm(loader, desc=desc, leave=False)):
                imgs   = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # -- MixUp / CutMix (training only) ------------
                if use_mix:
                    # Alternate between CutMix and MixUp each batch
                    if self.use_cutmix and step % 2 == 0:
                        imgs, y_a, y_b, lam = cutmix_data(imgs, labels, self.mixup_alpha)
                    else:
                        imgs, y_a, y_b, lam = mixup_data(imgs, labels, self.mixup_alpha)

                # -- Forward (AMP) -----------------------------
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(imgs)
                    if use_mix:
                        loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)

                    # Scale for accumulation
                    loss = loss / self.accum_steps

                # -- Backward (AMP-safe) -----------------------
                if train:
                    self.scaler.scale(loss).backward()

                    # Step only every accum_steps batches
                    if (step + 1) % self.accum_steps == 0 or (step + 1) == len(loader):
                        if self.config.gradient_clip:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.config.gradient_clip
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                        # EMA update
                        if self.use_ema:
                            self.ema.update(self.model)

                        # Step-level scheduler (OneCycleLR). Suspended once the
                        # SWA phase begins -- SWALR takes over per-epoch.
                        if self._is_onecycle and not self._swa_active:
                            self.scheduler.step()

                # -- Metrics -----------------------------------
                unscaled_loss = loss.item() * self.accum_steps
                total_loss += unscaled_loss * imgs.size(0)
                _, preds    = torch.max(outputs.detach(), 1)
                if use_mix:
                    # Accuracy against the primary label
                    correct += (preds == labels).sum().item()
                else:
                    correct += (preds == labels).sum().item()
                total += imgs.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    def _eval_ema(self, loader: DataLoader) -> float:
        """[DEPRECATED - EMA now evaluated inside _run_epoch to avoid double val pass]
        Kept for backward compat. Returns 0.0."""
        return 0.0

    def _run_val_with_ema(self, loader: DataLoader, epoch: int) -> tuple:
        """
        Single pass over val loader that SIMULTANEOUSLY evaluates
        both the raw model and the EMA model.
        Returns (val_loss, val_acc, ema_acc) in one pass.
        Eliminates the extra 62K-image EMA evaluation that was causing
        the long inter-epoch delay.
        """
        self.model.eval()
        if self.use_ema:
            self.ema.module.eval()

        total_loss, correct_raw, correct_ema, total = 0.0, 0, 0, 0
        desc = f"Ep {epoch}/{self.config.epochs} [Val]"

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=desc, leave=False):
                imgs   = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    out_raw = self.model(imgs)
                    loss    = self.criterion(out_raw, labels)

                total_loss += loss.item() * imgs.size(0)
                _, preds_raw = torch.max(out_raw, 1)
                correct_raw += (preds_raw == labels).sum().item()

                # EMA forward — no extra data loading, same batch
                if self.use_ema:
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        out_ema = self.ema.module(imgs)
                    _, preds_ema = torch.max(out_ema, 1)
                    correct_ema += (preds_ema == labels).sum().item()

                total += imgs.size(0)

        n = max(total, 1)
        val_loss = total_loss / n
        val_acc  = correct_raw / n
        ema_acc  = correct_ema / n if self.use_ema else 0.0
        return val_loss, val_acc, ema_acc


    # ----------------------------------------------------------
    def train(self) -> Dict:
        """
        Run the full training loop with modern techniques:
        AMP, MixUp/CutMix, EMA, gradient accumulation, OneCycleLR.
        """
        print(f"\n{'='*60}")
        print(f"Training {self.config.model_name}")
        print(f"  Device    : {self.device}")
        print(f"  Epochs    : {self.config.epochs}")
        print(f"  LR        : {self.config.learning_rate}")
        print(f"  Scheduler : {self.config.scheduler}")
        print(f"  Grad clip : {self.config.gradient_clip}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            t0 = time.perf_counter()

            train_loss, train_acc = self._run_epoch(self.train_loader, epoch=epoch, train=True)

            # Single val pass evaluates both raw model AND EMA simultaneously
            # (was previously TWO full passes over 62K images -- fixed)
            val_loss, val_acc, ema_acc = self._run_val_with_ema(self.val_loader, epoch=epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step epoch-level schedulers. Once the SWA phase starts, SWALR
            # drives the LR and we accumulate averaged weights instead.
            if self.use_swa and epoch >= self.swa_start:
                if not self._swa_active:
                    self._swa_active = True
                    print(f"  [SWA] Entering SWA phase at epoch {epoch}")
                self.swa_model.update_parameters(self._raw_model)
                self.swa_n += 1
                self.swa_scheduler.step()
            elif not self._is_onecycle:
                self.scheduler.step()

            elapsed = time.perf_counter() - t0

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["ema_val_acc"].append(ema_acc)
            self.history["lr"].append(current_lr)

            # Effective acc for early stopping and checkpointing.
            # Use max(val_acc, ema_acc) -- whichever is higher wins.
            # Rationale: after a checkpoint resume with changed freeze_blocks, the EMA
            # shadow model can diverge (it averages incompatible old+new weights) while
            # the raw val_acc keeps climbing. Blindly trusting EMA caused early-stop at
            # epoch 9 despite val_acc improving from 80% -> 84%. Using max() is safe:
            # if EMA is better we use it, if val_acc is better we use that.
            effective_acc = max(val_acc, ema_acc) if self.use_ema else val_acc

            ema_str = f"  ema_acc={ema_acc:.4f}" if self.use_ema else ""
            print(
                f"Ep {epoch:3d}/{self.config.epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{ema_str}  "
                f"lr={current_lr:.2e}  ({elapsed:.1f}s)"
            )

            # Best model checkpoint
            if effective_acc > self.best_val_acc:
                self.best_val_acc = effective_acc
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, is_best=True)
                print(f"  [BEST] New best val_acc={effective_acc:.4f} -- checkpoint saved")
            else:
                self.epochs_no_improve += 1

            # Periodic checkpoint every 5 epochs
            if epoch % 5 == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Early stopping -- disabled during the SWA phase (we want to run
            # the full averaging tail, not stop early).
            if not self._swa_active and self.epochs_no_improve >= self.config.early_stopping_patience:
                print(f"\nEarly stopping after {epoch} epochs "
                      f"(no improvement for {self.epochs_no_improve} epochs).")
                break

        # SWA finalization -- recompute BatchNorm running stats over the train
        # set for the averaged weights (mandatory), evaluate, and save.
        if self.use_swa and self.swa_n > 0:
            self._finalize_swa()

        # Final saves
        self._save_history()
        self._print_summary()
        return self.history

    # ----------------------------------------------------------
    def _finalize_swa(self):
        """Update BN stats for the SWA-averaged model, evaluate, and save it."""
        from torch.optim.swa_utils import update_bn
        print(f"\n  [SWA] Finalizing averaged model over {self.swa_n} snapshots ...")
        # update_bn accepts (input, target) batches and moves data to device;
        # it runs a forward-only pass to recompute BN running stats.
        self.swa_model.to(self.device)
        update_bn(self.train_loader, self.swa_model, device=self.device)

        # Evaluate SWA model on val
        self.swa_model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    out = self.swa_model(imgs)
                _, preds = torch.max(out, 1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)
        swa_acc = correct / max(total, 1)
        self.history["swa_val_acc"] = swa_acc
        print(f"  [SWA] Averaged-model val_acc = {swa_acc:.4f}")

        # Save SWA weights (unwrap AveragedModel.module -> arch-native state)
        path = self.ckpt_dir / f"{self.config.model_name.replace('/', '_')}_swa.pth"
        torch.save({
            "epoch": self.config.epochs,
            "model_state": self.swa_model.module.state_dict(),
            "val_acc": swa_acc,
            "config": asdict(self.config),
            "swa_snapshots": self.swa_n,
        }, path)
        print(f"  [SWA] Saved -> {path}")


    # ----------------------------------------------------------
    def _save_checkpoint(self, epoch: int, is_best: bool):
        suffix = "best" if is_best else f"epoch{epoch:03d}"
        path   = self.ckpt_dir / f"{self.config.model_name.replace('/', '_')}_{suffix}.pth"
        model_to_save = self.model.module if self.use_dp else self.model
        state = {
            "epoch":           epoch,
            "model_state":     model_to_save.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state":    self.scaler.state_dict() if self.use_amp else {},
            "val_acc":         self.best_val_acc,
            "config":          asdict(self.config),
        }
        if self.use_ema:
            state["ema_state"] = self.ema.state_dict()
        torch.save(state, path)

    def _save_history(self):
        path = self.log_dir / f"{self.config.model_name.replace('/', '_')}_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining history saved to {path}")

    def _print_summary(self):
        print(f"\n{'='*60}")
        print(f"Training Complete -- {self.config.model_name}")
        print(f"  Best Val Accuracy : {self.best_val_acc:.4f} "
              f"({'TARGET MET' if self.best_val_acc >= self.config.target_val_accuracy else 'BELOW TARGET'})")
        print(f"  Target            : {self.config.target_val_accuracy:.4f}")
        print(f"{'='*60}")


# -------------------------------------------------------------
# Hyperparameter Search  (Anuj -- Week 8-9 tuning task)
# -------------------------------------------------------------

class HyperparameterSearch:
    """
    Grid / random search over a set of training configurations.

    Usage::

        from src.models.train_advanced import HyperparameterSearch, EfficientNetTrainConfig

        search = HyperparameterSearch(
            base_config=EfficientNetTrainConfig(),
            param_grid={
                "learning_rate": [1e-3, 1e-4, 5e-5],
                "weight_decay":  [0.0, 1e-4, 1e-3],
            },
        )
        best_config, results = search.run(train_loader, val_loader, model_factory)
    """

    def __init__(self, base_config, param_grid: Dict[str, List]):
        self.base_config = base_config
        self.param_grid  = param_grid

    def _make_configs(self) -> List:
        """Enumerate all combinations in the grid."""
        from itertools import product
        keys   = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        configs = []
        for combo in product(*values):
            cfg = type(self.base_config)(**asdict(self.base_config))
            for k, v in zip(keys, combo):
                setattr(cfg, k, v)
            configs.append(cfg)
        return configs

    def run(self, train_loader, val_loader, model_factory) -> Tuple[object, List[Dict]]:
        """
        Run the full grid search.

        Args:
            train_loader: DataLoader for training.
            val_loader: DataLoader for validation.
            model_factory: Callable ``(config) -> nn.Module``.

        Returns:
            ``(best_config, results)`` where ``results`` is a list of dicts
            with ``config``, ``best_val_acc``, ``history``.
        """
        configs = self._make_configs()
        print(f"HyperparameterSearch: {len(configs)} configurations to evaluate")

        results = []
        best_acc, best_cfg = 0.0, None

        for i, cfg in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] lr={cfg.learning_rate}  wd={cfg.weight_decay}")
            model   = model_factory(cfg)
            device  = "cuda" if torch.cuda.is_available() else "cpu"
            trainer = AdvancedTrainer(model, train_loader, val_loader, cfg, device)
            history = trainer.train()

            val_acc = max(history["val_acc"])
            results.append({
                "config":       asdict(cfg),
                "best_val_acc": val_acc,
                "history":      history,
            })

            if val_acc > best_acc:
                best_acc = val_acc
                best_cfg  = cfg

        print(f"\nBest config: lr={best_cfg.learning_rate}, wd={best_cfg.weight_decay}")
        print(f"Best val_acc: {best_acc:.4f}")
        return best_cfg, results


# -------------------------------------------------------------
# Checkpoint Management  (Anuj -- monitoring / management task)
# -------------------------------------------------------------

class CheckpointManager:
    """
    Manages loading, listing, and comparing model checkpoints.

    Usage::

        mgr = CheckpointManager("models/checkpoints")
        summary = mgr.list_checkpoints()
        model_state = mgr.load_best("EfficientNet-B0")
    """

    def __init__(self, checkpoint_root: str = "models/checkpoints"):
        self.root = Path(checkpoint_root)

    def list_checkpoints(self) -> List[Dict]:
        """Return metadata for all saved checkpoints."""
        ckpts = []
        for path in sorted(self.root.rglob("*.pth")):
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                ckpts.append({
                    "path":     str(path),
                    "model":    data.get("config", {}).get("model_name", "unknown"),
                    "epoch":    data.get("epoch", -1),
                    "val_acc":  data.get("val_acc", 0.0),
                })
            except Exception:
                pass
        return ckpts

    def load_best(self, model_name: str) -> Optional[Dict]:
        """
        Load the best checkpoint for a given model name.

        Args:
            model_name: E.g., ``'EfficientNet-B0'`` or ``'ViT-B/16'``.

        Returns:
            Checkpoint dict (keys: model_state, optimizer_state, ...) or None.
        """
        candidates = [c for c in self.list_checkpoints() if c["model"] == model_name]
        if not candidates:
            print(f"No checkpoints found for {model_name}")
            return None
        best = max(candidates, key=lambda x: x["val_acc"])
        print(f"Loading best checkpoint: {best['path']}  (val_acc={best['val_acc']:.4f})")
        return torch.load(best["path"], map_location="cpu", weights_only=False)

    def print_summary(self):
        """Print a formatted table of all available checkpoints."""
        ckpts = self.list_checkpoints()
        if not ckpts:
            print("No checkpoints found.")
            return
        print(f"\n{'Model':<25} {'Epoch':>6} {'Val Acc':>10}  Path")
        print("-" * 80)
        for c in sorted(ckpts, key=lambda x: -x["val_acc"]):
            print(f"{c['model']:<25} {c['epoch']:>6} {c['val_acc']:>10.4f}  {c['path']}")


# -------------------------------------------------------------
# Quick smoke-test
# -------------------------------------------------------------

def _smoke_test():
    """Verify configs and trainer init without actual training."""
    print("Smoke-testing train_advanced.py...")

    eff_cfg = EfficientNetTrainConfig()
    vit_cfg = ViTTrainConfig()

    print(f"  EfficientNet config: lr={eff_cfg.learning_rate}, "
          f"epochs={eff_cfg.epochs}, freeze={eff_cfg.freeze_blocks} blocks")
    print(f"  ViT config         : lr={vit_cfg.learning_rate}, "
          f"epochs={vit_cfg.epochs}, grad_clip={vit_cfg.gradient_clip}, "
          f"warmup={vit_cfg.warmup_epochs} epochs")

    # Test JSON serialisation
    _ = json.dumps(asdict(eff_cfg))
    _ = json.dumps(asdict(vit_cfg))

    mgr = CheckpointManager()
    mgr.print_summary()

    print("  All checks passed.")


if __name__ == "__main__":
    _smoke_test()
