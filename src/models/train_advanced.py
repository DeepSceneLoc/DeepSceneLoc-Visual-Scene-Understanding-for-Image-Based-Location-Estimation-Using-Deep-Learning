"""
Advanced Model Training — EfficientNet-B0 & Vision Transformer (ViT)
DeepSceneLoc — Semester 2, Weeks 8–9

Author: Anuj Kondawar (Preprocessing & Pipeline Lead — training support)
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
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# Training Configuration Dataclasses  (Anuj — Week 8 task)
# ─────────────────────────────────────────────────────────────

@dataclass
class EfficientNetTrainConfig:
    """
    Hyperparameter configuration for EfficientNet-B0 fine-tuning.

    Designed to outperform the ResNet-50 baseline while remaining
    trainable on a free Colab T4 GPU.
    """
    # Model
    model_name: str = "EfficientNet-B0"
    num_classes: int = 5
    freeze_blocks: int = 7           # Freeze first 7 MBConv blocks (of 9)

    # Optimizer
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)

    # Schedule
    scheduler: str = "cosine"        # cosine | step | none
    warmup_epochs: int = 0
    min_lr: float = 1e-6

    # Training loop
    epochs: int = 40
    batch_size: int = 32
    early_stopping_patience: int = 5
    gradient_clip: Optional[float] = None   # EfficientNet is usually stable

    # Loss
    loss: str = "cross_entropy"
    label_smoothing: float = 0.0

    # Data
    image_size: int = 224
    augment_train: bool = True
    num_workers: int = 4

    # Paths
    checkpoint_dir: str = "models/checkpoints/efficientnet"
    log_dir: str = "logs/efficientnet"

    # Success targets (from Semester 2 plan)
    target_val_accuracy: float = 0.75   # 75–80 % per spec


@dataclass
class ViTTrainConfig:
    """
    Hyperparameter configuration for Vision Transformer (ViT-B/16) fine-tuning.

    ViTs are more sensitive to LR; linear warm-up + cosine decay is required.
    """
    # Model
    model_name: str = "ViT-B/16"
    num_classes: int = 5
    freeze_encoder_blocks: int = 10    # Freeze first 10 of 12 transformer blocks

    # Optimizer
    optimizer: str = "AdamW"
    learning_rate: float = 5e-5        # ½× EfficientNet — ViT needs smaller LR
    weight_decay: float = 0.05         # Strong WD is important for transformers
    betas: Tuple[float, float] = (0.9, 0.999)

    # Schedule: linear warmup for `warmup_epochs`, then cosine decay
    scheduler: str = "warmup_cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Training loop
    epochs: int = 40
    batch_size: int = 32               # Use 16 if GPU memory < 8 GB
    early_stopping_patience: int = 7   # ViT takes longer to converge
    gradient_clip: float = 1.0         # Mandatory — prevents exploding gradients

    # Loss
    loss: str = "cross_entropy"
    label_smoothing: float = 0.1       # Regularization helps ViT generalize

    # Data
    image_size: int = 224
    augment_train: bool = True
    num_workers: int = 4

    # Paths
    checkpoint_dir: str = "models/checkpoints/vit"
    log_dir: str = "logs/vit"

    # Success targets
    target_val_accuracy: float = 0.78


# ─────────────────────────────────────────────────────────────
# LR Schedule Builders
# ─────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer: optim.Optimizer,
    config,
    steps_per_epoch: int,
) -> optim.lr_scheduler._LRScheduler:
    """
    Build a learning-rate scheduler from a training config.

    Supports:
      - ``'cosine'``        — CosineAnnealingLR over all epochs
      - ``'warmup_cosine'`` — Linear warm-up then cosine decay (required for ViT)
      - ``'none'``          — Constant LR (no scheduler)

    Args:
        optimizer: The optimizer to schedule.
        config: EfficientNetTrainConfig or ViTTrainConfig instance.
        steps_per_epoch: Number of batches per epoch (used for step-level warm-up).

    Returns:
        A PyTorch LR scheduler.
    """
    scheduler_name = config.scheduler.lower()
    total_epochs = config.epochs

    if scheduler_name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=config.min_lr,
        )

    if scheduler_name == "warmup_cosine":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - config.warmup_epochs,
            eta_min=config.min_lr,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_epochs],
        )

    # 'none' — return a trivial scheduler that never changes LR
    return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=config.learning_rate)


# ─────────────────────────────────────────────────────────────
# Advanced Trainer
# ─────────────────────────────────────────────────────────────

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
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        # Scheduler
        self.scheduler = build_scheduler(
            self.optimizer, config, len(train_loader)
        )

        # History
        self.history: Dict[str, List] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "lr":         [],
        }
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0

    # ──────────────────────────────────────────────────────────
    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        """One forward (and optionally backward) pass over ``loader``."""
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        desc = "Train" if train else "Val"
        with torch.set_grad_enabled(train):
            for imgs, labels in tqdm(loader, desc=desc, leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.model(imgs)
                loss    = self.criterion(outputs, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping — critical for ViT
                    if self.config.gradient_clip:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip
                        )

                    self.optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                _, preds    = torch.max(outputs, 1)
                correct    += (preds == labels).sum().item()
                total      += imgs.size(0)

        return total_loss / total, correct / total

    # ──────────────────────────────────────────────────────────
    def train(self) -> Dict:
        """
        Run the full training loop.

        Returns:
            Training history dictionary.
        """
        print(f"\n{'='*60}")
        print(f"Training {self.config.model_name}")
        print(f"  Device    : {self.device}")
        print(f"  Epochs    : {self.config.epochs}")
        print(f"  LR        : {self.config.learning_rate}")
        print(f"  Scheduler : {self.config.scheduler}")
        print(f"  Grad clip : {self.config.gradient_clip}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.perf_counter()

            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss,   val_acc   = self._run_epoch(self.val_loader,   train=False)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            elapsed = time.perf_counter() - t0

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            print(
                f"Ep {epoch:3d}/{self.config.epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"lr={current_lr:.2e}  ({elapsed:.1f}s)"
            )

            # Best model checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best val_acc={val_acc:.4f} — checkpoint saved")
            else:
                self.epochs_no_improve += 1

            # Periodic checkpoint every 5 epochs
            if epoch % 5 == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.epochs_no_improve >= self.config.early_stopping_patience:
                print(f"\nEarly stopping after {epoch} epochs "
                      f"(no improvement for {self.epochs_no_improve} epochs).")
                break

        # Final saves
        self._save_history()
        self._print_summary()
        return self.history

    # ──────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, is_best: bool):
        suffix = "best" if is_best else f"epoch{epoch:03d}"
        path   = self.ckpt_dir / f"{self.config.model_name.replace('/', '_')}_{suffix}.pth"
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "val_acc":         self.best_val_acc,
                "config":          asdict(self.config),
            },
            path,
        )

    def _save_history(self):
        path = self.log_dir / f"{self.config.model_name.replace('/', '_')}_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining history saved to {path}")

    def _print_summary(self):
        print(f"\n{'='*60}")
        print(f"Training Complete — {self.config.model_name}")
        print(f"  Best Val Accuracy : {self.best_val_acc:.4f} "
              f"({'TARGET MET' if self.best_val_acc >= self.config.target_val_accuracy else 'BELOW TARGET'})")
        print(f"  Target            : {self.config.target_val_accuracy:.4f}")
        print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# Hyperparameter Search  (Anuj — Week 8–9 tuning task)
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Checkpoint Management  (Anuj — monitoring / management task)
# ─────────────────────────────────────────────────────────────

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
            Checkpoint dict (keys: model_state, optimizer_state, …) or None.
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


# ─────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────

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
