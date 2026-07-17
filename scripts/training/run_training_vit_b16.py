"""
Training Entry Point -- Vision Transformer (ViT-B/16)
DeepSceneLoc -- Semester 2, Week 9

Authors:
    Krishan Yadav  (Model Architecture Lead)
    Anuj Kondawar  (Pipeline & Training Lead)

This script is the entry point for training the ViT-B/16
advanced model. It wraps:
  - src.models.model_advanced   (ViT definition)
  - src.models.train_advanced   (AdvancedTrainer, ViTTrainConfig)
  - src.preprocessing.pipeline  (create_dataloaders)
  - src.evaluation.evaluate     (ModelEvaluator)
  - src.utils.visualizations    (create_all_visualizations)

Usage (activate venv first):
    # Train
    python run_training_vit_b16.py

    # Multi-GPU (e.g. Kaggle T4 x2) -- auto-detected, DataParallel used when
    # >1 CUDA device visible. --batch is the TOTAL batch across all GPUs.
    python run_training_vit_b16.py --batch 64                 # 32/GPU on T4 x2
    python run_training_vit_b16.py --no-multi-gpu             # force single-GPU

    # Max-accuracy full fine-tune on T4 x2 (LLRD + SWA)
    python run_training_vit_b16.py --batch 96 --epochs 45 \\
        --full-finetune --swa

    # Custom config
    python run_training_vit_b16.py \\
        --epochs 40 --batch 32 --lr 1e-4 --freeze-blocks 7

    # Smoke test (2 epochs, 5 batches)
    python run_training_vit_b16.py --dry-run

    # Evaluate a saved checkpoint
    python run_training_vit_b16.py --eval-only \\
        --resume models/checkpoints/vit/ViT-B_16_epoch030.pth

    # Resume interrupted training
    python run_training_vit_b16.py \\
        --resume models/checkpoints/vit/ViT-B_16_epoch010.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

# -- Project root on sys.path ---------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model_advanced import create_advanced_model, model_summary
from src.models.train_advanced import (
    AdvancedTrainer,
    CheckpointManager,
    ViTTrainConfig,
)
from src.preprocessing.pipeline import create_dataloaders
from src.preprocessing.transforms import get_modern_train_transforms, get_val_transforms
from src.evaluation.evaluate import ModelEvaluator
from src.utils.visualizations import create_all_visualizations

MODEL_NAME = "vit_b16"


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train ViT-B/16 on the DeepSceneLoc dataset"
    )
    # Data
    p.add_argument("--data", default="data/processed",
                   help="Path to processed dataset root (must have train/val/test subfolders)")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader workers (set 0 on Windows if spawn errors occur)")

    # Training hyperparameters (override config defaults)
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--batch",        type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None,
                   help="Initial learning rate")
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--patience",     type=int,   default=None,
                   help="Early stopping patience (epochs)")
    p.add_argument("--freeze-blocks", type=int,  default=None,
                   help="Number of encoder blocks to freeze")
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--grad-clip",    type=float, default=None)
    p.add_argument("--no-aug",       action="store_true",
                   help="Disable training augmentation (use plain transforms)")
    p.add_argument("--no-amp",       action="store_true",
                   help="Disable AMP mixed precision (default: ON on GPU)")
    p.add_argument("--no-ema",       action="store_true",
                   help="Disable EMA weight averaging")
    p.add_argument("--no-mixup",     action="store_true",
                   help="Disable MixUp / CutMix augmentation")
    p.add_argument("--accum",        type=int, default=None,
                   help="Gradient accumulation steps (default: from config)")

    # Operational modes
    p.add_argument("--resume",    default=None,
                   help="Path to .pth checkpoint to resume from")
    p.add_argument("--auto-resume", action="store_true",
                   help="Automatically find and resume from the latest checkpoint for the model")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training and only evaluate best saved checkpoint")
    p.add_argument("--dry-run",   action="store_true",
                   help="Smoke test: 2 epochs x 5 batches (no checkpoint saved)")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Allow CPU execution (disabled by default; slow!)")
    p.add_argument("--no-multi-gpu", action="store_true",
                   help="Disable nn.DataParallel even if multiple GPUs are visible")

    # Full fine-tune / regularization (T4x2 upgrade)
    p.add_argument("--full-finetune", action="store_true",
                   help="Unfreeze the whole encoder (freeze-blocks=0) + enable LLRD")
    p.add_argument("--llrd", type=float, default=None,
                   help="Layer-wise LR decay factor (e.g. 0.65 for ViT). Implied by --full-finetune")
    p.add_argument("--swa", action="store_true",
                   help="Enable Stochastic Weight Averaging over the training tail")

    return p.parse_args()


# -------------------------------------------------------------
# Device
# -------------------------------------------------------------

def get_device(allow_cpu: bool = False) -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        for i in range(n_gpu):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i} : {name}  ({vram:.1f} GB VRAM)")
        if n_gpu > 1:
            print(f"  Detected {n_gpu} GPUs -- DataParallel used unless --no-multi-gpu is set")
    else:
        if not allow_cpu:
            print("ERROR: CUDA not available and --allow-cpu not set.")
            print("       Use --allow-cpu only for debugging.")
            sys.exit(1)
        dev = torch.device("cpu")
        print("  GPU  : not available -- running on CPU (--allow-cpu enabled)")
    return dev


# -------------------------------------------------------------
# Build config from args
# -------------------------------------------------------------

def build_config(args):
    """
    Create the ViTTrainConfig, overriding defaults with any CLI
    arguments the user explicitly provided.
    """
    cfg = ViTTrainConfig()

    # Apply CLI overrides
    if args.epochs          is not None: cfg.epochs                  = args.epochs
    if args.batch           is not None: cfg.batch_size              = args.batch
    if args.lr              is not None: cfg.learning_rate           = args.lr
    if args.weight_decay    is not None: cfg.weight_decay            = args.weight_decay
    if args.patience        is not None: cfg.early_stopping_patience = args.patience
    if args.label_smoothing is not None: cfg.label_smoothing         = args.label_smoothing
    if args.grad_clip       is not None: cfg.gradient_clip           = args.grad_clip
    if args.accum           is not None: cfg.grad_accum_steps        = args.accum
    if args.no_aug:                      cfg.augment_train           = False
    if args.no_amp:                      cfg.use_amp                 = False
    if args.no_ema:                      cfg.use_ema                 = False
    if args.no_mixup:
        cfg.use_mixup  = False
        cfg.use_cutmix = False

    # Full fine-tune + LLRD + SWA
    if args.full_finetune:
        cfg.llrd_decay = args.llrd if args.llrd is not None else 0.65
    elif args.llrd is not None:
        cfg.llrd_decay = args.llrd
    if args.swa:
        cfg.use_swa = True

    # dry-run: minimal epochs/patience
    if args.dry_run:
        cfg.epochs               = 2
        cfg.early_stopping_patience = 999
        cfg.use_ema              = False   # EMA not useful on 5-batch dry-run
        cfg.use_amp              = cfg.use_amp  # keep AMP for smoke test

    return cfg


# -------------------------------------------------------------
# Evaluation helper
# -------------------------------------------------------------

CLASS_NAMES = ["Coastal", "Forest", "Mountain", "Rural", "Urban"]


def _run_evaluation(model, test_loader, device, results_dir: Path, log_dir: Path, cfg_model_name: str = None):
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        class_names=CLASS_NAMES,
        device=str(device),
    )
    metrics = evaluator.evaluate()
    evaluator.print_confusion_analysis()

    # Save metrics JSON
    results_dir.mkdir(parents=True, exist_ok=True)
    model_tag = model.__class__.__name__
    metrics_path = results_dir / f"{model_tag}_evaluation.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics -> {metrics_path}")

    # Visualisations
    # History JSON is saved as {ModelName}_history.json (e.g. ViT-B_16_history.json)
    # NOT as training_history.json -- was a bug causing history_path to always be None
    model_name_safe = cfg_model_name.replace('/', '_') if cfg_model_name else None
    if model_name_safe:
        history_path = log_dir / f"{model_name_safe}_history.json"
    else:
        # fallback: search for any *_history.json in log_dir
        candidates = list(log_dir.glob("*_history.json"))
        history_path = candidates[0] if candidates else log_dir / "training_history.json"
    vis_dir = results_dir / "visualizations"
    create_all_visualizations(
        metrics=metrics,
        history_path=str(history_path) if history_path.exists() else None,
        output_dir=str(vis_dir),
    )


# -------------------------------------------------------------
# Helper: load checkpoint into model
# -------------------------------------------------------------

def _load_checkpoint(model, ckpt_path: str, device: torch.device):
    path = Path(ckpt_path)
    if not path.exists():
        print(f"  [WARN] Checkpoint not found: {path}")
        return None
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    key = "model_state" if "model_state" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key])
    epoch   = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_acc", "?")
    print(f"  Checkpoint loaded  epoch={epoch}  val_acc={val_acc}")
    return ckpt


# -------------------------------------------------------------
# Dry-run patch (limit batches)
# -------------------------------------------------------------

def _patch_trainer_dry_run(trainer: AdvancedTrainer, max_batches: int = 100):
    """Monkey-patch the trainer to only process max_batches per phase."""
    from tqdm import tqdm

    orig_run_epoch = trainer._run_epoch

    def _dry_epoch(loader, train=True, epoch=0, **kwargs):
        trainer.model.train(train)
        total_loss = correct = total = 0

        with torch.set_grad_enabled(train):
            for i, (imgs, labels) in enumerate(loader):
                if i >= max_batches:
                    break
                imgs, labels = imgs.to(trainer.device), labels.to(trainer.device)
                outputs = trainer.model(imgs)
                loss    = trainer.criterion(outputs, labels)

                if train:
                    trainer.optimizer.zero_grad()
                    loss.backward()
                    if trainer.config.gradient_clip:
                        torch.nn.utils.clip_grad_norm_(
                            trainer.model.parameters(), trainer.config.gradient_clip
                        )
                    trainer.optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                _, preds    = torch.max(outputs, 1)
                correct    += (preds == labels).sum().item()
                total      += imgs.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    def _dry_val_with_ema(loader, epoch=0, **kwargs):
        """Patched validation: limits to max_batches and returns 3-tuple."""
        trainer.model.eval()
        total_loss = correct = total = 0

        with torch.no_grad():
            for i, (imgs, labels) in enumerate(loader):
                if i >= max_batches:
                    break
                imgs, labels = imgs.to(trainer.device), labels.to(trainer.device)
                outputs = trainer.model(imgs)
                loss    = trainer.criterion(outputs, labels)

                total_loss += loss.item() * imgs.size(0)
                _, preds    = torch.max(outputs, 1)
                correct    += (preds == labels).sum().item()
                total      += imgs.size(0)

        n = max(total, 1)
        return total_loss / n, correct / n, 0.0  # ema_acc = 0 for dry-run

    trainer._run_epoch = _dry_epoch
    trainer._run_val_with_ema = _dry_val_with_ema


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    args   = parse_args()
    device = get_device(allow_cpu=args.allow_cpu)
    cfg    = build_config(args)

    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR   = Path(f"results/{cfg.model_name.replace('/', '_')}_{RUN_TIMESTAMP}")
    LOG_DIR       = Path(cfg.log_dir)
    CKPT_DIR      = Path(cfg.checkpoint_dir)

    print("\n" + "=" * 65)
    print(f"  DeepSceneLoc -- ViT-B/16 Training (Modern Pipeline)")
    print(f"  Model     : {cfg.model_name}")
    print(f"  Data      : {args.data}")
    print(f"  Epochs    : {cfg.epochs}  Batch: {cfg.batch_size}")
    print(f"  LR        : {cfg.learning_rate}  Scheduler: {cfg.scheduler}")
    print(f"  Smoothing : {cfg.label_smoothing}  GradClip: {cfg.gradient_clip}")
    print(f"  Patience  : {cfg.early_stopping_patience}")
    print(f"  --- Modern Techniques (2024) ----------------------------")
    print(f"  AMP       : {'ON (2-3x faster + -40pct VRAM)' if cfg.use_amp else 'OFF'}")
    print(f"  EMA       : {'ON (decay=' + str(cfg.ema_decay) + ')' if cfg.use_ema else 'OFF'}")
    print(f"  MixUp     : {'ON (alpha=' + str(cfg.mixup_alpha) + ')' if cfg.use_mixup else 'OFF'}")
    print(f"  CutMix    : {'ON' if cfg.use_cutmix else 'OFF'}")
    print(f"  RandAug   : {'ON (N=' + str(cfg.randaugment_n) + ', M=' + str(cfg.randaugment_m) + ')' if cfg.use_randaugment else 'OFF'}")
    print(f"  RandErase : {'ON' if cfg.use_random_erasing else 'OFF'}")
    print(f"  GradAccum : {cfg.grad_accum_steps}x (eff. batch = {cfg.batch_size * cfg.grad_accum_steps})")
    print(f"  Device    : {device}")
    print(f"  Results   : {RESULTS_DIR}")
    print("=" * 65 + "\n")

    # -- Freeze blocks for model creation
    freeze_blocks = args.freeze_blocks  # None -> use architecture default
    if args.full_finetune:
        freeze_blocks = 0               # unfreeze entire encoder

    # -- Build model --------------------------------------------
    print("[1/4] Building model ...")
    model = create_advanced_model(
        model_name=MODEL_NAME,
        num_classes=len(CLASS_NAMES),
        pretrained=True,
        freeze_blocks=freeze_blocks if freeze_blocks is not None else -1,
    )
    model_summary(model)

    # -- Optionally load checkpoint -----------------------------
    resume_ckpt = None
    if args.auto_resume:
        mgr = CheckpointManager(str(CKPT_DIR))
        ckpts = [c for c in mgr.list_checkpoints() if c["model"] == cfg.model_name]
        if ckpts:
            latest = max(ckpts, key=lambda x: x["epoch"])
            print(f"\n  [Auto-Resume] Found latest checkpoint: {latest['path']} (Epoch {latest['epoch']})")
            resume_ckpt = _load_checkpoint(model, latest['path'], device)
        else:
            print(f"\n  [Auto-Resume] No checkpoints found for {cfg.model_name}. Starting fresh.")
    elif args.resume:
        print(f"\n  Resuming from: {args.resume}")
        resume_ckpt = _load_checkpoint(model, args.resume, device)

    # -- Data loaders -------------------------------------------
    print("\n[2/4] Building data loaders ...")
    n_workers = args.workers

    # Modern transforms for train split
    if cfg.augment_train:
        train_tfm = get_modern_train_transforms(
            image_size=cfg.image_size,
            use_randaugment=cfg.use_randaugment,
            randaugment_n=cfg.randaugment_n,
            randaugment_m=cfg.randaugment_m,
            use_random_erasing=cfg.use_random_erasing,
        )
    else:
        train_tfm = get_val_transforms(image_size=cfg.image_size)

    val_tfm = get_val_transforms(image_size=cfg.image_size)

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=args.data,
            batch_size=cfg.batch_size,
            num_workers=n_workers,
            image_size=cfg.image_size,
            augment_train=False,            # transforms handled above
            train_transform=train_tfm,
            val_transform=val_tfm,
            pin_memory=torch.cuda.is_available(),
        )
    except TypeError:
        # Older pipeline doesn't accept train_transform kw -- fall back
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=args.data,
            batch_size=cfg.batch_size,
            num_workers=n_workers,
            image_size=cfg.image_size,
            augment_train=cfg.augment_train,
        )
    except Exception as exc:
        if n_workers > 0:
            print(f"  Workers={n_workers} failed ({exc}); retrying with 0 ...")
            n_workers = 0
            train_loader, val_loader, test_loader = create_dataloaders(
                data_dir=args.data,
                batch_size=cfg.batch_size,
                num_workers=0,
                image_size=cfg.image_size,
                augment_train=cfg.augment_train,
            )
        else:
            raise

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Test  batches : {len(test_loader)}")

    # -- Eval-only mode -----------------------------------------
    if args.eval_only:
        if not args.resume:
            # Fall back to CheckpointManager best checkpoint
            mgr = CheckpointManager(str(CKPT_DIR))
            best = mgr.load_best(cfg.model_name)
            if best is None:
                print("ERROR: No checkpoint found and --resume not specified.")
                sys.exit(1)
            model.load_state_dict(best["model_state"])
        model.to(device)
        print("\n[3/4] Evaluation only (skipping training) ...")
        _run_evaluation(model, test_loader, device, RESULTS_DIR / "metrics", LOG_DIR, cfg.model_name)
        return

    # -- Training -----------------------------------------------
    print(f"\n[3/4] Training {cfg.model_name} ...")

    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=str(device),
        resume_checkpoint=resume_ckpt if not args.eval_only else None,
        multi_gpu=not args.no_multi_gpu,
    )

    if args.dry_run:
        print("  DRY-RUN: limiting to 5 batches per phase.")
        _patch_trainer_dry_run(trainer, max_batches=5)

    history = trainer.train()

    # -- Save config alongside history -------------------------
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = LOG_DIR / f"{cfg.model_name.replace('/', '_')}_config.json"
    with open(cfg_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"\n  Config saved -> {cfg_path}")

    # -- Post-training evaluation -------------------------------
    if not args.dry_run:
        print("\n[4/4] Post-training evaluation on test set ...")
        mgr  = CheckpointManager(str(CKPT_DIR))
        best = mgr.load_best(cfg.model_name)
        if best:
            model.load_state_dict(best["model_state"])
        model.to(device)
        _run_evaluation(model, test_loader, device, RESULTS_DIR / "metrics", LOG_DIR, cfg.model_name)
    else:
        print("\n[4/4] Dry-run complete -- skipping test-set evaluation.")


if __name__ == "__main__":
    main()
