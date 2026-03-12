"""
Training Entry Point — ResNet-50 Baseline (Weeks 5-7)
Krishan Yadav (Lead) | Anuj Kondawar (Pipeline) | Aditi Sah (Data) | Jensi Paneliya (Eval)

Trains the ResNet-50 scene classifier on the prepared Places365 subset,
then runs evaluation and generates all visualisations.

Usage:
    # Activate venv first:
    #   .\\venv\\Scripts\\activate   (Windows)
    #   source venv/bin/activate    (Linux/Mac)

    python run_training.py                          # default settings
    python run_training.py --epochs 20 --batch 32  # explicit
    python run_training.py --data data/processed    # custom data path
    python run_training.py --dry-run                # smoke-test (5 batches)
    python run_training.py --resume models/checkpoints/best_model.pth
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model import create_model
from src.models.train import Trainer
from src.preprocessing.pipeline import create_dataloaders, test_preprocessing_pipeline
from src.evaluation.evaluate import ModelEvaluator
from src.utils.visualizations import create_all_visualizations


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet-50 scene classifier")
    p.add_argument("--data",    default="data/processed",  help="Processed dataset root")
    p.add_argument("--model",   default="resnet50",         help="Model name (resnet50)")
    p.add_argument("--epochs",  type=int,   default=20)
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--lr",      type=float, default=0.001)
    p.add_argument("--workers", type=int,   default=4,
                   help="DataLoader workers (set 0 on Windows if you hit spawn errors)")
    p.add_argument("--no-aug",  action="store_true", help="Disable training augmentation")
    p.add_argument("--resume",  default=None,          help="Path to checkpoint .pth to resume from")
    p.add_argument("--dry-run", action="store_true",   help="5-batch smoke-test, no checkpointing")
    p.add_argument("--eval-only", action="store_true", help="Skip training, only evaluate best_model.pth")
    return p.parse_args()


# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU : {name}  ({vram:.1f} GB VRAM)")
    else:
        dev = torch.device("cpu")
        print("  GPU : not available — training on CPU (will be slow)")
    return dev


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = get_device()

    CLASS_NAMES = ["Coastal", "Forest", "Mountain", "Rural", "Urban"]
    NUM_CLASSES = len(CLASS_NAMES)
    CHECKPOINT_DIR = "models/checkpoints"
    LOG_DIR = "logs"
    RESULTS_DIR = "results"

    print("\n" + "=" * 60)
    print("DeepSceneLoc — ResNet-50 Training")
    print("=" * 60)
    print(f"  Data    : {args.data}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  LR      : {args.lr}")
    print(f"  Workers : {args.workers}")
    print(f"  Device  : {device}")
    print(f"  Dry-run : {args.dry_run}")
    print("=" * 60 + "\n")

    # ── Validate preprocessing ────────────────────────────────────────────────
    if not args.eval_only:
        print("[Step 1/4] Validating preprocessing pipeline ...")
        test_preprocessing_pipeline(data_dir=args.data)

    # ── Data loaders ─────────────────────────────────────────────────────────
    print("[Step 2/4] Building data loaders ...")
    # Windows multiprocessing: if workers>0 causes spawn errors, fall back to 0
    num_workers = args.workers
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=args.data,
            batch_size=args.batch,
            num_workers=num_workers,
            image_size=224,
            augment_train=not args.no_aug,
        )
    except Exception as e:
        if num_workers > 0:
            print(f"  DataLoader with {num_workers} workers failed ({e}), retrying with 0 ...")
            num_workers = 0
            train_loader, val_loader, test_loader = create_dataloaders(
                data_dir=args.data,
                batch_size=args.batch,
                num_workers=0,
                image_size=224,
                augment_train=not args.no_aug,
            )
        else:
            raise

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Test  batches : {len(test_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[Step 3/4] Building model ({args.model}) ...")
    model = create_model(model_name=args.model, num_classes=NUM_CLASSES, pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable:,}")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_val_acc = 0.0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f"  Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("best_val_acc", 0.0)
            print(f"  Resumed at epoch {start_epoch-1}, best val acc {best_val_acc:.2f}%")
        else:
            print(f"  Warning: checkpoint {ckpt_path} not found, starting fresh")

    if args.eval_only:
        # Load best model and evaluate
        best_path = Path(CHECKPOINT_DIR) / "best_model.pth"
        if not best_path.exists():
            print(f"ERROR: {best_path} not found. Train first.")
            sys.exit(1)
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        _run_evaluation(model, test_loader, CLASS_NAMES, device, RESULTS_DIR, LOG_DIR)
        return

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[Step 4/4] Training ...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # Restore optimiser/scheduler state when resuming
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        save_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR,
    )
    trainer.best_val_acc = best_val_acc

    # Inject a dry-run cap into the trainer if requested
    if args.dry_run:
        print("  DRY-RUN mode: limiting to 5 batches per epoch, 2 epochs")
        args.epochs = 2
        _patch_trainer_dry_run(trainer, max_batches=5)

    num_epochs = max(1, args.epochs - (start_epoch - 1))
    trainer.train(num_epochs=num_epochs, save_frequency=5)

    # ── Post-training evaluation ──────────────────────────────────────────────
    if not args.dry_run:
        print("\n" + "=" * 60)
        print("Post-training evaluation on test set ...")
        best_path = Path(CHECKPOINT_DIR) / "best_model.pth"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        _run_evaluation(model, test_loader, CLASS_NAMES, device, RESULTS_DIR, LOG_DIR)


def _run_evaluation(model, test_loader, class_names, device, results_dir, log_dir):
    """Run full evaluation + visualisations."""
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=str(device),
    )
    metrics = evaluator.evaluate()

    # Save metrics JSON
    metrics_dir = Path(results_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "resnet50_evaluation.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {metrics_path}")

    # Misclassification analysis
    evaluator.print_confusion_analysis()

    # All visualisations
    history_path = Path(log_dir) / "training_history.json"
    create_all_visualizations(
        metrics=metrics,
        history_path=str(history_path),
        output_dir=str(Path(results_dir) / "visualizations"),
    )


def _patch_trainer_dry_run(trainer: "Trainer", max_batches: int = 5):
    """Monkey-patch trainer so each phase only runs *max_batches* batches."""
    orig_train = trainer.train_epoch
    orig_val   = trainer.validate

    def _dry_train(epoch):
        trainer.model.train()
        from tqdm import tqdm
        import torch
        running_loss = correct = total = 0
        for i, (inputs, labels) in enumerate(trainer.train_loader):
            if i >= max_batches:
                break
            inputs, labels = inputs.to(trainer.device), labels.to(trainer.device)
            trainer.optimizer.zero_grad()
            outputs = trainer.model(inputs)
            loss = trainer.criterion(outputs, labels)
            loss.backward()
            trainer.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        return running_loss / max(total, 1), 100 * correct / max(total, 1)

    def _dry_val(epoch):
        trainer.model.eval()
        import torch
        running_loss = correct = total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(trainer.val_loader):
                if i >= max_batches:
                    break
                inputs, labels = inputs.to(trainer.device), labels.to(trainer.device)
                outputs = trainer.model(inputs)
                loss = trainer.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        return running_loss / max(total, 1), 100 * correct / max(total, 1)

    trainer.train_epoch = _dry_train
    trainer.validate    = _dry_val


if __name__ == "__main__":
    main()
