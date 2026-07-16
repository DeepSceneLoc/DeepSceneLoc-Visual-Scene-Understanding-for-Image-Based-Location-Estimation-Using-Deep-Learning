"""
Ensemble Evaluation Entry Point
DeepSceneLoc -- Semester 2 (T4x2 accuracy/robustness upgrade)

Combines several trained checkpoints into one calibrated, TTA-averaged
classifier and evaluates it on the test set. Ensembling independently-trained
backbones typically beats the best single model by 1-2% and is more robust on
arbitrary real-world images.

Usage (activate venv first):
    # Combine the three best checkpoints, val-acc weighted + calibrated
    python scripts/training/run_ensemble_eval.py \
        --ckpts models/checkpoints/efficientnet/EfficientNet-B0_best.pth \
                models/checkpoints/vit/ViT-B_16_best.pth \
                models/checkpoints/resnet/best_model.pth \
        --data data/processed --weight-by-val-acc --calibrate

    # Plain uniform average, no calibration
    python scripts/training/run_ensemble_eval.py --ckpts A.pth B.pth --data data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

# -- Project root on sys.path ---------------------------------
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.pipeline import create_dataloaders
from src.preprocessing.transforms import get_val_transforms
from src.evaluation.ensemble import EnsembleEvaluator
from src.evaluation.calibration import expected_calibration_error

CLASS_NAMES = ["Coastal", "Forest", "Mountain", "Rural", "Urban"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a multi-model ensemble on the DeepSceneLoc test set"
    )
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="Paths to checkpoint .pth files to ensemble")
    p.add_argument("--data", default="data/processed",
                   help="Processed dataset root (needs val/ + test/ subfolders)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--weight-by-val-acc", action="store_true",
                   help="Weight each model by its standalone val accuracy")
    p.add_argument("--calibrate", action="store_true",
                   help="Fit temperature scaling per model on the val set")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def get_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if not allow_cpu:
        print("ERROR: CUDA unavailable. Use --allow-cpu for debugging.")
        sys.exit(1)
    return torch.device("cpu")


def main():
    args = parse_args()
    device = get_device(args.allow_cpu)

    # Validate checkpoint paths
    missing = [c for c in args.ckpts if not Path(c).exists()]
    if missing:
        print("ERROR: checkpoints not found:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = Path(f"results/ensemble_{RUN_TS}")

    print("\n" + "=" * 65)
    print("  DeepSceneLoc -- Ensemble Evaluation")
    print(f"  Models    : {len(args.ckpts)}")
    for c in args.ckpts:
        print(f"    - {c}")
    print(f"  Data      : {args.data}")
    print(f"  Weighting : {'val-acc' if args.weight_by_val_acc else 'uniform'}")
    print(f"  Calibrate : {args.calibrate}")
    print(f"  Device    : {device}")
    print("=" * 65 + "\n")

    # Aspect-preserving val/test transforms (the fixed pipeline)
    val_tfm = get_val_transforms(image_size=224)
    _, val_loader, test_loader = create_dataloaders(
        data_dir=args.data,
        batch_size=args.batch,
        num_workers=args.workers,
        image_size=224,
        augment_train=False,
        val_transform=val_tfm,
        pin_memory=torch.cuda.is_available(),
    )

    evaluator = EnsembleEvaluator(
        ckpt_paths=args.ckpts,
        class_names=CLASS_NAMES,
        device=str(device),
    )

    if args.weight_by_val_acc:
        print("[1/3] Weighting by validation accuracy ...")
        evaluator.set_weights_from_val_acc(val_loader)

    if args.calibrate:
        print("[2/3] Temperature calibration ...")
        evaluator.calibrate(val_loader)

    print("[3/3] Ensemble test-set evaluation ...")
    metrics = evaluator.evaluate(test_loader)

    # Report calibration quality of the ensemble output (ECE on test)
    import numpy as np
    probs_t = torch.tensor(np.array(evaluator.all_probabilities))
    labels_t = torch.tensor(np.array(evaluator.all_labels))
    ece = expected_calibration_error(probs_t, labels_t)
    metrics["ensemble_ece"] = ece
    metrics["temperatures"] = evaluator.temperatures
    metrics["weights"] = list(map(float, evaluator.weights))

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "ensemble_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Overall accuracy : {metrics['overall_accuracy']:.4f}")
    print(f"  Ensemble ECE     : {ece:.4f}")
    print(f"  Metrics -> {out_path}")


if __name__ == "__main__":
    main()
