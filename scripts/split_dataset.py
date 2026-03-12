"""
Dataset Splitter — creates train / val / test splits
Week 3 — Dataset Preparation (Aditi Sah - Lead)

Takes the organised category folders (output of download_dataset.py) and
splits them 70 / 15 / 15 into data/processed/{train,val,test}/<Category>/.

Usage:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --data data/processed_raw --out data/processed
    python scripts/split_dataset.py --seed 42
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prepare_dataset import DatasetSplitter


def main():
    p = argparse.ArgumentParser(description="Split organised dataset into train/val/test")
    p.add_argument("--data",  default="data/processed_raw",
                   help="Source directory with <Category>/ folders (default: data/processed_raw)")
    p.add_argument("--out",   default="data/processed",
                   help="Destination root (default: data/processed)")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val",   type=float, default=0.15)
    p.add_argument("--test",  type=float, default=0.15)
    p.add_argument("--seed",  type=int,   default=42)
    p.add_argument("--copy",  action="store_true",
                   help="Copy files instead of symlinking (needed on some Windows configs)")
    args = p.parse_args()

    splitter = DatasetSplitter(
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )
    splitter.split_dataset(
        data_dir=args.data,
        output_dir=args.out,
        copy_files=args.copy,
    )

    print(f"\nSplit complete -> {args.out}/{{train,val,test}}/<Category>/")
    print("  Ready to train:  python run_training.py")


if __name__ == "__main__":
    main()
