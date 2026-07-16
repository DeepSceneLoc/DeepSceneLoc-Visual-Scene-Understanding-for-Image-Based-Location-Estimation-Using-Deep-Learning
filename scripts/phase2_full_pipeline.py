"""
Phase 2 full-data pipeline:
1) Resumable unzip of MIT Places365 archive into a dedicated folder.
2) Organize mapped images into 5 project classes (no cap).
3) Split into train/val/test.
4) Launch training on the new full-data split.

Run:
    .\\venv\\Scripts\\python.exe scripts\\phase2_full_pipeline.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_dataset import organise_images
from src.data.prepare_dataset import DatasetSplitter


def extract_zip_resumable(zip_path: Path, out_dir: Path, progress_every: int = 100000) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total = len(members)
        print(f"[extract] entries: {total:,}")
        print(f"[extract] target : {out_dir}")

        done = 0
        skipped = 0
        t0 = time.time()

        for i, member in enumerate(members, 1):
            if member.is_dir():
                continue
            target = out_dir / member.filename
            if target.exists():
                skipped += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, open(target, "wb") as dst:
                    dst.write(src.read())
                done += 1

            if i % progress_every == 0 or i == total:
                elapsed = (time.time() - t0) / 60
                pct = (i / total) * 100
                print(
                    f"[extract] {i:,}/{total:,} ({pct:.1f}%) | new={done:,} skipped={skipped:,} | {elapsed:.1f} min",
                    flush=True,
                )


def verify_required_folders(extract_root: Path) -> None:
    required = [
        extract_root / "train_256_places365standard",
        extract_root / "val_256",
        extract_root / "test_256",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        miss = "\n".join(str(p) for p in missing)
        raise RuntimeError(f"Missing expected extracted folders:\n{miss}")


def run_training(data_dir: Path, epochs: int, batch: int, workers: int, patience: int, min_delta: float) -> int:
    cmd = [
        str(ROOT / "venv" / "Scripts" / "python.exe"),
        "-u",
        str(ROOT / "run_training_resnet50.py"),
        "--data",
        str(data_dir),
        "--epochs",
        str(epochs),
        "--batch",
        str(batch),
        "--workers",
        str(workers),
        "--patience",
        str(patience),
        "--min-delta",
        str(min_delta),
    ]
    print("[train] command:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 full-data pipeline runner")
    parser.add_argument("--zip", default="dataset/Places-2_MIT_Dataset.zip")
    parser.add_argument("--extract", default="dataset/Places365_MIT_FULL_2026-03-15")
    parser.add_argument("--organised", default="data/processed_raw_mit_full_2026_03_15")
    parser.add_argument("--split-out", default="data/processed/places365_mit_full_2026_03_15")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--copy", action="store_true", help="Copy files during split (default: symlink)")
    args = parser.parse_args()

    zip_path = ROOT / args.zip
    extract_root = ROOT / args.extract
    organised_dir = ROOT / args.organised
    split_out = ROOT / args.split_out

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    print("=" * 70)
    print("Phase 2 Full Dataset Pipeline")
    print("=" * 70)
    print(f"zip       : {zip_path}")
    print(f"extract   : {extract_root}")
    print(f"organised : {organised_dir}")
    print(f"split out : {split_out}")
    print("=" * 70)

    print("\n[1/4] Resumable extraction")
    extract_zip_resumable(zip_path=zip_path, out_dir=extract_root)
    verify_required_folders(extract_root)

    print("\n[2/4] Organise mapped classes from MIT train folder")
    organise_images(
        raw_dir=extract_root / "train_256_places365standard",
        output_dir=organised_dir,
        categories_file=extract_root / "categories_places365.txt",
        max_per_class=None,
    )

    print("\n[3/4] Split into train/val/test")
    splitter = DatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    splitter.split_dataset(data_dir=str(organised_dir), output_dir=str(split_out), copy_files=args.copy)

    print("\n[4/4] Train model on new full split")
    code = run_training(
        data_dir=split_out,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    if code != 0:
        print(f"[train] failed with exit code {code}")
        return code

    print("[done] full pipeline completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
