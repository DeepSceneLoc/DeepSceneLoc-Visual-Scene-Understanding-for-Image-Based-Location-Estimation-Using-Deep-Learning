"""
Dataset Download Script — Places365
Week 3 — Dataset Preparation (Aditi Sah - Lead)

Downloads the Places365-Standard dataset via the Kaggle API and organises
images into the 5 project location categories:
  Urban | Rural | Coastal | Mountain | Forest

Prerequisites:
    1. Install kaggle:  pip install kaggle
    2. Get API key from https://www.kaggle.com/settings → API → Create New Token
    3. Place kaggle.json at %USERPROFILE%\\.kaggle\\kaggle.json  (Windows)
       or  ~/.kaggle/kaggle.json  (Linux/Mac)

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output data/raw
    python scripts/download_dataset.py --max-per-class 500   # quick smoke-test
    python scripts/download_dataset.py --demo               # tiny torchvision demo set
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# Kaggle dataset slug for Places365
KAGGLE_DATASET = "benjaminkz/places365"

# ── Category mapping (mirrors src/data/prepare_dataset.py) ────────────────────
LOCATION_MAPPING: Dict[str, List[str]] = {
    "Urban": [
        "street", "downtown", "building_facade", "skyscraper",
        "plaza", "shopping_mall", "parking_lot", "crosswalk",
        "highway", "alley", "bridge", "subway_station",
        "apartment_building", "office_building", "hotel_outdoor",
        "restaurant", "cafe", "bus_station", "parking_garage",
    ],
    "Rural": [
        "field", "farm", "barn", "pasture", "cornfield",
        "wheat_field", "rice_paddy", "vegetable_garden",
        "village", "farmhouse", "windmill", "vineyard",
        "orchard", "herb_garden", "cottage_garden", "scarecrow",
    ],
    "Coastal": [
        "beach", "coast", "ocean", "sea", "harbor", "pier",
        "lighthouse", "marina", "dock", "boardwalk",
        "seashore", "sandbar", "wave", "cliff_coastal",
        "bay", "cove", "peninsula", "beach_house",
    ],
    "Mountain": [
        "mountain", "mountain_snowy", "mountain_path", "canyon",
        "valley", "cliff", "rock_arch", "butte", "volcano",
        "summit", "ridge", "hill", "peak", "ski_slope",
        "ski_resort", "alpine", "highland", "plateau",
    ],
    "Forest": [
        "forest", "rainforest", "bamboo_forest", "tree_farm",
        "woodland", "jungle", "grove", "palm_grove",
        "forest_path", "forest_road", "sequoia", "swamp",
        "bog", "marsh", "wetland", "creek", "river_forest",
    ],
}

# Reverse mapping: places365_keyword → location_category
REVERSE_MAPPING: Dict[str, str] = {
    kw: cat
    for cat, keywords in LOCATION_MAPPING.items()
    for kw in keywords
}


def _map_places365_label(label: str) -> Optional[str]:
    """Map a raw Places365 label (e.g. '/a/abbey') to one of the 5 categories."""
    # labels look like: /a/airfield  or  airfield
    name = label.strip().lstrip("/").split("/")[-1].lower()

    # direct match
    if name in REVERSE_MAPPING:
        return REVERSE_MAPPING[name]

    # partial match
    for kw, cat in REVERSE_MAPPING.items():
        if kw in name or name in kw:
            return cat

    return None  # unmapped — will be skipped


class ProgressBar(tqdm):
    last_block = 0

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update((b - self.last_block) * bsize)
        self.last_block = b


def download_via_kaggle(dataset_slug: str, output_dir: Path) -> Path:
    """Download a Kaggle dataset to *output_dir* using the kaggle CLI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading Kaggle dataset: {dataset_slug}")
    print(f"  Destination: {output_dir}")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", dataset_slug, "-p", str(output_dir), "--unzip"],
        check=True,
    )
    print("  Kaggle download complete.")
    return output_dir


def create_demo_dataset(output_dir: Path, samples_per_class: int = 50):
    """
    Create a tiny demo dataset using CIFAR-10 images (~60 MB download).

    CIFAR-10 classes are mapped to the 5 project location categories:
      Mountain <- airplane
      Urban    <- automobile, truck
      Forest   <- bird, dog
      Rural    <- cat, deer, horse
      Coastal  <- frog, ship

    The demo dataset is NOT Places365 — only for testing the training pipeline.
    """
    try:
        import torchvision.datasets as tvd
        from PIL import Image as PILImage
    except ImportError:
        print("torchvision not installed — cannot create demo dataset.")
        sys.exit(1)

    # CIFAR-10 class index -> location category
    CIFAR10_TO_LOCATION: Dict[int, str] = {
        0: "Mountain",  # airplane
        1: "Urban",     # automobile
        2: "Forest",    # bird
        3: "Rural",     # cat
        4: "Rural",     # deer
        5: "Forest",    # dog
        6: "Coastal",   # frog
        7: "Rural",     # horse
        8: "Coastal",   # ship
        9: "Urban",     # truck
    }

    print("  Downloading CIFAR-10 (~60 MB) ...")
    raw_dir = output_dir / "_cifar10_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset = tvd.CIFAR10(root=str(raw_dir), train=True, download=True, transform=None)

    saved: Dict[str, int] = {c: 0 for c in LOCATION_MAPPING}
    for cat in LOCATION_MAPPING:
        (output_dir / cat).mkdir(parents=True, exist_ok=True)

    print(f"  Sampling up to {samples_per_class} images per location category ...")
    for img, label in tqdm(dataset, desc="Saving demo images"):
        loc_cat = CIFAR10_TO_LOCATION[label]
        if saved[loc_cat] >= samples_per_class:
            continue
        dest = output_dir / loc_cat / f"{loc_cat}_{saved[loc_cat]:05d}.jpg"
        img.resize((224, 224)).save(dest, quality=90)
        saved[loc_cat] += 1
        if all(v >= samples_per_class for v in saved.values()):
            break

    print(f"  Demo dataset ready at {output_dir}")
    for c, n in saved.items():
        print(f"    {c}: {n} images")



def organise_images(
    raw_dir: Path,
    output_dir: Path,
    categories_file: Path,
    max_per_class: Optional[int] = None,
):
    """
    Walk *raw_dir* (Places365 original structure: <raw_dir>/<letter>/<category>/
    or flat), map each image to one of the 5 location categories, and copy it
    into *output_dir*/<category>/.

    Creates a summary JSON at *output_dir*/organisation_stats.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a set of all official Places365 category names if the file is available
    p365_labels: Dict[str, int] = {}
    if categories_file.exists():
        with open(categories_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    p365_labels[parts[0]] = int(parts[1])

    counters: Dict[str, int] = {cat: 0 for cat in LOCATION_MAPPING}
    counters["unmapped"] = 0
    skipped = 0

    # Per-category destination dirs
    for cat in LOCATION_MAPPING:
        (output_dir / cat).mkdir(exist_ok=True)

    # Walk raw_dir recursively — each leaf directory IS the Places365 category
    print(f"\n  Organising images from {raw_dir} ...")
    for img_path in tqdm(list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png")),
                         desc="Mapping images"):
        # Derive category from parent folder name
        places_cat = img_path.parent.name
        loc_cat = _map_places365_label(places_cat)

        if loc_cat is None:
            counters["unmapped"] += 1
            continue

        # Respect max_per_class cap
        if max_per_class and counters[loc_cat] >= max_per_class:
            skipped += 1
            continue

        dest = output_dir / loc_cat / img_path.name
        # If a name collision, prefix with parent dir
        if dest.exists():
            dest = output_dir / loc_cat / f"{img_path.parent.name}_{img_path.name}"

        shutil.copy2(img_path, dest)
        counters[loc_cat] += 1

    # Save stats
    stats = {
        "category_counts": counters,
        "total_organised": sum(v for k, v in counters.items() if k != "unmapped"),
        "unmapped": counters["unmapped"],
        "skipped_over_cap": skipped,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "organisation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n  Organisation summary:")
    for cat, cnt in counters.items():
        print(f"    {cat:<12}: {cnt:,} images")
    print(f"    skipped (cap): {skipped:,}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download & organise Places365 dataset")
    parser.add_argument("--output", default="data/raw",
                        help="Directory to store raw downloads (default: data/raw)")
    parser.add_argument("--organised", default="data/processed_raw",
                        help="Directory for category-organised images (default: data/processed_raw)")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap images per location category after organisation")
    parser.add_argument("--demo", action="store_true",
                        help="Create a tiny demo dataset from STL10 (no Kaggle account needed)")
    parser.add_argument("--no-organise", action="store_true",
                        help="Skip the category-organisation step after download")
    args = parser.parse_args()

    raw_dir = Path(args.output)
    organised_dir = Path(args.organised)

    if args.demo:
        print("\n[DEMO MODE] Creating small pipeline smoke-test dataset ...")
        create_demo_dataset(organised_dir, samples_per_class=args.max_per_class or 50)
        print("\nDone. Use --data", organised_dir, "when running run_training.py")
        return

    # ── Download via Kaggle ────────────────────────────────────────────────────
    print("\n[1/2] Downloading Places365 via Kaggle API ...")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\nERROR: Kaggle API key not found.")
        print("  1. Go to https://www.kaggle.com/settings → API → Create New Token")
        print(f"  2. Copy kaggle.json to: {kaggle_json}")
        print("\nAlternatively, create a demo dataset for smoke-testing:")
        print("  python scripts/download_dataset.py --demo")
        sys.exit(1)

    download_via_kaggle(KAGGLE_DATASET, raw_dir)

    if args.no_organise:
        print("\nDone (--no-organise set).")
        return

    # ── Organise into 5 categories ────────────────────────────────────────────
    print("\n[2/2] Organising into location categories ...")
    organise_images(
        raw_dir=raw_dir,
        output_dir=organised_dir,
        categories_file=raw_dir / "categories_places365.txt",
        max_per_class=args.max_per_class,
    )

    print("\n✓ Dataset ready.")
    print(f"  Organised images: {organised_dir}")
    print("\nNext step: run the dataset splitter to create train/val/test splits:")
    print("  python scripts/split_dataset.py --data", organised_dir)


if __name__ == "__main__":
    main()
