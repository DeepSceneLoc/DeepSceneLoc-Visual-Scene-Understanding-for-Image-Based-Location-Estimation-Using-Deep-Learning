"""
Reorganize Raw Places365 Dataset
Extracts images from the alphabetized Places365 hierarchical folder structure
and pools them into the 5 DeepSceneLoc super-categories.

Uses parallel file I/O (16 threads) for fast copying on network-mounted
storage like Kaggle's /kaggle/input/.

Usage:
    python scripts/reorganize_places365.py --data /kaggle/input/.../data_256 --out data/processed_raw --copy
    python scripts/reorganize_places365.py --data ... --out data/processed_raw --copy --max-per-category 15000
"""

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prepare_dataset import Places365Mapper

def _copy_file(src: Path, dest: Path):
    """Copy a single file (called from thread pool)."""
    try:
        shutil.copy2(src, dest)
        return True
    except Exception:
        return False

def main():
    p = argparse.ArgumentParser(description="Reorganize hierarchical Places365 to flat super-categories")
    p.add_argument("--data", required=True, help="Path to data_256 folder (contains a/, b/, c/...)")
    p.add_argument("--out", default="data/processed_raw", help="Output directory")
    p.add_argument("--copy", action="store_true", help="Copy instead of symlink (REQUIRED on Kaggle)")
    p.add_argument("--workers", type=int, default=16,
                   help="Parallel copy threads (default: 16)")
    p.add_argument("--max-per-category", type=int, default=None,
                   help="Cap images per super-category for balanced, faster training "
                        "(e.g. 15000). Without this, all images are used (can be 120K+ per cat).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling when --max-per-category is set")
    args = p.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    random.seed(args.seed)
    mapper = Places365Mapper()
    super_categories = mapper.get_mapped_categories()

    # Create output directories for the 5 super-categories
    for scat in super_categories:
        (out_dir / scat).mkdir(parents=True, exist_ok=True)

    print(f"Scanning {data_dir} for Places365 categories...")
    
    # Phase 1: Build file list grouped by super-category
    cat_files = defaultdict(list)  # super_cat -> list of (src, dest)
    unmapped_count = 0

    letter_folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and len(d.name) == 1])
    
    for letter_dir in tqdm(letter_folders, desc="Scanning categories"):
        category_folders = [d for d in letter_dir.iterdir() if d.is_dir()]
        
        for cat_dir in category_folders:
            places_category = cat_dir.name
            super_cat = mapper.map_category(places_category)
            
            if super_cat is None:
                unmapped_count += 1
                continue
                
            images = list(cat_dir.glob('*.jpg'))
                
            for img_path in images:
                new_filename = f"{places_category}_{img_path.name}"
                dest = out_dir / super_cat / new_filename
                
                if not dest.exists():
                    cat_files[super_cat].append((img_path, dest))

    # Phase 2: Cap per category if requested
    file_tasks = []
    stats = {}
    
    for scat in super_categories:
        files = cat_files[scat]
        total_available = len(files)
        
        if args.max_per_category and len(files) > args.max_per_category:
            random.shuffle(files)
            files = files[:args.max_per_category]
            print(f"  {scat}: {total_available} available -> capped to {len(files)}")
        else:
            print(f"  {scat}: {total_available} images")
        
        file_tasks.extend(files)
        stats[scat] = len(files)

    total_files = len(file_tasks)
    total_all = sum(len(cat_files[s]) for s in super_categories)
    
    if args.max_per_category:
        print(f"\nUsing {total_files}/{total_all} images "
              f"(capped at {args.max_per_category}/category)")
    print(f"\n{'Copying' if args.copy else 'Linking'} {total_files} files...")

    # Phase 3: Copy/symlink files
    if args.copy and total_files > 0:
        print(f"  Using {args.workers} parallel threads...")
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_copy_file, src, dst): (src, dst) 
                       for src, dst in file_tasks}
            with tqdm(total=total_files, desc="Copying files", unit="file") as pbar:
                for future in as_completed(futures):
                    done += 1
                    pbar.update(1)
        print(f"  Copied {done} files.")
    elif total_files > 0:
        for src, dst in tqdm(file_tasks, desc="Creating symlinks"):
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)

    print("\nReorganization Complete!")
    print(f"Output directory: {out_dir}")
    print("\nImages per super-category:")
    for scat, count in stats.items():
        print(f"  {scat}: {count} images")
    print(f"\nIgnored (unmapped) categories: {unmapped_count}")
    print(f"Total images: {sum(stats.values())}")

if __name__ == "__main__":
    main()
