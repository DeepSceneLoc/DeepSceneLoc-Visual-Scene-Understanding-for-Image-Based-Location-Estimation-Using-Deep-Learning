"""
Reorganize Raw Places365 Dataset
Extracts images from the alphabetized Places365 hierarchical folder structure
and pools them into the 5 DeepSceneLoc super-categories.

Usage:
    python scripts/reorganize_places365.py --data /kaggle/input/.../train_256_places365standard/data_256 --out data/processed_raw
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prepare_dataset import Places365Mapper

def main():
    p = argparse.ArgumentParser(description="Reorganize hierarchical Places365 to flat super-categories")
    p.add_argument("--data", required=True, help="Path to data_256 folder (contains a/, b/, c/...)")
    p.add_argument("--out", default="data/processed_raw", help="Output directory")
    p.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = p.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    mapper = Places365Mapper()
    super_categories = mapper.get_mapped_categories()

    # Create output directories for the 5 super-categories
    for scat in super_categories:
        (out_dir / scat).mkdir(parents=True, exist_ok=True)

    print(f"Scanning {data_dir} for Places365 categories...")
    
    # The structure is data_256/<letter>/<category>/<images>
    # e.g., data_256/a/airport/image_001.jpg
    
    stats = {scat: 0 for scat in super_categories}
    unmapped_count = 0
    total_images_processed = 0

    letter_folders = [d for d in data_dir.iterdir() if d.is_dir() and len(d.name) == 1]
    
    for letter_dir in tqdm(letter_folders, desc="Processing alphabet folders"):
        category_folders = [d for d in letter_dir.iterdir() if d.is_dir()]
        
        for cat_dir in category_folders:
            places_category = cat_dir.name
            super_cat = mapper.map_category(places_category)
            
            if super_cat is None:
                # This Places365 category is not needed in our 5-class subset
                unmapped_count += sum(1 for _ in cat_dir.glob("*.*"))
                continue
                
            images = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                images.extend(cat_dir.glob(ext))
                
            for img_path in images:
                # Prefix filename with category to prevent collisions from different folders
                new_filename = f"{places_category}_{img_path.name}"
                dest = out_dir / super_cat / new_filename
                
                if not dest.exists():
                    if args.copy:
                        shutil.copy2(img_path, dest)
                    else:
                        try:
                            os.symlink(img_path, dest)
                        except OSError:
                            shutil.copy2(img_path, dest)
                            
                stats[super_cat] += 1
                total_images_processed += 1

    print("\nReorganization Complete!")
    print(f"Output directory: {out_dir}")
    print("\nImages per super-category:")
    for scat, count in stats.items():
        print(f"  {scat}: {count} images")
    print(f"\nIgnored (unmapped) images: {unmapped_count}")
    print(f"Total mapped images: {total_images_processed}")

if __name__ == "__main__":
    main()
