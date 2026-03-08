"""
Dataset Preparation Module
Handles downloading, organizing, and splitting the Places365 dataset
Week 3 - Dataset Preparation (Aditi Sah - Lead)
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm


class Places365Mapper:
    """Maps Places365 categories to 5 location categories"""
    
    LOCATION_MAPPING = {
        'Urban': [
            'street', 'downtown', 'building_facade', 'skyscraper',
            'plaza', 'shopping_mall', 'parking_lot', 'crosswalk',
            'highway', 'alley', 'bridge', 'subway_station',
            'apartment_building', 'office_building', 'hotel_outdoor',
            'restaurant', 'cafe', 'bus_station', 'parking_garage'
        ],
        'Rural': [
            'field', 'farm', 'barn', 'pasture', 'cornfield',
            'wheat_field', 'rice_paddy', 'vegetable_garden',
            'village', 'farmhouse', 'windmill', 'vineyard',
            'orchard', 'herb_garden', 'cottage_garden', 'scarecrow'
        ],
        'Coastal': [
            'beach', 'coast', 'ocean', 'sea', 'harbor', 'pier',
            'lighthouse', 'marina', 'dock', 'boardwalk',
            'seashore', 'sandbar', 'wave', 'cliff_coastal',
            'bay', 'cove', 'peninsula', 'beach_house'
        ],
        'Mountain': [
            'mountain', 'mountain_snowy', 'mountain_path', 'canyon',
            'valley', 'cliff', 'rock_arch', 'butte', 'volcano',
            'summit', 'ridge', 'hill', 'peak', 'ski_slope',
            'ski_resort', 'alpine', 'highland', 'plateau'
        ],
        'Forest': [
            'forest', 'rainforest', 'bamboo_forest', 'tree_farm',
            'woodland', 'jungle', 'grove', 'palm_grove',
            'forest_path', 'forest_road', 'sequoia', 'swamp',
            'bog', 'marsh', 'wetland', 'creek', 'river_forest'
        ]
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize mapper with configuration"""
        self.mapping = self.LOCATION_MAPPING
        self.reverse_mapping = self._create_reverse_mapping()
        
    def _create_reverse_mapping(self) -> Dict[str, str]:
        """Create reverse mapping from Places365 to location category"""
        reverse_map = {}
        for location, places_categories in self.mapping.items():
            for place in places_categories:
                reverse_map[place] = location
        return reverse_map
    
    def map_category(self, places_category: str) -> str:
        """Map a Places365 category to location category"""
        # Clean category name
        clean_name = places_category.strip().lower().replace('/', '_')
        
        # Check direct match
        if clean_name in self.reverse_mapping:
            return self.reverse_mapping[clean_name]
        
        # Check partial match
        for place, location in self.reverse_mapping.items():
            if place in clean_name or clean_name in place:
                return location
        
        # Default: return None for unmapped categories
        return None
    
    def get_mapped_categories(self) -> List[str]:
        """Get list of 5 location categories"""
        return list(self.mapping.keys())
    
    def get_category_stats(self, dataset_path: str) -> Dict:
        """Get statistics about category distribution"""
        stats = {cat: 0 for cat in self.mapping.keys()}
        stats['unmapped'] = 0
        
        # Scan dataset (placeholder - actual implementation would scan files)
        # This is a simulation
        return stats
    
    def save_mapping(self, output_path: str):
        """Save mapping to JSON file"""
        with open(output_path, 'w') as f:
            json.dump({
                'location_mapping': self.mapping,
                'reverse_mapping': self.reverse_mapping
            }, f, indent=2)
        print(f"Mapping saved to {output_path}")


class DatasetSplitter:
    """Splits dataset into train/val/test sets"""
    
    def __init__(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Initialize splitter with ratios
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        np.random.seed(seed)
    
    def split_dataset(self, data_dir: str, output_dir: str, copy_files=False):
        """
        Split dataset into train/val/test
        
        Args:
            data_dir: Source directory with category folders
            output_dir: Destination directory
            copy_files: If True, copy files; if False, create symlinks
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (output_path / split).mkdir(parents=True, exist_ok=True)
        
        # Process each category
        categories = [d for d in data_path.iterdir() if d.is_dir()]
        
        for category in tqdm(categories, desc="Splitting categories"):
            category_name = category.name
            
            # Get all images in category
            images = list(category.glob('*.jpg')) + list(category.glob('*.png'))
            
            # Shuffle
            np.random.shuffle(images)
            
            # Calculate split indices
            n_images = len(images)
            n_train = int(n_images * self.train_ratio)
            n_val = int(n_images * self.val_ratio)
            
            # Split
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Create category directories in splits
            for split in ['train', 'val', 'test']:
                (output_path / split / category_name).mkdir(exist_ok=True)
            
            # Copy or link files
            self._copy_images(train_images, output_path / 'train' / category_name, copy_files)
            self._copy_images(val_images, output_path / 'val' / category_name, copy_files)
            self._copy_images(test_images, output_path / 'test' / category_name, copy_files)
        
        # Save split statistics
        self._save_split_stats(output_path)
        
        print(f"\nDataset split complete!")
        print(f"Train: {self.train_ratio:.1%}, Val: {self.val_ratio:.1%}, Test: {self.test_ratio:.1%}")
    
    def _copy_images(self, images: List[Path], dest_dir: Path, copy: bool):
        """Copy or symlink images to destination"""
        for img in images:
            dest = dest_dir / img.name
            if copy:
                shutil.copy2(img, dest)
            else:
                # Create relative symlink if supported
                try:
                    os.symlink(img, dest)
                except OSError:
                    # Fallback to copy if symlink not supported
                    shutil.copy2(img, dest)
    
    def _save_split_stats(self, output_path: Path):
        """Save statistics about the split"""
        stats = {}
        
        for split in ['train', 'val', 'test']:
            split_path = output_path / split
            categories = [d for d in split_path.iterdir() if d.is_dir()]
            
            split_stats = {}
            total = 0
            
            for category in categories:
                n_images = len(list(category.glob('*.jpg'))) + len(list(category.glob('*.png')))
                split_stats[category.name] = n_images
                total += n_images
            
            split_stats['total'] = total
            stats[split] = split_stats
        
        # Save to JSON
        with open(output_path / 'split_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nSplit statistics saved to {output_path / 'split_statistics.json'}")


def verify_dataset(data_dir: str) -> Dict:
    """
    Verify dataset integrity and collect statistics
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Dictionary with verification results
    """
    data_path = Path(data_dir)
    results = {
        'total_images': 0,
        'categories': {},
        'corrupted_images': [],
        'valid': True
    }
    
    categories = [d for d in data_path.iterdir() if d.is_dir()]
    
    for category in tqdm(categories, desc="Verifying dataset"):
        category_name = category.name
        images = list(category.glob('*.jpg')) + list(category.glob('*.png'))
        
        valid_images = 0
        for img_path in images:
            try:
                # Try to open image
                with Image.open(img_path) as img:
                    img.verify()
                valid_images += 1
            except Exception as e:
                results['corrupted_images'].append(str(img_path))
                results['valid'] = False
        
        results['categories'][category_name] = {
            'total': len(images),
            'valid': valid_images,
            'corrupted': len(images) - valid_images
        }
        results['total_images'] += valid_images
    
    return results


if __name__ == "__main__":
    """Example usage"""
    
    print("="*60)
    print("DeepSceneLoc - Dataset Preparation")
    print("Week 3 - Aditi Sah (Lead)")
    print("="*60)
    
    # Initialize mapper
    print("\n1. Creating category mapper...")
    mapper = Places365Mapper()
    mapper.save_mapping("data/category_mapping.json")
    print(f"   Mapped to {len(mapper.get_mapped_categories())} categories:")
    for cat in mapper.get_mapped_categories():
        print(f"   - {cat}: {len(mapper.mapping[cat])} Places365 categories")
    
    # Example: Split dataset
    print("\n2. Dataset splitting (example)...")
    print("   Note: Actual dataset should be downloaded separately")
    print("   Command: python -m src.data.prepare_dataset --split")
    
    # Splitter example
    splitter = DatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    print(f"   Split ratios: Train={splitter.train_ratio:.0%}, Val={splitter.val_ratio:.0%}, Test={splitter.test_ratio:.0%}")
    
    print("\n[OK] Dataset preparation module ready!")
    print("  Run with: python -m src.data.prepare_dataset")
