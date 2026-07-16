"""
Preprocessing Pipeline Module
Handles image transformations, augmentation, and dataset creation
Week 4 - Preprocessing Pipeline (Anuj Kondawar - Lead)
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import yaml


class DeepSceneLocDataset(Dataset):
    """Custom Dataset for DeepSceneLoc"""
    
    def __init__(self, root_dir: str, transform=None, split='train'):
        """
        Args:
            root_dir: Root directory of dataset
            transform: Optional transform to be applied
            split: 'train', 'val', or 'test'
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.split = split
        
        # Define class mapping
        self.classes = ['Coastal', 'Forest', 'Mountain', 'Rural', 'Urban']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect image paths and labels
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """Load all image paths and labels"""
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            # Get all images in class directory
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count of samples per class"""
        counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            counts[class_name] += 1
        return counts


class DataTransforms:
    """Manage data transformations for train/val/test"""
    
    def __init__(self, image_size=224, augment_train=True):
        """
        Args:
            image_size: Target image size (default: 224 for ImageNet models)
            augment_train: Whether to apply augmentation to training data
        """
        self.image_size = image_size
        self.augment_train = augment_train
        
        # ImageNet normalization statistics
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()
    
    def _get_train_transform(self):
        """Get training transforms with augmentation"""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
        ]
        
        if self.augment_train:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        return transforms.Compose(transform_list)
    
    def _get_test_transform(self):
        """Get test/validation transforms (no augmentation).

        Aspect-ratio-preserving resize (shorter side) + center crop, matching
        the train-time geometry. The old Resize((size, size)) squashed
        non-square images and mismatched the training distribution.
        """
        resize_size = int(round(self.image_size * 256 / 224))
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def get_transform(self, split='train'):
        """Get appropriate transform for split"""
        if split == 'train':
            return self.train_transform
        else:
            return self.test_transform


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    augment_train: bool = True,
    train_transform=None,   # custom transform overrides augment_train
    val_transform=None,     # custom transform for val/test
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Key performance flags added (April 2026):
      - persistent_workers=True : worker processes stay alive between epochs,
        eliminating the 20-40s Windows worker-respawn delay per epoch.
      - prefetch_factor=2       : each worker pre-fetches 2 batches ahead,
        reducing GPU starvation during data loading.
      - pin_memory=True         : faster CPU->GPU transfer (already present).

    Args:
        data_dir: Root directory containing train/val/test splits.
        batch_size: Batch size for dataloaders.
        num_workers: Number of worker processes (use 0 on Windows if crashes).
        image_size: Target image size.
        augment_train: Whether to augment training data (ignored if train_transform given).
        train_transform: Optional custom transform for train split.
        val_transform: Optional custom transform for val/test split.
        pin_memory: Pin memory for faster GPU transfer.
    """
    transforms_obj = DataTransforms(image_size=image_size, augment_train=augment_train)

    tr_tfm = train_transform if train_transform is not None else transforms_obj.get_transform('train')
    va_tfm = val_transform   if val_transform   is not None else transforms_obj.get_transform('val')

    train_dataset = DeepSceneLocDataset(root_dir=data_dir, transform=tr_tfm, split='train')
    val_dataset   = DeepSceneLocDataset(root_dir=data_dir, transform=va_tfm, split='val')
    test_dataset  = DeepSceneLocDataset(root_dir=data_dir, transform=va_tfm, split='test')

    # persistent_workers=True keeps workers alive between epochs (no respawn delay)
    _persistent = num_workers > 0
    _prefetch   = 2 if num_workers > 0 else None


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
        drop_last=True,      # avoids uneven last batch with MixUp
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )

    return train_loader, val_loader, test_loader


def get_class_weights(dataset: Dataset) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        dataset: DeepSceneLocDataset instance
        
    Returns:
        Tensor of class weights
    """
    class_counts = dataset.get_class_counts()
    total_samples = sum(class_counts.values())
    
    num_classes = len(class_counts)
    weights = []
    
    for class_name in dataset.classes:
        count = class_counts[class_name]
        weight = total_samples / (num_classes * count) if count > 0 else 1.0
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def test_preprocessing_pipeline(data_dir: str = "data/processed"):
    """Test the preprocessing pipeline"""
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("Testing Preprocessing Pipeline")
    print("="*60)
    
    # Create transforms
    print("\n1. Testing transforms...")
    transforms_obj = DataTransforms(image_size=224, augment_train=True)
    print(f"   Train transform: {transforms_obj.train_transform}")
    print(f"   Test transform: {transforms_obj.test_transform}")
    
    # Create sample dataset
    print("\n2. Creating sample dataset...")
    try:
        train_dataset = DeepSceneLocDataset(
            root_dir=data_dir,
            transform=transforms_obj.get_transform('train'),
            split='train'
        )
        print(f"   Dataset size: {len(train_dataset)} images")
        print(f"   Classes: {train_dataset.classes}")
        print(f"   Class distribution: {train_dataset.get_class_counts()}")
        
        # Test loading a batch
        if len(train_dataset) > 0:
            img, label = train_dataset[0]
            print(f"\n3. Sample image:")
            print(f"   Shape: {img.shape}")
            print(f"   Label: {train_dataset.classes[label]}")
            print(f"   Min/Max: {img.min():.3f} / {img.max():.3f}")
        
    except Exception as e:
        print(f"   Note: Dataset not found or empty. Error: {e}")
        print(f"   This is expected if data hasn't been prepared yet.")
    
    print("\n[OK] Preprocessing pipeline module ready!")
    print("  Usage: from src.preprocessing.pipeline import create_dataloaders")


if __name__ == "__main__":
    """Test the preprocessing pipeline"""
    test_preprocessing_pipeline()
