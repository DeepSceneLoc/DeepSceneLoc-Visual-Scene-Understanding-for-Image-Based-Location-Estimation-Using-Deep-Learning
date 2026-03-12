"""
Transforms Module for DeepSceneLoc
Defines all image transformation and augmentation strategies

Author: Anuj Kondawar (Preprocessing & Pipeline Lead)
Week 4 Deliverable - Transforms Module
"""

import torch
import torchvision.transforms as T
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# ImageNet normalization constants
# ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Default target resolution for all models
DEFAULT_IMAGE_SIZE = 224


# ─────────────────────────────────────────────────────────────
# Base transform builders
# ─────────────────────────────────────────────────────────────

def get_train_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
    use_random_crop: bool = True,
    use_color_jitter: bool = True,
    use_random_flip: bool = True,
    use_random_rotation: bool = True,
    use_random_affine: bool = True,
) -> T.Compose:
    """
    Build the training transform pipeline with data augmentation.

    Augmentations applied (in order):
        1. Resize slightly larger than target, then RandomCrop to target size
        2. RandomHorizontalFlip
        3. RandomRotation (±15°)
        4. ColorJitter (brightness, contrast, saturation, hue)
        5. RandomAffine (translate ±10%, scale 90–110%)
        6. ToTensor
        7. Normalize (ImageNet statistics)

    Args:
        image_size: Target spatial resolution (H × W both set to this).
        mean: Channel means for normalization.
        std: Channel standard deviations for normalization.
        use_random_crop: Include RandomCrop augmentation.
        use_color_jitter: Include ColorJitter augmentation.
        use_random_flip: Include RandomHorizontalFlip.
        use_random_rotation: Include RandomRotation.
        use_random_affine: Include RandomAffine.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    transforms: List = []

    # Step 1 – resize (with small padding) + random crop
    if use_random_crop:
        scale_size = int(image_size * 1.15)          # e.g. 257 for size=224
        transforms += [
            T.Resize((scale_size, scale_size)),
            T.RandomCrop(image_size),
        ]
    else:
        transforms.append(T.Resize((image_size, image_size)))

    # Step 2 – horizontal flip
    if use_random_flip:
        transforms.append(T.RandomHorizontalFlip(p=0.5))

    # Step 3 – random rotation
    if use_random_rotation:
        transforms.append(T.RandomRotation(degrees=15))

    # Step 4 – color jitter (brightness & contrast = "brightness/contrast adjustment")
    if use_color_jitter:
        transforms.append(
            T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1,
            )
        )

    # Step 5 – affine (translate + scale)
    if use_random_affine:
        transforms.append(
            T.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            )
        )

    # Step 6–7 – convert + normalize
    transforms += [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]

    return T.Compose(transforms)


def get_val_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> T.Compose:
    """
    Build the validation / test transform pipeline (no augmentation).

    Args:
        image_size: Target spatial resolution.
        mean: Channel means for normalization.
        std: Channel standard deviations for normalization.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_test_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> T.Compose:
    """
    Alias for :func:`get_val_transforms`.  Test and validation preprocessing
    are identical (deterministic resize → tensor → normalize).
    """
    return get_val_transforms(image_size=image_size, mean=mean, std=std)


# ─────────────────────────────────────────────────────────────
# Minimal transforms (no augmentation, used for quick checks)
# ─────────────────────────────────────────────────────────────

def get_minimal_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> T.Compose:
    """
    Bare-minimum pipeline: resize → tensor → normalize.
    Useful for preprocessing sanity checks and benchmarking.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ─────────────────────────────────────────────────────────────
# Factory: get transforms by split name
# ─────────────────────────────────────────────────────────────

def get_transforms(
    split: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> T.Compose:
    """
    Return the correct transform pipeline for a dataset split.

    Args:
        split: One of ``'train'``, ``'val'``, ``'test'``.
        image_size: Target spatial resolution.
        mean: Normalization means.
        std: Normalization standard deviations.

    Returns:
        Appropriate ``Compose`` pipeline.

    Raises:
        ValueError: If ``split`` is not recognised.
    """
    split = split.lower()
    if split == 'train':
        return get_train_transforms(image_size=image_size, mean=mean, std=std)
    elif split in ('val', 'validation'):
        return get_val_transforms(image_size=image_size, mean=mean, std=std)
    elif split == 'test':
        return get_test_transforms(image_size=image_size, mean=mean, std=std)
    else:
        raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val', or 'test'.")


# ─────────────────────────────────────────────────────────────
# Inverse transform — for visualisation
# ─────────────────────────────────────────────────────────────

def denormalize(
    tensor: torch.Tensor,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> torch.Tensor:
    """
    Reverse the ImageNet normalization so a tensor can be displayed.

    Supports both batched (N, C, H, W) and single (C, H, W) tensors.

    Args:
        tensor: Normalized image tensor.
        mean: Channel means used during normalization.
        std: Channel standard deviations used during normalization.

    Returns:
        Denormalized tensor clamped to [0, 1].
    """
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t  = torch.tensor(std,  dtype=tensor.dtype, device=tensor.device)

    if tensor.ndim == 4:           # batched: (N, C, H, W)
        mean_t = mean_t[None, :, None, None]
        std_t  = std_t [None, :, None, None]
    elif tensor.ndim == 3:         # single:  (C, H, W)
        mean_t = mean_t[:, None, None]
        std_t  = std_t [:, None, None]

    return (tensor * std_t + mean_t).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# Augmentation policy descriptions (for documentation)
# ─────────────────────────────────────────────────────────────

AUGMENTATION_POLICY = {
    "resize_and_crop": {
        "description": "Resize to 257×257, then RandomCrop 224×224",
        "purpose": "Introduces spatial variation without distorting class features",
    },
    "horizontal_flip": {
        "p": 0.5,
        "purpose": "Simulates mirror-image real-world scenes",
    },
    "random_rotation": {
        "degrees": 15,
        "purpose": "Handles camera tilt and non-level horizon shots",
    },
    "color_jitter": {
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.2,
        "hue": 0.1,
        "purpose": "Simulates lighting and weather variation across locations",
    },
    "random_affine": {
        "translate": (0.1, 0.1),
        "scale": (0.9, 1.1),
        "purpose": "Handles slight perspective shifts and varying zoom levels",
    },
    "normalization": {
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "purpose": "Align input distribution with ImageNet pre-training statistics",
    },
}


def print_augmentation_policy():
    """Print a human-readable summary of all augmentation steps."""
    print("=" * 60)
    print("DeepSceneLoc — Augmentation Policy (Training)")
    print("=" * 60)
    for name, info in AUGMENTATION_POLICY.items():
        print(f"\n  {name}:")
        for k, v in info.items():
            print(f"    {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    print_augmentation_policy()

    # Quick smoke-test
    from PIL import Image
    import numpy as np

    dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))

    for split in ("train", "val", "test"):
        tfm   = get_transforms(split)
        t     = tfm(dummy_img)
        inv_t = denormalize(t)
        print(f"[{split}] output shape: {t.shape}, denorm range: [{inv_t.min():.3f}, {inv_t.max():.3f}]")

    print("\nAll transforms verified successfully.")
