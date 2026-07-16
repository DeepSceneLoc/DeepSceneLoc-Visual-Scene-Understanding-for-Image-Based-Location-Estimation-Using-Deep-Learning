# Dataset Setup Guide

## Dataset Information

**Dataset:** Places2 MIT Dataset  
**Kaggle Link:** https://www.kaggle.com/datasets/nickj26/places2-mit-dataset  
**Original:** 10M+ images, 365 scene categories  
**Our Use:** Filtered to 5 broad location categories

## Our 5 Categories

| Category | Description | Places2 Examples |
|----------|-------------|------------------|
| **Coastal** | Beaches, harbors, seaside | beach, harbor, coast, pier, lighthouse, boardwalk |
| **Forest** | Dense vegetation, woodland | forest, jungle, bamboo_forest, rainforest, grove |
| **Mountain** | Peaks, highlands, valleys | mountain, mountain_snowy, valley, canyon, cliff |
| **Rural** | Farmland, countryside | farmland, pasture, field, vineyard, barn, village |
| **Urban** | Cities, streets, buildings | street, city, downtown, skyscraper, plaza, highway |

## Setup Options

### Option 1: Use Pre-filtered Dataset (Recommended if Available)

If you already have a filtered dataset with only these 5 categories:

```python
# In Kaggle notebook
DATASET_DIR = "/kaggle/input/your-filtered-dataset"
DATA = DATASET_DIR

# Verify structure
!ls {DATA}/train/
# Should show: Coastal/ Forest/ Mountain/ Rural/ Urban/
```

### Option 2: Filter from Full Places2

If using the full Places2 dataset, you need to map and filter:

```python
# In Kaggle notebook - Cell before training

# Add this as input:
# https://www.kaggle.com/datasets/nickj26/places2-mit-dataset

PLACES2_DIR = "/kaggle/input/places2-mit-dataset"

# Map Places2 categories to our 5
CATEGORY_MAPPING = {
    'Coastal': [
        'beach', 'beach_house', 'boardwalk', 'coast', 'harbor', 
        'lighthouse', 'pier', 'promenade', 'sandbar', 'wave'
    ],
    'Forest': [
        'forest_path', 'forest_road', 'bamboo_forest', 'rainforest',
        'tree_farm', 'woodland', 'grove', 'jungle'
    ],
    'Mountain': [
        'mountain', 'mountain_path', 'mountain_snowy', 'valley',
        'canyon', 'cliff', 'butte', 'badlands', 'volcano'
    ],
    'Rural': [
        'barn', 'farm', 'farmland', 'field_cultivated', 'field_wild',
        'hayfield', 'orchard', 'pasture', 'rice_paddy', 'vineyard',
        'village', 'cottage_garden', 'wheat_field'
    ],
    'Urban': [
        'street', 'crosswalk', 'downtown', 'highway', 'plaza',
        'shopping_mall', 'skyscraper', 'apartment_building',
        'office_building', 'parking_lot', 'residential_neighborhood'
    ]
}

# Filter and organize
import os
import shutil
from pathlib import Path

output_dir = "data/processed"
for split in ['train', 'val']:  # Places2 might use 'val' instead of 'test'
    for our_category, places_categories in CATEGORY_MAPPING.items():
        out_path = Path(output_dir) / split / our_category
        out_path.mkdir(parents=True, exist_ok=True)
        
        for place_cat in places_categories:
            src = Path(PLACES2_DIR) / split / place_cat
            if src.exists():
                print(f"Copying {place_cat} -> {our_category}")
                # Copy images (limit to avoid too much data)
                images = list(src.glob("*.jpg"))[:1000]  # Limit per category
                for img in images:
                    shutil.copy(img, out_path / f"{place_cat}_{img.name}")

print("✓ Dataset prepared!")
DATA = output_dir
```

### Option 3: Manual Download and Prepare

If you want to prepare locally first:

1. **Download Places2 subset** from Kaggle
2. **Run filtering script:**

```bash
# On your local machine
python scripts/filter_places2.py \
    --input path/to/places2 \
    --output data/processed \
    --categories Coastal,Forest,Mountain,Rural,Urban
```

3. **Upload to Kaggle Dataset:**
   - Zip the filtered data
   - Create new Kaggle dataset
   - Add to your notebook

## Expected Structure

After preparation, you should have:

```
data/processed/
├── train/
│   ├── Coastal/
│   │   └── *.jpg (beach images, harbor, etc.)
│   ├── Forest/
│   │   └── *.jpg (forest images, jungle, etc.)
│   ├── Mountain/
│   │   └── *.jpg (mountain images, peaks, etc.)
│   ├── Rural/
│   │   └── *.jpg (farm images, countryside, etc.)
│   └── Urban/
│       └── *.jpg (city images, streets, etc.)
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

## Data Split Ratios

Recommended:
- **Train:** 70-80% of data
- **Validation:** 10-15% of data
- **Test:** 10-15% of data

## Image Requirements

- **Format:** JPG or PNG
- **Size:** Any (will be resized to 224x224)
- **Aspect ratio:** Any (aspect-preserving resize used)
- **Color:** RGB (3 channels)

## Quick Verification

```python
# In Kaggle notebook
import os
from pathlib import Path

data_dir = "data/processed"
categories = ['Coastal', 'Forest', 'Mountain', 'Rural', 'Urban']

for split in ['train', 'val', 'test']:
    print(f"\n{split.upper()}:")
    for cat in categories:
        path = Path(data_dir) / split / cat
        if path.exists():
            count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png")))
            print(f"  {cat:10s}: {count:,} images")
        else:
            print(f"  {cat:10s}: NOT FOUND")
```

Expected output:
```
TRAIN:
  Coastal   : 5,000 images
  Forest    : 5,000 images
  Mountain  : 5,000 images
  Rural     : 5,000 images
  Urban     : 5,000 images

VAL:
  Coastal   : 1,000 images
  ...
```

## Kaggle Notebook Setup

### Step 1: Add Dataset Input

In your Kaggle notebook:
1. Click **"+ Add data"**
2. Search: **"nickj26/places2-mit-dataset"**
3. Click **"Add"**

### Step 2: Check Available Data

```python
!ls /kaggle/input/places2-mit-dataset/
```

### Step 3: Use in Training

```python
# If already filtered
DATA = "/kaggle/input/places2-mit-dataset"

# If needs filtering
# Run filtering code above first
DATA = "data/processed"

# Then train
!python scripts/training/run_training_efficientnet_b0.py \
    --data {DATA} \
    --batch 128 \
    --epochs 45 \
    --full-finetune \
    --swa
```

## Troubleshooting

### "No images found"

Check paths:
```python
!ls -la /kaggle/input/places2-mit-dataset/
!ls -la data/processed/train/Coastal/
```

### "Category not found"

Verify category names match exactly (case-sensitive):
- ✅ `Coastal` (capital C)
- ❌ `coastal` (lowercase)

### "Not enough images"

You might need more images per category. Aim for:
- **Minimum:** 1,000 images per category
- **Good:** 5,000 images per category
- **Ideal:** 10,000+ images per category

### "Memory error"

Reduce images per category:
```python
images = list(src.glob("*.jpg"))[:500]  # Use 500 instead of 1000
```

## Alternative: Create Your Own Dataset

If Places2 doesn't work, collect your own:

1. **Google Images:** Search and download
2. **Flickr:** Use Flickr API
3. **Unsplash:** Free high-quality images
4. **Manual collection:** Take your own photos

Tools:
- `google-images-download`
- `flickr-api`
- `unsplash-api`

Then organize into the 5-category structure.

---

**Dataset URL:** https://www.kaggle.com/datasets/nickj26/places2-mit-dataset

**Next:** Once dataset is ready, proceed with training in `TRAINING_ON_KAGGLE.md`
