# Dataset

## Overview

This dataset contains synchronized ultrasound wrist images and finger joint angle measurements.

## Files Included

✅ **CSV Files** (included in this repository):
- `train.csv` - 2,844 training samples
- `val.csv` - 355 validation samples
- `test.csv` - 356 test samples

Each CSV contains:
```
image_filename, MCP_Flex, PIP_Flex
```

## Files NOT Included (Download Required)

⚠️ **Ultrasound Images** (not included due to size ~238MB):
- 3,555 PNG images in `251002_wrist/` folder

### Download Links

**Option 1: Google Drive**
- [Download ultrasound images (238MB)](#) - Coming soon

**Option 2: Kaggle**
- [Kaggle Dataset](#) - Coming soon

### Installation

After downloading, extract to:
```
data/
└── 251002_wrist/
    ├── ultrasound_frame_dsc_20251002_153224_869.png
    ├── ultrasound_frame_dsc_20251002_153225_033.png
    └── ... (3,555 images total)
```

Verify:
```bash
ls data/251002_wrist/ | wc -l
# Should output: 3555
```

## Data Statistics

| Split | Samples | Percentage | MCP Range | PIP Range |
|-------|---------|------------|-----------|-----------|
| Train | 2,844 | 80% | 3.9° - 93.3° | 6.1° - 110.0° |
| Val | 355 | 10% | 6.2° - 91.5° | 7.5° - 110.0° |
| Test | 356 | 10% | 11.8° - 78.6° | 7.8° - 110.0° |

## Data Format

### Images
- **Format**: PNG (grayscale ultrasound)
- **Resolution**: Variable (resized to 224×224 during training)
- **Frame rate**: ~6 FPS (170ms intervals)

### Angles
- **MCP_Flex**: Metacarpophalangeal joint flexion angle (degrees)
- **PIP_Flex**: Proximal Interphalangeal joint flexion angle (degrees)
- **Precision**: 0.1° (rounded to 1 decimal place)

## Data Collection

- **Date**: October 2, 2025
- **Subject**: Single subject (right hand)
- **Duration**: ~10 minutes
- **Synchronization**: Timestamp-based matching (±6.5ms average error)

## Temporal Properties

- **Ultrasound**: ~6 FPS (170.5ms intervals)
- **Angles**: 60 FPS (16.67ms intervals, downsampled to match images)
- **Window**: 7 consecutive frames used for prediction (~210ms temporal context)

## Usage

```python
import pandas as pd
from PIL import Image

# Load CSV
df = pd.read_csv('data/train.csv')

# Load image
img_path = f"data/251002_wrist/{df.iloc[0]['image_filename']}"
img = Image.open(img_path)

# Get angles
mcp_angle = df.iloc[0]['MCP_Flex']
pip_angle = df.iloc[0]['PIP_Flex']
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{ultrasound-finger-2025,
  author = {Your Name},
  title = {Ultrasound Wrist Images with Finger Angles},
  year = {2025},
  publisher = {GitHub}
}
```
