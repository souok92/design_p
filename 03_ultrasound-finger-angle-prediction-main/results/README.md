# Model Weights

## Download

The trained model file (`best_model.pth`) is **134MB** and exceeds GitHub's file size limit.

**Download from GitHub Releases:**
- 📦 [best_model.pth](https://github.com/eekimsj/ultrasound-finger-angle-prediction/releases/download/v1.0.0/best_model.pth) (134MB)

## Model Info

- **Architecture**: CNN-TCN (ResNet-18 + TCN)
- **Size**: 134MB
- **Parameters**: ~11.3M
- **Performance**: 
  - Overall MAE: 5.62°
  - MCP MAE: 5.75°
  - PIP MAE: 5.50°

## Usage

```python
import torch
from src.model import CNN_TCN_Model

# Load model
model = CNN_TCN_Model(window_size=7)
checkpoint = torch.load('results/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Alternative: Train Your Own

You can train the model yourself using the Colab notebook:
- See: [notebooks/Ultrasound_Angle_Prediction_Colab.ipynb](../notebooks/Ultrasound_Angle_Prediction_Colab.ipynb)
- Training time: ~1-2 hours on GPU T4
