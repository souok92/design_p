# 17_channel_stacked_2d: Channel-Stacked 2D CNN for Finger Angle Prediction

## Overview

**Novel approach**: Treats temporal frames as input channels for a 2D CNN.

- **Key innovation**: 16 grayscale frames → 16-channel input
- **Architecture**: ResNet18 (2D CNN) with modified first layer
- **Maximum efficiency**: 2D convolutions are faster than 3D
- **Trade-off**: Speed vs explicit temporal modeling

## Key Innovation

| Traditional 3D CNN | **Channel-Stacked 2D CNN** |
|-------------------|----------------------------|
| (B, 1, **T=16**, H, W) | (B, **C=16**, H, W) |
| 3D Conv(1, 64, (3,7,7)) | **2D Conv(16, 64, (7,7))** |
| Explicit temporal modeling | **Implicit temporal modeling** |
| Slower (3D convs) | **Faster (2D convs)** |
| Batch size: 12 | **Batch size: 16** |

**Core Insight**: Time is encoded as **channels** instead of a separate dimension!

## Architecture

```
Input: (Batch, Time=16, Height=224, Width=224) - Grayscale sequence
    ↓ Treat T as channels
(Batch, Channels=16, Height=224, Width=224) - 16-channel "image"
    ↓ Modified ResNet18
    │  - First Conv2d: 16 channels → 64 features
    │  - Weights initialized from averaged ImageNet RGB weights
    │  - Standard 2D convolutions (spatial patterns across channels)
    ↓
Feature vector: (Batch, 512)
    ↓ Regression Head
    │  - 512 → 256 → 128 → 8
    │  - ReLU activations + Dropout
    ↓
Output: (Batch, 8) - Finger angles
```

## Comparison with Other Models

| Aspect | 16_gray3d (3D CNN) | **17_channel_stacked_2d (2D CNN)** |
|--------|--------------------|------------------------------------|
| Input | (B, 1, 16, 224, 224) | **(B, 16, 224, 224)** |
| Architecture | R3D-18 (3D ResNet) | **ResNet18 (2D)** |
| Convolution type | 3D Conv | **2D Conv** |
| Temporal modeling | Explicit (3D filters) | **Implicit (channels)** |
| Batch size | 12 | **16 (33% higher)** |
| Training speed | Baseline | **2-3x faster** |
| Memory/batch | ~0.15 GB | **~0.13 GB** |
| Expected MAE | Similar to RGB | **Slightly higher (+5-10%)** |

## Advantages

1. **Maximum speed**: 2D convolutions are much faster than 3D
2. **Higher batch sizes**: Can train with batch_size=16
3. **Simplicity**: Standard 2D CNN architecture
4. **Flexibility**: Easy to swap backbones (ResNet34, EfficientNet, MobileNet)
5. **Transfer learning**: ImageNet pretrained weights

## Trade-offs

1. **No explicit temporal modeling**: Time is implicit in channel patterns
2. **May miss fine temporal dynamics**: 3D CNN explicitly models motion
3. **Unconventional approach**: Time-as-channels is not standard

## Training

### Quick Start

1. **Modify parameters** at the top of [train.py](train.py):
   ```python
   PERSON = "tk"  # or "sj"
   BATCH_SIZE = 16  # Higher than 3D CNN
   NUM_FRAMES = 16  # Temporal frames = channels
   NUM_EPOCHS = 100
   LEARNING_RATE = 0.0001
   # ... etc
   ```

2. **Run training**:
   ```bash
   cd 17_channel_stacked_2d
   python train.py
   ```

3. **Monitor with TensorBoard**:
   ```bash
   tensorboard --logdir=logs/tk
   ```

### Key Training Parameters

- **Batch size**: 16 (higher than 3D CNN due to efficiency)
- **Sequence length**: 16 frames (treated as 16 channels)
- **Image size**: 224×224
- **Optimizer**: AdamW
- **Learning rate**: 0.0001
- **Scheduler**: ReduceLROnPlateau
- **Loss**: MAE (Mean Absolute Error)
- **Early stopping**: 20 epochs patience (more than 3D CNN)

### Why Different Loss Function?

- **3D CNN**: SmoothL1 (robust to outliers, good for explicit modeling)
- **2D CNN**: MAE (simpler, interpretable, works well for implicit modeling)

You can change this in the training script if needed.

## Evaluation

1. **Update checkpoint path** in [evaluate.py](evaluate.py):
   ```python
   CHECKPOINT_PATH = "./checkpoints/tk/tk_channel2d_YYYYMMDD_HHMMSS/best_model.pth"
   ```

2. **Run evaluation**:
   ```bash
   python evaluate.py
   ```

3. **Output files** (in checkpoint directory):
   - `metrics.txt` - Per-angle MAE, RMSE, MSE
   - `predictions.npz` - Predictions and ground truth
   - `scatter_plots.png` - Prediction vs ground truth
   - `error_distribution.png` - Error histograms
   - `time_series.png` - Time series comparison

## Expected Performance

Based on hypothesis:
- **MAE**: 11-13° (5-10% higher than 3D CNN)
- **Training time**: **2-3x faster** than 16_gray3d
- **Inference time**: **3-5x faster** than 3D CNN
- **Memory usage**: Similar or slightly lower

**When to use**:
- Real-time inference is priority
- Limited GPU resources
- Need to train many models quickly
- Temporal dynamics are not critical

## Project Structure

```
17_channel_stacked_2d/
├── models/
│   ├── __init__.py
│   └── channel_stacked_2d.py  # ResNet18 with 16-channel input
├── data/
│   ├── __init__.py
│   └── dataset.py             # Same as 16_gray3d (grayscale loader)
├── checkpoints/
│   └── {person}/
│       └── {exp_name}/
│           ├── best_model.pth
│           └── last_model.pth
├── logs/
│   └── {person}/
│       └── {exp_name}/        # TensorBoard logs
├── results/
│   └── {person}/
│       └── {exp_name}/
│           └── experiment_info.json
├── train.py                   # Training script (hardcoded params)
├── evaluate.py                # Evaluation script (hardcoded params)
└── README.md                  # This file
```

## Technical Details

### How It Works

**Conceptually**:
```python
# Traditional approach:
# Process each frame separately, then combine with RNN/3D CNN
for frame in frames:
    feature = cnn_2d(frame)  # Per-frame features
combined = lstm(features)     # Temporal combination

# Channel-stacked approach:
# Stack frames as channels, single 2D CNN pass
stacked = stack_frames_as_channels(frames)  # (16, H, W)
output = cnn_2d(stacked)                    # Direct prediction
```

**Why this works**:
1. 2D CNN learns patterns across spatial + "temporal-channel" dimensions
2. Early conv layers can learn "motion filters" across channels
3. Similar to RGB channels encoding different color information

### Weight Initialization

```python
# Original ImageNet RGB weights: (64, 3, 7, 7)
old_weight = pretrained_conv.weight.data

# Average across RGB channels: (64, 1, 7, 7)
avg_weight = old_weight.mean(dim=1, keepdim=True)

# Repeat for all 16 temporal channels: (64, 16, 7, 7)
new_weight = avg_weight.repeat(1, 16, 1, 1) / 16
```

This allows transfer learning from ImageNet despite different input channels.

## Alternative Backbones

You can easily swap ResNet18 for other architectures:

```python
# In channel_stacked_2d.py, replace:
self.backbone = models.resnet18(pretrained=True)

# With:
self.backbone = models.resnet34(pretrained=True)       # More capacity
self.backbone = models.efficientnet_b0(pretrained=True) # More efficient
self.backbone = models.mobilenet_v3_small(pretrained=True) # Fastest
```

Just remember to adjust `self.feature_dim` accordingly!

## Use Cases

**Choose 17_channel_stacked_2d when**:
- ✅ Need fast training/inference
- ✅ Limited GPU memory
- ✅ Want to experiment quickly
- ✅ Temporal dynamics are not critical
- ✅ Real-time performance matters

**Choose 16_gray3d (3D CNN) when**:
- ✅ Accuracy is top priority
- ✅ Temporal dynamics are important
- ✅ Have sufficient GPU resources
- ✅ Training time is not critical

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` from 16 to 12 or 8
- Reduce `NUM_FRAMES` from 16 to 12 (but changes model input!)

### Slow Training
- This model should be FAST! Check GPU utilization
- Increase `NUM_WORKERS` (default: 4)

### Poor Performance
- Increase `NUM_EPOCHS`
- Try different `SCHEDULER` (e.g., CosineAnnealingLR)
- Consider using 16_gray3d (3D CNN) instead for better temporal modeling

## Citation

If you use this model, please cite:
- Original ultrasound dataset
- ImageNet for pretrained weights
- ResNet architecture

## License

Same as parent project.
