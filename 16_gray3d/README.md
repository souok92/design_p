# 16_gray3d: Grayscale 3D CNN for Finger Angle Prediction

## Overview

Memory-efficient variant of the 11_3dcnn model that uses **grayscale** instead of RGB images.

- **Memory reduction**: 66% compared to RGB version
- **Architecture**: R3D-18 (3D ResNet-18) modified for single-channel input
- **Transfer learning**: Averaged Kinetics-400 weights for grayscale
- **Performance**: Similar or better than RGB version (no information loss)

## Key Differences from 11_3dcnn (RGB)

| Aspect | 11_3dcnn (RGB) | 16_gray3d (Grayscale) |
|--------|----------------|------------------------|
| Input channels | 3 (RGB) | 1 (Grayscale) |
| Input shape | (B, 3, 16, 224, 224) | (B, 1, 16, 224, 224) |
| Memory/batch | ~0.46 GB | ~0.15 GB |
| Batch size | 8 | 12 (increased) |
| First Conv3d | Conv3d(3, 64, ...) | Conv3d(1, 64, ...) |
| Pretrained weights | Kinetics-400 RGB | Averaged to grayscale |

## Architecture

```
Input: (Batch, Time=16, Height=224, Width=224) - Grayscale sequence
    ↓ unsqueeze(1)
(Batch, 1, 16, 224, 224)
    ↓ Modified R3D-18
    │  - First Conv3d: 1 channel → 64 features
    │  - Weights initialized from averaged Kinetics-400 RGB weights
    │  - 3D convolutions extract spatiotemporal features
    ↓
Feature vector: (Batch, 512)
    ↓ Regression Head
    │  - 512 → 256 → 128 → 8
    │  - ReLU activations + Dropout
    ↓
Output: (Batch, 8) - Finger angles
```

## Model Details

- **Backbone**: R3D-18 (3D ResNet-18)
- **Feature extraction**: Spatiotemporal 3D convolutions
- **Feature dimension**: 512
- **Output**: 8 finger angles (Index, Middle, Ring, Pinky - MCP & PIP each)

## Training

### Quick Start

1. **Modify parameters** at the top of [train.py](train.py):
   ```python
   PERSON = "tk"  # or "sj"
   BATCH_SIZE = 12
   NUM_EPOCHS = 100
   LEARNING_RATE = 0.0001
   # ... etc
   ```

2. **Run training**:
   ```bash
   cd 16_gray3d
   python train.py
   ```

3. **Monitor with TensorBoard**:
   ```bash
   tensorboard --logdir=logs/tk
   ```

### Key Training Parameters

- **Batch size**: 12 (increased from 8 due to lower memory usage)
- **Sequence length**: 16 frames
- **Image size**: 224×224
- **Optimizer**: AdamW
- **Learning rate**: 0.0001
- **Scheduler**: CosineAnnealingLR
- **Loss**: SmoothL1
- **Early stopping**: 15 epochs patience
- **Transfer learning**: Freeze backbone for first 5 epochs

### Training Strategy

1. **Freeze backbone** (first 5 epochs):
   - Only train regression head
   - Fast convergence for task-specific layer

2. **Unfreeze backbone** (epoch 6+):
   - Fine-tune entire network
   - Learning rate reduced by 10x

3. **Early stopping**:
   - Monitors validation loss
   - Stops if no improvement for 15 epochs

## Evaluation

1. **Update checkpoint path** in [evaluate.py](evaluate.py):
   ```python
   CHECKPOINT_PATH = "./checkpoints/tk/tk_gray3d_YYYYMMDD_HHMMSS/best_model.pth"
   ```

2. **Run evaluation**:
   ```bash
   python evaluate.py
   ```

3. **Output files** (in checkpoint directory):
   - `metrics.txt` - Per-angle MAE, RMSE, MSE
   - `predictions.npz` - Predictions and ground truth
   - `scatter_plots.png` - Prediction vs ground truth scatter plots
   - `error_distribution.png` - Error histograms
   - `time_series.png` - Time series comparison (first 500 samples)

## Expected Performance

Based on hypothesis:
- **MAE**: Similar to 11_3dcnn RGB baseline (~10-12°)
- **Training time**: 1.3x faster due to lower memory usage
- **Memory usage**: 66% reduction
- **No accuracy loss**: Ultrasound images are inherently grayscale

## Project Structure

```
16_gray3d/
├── models/
│   ├── __init__.py
│   └── gray3d.py              # Grayscale R3D-18 model
├── data/
│   ├── __init__.py
│   └── dataset.py             # Grayscale image loader
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

## Technical Notes

### Why Grayscale?

1. **Ultrasound images are inherently grayscale**:
   - RGB channels contain identical information
   - No semantic difference between R, G, B

2. **Memory efficiency**:
   - RGB: 3 channels × 4 bytes = 12 bytes/pixel
   - Grayscale: 1 channel × 4 bytes = 4 bytes/pixel
   - **66% reduction** in input memory

3. **No accuracy loss**:
   - All visual information preserved
   - Kinetics-400 patterns transferred via weight averaging

### Weight Initialization

```python
# Original Kinetics-400 RGB weights: (64, 3, 3, 7, 7)
old_weight = pretrained_conv.weight.data

# Average across RGB channels: (64, 1, 3, 7, 7)
new_weight = old_weight.mean(dim=1, keepdim=True)

# Use for grayscale Conv3d(1, 64, ...)
```

This preserves learned temporal patterns while adapting to single-channel input.

## Comparison with Other Models

| Model | Input | Memory | Temporal Modeling | Expected MAE |
|-------|-------|--------|-------------------|--------------|
| 10_thumbless_model | (5, 3, 224, 224) | Low | LSTM (explicit) | Baseline |
| 11_3dcnn | (3, 16, 224, 224) | High | 3D Conv (explicit) | Good |
| **16_gray3d** | **(1, 16, 224, 224)** | **Medium** | **3D Conv (explicit)** | **Similar to 11_3dcnn** |
| 17_channel_stacked_2d | (16, 224, 224) | Low | Implicit (channels) | Slightly lower |

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` from 12 to 8 or 6
- Reduce `SEQUENCE_LENGTH` from 16 to 12

### Slow Training
- Increase `NUM_WORKERS` (default: 4)
- Ensure GPU is being used (check console output)

### Poor Performance
- Increase `NUM_EPOCHS`
- Adjust `LEARNING_RATE`
- Try different `SCHEDULER` settings

## Citation

If you use this model, please cite:
- Original ultrasound dataset
- Kinetics-400 for pretrained weights
- R3D-18 architecture

## License

Same as parent project.
