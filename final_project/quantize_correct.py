#!/usr/bin/env python3
"""
Corrected quantization script with proper calibration using multiple samples
"""

import random
import torch.multiprocessing
import pickle
import os
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import glob

from model_resnet50_channelstack import CNN_ResNet50_ChannelStack_DPU
from pytorch_nndct.apis import torch_quantizer

print("="*70)
print("Corrected Quantization Script")
print("="*70)

# Configuration
WINDOW_SIZE = 7
IMG_DIR = "./data/251002_wrist"
QUANT_MODE = 'test'  # Change to 'test' after calibration is done
QUANT_DIR = './build/quantized_corrected'
NUM_CALIB_SAMPLES = 100  # Use 100 samples for calibration

print(f"\nConfiguration:")
print(f"  Window size: {WINDOW_SIZE}")
print(f"  Image directory: {IMG_DIR}")
print(f"  Quantization mode: {QUANT_MODE}")
print(f"  Output directory: {QUANT_DIR}")
print(f"  Calibration samples: {NUM_CALIB_SAMPLES}")

# Setup device
print("\nDevice Setup...")
device = torch.device('cpu')

# Load model
print("Loading model...")
model = CNN_ResNet50_ChannelStack_DPU(window_size=WINDOW_SIZE)
checkpoint = torch.load("./checkpoints_dpu/best_model.pth", map_location=device)#, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
#model.load_state_dict(torch.load("./checkpoints_dpu/best_model.pth", map_location=device), strict=False)
model.eval()
print("[OK] Model loaded")

# Get image paths
print(f"\nLoading images from {IMG_DIR}...")
img_paths = []
for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
    img_paths.extend(glob.glob(os.path.join(IMG_DIR, ext)))
img_paths = sorted(img_paths)

if len(img_paths) == 0:
    raise RuntimeError(f"No image files found in {IMG_DIR}")

print(f"[OK] Found {len(img_paths)} images")

# Define preprocessing (must match training!)
transform = T.Compose([
    T.Resize((380, 508)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

def load_window(start_idx):
    """Load a window of WINDOW_SIZE frames starting from start_idx"""
    frames = []
    for i in range(WINDOW_SIZE):
        img_idx = (start_idx + i) % len(img_paths)  # Wrap around if needed
        img_path = img_paths[img_idx]
        img = Image.open(img_path).convert('L')
        img_t = transform(img)
        frames.append(img_t)

    # Stack and add batch dimension: (1, 7, 380, 508)
    stack = torch.cat(frames, dim=0)
    return stack.unsqueeze(0).to(device)

# Initialize quantizer with dummy input
print("\nInitializing quantizer...")
dummy_input = load_window(0)
print(f"  Input shape: {dummy_input.shape}")

os.makedirs(QUANT_DIR, exist_ok=True)

quantizer = torch_quantizer(
    QUANT_MODE,
    model,
    (dummy_input),
    output_dir=QUANT_DIR,
    device=device
)
quant_model = quantizer.quant_model.to(device)
print("[OK] Quantizer initialized")

# Run calibration or test
if QUANT_MODE == 'calib':
    print(f"\n{'='*70}")
    print("Running CALIBRATION with multiple samples")
    print(f"{'='*70}")

    # Determine how many samples we can create
    max_samples = len(img_paths) - WINDOW_SIZE + 1
    num_samples = min(NUM_CALIB_SAMPLES, max_samples) # calib 100번 돈다 

    print(f"\nCalibrating with {num_samples} samples...")
    print(f"  (Maximum possible: {max_samples})")

    with torch.no_grad():
        for i in range(num_samples):
            # Load window starting at frame i
            input_data = load_window(i)

            # Run through quantization model
            _ = quant_model(input_data)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_samples} samples...")

    print(f"\n[OK] Calibration complete with {num_samples} samples")

    # Export quantization config
    print("\nExporting quantization config...")
    quantizer.export_quant_config()
    print(f"[OK] Config saved to {QUANT_DIR}")

    print(f"\n{'='*70}")
    print("NEXT STEP:")
    print(f"{'='*70}")
    print("1. Change QUANT_MODE to 'test' in this script")
    print("2. Run this script again to export the xmodel")
    print(f"{'='*70}")

elif QUANT_MODE == 'test':
    print(f"\n{'='*70}")
    print("Running TEST mode - Exporting xmodel")
    print(f"{'='*70}")

    # Run one sample through the model
    print("\nRunning test inference...")
    with torch.no_grad():
        input_data = load_window(0)
        _ = quant_model(input_data)

    print("[OK] Test inference complete")

    # Export xmodel
    print("\nExporting xmodel...")
    quantizer.export_xmodel(deploy_check=False, output_dir=QUANT_DIR)
    print(f"[OK] xmodel saved to {QUANT_DIR}")

    print(f"\n{'='*70}")
    print("NEXT STEP:")
    print(f"{'='*70}")
    print("1. Check the output fix_point in quant_info.json")
    print("2. It should be much lower than 12 (e.g., 4-6)")
    print("3. Deploy the new xmodel to DPU for testing")
    print(f"{'='*70}")

else:
    raise ValueError(f"Invalid QUANT_MODE: {QUANT_MODE}")

print("\n" + "="*70)
print("Done!")
print("="*70)
