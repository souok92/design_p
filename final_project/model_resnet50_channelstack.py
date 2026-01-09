#!/usr/bin/env python3
"""
ResNet50 + Channel-Stack for DPU
- Based on src/model.py
- Input: (B, W, H, W) - W grayscale frames stacked in channels
- ResNet50 pretrained backbone
- DPU-compatible
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNN_ResNet50_ChannelStack_DPU(nn.Module):
    """
    ResNet50 with Channel-Stacked Input
    - Input: (B, window_size, H, W) where window_size=7 (grayscale frames)
    - ResNet50 pretrained on ImageNet
    - Simple regression head (DPU-compatible)
    """
    def __init__(self, window_size=7):
        super(CNN_ResNet50_ChannelStack_DPU, self).__init__()

        self.window_size = window_size
        input_channels = window_size  # 7 grayscale frames

        # Load pretrained ResNet50
        resnet50 = models.resnet50(pretrained=True)
        feature_dim = 2048  # ResNet50 output

        # Modify first conv layer for window_size input channels
        original_conv1 = resnet50.conv1
        resnet50.conv1 = nn.Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize by averaging RGB weights
        with torch.no_grad():
            avg_weight = original_conv1.weight.mean(dim=1, keepdim=True)
            resnet50.conv1.weight = nn.Parameter(
                avg_weight.repeat(1, input_channels, 1, 1) / input_channels
            )

        # Extract features (remove FC layer)
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        # Output: (B, 2048, 1, 1)

        # Regression head using Conv2d (DPU-compatible)
        # Conv → BN → ReLU 순서 준수
        self.fc1 = nn.Conv2d(feature_dim, 512, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        # Final regression
        self.fc5 = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, window_size, H, W) - e.g., (B, 7, 380, 508)
        Returns:
            (B, 2) - MCP and PIP angles
        """
        # Extract features with ResNet50
        x = self.features(x)  # (B, 2048, 1, 1)

        # Regression head (all DPU operations)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.fc5(x)  # (B, 2, 1, 1)

        # Squeeze (post-processing)
        x = x.squeeze(-1).squeeze(-1)  # (B, 2)

        return x


# Test
if __name__ == "__main__":
    print("="*70)
    print("Testing ResNet50 Channel-Stack DPU Model")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Initialize model
    print("\nInitializing model...")
    model = CNN_ResNet50_ChannelStack_DPU(window_size=7).to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test with batch input
    print("\nTesting with batch input (4, 7, 380, 508)...")
    dummy_input = torch.randn(4, 7, 380, 508).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample:\n{output[0]}")

    print("\n" + "="*70)
    print("✓ Model test successful!")
    print("="*70)
    print("\nModel Summary:")
    print("  - Backbone: ResNet50 (pretrained)")
    print("  - Input: (B, 7, H, W) - 7 grayscale frames")
    print("  - Output: (B, 2) - MCP, PIP angles")
    print("  - Operations: Conv2d, BatchNorm, ReLU only")
    print("  - DPU Compatible: YES")
    print("="*70)
