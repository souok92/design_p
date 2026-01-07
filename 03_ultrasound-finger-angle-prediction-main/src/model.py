#!/usr/bin/env python3
"""
CNN-TCN Hybrid Model for Ultrasound-to-Angle Prediction
- CNN Backbone: ResNet-18 for spatial feature extraction
- TCN: Dilated Causal Convolution for temporal dependency modeling
- Output: 2 regression values (MCP_Flex, PIP_Flex)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) with Dilated Causal Convolutions
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation: 1, 2, 4, 8, ...
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            # Causal padding to ensure only past frames influence current prediction
            padding = (kernel_size - 1) * dilation_size

            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                         padding=padding, dilation=dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(out_channels)
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, channels, sequence_length)
        return self.network(x)


class CNNBackbone(nn.Module):
    """
    CNN Backbone for spatial feature extraction from ultrasound images
    Using ResNet-18 pretrained on ImageNet
    """
    def __init__(self, feature_dim=512):
        super(CNNBackbone, self).__init__()

        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=True)

        # Remove final FC layer (we only need features)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Feature dimension from ResNet-18 is 512
        self.feature_dim = feature_dim

    def forward(self, x):
        # x shape: (batch, 3, H, W)
        features = self.features(x)  # (batch, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 512)
        return features


class CNN_TCN_Model(nn.Module):
    """
    Complete CNN-TCN Model for Ultrasound-to-Angle Prediction

    Architecture:
    1. CNN Backbone (ResNet-18): Extract spatial features from each frame
    2. TCN: Model temporal dependencies across frames
    3. Regression Head: Predict MCP and PIP angles
    """
    def __init__(self, window_size=7, tcn_channels=[256, 128, 64], dropout=0.2):
        super(CNN_TCN_Model, self).__init__()

        self.window_size = window_size

        # 1. CNN Backbone for spatial features
        self.cnn = CNNBackbone(feature_dim=512)

        # 2. TCN for temporal modeling
        # Input: 512 (CNN features), Output channels: tcn_channels
        self.tcn = TemporalConvNet(
            num_inputs=512,
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=dropout
        )

        # 3. Regression Head
        final_tcn_dim = tcn_channels[-1]
        self.regression_head = nn.Sequential(
            nn.Linear(final_tcn_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Output: [MCP_angle, PIP_angle]
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, window_size, 3, H, W)
               - batch: batch size
               - window_size: number of consecutive frames
               - 3: RGB channels
               - H, W: image height and width

        Returns:
            predictions: Tensor of shape (batch, 2) containing [MCP, PIP] angles
        """
        batch_size, window_size, C, H, W = x.shape

        # 1. Extract CNN features for each frame in the window
        # Reshape to process all frames at once
        x = x.view(batch_size * window_size, C, H, W)
        cnn_features = self.cnn(x)  # (batch * window_size, 512)

        # Reshape back to separate batch and temporal dimensions
        cnn_features = cnn_features.view(batch_size, window_size, -1)  # (batch, window_size, 512)

        # 2. Transpose for TCN: (batch, features, time)
        cnn_features = cnn_features.transpose(1, 2)  # (batch, 512, window_size)

        # 3. Apply TCN to model temporal dependencies
        tcn_features = self.tcn(cnn_features)  # (batch, tcn_channels[-1], window_size)

        # 4. Take the last time step (most recent prediction)
        tcn_output = tcn_features[:, :, -1]  # (batch, tcn_channels[-1])

        # 5. Regression to predict angles
        predictions = self.regression_head(tcn_output)  # (batch, 2)

        return predictions


def get_model(window_size=7, device='cuda'):
    """
    Factory function to create and initialize the model

    Args:
        window_size: Number of consecutive frames to use for prediction
        device: 'cuda' or 'cpu'

    Returns:
        model: Initialized CNN-TCN model
    """
    model = CNN_TCN_Model(
        window_size=window_size,
        tcn_channels=[256, 128, 64],
        dropout=0.2
    )

    return model.to(device)


if __name__ == "__main__":
    # Test the model
    print("=" * 70)
    print("CNN-TCN Model Architecture Test")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Model parameters
    window_size = 7
    batch_size = 4
    img_height, img_width = 224, 224

    # Create model
    model = get_model(window_size=window_size, device=device)

    # Create dummy input
    dummy_input = torch.randn(batch_size, window_size, 3, img_height, img_width).to(device)

    # Forward pass
    output = model(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nSample output (MCP, PIP angles):\n{output}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n" + "=" * 70)
    print("✓ Model architecture test passed!")
    print("=" * 70)
