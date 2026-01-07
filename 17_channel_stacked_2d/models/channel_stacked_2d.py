import torch
import torch.nn as nn
from torchvision import models


class ChannelStacked2D_FingerAngle(nn.Module):
    """
    Channel-Stacked 2D CNN for finger angle prediction.

    Novel approach: Treats temporal frames as input channels.
    - Input: (Batch, num_frames, Height, Width) where num_frames = 16
    - Treats 16 grayscale frames as 16-channel image
    - Uses 2D CNN (ResNet18) instead of 3D CNN

    Advantages:
    - Maximum efficiency (2D convs are faster than 3D)
    - Lower memory usage
    - Higher batch sizes possible
    - Standard 2D architecture (easy to swap backbones)

    Trade-off:
    - No explicit temporal modeling (time is implicit in channels)
    - May perform slightly worse than 3D CNN for temporal dynamics
    """

    def __init__(self, num_frames=16, num_outputs=8, dropout=0.3, pretrained=True):
        """
        Args:
            num_frames: Number of temporal frames (treated as channels)
            num_outputs: Number of output angles (8 for thumbless)
            dropout: Dropout rate
            pretrained: Use ImageNet pretrained weights for ResNet18
        """
        super(ChannelStacked2D_FingerAngle, self).__init__()

        self.num_frames = num_frames

        # Load pretrained ResNet18
        if pretrained:
            self.backbone = models.resnet18(pretrained=True)

            # Modify first conv: 3 channels → num_frames channels
            old_conv = self.backbone.conv1
            old_weight = old_conv.weight.data  # Shape: (64, 3, 7, 7)

            # Create new Conv2d with num_frames input channels
            self.backbone.conv1 = nn.Conv2d(
                num_frames,  # e.g., 16 grayscale frames
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

            # Initialize new conv weights by repeating averaged RGB weights
            # Strategy: Average RGB weights → tile across all channels
            avg_weight = old_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)

            # Repeat for all input channels and normalize
            self.backbone.conv1.weight.data = avg_weight.repeat(1, num_frames, 1, 1) / num_frames

        else:
            # Create from scratch without pretrained weights
            self.backbone = models.resnet18(pretrained=False)

            # Still modify first conv for num_frames channels
            self.backbone.conv1 = nn.Conv2d(
                num_frames,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        # Get feature dimension before FC layer
        self.feature_dim = self.backbone.fc.in_features  # 512 for ResNet18

        # Replace final FC layer with our regression head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, T, H, W) - grayscale sequence
               T = num_frames (e.g., 16)
               H, W = height, width (e.g., 224, 224)

        Returns:
            Output tensor of shape (batch_size, num_outputs)
        """
        # Input is (B, T, H, W) where T is treated as channels
        # x is already in the right format for 2D CNN: (B, C=T, H, W)

        # Forward through ResNet18
        output = self.backbone(x)  # (B, num_outputs)

        return output


def create_model(config):
    """
    Create model from config dictionary.

    Args:
        config: Dictionary with model parameters
            - model_type: 'channel_stacked_2d'
            - num_frames: Number of temporal frames (default: 16)
            - num_outputs: Number of output angles (default: 8)
            - dropout: Dropout rate (default: 0.3)
            - pretrained: Use pretrained weights (default: True)

    Returns:
        model: Channel-stacked 2D CNN model
    """
    model = ChannelStacked2D_FingerAngle(
        num_frames=config.get('num_frames', 16),
        num_outputs=config.get('num_outputs', 8),
        dropout=config.get('dropout', 0.3),
        pretrained=config.get('pretrained', True)
    )

    return model


if __name__ == '__main__':
    # Test model
    print("Testing ChannelStacked2D_FingerAngle...")
    print("="*60)

    model = ChannelStacked2D_FingerAngle(num_frames=16, num_outputs=8, pretrained=True)

    # Test input: (batch=4, frames=16, H=224, W=224) - grayscale sequence treated as 16 channels
    test_input = torch.randn(4, 16, 224, 224)

    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (4, 8)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Memory estimation
    batch_size = 16  # Can use higher batch size than 3D CNN
    num_frames = 16
    h, w = 224, 224

    # Input memory: B * T * H * W * 4 bytes (float32)
    input_mem = batch_size * num_frames * h * w * 4 / (1024**3)
    print(f"\nMemory usage estimation (batch_size={batch_size}):")
    print(f"  Input tensor: {input_mem:.3f} GB")
    print(f"  Compared to 3D CNN with batch=12: ~{12 * num_frames * h * w * 4 / (1024**3):.3f} GB")

    print("\n" + "="*60)
    print("Key Insight:")
    print("  - Time (16 frames) is encoded as 16 CHANNELS")
    print("  - 2D CNN processes spatial patterns across these channels")
    print("  - No explicit temporal modeling (implicit in channel patterns)")
    print("  - Much faster than 3D CNN!")
    print("="*60)

    print("\n" + "="*60)
    print("Testing with factory function...")

    config = {
        'model_type': 'channel_stacked_2d',
        'num_frames': 16,
        'num_outputs': 8,
        'dropout': 0.3,
        'pretrained': True
    }

    model2 = create_model(config)
    output2 = model2(test_input)
    print(f"Factory-created model output shape: {output2.shape}")

    print("\nModel created successfully!")
