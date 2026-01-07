import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class GrayscaleR3D18_FingerAngle(nn.Module):
    """
    Grayscale 3D ResNet (R3D-18) model for finger angle prediction.

    Key differences from RGB R3D-18:
    - Input: Single grayscale channel instead of 3 RGB channels
    - First Conv3d layer modified: 1 input channel instead of 3
    - Weight initialization: Averaged Kinetics-400 RGB weights
    - Memory usage: 66% reduction compared to RGB version

    Architecture:
    1. Grayscale 3D CNN (R3D-18) extracts spatiotemporal features
    2. FC layers predict 8 finger angles
    """

    def __init__(self, num_outputs=8, dropout=0.3, pretrained=True):
        """
        Args:
            num_outputs: Number of output angles (8 for thumbless)
            dropout: Dropout rate
            pretrained: Use pretrained weights from Kinetics-400 (averaged for grayscale)
        """
        super(GrayscaleR3D18_FingerAngle, self).__init__()

        # Load pretrained R3D-18
        if pretrained:
            weights = R3D_18_Weights.KINETICS400_V1
            self.r3d = r3d_18(weights=weights)

            # Modify first Conv3d layer: 3 channels → 1 channel
            # Original: Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
            old_conv = self.r3d.stem[0]
            old_weight = old_conv.weight.data  # Shape: (64, 3, 3, 7, 7)

            # Create new Conv3d with 1 input channel
            new_conv = nn.Conv3d(
                1,  # Grayscale input
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False
            )

            # Initialize weights by averaging RGB channels
            # This preserves the learned patterns from Kinetics-400
            new_conv.weight.data = old_weight.mean(dim=1, keepdim=True)  # Shape: (64, 1, 3, 7, 7)

            # Replace first conv layer
            self.r3d.stem[0] = new_conv

        else:
            # Create from scratch without pretrained weights
            self.r3d = r3d_18(weights=None)

            # Still modify first conv for grayscale
            old_conv = self.r3d.stem[0]
            new_conv = nn.Conv3d(
                1,  # Grayscale input
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False
            )
            self.r3d.stem[0] = new_conv

        # R3D-18 output feature size
        self.feature_size = self.r3d.fc.in_features  # 512

        # Replace the final FC layer with our regression head
        self.r3d.fc = nn.Identity()

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_size, 256),
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
               T = sequence_length (e.g., 16)
               H, W = height, width (e.g., 224, 224)

        Returns:
            Output tensor of shape (batch_size, num_outputs)
        """
        # Dataset outputs (B, T, H, W) for grayscale
        # R3D expects (B, C, T, H, W) format
        # We need to add channel dimension and permute

        if x.dim() == 4:
            # Input is (B, T, H, W) - grayscale sequence
            x = x.unsqueeze(1)  # Add channel dim: (B, 1, T, H, W)
        elif x.dim() == 5:
            # Input might already be (B, C, T, H, W) or (B, T, C, H, W)
            B, dim1, dim2, H, W = x.shape

            # If dim2 == 1, input is (B, T, 1, H, W) -> need to permute to (B, 1, T, H, W)
            if dim2 == 1 and dim1 != 1:
                x = x.permute(0, 2, 1, 3, 4)  # (B, T, 1, H, W) -> (B, 1, T, H, W)

        # Now x is (B, 1, T, H, W) - ready for 3D CNN

        # Extract features
        features = self.r3d(x)  # (B, 512)

        # Predict angles
        angles = self.regressor(features)  # (B, num_outputs)

        return angles


def create_model(config):
    """
    Create model from config dictionary.

    Args:
        config: Dictionary with model parameters
            - model_type: 'gray3d'
            - num_outputs: Number of output angles (default: 8)
            - dropout: Dropout rate (default: 0.3)
            - pretrained: Use pretrained weights (default: True)

    Returns:
        model: Grayscale R3D-18 model
    """
    model = GrayscaleR3D18_FingerAngle(
        num_outputs=config.get('num_outputs', 8),
        dropout=config.get('dropout', 0.3),
        pretrained=config.get('pretrained', True)
    )

    return model


if __name__ == '__main__':
    # Test model
    print("Testing GrayscaleR3D18_FingerAngle...")
    print("="*60)

    model = GrayscaleR3D18_FingerAngle(num_outputs=8, pretrained=True)

    # Test input: (batch=4, seq=16, H=224, W=224) - grayscale
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
    batch_size = 12  # Increased from 8 due to lower memory usage
    seq_len = 16
    h, w = 224, 224

    # Input memory (grayscale): B * T * H * W * 4 bytes (float32)
    input_mem = batch_size * seq_len * h * w * 4 / (1024**3)
    print(f"\nMemory usage estimation (batch_size={batch_size}):")
    print(f"  Input tensor: {input_mem:.3f} GB")
    print(f"  Compared to RGB (3 channels): ~{input_mem*3:.3f} GB")
    print(f"  Memory reduction: ~{((1 - 1/3) * 100):.1f}%")

    print("\n" + "="*60)
    print("Testing with factory function...")

    config = {
        'model_type': 'gray3d',
        'num_outputs': 8,
        'dropout': 0.3,
        'pretrained': True
    }

    model2 = create_model(config)
    output2 = model2(test_input)
    print(f"Factory-created model output shape: {output2.shape}")

    print("\nModel created successfully!")
