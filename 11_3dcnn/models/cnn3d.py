import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights, mc3_18, MC3_18_Weights


class R3D18_FingerAngle(nn.Module):
    """
    3D ResNet-18 model for finger angle prediction from image sequences.
    
    Architecture:
    1. 3D CNN (R3D-18) extracts spatiotemporal features
    2. FC layers predict 8 finger angles
    
    Input: (batch, channels, depth/time, height, width)
    Output: (batch, num_outputs)
    """
    
    def __init__(self, num_outputs=8, dropout=0.3, pretrained=True):
        """
        Args:
            num_outputs: Number of output angles (8 for thumbless)
            dropout: Dropout rate
            pretrained: Use pretrained weights from Kinetics-400
        """
        super(R3D18_FingerAngle, self).__init__()
        
        # Load pretrained R3D-18
        if pretrained:
            weights = R3D_18_Weights.DEFAULT
            self.backbone = r3d_18(weights=weights)
        else:
            self.backbone = r3d_18(weights=None)
        
        # Get the feature dimension before FC layer
        self.feature_dim = self.backbone.fc.in_features  # 512
        
        # Remove original FC layer
        self.backbone.fc = nn.Identity()
        
        # Custom regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, C, H, W)
               Will be permuted to (batch, C, seq_len, H, W) for 3D CNN
        
        Returns:
            Output tensor of shape (batch, num_outputs)
        """
        # Permute from (B, T, C, H, W) to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Extract spatiotemporal features
        features = self.backbone(x)  # (batch, 512)
        
        # Predict angles
        angles = self.regressor(features)
        
        return angles


class MC3_18_FingerAngle(nn.Module):
    """
    Mixed Convolution (MC3) model - uses 3D conv in early layers, 2D conv in later layers.
    More efficient than full 3D convolutions.
    
    Architecture:
    1. MC3-18 extracts spatiotemporal features
    2. FC layers predict 8 finger angles
    """
    
    def __init__(self, num_outputs=8, dropout=0.3, pretrained=True):
        """
        Args:
            num_outputs: Number of output angles (8 for thumbless)
            dropout: Dropout rate
            pretrained: Use pretrained weights from Kinetics-400
        """
        super(MC3_18_FingerAngle, self).__init__()
        
        # Load pretrained MC3-18
        if pretrained:
            weights = MC3_18_Weights.DEFAULT
            self.backbone = mc3_18(weights=weights)
        else:
            self.backbone = mc3_18(weights=None)
        
        # Get the feature dimension before FC layer
        self.feature_dim = self.backbone.fc.in_features  # 512
        
        # Remove original FC layer
        self.backbone.fc = nn.Identity()
        
        # Custom regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, C, H, W)
               Will be permuted to (batch, C, seq_len, H, W) for 3D CNN
        
        Returns:
            Output tensor of shape (batch, num_outputs)
        """
        # Permute from (B, T, C, H, W) to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Extract spatiotemporal features
        features = self.backbone(x)  # (batch, 512)
        
        # Predict angles
        angles = self.regressor(features)
        
        return angles


class R3D18_MultiScale(nn.Module):
    """
    3D ResNet-18 with multi-scale temporal input.
    Uses both original frames and frame differences (motion).
    
    Architecture:
    1. Two-stream input: original frames + frame differences
    2. Separate 3D CNN branches for each stream
    3. Feature fusion and FC layers
    """
    
    def __init__(self, num_outputs=8, dropout=0.3, pretrained=True):
        """
        Args:
            num_outputs: Number of output angles (8 for thumbless)
            dropout: Dropout rate
            pretrained: Use pretrained weights from Kinetics-400
        """
        super(R3D18_MultiScale, self).__init__()
        
        # Appearance stream (original frames)
        if pretrained:
            weights = R3D_18_Weights.DEFAULT
            self.appearance_stream = r3d_18(weights=weights)
        else:
            self.appearance_stream = r3d_18(weights=None)
        
        self.feature_dim = self.appearance_stream.fc.in_features  # 512
        self.appearance_stream.fc = nn.Identity()
        
        # Motion stream (frame differences)
        # Initialize with pretrained weights but will process different input
        if pretrained:
            self.motion_stream = r3d_18(weights=weights)
        else:
            self.motion_stream = r3d_18(weights=None)
        self.motion_stream.fc = nn.Identity()
        
        # Fusion and regression
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
    
    def compute_frame_diff(self, x):
        """
        Compute frame differences for motion stream.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            Frame differences of shape (B, C, T-1, H, W), padded to match original size
        """
        # Compute differences between consecutive frames
        diff = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        
        # Pad to match original temporal dimension (repeat last diff)
        padding = diff[:, :, -1:, :, :]
        diff = torch.cat([diff, padding], dim=2)
        
        return diff
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, C, H, W)
        
        Returns:
            Output tensor of shape (batch, num_outputs)
        """
        # Permute from (B, T, C, H, W) to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Appearance stream
        appearance_features = self.appearance_stream(x)  # (batch, 512)
        
        # Motion stream (frame differences)
        motion_input = self.compute_frame_diff(x)
        motion_features = self.motion_stream(motion_input)  # (batch, 512)
        
        # Fusion
        fused = torch.cat([appearance_features, motion_features], dim=1)  # (batch, 1024)
        fused = self.fusion(fused)  # (batch, 512)
        
        # Predict angles
        angles = self.regressor(fused)
        
        return angles


def create_model(config):
    """
    Create model from config dictionary.
    
    Args:
        config: Dictionary with model parameters
            - model_type: 'r3d18', 'mc3_18', or 'r3d18_multiscale'
            - num_outputs: Number of output angles
            - dropout: Dropout rate
            - pretrained: Use pretrained weights
    
    Returns:
        model: 3D CNN model
    """
    model_type = config.get('model_type', 'r3d18')
    num_outputs = config.get('num_outputs', 8)
    dropout = config.get('dropout', 0.3)
    pretrained = config.get('pretrained', True)
    
    if model_type == 'r3d18':
        model = R3D18_FingerAngle(
            num_outputs=num_outputs,
            dropout=dropout,
            pretrained=pretrained
        )
    elif model_type == 'mc3_18':
        model = MC3_18_FingerAngle(
            num_outputs=num_outputs,
            dropout=dropout,
            pretrained=pretrained
        )
    elif model_type == 'r3d18_multiscale':
        model = R3D18_MultiScale(
            num_outputs=num_outputs,
            dropout=dropout,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == '__main__':
    # Test models
    print("Testing 3D CNN models...\n")
    
    # Test input: (batch=4, seq=16, channels=3, H=112, W=112)
    # Note: 3D CNN typically uses 112x112 input and longer sequences
    test_input = torch.randn(4, 16, 3, 112, 112)
    
    models_to_test = ['r3d18', 'mc3_18', 'r3d18_multiscale']
    
    for model_type in models_to_test:
        print(f"{'='*60}")
        print(f"Model: {model_type}")
        print(f"{'='*60}")
        
        config = {
            'model_type': model_type,
            'num_outputs': 8,
            'dropout': 0.3,
            'pretrained': False  # False for quick testing
        }
        
        model = create_model(config)
        output = model(test_input)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: (4, 8)")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print()
