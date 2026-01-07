import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class R3D_FingerAngle(nn.Module):
    """
    3D ResNet (R3D-18) model for finger angle prediction from image sequences.
    
    Architecture:
    1. 3D CNN (R3D-18) extracts spatiotemporal features
    2. FC layers predict 8 finger angles
    
    3D CNN은 시간축과 공간축을 동시에 컨볼루션하여
    프레임 간의 움직임 패턴을 자연스럽게 포착합니다.
    """
    
    def __init__(self, num_outputs=8, dropout=0.3, pretrained=True):
        """
        Args:
            num_outputs: Number of output angles (8 for thumbless)
            dropout: Dropout rate
            pretrained: Use pretrained weights (Kinetics-400)
        """
        super(R3D_FingerAngle, self).__init__()
        
        # Load pretrained R3D-18
        if pretrained:
            weights = R3D_18_Weights.KINETICS400_V1
            self.r3d = r3d_18(weights=weights)
        else:
            self.r3d = r3d_18(weights=None)
        
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
            x: Input tensor of shape (batch_size, sequence_length, C, H, W)
               or (batch_size, C, sequence_length, H, W)
        
        Returns:
            Output tensor of shape (batch_size, num_outputs)
        """
        # R3D expects (B, C, T, H, W) format
        # Dataset outputs (B, T, C, H, W) where T=seq_len, C=3
        # We need to transpose to (B, C, T, H, W)
        
        if x.dim() == 5:
            B, dim1, dim2, H, W = x.shape
            
            # If dim2 == 3, input is (B, T, C, H, W) -> need transpose
            # If dim1 == 3, input is (B, C, T, H, W) -> already correct
            if dim2 == 3 and dim1 != 3:
                x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)
        
        # Extract features
        features = self.r3d(x)  # (B, 512)
        
        # Predict angles
        angles = self.regressor(features)  # (B, num_outputs)
        
        return angles


class R3D_FingerAngle_MultiScale(nn.Module):
    """
    Multi-scale 3D CNN model.
    원본 이미지와 프레임 간 차분(motion) 정보를 함께 사용합니다.
    """
    
    def __init__(self, num_outputs=8, dropout=0.3, pretrained=True):
        super(R3D_FingerAngle_MultiScale, self).__init__()
        
        # Appearance stream (원본 이미지)
        if pretrained:
            weights = R3D_18_Weights.KINETICS400_V1
            self.appearance_stream = r3d_18(weights=weights)
        else:
            self.appearance_stream = r3d_18(weights=None)
        
        # Motion stream (프레임 차분)
        self.motion_stream = r3d_18(weights=None)  # Motion은 pretrained 안 씀
        
        self.feature_size = self.appearance_stream.fc.in_features  # 512
        
        # Remove FC layers
        self.appearance_stream.fc = nn.Identity()
        self.motion_stream.fc = nn.Identity()
        
        # Combined regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_size * 2, 512),  # 두 스트림 결합
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_outputs)
        )
    
    def compute_motion(self, x):
        """
        프레임 간 차분 계산
        
        Args:
            x: (B, C, T, H, W)
        
        Returns:
            motion: (B, C, T-1, H, W) - zero-padded to (B, C, T, H, W)
        """
        # 프레임 간 차분: x[:,:,1:] - x[:,:,:-1]
        motion = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        
        # 첫 프레임에 zero padding (시퀀스 길이 유지)
        padding = torch.zeros_like(x[:, :, :1, :, :])
        motion = torch.cat([padding, motion], dim=2)
        
        return motion
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, C, H, W)
        
        Returns:
            Output tensor of shape (batch_size, num_outputs)
        """
        # Transpose to (B, C, T, H, W) if needed
        if x.dim() == 5:
            B, dim1, dim2, H, W = x.shape
            if dim2 == 3 and dim1 != 3:
                x = x.permute(0, 2, 1, 3, 4)
        
        # Compute motion
        motion = self.compute_motion(x)
        
        # Two-stream features
        appearance_feat = self.appearance_stream(x)
        motion_feat = self.motion_stream(motion)
        
        # Concatenate features
        combined = torch.cat([appearance_feat, motion_feat], dim=1)
        
        # Predict angles
        angles = self.regressor(combined)
        
        return angles


def create_model(config):
    """
    Create model from config dictionary.
    
    Args:
        config: Dictionary with model parameters
    
    Returns:
        model: R3D model
    """
    model_type = config.get('model_type', 'r3d')
    
    if model_type == 'r3d':
        model = R3D_FingerAngle(
            num_outputs=config.get('num_outputs', 8),
            dropout=config.get('dropout', 0.3),
            pretrained=config.get('pretrained', True)
        )
    elif model_type == 'r3d_multiscale':
        model = R3D_FingerAngle_MultiScale(
            num_outputs=config.get('num_outputs', 8),
            dropout=config.get('dropout', 0.3),
            pretrained=config.get('pretrained', True)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == '__main__':
    # Test model
    print("Testing R3D_FingerAngle...")
    model = R3D_FingerAngle(num_outputs=8, pretrained=True)
    
    # Test input: (batch=4, seq=16, channels=3, H=112, W=112)
    # R3D expects 112x112 input and at least 16 frames for optimal performance
    # But it can work with fewer frames
    test_input = torch.randn(4, 10, 3, 112, 112)  # 10 frames
    
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (4, 8)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*50)
    print("Testing R3D_FingerAngle_MultiScale...")
    model_ms = R3D_FingerAngle_MultiScale(num_outputs=8, pretrained=True)
    
    output_ms = model_ms(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output_ms.shape}")
    
    total_params_ms = sum(p.numel() for p in model_ms.parameters())
    print(f"Total parameters (MultiScale): {total_params_ms:,}")
