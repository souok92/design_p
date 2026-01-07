import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UNetEncoder(nn.Module):
    """
    U-Net Encoder for feature extraction from ultrasound images.
    
    출력: 시퀀스의 각 프레임에서 추출된 특징 벡터
    """
    def __init__(self, in_channels=3, base_features=64, depth=4):
        super().__init__()
        
        self.depth = depth
        self.inc = DoubleConv(in_channels, base_features)
        
        # Encoder layers
        self.encoders = nn.ModuleList()
        features = base_features
        for i in range(depth):
            self.encoders.append(Down(features, features * 2))
            features *= 2
        
        # Final feature size after encoding
        self.final_features = features  # 64 * 2^depth = 1024 (depth=4)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) single image or (B, T, C, H, W) sequence
        
        Returns:
            features: (B, final_features) or (B, T, final_features)
        """
        # Handle sequence input
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            is_sequence = True
        else:
            is_sequence = False
            B = x.shape[0]
            T = 1
        
        # Encoder forward
        x = self.inc(x)
        for encoder in self.encoders:
            x = encoder(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(B * T, -1)
        
        # Reshape back to sequence
        if is_sequence:
            x = x.view(B, T, -1)
        
        return x


class UNetGRU(nn.Module):
    """
    U-Net Encoder + GRU x 2 for finger angle prediction.
    
    Architecture:
    1. U-Net Encoder: 각 프레임에서 공간 특징 추출
    2. GRU x 2: 시간적 패턴 학습
    3. FC: 각도 예측
    
    Reference: "Improved fascicle length estimates from ultrasound 
               using a U-net-LSTM framework" (Harvard, 2024)
    """
    def __init__(
        self,
        in_channels=3,
        base_features=64,
        encoder_depth=4,
        gru_hidden_size=256,
        gru_num_layers=2,
        num_outputs=8,
        dropout=0.3
    ):
        super().__init__()
        
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        
        # U-Net Encoder
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_features=base_features,
            depth=encoder_depth
        )
        
        encoder_output_size = base_features * (2 ** encoder_depth)  # 1024 (depth=4)
        
        # Feature projection (optional, to reduce dimension before GRU)
        self.feature_proj = nn.Sequential(
            nn.Linear(encoder_output_size, gru_hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GRU layers (2 layers as in the paper)
        self.gru = nn.GRU(
            input_size=gru_hidden_size * 2,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_size // 2, num_outputs)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) image sequence
        
        Returns:
            angles: (B, num_outputs)
        """
        B, T, C, H, W = x.shape
        
        # Extract features from each frame using U-Net encoder
        # x: (B, T, C, H, W) -> features: (B, T, encoder_output_size)
        features = self.encoder(x)
        
        # Project features
        features = self.feature_proj(features)  # (B, T, gru_hidden_size * 2)
        
        # GRU forward
        gru_out, h_n = self.gru(features)  # gru_out: (B, T, gru_hidden_size)
        
        # Use last hidden state
        last_hidden = gru_out[:, -1, :]  # (B, gru_hidden_size)
        
        # Predict angles
        angles = self.regressor(last_hidden)  # (B, num_outputs)
        
        return angles


class UNetGRULight(nn.Module):
    """
    경량화 버전 - 더 적은 파라미터
    """
    def __init__(
        self,
        in_channels=3,
        base_features=32,  # 64 -> 32
        encoder_depth=3,   # 4 -> 3
        gru_hidden_size=128,  # 256 -> 128
        gru_num_layers=2,
        num_outputs=8,
        dropout=0.3
    ):
        super().__init__()
        
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_features=base_features,
            depth=encoder_depth
        )
        
        encoder_output_size = base_features * (2 ** encoder_depth)
        
        self.feature_proj = nn.Sequential(
            nn.Linear(encoder_output_size, gru_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_size // 2, num_outputs)
        )
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        features = self.encoder(x)
        features = self.feature_proj(features)
        
        gru_out, _ = self.gru(features)
        last_hidden = gru_out[:, -1, :]
        
        angles = self.regressor(last_hidden)
        
        return angles


def create_model(config):
    """Create model from config"""
    model_type = config.get('model_type', 'unet_gru')
    
    if model_type == 'unet_gru':
        model = UNetGRU(
            in_channels=config.get('in_channels', 3),
            base_features=config.get('base_features', 64),
            encoder_depth=config.get('encoder_depth', 4),
            gru_hidden_size=config.get('gru_hidden_size', 256),
            gru_num_layers=config.get('gru_num_layers', 2),
            num_outputs=config.get('num_outputs', 8),
            dropout=config.get('dropout', 0.3)
        )
    elif model_type == 'unet_gru_light':
        model = UNetGRULight(
            in_channels=config.get('in_channels', 3),
            base_features=config.get('base_features', 32),
            encoder_depth=config.get('encoder_depth', 3),
            gru_hidden_size=config.get('gru_hidden_size', 128),
            gru_num_layers=config.get('gru_num_layers', 2),
            num_outputs=config.get('num_outputs', 8),
            dropout=config.get('dropout', 0.3)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == '__main__':
    # Test models
    print("Testing UNetGRU...")
    
    model = UNetGRU(
        base_features=64,
        encoder_depth=4,
        gru_hidden_size=256,
        gru_num_layers=2
    )
    
    # Test input: (batch=2, seq=10, channels=3, H=224, W=224)
    test_input = torch.randn(2, 10, 3, 224, 224)
    
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*50)
    print("Testing UNetGRULight...")
    
    model_light = UNetGRULight()
    output_light = model_light(test_input)
    print(f"Output shape: {output_light.shape}")
    
    total_params_light = sum(p.numel() for p in model_light.parameters())
    print(f"Total parameters (Light): {total_params_light:,}")
