import torch
import torch.nn as nn
import torchvision.models as models


class CNN_LSTM_FingerAngle(nn.Module):
    """
    CNN + LSTM model for finger angle prediction from image sequences.

    Architecture:
    1. CNN (ResNet18) extracts spatial features from each frame
    2. LSTM processes temporal sequence
    3. FC layers predict 8 finger angles
    """

    def __init__(self, sequence_length=5, lstm_hidden_size=256,
                 lstm_num_layers=2, num_outputs=8, dropout=0.3,
                 pretrained=True):
        """
        Args:
            sequence_length: Number of frames in input sequence
            lstm_hidden_size: Hidden size of LSTM
            lstm_num_layers: Number of LSTM layers
            num_outputs: Number of output angles (8 for thumbless)
            dropout: Dropout rate
            pretrained: Use pretrained ResNet weights
        """
        super(CNN_LSTM_FingerAngle, self).__init__()

        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        # CNN backbone: ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        # Remove the final FC layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        # ResNet18 final feature size
        self.cnn_output_size = 512

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
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

        Returns:
            Output tensor of shape (batch_size, num_outputs)
        """
        batch_size, seq_len, C, H, W = x.size()

        # Reshape to process all frames through CNN
        # (batch_size * sequence_length, C, H, W)
        x = x.view(batch_size * seq_len, C, H, W)

        # Extract features with CNN
        cnn_features = self.cnn(x)  # (batch_size * seq_len, 512, 1, 1)
        cnn_features = cnn_features.view(batch_size * seq_len, -1)  # (batch_size * seq_len, 512)

        # Reshape for LSTM
        # (batch_size, sequence_length, 512)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)

        # Use the last hidden state
        # lstm_out[:, -1, :] has shape (batch_size, lstm_hidden_size)
        last_hidden = lstm_out[:, -1, :]

        # Predict angles
        angles = self.fc(last_hidden)  # (batch_size, num_outputs)

        return angles


def create_model(config):
    """
    Create model from config dictionary.

    Args:
        config: Dictionary with model parameters

    Returns:
        model: CNN_LSTM_FingerAngle model
    """
    model = CNN_LSTM_FingerAngle(
        sequence_length=config.get('sequence_length', 5),
        lstm_hidden_size=config.get('lstm_hidden_size', 256),
        lstm_num_layers=config.get('lstm_num_layers', 2),
        num_outputs=config.get('num_outputs', 8),
        dropout=config.get('dropout', 0.3),
        pretrained=config.get('pretrained', True)
    )
    return model


if __name__ == '__main__':
    # Test model
    model = CNN_LSTM_FingerAngle(sequence_length=5)

    # Test input: (batch=4, seq=5, channels=3, H=224, W=224)
    test_input = torch.randn(4, 5, 3, 224, 224)

    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (4, 8)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
