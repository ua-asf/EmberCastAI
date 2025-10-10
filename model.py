from torch import nn


class SimpleFireCNN(nn.Module):
    """Simple CNN for fire mask prediction from single time step to next"""

    def __init__(self, in_channels=4):
        """
        Args:
            in_channels: Number of input bands (default 4, flexible for future)
        """
        super().__init__()

        self.in_channels = in_channels

        # Encoder: Process N bands
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Decoder: Generate fire mask
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),  # Output single channel
            nn.Sigmoid(),  # Output values between 0 and 1
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            mask: Output tensor of shape (batch_size, 1, height, width)
        """
        features = self.encoder(x)
        mask = self.decoder(features)
        return mask

