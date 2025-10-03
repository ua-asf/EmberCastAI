from torch import nn

class SimpleFireCNN(nn.Module):
    """Simple CNN for fire mask prediction"""
    
    def __init__(self):
        super().__init__()
        
        # Encoder: Process 1 day × 3 bands = 3 channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder: Generate fire mask
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),  # Output single channel
            nn.Sigmoid()  # Output values between 0 and 1
        )
        
    def forward(self, x):
        # x shape: (batch_size, 100, 100, 3) - convert to (batch_size, 3, 100, 100)
        if x.dim() == 4 and x.shape[-1] == 3:
            # Input is (batch_size, H, W, C) - convert to (batch_size, C, H, W)
            x = x.permute(0, 3, 1, 2)
        
        # x shape: (batch_size, 3, 100, 100)
        features = self.encoder(x)
        mask = self.decoder(features)
        # mask shape: (batch_size, 1, 100, 100)
        
        # Convert back to (batch_size, H, W, C) format
        mask = mask.permute(0, 2, 3, 1)  # Shape: (batch_size, 100, 100, 1)
        
        return mask
