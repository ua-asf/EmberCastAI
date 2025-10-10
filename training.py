import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm
from model import SimpleFireCNN

# Configuration
SQUARE_SIZE = 100


class FireDataset(Dataset):
    """Dataset for fire mask prediction - slices large TIFFs into square tiles"""

    def __init__(
        self, data_dir="finalized_dataset", square_size=SQUARE_SIZE, normalize=True
    ):
        """
        Args:
            data_dir: Path to dataset directory containing data_point_{N} folders
            square_size: Size of square tiles to extract
            normalize: If True, normalize UINT16 values to [0, 1]
        """
        self.data_dir = Path(data_dir)
        self.square_size = square_size
        self.normalize = normalize
        self.tiles = []

        # Find all data_point_N directories
        data_points = sorted(self.data_dir.glob("data_point_*"))

        # Extract tiles with progress bar
        print("Extracting tiles from dataset...")
        for data_point in tqdm(data_points, desc="Processing files"):
            before_path = data_point / "before.tiff"
            after_path = data_point / "after.tiff"

            if before_path.exists() and after_path.exists():
                self._extract_tiles(before_path, after_path)

        print(f"Found {len(self.tiles)} tiles from dataset")

        if len(self.tiles) == 0:
            raise ValueError(f"No valid tiles found in {data_dir}")

    def _extract_tiles(self, before_path, after_path):
        """
        Extract square_size×square_size tiles from a pair of TIFF files

        Args:
            before_path: Path to before.tiff
            after_path: Path to after.tiff
        """
        # Read the full images
        with rasterio.open(before_path) as src:
            before_data = src.read()  # Shape: (bands, height, width)
            height, width = src.height, src.width

        with rasterio.open(after_path) as src:
            after_data = src.read(1)  # Read only first band, shape: (height, width)

        # Calculate number of tiles in each dimension
        n_tiles_h = height // self.square_size
        n_tiles_w = width // self.square_size

        # Extract non-overlapping tiles
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                y_start = i * self.square_size
                y_end = y_start + self.square_size
                x_start = j * self.square_size
                x_end = x_start + self.square_size

                # Extract tile from before.tiff (all bands)
                before_tile = before_data[:, y_start:y_end, x_start:x_end]

                # Extract tile from after.tiff (first band only)
                after_tile = after_data[y_start:y_end, x_start:x_end]

                # Store as tuple (input_tile, target_tile)
                self.tiles.append((before_tile.copy(), after_tile.copy()))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        """
        Returns:
            input_data: Tensor of shape (C, H, W) with all bands from tile
            target_mask: Tensor of shape (1, H, W) with first band from corresponding tile
        """
        before_tile, after_tile = self.tiles[idx]

        # Convert to float32
        input_data = before_tile.astype(np.float32)
        target_mask = after_tile.astype(np.float32)

        # Normalize if requested
        if self.normalize:
            input_data /= 65535.0
            target_mask /= 65535.0

        # Add channel dimension to target mask: (H, W) -> (1, H, W)
        target_mask = target_mask[np.newaxis, ...]

        # Convert to tensors
        input_tensor = torch.from_numpy(input_data).contiguous()
        target_tensor = torch.from_numpy(target_mask).contiguous()

        return input_tensor, target_tensor


def upload_to_s3(file_path, bucket_name, s3_key):
    """Upload file to S3 bucket"""
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_KEY"),
            aws_secret_access_key=os.environ.get("AWS_SECRET"),
            region_name=os.environ.get("AWS_REGION"),
        )

        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
        return True

    except ClientError as e:
        print(f"S3 upload error: {e}")
        return False


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0

    pbar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / len(val_loader)


def train_model(
    data_dir="finalized_dataset",
    square_size=SQUARE_SIZE,
    batch_size=8,
    learning_rate=0.001,
    num_epochs=50,
    val_split=0.2,
    device="cpu",
    checkpoint_dir="checkpoints",
    s3_bucket=None,
):
    """
    Train the SimpleFireCNN model

    Args:
        data_dir: Path to dataset directory
        square_size: Size of square tiles (should match SQUARE_SIZE used in data preparation)
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        val_split: Fraction of data for validation (0.2 = 20%)
        device: Device to train on ('cpu' or 'cuda')
        checkpoint_dir: Directory to save checkpoints
        s3_bucket: S3 bucket name for uploading checkpoints (optional)
    """
    # Load dataset
    print("Loading dataset and extracting tiles...")
    dataset = FireDataset(data_dir, square_size=square_size)

    # Get input channels from first sample
    sample_input, _ = dataset[0]
    in_channels = sample_input.shape[0]
    print(f"Input channels: {in_channels}")
    print(f"Tile size: {square_size}×{square_size}")

    # Split into train/validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    print(
        f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
    )

    # Create data loaders
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # Initialize model
    model = SimpleFireCNN(in_channels=in_channels).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")

    print(f"\nTraining on {device}")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "in_channels": in_channels,
            "square_size": square_size,
        }

        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Upload to S3
        if s3_bucket:
            s3_key = f"fire_cnn/checkpoints/checkpoint_epoch_{epoch + 1}.pth"
            upload_to_s3(checkpoint_path, s3_bucket, s3_key)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            print(f"Best model updated (Val Loss: {val_loss:.6f})")

            if s3_bucket:
                upload_to_s3(best_model_path, s3_bucket, "fire_cnn/best_model.pth")

    print("\n" + "=" * 60)
    print(f"Training complete! Best validation loss: {best_val_loss:.6f}")

    return model


def main():
    """Main training function"""
    # Load environment variables
    load_dotenv()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model
    model = train_model(
        data_dir="finalized_dataset",
        square_size=SQUARE_SIZE,
        batch_size=8,
        learning_rate=0.001,
        num_epochs=50,
        val_split=0.2,
        device=device,
        checkpoint_dir="checkpoints",
        s3_bucket=os.environ.get("S3_BUCKET"),
    )

    return model


if __name__ == "__main__":
    main()
