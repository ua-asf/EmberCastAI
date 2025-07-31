import sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from extract_bands import get_all_fires_squares
import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# CPU Optimization Settings
def setup_cpu_optimization():
    """Configure PyTorch and NumPy to use all available CPU cores"""
    import os
    
    # Get number of CPU cores from SLURM or system
    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    
    # Set PyTorch to use all available threads
    torch.set_num_threads(num_cores)
    
    # Set NumPy to use all available threads
    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ['MKL_NUM_THREADS'] = str(num_cores)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_cores)
    
    print(f"Using {num_cores} CPU cores")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    
    return num_cores

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Convolutional layers for input, forget, cell, and output gates
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Apply convolution
        conv_output = self.conv(combined)
        
        # Split the output into 4 parts for the gates
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)
        
        # Apply activation functions
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # Update cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM implementation"""
    def __init__(self, input_channels, hidden_channels, num_layers=1, kernel_size=3, 
                 batch_first=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        
        # Create ConvLSTM layers
        cell_list = []
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels
            cell_list.append(ConvLSTMCell(cur_input_channels, hidden_channels, kernel_size))
        
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, input_tensor, hidden_state=None):
        """
        Args:
            input_tensor: (b, t, c, h, w) or (t, b, c, h, w) depending on batch_first
            hidden_state: tuple of (h, c) for each layer
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        batch_size, time_steps, channels, height, width = input_tensor.size()
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width)
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(time_steps):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, height, width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append((
                torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device),
                torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device)
            ))
        return init_states

class FireDataset(Dataset):
    """Dataset for fire data"""
    def __init__(self, fire_data, sequence_length=5, target_length=1, transform=None):
        """
        Args:
            fire_data: Dictionary from get_all_fires_squares()
            sequence_length: Number of time steps to use as input
            target_length: Number of time steps to predict
            transform: Optional transform to apply
        """
        self.sequences = []
        self.targets = []
        self.transform = transform
        
        # Process each fire's data
        for fire_name, squares in fire_data.items():
            if len(squares) >= sequence_length + target_length:
                # Create sequences from the squares
                for i in range(len(squares) - sequence_length - target_length + 1):
                    sequence = squares[i:i + sequence_length]
                    target = squares[i + sequence_length:i + sequence_length + target_length]
                    
                    # Convert to tensors
                    sequence_tensor = torch.stack([torch.from_numpy(sq).float() for sq in sequence])
                    target_tensor = torch.stack([torch.from_numpy(tg).float() for tg in target])
                    
                    self.sequences.append(sequence_tensor)
                    self.targets.append(target_tensor)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
            target = self.transform(target)
        
        return sequence, target

class FirePredictor(nn.Module):
    """Complete model for fire prediction"""
    def __init__(self, input_channels, hidden_channels=64, num_layers=2, 
                 sequence_length=5, target_length=1, output_channels=None):
        super(FirePredictor, self).__init__()
        
        self.sequence_length = sequence_length
        self.target_length = target_length
        self.output_channels = output_channels or input_channels
        
        # ConvLSTM layers
        self.conv_lstm = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=3,
            batch_first=True,
            return_all_layers=False
        )
        
        # Output projection layer
        self.output_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=self.output_channels,
            kernel_size=1,
            padding=0
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, channels, height, width = x.size()
        
        # Pass through ConvLSTM
        lstm_out, _ = self.conv_lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_channels, height, width)
        
        # Take the last output
        last_output = lstm_out[0][:, -1, :, :, :]  # (batch_size, hidden_channels, height, width)
        
        # Project to output channels
        output = self.output_conv(last_output)  # (batch_size, output_channels, height, width)
        
        return output

def prepare_data(sequence_length=5, target_length=1, test_size=0.2, random_state=42):
    """Prepare the fire data for training"""
    print("Loading fire data...")
    fire_data = get_all_fires_squares()
    
    if not fire_data:
        print("No fire data found. Please run extract_bands.py first.")
        return None, None, None
    
    print(f"Loaded data for {len(fire_data)} fires")
    
    # Create dataset
    dataset = FireDataset(fire_data, sequence_length, target_length)
    
    if len(dataset) == 0:
        print("No valid sequences found. Try reducing sequence_length or target_length.")
        return None, None, None
    
    print(f"Created {len(dataset)} sequences")
    
    # Split into train and test
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    
    return train_dataset, test_dataset, dataset.sequences[0].shape

def train_model(train_dataset, test_dataset, input_shape, 
                hidden_channels=64, num_layers=2, learning_rate=0.001, 
                num_epochs=5, batch_size=4, device='cpu', s3_bucket=None):
    """Train the ConvLSTM model"""
    
    # Get number of CPU cores for DataLoader workers
    num_workers = min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count())))
    
    # Create data loaders with multiple workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Set to False for CPU-only training
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Initialize model
    input_channels = input_shape[1]  # channels dimension
    model = FirePredictor(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        sequence_length=input_shape[0],
        target_length=1
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, targets[:, 0, :, :, :])  # Take first target
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, targets[:, 0, :, :, :])
                test_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
                f'Train Loss: {avg_train_loss:.6f}, '
                f'Test Loss: {avg_test_loss:.6f}')
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
        }
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs('checkpoints', exist_ok=True)
        
        # Save checkpoint in checkpoints folder
        checkpoint_path = f'checkpoints/fire_predictor_checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch+1} at {checkpoint_path}")
        
        # Upload checkpoint to S3 if bucket is specified
        if s3_bucket:
            upload_checkpoint_to_s3(checkpoint_path, s3_bucket, epoch + 1)
    
    return model, train_losses, test_losses

def plot_training_history(train_losses, test_losses):
    """Plot training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

def upload_to_s3(file_path, bucket_name, s3_key=None):
    try:
        # Initialize S3 client with custom credentials from .env
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_KEY'),
            aws_secret_access_key=os.environ.get('AWS_SECRET'),
            region_name=os.environ.get('AWS_REGION')
        )
        
        # If s3_key is not provided, use the filename
        if s3_key is None:
            s3_key = os.path.basename(file_path)
        
        # Upload file
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"Successfully uploaded {file_path} to s3://{bucket_name}/{s3_key}")
        return True
        
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error uploading to S3: {e}")
        return False

def upload_checkpoint_to_s3(checkpoint_path, bucket_name, epoch):
    s3_key = f"model/checkpoints/fire_predictor_checkpoint_epoch_{epoch}.pth"
    return upload_to_s3(checkpoint_path, bucket_name, s3_key)

def main():
    """Main function to run the complete pipeline"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Setup CPU optimization
    num_cores = setup_cpu_optimization()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    train_dataset, test_dataset, input_shape = prepare_data(
        sequence_length=5,  # Use 5 time steps as input
        target_length=1,    # Predict 1 time step ahead
        test_size=0.2
    )
    
    if train_dataset is None:
        return
    
    print(f"Input shape: {input_shape}")
    
    # Train model
    model, train_losses, test_losses = train_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        input_shape=input_shape,
        hidden_channels=64,
        num_layers=2,
        learning_rate=0.001,
        num_epochs=5,
        batch_size=4,
        device=device,
        s3_bucket=os.environ.get('S3_BUCKET')  # Get from environment variable
    )
    
    # Plot results
    plot_training_history(train_losses, test_losses)
    
    # Save model
    model_path = 'fire_predictor_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'")
    
    # Upload final model to S3 if bucket is specified
    s3_bucket = os.environ.get('S3_BUCKET')
    if s3_bucket:
        upload_to_s3(model_path, s3_bucket, 'model/fire_predictor_model.pth')
    
    return model

def run_model_on_squares(squares):
    """Run the trained model on a list of squares"""
    load_dotenv()  # Load environment variables from .env file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"shape: {squares[0].shape}")
    print(f"shape[0] = {squares[0].shape[0]}")
    print(f"shape[1] = {squares[0].shape[1]}")
    print(f"shape[2] = {squares[0].shape[2]}")
    model = FirePredictor(
        input_channels=squares[0].shape[2],  # Number of channels in the input square
        hidden_channels=64,
        num_layers=2,
        sequence_length=1,  # Single square input
        target_length=1
    ).to(device)

    model.load_state_dict(torch.load('assets/model/fire_predictor_model.pth', map_location=device))

    model.eval()  # Set model to evaluation mode

    results = []
    
    with torch.no_grad():
        for square in squares:
            square_tensor = torch.from_numpy(square).unsqueeze(0).to(device)  # Add batch dimension
            output = model(square_tensor)
            results.append(output.cpu().numpy())
    
    return results

if __name__ == "__main__":
    model = main()