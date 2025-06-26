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
                num_epochs=50, batch_size=4, device='cpu'):
    """Train the ConvLSTM model"""
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Test Loss: {avg_test_loss:.6f}')
    
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

def main():
    """Main function to run the complete pipeline"""
    
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
        num_epochs=50,
        batch_size=4,
        device=device
    )
    
    # Plot results
    plot_training_history(train_losses, test_losses)
    
    # Save model
    torch.save(model.state_dict(), 'fire_predictor_model.pth')
    print("Model saved as 'fire_predictor_model.pth'")
    
    return model

if __name__ == "__main__":
    model = main()
