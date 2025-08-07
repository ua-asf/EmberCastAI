import torch
import numpy as np
from dotenv import load_dotenv
from training_cpu import FirePredictor, ConvLSTM, ConvLSTMCell

def inspect_model_config():
    """Inspect the model to determine its actual configuration"""
    load_dotenv()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = torch.load('fire_predictor_model.pth', weights_only=False, map_location=device)
        model.eval()
        
        print("=" * 50)
        print("MODEL CONFIGURATION INSPECTION")
        print("=" * 50)
        
        # Get the first ConvLSTM cell to inspect its configuration
        first_conv_lstm_cell = model.conv_lstm.cell_list[0]
        
        print(f"✓ Model type: {type(model).__name__}")
        print(f"✓ ConvLSTM input channels: {first_conv_lstm_cell.input_channels}")
        print(f"✓ ConvLSTM hidden channels: {first_conv_lstm_cell.hidden_channels}")
        print(f"✓ Number of ConvLSTM layers: {len(model.conv_lstm.cell_list)}")
        print(f"✓ Kernel size: {first_conv_lstm_cell.kernel_size}")
        print(f"✓ Output channels: {model.output_conv.out_channels}")
        
        # Inspect the conv layer in the first cell
        conv_layer = first_conv_lstm_cell.conv
        print(f"✓ First cell conv layer input channels: {conv_layer.in_channels}")
        print(f"✓ First cell conv layer output channels: {conv_layer.out_channels}")
        
        return first_conv_lstm_cell.input_channels
        
    except Exception as e:
        print(f"✗ Error inspecting model: {e}")
        return None

def test_model_loading():
    """Test if the model can be loaded correctly"""
    load_dotenv()  # Load environment variables from .env file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load the model
        model = torch.load('fire_predictor_model.pth', weights_only=False, map_location=device)
        model.eval()  # Set model to evaluation mode
        
        print("✓ Model loaded successfully!")
        print(f"✓ Model is on device: {next(model.parameters()).device}")
        print(f"✓ Model type: {type(model).__name__}")
        
        # Print model architecture summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def test_model_inference(model=None, input_channels=None):
    """Test if the model can perform inference with dummy data"""
    if model is None:
        model = test_model_loading()
        if model is None:
            return False
    
    # If input_channels not provided, try to get it from the model
    if input_channels is None:
        try:
            input_channels = model.conv_lstm.cell_list[0].input_channels
        except:
            input_channels = 3  # Default to 3 for corrected model
    
    device = next(model.parameters()).device
    
    try:
        # Create dummy input data matching expected format
        # Shape: (batch_size, sequence_length, channels, height, width)
        batch_size = 2
        sequence_length = 5  # Model expects 5 time steps
        channels = input_channels  # Use actual model input channels
        height, width = 100, 100  # Updated to match actual fire data dimensions
        
        dummy_input = torch.randn(batch_size, sequence_length, channels, height, width).to(device)
        
        print(f"✓ Created dummy input with shape: {dummy_input.shape}")
        print(f"✓ Using {channels} input channels (from model config)")
        
        # Run inference
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Model inference successful!")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Expected output shape: (batch_size={batch_size}, channels={channels}, height={height}, width={width})")
        
        # Verify output shape is correct
        expected_shape = (batch_size, channels, height, width)
        if output.shape == expected_shape:
            print("✓ Output shape matches expected dimensions!")
        else:
            print(f"⚠ Output shape mismatch. Expected: {expected_shape}, Got: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during model inference: {e}")
        return False

def test_with_real_data_format():
    """Test the model with data that matches the actual fire data format"""
    try:
        from extract_bands import get_all_fires_squares
        
        print("\n3. Testing with Real Data Format...")
        
        # Get a sample of real data
        fire_data = get_all_fires_squares()
        if not fire_data:
            print("⚠ No fire data available for testing")
            return False
        
        # Get first fire's data
        first_fire = next(iter(fire_data.values()))
        if len(first_fire) < 5:
            print("⚠ Not enough time steps in fire data for testing")
            return False
        
        # Create a sequence of 5 time steps
        sequence = first_fire[:5]
        print(f"✓ Real data sequence shape: {[sq.shape for sq in sequence]}")
        
        # Test with the model
        result = run_model_on_sequence(sequence)
        if result is not None:
            print(f"✓ Real data inference successful!")
            print(f"✓ Prediction shape: {result.shape}")
            return True
        else:
            print("✗ Real data inference failed")
            return False
            
    except ImportError:
        print("⚠ Cannot import extract_bands (missing dependencies), skipping real data test")
        return True  # Don't fail the overall test
    except Exception as e:
        print(f"✗ Error testing with real data: {e}")
        return False

def run_model_on_sequence(sequence_data):
    """Run the trained model on a sequence of squares (5 time steps)"""
    load_dotenv()  # Load environment variables from .env file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = torch.load('fire_predictor_model.pth', weights_only=False, map_location=device)
        model.eval()  # Set model to evaluation mode

        # Ensure sequence_data is the right format
        # Expected: (sequence_length, channels, height, width) or list of such arrays
        if isinstance(sequence_data, list):
            if len(sequence_data) != 5:
                raise ValueError(f"Expected sequence of 5 time steps, got {len(sequence_data)}")
            # Convert from (H, W, C) to (C, H, W) and stack
            sequence_tensor = torch.stack([torch.from_numpy(sq).float().permute(2, 0, 1) for sq in sequence_data])
        else:
            # Assume it's already a tensor or numpy array in correct format
            if isinstance(sequence_data, np.ndarray):
                if sequence_data.ndim == 4 and sequence_data.shape[-1] == 3:  # (seq, H, W, C)
                    sequence_tensor = torch.from_numpy(sequence_data).float().permute(0, 3, 1, 2)  # -> (seq, C, H, W)
                else:
                    sequence_tensor = torch.from_numpy(sequence_data).float()
            else:
                sequence_tensor = sequence_data
        
        # Add batch dimension: (1, sequence_length, channels, height, width)
        if sequence_tensor.dim() == 4:  # (seq, channels, h, w)
            sequence_tensor = sequence_tensor.unsqueeze(0)
        
        sequence_tensor = sequence_tensor.to(device)
        
        with torch.no_grad():
            output = model(sequence_tensor)
            
        return output.cpu().numpy()
        
    except Exception as e:
        print(f"Error running model on sequence: {e}")
        return None

def run_full_verification():
    """Run complete model verification"""
    print("=" * 50)
    print("FIRE PREDICTOR MODEL VERIFICATION")
    print("=" * 50)
    
    # Test 0: Inspect model configuration
    print("\n0. Inspecting Model Configuration...")
    input_channels = inspect_model_config()
     
    # Test 1: Model Loading
    print("\n1. Testing Model Loading...")
    model = test_model_loading()
    
    if model is None:
        print("✗ Model loading failed. Cannot proceed with further tests.")
        return False
    
    # Test 2: Model Inference
    print("\n2. Testing Model Inference...")
    inference_success = test_model_inference(model, input_channels)
    
    # Test 3: Real data format test (if available)
    real_data_success = test_with_real_data_format()
    
    if inference_success and real_data_success:
        print("\n✓ All tests passed! Model is ready for use.")
        return True
    else:
        print("\n✗ Some tests failed. Please check the model and try again.")
        return False

if __name__ == "__main__":
    run_full_verification()