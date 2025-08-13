
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import ndimage
from extract_bands import get_all_fires_squares
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def extract_features_from_square(square):
    """
    Extract meaningful features from a single square (100x100x3 numpy array)
    square: numpy array of shape (height, width, channels)
    """
    features = {}
    
    # Basic statistics for each channel (3 bands)
    for i in range(square.shape[2]):
        channel_data = square[:, :, i]
        features[f'band_{i}_mean'] = np.mean(channel_data)
        features[f'band_{i}_std'] = np.std(channel_data)
        features[f'band_{i}_max'] = np.max(channel_data)
        features[f'band_{i}_min'] = np.min(channel_data)
        features[f'band_{i}_median'] = np.median(channel_data)
    
    # Fire-specific features (assuming band 0 is primary fire indicator)
    fire_channel = square[:, :, 0]
    
    # Fire area (count of pixels above threshold)
    fire_threshold = 0.1
    features['fire_area'] = np.sum(fire_channel > fire_threshold)
    features['fire_area_ratio'] = features['fire_area'] / (square.shape[0] * square.shape[1])
    
    # Fire intensity
    features['fire_intensity_mean'] = np.mean(fire_channel)
    features['fire_intensity_max'] = np.max(fire_channel)
    features['fire_intensity_std'] = np.std(fire_channel)
    
    # Spatial features
    features['fire_perimeter'] = calculate_perimeter(fire_channel, fire_threshold)
    features['fire_compactness'] = calculate_compactness(fire_channel, fire_threshold)
    
    # Edge features
    features['fire_edge_density'] = calculate_edge_density(fire_channel, fire_threshold)
    
    # Texture features (simple)
    features['fire_texture_variance'] = calculate_texture_variance(fire_channel)
    
    return features

def calculate_perimeter(fire_channel, threshold=0.1):
    """Calculate approximate fire perimeter"""
    active_pixels = fire_channel > threshold
    if not np.any(active_pixels):
        return 0
    
    # Simple perimeter calculation using edge detection
    eroded = ndimage.binary_erosion(active_pixels)
    perimeter = np.sum(active_pixels) - np.sum(eroded)
    return perimeter

def calculate_compactness(fire_channel, threshold=0.1):
    """Calculate fire compactness (area/perimeter^2)"""
    active_pixels = fire_channel > threshold
    if not np.any(active_pixels):
        return 0
    
    area = np.sum(active_pixels)
    perimeter = calculate_perimeter(fire_channel, threshold)
    
    if perimeter == 0:
        return 0
    
    return area / (perimeter ** 2)

def calculate_edge_density(fire_channel, threshold=0.1):
    """Calculate edge density of fire"""
    active_pixels = fire_channel > threshold
    if not np.any(active_pixels):
        return 0
    
    eroded = ndimage.binary_erosion(active_pixels)
    edge_pixels = np.sum(active_pixels) - np.sum(eroded)
    total_pixels = fire_channel.shape[0] * fire_channel.shape[1]
    
    return edge_pixels / total_pixels

def calculate_texture_variance(fire_channel):
    """Calculate texture variance using simple local variance"""
    # Apply uniform filter to get local mean
    local_mean = ndimage.uniform_filter(fire_channel, size=3)
    local_var = ndimage.uniform_filter(fire_channel**2, size=3) - local_mean**2
    
    return np.mean(local_var)

def prepare_fire_dataset(fire_data, sequence_length=2, target_length=1):
    """
    Prepare dataset for fire growth prediction
    fire_data: dictionary from get_all_fires_squares()
    sequence_length: number of days to use as input
    target_length: number of days to predict
    """
    features_list = []
    targets_list = []
    
    print(f"Processing {len(fire_data)} fires...")
    
    for fire_name, squares in fire_data.items():
        print(f"Processing fire: {fire_name} ({len(squares)} squares)")
        
        if len(squares) >= sequence_length + target_length:
            # Create sequences for this fire
            for i in range(len(squares) - sequence_length - target_length + 1):
                # Extract features from input sequence
                sequence_features = []
                for j in range(sequence_length):
                    features = extract_features_from_square(squares[i + j])
                    sequence_features.extend(list(features.values()))
                
                # Extract target (fire area in next day)
                target_features = extract_features_from_square(squares[i + sequence_length])
                target_value = target_features['fire_area']  # Predict fire area
                
                features_list.append(sequence_features)
                targets_list.append(target_value)
        else:
            print(f"Skipping {fire_name}: only {len(squares)} squares (need {sequence_length + target_length})")
    
    return np.array(features_list), np.array(targets_list)

def analyze_feature_correlations(features, targets):
    """Analyze correlations between features and target to identify data leakage"""
    print("=" * 50)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 50)
    
    correlations = []
    for i in range(features.shape[1]):
        corr = abs(np.corrcoef(features[:, i], targets)[0, 1])
        correlations.append(corr)
    
    # Sort features by correlation
    sorted_indices = np.argsort(correlations)[::-1]
    
    print("Top 10 features with highest correlation to target:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"{i+1:2d}. feature_{idx}: {correlations[idx]:.4f}")
    
    # Identify potentially leaky features
    leaky_threshold = 0.8
    leaky_features = [i for i, corr in enumerate(correlations) if corr > leaky_threshold]
    
    if leaky_features:
        print(f"\nWARNING: Found {len(leaky_features)} potentially leaky features (correlation > {leaky_threshold}):")
        for idx in leaky_features:
            print(f"  - feature_{idx}: {correlations[idx]:.4f}")
    else:
        print(f"\nNo features with correlation > {leaky_threshold}")
    
    return correlations

def remove_leaky_features(features, targets, threshold=0.8):
    """Remove features that are too correlated with the target (potential data leakage)"""
    correlations = []
    for i in range(features.shape[1]):
        corr = abs(np.corrcoef(features[:, i], targets)[0, 1])
        correlations.append(corr)
    
    # Keep features with correlation below threshold
    good_features = [i for i, corr in enumerate(correlations) if corr < threshold]
    
    print(f"\nRemoving {features.shape[1] - len(good_features)} leaky features...")
    print(f"Keeping {len(good_features)} features (correlation < {threshold})")
    
    return features[:, good_features], good_features

def train_random_forest_model(features, targets, test_size=0.2, random_state=42):
    """
    Train Random Forest model for fire prediction with improved parameters
    """
    print("=" * 60)
    print("TRAINING RANDOM FOREST FOR FIRE PREDICTION")
    print("=" * 60)
    
    # Analyze feature correlations first
    correlations = analyze_feature_correlations(features, targets)
    
    # Remove leaky features
    features_clean, kept_features = remove_leaky_features(features, targets, threshold=0.8)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_clean, targets, test_size=test_size, random_state=random_state
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features per sample: {X_train.shape[1]}")
    
    # Initialize and train Random Forest with improved parameters
    rf_model = RandomForestRegressor(
        n_estimators=50,       # Fewer trees to reduce overfitting
        max_depth=5,           # Shorter trees to prevent overfitting
        min_samples_split=5,   # More samples required to split
        min_samples_leaf=3,    # More samples required in leaves
        max_features='sqrt',   # Use sqrt of features per split
        bootstrap=True,        # Use bootstrapping
        random_state=random_state,
        n_jobs=-1             # Use all CPU cores
    )
    
    print("\nTraining Random Forest with improved parameters...")
    rf_model.fit(X_train, y_train)
    
    # Cross-validation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(rf_model, features_clean, targets, cv=3, scoring='r2')
    print(f"\nCross-validation R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Make predictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n" + "=" * 40)
    print("MODEL PERFORMANCE")
    print("=" * 40)
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Overfitting gap (Train R² - Test R²): {train_r2 - test_r2:.4f}")
    
    # Feature importance analysis
    feature_importance = rf_model.feature_importances_
    feature_names = [f"feature_{kept_features[i]}" for i in range(len(kept_features))]
    
    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    
    print(f"\n" + "=" * 40)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 40)
    for i in range(min(15, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"{i+1:2d}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    # Check for feature dominance
    max_importance = np.max(feature_importance)
    if max_importance > 0.3:
        print(f"\nWARNING: Feature dominance detected!")
        print(f"Most important feature: {max_importance:.4f} ({max_importance*100:.1f}% of predictions)")
    
    return rf_model, {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'feature_importance': feature_importance,
        'feature_names': feature_names,
        'kept_features': kept_features,
        'correlations': correlations
    }

def plot_results(results, save_plot=True):
    """Plot training results and feature importance with improved metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Feature importance
    top_features = 10
    sorted_indices = np.argsort(results['feature_importance'])[-top_features:]
    
    feature_names = [results['feature_names'][i] for i in sorted_indices]
    importance_values = [results['feature_importance'][i] for i in sorted_indices]
    
    ax1.barh(range(len(feature_names)), importance_values)
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names)
    ax1.set_xlabel('Feature Importance')
    ax1.set_title(f'Top {top_features} Most Important Features')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance metrics
    metrics = ['Train MSE', 'Test MSE', 'Train R²', 'Test R²']
    values = [results['train_mse'], results['test_mse'], results['train_r2'], results['test_r2']]
    
    bars = ax2.bar(metrics, values, color=['blue', 'red', 'green', 'orange'])
    ax2.set_ylabel('Score')
    ax2.set_title('Model Performance Metrics')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 3: Cross-validation results
    cv_metrics = ['CV R² Mean', 'CV R² Std']
    cv_values = [results['cv_r2_mean'], results['cv_r2_std']]
    
    bars_cv = ax3.bar(cv_metrics, cv_values, color=['purple', 'brown'])
    ax3.set_ylabel('Score')
    ax3.set_title('Cross-Validation Results')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars_cv, cv_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 4: Overfitting analysis
    overfitting_gap = results['train_r2'] - results['test_r2']
    overfitting_data = [results['train_r2'], results['test_r2'], overfitting_gap]
    overfitting_labels = ['Train R²', 'Test R²', 'Overfitting Gap']
    colors = ['green', 'orange', 'red']
    
    bars_overfit = ax4.bar(overfitting_labels, overfitting_data, color=colors)
    ax4.set_ylabel('R² Score')
    ax4.set_title('Overfitting Analysis')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars_overfit, overfitting_data):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('random_forest_improved_results.png', dpi=300, bbox_inches='tight')
        print("Improved results plot saved as 'random_forest_improved_results.png'")
    
    plt.show()
    
    # Print summary
    print(f"\n" + "=" * 50)
    print("IMPROVED MODEL SUMMARY")
    print("=" * 50)
    print(f"Cross-validation R²: {results['cv_r2_mean']:.3f} (±{results['cv_r2_std']:.3f})")
    print(f"Overfitting gap: {overfitting_gap:.3f}")
    print(f"Features used: {len(results['kept_features'])} (after removing leaky features)")
    
    if overfitting_gap < 0.1:
        print("Good: Low overfitting gap")
    else:
        print("Warning: High overfitting gap detected")
    
    if results['cv_r2_mean'] > 0.6:
        print("Good: Reasonable cross-validation performance")
    else:
        print("Warning: Low cross-validation performance")

def predict_fire_growth(model, fire_data, fire_name, day_index, kept_features):
    """
    Make prediction for a specific fire and day using the same feature cleaning as training
    """
    if fire_name not in fire_data:
        print(f"Fire '{fire_name}' not found in dataset")
        return None
    
    squares = fire_data[fire_name]
    if day_index + 2 >= len(squares):  # Need 2 days for prediction
        print(f"Not enough data for fire '{fire_name}' at day {day_index}")
        return None
    
    # Extract features from the two days (same as training)
    features = []
    for i in range(2):  # Use 2 days as input
        day_features = extract_features_from_square(squares[day_index + i])
        features.extend(list(day_features.values()))
    
    # Apply the same feature cleaning as during training
    features = np.array(features).reshape(1, -1)  # Make it 2D
    features_clean = features[:, kept_features]  # Keep only the features the model was trained on
    
    # Make prediction
    prediction = model.predict(features_clean)[0]
    
    # Get actual value for comparison
    actual_features = extract_features_from_square(squares[day_index + 2])
    actual_value = actual_features['fire_area']
    
    print(f"\nPrediction for {fire_name} (Day {day_index} -> Day {day_index + 2}):")
    print(f"Predicted fire area: {prediction:.1f} pixels")
    print(f"Actual fire area: {actual_value:.1f} pixels")
    print(f"Error: {abs(prediction - actual_value):.1f} pixels")
    
    return prediction, actual_value

def analyze_feature_32(features, targets, fire_data):
    """Deep analysis of feature 32 to understand what it represents"""
    print("=" * 60)
    print("DEEP ANALYSIS OF FEATURE 32")
    print("=" * 60)
    
    # Get feature 32 values
    feature_32_values = features[:, 32]
    
    # Basic statistics
    print(f"Feature 32 Statistics:")
    print(f"  Mean: {np.mean(feature_32_values):.4f}")
    print(f"  Std: {np.std(feature_32_values):.4f}")
    print(f"  Min: {np.min(feature_32_values):.4f}")
    print(f"  Max: {np.max(feature_32_values):.4f}")
    print(f"  Range: {np.max(feature_32_values) - np.min(feature_32_values):.4f}")
    
    # Correlation with target
    correlation = np.corrcoef(feature_32_values, targets)[0, 1]
    print(f"\nCorrelation with target: {correlation:.4f}")
    
    # Check if it's the target itself or very similar
    target_correlation = np.corrcoef(feature_32_values, targets)[0, 1]
    if abs(target_correlation) > 0.95:
        print("WARNING: Feature 32 is almost identical to the target!")
        print("This is likely data leakage - the model is using future information")
    
    # Analyze feature 32 across different fires
    print(f"\nFeature 32 values by fire:")
    fire_names = list(fire_data.keys())
    for i, fire_name in enumerate(fire_names):
        if len(fire_data[fire_name]) >= 3:  # Need at least 3 days
            # Get the sequence indices for this fire
            fire_start_idx = i * (len(fire_data[fire_name]) - 2)  # Approximate
            fire_end_idx = min((i + 1) * (len(fire_data[fire_name]) - 2), len(feature_32_values))
            
            if fire_start_idx < len(feature_32_values):
                fire_feature_32 = feature_32_values[fire_start_idx:fire_end_idx]
                fire_targets = targets[fire_start_idx:fire_end_idx]
                
                print(f"  {fire_name}:")
                print(f"Feature 32: {fire_feature_32}")
                print(f"Targets: {fire_targets}")
                print(f"Correlation: {np.corrcoef(fire_feature_32, fire_targets)[0, 1]:.4f}")
    
    # Try to identify what feature 32 represents
    print(f"\nFeature 32 Analysis:")
    print(f"  Based on the feature extraction code, feature 32 might be:")
    
    # Calculate which feature index 32 corresponds to in the sequence
    sequence_length = 2  # We use 2 days as input
    features_per_day = len(extract_features_from_square(np.zeros((100, 100, 3))))  # ~25 features
    
    day_index = 32 // features_per_day
    feature_in_day = 32 % features_per_day
    
    print(f"  - Day {day_index + 1} (input sequence)")
    print(f"  - Feature {feature_in_day} within that day")
    
    # Map feature index to actual feature name
    sample_features = extract_features_from_square(np.zeros((100, 100, 3)))
    feature_names = list(sample_features.keys())
    
    if feature_in_day < len(feature_names):
        print(f"  - Likely represents: {feature_names[feature_in_day]}")
    else:
        print(f"  - Feature index {feature_in_day} is out of range")
    
    # Check if it's fire area from previous day
    if "fire_area" in feature_names and feature_names.index("fire_area") == feature_in_day:
        print("CONFIRMED: Feature 32 is 'fire_area' from previous day!")
    
    return feature_32_values, correlation

def main():
    """Main function to run the complete Random Forest pipeline"""
    print("Loading fire data from extract_bands.py...")
    fire_data = get_all_fires_squares()
    
    if not fire_data:
        print("No fire data found. Please run extract_bands.py first.")
        return
    
    print(f"Successfully loaded data for {len(fire_data)} fires")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    features, targets = prepare_fire_dataset(fire_data, sequence_length=2, target_length=1)
    
    if len(features) == 0:
        print("No valid sequences found. Try reducing sequence_length.")
        return
    
    print(f"Created {len(features)} training samples")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Target range: {targets.min():.1f} to {targets.max():.1f}")
    
    # Analyze feature 32 specifically
    if features.shape[1] > 32:
        analyze_feature_32(features, targets, fire_data)
    
    # Train model
    model, results = train_random_forest_model(features, targets)
    
    # Plot results
    plot_results(results)
    
    # Example predictions
    print(f"\n" + "=" * 40)
    print("EXAMPLE PREDICTIONS")
    print("=" * 40)
    
    # Make predictions for first few fires
    fire_names = list(fire_data.keys())
    for i, fire_name in enumerate(fire_names[:3]):  # Test first 3 fires
        if len(fire_data[fire_name]) >= 2:  # Need at least 3 days
            predict_fire_growth(model, fire_data, fire_name, 0, results['kept_features'])
    
    # Save model
    import joblib
    model_data = {
        'model': model,
        'kept_features': results['kept_features'],
        'feature_names': results['feature_names'],
        'results': results
    }
    joblib.dump(model_data, 'fire_prediction_random_forest.pkl')
    print(f"\nModel saved as 'fire_prediction_random_forest.pkl'")
    print(f"Model includes {len(results['kept_features'])} features for predictions")
    
    return model, results

if __name__ == "__main__":
    model, results = main()
