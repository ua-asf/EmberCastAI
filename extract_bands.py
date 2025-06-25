import os
import numpy as np
from osgeo import gdal
from datetime import datetime
from tqdm import tqdm

# Date format used by the dataset
date_format_str = '%Y-%m-%d-%H%M'

def get_fires_with_merged_wkt():
    """Get all fires that have merged_wkt.tiff files"""
    fires = {}
    # Search for merged_wkt.tiff files in organized_dataset
    for root, dirs, files in os.walk('organized_dataset'):
        for file in files:
            if file == 'merged_wkt.tiff':
                # Extract fire name from path
                path_parts = root.split(os.sep)
                if 'organized_dataset' in path_parts:
                    fire_idx = path_parts.index('organized_dataset') + 1
                    if fire_idx < len(path_parts):
                        fire_name = path_parts[fire_idx]
                        if fire_name not in fires:
                            fires[fire_name] = []
                        fires[fire_name].append(os.path.join(root, file))
    return fires

def extract_bands_from_tiff(tiff_path):
    """Extract all bands from a GeoTIFF and return as numpy array"""
    ds = gdal.Open(tiff_path)
    if ds is None:
        raise ValueError(f"Could not open {tiff_path}")
    num_bands = ds.RasterCount
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    bands_array = np.zeros((ysize, xsize, num_bands), dtype=np.float32)
    for band_idx in range(num_bands):
        band = ds.GetRasterBand(band_idx + 1)
        data = band.ReadAsArray()
        bands_array[:, :, band_idx] = data
    ds = None
    return bands_array

def expand_to_square(array, target_size=None):
    height, width, bands = array.shape
    if target_size is None:
        max_dim = max(height, width)
        target_size = ((max_dim // 100) + 1) * 100
        if target_size % 2 != 0:
            target_size += 100
    square_array = np.zeros((target_size, target_size, bands), dtype=array.dtype)
    y_offset = (target_size - height) // 2
    x_offset = (target_size - width) // 2
    square_array[y_offset:y_offset + height, x_offset:x_offset + width] = array
    return square_array

def cut_into_squares(array, square_size=100):
    height, width, bands = array.shape
    if height % square_size != 0 or width % square_size != 0:
        raise ValueError(f"Array dimensions ({height}, {width}) must be multiples of {square_size}")
    num_squares_y = height // square_size
    num_squares_x = width // square_size
    squares = []
    for i in range(num_squares_y):
        for j in range(num_squares_x):
            square = array[i * square_size:(i + 1) * square_size, j * square_size:(j + 1) * square_size]
            squares.append(square)
    return squares

def process_fire_data(fire_name, tiff_paths, square_size=100):
    """Process all merged_wkt.tiff files for a fire and return list of squares as numpy arrays"""
    all_squares = []
    for tiff_path in tiff_paths:
        try:
            print(f"Processing {tiff_path}")
            bands_array = extract_bands_from_tiff(tiff_path)
            square_array = expand_to_square(bands_array, target_size=None)
            squares = cut_into_squares(square_array, square_size)
            all_squares.extend(squares)
        except Exception as e:
            print(f"Error processing {tiff_path}: {e}")
            continue
    return all_squares

def get_all_fires_squares(square_size=100):
    """Returns a dictionary: fire_name -> list of squares (numpy arrays)"""
    print("Searching for merged_wkt.tiff files")
    fires = get_fires_with_merged_wkt()
    if not fires:
        print("No merged_wkt.tiff files found. Please run the preprocessing pipeline first.")
        return {}
    print(f"Found {len(fires)} fires with merged_wkt.tiff files:")
    for fire_name, paths in fires.items():
        print(f"  {fire_name}: {len(paths)} files")
    all_fires_data = {}
    for fire_name, tiff_paths in tqdm(fires.items(), desc="Processing fires"):
        print(f"\nProcessing fire: {fire_name}")
        squares = process_fire_data(fire_name, tiff_paths, square_size)
        if squares:
            all_fires_data[fire_name] = squares
    return all_fires_data

if __name__ == "__main__":
    all_fires_data = get_all_fires_squares()
    total_squares = sum(len(sq) for sq in all_fires_data.values())
    print(f"\nExtraction complete!")
    print(f"Total fires processed: {len(all_fires_data)}")
    print(f"Total squares extracted: {total_squares}")
    for fire_name, squares in all_fires_data.items():
        if squares:
            print(f"  {fire_name}: {len(squares)} squares, shape: {squares[0].shape if squares else 'N/A'}")