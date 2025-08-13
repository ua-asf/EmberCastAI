import os
import sys
from datetime import datetime, timedelta
import asf_search as asf
import math

sys.path.append('..')

import torch

from ship import generate_geocoded_grd
from geo import crop_and_scale_to_20x20, haversine_distance, draw_wkt_to_geotiff, merge_geotiffs
from shapely.geometry import Polygon

# Define the date format string
date_format_str = '%Y-%m-%dT%H:%M:%S.%f'

# Read terminal inputs
if len(sys.argv) < 4:
    print('Usage: python process.py <username> <password> <wkt_string> [date]')
    sys.exit(1)

print(f'{sys.argv}')

# Get user and password credentials from the inputs
username = sys.argv[1]
password = sys.argv[2]

# Get the WKT fire string from the inputs
wkt_string = sys.argv[3]

# Create a hash for the WKT string to use as a directory name
import hashlib
hash = hashlib.sha256(wkt_string.encode()).hexdigest()
file_path = f'assets/tmp/{hash}'

# GDAL configuration
from osgeo import gdal
gdal.UseExceptions()

# The width/height of the pixels in the SAR data, in meters
PIXEL_SIZE = 20

# The width/height of the square in the SAR data, in pixels (as referenced above)
SQUARE_SIZE = 100


lat_min = 90
lat_max = -90
long_min = 180
long_max = -180

extremes = (lat_min, lat_max, long_min, long_max)

print(f'Getting extremes for WKT: {wkt_string}')

# Remove the 'POLYGON ((' and '))\n' from the line
wkt_string = wkt_string.removeprefix('"').removeprefix('POLYGON').removeprefix(' ').removeprefix('((').removesuffix('"').removesuffix('))')
coords = wkt_string.split(', ')
for line in coords:
    long, lat = line.split(' ')
    lat = float(lat)
    long = float(long)

    # Update the extremes if necessary
    if lat < lat_min:
        lat_min = lat
    if lat > lat_max:
        lat_max = lat
    if long < long_min:
        long_min = long
    if long > long_max:
        long_max = long                

    # Scale the extremes to be a multiple of SQUARE_SIZE * PIXEL_SIZE meters, in a square

    # Get the distance of latitude and longitude
    lat_dist = haversine_distance(lat_min, long_min, lat_max, long_min)
    long_dist = haversine_distance(lat_min, long_min, lat_min, long_max)

    # Get the larger of the two differences
    width = max(lat_dist, long_dist)

    # Get the square size in meters
    square_size = SQUARE_SIZE * PIXEL_SIZE

    # Find the nearest multiple of SQUARE_SIZE * PIXEL_SIZE meters + some extra space
    new_width = (math.ceil(width / square_size) + 3) * square_size

    print(f'New width: {new_width}')

    # Adjust the extremes up and down to fit the new width, back into lat/long
    # Get the difference between the new width and the old width
    new_lat_diff = (new_width - lat_dist) / 2
    new_long_diff = (new_width - long_dist) / 2

    r_earth = 6378137

    # Adjust the extremes up and down to fit the new width, back into lat/long
    lat_min = lat_min - (new_lat_diff / r_earth) * (180 / math.pi)
    lat_max = lat_max + (new_lat_diff / r_earth) * (180 / math.pi)
    long_min = long_min - (new_long_diff / r_earth) * (180 / math.pi)
    long_max = long_max + (new_long_diff / r_earth) * (180 / math.pi)

    # Update the extremes for the fire
    extremes = (lat_min, lat_max, long_min, long_max)

print(f'Extremes: {extremes}')

session = asf.ASFSession()

if username is None or password is None:
    raise ValueError('Please set the EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables')

# Authenticate the session
session.auth_with_creds(username=username, password=password)

# in the format (FIRENAME: (LATMIN, LATMAX, LONGMIN, LONGMAX))
# or            (FIRENAME: (s_min,  n_max,  w_min,   e_max))

date = sys.argv[4]
date = datetime.strptime(os.path.basename(date), date_format_str)

# Create assets/tmp/{wkt_string}/data directory if it doesn't exist
if not os.path.isdir(f'{file_path}/data'):
    os.makedirs(f'{file_path}/data')

date_end = date

polygon = f'POLYGON(({extremes[2]:.8} {extremes[1]:.8}, {extremes[2]:.8} {extremes[0]:.8}, {extremes[3]:.8} {extremes[0]:.8}, {extremes[3]:.8} {extremes[1]:.8}, {extremes[2]:.8} {extremes[1]:.8}))'

print(f'Polygon: {polygon}')

delta = 36

date_start = date - timedelta(days=delta)

options = {
    'dataset': 'SENTINEL-1',
    'intersectsWith': polygon,
    'polarization': ['VV+VH'],
    'processingLevel': 'GRD_HD',
    'start': date_start.strftime(date_format_str),
    'end': date_end.strftime(date_format_str),
}

results = []

while len(results) == 0:
    results = asf.geo_search(**options)

    print(f'Found {len(results)} results')

    delta += delta
    options['end'] = options['start']
    options['start'] = (date - timedelta(days=delta)).strftime(date_format_str)

    if delta > 1000:
        raise ValueError(f'No data found for fire on {date}')

    print(f'Start: {options["start"]}, End: {options["end"]}')

    if len(results) > 0:
        # Sort by date
        results.sort(key=lambda x: datetime.strptime(x.properties['stopTime'], '%Y-%m-%dT%H:%M:%S%fZ'))                    
        
        # Ensure the polygon is within the bounds of the fire
        while results:
            # Get the first result
            result = results.pop()
            candidate = result.geometry['coordinates'][0]

            target = polygon.split('((')[1].split('))')[0].split(', ')
            # Split the coordinates into lat/long
            target = [coord.split(' ') for coord in target]
            # Convert the coordinates to floats
            target = [[float(coord[0]), float(coord[1])] for coord in target]
            
            # Convert the coordinates to a polygon
            candidate_polygon = Polygon(candidate)
            fire_polygon = Polygon(target)

            # Check if the coordinates are within the bounds of the fire
            # The order goes [0] = top left, [1] = top right, [2] = bottom right, [3] = bottom left
            if candidate_polygon.contains_properly(fire_polygon):
                print(f'Found {result.properties['sceneName']}')
                print(f'Candidate polygon: {candidate_polygon}')
                print(f'Fire polygon: {fire_polygon}')
                results = [result]
                break
            else:
                continue

if not os.path.isdir('tmp'):
    os.makedirs('tmp')

if not os.path.isfile(f'tmp/{results[0].properties['sceneName']}.zip'):
    results[0].download(path='tmp', session=session)

files = generate_geocoded_grd(results[0].properties['sceneName'], out_dir=f'{file_path}/data')

print(f'Files: {files}')

# Perform translation for 20x20 pixel sizes
if files:
    for geotiff in files:
        crop_and_scale_to_20x20(
            input_tiff_path=geotiff[0], 
            output_tiff_path=geotiff[0],
            nw_latlon=(extremes[1], extremes[2]), 
            sw_latlon=(extremes[0], extremes[2]), 
            se_latlon=(extremes[0], extremes[3]), 
            ne_latlon=(extremes[1], extremes[3]),
            pixel_size=PIXEL_SIZE,
            square_size=SQUARE_SIZE,
        )

# Merge all geotiffs into a single geotiff
# Get the SAR data files (.tiff)
sar_files = [file for file in os.listdir(f'{file_path}/data') if file.endswith('.tiff') and file != 'merged.tiff' and file != 'merged_wkt.tiff']

# Append the file path to each file
sar_files = [f'{file_path}/data/{file}' for file in sar_files]

# Sort such that the vv band is first, then the vh band
if 'vh' in sar_files[0]:
    sar_files.reverse()

# Merge the files into a single geotiff
merge_geotiffs(sar_files, output_file=f'{file_path}/data/merged.tiff')

polygons = [f'POLYGON(({wkt_string}))']

print(f'Merging polygons: {polygons}')

input_file = f'{file_path}/data/merged.tiff'
draw_wkt_to_geotiff(polygons, input_file, output_file=f'{file_path}/data/merged_wkt.tiff')

print(f'Finished processing {wkt_string}. Ready for model usage.')

# Cut tiff into squares of SQUARE_SIZExSQUARE_SIZE pixels

import os
import numpy as np
from datetime import datetime

# Date format used by the dataset
date_format_str = '%Y-%m-%d-%H%M'

def extract_bands_from_tiff(tiff_path):
    """Extract all bands from a GeoTIFF and return as numpy array"""
    ds = gdal.Open(tiff_path)
    if ds is None:
        raise ValueError(f"Could not open {tiff_path}")
    num_bands = ds.RasterCount
    print(f'Found {num_bands} bands!')
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
        target_size = ((max_dim // SQUARE_SIZE) + 1) * SQUARE_SIZE
        if target_size % 2 != 0:
            target_size += SQUARE_SIZE
    square_array = np.zeros((target_size, target_size, bands), dtype=array.dtype)
    y_offset = (target_size - height) // 2
    x_offset = (target_size - width) // 2
    square_array[y_offset:y_offset + height, x_offset:x_offset + width] = array
    return square_array

def cut_into_squares(array, square_size=SQUARE_SIZE):
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

    return torch.from_numpy(np.array(squares))

bands = extract_bands_from_tiff(f'{file_path}/data/merged_wkt.tiff')
squares = cut_into_squares(bands, square_size=SQUARE_SIZE)

print(squares.shape)
print(type(squares))

results = []

# Run the model
import joblib
from sklearn.ensemble import RandomForestRegressor
model = joblib.load(f'{os.getcwd()}/assets/model/fire_prediction_random_forest.pkl')['model']

print(model)

for square in squares:
    results.append(model.predict(square))

print('Model run complete. Results:', len(results))

# Stitch the results back together

def stitch_results(squares, square_size=SQUARE_SIZE):
    """Stitch squares back into a single array"""
    num_squares = len(squares)
    if num_squares == 0:
        return np.array([])
    
    # Calculate dimensions
    num_rows = int(np.sqrt(num_squares))
    num_cols = (num_squares + num_rows - 1) // num_rows  # Ceiling division
    stitched_height = num_rows * square_size
    stitched_width = num_cols * square_size
    
    stitched_array = np.zeros((stitched_height, stitched_width, squares[0].shape[2]), dtype=squares[0].dtype)
    
    for idx, square in enumerate(squares):
        row = idx // num_cols
        col = idx % num_cols
        y_offset = row * square_size
        x_offset = col * square_size
        stitched_array[y_offset:y_offset + square_size, x_offset:x_offset + square_size] = square
    
    return stitched_array

print('Stitching results back together...')
stitched_results = stitch_results(results)

print(f'Stitched results shape: {stitched_results.shape}')
# Save the stitched results
output_dir = f'assets/tmp/{date}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, f'results.png')
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(output_file, stitched_results.shape[1], stitched_results.shape[0], stitched_results.shape[2], gdal.GDT_Float32)
if out_ds is None:
    raise ValueError(f"Could not create output file {output_file}")

# Turn the stitched results into a png
from PIL import Image
def save_as_png(array, output_path):
    """Save a numpy array as a PNG image"""
    if array.ndim == 3 and array.shape[2] == 1:
        # Single band, convert to grayscale
        array = array[:, :, 0]
    elif array.ndim == 3:
        # Multi-band, convert to RGB
        array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        raise ValueError("Array must be 2D or 3D with shape (height, width) or (height, width, bands)")
    
    img = Image.fromarray(array)
    img.save(output_path)

print(f'Saving stitched results to {output_file}')
# Data should always be 3 bands, so we can save it as RGB
save_as_png(stitched_results, f'assets/tmp/{date}/output.png')
