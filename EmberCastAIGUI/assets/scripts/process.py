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

def process(username=None, password=None, wkt_string=None, date_str=None):

# Read terminal inputs
if len(sys.argv) < 4:
    print('Usage: python process.py <username> <password> <wkt_string> [date]')
    sys.exit(1)

# Get user and password credentials from the inputs
username = sys.argv[1]
password = sys.argv[2]

# Get the WKT fire string from the inputs
wkt_string = sys.argv[3]

date_str = sys.argv[4]

output_dir = f'{os.getcwd()}/assets/tmp/{date_str}'

# Function to write status to a shared file
def write_status(message):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f'{output_dir}/status.txt', 'w') as f:
        f.write(message)

write_status(f'Starting processing for WKT...')

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

session = asf.ASFSession()

if username is None or password is None:
    raise ValueError('Please set the EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables')

# Authenticate the session
session.auth_with_creds(username=username, password=password)

# in the format (FIRENAME: (LATMIN, LATMAX, LONGMIN, LONGMAX))
# or            (FIRENAME: (s_min,  n_max,  w_min,   e_max))

date = datetime.strptime(os.path.basename(date_str), date_format_str)

write_status('Searching for relevant Sentinel-1 data...')

# Create assets/tmp/{wkt_string}/data directory if it doesn't exist
if not os.path.isdir(f'{file_path}/data'):
    os.makedirs(f'{file_path}/data')

date_end = date

polygon = f'POLYGON(({extremes[2]:.8} {extremes[1]:.8}, {extremes[2]:.8} {extremes[0]:.8}, {extremes[3]:.8} {extremes[0]:.8}, {extremes[3]:.8} {extremes[1]:.8}, {extremes[2]:.8} {extremes[1]:.8}))'

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

    delta += delta
    options['end'] = options['start']
    options['start'] = (date - timedelta(days=delta)).strftime(date_format_str)

    if delta > 1000:
        raise ValueError(f'No data found for fire on {date}')

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
                results = [result]
                break
            else:
                continue

write_status('Found Sentinel-1 data. Downloading...')

if not os.path.isdir('tmp'):
    os.makedirs('tmp')

if not os.path.isfile(f'tmp/{results[0].properties['sceneName']}.zip'):
    results[0].download(path='tmp', session=session)

write_status('Download complete. Generating geocoded GRDs...')

files = generate_geocoded_grd(results[0].properties['sceneName'], out_dir=f'{file_path}/data')

write_status(f'Geocoded GRD generated. Cropping and scaling data to 20x20 meter resolution...')

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

write_status('Cropping and scaling complete. Merging geotiffs...')

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


input_file = f'{file_path}/data/merged.tiff'
draw_wkt_to_geotiff(polygons, input_file, output_file=f'{file_path}/data/merged_wkt.tiff')

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
    
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    
    # Option 1: Direct CHW allocation (most memory efficient)
    bands_array = np.zeros((num_bands, ysize, xsize), dtype=np.float32)
    
    for band_idx in range(num_bands):
        band = ds.GetRasterBand(band_idx + 1)
        bands_array[band_idx] = band.ReadAsArray()
    
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
    channels, height, width = array.shape
    
    if height % square_size != 0 or width % square_size != 0:
        raise ValueError(f"Array dimensions ({height}, {width}) must be multiples of {square_size}")
    
    num_squares_y = height // square_size
    num_squares_x = width // square_size
    
    squares = []
    for i in range(num_squares_y):
        for j in range(num_squares_x):
            # Extract square preserving all channels
            square = array[:, i * square_size:(i + 1) * square_size, j * square_size:(j + 1) * square_size]
            squares.append(square)

    # Add batch dimension: (num_patches, channels, H, W) -> (num_patches, 1, channels, H, W)
    return torch.from_numpy(np.array(squares)).unsqueeze(1)

write_status('Extracting bands from merged_wkt.tiff...')

bands = extract_bands_from_tiff(f'{file_path}/data/merged_wkt.tiff')
squares = cut_into_squares(bands, square_size=SQUARE_SIZE)

results = []

# Run the model
import torch
from model import SimpleFireCNN

write_status('Loading model and running inference...')

model = SimpleFireCNN()
model.load_state_dict(torch.load(f'{os.getcwd()}/assets/model/fire_predictor_model.pth'))
model.eval()

with torch.no_grad():
    for square in squares:
        results.append(model(square).numpy())

write_status('Model run complete. Stitching results...')

# Stitch the results back together

def stitch_results(squares, square_size=SQUARE_SIZE):
    """Stitch squares back into a single array"""
    num_squares = len(squares)
    if num_squares == 0:
        return np.array([])
    
    # Calculate dimensions
    num_rows = int(np.sqrt(num_squares))
    num_cols = (num_squares + num_rows - 1) // num_rows
    stitched_height = num_rows * square_size
    stitched_width = num_cols * square_size
    
    stitched_array = np.zeros((stitched_height, stitched_width), dtype=squares[0].dtype)
    
    for idx, square in enumerate(squares):
        row = idx // num_cols
        col = idx % num_cols
        y_offset = row * square_size
        x_offset = col * square_size
        stitched_array[y_offset:y_offset + square_size, x_offset:x_offset + square_size] = square.squeeze() * 255
    
    return stitched_array.astype(np.uint8)

stitched_results = stitch_results(results)

# Save the stitched results
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Turn the stitched results into a png
from PIL import Image
def save_as_png(array, output_path):
    """Save a numpy array as a PNG image"""
    img = Image.fromarray(array)
    img.save(output_path)

# Save the original WKT band as an image for comparison
# Convert the WKT band to a u8 int array
print(bands[0])
data = bands[0].astype(np.uint8)
print(data)
save_as_png(data, f'{output_dir}/original.png')

# Save the stitched results as a PNG
save_as_png(stitched_results, f'{output_dir}/output.png')
