import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import asf_search as asf
import math
from ship import generate_cropped_geocode_grd
from geo import crop_and_scale_to_20x20, haversine_distance, draw_wkt_to_geotiff
from shapely.geometry import Polygon

# GDAL configuration
from osgeo import gdal
gdal.UseExceptions()

# Dates used by the dataset are usually in the format YYYYMMDD
date_format_str = '%Y-%m-%d'

# Fires is a dict of each fire, with each fire being a 
# list of the files associated with that fire, sorted
# by date
fires = dict()

# Grab all fires (they are organized as organized_dataset/<FIRENAME>/<FILES>)
dirnames = os.listdir('organized_dataset')
for dirname in dirnames:
    # Check if the directory is a fire
    if os.path.isdir(os.path.join('organized_dataset', dirname)):
        try:
            # Get all folder in the directory
            dates = os.listdir(os.path.join('organized_dataset', dirname))
            # Purge any 'unknown date' folders
            dates = [date for date in dates if date != 'UnknownDate' and date != 'data']
            # Sort the files by date
            dates.sort(key=lambda x: datetime.strptime(x, date_format_str))
            # Skip any fires with less than 2 days of data
            if len(dates) < 2:
                print(f'Skipping {dirname} with {len(dates)} days of data')
                continue
            # Add the fire to the fires dict
            fires[dirname] = [os.path.join('organized_dataset', dirname, file) for file in dates]
        except Exception as e:
            print(f'Error with {dirname}: {e}')
            continue

# Purge any fires not in the fires dict
for fire in dirnames:
    if fire not in fires:
        # Check if the directory is a fire
        if os.path.isdir(os.path.join('organized_dataset', fire)):
            # Remove the directory
            os.system(f'rm -rf organized_dataset/{fire}')
            print(f'Removed {fire}')

print(f'Found {len(fires)} fires')

# Purge empty fires
fires = {fire: data for fire, data in fires.items() if len(data) > 0}

for fire, data in fires.items():
    print(f'{fire}: {data}')

# Correct fires such that each pixel is 20m


# Find the lat/long extremes for each fire

# Extremes stores the lat/long extremes for each fire
# in the format (FIRENAME: (LATMIN, LATMAX, LONGMIN, LONGMAX))
extremes = fires.copy()

for fire_name, data in fires.items():
    for day in data:
        for file in os.listdir(day):
            # Only open the .wkt files
            if not file.endswith('.wkt'):
                continue

            lat_max = -90
            lat_min = 90
            long_max = -180
            long_min = 180

            # Parse the lat/long from the file
            with open(f'{day}/{file}', 'r') as f:
                print(f'{day}/{file}')
                # Every line is a polygon
                for line in f.readlines():
                    # Remove the 'POLYGON ((' and '))\n' from the line
                    line = line.removeprefix('POLYGON ((').removesuffix('))\n')
                    coords = line.split(', ')
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

                

                # Scale the extremes to be a multiple of 2000x2000 meters, in a square
                # Get the distance of latitude and longitude
                lat_dist = haversine_distance(lat_min, long_min, lat_max, long_min)
                long_dist = haversine_distance(lat_min, long_min, lat_min, long_max)

                # Get the larger of the two differences
                width = max(lat_dist, long_dist)

                # Find the nearest multiple of 2000 meters
                new_width = math.ceil(width / 1000) * 1000

                # Adjust the extremes up and down to fit the new width, back into lat/long
                # Get the difference between the new width and the old width
                new_lat_diff = (new_width - lat_dist) / 2
                new_long_diff = (new_width - long_dist) / 2

                r_earth = 6378000

                # Adjust the extremes up and down to fit the new width, back into lat/long
                lat_min = lat_min - (new_lat_diff / r_earth) * (180 / math.pi)
                lat_max = lat_max + (new_lat_diff / r_earth) * (180 / math.pi)
                long_min = long_min - (new_long_diff / r_earth) * (180 / math.pi)
                long_max = long_max + (new_long_diff / r_earth) * (180 / math.pi)

                # Update the extremes for the fire
                extremes[fire_name] = (lat_min, lat_max, long_min, long_max)

# Print the extremes for each fire
for fire, data in extremes.items():
    print(f'{fire}: {data}')

total_data = 0

for fire, data in fires.items():
    data_len = len(data)
    total_data += data_len

session = asf.ASFSession()

load_dotenv()

# Get user and password credentials
username = os.getenv('EARTHDATA_USERNAME')
password = os.getenv('EARTHDATA_PASSWORD')

if username is None or password is None:
    username = input('Enter your EARTHDATA username: ')
    password = input('Enter your EARTHDATA password: ')

if username is None or password is None:
    raise ValueError('Please set the EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables')

# Authenticate the session
session.auth_with_creds(username=username, password=password)

# in the format (FIRENAME: (LATMIN, LATMAX, LONGMIN, LONGMAX))
# or            (FIRENAME: (s_min,  n_max,  w_min,   e_max))
for fire, data in extremes.items():
    path = os.path.join('organized_dataset', fire)

    # Check if the {fire}/data folder exists
    if not os.path.isdir(path + '/data') or len(os.listdir(path + '/data')) == 0:
        os.makedirs(path + '/data', exist_ok=True)

        # Get the earliest date for the fire

        print(f'{fire}: {fires[fire]}')

        date = min(fires[fire], key=lambda x: datetime.strptime(x.split('/')[-1], date_format_str))
        date = datetime.strptime(date.split('/')[-1], date_format_str)

        print(f'Getting data for {fire} on {date}')

        date_end = date

        try:

            polygon = f'POLYGON(({data[2]:.6} {data[1]:.6}, {data[2]:.6} {data[0]:.6}, {data[3]:.6} {data[0]:.6}, {data[3]:.6} {data[1]:.6}, {data[2]:.6} {data[1]:.6}))'

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
                    raise ValueError(f'No data found for {fire} on {date}')

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
                        target_polygon = Polygon(candidate)
                        fire_polygon = Polygon(target)

                        # Check if the coordinates are within the bounds of the fire
                        # The order goes [0] = top left, [1] = top right, [2] = bottom right, [3] = bottom left
                        if target_polygon.contains(fire_polygon):
                            print(f'Found {result.properties['sceneName']}')
                            results = [result]
                            break
                        else:
                            continue

            print(f'{results[0].properties['sceneName']}')

            if not os.path.isdir('tmp'):
                os.makedirs('tmp')

            if not os.path.isfile(f'tmp/{results[0].properties['sceneName']}.zip'):
                results[0].download(path='tmp', session=session)

            files = generate_cropped_geocode_grd(results[0].properties['sceneName'], out_dir=f'organized_dataset/{fire}/data')

            print(f'Files: {files}')

            # Perform translation for 20x20 pixel sizes
            for geotiff in files:
                crop_and_scale_to_20x20(
                    input_tiff_path=geotiff[0], 
                    output_tiff_path=geotiff[0],
                    nw_latlon=(data[1], data[2]), 
                    sw_latlon=(data[0], data[2]), 
                    se_latlon=(data[0], data[3]), 
                    ne_latlon=(data[1], data[3]), 
                )
            
        except Exception as e:
            print(f'Error with {fire}: {e}')
            continue

# Put the WKT fire pixel polygons onto the GeoTIFF
for fire, data in fires.items():
    # Get the SAR data files (.tiff)
    sar_files = [file for file in os.listdir(f'organized_dataset/{fire}/data') if file.endswith('.tiff')]

    for day in data:
        # Search for the .wkt file for the fire polygon
        wkt_file = os.listdir(day)
        wkt_file = [file for file in wkt_file if file.endswith('.wkt')].pop()

        # Open the wkt file
        with open(f'{day}/{wkt_file}', 'r') as f:
            polygons = []

            # Every line is a polygon
            for line in f.readlines():
                polygons.append(line.strip())

            for tiff in sar_files:
                input_file = f'organized_dataset/{fire}/data/{tiff}'
                draw_wkt_to_geotiff(polygons, input_file, output_file=f'{day}/{tiff.split(".")[0]}_wkt.tiff')