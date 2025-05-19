import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import asf_search as asf
import rasterio
import numpy as np
from osgeo import gdal

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
        # Get all folder in the directory
        dates = os.listdir(os.path.join('organized_dataset', dirname))
        # Purge any 'unknown date' folders
        dates = [date for date in dates if date != 'UnknownDate']
        # Sort the files by date
        dates.sort(key=lambda x: datetime.strptime(x, date_format_str))
        # Skip any fires with less than 2 days of data
        if len(dates) < 2:
            print(f'Skipping {dirname} with {len(dates)} days of data')
            continue
        # Add the fire to the fires dict
        fires[dirname] = [os.path.join('organized_dataset', dirname, file) for file in dates]

print(f'Found {len(fires)} fires')

# Purge empty fires
fires = {fire: data for fire, data in fires.items() if len(data) > 0}

print(fires)

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

                # Update the extremes for the fire
                extremes[fire_name] = (lat_min, lat_max, long_min, long_max)

total_data = 0

for fire, data in fires.items():
    data_len = len(data)
    total_data += data_len

session = asf.ASFSession()

load_dotenv()

# Get user and password credentials
username = os.getenv('ASF_USERNAME')
password = os.getenv('ASF_PASSWORD')
if username is None or password is None:
    raise ValueError('Please set the ASF_USERNAME and ASF_PASSWORD environment variables')

# Authenticate the session
session.auth_with_creds(username=username, password=password)

# in the format (FIRENAME: (LATMIN, LATMAX, LONGMIN, LONGMAX))
# or            (FIRENAME: (s_min,  n_max,  w_min,   e_max))
for fire, data in extremes.items():
    for day in fires[fire]:
        # Skip any days that have data
        if os.path.isdir(day):
            continue

        print(f'Getting data for {fire} on {day}')
        
        date = datetime.strptime(day.split('/')[-1], date_format_str)
        date_end = date

        try:
            polygon = f'POLYGON(( {data[3]} {data[1]}, {data[3]} {data[0]}, {data[2]} {data[0]}, {data[2]} {data[1]}, {data[3]} {data[1]} ))'

            delta = 15

            date_start = date - timedelta(days=delta)

            results = asf.geo_search(platform=[asf.PLATFORM.SENTINEL1], intersectsWith=polygon, processingLevel='GRD_HS', start=date_start, end=date_end)

            while len(results) == 0:
                delta += delta
                date_start = date - timedelta(days=delta)
                date_end = date + timedelta(days=delta)

                results = asf.geo_search(platform=[asf.PLATFORM.SENTINEL1], intersectsWith=polygon, processingLevel='GRD_HS', start=date_start, end=date_end)

            result = results.pop()
            results.download(path=day, session=session)
        except Exception as e:
            print(f'Error with {fire}: {e}')
            continue

# Unzip the S1A_IW_SLC file
for fire, data in fires.items():
    for day in data:
        print(day)

        # Unzip the S1A_IW_SLC file
        for file in os.listdir(day):
            # Only unzip the .zip files if the data folder doesn't exist
            if not os.path.isdir(os.path.join(day, 'data')):
                if file.endswith('.zip') and file.startswith('S1A_IW_GRD'):
                    # Unzip the file
                    os.system(f'unzip {day}/{file} -d {day}/data')

                    # Move all of the data in the /data/*.SAFE/measurement folder up
                    os.system(f'mv {day}/data/*.SAFE/measurement/*.tiff {day}/data/')

                    # Remove everything else that isn't .tiff
                    os.system(f'rm -rf {day}/data/*.SAFE')

# Use GDAL to convert the split tiff files to a single tiff file
for fire, data in fires.items():
    for day in data:
        files = []
        for file in os.listdir(os.path.join(day, 'data')):
            # Only add the .tiff files
            if file.endswith('.tiff'):
                files.append(os.path.join(day, 'data', file))

        print(f'Combining {len(files)} files')
        # Open the files (YOU WILL NEED A LOT OF RAM FOR THIS)
        datasets = [rasterio.open(file_path) for file_path in files]
        print('Reading files')
        # Read the data from the files
        arrays = [dataset.read() for dataset in datasets]
        # Comine the arrays into a single array
        combined_array = np.concatenate(arrays, axis=0)
        print('Combining arrays')
        # Write the combined array to a new file
        with rasterio.open(os.path.join(day, 'data', 'merged.tiff'), 'w', height=combined_array.shape[1], width=combined_array.shape[2], count=combined_array.shape[0], dtype=combined_array.dtype) as dst:
            dst.write(combined_array)
