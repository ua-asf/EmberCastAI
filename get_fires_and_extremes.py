import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import asf_search as asf
import numpy as np
import shutil

# Dates used by the dataset are usually in the format YYYYMMDD
date_format_str = '%Y-%m-%d-%H%M'

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
            shutil.rmtree(os.path.join('organized_dataset', fire), ignore_errors=True)
            print(f'Removed {fire}')

print(f'Found {len(fires)} fires')

# Purge empty fires
fires = {fire: data for fire, data in fires.items() if len(data) > 0}

print(fires)

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
    # Check if the data is already downloaded
    if os.path.exists(f'{fire}/downloaded'):
        print(f'{fire} already downloaded')
        continue

    try:
        polygon = f'POLYGON(( {data[2]} {data[1]}, {data[3]} {data[1]}, {data[3]} {data[0]}, {data[2]} {data[0]}, {data[2]} {data[1]} ))'
        results = asf.geo_search(platform=[asf.PLATFORM.SENTINEL1], intersectsWith=polygon, polarization=['VV+VH', 'HH+HV'], start='1 month ago', end='now')
        result = results.pop()
        # Make directory for the fire
        os.makedirs(f'{fire}/downloaded', exist_ok=True)
        results.download(path=f'{fire}/downloaded', session=session)
        print(f'{fire}: {len(results)}')
    except Exception as e:
        print(f'Error with {fire}: {e}')
        continue