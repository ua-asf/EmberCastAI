#!python

import os
from dotenv import load_dotenv
from datetime import datetime
import asf_search as asf

# Dates used by the dataset are usually in the format YYYYMMDD
date_format_str = '%Y%m%d'

# Parse /dataset/coordinates folder

# Fires is a dict of each fire, with each fire being a 
# list of the files associated with that fire, sorted
# by date
fires = {}

for dirpath, dirnames, filenames in os.walk('dataset/coordinates'):
    for filename in filenames:
        # Search for files with the .wkt extension
        if filename.endswith('.wkt'):
            file = os.path.join(dirpath, filename)
            # Insert into fires list
            # Search for last list that contains the same beginning filepath
            fire_found = False
            for fire in fires.keys():
                if file.startswith(fire):
                    fires[fire].append(file)
                    fire_found = True
                    break

            # In the case of a new fire being found, add it to the fires dict
            if not fire_found:
                file_dir = file.split('/')

                for i in range(len(file_dir)):
                    # All days for a fire are the same except for after the 'IR' or 'KML_KMZ' folder
                    if file_dir[i].upper() in ['IR', 'KML_KMZ', 'KMZ', 'GIS']:
                        file_dir = file_dir[:i]
                        break
                    else:
                        try:
                            # See if the current string is a date, at which point the fire name is found
                            datetime.strptime(file_dir[i], date_format_str)
                            file_dir = file_dir[:i]
                            break
                        except ValueError:
                            continue

                fires['/'.join(file_dir)] = [file]

# Find the lat/long extremes for each fire

# Extremes stores the lat/long extremes for each fire
# in the format (FIRENAME: (LATMIN, LATMAX, LONGMIN, LONGMAX))
extremes = fires.copy()

for fire_name, data in fires.items():
    lat_min = 180
    lat_max = -180
    long_min = 180
    long_max = -180

    for file in data:
        # Parse the lat/long from the file
        with open(file, 'r') as f:
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
