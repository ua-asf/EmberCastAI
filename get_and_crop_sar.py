import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import asf_search as asf
import math
from ship import generate_geocoded_grd
from geo import (
    crop_and_scale_to_20x20,
    haversine_distance,
    draw_wkt_to_geotiff,
    merge_geotiffs,
)
from shapely.geometry import Polygon
import shutil
from utils import get_fires

# GDAL configuration
from osgeo import gdal

gdal.UseExceptions()

# Dates used by the dataset are usually in the format YYYYMMDD
from utils import date_format_str

# The width/height of the pixels in the SAR data, in meters
PIXEL_SIZE = 20

# The width/height of the square in the SAR data, in pixels (as referenced above)
SQUARE_SIZE = 100

fires = get_fires()

print(f"Found {len(fires)} fires")

# Purge empty fires
fires = {fire: data for fire, data in fires.items() if len(data) > 0}

for fire, data in fires.items():
    print(f"{fire}: {data}")

# Find the lat/long extremes for each fire

# Extremes stores the lat/long extremes for each fire
# in the format (FIRENAME: (LATMIN, LATMAX, LONGMIN, LONGMAX))
extremes = fires.copy()

for fire_name, data in fires.items():
    lat_min = 90
    lat_max = -90
    long_min = 180
    long_max = -180

    for day in data:
        for file in os.listdir(day):
            # Only open the .wkt files
            if not file.endswith(".wkt"):
                continue

            # Parse the lat/long from the file
            with open(f"{day}/{file}", "r") as f:
                print(f"{day}/{file}")
                # Every line is a polygon
                for line in f.readlines():
                    # Remove the 'POLYGON ((' and '))\n' from the line
                    line = line.removeprefix("POLYGON ((").removesuffix("))\n")
                    coords = line.split(", ")
                    for line in coords:
                        long, lat = line.split(" ")
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

    print(f"New width: {new_width}")

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
    extremes[fire_name] = (lat_min, lat_max, long_min, long_max)

# Print the extremes for each fire
for fire, data in extremes.items():
    print(f"{fire}: {data}")

total_data = 0

for fire, data in fires.items():
    data_len = len(data)
    total_data += data_len

session = asf.ASFSession()

load_dotenv()

# Get user and password credentials
username = os.getenv("EARTHDATA_USERNAME")
password = os.getenv("EARTHDATA_PASSWORD")

if username is None or password is None:
    username = input("Enter your EARTHDATA username: ")
    password = input("Enter your EARTHDATA password: ")

if username is None or password is None:
    raise ValueError(
        "Please set the EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables"
    )

# Authenticate the session
session.auth_with_creds(username=username, password=password)

# in the format (FIRENAME: (LATMIN, LATMAX, LONGMIN, LONGMAX))
# or            (FIRENAME: (s_min,  n_max,  w_min,   e_max))
for fire, data in extremes.items():
    path = os.path.join("organized_dataset", fire)

    # Check if the {fire}/data folder exists
    if not os.path.isdir(path + "/data") or len(os.listdir(path + "/data")) == 0:
        os.makedirs(path + "/data", exist_ok=True)

        # Get the earliest date for the fire

        print(f"{fire}: {fires[fire]}")

        date = min(
            fires[fire],
            key=lambda x: datetime.strptime(os.path.basename(x), date_format_str),
        )
        date = datetime.strptime(os.path.basename(date), date_format_str)

        print(f"Getting data for {fire} on {date}")

        date_end = date

        try:
            polygon = f"POLYGON(({data[2]:.8} {data[1]:.8}, {data[2]:.8} {data[0]:.8}, {data[3]:.8} {data[0]:.8}, {data[3]:.8} {data[1]:.8}, {data[2]:.8} {data[1]:.8}))"

            print(f"Polygon: {polygon}")

            delta = 36

            date_start = date - timedelta(days=delta)

            options = {
                "dataset": "SENTINEL-1",
                "intersectsWith": polygon,
                "polarization": ["VV+VH"],
                "processingLevel": "GRD_HD",
                "start": date_start.strftime(date_format_str),
                "end": date_end.strftime(date_format_str),
            }

            results = []

            while len(results) == 0:
                results = asf.geo_search(**options)

                print(f"Found {len(results)} results")

                delta += delta
                options["end"] = options["start"]
                options["start"] = (date - timedelta(days=delta)).strftime(
                    date_format_str
                )

                if delta > 1000:
                    raise ValueError(f"No data found for {fire} on {date}")

                print(f"Start: {options['start']}, End: {options['end']}")

                if len(results) > 0:
                    # Sort by date
                    results.sort(
                        key=lambda x: datetime.strptime(
                            x.properties["stopTime"], "%Y-%m-%dT%H:%M:%S%fZ"
                        )
                    )

                    # Ensure the polygon is within the bounds of the fire
                    while results:
                        # Get the first result
                        result = results.pop()
                        candidate = result.geometry["coordinates"][0]

                        target = polygon.split("((")[1].split("))")[0].split(", ")
                        # Split the coordinates into lat/long
                        target = [coord.split(" ") for coord in target]
                        # Convert the coordinates to floats
                        target = [
                            [float(coord[0]), float(coord[1])] for coord in target
                        ]

                        # Convert the coordinates to a polygon
                        candidate_polygon = Polygon(candidate)
                        fire_polygon = Polygon(target)

                        # Check if the coordinates are within the bounds of the fire
                        # The order goes [0] = top left, [1] = top right, [2] = bottom right, [3] = bottom left
                        if candidate_polygon.contains_properly(fire_polygon):
                            print(f"Found {result.properties['sceneName']}")
                            print(f"Candidate polygon: {candidate_polygon}")
                            print(f"Fire polygon: {fire_polygon}")
                            results = [result]
                            break
                        else:
                            continue

            print(f"{results[0].properties['sceneName']}")

            if not os.path.isdir("tmp"):
                os.makedirs("tmp")

            if not os.path.isfile(f"tmp/{results[0].properties['sceneName']}.zip"):
                results[0].download(path="tmp", session=session)

            files = generate_geocoded_grd(
                results[0].properties["sceneName"],
                out_dir=f"organized_dataset/{fire}/data",
            )

            print(f"Files: {files}")

            # Perform translation for 20x20 pixel sizes
            for geotiff in files:
                crop_and_scale_to_20x20(
                    input_tiff_path=geotiff[0],
                    output_tiff_path=geotiff[0],
                    nw_latlon=(data[1], data[2]),
                    sw_latlon=(data[0], data[2]),
                    se_latlon=(data[0], data[3]),
                    ne_latlon=(data[1], data[3]),
                    pixel_size=PIXEL_SIZE,
                    square_size=SQUARE_SIZE,
                )

        except Exception as e:
            print(f"Error with {fire}: {e}")
            # Remove the fire from the fires dict, extremes, and organized_dataset
            del fires[fire]
            shutil.rmtree(f"organized_dataset/{fire}")
            continue

# Merge all geotiffs into a single geotiff
for fire, data in fires.items():
    # Get the SAR data files (.tiff)
    sar_files = [
        file
        for file in os.listdir(f"organized_dataset/{fire}/data")
        if file.endswith(".tiff") and file != "merged.tiff"
    ]

    # Sort such that the vv band is first, then the vh band
    if "vh" in sar_files[0]:
        sar_files.reverse()

    # Append full file path to each file
    sar_files = [
        os.path.join(f"organized_dataset/{fire}/data", file) for file in sar_files
    ]

    # Merge the files into a single geotiff
    merge_geotiffs(sar_files, output_file=f"organized_dataset/{fire}/data/merged.tiff")


# Put the WKT fire pixel polygons onto the GeoTIFF
for fire, data in fires.items():
    # Get the SAR data files (.tiff)
    sar_files = [
        file
        for file in os.listdir(f"organized_dataset/{fire}/data")
        if file == "merged.tiff"
    ]

    for day in data:
        # Search for the .wkt file for the fire polygon
        wkt_file = os.listdir(day)
        wkt_file = [file for file in wkt_file if file.endswith(".wkt")].pop()

        # Open the wkt file
        with open(f"{day}/{wkt_file}", "r") as f:
            polygons = []

            # Every line is a polygon
            for line in f.readlines():
                polygons.append(line.strip())

            for tiff in sar_files:
                input_file = f"organized_dataset/{fire}/data/{tiff}"
                draw_wkt_to_geotiff(
                    polygons,
                    input_file,
                    output_file=f"{day}/{tiff.split('.')[0]}_wkt.tiff",
                )
