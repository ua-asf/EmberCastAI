import os
from datetime import datetime, timedelta
import asf_search as asf
import math
import hashlib

# GDAL configuration
from osgeo import gdal

import numpy as np
import torch

from ship import generate_geocoded_grd
from geo import (
    crop_and_scale_to_20x20,
    haversine_distance,
    draw_wkt_to_geotiff,
    merge_geotiffs,
)
from dem import download_dem_from_bounds

from shapely.geometry import Polygon

from model import SimpleFireCNN

gdal.UseExceptions()

# Define the date format string
date_format_str = "%Y-%m-%dT%H:%M:%S.%f"

# The width/height of the pixels in the SAR data, in meters
PIXEL_SIZE = 20

# The width/height of the square in the SAR data, in pixels (as referenced above)
SQUARE_SIZE = 100


def get_wkt_extremes(
    wkt_list: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    lat_min = 90
    lat_max = -90
    long_min = 180
    long_max = -180

    for long, lat in wkt_list:
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

    return (lat_min, lat_max, long_min, long_max)


def extract_bands_from_tiff(tiff_path: str) -> np.ndarray:
    """Extract all bands from a GeoTIFF and return as numpy array"""
    ds = gdal.Open(tiff_path)
    if ds is None:
        raise ValueError(f"Could not open {tiff_path}")

    num_bands = ds.RasterCount

    xsize = ds.RasterXSize
    ysize = ds.RasterYSize

    # Option 1: Direct CHW allocation (most memory efficient)
    bands_array = np.zeros((num_bands, ysize, xsize), dtype=np.uint16)

    for band_idx in range(num_bands):
        band = ds.GetRasterBand(band_idx + 1)
        bands_array[band_idx] = band.ReadAsArray()

    ds = None
    return bands_array


def get_drawn_wkt(
    username: str,
    password: str,
    date_str: str,
    wkt_list: list[list[tuple[float, float]]],
) -> np.ndarray:
    """
    Downloads SAR data for the given WKT polygons and date, processes it, and draws the WKT ontop of an np.
    """

    session = asf.ASFSession()

    # Authenticate the session
    session.auth_with_creds(username=username, password=password)

    date = datetime.strptime(os.path.basename(date_str), date_format_str)

    extremes = get_wkt_extremes([point for sublist in wkt_list for point in sublist])

    print(f"Using extremes: {extremes}")

    # Create a hash for the WKT string to use as a directory name
    hash = hashlib.sha256(str(extremes).encode()).hexdigest()
    file_path = f"tmp/{hash}"

    # Create assets/tmp/{wkt_string}/data directory if it doesn't exist
    if not os.path.isdir(f"{file_path}/data"):
        os.makedirs(f"{file_path}/data")

    date_end = date

    polygon = f"POLYGON(({extremes[2]:.8} {extremes[1]:.8}, {extremes[2]:.8} {extremes[0]:.8}, {extremes[3]:.8} {extremes[0]:.8}, {extremes[3]:.8} {extremes[1]:.8}, {extremes[2]:.8} {extremes[1]:.8}))"

    print(polygon)

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

        delta += delta
        options["end"] = options["start"]
        options["start"] = (date - timedelta(days=delta)).strftime(date_format_str)

        if delta > 1000:
            raise ValueError(f"No data found for fire on {date}")

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

    if not os.path.isfile(f"tmp/{results[0].properties['sceneName']}.zip"):
        # Check if tmp/downloads directory exists, if not create it
        if not os.path.isdir("tmp"):
            os.makedirs("tmp")

        results[0].download(path="tmp", session=session)

    generate_geocoded_grd(
        results[0].properties["sceneName"],
        in_dir="tmp",
        out_dir=f"{file_path}/data",
    )

    # Download DEM data for the area.
    # We need to add a bit of padding to ensure we cover the entire area
    dem = download_dem_from_bounds(
        extremes[0] - 0.05,
        extremes[1] + 0.05,
        extremes[2] - 0.05,
        extremes[3] + 0.05,
        f"{file_path}/data",
    )

    tiff_files = [
        f"{file_path}/data/{file}"
        for file in os.listdir(f"{file_path}/data")
        if (file.endswith(".tiff") or file.endswith(".tif")) and "merged" not in file
    ]

    # Perform translation for 20x20 pixel sizes
    for geotiff in tiff_files:
        crop_and_scale_to_20x20(
            input_tiff_path=geotiff,
            output_tiff_path=geotiff,
            nw_latlon=(extremes[1], extremes[2]),
            sw_latlon=(extremes[0], extremes[2]),
            se_latlon=(extremes[0], extremes[3]),
            ne_latlon=(extremes[1], extremes[3]),
            pixel_size=PIXEL_SIZE,
            square_size=SQUARE_SIZE,
        )

    vv_band = [file for file in tiff_files if "vv" in file.lower()][0]
    vh_band = [file for file in tiff_files if "vh" in file.lower()][0]
    dem_band = [file for file in tiff_files if "dem" in file.lower()][0]

    tiffs = [vv_band, vh_band, dem_band]

    if len(tiff_files) != len(tiffs):
        raise ValueError(
            "Could not find all required bands (VV, VH, DEM). Found: "
            + ", ".join(tiff_files)
        )

    # Merge the files into a single geotiff
    merge_geotiffs(tiffs, output_file=f"{file_path}/data/merged.tiff")

    point_str = ", ".join([f"{coord[0]:.8} {coord[1]:.8}" for coord in wkt_list[0]])
    polygon_str = f"POLYGON(({point_str}))"

    input_file = f"{file_path}/data/merged.tiff"
    draw_wkt_to_geotiff(
        [polygon_str], input_file, output_file=f"{file_path}/data/merged_wkt.tiff"
    )

    raise Exception("Debug stop")

    # Open the file and extract the band with the WKT drawn on it
    return extract_bands_from_tiff(f"{file_path}/data/merged_wkt.tiff")


def expand_to_square(array, target_size=None):
    bands, width, height = array.shape
    if target_size is None:
        max_dim = max(height, width)
        target_size = ((max_dim // SQUARE_SIZE) + 1) * SQUARE_SIZE
        if target_size % 2 != 0:
            target_size += SQUARE_SIZE
    square_array = np.zeros((target_size, target_size, bands), dtype=array.dtype)
    y_offset = (target_size - height) // 2
    x_offset = (target_size - width) // 2
    square_array[y_offset : y_offset + height, x_offset : x_offset + width] = array
    return square_array


def cut_into_squares(array, square_size=SQUARE_SIZE):
    channels, height, width = array.shape

    if height % square_size != 0 or width % square_size != 0:
        raise ValueError(
            f"Array dimensions ({height}, {width}) must be multiples of {square_size}"
        )

    num_squares_y = height // square_size
    num_squares_x = width // square_size

    squares = []
    for i in range(num_squares_y):
        for j in range(num_squares_x):
            # Extract square preserving all channels
            square = array[
                :,
                i * square_size : (i + 1) * square_size,
                j * square_size : (j + 1) * square_size,
            ]
            squares.append(square)

    # Add batch dimension: (num_patches, channels, H, W) -> (num_patches, 1, channels, H, W)
    return torch.from_numpy(np.array(squares)).unsqueeze(1)


def run_inference(squares):
    """Run inference on the provided data and return the stitched results as a numpy array"""
    results = []
    model = SimpleFireCNN()
    model.load_state_dict(torch.load(f"{os.getcwd()}/fire_predictor_model.pth"))
    model.eval()

    with torch.no_grad():
        for square in squares:
            results.append(model(square).numpy())

    return results


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

    stitched_array = np.zeros((stitched_height, stitched_width), dtype=np.float32)

    print(f"{square_size=}")

    for idx, square in enumerate(squares):
        row = idx // num_cols
        col = idx % num_cols
        y_offset = row * square_size
        x_offset = col * square_size

        print(f"Placing square {idx} at ({y_offset}, {x_offset})")
        print(f"{square.shape=}")
        print(f"{stitched_array.shape=}")

        stitched_array[
            y_offset : y_offset + square_size, x_offset : x_offset + square_size
        ] = square.squeeze() * 255

    return stitched_array.astype(np.uint8)


def process(
    username: str,
    password: str,
    wkt_list: list[list[tuple[float, float]]],
    date_str: str,
) -> tuple[list[int], list[int], list[int]]:
    data = get_drawn_wkt(
        username=username, password=password, date_str=date_str, wkt_list=wkt_list
    )

    print(f"Downloaded data shape: {data.shape}")

    squares = cut_into_squares(data)

    results = run_inference(squares)

    print(f"Got {len(results)} results from inference")
    print(f"First result shape: {results[0].shape if results else 'N/A'}")

    stitched_results: list[int] = [int(x) for x in stitch_results(results).flatten()]

    return (
        # Original mask (first band)
        data[0].flatten().astype(int).tolist(),
        # Results from inference
        stitched_results,
        # Digital elevation model (4th band)
        data[3].flatten().astype(int).tolist(),
    )
