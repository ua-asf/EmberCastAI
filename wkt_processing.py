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

# Define the date format string
from utils import date_format_str

gdal.UseExceptions()

# The width/height of the pixels in the SAR data, in meters
PIXEL_SIZE = 20

# The width/height of the square in the SAR data, in pixels (as referenced above)
SQUARE_SIZE = 100


def get_wkt_extremes(
    wkt_list: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """
    Calculate a rectilinear bounding box around WKT points, padded to a grid multiple.

    Args:
        wkt_list: List of (longitude, latitude) coordinate pairs

    Returns:
        Tuple of (lat_min, lat_max, long_min, long_max) forming a square in meters
    """
    # Find initial bounding box and center
    longitudes, latitudes = zip(*wkt_list)
    lat_center = (min(latitudes) + max(latitudes)) / 2
    long_center = (min(longitudes) + max(longitudes)) / 2

    # Calculate current spans in meters
    lat_dist = haversine_distance(
        min(latitudes), long_center, max(latitudes), long_center
    )
    long_dist = haversine_distance(
        lat_center, min(longitudes), lat_center, max(longitudes)
    )

    # Target size: max dimension rounded up to grid + padding
    grid_size = SQUARE_SIZE * PIXEL_SIZE
    target_long = (math.ceil(long_dist / grid_size) + 3) * grid_size
    target_lat = (math.ceil(lat_dist / grid_size) + 3) * grid_size

    r_earth = 6378137
    lat_center_rad = math.radians(lat_center)

    # Set latitude dimension (more stable due to constant conversion factor)
    lat_delta = math.degrees(target_lat / 2 / r_earth)
    lat_min = lat_center - lat_delta
    lat_max = lat_center + lat_delta

    # Calculate longitude delta that gives the same measured distance
    long_delta = math.degrees((target_long / 2) / (r_earth * math.cos(lat_center_rad)))

    return (
        lat_min,
        lat_max,
        long_center - long_delta,
        long_center + long_delta,
    )


def get_sar_over_area(
    username: str,
    password: str,
    date_str: str,
    polygon: list[tuple[float, float]],
    file_path: str,
) -> tuple[str, str]:
    """
    Gets SAR data over the given polygon and date, processing it as needed.
    Will mosaic multiple scenes if required.

    Inputs:
        username: ASF username
        password: ASF password
        date_str: Date string in YYYY-MM-DD format
        polygon: List of (longitude, latitude) tuples defining the polygon
    Outputs:
        List of paths to generated GRD files
        Format: `tuple[VV_BAND, VH_BAND]`
    """

    session = asf.ASFSession()

    # Authenticate the session
    session.auth_with_creds(username=username, password=password)

    date = datetime.strptime(os.path.basename(date_str), date_format_str)

    date_end = date

    delta = 15

    date_start = date - timedelta(days=delta)

    polygon_str = (
        f"POLYGON(({', '.join([f'{coord[0]} {coord[1]}' for coord in polygon])}))"
    )

    options = {
        "dataset": "SENTINEL-1",
        "intersectsWith": polygon_str,
        "polarization": ["VV+VH"],
        "processingLevel": "GRD_HD",
        "start": date_start.strftime(date_format_str),
        "end": date_end.strftime(date_format_str),
        "maxResults": 1000,
    }

    results = []
    final_results = []

    fire_polygon = Polygon(polygon)

    while len(results) == 0:
        # Extend the search window and add new results.
        # We do this to save querying for the same date range multiple times,
        # say we miss any relevant data the first time, Oct-Nov. We save that,
        # then query again for Nov-Jan.
        results.extend(asf.geo_search(**options))

        delta += delta
        options["end"] = options["start"]
        options["start"] = (date - timedelta(days=delta)).strftime(date_format_str)

        if delta > 1000:
            raise ValueError(f"No data found for fire on {date}")

        if len(results) > 0:
            # Sort the data on the time of acquisition, newest first
            results.sort(
                key=lambda x: datetime.strptime(
                    x.properties["stopTime"], "%Y-%m-%dT%H:%M:%SZ"
                ),
                reverse=True,
            )

        # === Step 1: Check if a polygon covers the area of the fire polygon ===

        for result in results:
            candidate_polygon = Polygon(result.geometry["coordinates"][0])
            if candidate_polygon.contains_properly(fire_polygon):
                final_results = [result]
                break

        # Check if we found a suitable polygon
        if len(final_results) == 1:
            break

        # === Step 2: Mosacking. Search for polygons that combine to cover the fire polygon ===

        # Check again if we found a suitable set of polygons
        if len(final_results) > 0:
            break

        ascending_granules = get_containing_flight_dir_polygon(
            results, "ASCENDING", fire_polygon
        )
        descending_granules = get_containing_flight_dir_polygon(
            results, "DESCENDING", fire_polygon
        )

        if len(ascending_granules) == 0 and len(descending_granules) == 0:
            print(
                f"No suitable mosaic found with {len(results)} granules, expanding search"
            )
        else:
            # Return whichever list has a shorter length
            if len(ascending_granules) > len(descending_granules):
                final_results = descending_granules
            else:
                final_results = ascending_granules

        if len(final_results) > 0:
            break

        # === Step 3: No mosaic found, repeat with larger date range ===
        results = []

    generated_grds = []

    print(
        f"Final results: {len(final_results)} granules: {[result.properties['sceneName'] for result in final_results]}"
    )

    for result in final_results:
        if not os.path.isfile(f"tmp/{result.properties['sceneName']}.zip"):
            # Check if tmp/downloads directory exists, if not create it
            if not os.path.isdir("tmp"):
                os.makedirs("tmp")

            result.download(path="tmp", session=session)

        generated_grds.append(
            generate_geocoded_grd(
                result.properties["sceneName"],
                in_dir="tmp",
                # Output to tmp so we can cache the GRD files for later use
                out_dir="tmp",
            )
        )

    # Flatten the list of tuples into a single list
    generated_grds = [grd for pair in generated_grds for grd in pair]

    # Append file path to each generated GRD file
    generated_grds = [f"tmp/{grd}" for grd in generated_grds]

    print(f"Generated {len(generated_grds)} GRD files: {generated_grds}")

    if len(generated_grds) > 2:
        return mosaic_sar_bands(
            generated_grds, out_dir=f"{file_path}/data", out_file_name="mosaic"
        )
    elif len(generated_grds) == 2:
        return (generated_grds[0], generated_grds[1])
    else:
        raise ValueError(f"Unexpected number of GRD files: {len(generated_grds)}")


def get_containing_flight_dir_polygon(results, direction, fire_polygon):
    candidate_polygon = Polygon()
    mosaic_results = []

    for result in results:
        if not result.properties["flightDirection"] == direction:
            continue

        new_polygon = Polygon(result.geometry["coordinates"][0])

        intersection = candidate_polygon.intersection(new_polygon)

        # Skip if the new polygon has more than a certain percentage overlap with the existing candidate polygon
        if candidate_polygon.area != 0 and intersection.area / new_polygon.area > 0.3:
            continue

        candidate_polygon = candidate_polygon.union(new_polygon)
        mosaic_results.append(result)

        if candidate_polygon.contains_properly(fire_polygon):
            break

    if candidate_polygon.contains_properly(fire_polygon):
        return mosaic_results
    else:
        return []


def mosaic_sar_bands(
    input_files: list[str], out_dir: str, out_file_name: str
) -> tuple[str, str]:
    """
    Takes a set of VV and VH files and creates a mosaic for each polarization.
    """
    vv_files = [f for f in input_files if "vv" in f.lower()]
    vh_files = [f for f in input_files if "vh" in f.lower()]

    mosaic_geotiffs(vv_files, dst_path=f"{out_dir}/{out_file_name}_vv.tiff")
    mosaic_geotiffs(vh_files, dst_path=f"{out_dir}/{out_file_name}_vh.tiff")

    return (f"{out_dir}/{out_file_name}_vv.tiff", f"{out_dir}/{out_file_name}_vh.tiff")


from osgeo import gdal


def mosaic_geotiffs(tiffs: list[str], dst_path: str):
    options = gdal.WarpOptions(
        format="GTiff",
        outputType=gdal.GDT_UInt16,
        multithread=True,
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
    )

    print(f"\n\n\n\n\n\n\nMosaicking {len(tiffs)} files into {dst_path}\n\n\n\n\n\n")

    gdal.Warp(
        dst_path,
        tiffs,
        options=options,
    )


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


def get_hash(extremes):
    return hashlib.sha256(str(extremes).encode()).hexdigest()


def get_drawn_wkt(
    username: str,
    password: str,
    date_str: str,
    wkt_list: list[list[tuple[float, float]]],
    extremes: tuple[float, float, float, float],
    file_path: str,
) -> np.ndarray:
    """
    Downloads SAR data for the given WKT polygons and date, processes it, and draws the WKT ontop of an np.
    """

    # Create assets/tmp/{wkt_string}/data directory if it doesn't exist
    if not os.path.isdir(f"{file_path}/data"):
        os.makedirs(f"{file_path}/data")

    # Make a square polygon from the extremes - add padding so the SAR data covers the entire area
    polygon = [
        (extremes[2] - 0.1, extremes[1] + 0.1),
        (extremes[2] - 0.1, extremes[0] - 0.1),
        (extremes[3] + 0.1, extremes[0] - 0.1),
        (extremes[3] + 0.1, extremes[1] + 0.1),
        (extremes[2] - 0.1, extremes[1] + 0.1),
    ]

    results = get_sar_over_area(
        username=username,
        password=password,
        date_str=date_str,
        polygon=polygon,
        file_path=file_path,
    )

    # Download DEM data for the area.
    # We need to add a bit of padding to ensure we cover the entire area
    # dem = download_dem_from_bounds(
    # extremes[0] - 0.05,
    # extremes[1] + 0.05,
    # extremes[2] - 0.05,
    # extremes[3] + 0.05,
    # f"{file_path}/data",
    # )

    tiff_files = [
        results[0],  # VV band
        results[1],  # VH band
        # dem       , # DEM
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

    # Merge the files into a single geotiff
    merge_geotiffs(tiff_files, output_file=f"{file_path}/data/merged.tiff")

    wkt_strs = []
    for wkt in wkt_list:
        point_str = ", ".join([f"{coord[0]:.8} {coord[1]:.8}" for coord in wkt])
        wkt_strs.append(f"POLYGON(({point_str}))")

    input_file = f"{file_path}/data/merged.tiff"
    draw_wkt_to_geotiff(
        wkt_strs, input_file, output_file=f"{file_path}/data/merged_wkt.tiff"
    )

    # raise Exception("Debug stop")

    # Open the file and extract the band with the WKT drawn on it
    return extract_bands_from_tiff(f"{file_path}/data/merged_wkt.tiff")


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
    return (
        torch.from_numpy(np.array(squares)).unsqueeze(1),
        num_squares_x,
        num_squares_y,
    )


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
        username=username,
        password=password,
        date_str=date_str,
        wkt_list=wkt_list,
        extremes=get_wkt_extremes([pt for sublist in wkt_list for pt in sublist]),
        file_path=f"tmp/{get_hash(wkt_list)}",
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

