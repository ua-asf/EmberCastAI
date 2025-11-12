import os
import numpy
from cmr import get_scene_metadata
from geo import geocode_geotiff
from cde_data import get_scene_data, get_image_files, get_gdal_data
from gamma import extract_calibrated_tiff

#################################
#
# Tweakable parameters
#

BRIGHT_THRESHOLD = 6.0  # Number of STD's less than mean-adjusted,
# outlier-removed max value of dataset
# Higher Num = More Ships / False Positives
LABEL_THRESHOLD = 5  # Number of pixels that make a ship
OUTLIER_STD_THRESHOLD = 25  # Std deviations past mean to clip data
CALIBRATE_DATA = False  # Apply sentinel calibration LUTs
NORM_SCENE1_TO_SCENE2 = True  # Roughly match scene min/max's

# Experimental
ARBITRARY_PIXEL_CUTOFF = 1700
ARBITRARY_BRIGHT_PIXEL = 420


def utm_from_lon_lat(lon: float, lat: float) -> int:
    hemisphere = 32600 if lat >= 0 else 32700
    zone = int(lon // 6 + 30) % 60 + 1
    return hemisphere + zone


def generate_geocoded_grd(granule, epsg_code_poly=None, in_dir=None, out_dir=None):
    if not epsg_code_poly:
        print(f"Loading raster data for dataset {granule}")
        print("Querying CMR for scene info")
        _, _, _, epsg_code_poly, _ = get_scene_metadata(granule)

    found_existing = []

    for file in os.listdir(out_dir or "/tmp"):
        if file.upper().endswith(".TIFF"):
            granule_data = "-".join(granule.upper().split("_")[4:8])

            if granule_data in file.upper():
                found_existing.append(file)

    if len(found_existing) == 2:
        print(f"Reusing previously subsetted products {found_existing}")
        return found_existing

    print(f"Local fetching {granule} dataset")
    local_measure_path, zip_file = get_scene_data(granule)

    if CALIBRATE_DATA:  # Make true to do calibration stuff
        print(f"Calibrating {granule}")
        calibrated = extract_calibrated_tiff(zip_file, granule)
        print(f"Loading calibrated data from {calibrated}")
        # data = get_gdal_data (calibrated)
        data = [calibrated]
    else:
        print(f"Getting tiff list for {local_measure_path}")
        files = get_image_files(local_measure_path)
        print("Loading uncalibrated VV and VH tiffs")
        # data = get_gdal_data (files['VV'])
        data = [files["VV"], files["VH"]]

    print("Geocoding products")

    product_paths = []

    for product in data:
        product_name = product.split(".")[-2].split("/")[-1]

        gcgt = geocode_geotiff(
            granule,
            epsg_code_poly,
            product,
            output_file=f"{out_dir}/{product_name}.tiff",
        )
        product_paths.append(f"{product_name}.tiff")
        print(
            f"Completed geocoding {product}, subsetting to {out_dir}/{product_name}.tiff"
        )

    return product_paths


def get_dataset_as_array(dataset):
    if type(dataset) is str:
        dataset = get_gdal_data(dataset)

    print("Reading Band")
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray().astype(numpy.int16)
    print(f"Clipping data to {ARBITRARY_PIXEL_CUTOFF}")
    arr = numpy.clip(arr, 0, ARBITRARY_PIXEL_CUTOFF)
    return arr
