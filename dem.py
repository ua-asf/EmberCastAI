#!/usr/bin/env python3
"""
Utility functions for downloading and processing DEM data from OpenTopography
using the bmi-topography library to add as a 4th band to SAR GeoTIFFs.
"""

import os
from bmi_topography import Topography


def download_dem_from_bounds(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    output_dir: str,
    dem_type: str = "NASADEM",
):
    """
    Download DEM data for a fire area using bmi-topography.

    Args:
        lat_min, lat_max, lon_min, lon_max: Bounding box coordinates
        output_dir: Directory to save the DEM file (optional)
        dem_type: DEM dataset to use ('NASADEM', 'SRTMGL3', 'SRTMGL1', etc.)

    Returns:
        str: Path to the downloaded DEM file
    """
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"Downloading DEM data for bounds: lat({lat_min}, {lat_max}), lon({lon_min}, {lon_max})"
    )
    print(f"Using DEM type: {dem_type}")

    try:
        # Initialize Topography object with correct parameters
        topo = Topography(
            dem_type=dem_type,
            south=lat_min,
            north=lat_max,
            west=lon_min,
            east=lon_max,
            output_format="GTiff",
            cache_dir=output_dir,
        )

        # Download the DEM data
        file_path = topo.fetch()

        # Load data to verify it's not empty
        try:
            data = topo.load()
        except Exception as e:
            os.remove(file_path)
            raise e

        print(data)

        if data is None or len(data) == 0:
            os.remove(file_path)
            raise ValueError("Downloaded DEM data is empty.")

        # Rename file to "dem"
        os.rename(file_path, os.path.join(output_dir, "dem.tiff"))

        return os.path.join(output_dir, "dem.tiff")

    except Exception as e:
        print(f"Error downloading DEM with {dem_type}: {e}")
        if dem_type == "NASADEM":
            print("Trying fallback DEM type SRTMGL1...")
            return download_dem_from_bounds(
                lat_min, lat_max, lon_min, lon_max, output_dir, "SRTMGL1"
            )
        elif dem_type == "SRTMGL1":
            print("Trying fallback DEM type COP30...")
            return download_dem_from_bounds(
                lat_min, lat_max, lon_min, lon_max, output_dir, "COP30"
            )
        elif dem_type == "COP30":
            print("Trying fallback DEM type AW3D30...")
            return download_dem_from_bounds(
                lat_min, lat_max, lon_min, lon_max, output_dir, "AW3D30"
            )
        else:
            raise e
