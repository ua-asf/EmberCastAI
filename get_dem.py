#!/usr/bin/env python3
"""
Simple script to download DEM data using bmi-topography.
This is a minimal example - see dem_utils.py for more comprehensive functionality.
"""

from bmi_topography import Topography
import sys

def download_dem(lat_min, lat_max, lon_min, lon_max, output_file='dem.tif', dem_type='NASADEM'):
    """Download DEM for specified bounds"""
    try:
        topo = Topography(
            dem_type=dem_type,
            south=lat_min,
            north=lat_max,
            west=lon_min,
            east=lon_max,
            output_format='GTiff'
        )
        topo.output_file = output_file
        topo.fetch()
        print(f"DEM downloaded successfully: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error downloading DEM: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python get_dem.py <lat_min> <lat_max> <lon_min> <lon_max> <output_file>")
        print("Example: python get_dem.py 37.5 37.8 -122.5 -122.2 my_dem.tif")
        sys.exit(1)
    
    lat_min = float(sys.argv[1])
    lat_max = float(sys.argv[2])
    lon_min = float(sys.argv[3])
    lon_max = float(sys.argv[4])
    output_file = sys.argv[5]
    
    download_dem(lat_min, lat_max, lon_min, lon_max, output_file)
