#!/usr/bin/env python3
"""
Simplified DEM utilities for Docker environments.
Falls back to basic functionality if complex dependencies are not available.
"""

import os
import tempfile
import shutil
from osgeo import gdal

def simple_get_dem_for_fire_area(lat_min: float, lat_max: float, lon_min: float, lon_max: float, 
                                output_dir: str = None, dem_type: str = 'NASADEM') -> str:
    """
    Simplified DEM download for Docker environments.
    Uses basic bmi-topography functionality with fallbacks.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    dem_filename = f'dem_{dem_type.lower()}_{lat_min:.4f}_{lat_max:.4f}_{lon_min:.4f}_{lon_max:.4f}.tif'
    dem_path = os.path.join(output_dir, dem_filename)
    
    print(f"Downloading DEM data for bounds: lat({lat_min}, {lat_max}), lon({lon_min}, {lon_max})")
    print(f"Using DEM type: {dem_type}")
    
    try:
        # Try the full implementation first
        from bmi_topography import Topography
        
        topo = Topography(
            dem_type=dem_type,
            south=lat_min,
            north=lat_max,
            west=lon_min,
            east=lon_max,
            output_format='GTiff',
            cache_dir=output_dir
        )
        
        # Download and load the DEM data
        topo.fetch()
        topo.load()
        
        # Save using different methods based on what's available
        if hasattr(topo, 'da') and topo.da is not None:
            try:
                # Try rioxarray first
                topo.da.rio.to_raster(dem_path)
                print(f"DEM downloaded and saved successfully: {dem_path}")
                return dem_path
            except:
                # Fallback to basic numpy/gdal approach
                import numpy as np
                data = topo.da.values
                
                # Create GeoTIFF using GDAL
                driver = gdal.GetDriverByName('GTiff')
                rows, cols = data.shape
                
                # Create the dataset
                dataset = driver.Create(dem_path, cols, rows, 1, gdal.GDT_Float32)
                
                # Set geotransform
                geotransform = (
                    lon_min,  # x origin
                    (lon_max - lon_min) / cols,  # pixel width
                    0,  # rotation
                    lat_max,  # y origin
                    0,  # rotation
                    -(lat_max - lat_min) / rows  # pixel height (negative)
                )
                dataset.SetGeoTransform(geotransform)
                
                # Set projection to WGS84
                dataset.SetProjection('EPSG:4326')
                
                # Write the data
                band = dataset.GetRasterBand(1)
                band.WriteArray(data)
                band.SetNoDataValue(-9999)
                
                # Clean up
                dataset.FlushCache()
                dataset = None
                
                print(f"DEM downloaded and saved successfully (fallback method): {dem_path}")
                return dem_path
        else:
            raise ValueError("No DEM data was loaded")
            
    except Exception as e:
        print(f"Error downloading DEM with {dem_type}: {e}")
        
        # Try with SRTMGL3 as fallback
        if dem_type != 'SRTMGL3':
            print("Trying fallback DEM type SRTMGL3...")
            return simple_get_dem_for_fire_area(lat_min, lat_max, lon_min, lon_max, output_dir, 'SRTMGL3')
        else:
            # Final fallback - create a dummy DEM file with zeros
            print("Creating dummy DEM file (all zeros) as final fallback...")
            return create_dummy_dem(lat_min, lat_max, lon_min, lon_max, dem_path)

def create_dummy_dem(lat_min: float, lat_max: float, lon_min: float, lon_max: float, output_path: str) -> str:
    """Create a dummy DEM file filled with zeros for testing purposes."""
    import numpy as np
    
    # Create a small DEM (100x100 pixels)
    rows, cols = 100, 100
    data = np.zeros((rows, cols), dtype=np.float32)
    
    # Create GeoTIFF using GDAL
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
    
    # Set geotransform
    geotransform = (
        lon_min,  # x origin
        (lon_max - lon_min) / cols,  # pixel width
        0,  # rotation
        lat_max,  # y origin
        0,  # rotation
        -(lat_max - lat_min) / rows  # pixel height (negative)
    )
    dataset.SetGeoTransform(geotransform)
    
    # Set projection to WGS84
    dataset.SetProjection('EPSG:4326')
    
    # Write the data
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(-9999)
    
    # Clean up
    dataset.FlushCache()
    dataset = None
    
    print(f"Dummy DEM created: {output_path}")
    return output_path

# Use the simple version if we're in a constrained environment
try:
    from dem_utils import *
    print("Using full DEM utilities")
except ImportError:
    print("Using simplified DEM utilities")
    get_dem_for_fire_area = simple_get_dem_for_fire_area
