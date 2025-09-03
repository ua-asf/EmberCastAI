#!/usr/bin/env python3
"""
Utility functions for downloading and processing DEM data from OpenTopography
using the bmi-topography library to add as a 4th band to SAR GeoTIFFs.
"""

import os
import numpy as np
from osgeo import gdal, gdalconst
from bmi_topography import Topography
import tempfile
import shutil

# Try to import rioxarray for saving xarray data to raster
try:
    import rioxarray
except ImportError:
    print("Warning: rioxarray not installed. Some DEM functionality may be limited.")
    rioxarray = None

def get_dem_for_fire_area(lat_min: float, lat_max: float, lon_min: float, lon_max: float, 
                         output_dir: str = None, dem_type: str = 'NASADEM') -> str:
    """
    Download DEM data for a fire area using bmi-topography.
    
    Args:
        lat_min, lat_max, lon_min, lon_max: Bounding box coordinates
        output_dir: Directory to save the DEM file (optional)
        dem_type: DEM dataset to use ('NASADEM', 'SRTMGL3', 'SRTMGL1', etc.)
    
    Returns:
        str: Path to the downloaded DEM file
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
        # Initialize Topography object with correct parameters
        topo = Topography(
            dem_type=dem_type,
            south=lat_min,
            north=lat_max,
            west=lon_min,
            east=lon_max,
            output_format='GTiff',
            cache_dir=output_dir
        )
        
        # Download the DEM data
        topo.fetch()
        
        # Load the data into memory
        topo.load()
        
        # The data is now available as topo.da (xarray DataArray)
        if hasattr(topo, 'da') and topo.da is not None:
            # Save the merged DEM to our desired output file
            topo.da.rio.to_raster(dem_path)
            print(f"DEM downloaded and saved successfully: {dem_path}")
            return dem_path
        else:
            raise ValueError("No DEM data was loaded")
        
    except Exception as e:
        print(f"Error downloading DEM with {dem_type}: {e}")
        # Try with SRTMGL3 as fallback
        if dem_type != 'SRTMGL3':
            print("Trying fallback DEM type SRTMGL3...")
            return get_dem_for_fire_area(lat_min, lat_max, lon_min, lon_max, output_dir, 'SRTMGL3')
        else:
            raise e

def crop_dem_to_exact_bounds(dem_file: str, lat_min: float, lat_max: float, 
                           lon_min: float, lon_max: float, output_file: str = None) -> str:
    """
    Crop DEM to exact bounding box coordinates.
    
    Args:
        dem_file: Path to input DEM file
        lat_min, lat_max, lon_min, lon_max: Target bounding box
        output_file: Output file path (optional)
    
    Returns:
        str: Path to cropped DEM file
    """
    if output_file is None:
        base, ext = os.path.splitext(dem_file)
        output_file = f"{base}_cropped{ext}"
    
    # Use gdal.Warp to crop to exact bounds
    warp_options = gdal.WarpOptions(
        outputBounds=[lon_min, lat_min, lon_max, lat_max],
        format='GTiff',
        dstSRS='EPSG:4326'
    )
    
    gdal.Warp(output_file, dem_file, options=warp_options)
    print(f"DEM cropped to: {output_file}")
    return output_file

def resample_dem_to_match_sar(dem_file: str, sar_file: str, output_file: str = None) -> str:
    """
    Resample DEM to match the resolution and projection of SAR data.
    
    Args:
        dem_file: Path to DEM file
        sar_file: Path to SAR GeoTIFF file to match
        output_file: Output file path (optional)
    
    Returns:
        str: Path to resampled DEM file
    """
    if output_file is None:
        base, ext = os.path.splitext(dem_file)
        output_file = f"{base}_resampled{ext}"
    
    # Open SAR file to get target projection and resolution
    sar_ds = gdal.Open(sar_file)
    if sar_ds is None:
        raise ValueError(f"Could not open SAR file: {sar_file}")
    
    # Get SAR file properties
    sar_projection = sar_ds.GetProjection()
    sar_geotransform = sar_ds.GetGeoTransform()
    sar_xsize = sar_ds.RasterXSize
    sar_ysize = sar_ds.RasterYSize
    
    # Calculate target bounds in SAR projection
    ulx = sar_geotransform[0]
    uly = sar_geotransform[3]
    lrx = ulx + sar_geotransform[1] * sar_xsize
    lry = uly + sar_geotransform[5] * sar_ysize
    
    sar_ds = None
    
    # Resample DEM to match SAR
    warp_options = gdal.WarpOptions(
        dstSRS=sar_projection,
        xRes=abs(sar_geotransform[1]),
        yRes=abs(sar_geotransform[5]),
        outputBounds=[ulx, lry, lrx, uly],
        width=sar_xsize,
        height=sar_ysize,
        resampleAlg=gdalconst.GRA_Bilinear,
        format='GTiff'
    )
    
    gdal.Warp(output_file, dem_file, options=warp_options)
    print(f"DEM resampled to match SAR: {output_file}")
    return output_file

def add_dem_band_to_geotiff(sar_file: str, dem_file: str, output_file: str = None) -> str:
    """
    Add DEM data as an additional band to an existing SAR GeoTIFF.
    
    Args:
        sar_file: Path to SAR GeoTIFF file
        dem_file: Path to DEM file (should be resampled to match SAR)
        output_file: Output file path (optional)
    
    Returns:
        str: Path to output file with DEM band added
    """
    if output_file is None:
        base, ext = os.path.splitext(sar_file)
        output_file = f"{base}_with_dem{ext}"
    
    # Open source files
    sar_ds = gdal.Open(sar_file)
    dem_ds = gdal.Open(dem_file)
    
    if sar_ds is None:
        raise ValueError(f"Could not open SAR file: {sar_file}")
    if dem_ds is None:
        raise ValueError(f"Could not open DEM file: {dem_file}")
    
    # Get SAR file properties
    sar_bands = sar_ds.RasterCount
    xsize = sar_ds.RasterXSize
    ysize = sar_ds.RasterYSize
    projection = sar_ds.GetProjection()
    geotransform = sar_ds.GetGeoTransform()
    
    # Get data type from first SAR band
    sar_band = sar_ds.GetRasterBand(1)
    dtype = sar_band.DataType
    
    # Create output dataset with SAR bands + 1 DEM band
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_file, xsize, ysize, sar_bands + 1, dtype)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    
    # Copy SAR bands
    for i in range(1, sar_bands + 1):
        sar_band = sar_ds.GetRasterBand(i)
        out_band = out_ds.GetRasterBand(i)
        data = sar_band.ReadAsArray()
        out_band.WriteArray(data)
        
        # Copy band metadata
        out_band.SetDescription(sar_band.GetDescription() or f"SAR_Band_{i}")
    
    # Add DEM as the last band
    dem_band = dem_ds.GetRasterBand(1)
    out_dem_band = out_ds.GetRasterBand(sar_bands + 1)
    dem_data = dem_band.ReadAsArray()
    
    # Handle nodata values in DEM
    dem_nodata = dem_band.GetNoDataValue()
    if dem_nodata is not None:
        # Replace nodata with 0 or interpolate
        dem_data = np.where(dem_data == dem_nodata, 0, dem_data)
    
    out_dem_band.WriteArray(dem_data)
    out_dem_band.SetDescription("DEM_Elevation")
    
    # Clean up
    out_ds.FlushCache()
    sar_ds = None
    dem_ds = None
    out_ds = None
    
    print(f"Added DEM band to SAR file: {output_file}")
    return output_file

def process_dem_for_fire(fire_bounds: tuple, sar_file: str, output_dir: str = None) -> str:
    """
    Complete pipeline to download, process, and add DEM data to SAR file.
    
    Args:
        fire_bounds: Tuple of (lat_min, lat_max, lon_min, lon_max)
        sar_file: Path to SAR GeoTIFF file
        output_dir: Directory for intermediate files (optional)
    
    Returns:
        str: Path to SAR file with DEM band added
    """
    lat_min, lat_max, lon_min, lon_max = fire_bounds
    
    if output_dir is None:
        output_dir = os.path.dirname(sar_file)
    
    temp_dir = os.path.join(output_dir, 'temp_dem')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Step 1: Download DEM
        print("1. Downloading DEM data...")
        dem_file = get_dem_for_fire_area(lat_min, lat_max, lon_min, lon_max, temp_dir)
        
        # Step 2: Crop to exact bounds
        print("2. Cropping DEM to exact bounds...")
        cropped_dem = crop_dem_to_exact_bounds(dem_file, lat_min, lat_max, lon_min, lon_max,
                                             os.path.join(temp_dir, 'cropped_dem.tif'))
        
        # Step 3: Resample to match SAR
        print("3. Resampling DEM to match SAR resolution...")
        resampled_dem = resample_dem_to_match_sar(cropped_dem, sar_file,
                                                os.path.join(temp_dir, 'resampled_dem.tif'))
        
        # Step 4: Add DEM band to SAR
        print("4. Adding DEM band to SAR file...")
        output_file = add_dem_band_to_geotiff(sar_file, resampled_dem, sar_file)
        
        print(f"✓ Successfully added DEM band to {output_file}")
        return output_file
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary DEM files")

def merge_geotiffs_with_dem(geotiffs: list, output_file: str, dem_bounds: tuple = None):
    """
    Enhanced version of merge_geotiffs that can handle DEM data.
    
    Args:
        geotiffs: List of GeoTIFF file paths to merge
        output_file: Output merged file path
        dem_bounds: Optional tuple of (lat_min, lat_max, lon_min, lon_max) for DEM
    """
    first_ds = gdal.Open(geotiffs[0])
    if first_ds is None:
        raise ValueError(f"Could not open {geotiffs[0]}")

    # Get metadata from first image
    xsize = first_ds.RasterXSize
    ysize = first_ds.RasterYSize
    projection = first_ds.GetProjection()
    geotransform = first_ds.GetGeoTransform()
    dtype = first_ds.GetRasterBand(1).DataType

    # Determine if we should add DEM band
    add_dem = dem_bounds is not None
    total_bands = len(geotiffs) + (1 if add_dem else 0)

    # Create output dataset
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_file, xsize, ysize, total_bands, dtype)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)

    # Copy each geotiff to a band in the output
    for i, geotiff in enumerate(geotiffs, 1):
        ds = gdal.Open(geotiff)
        if ds is None:
            raise ValueError(f"Could not open {geotiff}")
        
        # Copy data to output band
        out_band = out_ds.GetRasterBand(i)
        data = ds.GetRasterBand(1).ReadAsArray()
        out_band.WriteArray(data)
        
        # Set band description
        band_name = os.path.basename(geotiff).split('.')[0].upper()
        out_band.SetDescription(band_name)
        
        ds = None

    # Add DEM band if requested
    if add_dem:
        print("Adding DEM band to merged file...")
        temp_merged = output_file + "_temp.tiff"
        os.rename(output_file, temp_merged)
        
        try:
            final_output = process_dem_for_fire(dem_bounds, temp_merged, os.path.dirname(output_file))
            if final_output != output_file:
                os.rename(final_output, output_file)
        finally:
            if os.path.exists(temp_merged):
                os.remove(temp_merged)

    # Clean up
    out_ds.FlushCache()
    first_ds = None
    out_ds = None

    band_desc = f"{len(geotiffs)} SAR bands" + (" + DEM" if add_dem else "")
    print(f"Merged {len(geotiffs)} geotiffs with {band_desc} into: {output_file}")
