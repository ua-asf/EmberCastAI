import os
import shutil
from osgeo import gdal, ogr, osr
import math

cutline_shp = '/vsimem/cutline.shp'

def pix_to_lat_lon_basic ( line, pixel, geomatrix, proj ):
    # Read geotransform matrix and calculate ground coordinates
    X = geomatrix[0] + geomatrix[1] * pixel + geomatrix[2] * line
    Y = geomatrix[3] + geomatrix[4] * pixel + geomatrix[5] * line

        # Shift to the center of the pixel
    X += geomatrix[1] / 2.0
    Y += geomatrix[5] / 2.0

    # Build Spatial Reference object based on coordinate system, fetched from the
    # opened dataset
    srs = osr.SpatialReference()
    if srs.ImportFromWkt(proj) != 0:
        print(f"Cannot import projection '{proj}'")

    srsLatLong = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srs, srsLatLong)
    (lat, lon, _) = ct.TransformPoint(X, Y)
    return (lat, lon,  line, pixel )


def pix_to_lat_lon( indataset, line, pixel ):
    # Read geotransform matrix and calculate ground coordinates
    geomatrix = indataset.GetGeoTransform()
    proj = indataset.GetProjection()
    return pix_to_lat_lon_basic ( line, pixel, geomatrix, proj )

def geocode_geotiff(scene, poly, dataset, output_file=None, width=None):

    output_file = output_file or f'/tmp/{scene}_geocoded.tif'

    print(f"Geocoding {dataset} to {output_file}")

    centroid = poly.centroid.coords[0]
    epsg_code = utm_from_lon_lat(centroid[0],centroid[1])

    params = {  "dstNodata": None, "srcNodata": 0, "format": 'gtiff'}
    if width:
        params['width'] = width

    if type(dataset) == str:
        pixel_sizes = get_image_pixel_size (dataset)
        params["xRes"] = pixel_sizes["x_size"]
        params["yRes"] = pixel_sizes["y_size"]

    print("Starting gdal.Warp()")
    gdal.Warp(output_file, dataset, dstSRS=f'EPSG:{epsg_code}', **params )
    print(f"Finished writing gdal.Warp() to {output_file}")
    dataset = None
    return output_file

def create_cutline_shp_file(crop_wkt, epsg_code, cutline_shp_path='/vsimem/cutline.shp'):

    # Create a shapefile for cropping
    mem_shp = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(cutline_shp_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    lyr = mem_shp.CreateLayer('cutline', srs)
    f = ogr.Feature(lyr.GetLayerDefn())

    # Translate WKT (4326) to UTM
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(4326)
    inSpatialRef.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg_code)
    outSpatialRef.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    crop_shape = ogr.CreateGeometryFromWkt(crop_wkt)
    crop_shape.Transform(coordTrans)

    # Add Geometery to shapefile
    f.SetGeometry(crop_shape)
    lyr.CreateFeature(f)
    return cutline_shp_path

def get_image_pixel_size (image_path):
    # This wasn't pulling the proper value, its hard coded now
    print(f"Extracting pixel size from {image_path}")
    pixelSizeX = 10 #gt[1]
    pixelSizeY = 10 #-gt[5]

    print(f"XPixelSize/YPixelSize is {pixelSizeX}/{pixelSizeY} in {image_path}")

    return {"x_size":pixelSizeX, "y_size":pixelSizeY}

def crop_image(image_path, outfile, epsg_code, cutline_shp_path='/vsimem/cutline.shp'):

    pixel_sizes = get_image_pixel_size (image_path)

    return gdal.Warp(outfile, image_path, \
                     cutlineDSName=cutline_shp_path, \
                     dstSRS=f'EPSG:{epsg_code}', \
                     targetAlignedPixels=True, \
                     xRes=pixel_sizes["x_size"], yRes=pixel_sizes["y_size"], \
                     resampleAlg=None, \
                     cropToCutline=True, \
                     dstNodata=None)

def utm_from_lon_lat(lon: float, lat: float) -> int:
    hemisphere = 32600 if lat >= 0 else 32700
    zone = int(lon // 6 + 30) % 60 + 1
    return hemisphere + zone

from osgeo import gdal, osr

def crop_and_scale_to_20x20(
    input_tiff_path,
    output_tiff_path,
    nw_latlon,
    ne_latlon,
    se_latlon,
    sw_latlon,
    pixel_size=20,
    square_size=50,
):
    ds = gdal.Open(input_tiff_path)
    if ds is None:
        raise FileNotFoundError(f"Could not open input file: {input_tiff_path}")

    raster_wkt = ds.GetProjection()
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_wkt)

    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(source_srs, raster_srs)

    def project(lon, lat):
        x, y, _ = transform.TransformPoint(lon, lat)
        return x, y

    # Project corners and print for debugging
    corners_ll = [nw_latlon, ne_latlon, se_latlon, sw_latlon]
    corners_proj = [project(lon, lat) for lon, lat in corners_ll]

    x_vals = [pt[0] for pt in corners_proj]
    y_vals = [pt[1] for pt in corners_proj]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)

    # Ensure we get a square image - translate to pixels
    min_x_pixel = min_x / pixel_size
    max_x_pixel = max_x / pixel_size
    min_y_pixel = min_y / pixel_size
    max_y_pixel = max_y / pixel_size

    width = max_x_pixel - min_x_pixel
    height = max_y_pixel - min_y_pixel

    new_width = max(width, height)

    new_width = math.ceil(new_width / square_size) * square_size

    new_width_diff = (new_width - width) / 2

    min_x = (min_x_pixel - new_width_diff) * pixel_size
    max_x = (max_x_pixel + new_width_diff) * pixel_size

    new_height_diff = (new_width - height) / 2

    min_y = (min_y_pixel - new_height_diff) * pixel_size
    max_y = (max_y_pixel + new_height_diff) * pixel_size

    # Final warp
    result = gdal.Warp(
        output_tiff_path,
        input_tiff_path,
        xRes=pixel_size,
        yRes=pixel_size,
        outputBounds=(min_x, min_y, max_x, max_y),
        resampleAlg='bilinear',
        dstSRS=raster_srs.ExportToWkt(),
        format='GTiff'
    )

    if result is None:
        raise RuntimeError("gdal.Warp failed.")

    print(f"Output saved to: {output_tiff_path}")


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the distance between two points on Earth using the Haversine formula.

    Args:
        lat1: Latitude of the first point in degrees.
        lon1: Longitude of the first point in degrees.
        lat2: Latitude of the second point in degrees.
        lon2: Longitude of the second point in degrees.

    Returns:
        The distance in meters.
    """
    radius = 6371000  # Earth radius in meters

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = radius * c
    return distance

def get_utm_epsg(lat, lon):
    if not (-80 <= lat <= 84):
        raise ValueError("Latitude out of UTM bounds")
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone  # Northern hemisphere
    else:
        return 32700 + zone  # Southern hemisphere

def draw_wkt_to_geotiff(wkt_strs: list[str], input_file: str, output_file: str, fill_value: int = 255):

    # Open input raster
    src_ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    xsize = src_ds.RasterXSize
    ysize = src_ds.RasterYSize

    wkt_strs = [wkt for wkt in wkt_strs if ogr.CreateGeometryFromWkt(wkt) is not None]

    # Get data type from first band
    in_band = src_ds.GetRasterBand(1)
    dtype = in_band.DataType

    # Keep same number of bands (don't add extra)
    bands = src_ds.RasterCount

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_file, xsize, ysize, bands, dtype)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)

    # Copy ALL input bands to output
    for band_idx in range(1, bands + 1):
        in_band = src_ds.GetRasterBand(band_idx)
        data = in_band.ReadAsArray()
        out_band = out_ds.GetRasterBand(band_idx)
        out_band.WriteArray(data)

    # Create mask for WKT shapes
    mem_driver = gdal.GetDriverByName("MEM")
    mask_ds = mem_driver.Create("", xsize, ysize, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(geotransform)
    mask_ds.SetProjection(projection)
    mask_band = mask_ds.GetRasterBand(1)
    mask_band.Fill(fill_value)

    # Prepare layer and geometries
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    
    # Get MEM driver with error checking
    mem_driver = ogr.GetDriverByName("MEM")
    if mem_driver is None:
        mem_driver = ogr.GetDriverByName("Memory")
    if mem_driver is None:
        raise RuntimeError("Could not get MEM or Memory driver from OGR")
    
    mem_vector_ds = mem_driver.CreateDataSource("")
    mem_layer = mem_vector_ds.CreateLayer("layer", srs=srs, geom_type=ogr.wkbPolygon)

    for wkt in wkt_strs:
        geom = ogr.CreateGeometryFromWkt(wkt)
        if geom is None:
            continue
        feature = ogr.Feature(mem_layer.GetLayerDefn())
        feature.SetGeometry(geom)
        mem_layer.CreateFeature(feature)

    # Rasterize WKT shapes into mask (value = 1)
    gdal.RasterizeLayer(mask_ds, [1], mem_layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
    mask = mask_band.ReadAsArray()

    # Write WKT mask to band 3 (between SAR bands 1,2 and DEM band 4)
    if bands >= 3:
        out_band3 = out_ds.GetRasterBand(3)
        out_band3.WriteArray(mask)
    else:
        # Fallback for files with fewer bands
        out_band2 = out_ds.GetRasterBand(2)
        out_band2.WriteArray(mask)

    # Clean up
    out_ds.FlushCache()
    src_ds = None
    out_ds = None

    print(f"Finalized output saved to: {output_file}")

def merge_geotiffs(geotiffs, output_file, dem_bounds=None):
    """
    Merge multiple GeoTIFF files into a single multi-band GeoTIFF.
    
    Args:
        geotiffs: List of GeoTIFF file paths to merge
        output_file: Output merged file path
        dem_bounds: Optional tuple of (lat_min, lat_max, lon_min, lon_max) to add DEM band
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

    # Create output dataset with n bands (+ optional DEM band)
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
        
        # Set band description based on filename
        band_name = os.path.basename(geotiff).split('.')[0].upper()
        out_band.SetDescription(band_name)
        
        ds = None

    # Add DEM band if requested
    if add_dem:
        print("Adding DEM band to merged file...")
        temp_merged = output_file + "_temp.tiff"
        
        # Close current output dataset
        out_ds.FlushCache()
        out_ds = None
        first_ds = None
        
        # Rename current output to temp
        os.rename(output_file, temp_merged)
        
        try:
            # Import here to avoid circular imports
            # Try full version first, then fallback to simple version
            try:
                from dem_utils import process_dem_for_fire
            except ImportError:
                print("Using simplified DEM processing for Docker environment")
                from dem_utils_simple import simple_get_dem_for_fire_area
                from dem_utils import add_dem_band_to_geotiff, resample_dem_to_match_sar
                
                # Simplified process_dem_for_fire equivalent
                def process_dem_for_fire(fire_bounds, sar_file, output_dir):
                    lat_min, lat_max, lon_min, lon_max = fire_bounds
                    temp_dir = os.path.join(output_dir, 'temp_dem')
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    try:
                        # Download DEM
                        dem_file = simple_get_dem_for_fire_area(lat_min, lat_max, lon_min, lon_max, temp_dir)
                        
                        # Resample to match SAR
                        resampled_dem = resample_dem_to_match_sar(dem_file, sar_file,
                                                                os.path.join(temp_dir, 'resampled_dem.tif'))
                        
                        # Add DEM band to SAR
                        output_file = add_dem_band_to_geotiff(sar_file, resampled_dem, sar_file)
                        return output_file
                    finally:
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
            
            final_output = process_dem_for_fire(dem_bounds, temp_merged, os.path.dirname(output_file))
            if final_output != output_file:
                os.rename(final_output, output_file)
        finally:
            if os.path.exists(temp_merged):
                os.remove(temp_merged)
    else:
        # Clean up normally if no DEM
        out_ds.FlushCache()
        first_ds = None
        out_ds = None

    band_desc = f"{len(geotiffs)} SAR bands" + (" + DEM" if add_dem else "")
    print(f"Merged {len(geotiffs)} geotiffs with {band_desc} into: {output_file}")
