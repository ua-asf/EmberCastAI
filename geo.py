from osgeo import gdal, ogr, osr
import math

cutline_shp = "/vsimem/cutline.shp"


def pix_to_lat_lon_basic(line, pixel, geomatrix, proj):
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
    return (lat, lon, line, pixel)


def pix_to_lat_lon(indataset, line, pixel):
    # Read geotransform matrix and calculate ground coordinates
    geomatrix = indataset.GetGeoTransform()
    proj = indataset.GetProjection()
    return pix_to_lat_lon_basic(line, pixel, geomatrix, proj)


def geocode_geotiff(scene, poly, dataset, output_file=None, width=None):
    output_file = output_file or f"/tmp/{scene}_geocoded.tif"

    print(f"Geocoding {dataset} to {output_file}")

    centroid = poly.centroid.coords[0]
    epsg_code = utm_from_lon_lat(centroid[0], centroid[1])

    params = {"dstNodata": None, "srcNodata": 0, "format": "gtiff"}
    if width:
        params["width"] = width

    if type(dataset) == str:
        pixel_sizes = get_image_pixel_size(dataset)
        params["xRes"] = pixel_sizes["x_size"]
        params["yRes"] = pixel_sizes["y_size"]

    print("Starting gdal.Warp()")
    gdal.Warp(output_file, dataset, dstSRS=f"EPSG:{epsg_code}", **params)
    print(f"Finished writing gdal.Warp() to {output_file}")
    dataset = None
    return output_file


def create_cutline_shp_file(
    crop_wkt, epsg_code, cutline_shp_path="/vsimem/cutline.shp"
):
    # Create a shapefile for cropping
    mem_shp = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(cutline_shp_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    lyr = mem_shp.CreateLayer("cutline", srs)
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


def get_image_pixel_size(image_path):
    # This wasn't pulling the proper value, its hard coded now
    print(f"Extracting pixel size from {image_path}")
    pixelSizeX = 10  # gt[1]
    pixelSizeY = 10  # -gt[5]

    print(f"XPixelSize/YPixelSize is {pixelSizeX}/{pixelSizeY} in {image_path}")

    return {"x_size": pixelSizeX, "y_size": pixelSizeY}


def crop_image(image_path, outfile, epsg_code, cutline_shp_path="/vsimem/cutline.shp"):
    pixel_sizes = get_image_pixel_size(image_path)

    return gdal.Warp(
        outfile,
        image_path,
        cutlineDSName=cutline_shp_path,
        dstSRS=f"EPSG:{epsg_code}",
        targetAlignedPixels=True,
        xRes=pixel_sizes["x_size"],
        yRes=pixel_sizes["y_size"],
        resampleAlg=None,
        cropToCutline=True,
        dstNodata=None,
    )


def utm_from_lon_lat(lon: float, lat: float) -> int:
    hemisphere = 32600 if lat >= 0 else 32700
    zone = int(lon // 6 + 30) % 60 + 1
    return hemisphere + zone


from osgeo import gdal, osr
import math


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
    """
    Crop and resample a GeoTIFF to specified coordinates with target resolution.

    Automatically reprojects to appropriate UTM zone for metric pixel sizing.

    Args:
        input_tiff_path: Path to input GeoTIFF file
        output_tiff_path: Path to output GeoTIFF file
        nw_latlon: Northwest corner (lat, lon) tuple
        ne_latlon: Northeast corner (lat, lon) tuple
        se_latlon: Southeast corner (lat, lon) tuple
        sw_latlon: Southwest corner (lat, lon) tuple
        pixel_size: Target pixel size in meters (default: 20)
        square_size: Output dimensions must be multiples of this value (default: 50)
    """
    src_ds = gdal.Open(input_tiff_path)
    if not src_ds:
        raise FileNotFoundError(f"Cannot open input file: {input_tiff_path}")

    # Calculate center point to determine UTM zone
    corners_latlon = [nw_latlon, ne_latlon, se_latlon, sw_latlon]
    avg_lat = sum(lat for lat, _ in corners_latlon) / 4
    avg_lon = sum(lon for _, lon in corners_latlon) / 4

    # Determine UTM zone and hemisphere
    utm_zone = int((avg_lon + 180) / 6) + 1
    is_north = avg_lat >= 0
    epsg_code = 32600 + utm_zone if is_north else 32700 + utm_zone

    # Create UTM spatial reference
    utm_srs = osr.SpatialReference()
    utm_srs.ImportFromEPSG(epsg_code)

    # Create WGS84 spatial reference
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # Transform corners from WGS84 to UTM
    transform = osr.CoordinateTransformation(wgs84_srs, utm_srs)
    corners_utm = [
        transform.TransformPoint(lon, lat)[:2] for lat, lon in corners_latlon
    ]

    # Calculate bounding box in UTM
    x_coords = [x for x, _ in corners_utm]
    y_coords = [y for _, y in corners_utm]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Calculate dimensions to ensure they're multiples of square_size
    width_m = max_x - min_x
    height_m = max_y - min_y

    width_px = int(math.ceil(width_m / pixel_size))
    height_px = int(math.ceil(height_m / pixel_size))

    width_px = ((width_px + square_size - 1) // square_size) * square_size
    height_px = ((height_px + square_size - 1) // square_size) * square_size

    # Adjust bounds to match the rounded dimensions
    max_x = min_x + (width_px * pixel_size)
    max_y = min_y + (height_px * pixel_size)

    # Warp to UTM with specified resolution
    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=utm_srs,
        outputBounds=(min_x, min_y, max_x, max_y),
        xRes=pixel_size,
        yRes=pixel_size,
        resampleAlg=gdal.GRA_Bilinear,
    )

    gdal.Warp(output_tiff_path, src_ds, options=warp_options)
    src_ds = None


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

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
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


def draw_wkt_to_geotiff(
    wkt_strs: list[str], input_file: str, output_file: str, fill_value: int = 65535
) -> None:
    """
    Adds a WKT mask layer as the first band of a new GeoTIFF.

    Args:
        wkt_strs: List of WKT polygon strings to rasterize
        input_file: Path to input GeoTIFF
        output_file: Path to output GeoTIFF
        fill_value: Value to burn into mask for matched geometries
    """
    src_ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    xsize = src_ds.RasterXSize
    ysize = src_ds.RasterYSize
    dtype = src_ds.GetRasterBand(1).DataType

    wkt_strs = [wkt for wkt in wkt_strs if ogr.CreateGeometryFromWkt(wkt) is not None]

    bands = src_ds.RasterCount + 1
    print(f"bands: {bands}")

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_file, xsize, ysize, bands, dtype, options=["COMPRESS=LZW", "TILED=YES"]
    )
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)

    # === STEP 1: Create and write mask to band 1 FIRST ===
    mem_driver = gdal.GetDriverByName("MEM")
    mask_ds = mem_driver.Create("", xsize, ysize, 1, gdal.GDT_UInt16)
    mask_ds.SetGeoTransform(geotransform)
    mask_ds.SetProjection(projection)
    mask_band = mask_ds.GetRasterBand(1)
    mask_band.Fill(0)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    mem_vector_ds = ogr.GetDriverByName("MEM").CreateDataSource("")
    mem_layer = mem_vector_ds.CreateLayer("layer", srs=srs, geom_type=ogr.wkbPolygon)

    for wkt in wkt_strs:
        geom = ogr.CreateGeometryFromWkt(wkt)
        if geom is None:
            continue
        feature = ogr.Feature(mem_layer.GetLayerDefn())
        feature.SetGeometry(geom)
        mem_layer.CreateFeature(feature)
        feature = None

    gdal.RasterizeLayer(
        mask_ds,
        [1],
        mem_layer,
        burn_values=[fill_value],
        options=["ALL_TOUCHED=TRUE"],
    )

    mask = mask_band.ReadAsArray()
    print(f"Mask min/max: {mask.min()}/{mask.max()}")

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(mask)
    out_band.FlushCache()
    out_band = None
    mask_band = None
    mask_ds = None
    mem_vector_ds = None

    # === STEP 2: Copy original bands sequentially ===
    for band_idx in range(1, src_ds.RasterCount + 1):
        print(f"Copying band {band_idx} of {src_ds.RasterCount}")
        in_band = src_ds.GetRasterBand(band_idx)
        data = in_band.ReadAsArray()
        print(f"Band {band_idx} min/max: {data.min()}/{data.max()}")

        out_band = out_ds.GetRasterBand(band_idx + 1)
        print(f"Writing input band {band_idx} to output band {band_idx + 1}")
        out_band.WriteArray(data)
        out_band.FlushCache()
        out_band = None
        in_band = None

    # Force final flush and close
    out_ds.FlushCache()
    out_ds = None
    src_ds = None

    # After calling the function:
    verify_ds = gdal.Open(output_file)
    for i in range(1, verify_ds.RasterCount + 1):
        band = verify_ds.GetRasterBand(i)
        data = band.ReadAsArray()
        print(f"Verify band {i}: min={data.min()}, max={data.max()}")
    verify_ds = None

    print(f"Finalized output saved to: {output_file}")


def merge_geotiffs(geotiffs, output_file):
    first_ds = gdal.Open(geotiffs[0])
    if first_ds is None:
        raise ValueError(f"Could not open {geotiffs[0]}")

    # Get metadata from first image
    xsize = first_ds.RasterXSize
    ysize = first_ds.RasterYSize
    projection = first_ds.GetProjection()
    geotransform = first_ds.GetGeoTransform()
    dtype = first_ds.GetRasterBand(1).DataType

    # Create output dataset with n bands
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_file, xsize, ysize, len(geotiffs), dtype)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)

    # Copy each geotiff to a band in the output
    for i, geotiff in enumerate(geotiffs):
        print(f"{i}: {geotiff}")
        ds = gdal.Open(geotiff, gdal.GA_Update)
        if ds is None:
            raise ValueError(f"Could not open {geotiff}")

        # Copy data to output band
        # All bands are 1-indexed in GDAL!!!
        out_band = out_ds.GetRasterBand(i + 1)

        in_band = ds.GetRasterBand(1)

        out_band.WriteArray(in_band.ReadAsArray())

        ds = None

    # Clean up
    out_ds.FlushCache()
    first_ds = None
    out_ds = None

    print(f"Merged {len(geotiffs)} geotiffs into: {output_file}")


import numpy as np


def convert_to_uint16(band):
    """
    Rescale a GDAL band to uint16 in-place with linear scaling.
    Maps min value -> 0, max value -> 65535.

    Args:
        band: GDAL raster band object
    """
    data = band.ReadAsArray()

    # Get min/max from valid data
    data_min, data_max = data.min(), data.max()

    # Linear rescale
    output = np.zeros(data.shape, dtype=np.uint16)
    if data_max > data_min:
        output = np.clip(
            (data - data_min) / (data_max - data_min) * 65535, 0, 65535
        ).astype(np.uint16)

    return output
