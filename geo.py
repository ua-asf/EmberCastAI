import os
import logging
from osgeo import ogr, osr, gdal

log = logging.getLogger(__name__)

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
        log.error(f"Cannot import projection '{proj}'")

    srsLatLong = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srs, srsLatLong)
    (lat, lon, _) = ct.TransformPoint(X, Y)
    log.debug(f'latitude: {lat}\tlongitude: {lon}')
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

    # if os.path.exists( output_file):
    #     print(f"Reusing previously geocoded image {output_file}")
    #     return output_file

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
    pixel_size=20
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


