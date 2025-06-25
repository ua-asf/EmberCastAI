import sys

# Patch Requests if in pyscript
if sys.platform == 'emscripten':
    import pyodide_http
    pyodide_http.patch_all()

from PIL import Image, ImageOps
import numpy
import xml.etree.ElementTree as ET 

import zipfile
import glob
import shutil
import os
import sys
import base64
import json
from io import BytesIO

import requests
import asf_search
import logging
import gc


BROWSER = True if sys.platform == 'emscripten' else False

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def download_product(grd_product, edl_token, download_path='/tmp/data_download'):
    logging.info('Setting up download environment')
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    session = asf_search.ASFSession()

    logging.info(f'Logging into EDL w/ token {edl_token[0:10]}...')
    session.auth_with_token(edl_token)

    # Attempt to avoid
    if BROWSER:
        #dl_url = grd_product.properties["url"].replace("datapool", "sentinel1")
        #dl_url = grd_product.properties["url"].replace("datapool", "tea-test-jenk-0")
        #dl_url = dl_url.replace("SLC/SA", "SA/SLC") #test-test hack
        dl_url = grd_product.properties["url"]
        logging.info(f"Attepting HEAD request @ {dl_url}") 
        headers = {"Authorization": f"Bearer {edl_token}"}
        logging.info(requests.head(dl_url, headers=headers))

    logging.info(f'Downloading {grd_product.properties["fileID"]}')
    grd_product.download(path=download_path, session=session)

    return f'{download_path}/{grd_product.properties["sceneName"]}.zip'


def find_product(scene_name): 
    logging.info(f'Searching for scene {scene_name}')
    results = asf_search.granule_search([scene_name])

    # Cherry pick @ download the GRD product
    grd_product = [ r for r in results if r.properties['processingLevel'].startswith('GRD') ][0]
    #grd_product = [ r for r in results if r.properties['processingLevel'].startswith('SLC') ][0]
    logging.info(f'Found {scene_name} @ {grd_product.properties["url"]}')
    logging.info(json.dumps(grd_product.properties, indent=2))

    if BROWSER:
        grd_product.properties["url"] = f'https://local.asf.alaska.edu/download/{scene_name}.zip'

    return grd_product

def unzip_product(zip_path):

    download_path = os.path.split(zip_path)[0]

    # Unzip 
    logging.info(f'Extracting {zip_path} to {download_path}')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)

    # Remove original zip
    if BROWSER: 
        os.remove(zip_path)
    gc.collect()

    # So tacky
    return zip_path[0:-4] + ".SAFE"

def get_geo_points(hh_metadata_path):
    logging.info(f'Reading coords from {hh_metadata_path}')
    tree = ET.parse(hh_metadata_path)
    root = tree.getroot()
    points_path = './geolocationGrid/geolocationGridPointList/geolocationGridPoint'
    points_list = []
    for point in root.findall(points_path):
        points_list.append(
            {
                'lat': float(point.find('latitude').text),
                'lon': float(point.find('longitude').text),
                'y':   int(point.find('line').text),
                'x':   int(point.find('pixel').text),
            }
        )
    return points_list

def open_image (hh_image_path, product_path):
    logging.info(f'Reading product TIFF: {hh_image_path}')
    # Open image as a TIFF 

    # This throws "decompression bomb" warnings that can be ignored.
    im = Image.open(hh_image_path)

    # Delete the product to free space
    if BROWSER:
        shutil.rmtree(product_path)

    return im


def open_image_as_scaled_array(hh_image_path, product_path):
    im = open_image (hh_image_path, product_path)
    im = scale_image ( im )
    return numpy.array(im)

def open_image_as_array (hh_image_path, product_path):
    im = open_image (hh_image_path, product_path)
    return numpy.array(im)

def normalize_greys(imarray):
    average_pixel = imarray.mean() 
    logging.info(f'Normalizing greyscale to average pixel {average_pixel}')
 
    # Change to ints
    imarray = numpy.floor(imarray)
    gc.collect()

    imarray = (imarray/average_pixel)*255
    gc.collect()

    return imarray

def scale_image ( image ):
    logging.info('Scaling image by 1:10')
    scaled_size = [int(x/10) for x in list(image.size)]
    out_image_small = image.resize( scaled_size )
    del image
    gc.collect()
    # return numpy.array(out_image_small)
    return out_image_small

def create_output_image( imarray ):
    return Image.fromarray(numpy.uint8(imarray))

def mask_no_data_alpha ( out_image ):
    # Create mask
    logging.info('Generating no-data mask')
    mask = numpy.array(out_image)
    mask = numpy.where(mask > 1, 255, 0)
    mask_image = Image.fromarray(numpy.uint8(mask)).convert('L')

    del mask

    # Remove black borders 
    masked_image = out_image.convert('RGBA')
    masked_image.putalpha(mask_image)

    gc.collect()
    return masked_image

def create_geotiff_md (points_list):
    ### Create a geo-referenced image
    logging.info('Creating Geotiff Metadata')

    # Geotiff Tag Keys
    gtk = { 
        'GeoKeyDirectoryTag': 34735,
        'ModelTiepointTag': 33922,
        'GeoDoubleParamsTag': 34736,
        'GeoAsciiParamsTag': 34737,
    }

    # Add Tags
    tiff_info = {} 

    ### Use 1/10th of points as GCP (Method 2)
    gcp_points = []

    logging.info('Applying GCPs')

    #   GCP's are a tuple of
    #   ( X1, Y1, Z1, Lon1, Lat1, Z_cor1, X2, Y2, Z2, Lon2, Lat2, Z_cor2 ... )

    # Grab the coords of every 10th point from metadata
    for x in range(0,int(len(points_list)/10)):
       p = points_list[x*10]
       gcp_points += [float(int(p['x']/10)), float(int(p['y']/10)), 0.0, p['lon'], p['lat'], 0.0 ]

    # If we didn't get the last point, add it:
    if (len(points_list)/10)*10 < len(points_list):
       p = points_list[-1]
       gcp_points += [float(int(p['x']/10)), float(int(p['y']/10)), 0.0, p['lon'], p['lat'], 0.0 ]

    # Apply list of GCP's
    tiff_info[gtk['ModelTiepointTag']] = tuple (gcp_points)

    # Other GeoTIFF Headers 
    tiff_info[gtk['GeoKeyDirectoryTag']] = (
	1, 1, 0, 7, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326, 2049, 34737, 7, 0, 
        2054, 0, 1, 9102, 2057, 34736, 1, 1, 2059, 34736, 1, 0,
    )
    tiff_info[gtk['GeoDoubleParamsTag']] = (298.257223563, 6378137.0,)
    tiff_info[gtk['GeoAsciiParamsTag']] = ('WGS 84|',)

    return tiff_info

def save_geotiff( out_image, tiff_info, out_path):
    logging.info('Saving output product')
    out_image.save(out_path,  tiffinfo=tiff_info)

def orient (out_image):

    # Mirror data
    out_image_oriented = ImageOps.mirror(out_image)

    # Rotate 180
    out_image_oriented = out_image_oriented.rotate(190, Image.NEAREST, expand = 1)

    return out_image_oriented

def scale_to_proper_size ( hh_image_path, product_path ):

    imarray = open_image_as_array(hh_image_path, product_path)
    #imarry = scale_image ( imarray )
    gc.collect()
    imarray = normalize_greys( imarray )
    gc.collect()
    out_image = create_output_image( imarray )
    gc.collect()
    #out_image = scale_image ( out_image )
    out_image = scale_image ( out_image )
    return out_image

def scale_to_proper_size_lowmem ( hh_image_path, product_path ):
    imarray = open_image_as_scaled_array(hh_image_path, product_path)
    gc.collect()
    imarray = normalize_greys( imarray )
    gc.collect()
    return create_output_image( imarray )
    

def run_tif(scene_name, token, out_path='/tmp/georef_output.tif'):

    # Find product
    grd_product = find_product(scene_name)

    zip_path = download_product(grd_product, token)

    product_path = unzip_product(zip_path)

    hh_image_path = glob.glob(f'{product_path}/measurement/s1*-*-grd-hh-*.tiff')[0]
    hh_metadata_path = glob.glob(f'{product_path}/annotation/s1*-*-grd-hh-*.xml')[0]

    points_list = get_geo_points(hh_metadata_path)
   
    scale_imaged = scale_to_proper_size( hh_image_path, product_path )

    scale_imaged =  mask_no_data_alpha ( scale_imaged )

    # Create a geotiff metadata
    tiff_info = create_geotiff_md (points_list)

    # Save output to file
    save_geotiff( scale_imaged, tiff_info, out_path)

    # Show Oriented
    orient (scale_imaged).show()

    # Return image as data
    buffered = BytesIO()
    scale_imaged.save(buffered, format="PNG")
    data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"
