import os
import requests
import shutil
import asf_search as asf
from osgeo import gdal

def get_image_files (path):
    print(f"Reading files from {path}")
    files = gdal.ReadDir(path)
    print(f"files: {files}")
    VV = next ( ( z for z in files if '-vv-' in z ), None)
    VH = next ( ( z for z in files if '-vh-' in z ), None)

    return { 'VV': f'{path}/{VV}', 'VH': f'{path}/{VH}' }

def get_gdal_data (raster_path):

    data = gdal.Open(raster_path)

    ulx, xres, _, uly, _, yres  = data.GetGeoTransform()
    lrx = ulx + (data.RasterXSize * xres)
    lry = uly + (data.RasterYSize * yres)
    print(f"Upper Left: ({ulx}, {uly}); Lower Right: ({lrx},{lry})")
    return data

def download_local(scene, path='/tmp'):
    raise Exception(f"Scene {scene} not found")

def get_scene_data (scene):

    SCENE_URL = f'https://datapool.asf.alaska.edu/GRD_HD/{scene[0]}{scene[2]}/{scene}.zip'
    WORKING_DIR = os.getcwd()
    LOCAL_ZIP_FILE = f'{WORKING_DIR}/tmp/{scene}.zip'
    LOCAL_VSIZIP = f'/vsizip/{LOCAL_ZIP_FILE}'
    LOCAL_SAFE_DIR = f'{LOCAL_VSIZIP}/{scene}.SAFE'
    LOCAL_MEASURE_PATH = f'{LOCAL_SAFE_DIR}/measurement'


    if not os.path.exists(LOCAL_ZIP_FILE):
        download_local(SCENE_URL, path=f'{WORKING_DIR}/tmp')

    return LOCAL_MEASURE_PATH, LOCAL_ZIP_FILE
