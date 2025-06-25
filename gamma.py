import logging
import os
import zipfile
import tempfile
import numpy
import glob
from osgeo import gdal
from math import ceil, floor

WKT = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["unnamed",6378137,298.25722356049,' + \
      'AUTHORITY["EPSG","4326"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UN' + \
      'IT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NOR' + \
      'TH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'

log = logging.getLogger(__name__)

def run(cmd):
    os.system(cmd)

def overwrite_tiff(tif_file, output, data_file):

    # find the original size
    ds = gdal.Open(tif_file)
    y_size = ds.RasterXSize  # Width (DOES change)
    x_size = ds.RasterYSize  # Lines (Does not change)

    # Extract GCP's for later.
    original_gcp = ds.GetGCPs()
    try:
        original_gcpproj = ds.GetGCPProjection()
    except RuntimeError:
        log.error("Could not get default GCP Projection, using espg:4326")
        original_gcpproj = WKT
    ds = None

    # Read in the data
    print(f"reading data from {data_file}")
    data = numpy.fromfile(data_file, dtype=numpy.float32)

    new_width = ceil ( len(data) / x_size )
    y_scale_factor = new_width / y_size
    print(f"Data was stetched {y_size} -> {new_width}, scaling factor: {y_scale_factor:0.4f}")

    data = numpy.reshape(data, (x_size, new_width))

    MAX_VALUE = 2048
    print("Byteswapping data...")
    data = data.byteswap()
    # Converting from 0-1 float to 0-MAX_VALUE float
    print(f"Converting data to 0-{MAX_VALUE} ")
    data = data * MAX_VALUE
    # Formating as int
    print("Converting data to int ")
    data = data.astype(numpy.int16)
    # Clip to MAX_VALUE
    print(f"Clipping rouge values over {MAX_VALUE}")
    #data = numpy.putmask(data, data>MAX_VALUE, 0)
    data = numpy.clip(data, 0, MAX_VALUE)

    original_data_size = y_size * x_size
    print(f"Original data was {y_size} x {x_size} ({original_data_size}b), in data is {len(data)}")

    # Scale the GCP's by y_scale_factor:
    new_gcps = []
    print(f"Shifting GCP's in the Y direction by {y_scale_factor}")
    for gcp in original_gcp:
        # 'GCPLine', 'GCPPixel', 'GCPX', 'GCPY', 'GCPZ', 'Id',
        # gdal.GCP(x, y, z, pixel, line)
        new_y_value = floor(gcp.GCPPixel * y_scale_factor)
        log.debug(f"Shifting (x{gcp.GCPX},y{gcp.GCPY},z{gcp.GCPZ}) from ({gcp.GCPLine},{gcp.GCPPixel}) to ({gcp.GCPLine},{new_y_value})")
        new = gdal.GCP(gcp.GCPX, gcp.GCPY, gcp.GCPZ, new_y_value, gcp.GCPLine )
        log.debug(f"Old: {gcp}; New: {new}")
        new_gcps.append(new)

    print("Create a NEW geotiff with all our params and GCPs!")
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output, new_width, x_size, 1, gdal.GDT_Int16)

    outdata.GetRasterBand(1).WriteArray(data)
    outdata.SetGCPs( new_gcps, original_gcpproj )

    #free them
    new_gcps = None
    original_gcpproj = None
    outdata = None
    driver = None

    print(f"Completed creation of new geotiff @ {output}")

    return output

def apply_calibation(safe_dir, output, pol='vv'):

    pol = pol.lower()
    annotation_xml = f'{safe_dir}/annotation/*-{pol}-*.xml'
    calibration_xml = f'{safe_dir}/annotation/calibration/calibration*-{pol}-*.xml'
    noise_xml = f'{safe_dir}/annotation/calibration/noise*-{pol}-*.xml'
    tiff = f'{safe_dir}/measurement/*-{pol}-*.tiff'

    par_file = f'{output}.par'
    data_file = f'{output}.data'
    if not os.path.exists( data_file ):
        par_cmd = f'par_S1_GRD {tiff} {annotation_xml} {calibration_xml} {noise_xml} {par_file} {data_file}'
        print(f"Running command {par_cmd}")
        run(par_cmd)

    # copy the raw data output from par_S1_GRD overtop of the GRD tif
    tif_file = glob.glob(tiff)[0]
    output = overwrite_tiff(tif_file, output, data_file)

    return output

def extract_calibrated_tiff(zip_file, scene):
    zf = zipfile.ZipFile(zip_file)

    output = f"/tmp/{scene}_cal.tif"
    if os.path.exists(output):
        print(f"Reusing old {output} calibrated prodcut")
        return output

    print(f"Extracting SAFE dir from {zip_file}")
    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)

        calibrated_tif = apply_calibation( f"{tempdir}/{scene}.SAFE", output)
        print(f"calibrated_tif is {calibrated_tif}")

        return calibrated_tif

    log.error("Could not generate calibrated data")
    return None

if __name__ == "__main__":

    # Set up logging.
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.INFO)
    SCENE = 'S1A_IW_GRDH_1SDV_20190827T175044_20190827T175109_028758_0341BF_2193'
    ZIP_NAME = '/tmp/S1A_IW_GRDH_1SDV_20190827T175044_20190827T175109_028758_0341BF_2193.zip'
    print(f"calibrating {SCENE} @ {ZIP_NAME}")
    extract_calibrated_tiff( ZIP_NAME, SCENE )


