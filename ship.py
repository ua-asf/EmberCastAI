import os
import numpy
import matplotlib
from osgeo import gdal
from matplotlib import pyplot, patches
from matplotlib.collections import PatchCollection
from shapely.geometry import Point
from cmr import get_scene_metadata
from geo import geocode_geotiff, crop_image, pix_to_lat_lon_basic
from s3 import download_file
from cde_data import get_scene_data, get_image_files, get_gdal_data
from gamma import extract_calibrated_tiff
from ais import match_ships
from skimage import measure

#################################
#
# Tweakable parameters
#

BRIGHT_THRESHOLD = 6.0         # Number of STD's less than mean-adjusted,
                               # outlier-removed max value of dataset
                               # Higher Num = More Ships / False Positives
LABEL_THRESHOLD = 5            # Number of pixels that make a ship
OUTLIER_STD_THRESHOLD = 25     # Std deviations past mean to clip data
CALIBRATE_DATA = False         # Apply sentinel calibration LUTs
NORM_SCENE1_TO_SCENE2 = True   # Roughly match scene min/max's

# Experimental
ARBITRARY_PIXEL_CUTOFF = 1700
ARBITRARY_BRIGHT_PIXEL = 420

#
#####

def utm_from_lon_lat(lon: float, lat: float) -> int:
    hemisphere = 32600 if lat >= 0 else 32700
    zone = int(lon // 6 + 30) % 60 + 1
    return hemisphere + zone

def generate_geocoded_grd( granule, out_dir=None ):

    print(f"Loading raster data for dataset {granule}")
    print("Querying CMR for scene info")
    _, _, _, epsg_code_poly, _ = get_scene_metadata (granule)

    centroid = epsg_code_poly.centroid.coords[0]
    epsg_code = utm_from_lon_lat(centroid[0],centroid[1])
    print (f"Data is in UTM zone/EPSG {epsg_code}")

    subset_product = out_dir or f'/tmp/{granule}'
    
    # if os.path.exists(subset_product):
    #     print(f"Reusing previously subsetted product {subset_product}")
    #     return

    print(f"Local fetching {granule} dataset")
    local_measure_path, zip_file = get_scene_data(granule)

    if CALIBRATE_DATA: # Make true to do calibration stuff
        print(f"Calibrating {granule}")
        calibrated = extract_calibrated_tiff(zip_file, granule)
        print(f"Loading calibrated data from {calibrated}")
        #data = get_gdal_data (calibrated)
        data = [calibrated]
    else:
        print(f"Getting tiff list for {local_measure_path}")
        files = get_image_files (local_measure_path)
        print("Loading uncalibrated VV and VH tiffs")
        #data = get_gdal_data (files['VV'])
        data = [files['VV'], files['VH']]

    print("Geocoding products")

    product_paths = []

    for product in data:
        product_name = product.split('.')[-2].split('/')[-1]

        gcgt = geocode_geotiff(granule, epsg_code_poly, product, output_file=f'{subset_product}/{product_name}.tiff')
        product_paths.append((f'{subset_product}/{product_name}.tiff', epsg_code))
        print(f"Completed geocoding {product}, subsetting to {subset_product}/{product_name}.tiff")

    return product_paths

def get_dataset_as_array ( dataset ):
    if type(dataset) == str:
       dataset = get_gdal_data(dataset)
    
    print("Reading Band")
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray().astype(numpy.int16)
    print(f"Clipping data to {ARBITRARY_PIXEL_CUTOFF}")
    arr = numpy.clip(arr, 0, ARBITRARY_PIXEL_CUTOFF)
    return arr

def get_geo_trans (dataset):
    if type(dataset) == str:
       dataset = get_gdal_data(dataset)
    
    geomatrix = dataset.GetGeoTransform()
    proj = dataset.GetProjection()

    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize

    return geomatrix, proj, x_size, y_size

def dump_data_to_image(data, name=None):

    name = name or 'plot.jpg'
    out_name = f'/tmp/{name}'
    print(f"Writing {out_name}")

    fig, ax = pyplot.subplots()
    im = ax.imshow(data)
    cmap = pyplot.get_cmap('viridis',100)
    norm = matplotlib.colors.Normalize(vmin=0,vmax=1)
    fig.colorbar(pyplot.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    im.set_clim(0,1)
    fig.savefig(out_name)

    print(f"Finished writing {out_name}")

def erase_beach_noise (arr, waterMask):

    MIN_NOISE_BRIGHTNESS = 150
    MAX_NOISE_BRIGHTNESS = 900
    TOO_BIG_TO_BE_A_SHIP = 750
    NOISE_ERASER = 20

    print(f"Removing blocks of +{TOO_BIG_TO_BE_A_SHIP} pixels between {MIN_NOISE_BRIGHTNESS} and {MAX_NOISE_BRIGHTNESS}")
    is_dull = numpy.logical_and( arr > MIN_NOISE_BRIGHTNESS, arr < MAX_NOISE_BRIGHTNESS)
    is_dull = numpy.logical_and(is_dull,waterMask)

    dump_data_to_image(is_dull, 'is_dull.png')

    print("Grouping Pixels")
    labels, num_labels = measure.label(is_dull, return_num=True)
    print("Building beach noise histogram")
    histogram, _ = numpy.histogram(labels,bins=num_labels,range=(0.5,num_labels+0.5))
    labels_pass = numpy.nonzero(histogram>=TOO_BIG_TO_BE_A_SHIP)[0]+1
    histogram = None
    label_count = len(labels_pass)

    print("Filtering Pixels")

    chunk_count = 0
    is_labels_pass = numpy.full(arr.shape, False, dtype=bool)
    for ii in labels_pass:
        chunk_count += 1
        is_ii = labels==ii
        #is_ii = numpy.isin(labels,ii)
        #chunk_size = numpy.sum(is_ii)
        chunk_size = numpy.count_nonzero(is_ii)
        if chunk_size > TOO_BIG_TO_BE_A_SHIP:
            print(f"Removing noise chunk {ii} {chunk_count}/{label_count}. Size: {chunk_size}")
            is_labels_pass = numpy.logical_or(is_labels_pass,is_ii)
    arr[is_labels_pass] = NOISE_ERASER

    return arr

def basic_is_bright(dataset_01, dataset_02, waterMask):
    print("Reading Dataset 2")
    arr_02 = get_dataset_as_array ( dataset_02 )
    arr_02 = erase_beach_noise( arr_02, waterMask )
     

    print("Reading Dataset 1")
    arr_01 = get_dataset_as_array ( dataset_01 )

    print("Getting basic geometry for Dataset 2")
    geomatrix, proj, x_size, y_size = get_geo_trans (dataset_02)

    print("Calculating Difference Array")
    diffArr = numpy.subtract(arr_02, arr_01, where=waterMask)

    isBright = diffArr > ARBITRARY_BRIGHT_PIXEL

    return isBright, int(numpy.minimum(x_size, y_size)/50.), geomatrix, proj

def big_is_bright(dataset_01, dataset_02, waterMask):

    print("Gettings both bands....")

    print("Reading Dataset 1")
    if type(dataset_01) == str:
       dataset_01 = get_gdal_data(dataset_01)
    band_01 = dataset_01.GetRasterBand(1)
    (min1,max1,mean1,std1) = band_01.ComputeStatistics(False)
    print(f"Band 1: min/max/mean/std: {min1}/{max1}/{mean1}/{std1}")
    print("Reading Band 1")
    arr_01 = band_01.ReadAsArray().astype(numpy.int16)

    print("Calculating watermasted Mean for dataset1...")
    mean1 = numpy.mean(arr_01, where=waterMask)
    print("Calculating watermasted Std for dataset1...")
    std1 = numpy.std(arr_01, where=waterMask)
    print("Calculating watermasted min/max for dataset1...")
    min1 = numpy.amin(arr_01, initial=10000, where=waterMask)
    max1 = numpy.amax(arr_01, initial=1, where=waterMask)
    print(f"Band 1 WM'd: min/max/mean/std: {min1}/{max1}/{mean1}/{std1}")

    band_01 = None

    print("Reading Dataset 2")
    if type(dataset_02) == str:
       dataset_02 = get_gdal_data(dataset_02)

    # for pixel lookup later
    geomatrix = dataset_02.GetGeoTransform()
    proj = dataset_02.GetProjection()
    band_02 = dataset_02.GetRasterBand(1)
    (min2,max2,mean2,std2) = band_02.ComputeStatistics(False)
    print(f"Band 2: min/max/mean/std: {min2}/{max2}/{mean2}/{std2}")
    print("Reading Band 2")
    x_size = dataset_02.RasterXSize
    y_size = dataset_02.RasterYSize
    arr_02 = band_02.ReadAsArray().astype(numpy.int16)

    print("Calculating watermasted Mean for dataset2...")
    mean2 = numpy.mean(arr_02, where=waterMask)
    print("Calculating watermasted Std for dataset2...")
    std2 = numpy.std(arr_02, where=waterMask)
    print("Calculating watermasted min/max for dataset2...")
    min2 = numpy.amin(arr_02, initial=10000, where=waterMask)
    max2 = numpy.amax(arr_02, initial=1, where=waterMask)
    print(f"Band 2 WM'd: min/max/mean/std: {min2}/{max2}/{mean2}/{std2}")

    band_02 = None

    print("Creating Diff array")
    #diffArr = numpy.zeros_like(arr_02).astype(numpy.int16) # pre-allocate
    #numpy.subtract(arr_02, arr_01, out=diffArr, where=waterMask)
    diffArr = numpy.subtract(arr_02, arr_01, where=waterMask)

    # Remove Outliers
    print("Removing Outliers")
    #arr_01_mean = numpy.mean(arr_01)
    print("std...")
    #arr_01_std = numpy.std(arr_01)
    clip_1 = int(mean1 + OUTLIER_STD_THRESHOLD * std1 )
    print(f"Clipping arr_1 to values < {clip_1}")
    arr_01 = numpy.clip(arr_01, 0, clip_1)
    #arr_02_mean = numpy.mean(arr_02)
    #arr_02_std = numpy.std(arr_02)
    clip_2 = int(mean2 + OUTLIER_STD_THRESHOLD * std2 )
    print(f"Clipping arr_2 to values < {clip_2}")
    arr_02 = numpy.clip(arr_02, 0, clip_2)

    # Normalizing the data to the same min/max
    if NORM_SCENE1_TO_SCENE2:
        print("Normalizing MIN/MAX between datasets")
        max2 = numpy.max(arr_02)
        min1 = numpy.min(arr_01)
        max1 = numpy.max(arr_01)
        arr_01 = ( arr_01 -  min1 ) / ( ( max1 - min1 ) / max2 )

    print("Getting updated metrics...")
    mean1 = numpy.mean(arr_01)
    arr_01 = None
    mean2 = numpy.mean(arr_02)
    pixel_range = numpy.std(arr_02) * BRIGHT_THRESHOLD
    arr_02 = None

    bright_cutoff = int( max2 - (mean1+mean2) - pixel_range)
    print(f"bright_cutoff: {max2} - ({mean1} + {mean2}) - {pixel_range} = {bright_cutoff}")

    isBright = diffArr > bright_cutoff

    return isBright, int(numpy.minimum(x_size, y_size)/50.), geomatrix, proj

def group_pixels ( labels, labels_pass, dataset=None):

    loc_labels_list = []

    label_count = len(labels_pass)
    label_counter = 0

    if dataset is not None:
        dataset = get_dataset_as_array ( dataset )

    for label in labels_pass: #[::100]: #[::1000]:
        label_counter +=1

        #if label_counter == 1:
        #    print(f"Skipping first label: {label}")
        #    continue

        label_filter = label == labels
        non_zero_index = numpy.nonzero( label_filter )
  
        if dataset is not None:
            pass_mean = numpy.mean(dataset[non_zero_index])
            MIN_MEAN = 600
            if pass_mean < MIN_MEAN:
                log.debug(f"Skipping {label} w/ mean {pass_mean} < MIN_MEAN")
                continue
         
        ship_size = len(non_zero_index[0])
        print(f"Grouping label pass {label} ({label_counter}/{label_count}). Size: {ship_size}")
        log.debug("Calculating min/max range...")
        loc_labels_list.append(
              [ numpy.amin( non_zero_index[0] ),
                numpy.amax( non_zero_index[0] ),
                numpy.amin( non_zero_index[1] ),
                numpy.amax( non_zero_index[1] )
              ] )
        log.debug(f"adding label {loc_labels_list[-1]}, {len(non_zero_index[0])}")

    print(f"Found {len(loc_labels_list)} Label areas with BRIGHTNESS > {BRIGHT_THRESHOLD}")
    return loc_labels_list

def plot_grid( data_array, scene, water_mask=None ):
    figure, axes = pyplot.subplots()

    if water_mask is None:
       if type(data_array) == str:
           print(f"Reading Geotiff... {data_array}")
           data_array = read_geotiff_as_numpy (data_array)
       image = axes.imshow( data_array )
    else:
       image = axes.imshow(numpy.logical_not(water_mask))

    axes.set_title(scene[0:48])
    cmap = pyplot.get_cmap('viridis',100)
    norm = matplotlib.colors.Normalize(vmin=0)
    figure.colorbar(pyplot.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes)

    return image, figure, axes

def get_rectangle ( c_center, r_center, rectLen ):
    xy_array = (int(c_center-rectLen/2.), int(r_center-rectLen/2.))
    params = {"linewidth": 2, "edgecolor": 'r', "facecolor": 'none' }
    return xy_array, rectLen, rectLen, params

def overlay_rectangles ( loc_labels_list, rectLen, axes, geomatrix, proj ):
    rects = []
    all_points = []

    for ship in numpy.arange(len(loc_labels_list)):
        r_center = numpy.mean(loc_labels_list[ship][0:2])
        c_center = numpy.mean(loc_labels_list[ship][2:])
        lat,lon,_,_ = pix_to_lat_lon_basic ( r_center, c_center, geomatrix, proj )
        log.debug(f"Ship {ship} center @ pixel ({r_center},{c_center}) = ({lat},{lon})")
        all_points.append( Point( lon, lat ) )
        xy, w, h, r_kwargs = get_rectangle ( c_center, r_center, rectLen  )
        rectangle = patches.Rectangle( xy, w, h, **r_kwargs )
        rects.append(rectangle)

    for ship in all_points:
        log.debug(f"Ship Point @ {ship.wkt}")

    p = PatchCollection(rects, match_original=True)
    axes.add_collection(p)
    pyplot.show()

    return all_points

def get_geotiff_watermask( bucket, key, cut_poly, scene):

    local_mask = download_file ( bucket, key )

    crop_local_mask = f"/tmp/{scene}_wm_crop.tiff"
    if os.path.exists(crop_local_mask):
        print(f"Reusing existing cropped watermask: {crop_local_mask}")
    else:

        centroid = cut_poly.centroid.coords[0]
        epsg_code = utm_from_lon_lat(centroid[0],centroid[1])

        print(f"Subsetting watermask to {crop_local_mask}")
        crop_image(local_mask, crop_local_mask, epsg_code)
        print("Finsished subsetting watermask")

    return read_geotiff_as_numpy( crop_local_mask, True)

def read_geotiff_as_numpy( crop_local_mask, binary_result=False):
    ds = gdal.Open( crop_local_mask )
    geotiff_data = numpy.array(ds.GetRasterBand(1).ReadAsArray())
    ds = None
    if binary_result:
        geotiff_data = geotiff_data == 0
    return geotiff_data

def ship_low_mem ( scene_1, scene_2, output_file_name=None ):

    output_file_name = output_file_name or f"/tmp/{scene_2}.ships.png"

    print(f"Loading {scene_2} raster")
    later_scene_data_path, cut_poly = generate_cropped_geocode_grd( scene_2 )
    print(f"Loading {scene_1} raster")
    early_scene_data_path, cut_poly = generate_cropped_geocode_grd(scene_1, cut_poly)

    print("Loading Watermask")
    water_mask = get_geotiff_watermask( 'coop-static-input', 'water_masks/ak_watermask.tiff',  cut_poly, scene_2)

    print("Running bight pixel detection")
    #is_bright, rect_len, geomatrix, proj = big_is_bright( early_scene_data_path, later_scene_data_path, water_mask)
    is_bright, rect_len, geomatrix, proj = basic_is_bright( early_scene_data_path, later_scene_data_path, water_mask)

    # Free Memory
    early_scene_data_path = None

    labels, num_labels = measure.label(is_bright, return_num=True)
    if num_labels == 0:
        log.warning("No bright pixels found?!")
        return None

    is_bright = None

    histogram, _ = numpy.histogram(labels,bins=num_labels,range=(0.5,num_labels+0.5))
    labels_pass = numpy.nonzero(histogram>=LABEL_THRESHOLD)[0]+1

    histogram = None

    #loc_labels_list = group_pixels ( labels, labels_pass, later_scene_data_path)
    loc_labels_list = group_pixels ( labels, labels_pass )

    labels = None
    labels_pass = None

    print("Generate output image")
    _, figure, axes = plot_grid( later_scene_data_path, scene_2 )

    print("Overlaying Rectangles")
    ships = overlay_rectangles ( loc_labels_list, rect_len, axes, geomatrix, proj)

    figure.savefig( output_file_name )

    # Match Ships in SAR to Ships in AIS 
    match_ships ( scene_2, cut_poly, ships)

def utm_from_lon_lat(lon: float, lat: float) -> int:
    hemisphere = 32600 if lat >= 0 else 32700
    zone = int(lon // 6 + 30) % 60 + 1
    return hemisphere + zone
