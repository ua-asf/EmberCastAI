import os
import logging
import pandas
import shapely
import shapely.wkt
import fiona
from zipfile import ZipFile
from shapely.ops import nearest_points
from datetime import timedelta
from fiona.crs import from_epsg
from shapely.geometry import Point, LineString, mapping
from dateutil import parser
from osgeo import ogr
from cde_data import download_local
from pandas.core.frame import DataFrame
from pyproj import Geod


log = logging.getLogger(__name__)
LOG_LEVEL = logging.INFO

USE_PANDAS = True
AOI_WINDOW_MINUTES = 60
AIS_CACHE = '/tmp'
MAXIMUM_SHIP_DISTANCE_METERS = 1000

AOI_WKT = 'POLYGON(( -167.22426 71.90211, -140.57578 72.30599, -140.60652 56.98192, -170.16160 52.21863, -170.17172 71.86674, -167.22426 71.90211))'
DATA_WKT = 'POLYGON ((-168.317749 63.807453, -167.467316 65.280991, -172.854034 65.73967, -173.423233 64.252975, -168.317749 63.807453))'
DATA_WKT = 'POLYGON ((-164.424057 63.405624, -163.599335 64.880775, -168.905151 65.335281, -169.457214 63.847801, -164.424057 63.405624))'

SHIPS_WKT = 'MULTIPOINT(-170.61160824964693 64.58327804870768)'
SHIPS_WKT = 'MULTIPOINT((-167.50946359574493 65.04149495048573),(-167.12624518553068 65.04716620139345),(-165.62992014583804 64.51684731085108),(-165.4392839433779 64.49537433225952),(-165.43701060632415 64.48335649879124),(-165.43545028029294 64.48336112299172),(-166.9601104699508 64.46071647045139),(-168.72696935008776 63.94857152460653))'

def download_ais_csv (zip_path, csv_name):
    # Download the file to local!
    
    local_zip_path = download_local(zip_path, AIS_CACHE)
    
    with ZipFile(local_zip_path, 'r') as zipObj:
        log.info(f"Extracing {csv_name} from {local_zip_path} to {AIS_CACHE}")
        zipObj.extract(csv_name, AIS_CACHE)

    csv_path = f"{AIS_CACHE}/{csv_name}"
    if os.path.exists(csv_path):
        log.info(f"Extraction Successful: {csv_path}, removing Zip")
        os.remove(local_zip_path)
        return csv_path
    
    log.error("Could not download and extract AIS CSV")
    return None
      
def get_ship_data(date):
    # 2019-08-22T17:42:07.000Z
    (year, month, day) = (date[:4], date[5:7], date[8:10])

    local_path = f'{AIS_CACHE}/AIS_{year}_{month}_{day}.csv'
    shape_path = f'{AIS_CACHE}/{year}{month}{day}/{year}{month}{day}.shp'
    if os.path.exists(shape_path):
        csv_path = shape_path
        log.info(f"Using local SHP: {csv_path}")
    elif os.path.exists(local_path):
        csv_path = local_path
        log.info(f"Using local csv: {csv_path}")
    else:
        gdal_pref = '/vsizip/vsicurl'
        host_path = 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler'
        file_name = f'AIS_{year}_{month}_{day}'
        zip_path = f'{host_path}/{year}/{file_name}.zip'

        if USE_PANDAS:
            csv_path = download_ais_csv (zip_path, f'{file_name}.csv')
        else:
            csv_path = f'{gdal_pref}/{zip_path}/{file_name}.csv'

    return csv_path

def open_csv_ogr(csv_path):
    log.info(f"Opening {csv_path}")
    data = ogr.Open(csv_path)
    return data

def open_pandas( csv_path, poly = None, time_range = None):
    log.info(f"Opening {csv_path} using Pandas")
    ais = pandas.read_csv(csv_path, parse_dates=['BaseDateTime'])

    if poly:
       log.info("Applying geographic filter to AIS data")
       (min_lon, min_lat, max_lon, max_lat) = poly.bounds
       old_row_count = len(ais)
       ais = ais[ ( min_lat <= ais.LAT ) & (ais.LAT <= max_lat ) & (min_lon <= ais.LON) & (ais.LON <= max_lon) ]
       new_row_count = len(ais)
       log.info(f"Reduced AIS dataset from {old_row_count} to {new_row_count}")
    if time_range is not None:
       log.info(f"Applying temporal filter to AIS data {time_range[0]} - {time_range[1]}")
       old_row_count = len(ais)
       ais = ais[ ( time_range[0] <= ais.BaseDateTime ) & ( ais.BaseDateTime <= time_range[1]) ]
       new_row_count = len(ais)
       log.info(f"Reduced AIS dataset from {old_row_count} to {new_row_count}")

    ais.VesselName = ais.VesselName.fillna('')

    return ais

def filter_data_for_aoi(data, aoi):
    f_cnt = 0

    # look up datatype
    if type(data) == DataFrame:
        log.info("Data is PANDAS")
        data_type = 'pandas'
        data = [ data.values ]
    elif data[0].GetLayerDefn().GetFieldDefn(7).name == 'DateTime':
        log.info("Data is a SHAPEFILE")
        data_type = 'shape'
    elif data[0].GetLayerDefn().GetFieldDefn(7).name == 'VesselName':
        log.info("Data is a CSV")
        data_type = 'csv'
    elif data[0].GetLayerDefn().GetFieldDefn(7).name == 'field_8':
        log.info("Data is probably a CSV but missing a header")
        data_type = 'csv'
    else:
        log.warning("I don't know what kinda of data this input is!")
        exit(0)

    time_points = {}

    for layer in data:
        total_cnt = len(layer)
        for f in layer:
            f_cnt += 1

            if f_cnt % 10000 == 0:
                log.info(f" Processed {f_cnt}/{total_cnt} {(f_cnt/total_cnt)*100:.2f} of dataset")

            # parse the line
            if data_type == 'pandas':
                (time, lat, lon, ship) = (f[1], f[2], f[3], f[7] or "UNKNOWN")
                ship = 'UNKNOWN' if ship == 'nan' else ship
            if data_type == 'csv':
                (time, lat, lon, ship) = (parser.parse(f[1]), float(f[2]), float(f[3]), f[7] or "UNKNOWN")
            if data_type == 'shape':
                lat = f.GetGeometryRef().GetY()
                lon = f.GetGeometryRef().GetX()
                (time, ship) = (parser.parse(f[7]), f[8] or "UNKNOWN" )


            if not aoi["date_start"] <= time <= aoi["date_end"]:
                continue

            f_point = Point( lon, lat )
            if not aoi["poly"].contains( f_point ):
                continue

            # format results
            if not ship in time_points:
                time_points[ship] = []

            # We found a point in our GEO/TEMPORAL AOI
            log.debug(f" - Found {ship} @ {time}: {f_point.wkt}")
            time_points[ship].append([time, f_point])

    return time_points

def get_nearest_points(ship_track, center_date):

    # We only have one point in our time range
    if len(ship_track) == 1:
        return ship_track[0], ship_track[0]

    # sort track by time from center data
    ship_track.sort(key=lambda a: abs(center_date - a[0]))    
    log.debug(f"Closest time to {center_date} is {ship_track[0]}. 2nd Closest is {ship_track[1]}")
    return ship_track[0], ship_track[1]

def get_min_max_center_time( scene ):
    date_aoi = "-".join([scene[17:21], scene[21:23], scene[23:25] ] )
    scene_start = parser.parse(scene[17:32])
    scene_end = parser.parse(scene[33:48])
    min_date = scene_start - timedelta(minutes=AOI_WINDOW_MINUTES)
    max_date = scene_end + timedelta(minutes=AOI_WINDOW_MINUTES)
    center_date = min_date + (max_date-min_date)/2

    log.info(f"{scene}/{date_aoi}, {min_date}-{max_date}")
    return min_date, max_date, center_date, date_aoi

def get_ship_tracks_by_scene ( scene, aoi_poly, scene_poly ):
    
    min_date, max_date, center_date, date_aoi = get_min_max_center_time( scene )
    aoi = { "poly": scene_poly, "date_start": min_date, "date_end": max_date }
    all_ship_data = get_ship_data( date_aoi )
    ship_data = open_pandas(all_ship_data, aoi_poly, (min_date, max_date)) 
    aoi_ships = filter_data_for_aoi(ship_data, aoi)
    log.info(f"Found {len(aoi_ships)} ships in pandas dataset")
    return aoi_ships, center_date

def time_delta_sec(first, second):
    delta = (first - second).seconds
    # if delta > 43200, we wrapped into the next/previous day
    delta = delta - 86400 if delta > 43200 else delta
    return delta

def get_third_point( first, second, center_date):
    geod = Geod(ellps="WGS84")
    past_time_delta = time_delta_sec(second[0], first[0])

    if past_time_delta == 0:
        log.info("Ship did not move between time stamps")
        # Stop here, otherwise we hit divided-by-0
        return first[1], 0

    future_time_delta = time_delta_sec(first[0], center_date) 

    log.debug(f"{past_time_delta}s between {second[0]} and {first[0]}")
    log.debug(f"{future_time_delta}s beween {first[0]} and {center_date}")

    # Nearest and further point
    ( near, far)  = ( second[1] , first[1] )
    
    # Calculate direction and distance between far/near points 
    fwd_azimuth,_,distance = geod.inv(near.x, near.y, far.x, far.y)
    # Calculate distance past near point based on elapsed time
    future_dist = (distance/past_time_delta) * future_time_delta
    log.debug(f"Distance outside our known AIS range: {future_dist}")
    new_lon,new_lat,_ = geod.fwd(near.x,near.y, fwd_azimuth,future_dist) 
    third_point = Point( new_lon, new_lat )
    log.debug(f"Projected point is {third_point}")
    log.debug(f" {far} -> {near} -> {third_point} ")

    return third_point, future_time_delta > 0
   
def get_ship_track_lines ( aoi_ships, center_date):
    
    ship_tracks = {}

    if not len (aoi_ships):
        log.warning("No AOI Ships to create tracks!")
        return None

    for ship in aoi_ships:
        first, second = get_nearest_points(aoi_ships[ship], center_date)
        early_point = min( [ first[0], second[0]] )
        late_point = max( [ first[0], second[0]] ) 
        track_line = [ first[1], second[1] ]
        if not early_point <= center_date <= late_point:
            log.warning(f" {center_date} not between {early_point} and {late_point}")
            log.debug(f"{[x[1].wkt for x in aoi_ships[ship]]}")
            third_point, fwd_time = get_third_point( first, second, center_date )
            if fwd_time: 
                track_line.insert(0, third_point)
            else:
                track_line.append(third_point)
            
        ship_track_line = LineString( track_line )
        log.info(f"Ship track for {ship} is {ship_track_line}")
        ship_tracks[ship] = ship_track_line

    return ship_tracks

def map_ships_to_tracks ( ship_tracks, ships_points, geod):

    closest_pair = False

    for ship in ship_tracks:
        for point in ships_points:
            nearest_point = nearest_points(ship_tracks[ship], point)[0]
            distance = ship_tracks[ship].distance(nearest_point)
            distance_meters = geod.geometry_length( LineString( [point, nearest_point] ) )
            log.debug(f"Distance between ship ({nearest_point}) & {point} is {distance}/{distance_meters}")

            if distance_meters > MAXIMUM_SHIP_DISTANCE_METERS:
                log.debug("Distance between ship & track is too great to be considered a match")
                continue

            if closest_pair == False or closest_pair["distance"] > distance_meters:
                closest_pair = { "distance": distance_meters, "ship": ship, 
                                 "point": point, "track": ship_tracks[ship] }

    return closest_pair

def create_geopkg_file ( scene, pairs, ships_points, ship_tracks, outname = None):
   
    outname = outname or f"{AIS_CACHE}/{scene}.gpkg"

    schema =  {
               'properties': {
                  'description': 'str',
                  'ship_name': 'str'
               }
              }
    
    with fiona.open(outname, 'w', 'GPKG', schema={**schema,'geometry':'Point'}, layer='confirmed', crs = from_epsg(4326)) as out:
        for pair in pairs:
            out.write({ 'geometry': mapping( pair['point'] ),
                        'properties': {
                            'ship_name': pair['ship'],
                            'description': "Confirmed Ship" }})

    with fiona.open(outname, 'w', 'GPKG', schema={**schema,'geometry':'Point'}, layer='ghosts', crs = from_epsg(4326)) as out:
        for point in ships_points:
            out.write({ 'geometry': mapping( point ),
                        'properties': {
                            'ship_name': "NOT FOUND",
                            'description': "Ghost Ship"}})

    with fiona.open(outname, 'w', 'GPKG', schema={**schema,'geometry':'LineString'}, layer='missing', crs = from_epsg(4326)) as out:
        for ship in ship_tracks:
            out.write({ 'geometry': mapping( ship_tracks[ship] ), # ship_tracks[ship].centroid ] )),
                        'properties': {
                            'ship_name': ship,
                            'description': "Missing Ship" }})

    return outname

def filter_pairs (ship_tracks, ships_points):

    pairs = []

    geod = Geod(ellps="WGS84")

    while True:
         pair = map_ships_to_tracks ( ship_tracks, ships_points, geod)

         if not pair:
             break
         else:
             log.debug(f"Match {pair['distance']} @ {pair['ship']}: {pair['point']} & {pair['track']}")
             pairs.append(pair)
    
             # remove point from ship track and ship points
             log.debug(f"Removing {pair['ship']} from ship_tracks")
             del ship_tracks[pair['ship']]
             log.debug(f"Removing {pair['point']} from ships_points")
             ships_points.remove(pair['point'])
             log.debug(f"New Sets: {ship_tracks} & {ships_points}")

    return pairs, ship_tracks, ships_points

def log_results ( scene, pairs, ships_points, ship_tracks ):

    if len( ship_tracks ):
        log.info("Unable to find ships (MISSING):")
        for ship in ship_tracks:
             log.info(f" - {ship_tracks[ship]} \t {ship}")
    if len( ships_points ):
        log.info("Unable to match points in SAR to known ships (GHOSTS): ")
        for ship in ships_points:
             log.info(f" - {ship}")
    if len (pairs):
        log.info(f"Known Ships in {scene} (CONFIRMED):")
        for pair in pairs:
            log.info(f" - {pair['point']} \t {pair['ship']}")

def match_ships ( scene, scene_poly, ships_points, aoi_poly=None):

    if not aoi_poly:
        log.info("Using All of alaska for AIS filtering")
        aoi_poly = shapely.wkt.loads( AOI_WKT )

    # Get ship tracks based from AIS data
    aoi_ships, center_date = get_ship_tracks_by_scene ( scene, aoi_poly, scene_poly )

    # Generate LineString for ships in our scenes
    ship_tracks = get_ship_track_lines ( aoi_ships, center_date)

    if not ship_tracks:
       log.warning(f"Found no ships in {scene}")

    # Match ships in SAR image to ship tracks in AIS data
    pairs, ship_tracks, ships_points = filter_pairs (ship_tracks, ships_points)

    # create a GeoPackage file with known ships, missing ships, and ghost ships
    gpkg =  create_geopkg_file ( scene, pairs, ships_points, ship_tracks )
    log.info(f"Wrote GeoPackage file to {gpkg}")

    # Log the results
    log_results ( scene, pairs, ships_points, ship_tracks )

def runner( scene ): 

    # Pull in our AOI shape
    scene_poly = shapely.wkt.loads( DATA_WKT )
    ships_points_mp = shapely.wkt.loads(SHIPS_WKT)
    ships_points = [Point(pt) for pt in ships_points_mp.geoms]
    match_ships ( scene, scene_poly, ships_points)

if __name__ == "__main__":

    # Set up logging.
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=LOG_LEVEL)

    SCENE = 'S1A_IW_GRDH_1SDV_20190908T175044_20190908T175109_028933_0347D0_D4C8'
    SCENE = 'S1A_IW_GRDH_1SDV_20190910T173424_20190910T173449_028962_0348D8_A19E'

    runner(SCENE)
