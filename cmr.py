import logging
import requests
from shapely.geometry import Polygon

log = logging.getLogger(__name__)
LOG_LEVEL = logging.INFO


# CMR search datasets
CMR_BASEURL = 'https://cmr.earthdata.nasa.gov/search/granules.umm_json'
CONCEPT_IDS = ['C1214470533-ASF', 'C1327985645-ASF', 'C1214471521-ASF', 'C1327985660-ASF', 'C1214470682-ASF', 'C1214472994-ASF', 'C1327985740-ASF', 'C1327985571-ASF', 'C1214471197-ASF']

def get_scene_metadata (scene):

    scene_metadata = get_granule_from_cmr ( scene )
    return get_relevant_metadata(scene_metadata)

def execute_cmr_search( url):

    granules = []
    search_after = None

    while True:
        headers = {}
        if search_after:
            headers={'CMR-Search-After': search_after}
        r = requests.get(url, headers=headers)

        search_after = r.headers.get('CMR-Search-After')

        cmr_json = r.json()
        if not cmr_json.get('items',[]):
            break

        granules += cmr_json['items']

        if not search_after:
            break

    return granules

def get_granule_from_cmr ( scene ):

    # quiet noisy requets logger.
    logging.getLogger('requests').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)

    granule_id = f"producer_granule_id[]={scene}"
    concept_ids = '&'.join([ f'concept_id[]={cid}' for cid in CONCEPT_IDS])
    cmr_params = [concept_ids, granule_id, 'page_size=200']
    cmr_search_url = CMR_BASEURL + '?' + '&'.join(cmr_params)
    return execute_cmr_search(cmr_search_url)[0]["umm"]


def get_granules_by_aoi ( aoi_poly, date_aoi ):

    start = f'{date_aoi}T00:00:00Z'
    end = f'{date_aoi}T23:59:59Z'
    external_ring = [f"{x},{y}" for x,y in aoi_poly.exterior.coords]
    if not aoi_poly.exterior.is_ccw:
        external_ring.reverse()

    cmr_poly = "polygon[]=" + ",".join(external_ring)
    concept_ids = '&'.join([ f'concept_id[]={cid}' for cid in CONCEPT_IDS])
    date_filter = f'production_date[]={start},{end}'
    cmr_params = [concept_ids, cmr_poly, date_filter, 'page_size=200']
    cmr_search_url = CMR_BASEURL + '?' + '&'.join(cmr_params)

    return execute_cmr_search(cmr_search_url)

def get_relevant_metadata(scene_metadata):

    scene_name = scene_metadata["DataGranule"]["Identifiers"][0]["Identifier"]
    start_date = scene_metadata["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
    stop_date = scene_metadata["TemporalExtent"]["RangeDateTime"]["EndingDateTime"]
    geometry = scene_metadata["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
    polygon = Polygon( [ (p['Longitude'], p['Latitude']) for p in geometry ] )

    log.info(f"Data polgon is {polygon.wkt}")

    urls = scene_metadata["RelatedUrls"]
    browse = next((item["URL"] for item in urls if item["Type"] == "GET RELATED VISUALIZATION"), None)

    return scene_name, start_date, stop_date, polygon, browse
