import datetime as dt

import geopandas as gpd
from owslib.ogcapi.features import Features
import requests

OGC_URL = "https://openveda.cloud/api/features"

w = Features(url=OGC_URL)
print(w.feature_collections())

collection_id = "public.eis_fire_snapshot_perimeter_nrt"
perm = w.collection(collection_id)

print(perm)

perm_q = w.collection_queryables(collection_id)
print(perm_q["properties"])
