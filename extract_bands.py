from utils import get_fires
from osgeo import gdal


fires = get_fires()

for fire in fires.items():
    print(fire)
