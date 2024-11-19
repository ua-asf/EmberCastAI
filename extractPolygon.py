import zipfile
import os
from xml.etree import ElementTree as ET
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# Path to the KMZ file
kmz_file = '/home/dmmaltos/Desktop/wildfire prediction/data/20240919_BachelorCx_IR(9).kmz'

# Extract the KML file from the KMZ archive
extracted_dir = 'extracted_kmz'
with zipfile.ZipFile(kmz_file, 'r') as kmz:
    kmz.extractall(extracted_dir)

# Locate the KML file
kml_file = None
for root, dirs, files in os.walk(extracted_dir):
    for file in files:
        if file.endswith('.kml'):
            kml_file = os.path.join(root, file)
            break

if not kml_file:
    raise FileNotFoundError("No KML file found in the KMZ archive")

# Parse the KML file to find the 'Heat Perimeter' layer
tree = ET.parse(kml_file)
root = tree.getroot()

# Extract namespace
namespace = {'kml': 'http://www.opengis.net/kml/2.2'}

# Find the coordinates for the 'Heat Perimeter' layer
coordinates = None
for placemark in root.findall('.//kml:Placemark', namespace):
    name = placemark.find('./kml:name', namespace)
    if name is not None and 'Heat Perimeter' in name.text:
        polygon = placemark.find('.//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespace)
        if polygon is not None:
            coordinates = polygon.text.strip()
            break

if not coordinates:
    raise ValueError("No 'Heat Perimeter' layer found in the KML file")

# Parse the coordinates into a list of (longitude, latitude) tuples
coords = []
raw_coordinates = coordinates.split()  # Split on whitespace to get individual coordinate triples

for triple in raw_coordinates:
    try:
        triples = triple.split(',')
        for i in range(0, len(triples), 3):
            lon, lat, _ = map(float, triples[i:i+3])  # Split into lon, lat, alt
            coords.append((lon, lat))
    except ValueError as e:
        print(f"Skipping invalid coordinate triple {triple}: {e}")

# Ensure the polygon is closed (first and last points must be identical)
if len(coords) > 0 and coords[0] != coords[-1]:
    coords.append(coords[0])

# Check if there are enough coordinates to form a polygon
if len(coords) < 4:
    raise ValueError("Insufficient coordinates to form a polygon")

# Create a Polygon from the coordinates
polygon = Polygon(coords)

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:4326")

# Plot the polygon
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)  # No fill color, red perimeter 
ax.set_axis_off()  # Remove axes

# Save the image with a transparent background
output_png = 'heat_perimeter_polygon.png'
fig.savefig(output_png, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"Polygon saved as {output_png}")
