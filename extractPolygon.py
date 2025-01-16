import zipfile
import os
import shutil  # For removing directories
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

# Collect all polygons from the 'Heat Perimeter' layer
polygons = []
for placemark in root.findall('.//kml:Placemark', namespace):
    name = placemark.find('./kml:name', namespace)
    if name is not None and 'Heat Perimeter' in name.text:
        polygon_elements = placemark.findall('.//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespace)
        for polygon in polygon_elements:
            coordinates = polygon.text.strip()
            if coordinates:
                coords = []
                raw_coordinates = coordinates.split()  # Split on whitespace to get individual coordinate strings
                for triple in raw_coordinates:
                    try:
                        # Split the triple into components by commas
                        components = triple.split(',')
                        # Process each group of three values (lon, lat, alt)
                        for i in range(0, len(components) - 2, 3):  # Ensure we don't exceed bounds
                            lon, lat, alt = map(float, components[i:i+3])  # Unpack the three values
                            coords.append((lon, lat))
                    except ValueError as e:
                        print(f"Skipping invalid coordinate triple: {triple} ({e})")


                # Ensure the polygon is closed
                if len(coords) > 0 and coords[0] != coords[-1]:
                    coords.append(coords[0])

                # Check if there are enough coordinates to form a polygon
                if len(coords) >= 4:
                    polygons.append(Polygon(coords))

if not polygons:
    raise ValueError("No valid polygons found in the 'Heat Perimeter' layer")

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")

# Plot all polygons
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, facecolor='red', edgecolor='red', linewidth=2)  # No fill color, red perimeter
ax.set_axis_off()  # Remove axes

# Save the image with a transparent background
output_geotiff = 'dataset/geotiff/heat_perimeter_polygons.tif'
fig.savefig(output_geotiff, format='tiff', transparent=True, bbox_inches='tight', pad_inches=0)
plt.close()

# Remove the extracted directory
try:
    shutil.rmtree(extracted_dir)
    print(f"Deleted folder: {extracted_dir}")
except Exception as e:
    print(f"Error deleting folder {extracted_dir}: {e}")

print(f"Polygons saved as {output_geotiff}")
