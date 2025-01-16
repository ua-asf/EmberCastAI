# EmberCastAI - extractPolygon.py
# Extracts wildfire polygon data from kmz files
# Last Updated: 1/16/24

# Libraries
import zipfile
import os
import shutil
from xml.etree import ElementTree as ET
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# Define input and output directories
kmz_dir = 'kmz_data'
output_geotiff = 'dataset/geotiff'
output_png = 'dataset/png'
output_wkt = 'dataset/coordinates'

# Process each KMZ file in the directory
for kmz_file in os.listdir(kmz_dir):
    if kmz_file.endswith('.kmz'):
        kmz_path = os.path.join(kmz_dir, kmz_file)
        extracted_dir = 'extracted_kmz'

        # Extract the KMZ file
        with zipfile.ZipFile(kmz_path, 'r') as kmz:
            kmz.extractall(extracted_dir)

        # Locate the KML file
        kml_file = None
        for root, dirs, files in os.walk(extracted_dir):
            for file in files:
                if file.endswith('.kml'):
                    kml_file = os.path.join(root, file)
                    break

        if not kml_file:
            print(f"No KML file found in KMZ archive: {kmz_file}")
            continue

        # Parse the KML file to find the 'Heat Perimeter' layer
        tree = ET.parse(kml_file)
        root = tree.getroot()
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
                        raw_coordinates = coordinates.split()
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
            print(f"No valid polygons found in KMZ file: {kmz_file}")
            shutil.rmtree(extracted_dir, ignore_errors=True)
            continue

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")

        # Prepare file names
        base_name = os.path.splitext(kmz_file)[0]
        geotiff_file = os.path.join(output_geotiff, f"{base_name}.tif")
        png_file = os.path.join(output_png, f"{base_name}.png")
        wkt_file = os.path.join(output_wkt, f"{base_name}.wkt")

        # Save the raw coordinate strings as WKT
        with open(wkt_file, 'w') as wkt_out:
            for polygon in polygons:
                wkt_out.write(f"{polygon.wkt}\n")

        # Plot all polygons and save images
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(ax=ax, facecolor='red', edgecolor='red', linewidth=2)
        ax.set_axis_off()

        # Save as GeoTIFF
        fig.savefig(geotiff_file, format='tiff', transparent=True, bbox_inches='tight', pad_inches=0)

        # Save as PNG
        fig.savefig(png_file, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Clean up extracted directory
        shutil.rmtree(extracted_dir, ignore_errors=True)

        print(f"Processed {kmz_file}:")
        print(f"  GeoTIFF saved to {geotiff_file}")
        print(f"  PNG saved to {png_file}")
        print(f"  WKT saved to {wkt_file}")

print("Processing complete for all KMZ files.")
