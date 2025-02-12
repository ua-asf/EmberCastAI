# EmberCastAI - extractPolygon.py
# Extracts wildfire polygon data from kmz files
# Last Updated: 2/12/25

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
output_dir = 'dataset'

# Function to ensure the directory structure is mirrored
def ensure_output_dir_structure(base_input_dir, base_output_dir, relative_path):
    output_path = os.path.join(base_output_dir, relative_path)
    os.makedirs(output_path, exist_ok=True)
    return output_path

# Function to check if KMZ file is valid before extracting
def is_valid_kmz(file_path):
    try:
        with zipfile.ZipFile(file_path, 'r') as kmz:
            # Test if all files inside the KMZ can be read
            bad_file = kmz.testzip()
            if bad_file is not None:
                print(f"Corrupt file detected in KMZ: {file_path}. Skipping.")
                return False
        return True
    except zipfile.BadZipFile:
        print(f"Invalid or corrupt KMZ file: {file_path}. Skipping.")
        return False
    except Exception as e:
        print(f"Unexpected error checking KMZ file: {file_path}. Skipping. ({e})")
        return False

# Process each KMZ file in the directory structure
for current_dir, dirs, files in os.walk(kmz_dir):
    for kmz_file in files:
        if kmz_file.endswith('.kmz'):
            relative_dir = os.path.relpath(current_dir, kmz_dir)  # Get the relative directory path
            kmz_path = os.path.join(current_dir, kmz_file)

            # Check if the KMZ file is valid before proceeding
            if not is_valid_kmz(kmz_path):
                continue

            extracted_dir = 'extracted_kmz'

            # Extract the KMZ file
            try:
                with zipfile.ZipFile(kmz_path, 'r') as kmz:
                    kmz.extractall(extracted_dir)
            except Exception as e:
                print(f"Error extracting KMZ file: {kmz_path}. Skipping. ({e})")
                continue

            # Locate the KML file
            kml_file = None
            for root, _, filenames in os.walk(extracted_dir):
                for file in filenames:
                    if file.endswith('.kml'):
                        kml_file = os.path.join(root, file)
                        break

            if not kml_file:
                print(f"No KML file found in KMZ archive: {kmz_path}")
                shutil.rmtree(extracted_dir, ignore_errors=True)
                continue

            try:
                tree = ET.parse(kml_file)
                root = tree.getroot()
            except ET.ParseError as e:
                print(f"Error parsing KML file: {kml_file}. Skipping. ({e})")
                shutil.rmtree(extracted_dir, ignore_errors=True)
                continue

            namespace = {'kml': 'http://www.opengis.net/kml/2.2'}

            # Collect all polygons from the 'Heat Perimeter' layer
            polygons = []
            for placemark in root.findall('.//kml:Placemark', namespace):
                name = placemark.find('./kml:name', namespace)
                if name is not None and name.text is not None and 'heat perimeter' in name.text.lower():
                    polygon_elements = placemark.findall('.//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespace)
                    for polygon in polygon_elements:
                        if polygon.text:
                            coords = []
                            raw_coordinates = polygon.text.strip().split()
                            for triple in raw_coordinates:
                                try:
                                    components = triple.split(',')
                                    lon, lat, alt = map(float, components[:3])
                                    coords.append((lon, lat))
                                except ValueError as e:
                                    print(f"Skipping invalid coordinate triple: {triple} ({e})")
                            if len(coords) > 0 and coords[0] != coords[-1]:
                                coords.append(coords[0])
                            if len(coords) >= 4:
                                polygons.append(Polygon(coords))

            # If no polygons were found, skip processing this KMZ file
            if not polygons:
                print(f"No valid 'Heat Perimeter' polygons found in KMZ file: {kmz_path}. Skipping.")
                shutil.rmtree(extracted_dir, ignore_errors=True)
                continue

            # Now that we have polygons, create directories
            output_geotiff_dir = ensure_output_dir_structure(kmz_dir, os.path.join(output_dir, 'geotiff'), relative_dir)
            output_png_dir = ensure_output_dir_structure(kmz_dir, os.path.join(output_dir, 'png'), relative_dir)
            output_wkt_dir = ensure_output_dir_structure(kmz_dir, os.path.join(output_dir, 'coordinates'), relative_dir)

            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")

            # Prepare file names
            base_name = os.path.splitext(kmz_file)[0]
            geotiff_file = os.path.join(output_geotiff_dir, f"{base_name}.tif")
            png_file = os.path.join(output_png_dir, f"{base_name}.png")
            wkt_file = os.path.join(output_wkt_dir, f"{base_name}.wkt")

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

            print(f"Processed {kmz_path}:")
            print(f"  GeoTIFF saved to {geotiff_file}")
            print(f"  PNG saved to {png_file}")
            print(f"  WKT saved to {wkt_file}")

print("Processing complete for all KMZ files.")
