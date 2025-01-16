# EmberCastAI

## Overview
`extractPolygon.py` is a script that processes KMZ files to extract polygon data and outputs them in multiple formats, organized into respective directories for ease of use.

## Features
- Iterates through each KMZ file in the `/kmz_data` directory.
- Extracts polygon images and data in the following formats:
  - **PNG**: Polygon images saved in `/dataset/png`.
  - **GeoTIFF**: Polygon geospatial data saved in `/dataset/geotiff`.
  - **WKT**: Polygon coordinate data WKT format saved in `/dataset/coordinates`.
