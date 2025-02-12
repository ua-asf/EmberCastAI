## EmberCastAI

## Overview
`extractPolygon.py` is a script that processes KMZ files to extract polygon data and outputs them in multiple formats, organized into respective directories for ease of use.  
`webScraper.go` scrapes KMZ files from [ftp.wildfire.gov](https://ftp.wildfire.gov/public/incident_specific_data/) into the `/kmz_data` directory for `extractPolygon.py` to use.

## extractPolygon.py
- Iterates through each KMZ file in the `/kmz_data` directory.
- Extracts polygon images and data in the following formats:
  - **PNG**: Polygon images saved in `/dataset/png`.
  - **GeoTIFF**: Polygon geospatial data saved in `/dataset/geotiff`.
  - **WKT**: Polygon coordinate data in WKT format saved in `/dataset/coordinates`.

## webScraper.go
- Scrapes KMZ files from [ftp.wildfire.gov](https://ftp.wildfire.gov/public/incident_specific_data/) into `/kmz_data` for `extractPolygon.py` to use.