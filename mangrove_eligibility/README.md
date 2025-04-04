### Eligibility for Mangrove Projects
This code will run in QGIS 3.34 or above

### Requirements
- python 3.13 or above
- QGIS 3.34 or above
- libraries: sys, pandas, geopandas, processing, numpy, pathlib, os, glob, shapely
- ESA_WorldCover tif file
- Global mangrove watch shapefile for 2010 and 2020
- 1 shapefile of the project area

> The .py files must be in a folder with the name of the project/country. Inside that same folder, you need to have 2 folders - input and output. The input folder will have the project shapefile, the ESA worldcover rasters, and the GMW shapefiles, and the output folder will store any files created by the code

### Working directory
COUNTRY/PROJECT_FOLDER
- input folder
  - project shapefile
  - ESA_WorldCover_(...).tif
  - gmw_v3_2010_vec.shp
  - gmw_v3_2020_vec.shp
- output folder
- ESA_eligibility.py
- GMW_eligibility.py
- my_functions.py

The ESA_eligibility.py takes the project shapefile and the ESA raster files and returns a final shapefile with the eligible areas according to ESA\
The GMW_eligibility.py takes the project shapefile and both GMW files and returns the two shapefiles with eligible areas - one for each year (2010 and 2020)\
The my_functions.py is where all the needed functions to run the 2 codes are stored. **Without this file the code will not run**
