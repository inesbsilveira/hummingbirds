#### MANGROVE ELIGIBILITY
#######GMW FILES

import sys
import pandas as pd
import geopandas as gpd
import processing 
import numpy
from osgeo import ogr, gdal, osr
from console.console import _console
from pathlib import Path
from qgis.core import QgsDistanceArea, QgsProject, QgsCoordinateReferenceSystem, QgsGeometry
import os
from qgis.core import (QgsRasterLayer,QgsVectorLayer,QgsProcessingFeatureSourceDefinition,QgsProcessing,QgsProject,QgsProcessingAlgorithm,QgsRaster,QgsCoordinateReferenceSystem)
import processing
import glob
from shapely.geometry import mapping


#project
myproject = QgsProject.instance()
project_path = myproject.homePath()

# project coordinate reference system
print(myproject.crs())  # QgsCoordinateReferenceSystem
print(myproject.crs().authid())  # string
#change the project CRS
if False: #choose the CRS
    myproject.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
    myproject.setCrs(QgsCoordinateReferenceSystem("EPSG:3763"))


# data folders
input_folder = 'input'
output_folder ='output'
#path to working directory (my_folder)
script_path =Path(_console.console.tabEditorWidget.currentWidget().path)
my_folder = script_path.parent
## load functions files
exec(Path(my_folder/'my_functions.py').read_text())

#vector layer names
ln_shp = 'sierra_leone_mangrove_forests'
ln_gmw_2010 = 'gmw_v3_2010_vec'
ln_gmw_2020 = 'gmw_v3_2020_vec'

#path to layers
fn_shp = str(my_folder/input_folder/(ln_shp + '.shp'))
fn_gmw_2010 = str(my_folder/input_folder/(ln_gmw_2010 + '.shp'))
fn_gmw_2020 = str(my_folder/input_folder/(ln_gmw_2020 + '.shp'))


#1ST - FIX GEOMETRY 
# Define paths for the fixed shapefile
fn_shp_fixed = str(my_folder / output_folder / (ln_shp + '_fixed.shp'))
# Call the function to fix geometries
fix_shapefile_geometries(fn_shp, fn_shp_fixed)

#2ND STEP - CONVERT FROM MULTIPART TO SINGLEPART IF NEEDED
# # Define paths for the singlepart shapefiles
# fn_gmw_2010_singlepart = str(my_folder / output_folder / (ln_gmw_2010 + '_singlepart.shp'))
# fn_gmw_2020_singlepart = str(my_folder / output_folder / (ln_gmw_2020 + '_singlepart.shp'))
# # call the function to convert shp
# process_singlepart_conversion(fn_gmw_2010, fn_gmw_2010_singlepart, 2010)
# process_singlepart_conversion(fn_gmw_2020, fn_gmw_2020_singlepart, 2020)

#3RD STEP - CLIP GMW SHP WITH FIXED SHAPEFILE
# Define paths for the clipped shapefiles
fn_gmw_2010_clipped = str(my_folder / output_folder / (ln_gmw_2010 + '_clipped.shp'))
fn_gmw_2020_clipped = str(my_folder / output_folder / (ln_gmw_2020 + '_clipped.shp'))
# Call the function for clipping for both years
process_clipping(fn_gmw_2010, fn_shp_fixed, fn_gmw_2010_clipped, 2010)
process_clipping(fn_gmw_2020, fn_shp_fixed, fn_gmw_2020_clipped, 2020)

#4RD STEP - CALCULATE THE ELIGIBLE AREAS
#read gmw_shp as gdf
gdf_gmw_2010 = gpd.read_file(fn_gmw_2010_clipped)
gdf_gmw_2020 = gpd.read_file(fn_gmw_2020_clipped)
# call the function to calculate total area for both years
total_area_ha_2010 = calculate_total_area_ha(gdf_gmw_2010, 2010)
total_area_ha_2020 = calculate_total_area_ha(gdf_gmw_2020, 2020)

# Print final results
print(f"\nGMW Total Area:\n2010: {total_area_ha_2010} hectares\n2020: {total_area_ha_2020} hectares")







