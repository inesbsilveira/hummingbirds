#### MANGROVE ELIGIBILITY
#######ESA FILES

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
ln_shp = 'sierra_leone_mangrove_forests' #change file name here
# Get all raster files that start with 'ESA_WorldCover' in the input folder
input_folder_path = Path(my_folder/input_folder)
raster_files = list(input_folder_path.glob("ESA_WorldCover*.tif"))



# 1ST STEP - CLIP THE RASTER FILES USING THE FIXED GEOMETRY SHAPEFILE
# Create a list to hold the clipped rasters
clipped_rasters = []
clipped_rasters = clip_raster_with_mask(raster_files, fn_shp_fixed)
# Print the list of clipped rasters to verify
#print(f"Clipped rasters: {clipped_rasters}")


# 2ND STEP - MERGE THE CLIPPED RASTER FILES
merged_raster_path = str(my_folder / output_folder / f"{ln_shp}_ESA_merged.tif")
merge_raster_files(clipped_rasters, merged_raster_path)


# 3RD STEP - POLIGONIZE RASTER
polygonized_shapefile_path = str(my_folder / output_folder / f"{ln_shp}_ESA_polygonized.shp")
polygonize_raster(merged_raster_path, polygonized_shapefile_path)
# Rename de DN field (Digital Number) to a name of your choosing
rename_field(polygonized_shapefile_path, "DN", "class_nr")

# 4TH STEP - ADD A NEW FIELD - CLASS_NAME
# read file as gdf
gdf_esa_polygonized = gpd.read_file(polygonized_shapefile_path)
# Add a new field and map it to the corresponding class description using the class_nr field
class_mapping = {
    '10': 'Tree Cover',
    '20': 'Shrubland',
    '30': 'Grassland',
    '40': 'Cropland',
    '50': 'Built up',
    '60': 'Bare land',
    '70': 'Snow and ice'
    '80': 'Water bodies',
    '90': 'Herbaceous wetland',
    '95': 'Mangroves',
    '100': 'Moss and lichen'
}

# Add the new field "class_name" based on "class_nr" 
gdf_esa_polygonized['class_name'] = gdf_esa_polygonized['class_nr'].astype(str).map(class_mapping)
# Save the updated shapefile
polygonized_shapefile_classes_path = str(my_folder / output_folder / f"{ln_shp}_ESA_with_classes.shp")
gdf_esa_polygonized.to_file(polygonized_shapefile_classes_path)


# 5TH STEP - CALCULATE THE TOTAL AREA PER CLASS
# Load the shapefile
gdf_esa_classes = gpd.read_file(polygonized_shapefile_classes_path)
#calculate the area in hectares, per feature
area_ha = calculate_area_in_hectares(gdf_esa_classes, crs=None)
# Add the area in hectares as a new column to the GeoDataFrame
gdf_esa_classes['area_ha'] = area_ha
# Save the shapefile with the new area column
output_path_with_areas = str(my_folder / output_folder / f"{ln_shp}_ESA_with_areas.shp")
gdf_esa_classes.to_file(output_path_with_areas)


#calculate the area, in hectares, by class
area_by_class = gdf_esa_classes.groupby('class_name')['area_ha'].sum()
# Print the results in the desired format
for class_name, area_ha in area_by_class.items():
    print(f"{class_name}: {round(area_ha, 2)} hectares")



## 6TH STEP - EXTRACT ELIGIBLE AREAS BY EXPRESSION
classes_of_interest = ['Tree Cover', 'Shrubland', 'Grassland', 'Cropland', 'Bare land','Herbaceous wetland', 'Mangroves']
gdf_esa_areas = gpd.read_file(output_path_with_areas)
# Check a few rows before applying the expression
print(gdf_esa_areas['class_name'].unique())

#expressionto extract classes
expression = "lower(class_name) IN ({})".format(", ".join([f"'{cls.lower()}'" for cls in classes_of_interest]))

#output shapefile path
eligible_areas_path = str(my_folder / output_folder / f"{ln_shp}_ESA_eligible_areas.shp")
# extract features from classes of interest
extracted_features = extract_features_by_expression(output_path_with_areas, eligible_areas_path)
