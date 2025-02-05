
#function to fix geometry of a shapefile
def fix_geometries(input_filepath, output_filepath):
    fixed_shp = processing.run("native:fixgeometries", 
    {'INPUT': input_filepath,
    'METHOD':1,
    'OUTPUT':output_filepath
    })['OUTPUT']
    
    return fixed_shp

#function to convert from multipart to singlepart
def multi_to_singlepart(input_filepath, output_filepath): 
    singlepart_shp = processing.run("native:multiparttosingleparts", {
    'INPUT':input_filepath,
    'OUTPUT':output_filepath
    })['OUTPUT']
    
    return singlepart_shp

#function to clip a shapefile
def clip(input_shp, overlay_shp, output_filepath):
    clipped_shp = processing.run("native:clip", {
    'INPUT': input_shp,
    'OVERLAY':overlay_shp,
    'OUTPUT': output_filepath
    })['OUTPUT']
    
    return clipped_shp

def fix_shapefile_geometries(input_shp, output_shp):
    """
    Fixes the geometries of a shapefile and saves the fixed version.
    
    Parameters:
        input_shp (str): Path to the input shapefile.
        output_shp (str): Path to the output fixed shapefile.
    """
    if not os.path.exists(output_shp):
        print(f"Fixing {input_shp} to {output_shp}")
        try:
            fixed_shp = fix_geometries(input_shp, output_shp)
            print("Fixing completed successfully.")
        except Exception as e:
            print(f"Error during fixing process: {e}")
    else:
        print(f"Fixed shapefile already exists: {output_shp}")



def process_singlepart_conversion(input_shp, output_shp, year):
    """
    Processes conversion from multipart to singlepart for a given input shapefile.
    
    Parameters:
        input_shp (str): Path to the input shapefile.
        output_shp (str): Path to the output singlepart shapefile.
        year (str or int): Year of the dataset for logging purposes.
    """
    if not os.path.exists(output_shp):
        print(f"Converting {input_shp} to singlepart: {output_shp}")
        try:
            singlepart_shp = multi_to_singlepart(input_shp, output_shp)
            print(f"Conversion to singlepart ({year}) completed successfully.")
        except Exception as e:
            print(f"Error during singlepart conversion ({year}): {e}")
    else:
        print(f"Singlepart shapefile ({year}) already exists: {output_shp}")



def process_clipping(input_shp, overlay_shp, output_shp, year):
    """
    Processes clipping for a given input shapefile.
    
    Parameters:
        input_shp (str): Path to the input shapefile.
        overlay_shp (str): Path to the Country shapefile.
        output_shp (str): Path to the output clipped shapefile.
        year (str or int): Year of the dataset for logging purposes.
    """
    if not os.path.exists(output_shp):
        print(f"Clipping {input_shp} with {overlay_shp} to create {output_shp}")
        try:
            clipped_shp = clip(input_shp, overlay_shp, output_shp)
            print(f"Clipping ({year}) completed successfully.")
        except Exception as e:
            print(f"Error during clipping ({year}): {e}")
    else:
        print(f"Clipped shapefile ({year}) already exists: {output_shp}")


# Function to calculate the area using ellipsoid-based measurements
def calculate_total_area_ha(gdf, year):
    # Create a QgsDistanceArea object for area calculation
    d = QgsDistanceArea()
    
    # Convert GeoDataFrame CRS to QgsCoordinateReferenceSystem
    qgis_crs = QgsCoordinateReferenceSystem(gdf.crs.to_string())  # Convert GeoPandas CRS to QGIS CRS
    
    # Set the CRS of the layer and the ellipsoid from the project settings
    d.setSourceCrs(qgis_crs, QgsProject.instance().transformContext())
    d.setEllipsoid(QgsProject.instance().ellipsoid())
    
    # Calculate the area for each geometry and sum them
    total_area_m2 = 0
    for geom in gdf['geometry']:
        # Convert shapely geometry to QgsGeometry
        qgis_geom = QgsGeometry.fromWkt(geom.wkt)  # Convert shapely geometry to QgsGeometry
        
        # Calculate area of geometry
        total_area_m2 += d.measureArea(qgis_geom)
    
    # Convert square meters to hectares
    total_area_ha = total_area_m2 / 10000  # 1 hectare = 10,000 m²
    
    # Print the result for this year
    print(f"Total area for {year}: {round(total_area_ha, 2)} hectares")
    
    return total_area_ha


def merge_raster_files(list_raster_files, output_path):
    # gdalbuildvrt command should be constructed as a string
    gdal_command = f"gdalbuildvrt {output_path} " + " ".join(list_raster_files)
    # Run the gdalbuildvrt command
    os.system(gdal_command)



def clip_raster_with_mask(raster_files, mask_shp):
    # Loop through each raster file and process it
    for raster in raster_files:
        # Get the raster name (without the extension)
        raster_name = raster.stem  # Corrected variable name
        
        # Load the raster layer
        raster_layer = QgsRasterLayer(str(raster), raster_name)  # Use 'raster' instead of 'fn_raster'
        if not raster_layer.isValid():
            raise ValueError(f"Failed to load raster file: {raster}")
        
        # Load the mask vector layer
        mask_layer = QgsVectorLayer(mask_shp, "Mask Shapefile", "ogr")
        if not mask_layer.isValid():
            raise ValueError(f"Failed to load shapefile: {mask_shp}")
        
        # Define output path for the clipped raster
        output_raster_path = str(my_folder / output_folder / f"{raster_name}_clipped.tif")
        
        # Use QgsProcessing to clip raster with vector mask
        processing_algorithm = 'gdal:cliprasterbymasklayer'
        params = {
            'INPUT': raster_layer,
            'MASK': mask_layer,
            'OUTPUT': output_raster_path
        }
        
        # Run the processing algorithm
        processing.run(processing_algorithm, params)
        print(f"Clipping complete for {raster}. Output saved to {output_raster_path}")
        
        # Append the output raster path to the list
        clipped_rasters.append(output_raster_path)
    
    return clipped_rasters

def polygonize_raster(input_raster_path, output_raster_path):
# Use QgsProcessing to polygonize the VRT raster
    # Define the processing algorithm for polygonization
    processing_algorithm = 'gdal:polygonize'

    # Set parameters for the processing algorithm
    params = {
        'INPUT': input_raster_path,
        'OUTPUT': output_shapefile_path
    }

    try:
        # Run the polygonization process
        processing.run(processing_algorithm, params, QgsProcessingFeedback())
        print(f"Raster successfully polygonized. Output saved to: {output_shapefile_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to polygonize raster {input_raster_path}: {e}")


def rename_field(layer_path, old_field_name, new_field_name):
    """
    Renames a field in a shapefile layer.
    
    :param layer_path: Path to the shapefile
    :param old_field_name: Name of the field to rename
    :param new_field_name: New name for the field
    """
    # Load the layer
    layer = QgsVectorLayer(layer_path, "layer", "ogr")
    if not layer.isValid():
        raise ValueError(f"Failed to load shapefile: {layer_path}")
    
    # Start editing the layer
    layer.startEditing()
    
    # Check if the field exists and rename it
    fields = layer.fields()
    field_index = fields.indexOf(old_field_name)
    if field_index != -1:
        # Rename the field
        layer.renameAttribute(field_index, new_field_name)
        print(f"Field '{old_field_name}' renamed to '{new_field_name}'.")
    else:
        print(f"Field '{old_field_name}' not found, no renaming performed.")
    
    # Commit changes
    if not layer.commitChanges():
        raise RuntimeError(f"Failed to commit changes to the layer: {layer_path}")




# Function to calculate areas per class considering Earth's curvature
def calculate_area_in_hectares(gdf, crs=None):
    """
    Calculate the area of each geometry in a GeoDataFrame in hectares.
    
    Parameters:
    - gdf: GeoDataFrame containing the geometries.
    - crs: Optional; Coordinate reference system to be used for area calculation. 
           If None, uses the CRS of the GeoDataFrame.
    
    Returns:
    - area_ha: List of area values in hectares.
    """
    # Create a QgsDistanceArea object for area calculation
    d = QgsDistanceArea()

    # Convert GeoDataFrame CRS to QgsCoordinateReferenceSystem
    if crs is None:
        qgis_crs = QgsCoordinateReferenceSystem(gdf.crs.to_string())  # Convert GeoPandas CRS to QGIS CRS
    else:
        qgis_crs = QgsCoordinateReferenceSystem(crs)

    # Set the CRS of the layer and the ellipsoid from the project settings
    d.setSourceCrs(qgis_crs, QgsProject.instance().transformContext())
    d.setEllipsoid(QgsProject.instance().ellipsoid())

    # Initialize a list to store area values in hectares
    area_ha = []

    # Calculate the area for each geometry
    for geom in gdf['geometry']:
        # Convert shapely geometry to QgsGeometry
        qgis_geom = QgsGeometry.fromWkt(geom.wkt)  # Convert shapely geometry to QgsGeometry
        
        # Calculate area of geometry in square meters
        area_m2 = d.measureArea(qgis_geom)
        
        # Convert square meters to hectares (1 hectare = 10,000 m²)
        area_ha.append(area_m2 / 10000)

    return area_ha








def extract_features_by_expression(input_shp, output_shp, expression):
    # Check if the output file already exists
    if os.path.exists(output_shp):
        print(f"Output file '{output_shp}' already exists. Skipping extraction.")
        return output_shp  # Return the existing output file path
    
    # Run the extraction if the file does not exist
    extracted_features = processing.run("native:extractbyexpression", 
                                        {'INPUT': input_shp,
                                         'EXPRESSION': expression,
                                         'OUTPUT': output_shp
                                        })['OUTPUT']
    
    print(f"Extraction completed. Output saved to '{output_shp}'")
    return extracted_features










