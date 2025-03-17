import os
import csv
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.validation import make_valid
import geojson
import zipfile
import ee
import geemap
import streamlit as st
import tempfile
import json 

my_project = 'ee-ineshummingbirds'
ee.Authenticate()
ee.Initialize(project= my_project)

# Function to process the uploaded files and calculate areas
def process_files(shp_file, xlsx_file, country, project_area_name, year_0, year_10, start_date, end_date, year_0_2020, year_1_2020, start_date_2020, end_date_2020, slope_percentage):
    # Read the shapefile
    gdf = gpd.read_file(shp_file).to_crs('EPSG:4326')
    File = geemap.geopandas_to_ee(gdf)

    # Read the Excel file
    df_forest_definition = pd.read_excel(xlsx_file)
    country_data = df_forest_definition[df_forest_definition['Country'] == country]
    cover_threshold = country_data['Tree_crown_cover_%'].values[0]
    cover_threshold = int(cover_threshold)
    height_threshold = country_data['Tree_height_m'].values[0]
    height_threshold = int(height_threshold)
    forest_size_ha = country_data['Area_ha'].values[0]
    forest_size_pixels = forest_size_ha * 11
    #minimum forest size for eligibility (1ha=11pixels)
    min_forest_pixels_list = [11, 55, 110] #1ha, 5ha, and 10ha

    #Define a function to reclassify the 'tree' class of ESA
    def reclassify(image):
        return image.where(image.eq(10), result)

    # Define functions for Landsat processing
    def apply_scale_factors(image):
        opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

    def maskSrClouds(image):
        qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
        saturation_mask = image.select('QA_RADSAT').eq(0)
        masked_image = image.updateMask(qa_mask).updateMask(saturation_mask)
        return masked_image

    def fillGap(image):
        return image.focalMedian(1.5, 'square', 'pixels', 2).blend(image)

    def rename(image):
        return image.select(
            ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
            ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])

    def renamel9(image):
        return image.select(
            ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])

    def create_landsat_collection(collection, start_year, end_year, start_date, end_date, region_of_interest, apply_scale_factors, maskSrClouds):
        collection = (
            ee.ImageCollection(collection)
            .filterBounds(region_of_interest)
            .filterDate(str(start_year) + start_date, str(end_year) + end_date)
            .filter(ee.Filter.lt('CLOUD_COVER', 30))
            .map(apply_scale_factors)
            .map(maskSrClouds)
        )
        return collection

    def create_landsat_collection_with_clouds(collection, start_year, end_year, start_date, end_date, region_of_interest, apply_scale_factors, maskSrClouds, renamel9):
        collection_l7 = (
            ee.ImageCollection(collection)
            .filterBounds(region_of_interest)
            .filterDate(str(start_year) + start_date, str(end_year) + end_date)
            .filter(ee.Filter.lt('CLOUD_COVER', 50))
            .map(apply_scale_factors)
            .map(maskSrClouds)
            .map(renamel9))
        return collection_l7

    def create_composite(collection_l7, collection, fillGap, File):
        landsat78 = collection_l7.merge(collection.select(
            ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']))
        composite78 = landsat78.map(fillGap).merge(collection.select(
            ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']))
        landsat78composite = composite78.median().clip(File)
        return landsat78composite

    def get_shapefile_centroid(gdf):
        """Ensure CRS is geographic and return the centroid coordinates."""
        if gdf.crs is None or gdf.crs.is_projected:
            gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84 (lat/lon)

        centroid = gdf.unary_union.centroid
        return centroid.y, centroid.x  # (latitude, longitude)

    def get_best_crs(latitude, longitude):
        utm_zone = int((180 + longitude) / 6) + 1
        return f"EPSG:{32600 + utm_zone if latitude >= 0 else 32700 + utm_zone}"


    def calculateTotalPixelArea(image, geometry):
        # Ensure the image is in a projected CRS (Web Mercator or UTM)
        image = image.reproject(best_epsg, None, 30)

        # Compute pixel area in hectares
        total_area = ee.Image.pixelArea().addBands(image).divide(10_000).reduceRegion(
            reducer=ee.Reducer.sum().group(1),  # Sum areas by class
            geometry=geometry,
            scale=30,
            bestEffort=True,
            tileScale=16  # Reduce memory usage
        )

        # Retrieve results
        try:
            result = total_area.getInfo()
            if not result:
                print("No area data found.")
                return None

            # Convert to a DataFrame
            df = pd.DataFrame.from_dict(result, orient='columns')
            print(df)
            return result

        except Exception as e:
            print(f"Error encountered: {e}")
            return None
        
    esa_legend_dict = {
        'Forest': '006400',
        'Shrubland': 'ffbb22',
        'Grassland': 'ffff4c',
        'Cropland': 'f096ff',
        'Built-up': 'fa0000',
        'Bare/sparse vegetation': 'b4b4b4',
        'Snow and ice': 'f0f0f0',
        'Permanent water bodies': '0064c8',
        'Herbaceous wetland': '0096a0',
        'Mangroves': '00cf75',
        'Moss and lichen': 'fae6a0'
    }

    legend_dict = {
        'Forest': '006400',
        'Non-forest': 'ffff4c',
        'Built-up': 'fa0000',
        'Permanent water bodies': '0064c8',
        'Other land': '0096a0',
    }

    #Get ESA World Classification
    gfc = ee.Image('UMD/hansen/global_forest_change_2023_v1_11').clip(File);
    canopyCover = gfc.select(['treecover2000']).clip(File).gte(cover_threshold)

    # Tree height
    tree_height = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').clip(File.geometry()).gte(height_threshold)

    # Overlay tree cover and tree height to have a layer presenting the threshold
    forest_mask = tree_height.multiply(canopyCover)  # Overlay the two layers

    #Load the ESA WorldCover and clip the area of interest ##IF NOT HAVING FOREST DEFINITION BY THE COUNTRY and small project area <100.000 ha
    esa = ee.ImageCollection('ESA/WorldCover/v200').first().clip(File.geometry())
    #Create a binary image from ESA just for tree (10)
    esa_10 = esa.eq(10)
    #Overlay ESA forest image and canopy cover
    result = esa_10.multiply(forest_mask)

    reclassify_map = reclassify(esa)
    # Reclassify the reference land-use map
    forest = reclassify_map.eq(1).selfMask().multiply(1)
    non_forest = reclassify_map.eq(0).Or(reclassify_map.eq(20)).Or(reclassify_map.eq(30)).Or(reclassify_map.eq(40)).selfMask().multiply(2)
    built_up = reclassify_map.eq(50).selfMask().multiply(3)
    water = reclassify_map.eq(80).selfMask().multiply(4)
    other_land = reclassify_map.eq(60).Or(reclassify_map.eq(70)).Or(reclassify_map.eq(90)).Or(reclassify_map.eq(95)).Or(reclassify_map.eq(100)).selfMask().multiply(5)
    new_esa = forest.blend(non_forest).blend(built_up).blend(water).blend(other_land)

    #get sample points for training and validation
    #Change according to the sampling method
    points = esa.sample(
        **{
            "region": File.geometry(),
            "scale": 30,
            "numPixels": 10000,
            "seed": 0,
            "geometries": True,
        })

    # Create Landsat collections
    collection_y0 = create_landsat_collection('LANDSAT/LC08/C02/T1_L2', year_0, year_0, start_date, end_date, File, apply_scale_factors, maskSrClouds)
    collection_l7_y0 = create_landsat_collection_with_clouds('LANDSAT/LE07/C02/T1_L2', year_0, year_0, start_date, end_date, File, apply_scale_factors, maskSrClouds, rename)
    landsat78composite_y0 = create_composite(collection_l7_y0, collection_y0, fillGap, File)

    collection_y10 = create_landsat_collection('LANDSAT/LC08/C02/T1_L2', year_10, year_10, start_date, end_date, File, apply_scale_factors, maskSrClouds)
    collection_l7_y10 = create_landsat_collection_with_clouds('LANDSAT/LC09/C02/T1_L2', year_10, year_10, start_date, end_date, File, apply_scale_factors, maskSrClouds, renamel9)
    landsat78composite_y10 = create_composite(collection_l7_y10, collection_y10, fillGap, File)

    #Landsat collection for 2020
    collection_2020 = create_landsat_collection('LANDSAT/LC08/C02/T1_L2', year_0_2020, year_1_2020, start_date_2020, end_date_2020, File, apply_scale_factors, maskSrClouds)
    collection_l7_2020 = create_landsat_collection_with_clouds('LANDSAT/LE07/C02/T1_L2', year_0_2020, year_1_2020, start_date_2020, end_date_2020, File, apply_scale_factors, maskSrClouds, rename)
    landsat78composite_2020 = create_composite(collection_l7_2020, collection_2020, fillGap, File)

    # Streamlit Map
    Map = geemap.Map()
    Map.add_basemap('HYBRID', False)
    Map.centerObject(File, 10)

    # Add Landsat composites to the map
    Map.addLayer(landsat78composite_y0, {'min': 0, 'max': 0.3, 'bands': ['SR_B4', 'SR_B3', 'SR_B2']}, 'Composite year 0')
    Map.addLayer(landsat78composite_y10, {'min': 0, 'max': 0.3, 'bands': ['SR_B4', 'SR_B3', 'SR_B2']}, 'Composite year 10')
    
    # Display the map
    #st.write(Map)

    # Classify the images
    bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    label = 'Map'

    training_2020 = landsat78composite_2020.select(bands).sampleRegions (**{
    'collection' : points,
    'properties' : [label],
    'scale'      : 30  # 30 m resolution based on the Landsat 8 resolution
})

    # Add a column for the accuracy assessment
    training_2020 = training_2020.randomColumn()

    training_2020_new = training_2020.filter(ee.Filter.lt('random', 0.7))
    validation_2020_new = training_2020.filter(ee.Filter.gte('random', 0.7))

    # Using the Classifier.smilecart machine learning to predict and classify the land cover
    trained_2020 = ee.Classifier.smileRandomForest(10).train(training_2020_new, label, bands)# Train the classifier using the trianing data generated

    # Reclassifying the image classes + values
    result_y10 = landsat78composite_y10.select(bands).classify(trained_2020) # classify the image/raster
    result_y0 = landsat78composite_y0.select(bands).classify(trained_2020)

    # Reclassify the results
    forest_y10 = result_y10.eq(10).selfMask()
    forest_y0 = result_y0.eq(10).selfMask()

    contArea_y10 = forest_y10.connectedPixelCount()
    area_y10 = contArea_y10.gte(forest_size_pixels).selfMask()

    contArea_y0 = forest_y0.connectedPixelCount()
    area_y0 = contArea_y0.gte(forest_size_pixels).selfMask()

    reclassify_y10 = result_y10.where(result_y10.eq(10), area_y10)
    reclassify_y0 = result_y0.where(result_y0.eq(10), area_y0)

    class_values = esa.get('Map_class_values')#.getInfo()
    class_palette = esa.get('Map_class_palette')#.getInfo()
    class_names = esa.get('Map_class_names')
    # Reclassifying the class using the original class names and class palette
    landcover_y10 = reclassify_y10.set ('classification_class_values', class_values)
    landcover_y10 = landcover_y10.set('classification_class_palette', class_palette)
    landcover_y10 = landcover_y10.set('classification_class_names', class_names)
    landcover_y0 = reclassify_y0.set ('classification_class_values', class_values)
    landcover_y0 = landcover_y0.set('classification_class_palette', class_palette)
    landcover_y0 = landcover_y0.set('classification_class_names', class_names)

    #Overall accuracy
    #Training dataset
    training_accuracy = trained_2020.confusionMatrix()
    overall_accuracy = training_accuracy.accuracy()
    #print(overall_accuracy.getInfo())

    # Accuracy on Validation dataset
    validation = validation_2020_new.classify(trained_2020)
    validation.first().getInfo()

    #validation_accuracy = validation.errorMatrix('Map', 'classification')

    ##FOR ESA ONLY
    non_forest_y0 = reclassify_y0.eq(20).Or(reclassify_y0.eq(30)).Or(reclassify_y0.eq(40)).Or(reclassify_y0.eq(60))
    non_forest_y10 = reclassify_y10.eq(20).Or(reclassify_y10.eq(30)).Or(reclassify_y10.eq(40)).Or(reclassify_y10.eq(60))

    dataset = ee.Image('USGS/SRTMGL1_003').select('elevation')
    slope = ee.Terrain.slope(dataset)                                                    # Getting the slope
    #possible to change the slope in here (if not consider just change to 0)
    slope_30 = slope.updateMask(slope.gt(30)).updateMask(slope.lt(100)).gt(30)
    #### Getting the difference between the land cover in 2023 and 2013 by overlaying those 2 rasters on each other - Multiplication "Py language" ####

    # Overlaying to see the differences between 2023 and 2013
    overlayed_10year = reclassify_y0.multiply(reclassify_y10)

    # Overlaying to see the differences between 2023 and 2013 ONLY FOR ESA
    overlayed_10year = non_forest_y0.multiply(non_forest_y10)

    # Overlaying with slope
    Slope_Mul_Overlayed = slope_30.multiply(overlayed_10year.eq(1))  
    print(Slope_Mul_Overlayed.getInfo())   

    latitude, longitude = get_shapefile_centroid(gdf)
    #print(f"Central Point: ({latitude}, {longitude})")
    best_epsg = get_best_crs(latitude, longitude)  # Replace with actual latitude
    #print(best_epsg)

    gdf_crs = gdf.to_crs(best_epsg)
    total_area_ha = (gdf_crs['geometry'].area/10000).sum()
    print(f"Total area in hectares: {total_area_ha}")

    #CALCULATE THE FOREST AREA FOR YEAR 0 AND YEAR 10
    print('Forest year 0:')
    forest_year0 = calculateTotalPixelArea(forest_y0, File)

    print('Forest year 10:')
    forest_year10 = calculateTotalPixelArea(forest_y10, File)

    #Calculate total eligible area
    print('Non-eligible and eligible area are:')
    eligible_area = calculateTotalPixelArea(overlayed_10year, File)

    results = []

    for min_forest_pixels in min_forest_pixels_list:
        contArea = overlayed_10year.eq(1).selfMask().connectedPixelCount()
        area = contArea.gte(min_forest_pixels).selfMask()
        areas_slope = calculateTotalPixelArea(area.selfMask(), File)

        # Append results to the list
        results.append({"min_forest_pixels": min_forest_pixels, "Total Area (ha)": areas_slope})

    # Convert to DataFrame for a structured table
    df = pd.DataFrame(results)

    # Return the results (in a displayable format)
    return forest_year0, forest_year10, total_area_ha, eligible_area, overall_accuracy, df



# Streamlit app
st.title("Forest Eligibility Analysis")

# File upload
uploaded_shp = st.file_uploader("Upload a zip file containing shapefile files", type="zip")
uploaded_xlsx = st.file_uploader("Upload the Countries_Forest_Definition.xlsx file", type="xlsx")

if uploaded_shp and uploaded_xlsx:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract the shapefile
        with zipfile.ZipFile(uploaded_shp, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # Find the .shp file
        shp_file = None
        for file in os.listdir(tmpdir):
            if file.endswith('.shp'):
                shp_file = os.path.join(tmpdir, file)
                break

        if shp_file:
            # Input parameters
            country = st.text_input("Enter the country name", "Ivory Coast")
            project_area_name = st.text_input("Enter the project area name", "Calao")
            year_0 = st.number_input("Enter the start year", value=2014)
            year_10 = st.number_input("Enter the end year", value=2024)
            start_date = st.text_input("Enter the start date for 2014/2024 (format: -mm-dd)", "-01-01")
            end_date = st.text_input("Enter the end date for 2014/2024 (format: -mm-dd)", "-03-30") 
            year_0_2020 = st.number_input("Enter the start year", value=2019)
            year_1_2020 = st.number_input("Enter the start year", value=2020)
            start_date_2020 = st.text_input("Enter the start date for 2020 (format: -mm-dd)", "-12-01")
            end_date_2020 = st.text_input("Enter the start date for 2020 (format: -mm-dd)", "-02-01")
            slope_percentage = st.number_input("Enter the slope percentage", value=30)

            if st.button("Process"):
                # Process the files
                forest_year0, forest_year10, total_area_ha, eligible_area, overall_accuracy, df = process_files(
                    shp_file, uploaded_xlsx, country, project_area_name, year_0, year_10, start_date, end_date, year_0_2020, year_1_2020, start_date_2020, end_date_2020, slope_percentage
                )

                # Display results
                st.header('Project Summary')
                st.subheader('Forest Area')

                # Function to extract sum values
                def extract_sum(data):
                    return sum(item["sum"] for item in data["groups"])

                # Display forest area
                st.write(f"**Forest area in year 0:** {extract_sum(forest_year0):,.2f} ha")
                st.write(f"**Forest area in year 10:** {extract_sum(forest_year10):,.2f} ha")
                st.write(f"**Overall Accuracy of the model:** {((overall_accuracy.getInfo())*100):,.2f} %")

                # Display total project area
                st.subheader("Total Project Area")
                st.write(f"**{total_area_ha:,.2f} ha**")

                # Display eligible area
                st.subheader("Eligible Area")
                #st.write(f"**Total Eligible Area:** {extract_sum(eligible_area):,.2f} ha")

                group_names = {0: 'Non-eligible', 1: 'Eligible'}

                # Breakdown of eligible areas
                for i, group in enumerate(eligible_area["groups"]):
                    group_label = group_names.get(i, f'Group{i}')
                    st.write(f"• **{group_label}:** {group['sum']:,.2f} ha")

                st.dataframe(df)

                
        else:
            st.error("No shapefile found in the uploaded zip file.")
