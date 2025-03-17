import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.validation import make_valid
import zipfile
import ee
#import geemap
import streamlit as st
import tempfile
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

my_project = 'ee-ineshummingbirds'
ee.Authenticate()
ee.Initialize(project= my_project)

# Function to process the uploaded files and calculate areas
def process_files(shp_file, start_date, end_date, dry_season_1stmonth, dry_season_lastmonth, wet_season_1stmonth, wet_season_lastmonth, wf_startDate, wf_endDate):
    # Read the shapefile
    gdf = gpd.read_file(shp_file).to_crs('EPSG:4326')
    region = geemap.geopandas_to_ee(gdf)

    # Count the number of days with at least one pixel exceeding the threshold
    def count_hot_days(image):
        mask = image.reduceRegion(
            reducer=ee.Reducer.anyNonZero(),
            geometry=region,
            scale=1000,
            maxPixels=1e8
        )
        is_above = ee.Algorithms.If(mask.get('temperature_2m'), 1, 0)
        return ee.Feature(None, {'date': image.get('system:time_start'), 'day_above_32': is_above})

    def count_flood_events(year):
        start_date = ee.Date.fromYMD(year, 1, 1)
        end_date = ee.Date.fromYMD(year, 12, 31)

        yearly_floods = (gfd.filterDate(start_date, end_date)
                        .select('flooded')
                        .map(lambda img: img.gt(0).And(jrc.Not()))
                        .sum()
                        .gt(0))

        clipped_flood = yearly_floods.clip(region)

        flood_check = clipped_flood.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region.geometry(),
            scale=500,
            maxPixels=1e8
        )

        return ee.Feature(None, {'year': year, 'flood_count': flood_check.get('flooded')})

    # Process burned area per year
    def process_year(n):
        ini = startDate.advance(n, 'year')
        end = ini.advance(1, 'year')

        result = sst.filterDate(ini, end).max()

        # Ensure burned areas are correctly masked
        result = result.updateMask(result.gt(0))  # Keep only burned pixels

        # Compute burned area in hectares
        burned_area = ee.Image.pixelArea() \
                        .divide(10000) \
                        .updateMask(result) \
                        .reduceRegion(
                            reducer=ee.Reducer.sum(),
                            geometry=region,
                            scale=500,
                            maxPixels=1e12
                        )

        return ee.Feature(None, {
            'year': ini.get('year'),  # Ensure year is stored correctly
            'burned_area_ha': burned_area.get('area')
        })


    def get_shapefile_centroid(gdf):
        """Ensure CRS is geographic and return the centroid coordinates."""
        if gdf.crs is None or gdf.crs.is_projected:
            gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84 (lat/lon)

        centroid = gdf.unary_union.centroid
        return centroid.y, centroid.x  # (latitude, longitude)

    def get_best_crs(latitude, longitude):
        """ Returns the best UTM zone EPSG code based on latitude """
        utm_zone = int((180 + longitude) / 6) + 1
        return f"EPSG:{32600 + utm_zone if latitude >= 0 else 32700 + utm_zone}"
    
    #load the datasets
    #temperature - ERA5-Land dataset (temperature in Kelvin)
    temp_dataset = (ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                    .filterDate(start_date, end_date)
                    .select('temperature_2m'))

    #precipitation - ERA5-Land dataset (precipitation in meters)
    precip_dataset = (ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                    .filterDate(start_date, end_date)
                    .select('total_precipitation'))

    #floods -  MODIS Global Flood Database (GFD) and JRC permanent water mask
    gfd = ee.ImageCollection('GLOBAL_FLOOD_DB/MODIS_EVENTS/V1')
    jrc = (ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
        .select('waterClass')
        .map(lambda img: img.eq(3))  # Permanent water class
        .max())

    #drought - SPEI dataset
    spei_dataset = ee.ImageCollection("CSIC/SPEI/2_10") \
        .filterBounds(region) \
        .filterDate("2000-01-01", "2022-01-01")
    
    # MODIS Burned Area dataset
    sst = ee.ImageCollection("MODIS/061/MCD64A1") \
                .select('BurnDate') \
                .filterDate(wf_startDate, wf_endDate)

    #-----------------------------------------------------------------------------------
    #-----------------------------------TEMPERATURE-------------------------------------
    #-----------------------------------------------------------------------------------

    # Calculate the mean, min, and max annual temperature
    mean_temp = temp_dataset.mean().clip(region)
    min_temp = temp_dataset.min().clip(region)
    max_temp = temp_dataset.max().clip(region)

    # Convert temperature from Kelvin to Celsius
    mean_temp_celsius = mean_temp.subtract(273.15)
    min_temp_celsius = min_temp.subtract(273.15)
    max_temp_celsius = max_temp.subtract(273.15)

    # Calculate the average, min, and max temperature over the region
    temp_stats = mean_temp_celsius.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region.geometry(),
        scale=1000,
        bestEffort=True
    )

    min_stats = min_temp_celsius.reduceRegion(
        reducer=ee.Reducer.min(),
        geometry=region.geometry(),
        scale=1000,
        bestEffort=True
    )

    max_stats = max_temp_celsius.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=region.geometry(),
        scale=1000,
        bestEffort=True
    )

    # Extract and print temperature values
    avg_temp = temp_stats.get('temperature_2m').getInfo()
    min_temp_value = min_stats.get('temperature_2m').getInfo()
    max_temp_value = max_stats.get('temperature_2m').getInfo()

    # Define threshold temperature in Celsius
    threshold = 32
    threshold_k = threshold + 273.15  # Convert to Kelvin

    # Apply threshold to identify hot pixels
    col_threshold = temp_dataset.map(lambda image: image.gt(threshold_k).set('system:time_start', image.get('system:time_start')))

    # Convert to FeatureCollection
    days_above_32_fc = ee.FeatureCollection(col_threshold.map(count_hot_days))

    # Sum the number of hot hours
    total_hot_hours = days_above_32_fc.aggregate_sum('day_above_32')

    # Convert hot hours to days
    total_days_above_32 = ee.Number(total_hot_hours).divide(24).getInfo()

    #-----------------------------------------------------------------------------------
    #-----------------------------------PRECIPITATION-----------------------------------
    #-----------------------------------------------------------------------------------

    # Convert hourly precipitation to daily precipitation (sum over each day)
    daily_precip = precip_dataset.map(lambda image: image.multiply(1000).set('date', ee.Date(image.date()).format('YYYY-MM-dd')))

    # Compute cumulative annual precipitation
    cumulative_annual_precip = daily_precip.sum().clip(region)
    cumulative_annual_precip_stats = cumulative_annual_precip.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region.geometry(),
        scale=1000,
        bestEffort=True
    )

    # Compute daily average precipitation
    daily_avg_precip_stats = ee.Dictionary(cumulative_annual_precip_stats).map(lambda key, value: ee.Number(value).divide(365))

    # Compute cumulative precipitation for wet season
    wet_season_precip = precip_dataset.filter(ee.Filter.calendarRange(wet_season_1stmonth, wet_season_lastmonth, 'month')).sum().multiply(1000).clip(region)
    wet_season_precip_stats = wet_season_precip.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region.geometry(),
        scale=1000,
        bestEffort=True
    )

    # Compute cumulative precipitation for dry season
    dry_season_precip = precip_dataset.filter(ee.Filter.calendarRange(dry_season_1stmonth, dry_season_lastmonth, 'month')).sum().multiply(1000).clip(region)
    dry_season_precip_stats = dry_season_precip.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region.geometry(),
        scale=1000,
        bestEffort=True
    )

    # Extract and print precipitation values
    cumulative_annual_precip_value = cumulative_annual_precip_stats.get('total_precipitation').getInfo()
    daily_avg_precip_value = daily_avg_precip_stats.get('total_precipitation').getInfo()
    wet_season_precip_value = wet_season_precip_stats.get('total_precipitation').getInfo()
    dry_season_precip_value = dry_season_precip_stats.get('total_precipitation').getInfo()

    #-----------------------------------------------------------------------------------
    #-----------------------------------FLOODS------------------------------------------
    #-----------------------------------------------------------------------------------

    # Define years for flood analysis
    years = ee.List.sequence(2000, 2018)

    # Convert flood counts to FeatureCollection
    flood_counts_fc = ee.FeatureCollection(years.map(count_flood_events))

    total_floods = flood_counts_fc.aggregate_sum('flood_count').getInfo()

    # Classify risk
    if total_floods == 0:
        risk_level_f = "Low Risk"
    elif total_floods == 1:
        risk_level_f = "Medium Risk"
    else:
        risk_level_f = "High Risk"
    
    #-----------------------------------------------------------------------------------
    #-----------------------------------DROUGHT-----------------------------------------
    #-----------------------------------------------------------------------------------

    # Compute average SPEI indices
    spei_avg = spei_dataset.reduce(ee.Reducer.mean()).clip(region).select([
        "SPEI_03_month_mean", "SPEI_06_month_mean", "SPEI_09_month_mean", "SPEI_12_month_mean"
    ])

    # Compute drought risk based on SPEI-9
    chart_data = spei_dataset.map(lambda image:
        ee.Feature(None, {
            "Date": image.get("system:time_start"),
            "SPEI_09_month": image.select("SPEI_09_month").reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region.geometry(),
                scale=55660,
                maxPixels=1e9
            ).get("SPEI_09_month")
        })
    )

    # Convert to FeatureCollection
    chart_list = chart_data.toList(chart_data.size())

    # Filter drought events (SPEI-9 < -1.5)
    drought_events = chart_list.filter(ee.Filter.lt("SPEI_09_month", -1.5))

    # Compute drought risk percentage
    total_features = chart_list.size()
    num_drought_events = drought_events.size()
    percentage_drought = ee.Number(num_drought_events).divide(total_features).multiply(100)

    # Classify drought risk
    risk_level = ee.Algorithms.If(
        percentage_drought.lt(5), "Low risk",
        ee.Algorithms.If(percentage_drought.lt(15), "Medium risk", "High risk")
    )

    percentage_drought = percentage_drought.getInfo()

    #-----------------------------------------------------------------------------------
    #-----------------------------------WILDFIRES---------------------------------------
    #-----------------------------------------------------------------------------------

    startDate = ee.Date(wf_startDate)
    endDate = ee.Date(wf_endDate)

    # calculate number of years to process
    nYears = ee.Number(endDate.difference(startDate, 'year')).round().subtract(1)
    #print(f'Number of years: {nYears.getInfo()}')

    # processs burned area per year
    byYear = ee.FeatureCollection(
        ee.List.sequence(0, nYears).map(process_year)
    )

    #features from the Earth Engine FeatureCollection
    features = byYear.getInfo()['features']

    #'area_ha' values and their corresponding years
    data = []
    for feature in features:
        year = feature['id']  # The id corresponds to the year index (0-9 in your case)
        area_ha = feature['properties']['burned_area_ha']
        data.append({'year': int(year), 'burned_area_ha': area_ha})

    #convert to pandas dataframe
    df_wf = pd.DataFrame(data)
    #retrive lat and long to get the adequate CRS for a correct area calculation
    latitude, longitude = get_shapefile_centroid(gdf)
    best_epsg = get_best_crs(latitude, longitude)

    #calculate total area
    gdf_crs = gdf.to_crs(best_epsg)
    total_area_ha = (gdf_crs['geometry'].area/10000).sum()
    #add new column to the df with the percentage of burned area per year
    df_wf['burned_area_percentage'] = (df_wf['burned_area_ha'] / total_area_ha) * 100

    # Calculate mean and standard deviation for area burned in hectares
    #mean_area = df['burned_area_ha'].mean()
    #std_area = df['burned_area_ha'].std()
    mean_area_percentage = df_wf['burned_area_percentage'].mean()
    #std_area_percentage = df_wf['burned_area_percentage'].std()

    # Step 1: Identify big fire years
    df_wf['is_big_fire_year'] = df_wf['burned_area_percentage'] > 30 #(mean_area_percentage + std_area_percentage)
    # Step 2: Calculate frequency of big fire years
    big_fire_frequency = df_wf['is_big_fire_year'].mean() * 100  # Frequency in percentage
    # Step 3: Classify fire risk
    if mean_area_percentage < 10:
        if big_fire_frequency > 20:
            risk_level_wf = "Medium"
        else:
            risk_level_wf = "Low"
    elif 10 <= mean_area_percentage <= 30:
        if big_fire_frequency > 20:
            risk_level_wf = "High"
        else:
            risk_level_wf = "Medium"
    else:
        risk_level_wf = "High"

    return avg_temp, min_temp_value, max_temp_value, total_days_above_32, cumulative_annual_precip_value, daily_avg_precip_value, wet_season_precip_value, dry_season_precip_value, total_floods, risk_level_f, percentage_drought, risk_level, mean_area_percentage, big_fire_frequency, risk_level_wf, df_wf

# Streamlit app
st.title("Risk Classification")

# File upload
uploaded_shp = st.file_uploader("Upload a zip file containing shapefile files", type="zip")

if uploaded_shp:
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
            start_date = st.text_input("Enter the start date (format: yyyy-mm-dd)", "2024-01-01")
            end_date = st.text_input("Enter the end date (format: yyyy-mm-dd)", "2024-12-31") 
            dry_season_1stmonth = st.number_input("Enter the number of the first month of dry season", 11)
            dry_season_lastmonth = st.number_input("Enter the number of the last month of dry season", 5)
            wet_season_1stmonth = st.number_input("Enter the number of the first month of wet season",6)
            wet_season_lastmonth = st.number_input("Enter the number of the last month of wet season", 10)
            wf_startDate = st.text_input("Enter the start date for wildfires (format: yyyy-mm-dd)", "2000-01-01")
            wf_endDate = st.text_input("Enter the start date for wildfires (format: yyyy-mm-dd)", "2024-12-31")


            if st.button("Process"):
                # Process the files
                avg_temp, min_temp_value, max_temp_value, total_days_above_32, cumulative_annual_precip_value, daily_avg_precip_value, wet_season_precip_value, dry_season_precip_value, total_floods, risk_level_f, percentage_drought, risk_level, mean_area_percentage, big_fire_frequency, risk_level_wf, df_wf = process_files(
                    shp_file, start_date, end_date, dry_season_1stmonth, dry_season_lastmonth, wet_season_1stmonth, wet_season_lastmonth, wf_startDate, wf_endDate
                )

                # Display results
                st.header(f'Risk Classification in {project_area_name}, {country}')
                # Display temperatura
                st.subheader('Temperature 2024')

                st.write(f'Average Annual Temperature (°C): {avg_temp:.2f}')
                st.write(f'Minimum Annual Temperature (°C): {min_temp_value:.2f}')
                st.write(f'Maximum Annual Temperature (°C): {max_temp_value:.2f}')
                st.write(f'Total number of days with at least one pixel above 32°C: {total_days_above_32:.2f}')

                # Display precipitation
                st.subheader("Precipitation 2024")
                st.write(f'Cumulative Annual Precipitation (mm): {cumulative_annual_precip_value:.2f}')
                st.write(f'Daily Average Precipitation (mm): {daily_avg_precip_value:.2f}')
                st.write(f'Wet Season Cumulative Precipitation (mm): {wet_season_precip_value:.2f}')
                st.write(f'Dry Season Cumulative Precipitation (mm): {dry_season_precip_value:.2f}')

                # Display the flood risks
                st.subheader("Floods 2000-2018")

                if total_floods > 0:
                    st.write('Floods detected in the project area')
                else:
                    st.write('No floods detected in the project area')
                
                st.write('Number of total flood events:', total_floods)
                st.write(f'Flood Risk Level: {risk_level_f}')

                # Display the drought risks
                st.subheader("Drought 2000-2022")
                st.write(f"Percentage of months with severe drought: {percentage_drought:.2f}")
                st.write("Drought Risk Level:", risk_level.getInfo())

                #Display the wildfire risks
                st.subheader("Wildfires 2000-2024")
                st.write(f'Mean of burned area (%): {mean_area_percentage:.2f}')
                st.write(f'Frequency of big fire years (%): {big_fire_frequency:.2f}')
                st.write(f'Wildfire risk Level: {risk_level_wf}')

                # Define risk thresholds
                low_risk_threshold = 10  # Low risk threshold (10%)
                high_risk_threshold = 30  # High risk threshold (30%)
                mean_threshold = df_wf['burned_area_percentage'].mean()  # Mean burned area percentage

                # Add a column for point size (optional, for visualization)
                df_wf['point_size'] = df_wf['burned_area_percentage'] * 10  # Scale size for better visualization

                # Streamlit app layout
                st.subheader("Burned Area Percentage Analysis")
                st.write("Scatter plot showing annual burned area percentage with risk thresholds.")

                # Create the scatter plot
                fig, ax = plt.subplots(figsize=(12, 7))
                sns.scatterplot(
                    data=df_wf,
                    x='year',
                    y='burned_area_percentage',
                    size='point_size',
                    hue='burned_area_percentage',
                    palette='coolwarm',
                    legend=False,
                    ax=ax
                )

                # Add horizontal lines for risk thresholds
                ax.axhline(y=low_risk_threshold, color='green', linestyle='--', label='Low Risk Threshold (10%)')
                ax.axhline(y=high_risk_threshold, color='red', linestyle='--', label='High Risk Threshold (30%)')
                ax.axhline(y=mean_threshold, color='blue', linestyle='-', label=f'Mean Burned Area ({mean_threshold:.2f}%)')

                # Set labels and title
                ax.set_title('Percentage of Burned Area per Year with Risk Thresholds', fontsize=16)
                ax.set_xlabel('Year', fontsize=14)
                ax.set_ylabel('Burned Area (%)', fontsize=14)

                # Set y-axis limits
                ax.set_ylim(0, df_wf['burned_area_percentage'].max() * 1.1)

                # Show legend
                ax.legend(loc='upper right', fontsize=12)

                # Display the plot in Streamlit
                st.pyplot(fig)

        else:
            st.error("No shapefile found in the uploaded zip file.")

    



