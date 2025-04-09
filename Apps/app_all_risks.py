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
import pandas as pd
import math
import seaborn as sns
from google.oauth2 import service_account
from ee import oauth
import statistics
import numpy as np
import geemap.foliumap as geemap

# Access credentials from Streamlit secrets
secrets = st.secrets["GOOGLE_CREDENTIALS"]

# Convert the secrets into a dictionary that can be used with ServiceAccountCredentials
service_account_info = {
    "type": secrets["type"],
    "project_id": secrets["project_id"],
    "private_key_id": secrets["private_key_id"],
    "private_key": secrets["private_key"],
    "client_email": secrets["client_email"],
    "client_id": secrets["client_id"],
    "auth_uri": secrets["auth_uri"],
    "token_uri": secrets["token_uri"],
    "auth_provider_x509_cert_url": secrets["auth_provider_x509_cert_url"],
    "client_x509_cert_url": secrets["client_x509_cert_url"],
    "universe_domain": secrets["universe_domain"]
}

# Use the service account credentials to authenticate with Google Earth Engine
credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=oauth.SCOPES
)

# Initialize the Earth Engine API with the credentials
#ee.Authenticate()
ee.Initialize(credentials)

#my_project = 'ee-ineshummingbirds'
#ee.Authenticate()
#ee.Initialize()
#ee.Initialize(project= my_project)

# Function to process the uploaded files and calculate areas
def process_files(shp_file, start_date, end_date, dry_season_1stmonth, dry_season_lastmonth, wet_season_1stmonth, wet_season_lastmonth):
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
    
    # Define slope class ranges
    def classify_slope(slope):
        return (slope
                .where(slope.lte(15), 1)   # Flat to very gently sloping (Very low)
                .where(slope.gt(15).And(slope.lte(30)), 2)  # Gently sloping (Low)
                .where(slope.gt(30), 3))  # Steep (Extremely high)

    # Compute area per class
    def compute_area(class_value, region):
        area = area_per_pixel.updateMask(slope_classified.eq(class_value))
        total_area = area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=30,
            maxPixels=1e13
        ).getInfo()
        return total_area['area']

    # Create a function to process burned area by year
    def process_year(n):
        # Calculate the start and end date for each year
        ini = startDate.advance(n, 'year')
        end = ini.advance(1, 'year')

        # Filter the burned area collection for the given year
        result = sst.filterDate(ini, end)
        result = result.max().set('system:time_start', ini)

        # Get the burned area (where BurnDate is not 0) and mask it
        result = ee.Image.pixelArea() \
                .divide(10000) \
                .updateMask(result.neq(0))  # Mask out non-burned areas

        # Sum the area of burned forest for the year
        result = result.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=500,
            maxPixels=1e12,
            tileScale=4
        )

        # Extract the area burned in the forest for that year
        burnedArea = result.get('area')

        # Return the area burned in the forest for that year
        return ee.Feature(None, {'burned_area_ha': burnedArea})

    # Process burned area per year
    def process_year1(n):
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

    # Classify risk based on the average yearly extreme heat days
    def classify_risk(total_days):
        if total_days < 30:
            return 'Low risk'
        elif total_days <= 90:
            return 'Medium risk'
        else:
            return 'High risk'

    # Function to remove duplicates based on the date within each collection
    def remove_duplicates(collection):
        return collection.map(lambda image: image.set('date', ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'))).distinct('date')

    # Apply threshold to the averaged collection
    def apply_threshold(image):
        thresholdImage = image.gt(thresholdK)  # Identify pixels above 32°C
        return thresholdImage.set('system:time_start', image.get('system:time_start'))


    # Count the number of days where at least one pixel exceeded the threshold
    def count_days_above_35(image):
        mask = image.reduceRegion(
            reducer=ee.Reducer.anyNonZero(),
            geometry=region,
            scale=5000,
            bestEffort=True
        )

        isAbove = ee.Algorithms.If(mask.get('tasmax'), 1, 0)

        return ee.Feature(None, {'date': image.get('system:time_start'), 'day_above_32': isAbove})

    # Function to compute the average image at a given index
    def mean_image_list(collections, indices):
        def compute_mean(i):
            images = [ee.Image(collection.get(i)) for collection in collections]
            mean_img = ee.ImageCollection(images).mean()
            return mean_img.set('system:time_start', images[0].get('system:time_start'))

        return indices.map(compute_mean)

    # Function to filter images by season
    def filter_by_season(image_collection, start_month, end_month):
        return image_collection.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))

    # Extract the month and year from the date to group by month
    def add_month_year(image):
        date = ee.Date(image.get('system:time_start'))
        month = date.get('month')
        year = date.get('year')
        return image.set('month', month).set('year', year)

    # Group by month and calculate the mean for both temperature and precipitation
    def calculate_monthly_means(month):
        # Filter the datasets by the current month
        tempData = tempWithMonth.filter(ee.Filter.eq('month', month))
        precipData = precipWithMonth.filter(ee.Filter.eq('month', month))

        # Calculate the mean temperature for that month across all years
        tempMean = tempData.mean().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10000,  # Adjust based on your area of interest
            maxPixels=1e8
        )

        # Calculate the mean precipitation for that month across all years
        precipMean = precipData.mean().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10000,  # Adjust based on your area of interest
            maxPixels=1e8
        )

        # Convert temperature from Kelvin to Celsius (subtract 273.15)
        temperatureCelsius = ee.Number(tempMean.get('temperature_2m')).subtract(273.15)

        # Convert precipitation from meters to millimeters (multiply by 1000)
        precipitationMillimeters = ee.Number(precipMean.get('total_precipitation_sum')).multiply(1000)

        # Create a feature with the month, temperature in Celsius, and precipitation in millimeters
        return ee.Feature(None, {
            'month': month,
            'mean_temperature_celsius': temperatureCelsius,
            'mean_precipitation_mm': precipitationMillimeters
        })

    #load the datasets
    #temperature - ERA5-Land dataset (temperature in Kelvin)
    temp_dataset = (ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                    .filterDate(start_date, end_date)
                    .select('temperature_2m'))

    #precipitation - ERA5-Land dataset (precipitation in meters)
    precip_dataset = (ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD')
              .filterDate(start_date, end_date))

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
                .filterDate("2000-01-01", "2024-12-31")

    #-----------------------------------------------------------------------------------
    #-----------------------------------ELEVATION---------------------------------------
    #-----------------------------------------------------------------------------------
    #DEM
    dem_dataset = ee.Image('USGS/SRTMGL1_003').clip(region)
    elevation = dem_dataset.select('elevation')
    #Elevation
    #calculate mean, min and max elevation value
    elevation_stats = elevation.reduceRegion(
        reducer=ee.Reducer.min().combine(ee.Reducer.max(), None, True).combine(ee.Reducer.mean(), None, True),
        geometry=region.geometry(),
        scale=30,
        bestEffort=True
    )
    
    elevation_min_value = elevation_stats.get('elevation_min').getInfo()
    elevation_max_value = elevation_stats.get('elevation_max').getInfo()
    elevation_mean_value = elevation_stats.get('elevation_mean').getInfo()
    
    #calculate the slope
    slope = ee.Terrain.slope(elevation).clip(region)
    
    slope_stats = slope.reduceRegion(
        reducer=ee.Reducer.min().combine(ee.Reducer.max(), None, True).combine(ee.Reducer.mean(), None, True).combine(ee.Reducer.mode(), None, True),
        geometry=region,
        scale=30,  # change resolution if needed
        maxPixels=1e13
    )
    
    slope_min = slope_stats.get('slope_min').getInfo()
    slope_max = slope_stats.get('slope_max').getInfo()
    slope_mean = slope_stats.get('slope_mean').getInfo()
    slope_mode = slope_stats.get('slope_mode').getInfo()
    
    #convert from degrees to percentage
    slope_min_percentage = math.tan(math.radians(slope_min)) * 100
    slope_max_percentage = math.tan(math.radians(slope_max)) * 100
    slope_mean_percentage = math.tan(math.radians(slope_mean)) * 100
    slope_mode_percentage = math.tan(math.radians(slope_mode)) * 100
    
    # Classify the risk of erosion based on degrees
    if slope_mode_percentage < 15:
      risk_level_erosion = "Low risk"
    elif slope_mode_percentage <= 30:
      risk_level_erosion = "Medium risk"
    else:
      risk_level_erosion = "High risk"
    
    # Apply classification
    slope_classified = classify_slope(slope)
    
    # Convert to area (hectares)
    area_per_pixel = ee.Image.pixelArea().divide(10000)  # Convert to hectares
    
    # Create table data
    classes = [
        (1, "0-15", "Flat to very gently", "Low"),
        (2, "15-30", "Gently slope", "Medium"),
        (3, ">30", "Sloping", "High")
    ]
    
    # Compute areas
    data = []
    total_land = sum([compute_area(c[0], region) for c in classes])  # Total land area
    
    for c in classes:
        area_ha = compute_area(c[0], region)
        percentage = (area_ha / total_land) * 100 if total_land else 0
        data.append([c[0], c[1], c[2], c[3], f"{area_ha:,.2f}", f"{percentage:.1f}%"])
    
    # Create DataFrame
    df_elevation = pd.DataFrame(data, columns=["No", "Classes (°)", "Characteristics", "Susceptibility", "Area (ha)", "Area (%)"])
    
    # Display the table
    #print(df)

    # Define visualization parameters for a green-yellow-green color scheme
    vis_params = {
        'min': elevation_min_value,
        'max': elevation_max_value,
        'palette': ['green', 'yellow', 'red']  # Green for low, yellow for mid, green for high
    }

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
    
    #-----------------------------------------------------------------------------------
    #----------------------------------THERMAL-STRESS-----------------------------------
    #-----------------------------------------------------------------------------------

    # 2024
    # Load the ERA5-Land Daily Aggregated dataset
    temp_dataset = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .select('temperature_2m_max')  # Select daily max temperature
    
    # Define the temperature threshold (35°C in Kelvin)
    temp_threshold = ee.Number(35).add(273.15)
    
    # Count the number of days exceeding 35°C
    hot_days = temp_dataset.map(lambda image: image.gt(temp_threshold).set('date', image.date())).sum()
    
    # Reduce the count over the region (AOI)
    hot_days_count_24 = hot_days.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=1000,
        maxPixels=1e9
    )
    
    
    # 2050
    models = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL']
    hot_days_list = []
    
    for model in models:
      temp_dataset = ee.ImageCollection('NASA/GDDP-CMIP6') \
          .filter(ee.Filter.date('2050-01-01', '2050-12-31')) \
          .filter(ee.Filter.eq('model', model)) \
          .select('tasmax')
    
      # Define the temperature threshold (35°C in Kelvin)
      temp_threshold = ee.Number(35).add(273.15)
    
      # Count the number of days exceeding 35°C
      hot_days = temp_dataset.map(lambda image: image.gt(temp_threshold).set('date', image.date())).sum()
    
      # Reduce the count over the region (AOI)
      hot_days_count = hot_days.reduceRegion(
          reducer=ee.Reducer.mean(),
          geometry=region,
          scale=1000,
          maxPixels=1e9
      )
      hot_days_list.append(hot_days_count.get('tasmax').getInfo())
      # Print the result for 2024
      #print('Number of days above 35C:', hot_days_count.get('tasmax').getInfo())
    
    average = statistics.mean(hot_days_list)
    
    # Classify risk
    if average < 30:
        risk_level_thermal = "Low risk"
    elif average > 90:
        risk_level_thermal = "High risk"
    else:
        risk_level_thermal = "Medium risk"


    #-----------------------------------------------------------------------------------
    #-----------------------------------PRECIPITATION-----------------------------------
    #-----------------------------------------------------------------------------------

    # Sum precipitation over the selected period
    total = precip_dataset.reduce(ee.Reducer.sum())
    
    # Compute mean precipitation within the given region
    stats = total.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=5000
    )
    
    # Filter the dataset for each season
    dry_season = filter_by_season(precip_dataset, dry_season_1stmonth, dry_season_lastmonth).filterDate(start_date, end_date)
    wet_season = filter_by_season(precip_dataset, wet_season_1stmonth, wet_season_lastmonth).filterDate(start_date, end_date)
    
    # Compute total precipitation sum for dry season
    total_dry = dry_season.reduce(ee.Reducer.sum()).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=5000
    )
    
    # Compute total precipitation sum for wet season
    total_wet = wet_season.reduce(ee.Reducer.sum()).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=5000
    )
    
    # Extract the precipitation sum values
    dry_precip_value = total_dry.getInfo().get('precipitation_sum')
    wet_precip_value = total_wet.getInfo().get('precipitation_sum')
    total_precipitation = dry_precip_value + wet_precip_value

    # Define the time range
    start_year_prec = 1994
    end_year_prec = 2024
    
    # Load CHIRPS precipitation data
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD')
    
    # Initialize a list to store the precipitation values for each year
    annual_precipitation = []
    
    # Loop through each year in the range
    for year in range(start_year_prec, end_year_prec + 1):
        # Filter CHIRPS data for the specific year
        chirps_year = chirps.filter(ee.Filter.calendarRange(year, year, 'year'))
    
        # Sum precipitation over the selected year
        total_year = chirps_year.reduce(ee.Reducer.sum())
    
        # Compute mean precipitation for the year within the given region
        stats_year = total_year.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5000
        )
    
        # Extract the total precipitation for the year
        yearly_precip = stats_year.get('precipitation_sum')
    
        # Append the result to the list
        annual_precipitation.append((year, yearly_precip.getInfo()))
    
    # Print out the results
    #for year, precip in annual_precipitation:
        #print(f"Year {year}: Total Precipitation = {precip:.2f} mm")
    
    # Extract years and precipitation values for plotting
    years = [year for year, _ in annual_precipitation]
    precipitation = [precip for _, precip in annual_precipitation]
    
    # Calculate mean annual precipitation
    mean_precipitation = np.mean(precipitation)

    # Load the ERA5 Monthly dataset for temperature and precipitation
    tempDataset = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
                    .filterDate('1994-01-01', '2024-12-31') \
                    .select('temperature_2m')  # Temperature dataset
    
    precipDataset = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
                    .filterDate('1994-01-01', '2024-12-31') \
                    .select('total_precipitation_sum')  # Precipitation dataset
    
    tempWithMonth = tempDataset.map(add_month_year)
    precipWithMonth = precipDataset.map(add_month_year)
    
    # Create a list of months (1 to 12) and calculate the monthly means
    monthlyMeans = ee.List.sequence(1, 12).map(calculate_monthly_means)
    
    # Convert the list of features to a FeatureCollection
    monthlyMeansFC = ee.FeatureCollection(monthlyMeans)


    
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

    spei_avg_dict = spei_avg.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,  
        scale=55660,
        maxPixels=1e9
    )
    
    # get the results as a dictionary so we can convert them into a df
    spei_avg_values = spei_avg_dict.getInfo()
    # convert to a df
    df = pd.DataFrame([spei_avg_values])
    
    # rename columns
    df.rename(columns={
        'SPEI_03_month_mean': 'SPEI_03',
        'SPEI_06_month_mean': 'SPEI_06',
        'SPEI_09_month_mean': 'SPEI_09',
        'SPEI_12_month_mean': 'SPEI_12'
    }, inplace=True)
    
    print(df)
    
    if df["SPEI_09"].isnull().all():
      buffered_region = region.geometry().buffer(10000)
      # Compute drought risk based on SPEI-9
      chart_data = spei_dataset.map(lambda image:
          ee.Feature(None, {
              "Date": image.get("system:time_start"),
              "SPEI_09_month": image.select("SPEI_09_month").reduceRegion(
                  reducer=ee.Reducer.mean(),
                  geometry=buffered_region,
                  scale=55660,
                  maxPixels=1e9
              ).get("SPEI_09_month")
          })
      )
    else:
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

    wf_startDate = "2000-01-01"
    wf_endDate = "2024-12-31"
    startDate = ee.Date(wf_startDate)
    endDate = ee.Date(wf_endDate)

    #retrive lat and long to get the adequate CRS for a correct area calculation
    latitude, longitude = get_shapefile_centroid(gdf)
    #print(f"Central Point: ({latitude}, {longitude})")
    best_epsg = get_best_crs(latitude, longitude)
    #print(best_epsg)
    #calculate total area
    gdf_crs = gdf.to_crs(best_epsg)
    total_area_ha = (gdf_crs['geometry'].area/10000).sum()

    # buffered area
    gdf_buffered = gdf_crs.buffer(10000) # 10km buffer
    total_area_buffered_ha = (gdf_buffered.area/10000).sum()
    gdf_buffered_df = gpd.GeoDataFrame(geometry=gdf_buffered, crs=best_epsg)
    region = geemap.geopandas_to_ee(gdf_buffered_df)

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
    df_wf['burned_area_percentage'] = (df_wf['burned_area_ha'] / total_area_buffered_ha) * 100

    # Calculate mean and standard deviation for area burned in hectares
    #mean_area = df['burned_area_ha'].mean()
    #std_area = df['burned_area_ha'].std()
    mean_area_percentage = df_wf['burned_area_percentage'].mean()
    #std_area_percentage = df_wf['burned_area_percentage'].std()

    # Step 1: Identify big fire years
    df_wf['is_big_fire_year'] = df_wf['burned_area_percentage'] > 10 #(mean_area_percentage + std_area_percentage)
    df_wf['is_medium_fire_year'] = (df_wf['burned_area_percentage'] > 5) & (df_wf['burned_area_percentage'] <= 10)
    # Step 2: Calculate frequency of big fire years
    big_fire_frequency = df_wf['is_big_fire_year'].mean() * 100  # Frequency in percentage
    nr_big_fire_years = df_wf['is_big_fire_year'].sum()
    nr_medium_fire_years = df_wf['is_medium_fire_year'].sum()
    
    # Step 3: Classify fire risk
    if mean_area_percentage < 2:
        if nr_big_fire_years >= 1:
            risk_level_wf = "High risk"
        if nr_medium_fire_years >= 1:
            risk_level_wf = "Medium risk"
        else:
            risk_level_wf = "Low risk"
    elif 2 <= mean_area_percentage <= 5:
        if nr_big_fire_years >= 1:
            risk_level_wf = "High risk"
        if nr_medium_fire_years >= 1:
            risk_level_wf = "Medium risk"
        else:
            risk_level_wf = "Medium risk"
    else:
        risk_level_wf = "High risk"

    # Convert BurnDate values into a binary burn occurrence (1 for burned, 0 otherwise)
    burned_binary = sst.map(lambda img: img.gt(0).unmask(0))
    
    # Sum the occurrences of burned areas over time
    burn_count = burned_binary.sum().clip(region)
    
    # Get the statistics of burn_count over the region
    burn_stats = burn_count.reduceRegion(
        reducer=ee.Reducer.minMax().combine(
            ee.Reducer.mean(), sharedInputs=True
        ).combine(
            ee.Reducer.stdDev(), sharedInputs=True
        ),
        geometry=region,
        scale=500,  # MODIS resolution is ~500m
        bestEffort=True
    )
    
    
    burnDate_max = burn_stats.get('BurnDate_max')
    burnDate_min = burn_stats.get('BurnDate_min')
    
    gdf = gpd.read_file(shp_file).to_crs('EPSG:4326')
    region_o = geemap.geopandas_to_ee(gdf)
    
    # Retrieve the actual min and max values as Python numbers
    burnDate_min_val = burnDate_min.getInfo()
    burnDate_max_val = burnDate_max.getInfo()
    
    # Define visualization parameters using actual numeric values
    vis_params_wf = {
        'min': burnDate_min_val,
        'max': burnDate_max_val,
        'palette': ['white', '#FFFFB2', '#FECC5C', '#FD8D3C', '#E31A1C']  # YlOrRd_04 color palette
    }


    
    return monthlyMeansFC, region, region_o, elevation, vis_params, elevation_mean_value, elevation_min_value, elevation_max_value, slope_mode, slope_mean, slope_min, slope_max, df_elevation, risk_level_erosion, avg_temp, min_temp_value, max_temp_value, hot_days_count_24, average, risk_level_thermal, total_precipitation, wet_precip_value, dry_precip_value, mean_precipitation, precipitation, years, total_floods, risk_level_f, percentage_drought, chart_list, chart_data, risk_level, mean_area_percentage, big_fire_frequency, risk_level_wf, df_wf, vis_params_wf, burnDate_min_val, burnDate_max_val, burn_count

# Streamlit app
st.title("Non-permanence Natural Risks")

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
            dry_season_1stmonth = st.selectbox("First month of dry season", options=list(range(1,13)), index=10)
            dry_season_lastmonth = st.selectbox("Last month of dry season",  options=list(range(1,13)), index=4)
            wet_season_1stmonth = st.selectbox("First month of wet season", options=list(range(1,13)), index=5)
            wet_season_lastmonth = st.selectbox("Last month of wet season", options=list(range(1,13)), index=9)
            #wf_startDate = st.text_input("Enter the start date for wildfires (format: yyyy-mm-dd)", "2000-01-01")
            #wf_endDate = st.text_input("Enter the start date for wildfires (format: yyyy-mm-dd)", "2024-12-31")


            if st.button("Process"):
                # Process the files
                monthlyMeansFC, region, region_o, elevation, vis_params, elevation_mean_value, elevation_min_value, elevation_max_value, slope_mode, slope_mean, slope_min, slope_max, df_elevation, risk_level_erosion, avg_temp, min_temp_value, max_temp_value, hot_days_count_24, average, risk_level_thermal, total_precipitation, wet_precip_value, dry_precip_value, mean_precipitation, precipitation, years, total_floods, risk_level_f, percentage_drought, chart_list, chart_data, risk_level, mean_area_percentage, big_fire_frequency, risk_level_wf, df_wf, vis_params_wf, burnDate_min_val, burnDate_max_val, burn_count = process_files(
                    shp_file, start_date, end_date, dry_season_1stmonth, dry_season_lastmonth, wet_season_1stmonth, wet_season_lastmonth
                )

                # Display results
                st.header(f'Risk Classification in {project_area_name}, {country}')
                
                # Display temperature
                st.subheader('Temperature 2024')
                st.write(f'Average Annual Temperature: {avg_temp:.2f} °C')
                st.write(f'Minimum Annual Temperature: {min_temp_value:.2f} °C')
                st.write(f'Maximum Annual Temperature: {max_temp_value:.2f} °C')

                # Display Thermal stress
                st.subheader("Thermal stress")
                st.write("Number of days with temperatures above 35C in 2024:", str(int(hot_days_count_24.get('temperature_2m_max').getInfo())))
                st.write(f"Number of days with temperatures above 35C in 2050: {average:.0f}")
                st.write(f'**Thermal Stress Risk Level:** {risk_level_thermal}')

                # Display precipitation
                st.subheader("Precipitation 2024")
                st.write(f'Cumulative Annual Precipitation: {total_precipitation:.2f} mm')
                st.write(f'Wet Season Cumulative Precipitation: {wet_precip_value:.2f} mm')
                st.write(f'Dry Season Cumulative Precipitation: {dry_precip_value:.2f} mm')
                st.write(f"Mean Annual Precipitation (1994-2024): {mean_precipitation:.2f} mm")
                
                # Create the combo chart (Bar + Line)
                # Extract the monthly data from the FeatureCollection
                features = monthlyMeansFC.getInfo()['features']
                months = [feature['properties']['month'] for feature in features]
                temperatures = [feature['properties']['mean_temperature_celsius'] for feature in features]
                precipitation = [feature['properties']['mean_precipitation_mm'] for feature in features]
                
                # Plot the data
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                # Plot precipitation as bars
                ax1.bar(months, precipitation, color='#4b8292', width=0.4, label='Precipitation (mm)', align='center')
                ax1.set_xlabel('Month')
                ax1.set_ylabel('Precipitation (mm)', color='#4b8292')
                ax1.tick_params(axis='y', labelcolor='#4b8292')
                
                # Set the y-axis limit for precipitation (0 to 200 mm)
                ax1.set_ylim(0, 200)
                
                # Create a second y-axis for the temperature
                ax2 = ax1.twinx()
                ax2.plot(months, temperatures, color='#E77577', label='Temperature (°C)', marker='o', linestyle='-', linewidth=2)
                ax2.set_ylabel('Temperature (°C)', color='#E77577')
                ax2.tick_params(axis='y', labelcolor='#E77577')
                
                # Set the y-axis limit for temperature (0 to 30°C)
                ax2.set_ylim(0, 30)
                
                # Set chart title
                #plt.title('Average Monthly Precipitation and Temperature (1994-2024)')
                
                # Show the plot
                fig.tight_layout()
                
                # Show plot in Streamlit
                st.pyplot(fig)

                
                #Display Elevation and slope
                st.subheader("Elevation and Slope")
                st.write(f"Mean Elevation: {elevation_mean_value:.0f} m")
                st.write(f"Min Elevation: {elevation_min_value:.0f} m")
                st.write(f"Max Elevation: {elevation_max_value:.0f} m")
                
                st.write(f"Most Common Slope: {slope_mode:.2f}%")
                st.write(f"Mean Slope: {slope_mean:.2f}%")
                st.write(f"Minimum Slope: {slope_min:.2f}%")
                st.write(f"Maximum Slope: {slope_max:.2f}%")
                st.write(f"**Erosion Risk Level: {risk_level_erosion}**")
                st.dataframe(df_elevation)
                # Create the map
                Map = geemap.Map()
                Map.addLayer(elevation, vis_params, "Elevation Map")
                
                # Style and add the region layer
                fc = region.style(fillColor='00000000')  # Transparent fill
                Map.addLayer(fc, {}, "Transparent Vector")
                
                # Center on the region
                Map.centerObject(region, 11)
                
                # Add colorbar
                Map.add_colorbar(
                    vis_params=vis_params,
                    label="Elevation (m)",
                    orientation="horizontal"
                )
                
                # Display the map in Streamlit
                Map.to_streamlit(height=600) 
                
                # Display the flood risks
                total_floods_int = int(total_floods)
                st.subheader("Floods 2000-2018")
                st.write(f'**Flood Risk Level: {risk_level_f}**')

                if total_floods > 0:
                    st.write('Floods detected in the project area')
                else:
                    st.write('No floods detected in the project area')
                
                st.write('Number of total flood events:', total_floods_int)

                # Display the drought risks
                st.subheader("Drought 1992-2022")
                st.write(f"Months with severe drought: {percentage_drought:.2f} %")
                st.write("**Drought Risk Level:**", risk_level.getInfo())

                # Convert chart_data to a list
                chart_list = chart_data.aggregate_array("Date").getInfo()
                spei_values = chart_data.aggregate_array("SPEI_09_month").getInfo()
                
                # Convert dates from milliseconds to datetime format
                dates = pd.to_datetime(chart_list, unit='ms')
                
                # Create DataFrame
                df = pd.DataFrame({"Date": dates, "SPEI_09_month": spei_values})
                # Ensure 'Date' is in datetime format
                df['Date'] = pd.to_datetime(df['Date'])
                count_drought = (df['SPEI_09_month'] < -1.5).sum()
                print(count_drought)
                #print(df)
                
               # Plot SPEI-9 over time with better handling for many bars
                fig, ax = plt.subplots(figsize=(16, 8))  # <== Use fig, ax instead of plt.figure()
                
                # Create bars
                bars = ax.bar(df['Date'], df["SPEI_09_month"],
                              width=20,
                              color=np.where(df["SPEI_09_month"] < -1.5, 'red', 'b'),
                              label="SPEI-9 Month Avg")
                
                threshold = -1.5
                
                # Formatting
                ax.axhline(y=0, color='black', linewidth=1)
                ax.axhline(y=threshold, color='red', linestyle='--', label='Severe Drought Threshold (-1.5)')
                ax.set_xlabel("Year")
                ax.set_ylabel("SPEI-9 Index")
                ax.set_title("Monthly SPEI-9 Over Time (1992-2021)")
                
                # Improve x-axis ticks
                years = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='YS')
                ax.set_xticks(years)
                ax.set_xticklabels([y.strftime('%Y') for y in years], rotation=45)
                
                ax.legend()
                ax.grid(False)
                
                # Adjust layout
                fig.tight_layout()
                
                # Show plot in Streamlit
                st.pyplot(fig)  # <== THIS is the key

                #Display the wildfire risks
                st.subheader("Wildfires 2000-2024")
                st.write(f'Mean of burned area: {mean_area_percentage:.2f} %')
                st.write(f'Frequency of big fire years: {big_fire_frequency:.2f} %')
                st.write(f'**Wildfire risk Level: {risk_level_wf}**')

                # Define risk thresholds
                low_risk_threshold = 2  # Low risk threshold (10%)
                high_risk_threshold = 10  # High risk threshold (30%)
                mean_threshold = df_wf['burned_area_percentage'].mean()  # Mean burned area percentage

                # Add a column for point size (optional, for visualization)
                df_wf['point_size'] = df_wf['burned_area_percentage'] * 10  # Scale size for better visualization

                # Streamlit app layout
                st.subheader("Burned Area Percentage Analysis")
                #st.write("Scatter plot showing annual burned area percentage with risk thresholds.")

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
                ax.axhline(y=low_risk_threshold, color='green', linestyle='--', label='Low Risk Threshold (2%)')
                ax.axhline(y=high_risk_threshold, color='red', linestyle='--', label='High Risk Threshold (10%)')
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

                # Create an interactive map
                Map = geemap.Map()
                
                # Center the map around your region
                Map.centerObject(region, 11)
                # Add the burn frequency layer
                Map.addLayer(burn_count, vis_params, "Burn Frequency")
                fc = region.style(fillColor='00000000')  # Transparent fill for 'region'
                Map.addLayer(fc, {}, "Transparent Region Boundary")
                #region original
                fc_o = region_o.style(fillColor='00000000')  # Transparent fill for 'region_o'
                Map.addLayer(fc_o, {}, "Transparent Region O Boundary")
                Map.add_colorbar(
                    vis_params=vis_params_wf,
                    label="Fire Frequency (years)",
                    orientation="horizontal"
                )
                # Display the map
                Map.to_streamlit(height=600) 

        else:
            st.error("No shapefile found in the uploaded zip file.")

    



