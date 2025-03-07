### Natural Risk Classification for Project areas
Understand the historical data on floods, drought and wildfires. Analyse estimated future climate data

### Requirements
- Google Earth Engine account
- 1 shapefile of the project area

### Content
1. drought_risk_classification (GEE)
2. extreme_heat_tasmax (GEE)
3. flood_risk_classification (GEE)
4. precipitation_analysis (GEE)
5. temperature_analysis_2024 (GEE)
6. temperature_analysis_2041 (GEE)
7. wildfire_risk_classification_v2 (Google Colab)

### Workflow for Google Earth Engine
1. add project area shapefile to the Assets in GEE
2. add the asset to the script
3. run the code

### Workflow for Google Colab
1. Imports - Install and import the necessary libraries
2. Connect to your GEE - add your own GEE project at `my_project`
3. Input file and Variables - Upload the shp into google Colab and change the name of the `input_shp`. Change the variables **THIS IS THE ONLY STEP IN THE CODE WHERE YOU WILL MODIFY ANYTHING**
   - `country`: this serves only for the chart created
   - `project_area_name`= this serves only for the chart created
   - `start_date`: yyyy-mm-dd format
   - `end_date`: yyyy-mm-dd format
6. Functions - here are stored all the functions that the code needs to run. This step is absolutely necessary. **DO NOT CHANGE ANY FUNCTION**
7. Main - from the burned areas dataset, we retrieve the burned area per each year, in hectares and in percentage. Calculate the mean and the standard deviation
8. Results - The results will identify the percentage of years where the burned area was above the threshold and retrieve the Fire Risk Classification

**Example for Calao for Wildfires**
   ```
input_shp = "original_calao_shp.shp"
country = 'Ivory Coast'
project_area_name = 'Calao' #region/country/project name
start_date = '2000-01-01'
end_date = '2024-12-31'
```

Last updated on March 2025
