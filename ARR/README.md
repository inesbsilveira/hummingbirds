### Eligibility analysis for ARR projects 
From a shapefile or geojson

### Requirements:
- Google Earth Engine (GEE) account
- 1 shapefile OR 1 geojson
- Countries_Forest_Definition.xlsx (if you don't have this file, the code WILL NOT run). Find the file [here](https://hummingbirdsnbs.sharepoint.com/:x:/s/hummingbirds-GnralpartagTTA/EZROObh4yJFHglX48JPVsTUByDptRm9QpymrkZs-sg09yw?e=CcsuKT). If the Country is not in the list, please add it and its data

### Workflow
1. Imports - Install and import the necessary libraries
2. Connect to your GEE - add your own GEE project at `my_project`
3. Input file - Upload the shp or geojson into google Colab and change the name of the `input_shp` or `input_geojson`, accordingly. This will read only one file
4. Variables - Change the variables **THIS IS THE ONLY STEP IN THE CODE WHERE YOU WILL MODIFY ANYTHING**
   - `country`: Important to later retrieve the forest definition of each country
   - `project_area_name`= this serves only for the filenames created
   - `year_0` and `year_10`: 2014 and 2024 (10 years apart) DO NOT CHANGE THIS
   - `start_date` and `end_date`: -mm-dd format. change to the dry season of the country
   - `year_0_2020` and `year_10_2020`: should be 2020 or late 2019
   - `start_date_2020` and `end_date_2020`: -mm-dd format. change to the dry season of the country
   - `slope_percentage`: change if necessary. unit is percentage
   - `min_forest_pixels_list`: This will return the results of eligible area for forest with minimum patches of 1ha, 5 ha and 10 ha. Change if you want different sizes but keep in mind that 1ha=11pixels
6. Functions and Legends - here are stored all the functions and legends (colors of maps, etc) that the code needs to run. This step is absolutely necessary. **DO NOT CHANGE ANY FUNCTION**
7. Main - in this part, the code gets the ESA World Cover for 2020 and samples random points from that dataset. These points will be used to train a Random Forest model to estimate the World Cover for 2014 and 2024
8. Results - The results will show the area calculation in the best EPSG according to each country. This will calculate the area of forest for year 0 and year 10, as well as the eligible areas for the ARR project

**Example for Calao**
   ```
country = 'Ivory Coast'
project_area_name = 'Calao'
year_0 = 2014
year_10 = 2024
start_date = '-01-01'
end_date = '03-30'
year_0_2020 = 2019
year_1_2020 = 2020
start_date_2020 = '-12-01'
end_date_2020 = '-02-01'
slope_percentage = 30
min_forest_pixels_list = [11, 55, 110]
```

Last updated on March 2025
