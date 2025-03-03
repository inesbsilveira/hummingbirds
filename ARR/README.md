**README FILE to run the ARR_eligibility notebook**\
Updated on March 2025

### Requirements:
- Google Earth Engine (GEE) account
- 1 shapefile OR 1 geojson
- Countries_Forest_Definition.xlsx (if you don't have this file, the code WILL NOT run). Find the file here ##addpath##

### Workflow

1. Imports - Install and import the necessary libraries
2. Connect to your GEE 
3. Input file - Upload the shp or geojson into google Colab and change the name of the 'input_shp' or 'input_geojson', accordingly. This will read only one file
4. Variables - Change the variables **THIS IS THE ONLY STEP IN THE CODE WHERE YOU WILL MODIFY ANYTHING!!**
   - 'country': Important to later retrieve the forest definition of each country
   - 'project_area_name'= this serves only for the filenames created
   - 'year_0' and 'year_10': 2014 and 2024 (10 years apart) DO NOT CHANGE THIS
   - 'start_date' and 'end_date': mm-dd format. change this to for the dry season of the country
   - 'year_0_2020' and 'year_10_2020': should be 2020 or late 2019
   - 'start_date_2020' and 'end_date_2020': mm-dd format. change this to for the dry season of the country
   - 'slope': change if necessary. unit is percentage
   - 'min_forest_pixels_list': This will return the results of eligible area for forest with minimum patches of 1ha, 5ha and 10ha. Change if you want different sizes but keep in mind that 1ha=11pixels
6. Functions and Legends - here are stores all the functions and legends (colors of maps, etc) that the code needs to run. This step is absolutely necessary. **DO NOT CHANGE ANY FUNCTION**
7. Main
8. Results
