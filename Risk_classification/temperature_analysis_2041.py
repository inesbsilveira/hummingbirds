// Load your shapefile from assets
var region = ee.FeatureCollection(table);

// Function to remove duplicates based on the date within each collection
var remove_duplicates = function(collection) {
  return collection
    .map(function(image) {
      var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');  // Get the date from each image
      return image.set('date', date);  // Add date as a property
    })
    .distinct('date');  // Remove duplicates based on the 'date' property
};

// Dataset for each model
var dataset_gfdl = ee.ImageCollection('NASA/GDDP-CMIP6')
    .filter(ee.Filter.date('2041-01-01', '2041-12-31'))
    .filter(ee.Filter.eq('model', 'GFDL-ESM4'));
var max_air_temp_gfdl = remove_duplicates(dataset_gfdl.select('tasmax'));
var min_air_temp_gfdl = remove_duplicates(dataset_gfdl.select('tasmin'));
var avg_air_temp_gfdl = remove_duplicates(dataset_gfdl.select('tas'));

var dataset_ipsl = ee.ImageCollection('NASA/GDDP-CMIP6')
    .filter(ee.Filter.date('2041-01-01', '2041-12-31'))
    .filter(ee.Filter.eq('model', 'IPSL-CM6A-LR'));
var max_air_temp_ipsl = remove_duplicates(dataset_ipsl.select('tasmax'));
var min_air_temp_ipsl = remove_duplicates(dataset_ipsl.select('tasmin'));
var avg_air_temp_ipsl = remove_duplicates(dataset_ipsl.select('tas'));

var dataset_mpi = ee.ImageCollection('NASA/GDDP-CMIP6')
    .filter(ee.Filter.date('2041-01-01', '2041-12-31'))
    .filter(ee.Filter.eq('model', 'MPI-ESM1-2-HR'));
var max_air_temp_mpi = remove_duplicates(dataset_mpi.select('tasmax'));
var min_air_temp_mpi = remove_duplicates(dataset_mpi.select('tasmin'));
var avg_air_temp_mpi = remove_duplicates(dataset_mpi.select('tas'));

var dataset_mri = ee.ImageCollection('NASA/GDDP-CMIP6')
    .filter(ee.Filter.date('2041-01-01', '2041-12-31'))
    .filter(ee.Filter.eq('model', 'MRI-ESM2-0'));
var max_air_temp_mri = remove_duplicates(dataset_mri.select('tasmax'));
var min_air_temp_mri = remove_duplicates(dataset_mri.select('tasmin'));
var avg_air_temp_mri = remove_duplicates(dataset_mri.select('tas'));

var dataset_ukesm = ee.ImageCollection('NASA/GDDP-CMIP6')
    .filter(ee.Filter.date('2041-01-01', '2041-12-31'))
    .filter(ee.Filter.eq('model', 'UKESM1-0-LL'));
var max_air_temp_ukesm = remove_duplicates(dataset_ukesm.select('tasmax'));
var min_air_temp_ukesm = remove_duplicates(dataset_ukesm.select('tasmin'));
var avg_air_temp_ukesm = remove_duplicates(dataset_ukesm.select('tas'));

print(max_air_temp_gfdl.size());

// Get the list of images from each collection
var list_gfdl_max = max_air_temp_gfdl.toList(max_air_temp_gfdl.size());
var list_ipsl_max = max_air_temp_ipsl.toList(max_air_temp_ipsl.size());
var list_mpi_max = max_air_temp_mpi.toList(max_air_temp_mpi.size());
var list_mri_max = max_air_temp_mri.toList(max_air_temp_mri.size());
var list_ukesm_max = max_air_temp_ukesm.toList(max_air_temp_ukesm.size());
var list_gfdl_min = min_air_temp_gfdl.toList(min_air_temp_gfdl.size());
var list_ipsl_min = min_air_temp_ipsl.toList(min_air_temp_ipsl.size());
var list_mpi_min = min_air_temp_mpi.toList(min_air_temp_mpi.size());
var list_mri_min = min_air_temp_mri.toList(min_air_temp_mri.size());
var list_ukesm_min = min_air_temp_ukesm.toList(min_air_temp_ukesm.size());
var list_gfdl_avg = avg_air_temp_gfdl.toList(avg_air_temp_gfdl.size());
var list_ipsl_avg = avg_air_temp_ipsl.toList(avg_air_temp_ipsl.size());
var list_mpi_avg = avg_air_temp_mpi.toList(avg_air_temp_mpi.size());
var list_mri_avg = avg_air_temp_mri.toList(avg_air_temp_mri.size());
var list_ukesm_avg = avg_air_temp_ukesm.toList(avg_air_temp_ukesm.size());

// Define the number of images (728)
var num_images_max = max_air_temp_gfdl.size();
var num_images_min = min_air_temp_gfdl.size();
var num_images_avg = avg_air_temp_gfdl.size();

// Generate a sequence of indices from 0 to 727
var indices_max = ee.List.sequence(0, num_images_max.subtract(1));
var indices_min = ee.List.sequence(0, num_images_min.subtract(1));
var indices_avg = ee.List.sequence(0, num_images_avg.subtract(1));

// Function to compute the average image at a given index
var mean_image_list_max = indices_max.map(function(i) {
  var img_gfdl_max = ee.Image(list_gfdl_max.get(i));
  var img_ipsl_max = ee.Image(list_ipsl_max.get(i));
  var img_mpi_max = ee.Image(list_mpi_max.get(i));
  var img_mri_max = ee.Image(list_mri_max.get(i));
  var img_ukesm_max = ee.Image(list_ukesm_max.get(i));

  // Compute the mean across the five images
  var mean_img_max = img_gfdl_max.add(img_ipsl_max).add(img_mpi_max).add(img_mri_max).add(img_ukesm_max).divide(5);

  // Copy metadata from one of the images (date, projection, etc.)
  return mean_img_max.set('system:time_start', img_gfdl_max.get('system:time_start'));
});

// Function to compute the average image at a given index
var mean_image_list_min = indices_min.map(function(i) {
  var img_gfdl_min = ee.Image(list_gfdl_min.get(i));
  var img_ipsl_min = ee.Image(list_ipsl_min.get(i));
  var img_mpi_min = ee.Image(list_mpi_min.get(i));
  var img_mri_min = ee.Image(list_mri_min.get(i));
  var img_ukesm_min = ee.Image(list_ukesm_min.get(i));

  // Compute the mean across the five images
  var mean_img_min = img_gfdl_min.add(img_ipsl_min).add(img_mpi_min).add(img_mri_min).add(img_ukesm_min).divide(5);

  // Copy metadata from one of the images (date, projection, etc.)
  return mean_img_min.set('system:time_start', img_gfdl_min.get('system:time_start'));
});

// Function to compute the average image at a given index
var mean_image_list_avg = indices_avg.map(function(i) {
  var img_gfdl_avg = ee.Image(list_gfdl_avg.get(i));
  var img_ipsl_avg = ee.Image(list_ipsl_avg.get(i));
  var img_mpi_avg = ee.Image(list_mpi_avg.get(i));
  var img_mri_avg = ee.Image(list_mri_avg.get(i));
  var img_ukesm_avg = ee.Image(list_ukesm_avg.get(i));

  // Compute the mean across the five images
  var mean_img_avg = img_gfdl_avg.add(img_ipsl_avg).add(img_mpi_avg).add(img_mri_avg).add(img_ukesm_avg).divide(5);

  // Copy metadata from one of the images (date, projection, etc.)
  return mean_img_avg.set('system:time_start', img_gfdl_avg.get('system:time_start'));
});

// Convert the list of images into an ImageCollection
var dataset_max = ee.ImageCollection(mean_image_list_max);
var dataset_min = ee.ImageCollection(mean_image_list_min);
var dataset_avg = ee.ImageCollection(mean_image_list_avg);

// Calculate the mean, min, and max annual temperature
var meanTemp = dataset_avg.mean().clip(region);
var minTemp = dataset_min.min().clip(region);
var maxTemp = dataset_max.max().clip(region);

// Convert temperature from Kelvin to Celsius
var meanTempCelsius = meanTemp.subtract(273.15);
var minTempCelsius = minTemp.subtract(273.15);
var maxTempCelsius = maxTemp.subtract(273.15);

// Calculate the average, min, and max temperature over the region
var tempStats = meanTempCelsius.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: region.geometry(),
  scale: 1000,
  bestEffort: true
});

var minStats = minTempCelsius.reduceRegion({
  reducer: ee.Reducer.min(),
  geometry: region.geometry(),
  scale: 1000,
  bestEffort: true
});

var maxStats = maxTempCelsius.reduceRegion({
  reducer: ee.Reducer.max(),
  geometry: region.geometry(),
  scale: 1000,
  bestEffort: true
});

// Extract the temperature values and print them as numbers
var avgTemp = tempStats.get('tas');
var minTemp = minStats.get('tasmin');  
var maxTemp = maxStats.get('tasmax');  

// Print the results to the console as numbers
print('Average Annual Temperature (°C):', avgTemp.getInfo());
print('Minimum Annual Temperature (°C):', minTemp.getInfo());
print('Maximum Annual Temperature (°C):', maxTemp.getInfo());

var threshold = 32; // Threshold temperature in Celsius
// Convert to Kelvin (if needed, depends on dataset)
var thresholdK = threshold + 273.15; 
// Apply threshold to the averaged collection
var colThreshold = dataset_max.map(function (image) {
  var thresholdImage = image.gt(thresholdK); // Identify pixels above 32°C
  return thresholdImage.set('system:time_start', image.get('system:time_start'));
});

// Sum the number of days where at least one pixel exceeded the threshold
var countDaysAbove32 = colThreshold.map(function(image) {
  // Reduce over the region: if any pixel is above threshold, return 1 for that day
  var mask = image.reduceRegion({
    reducer: ee.Reducer.anyNonZero(), 
    geometry: region,
    scale: 5000, // Adjust scale based on data resolution
    bestEffort: true
  });

  // Get the result and assign 1 if true, otherwise 0
  var isAbove = ee.Algorithms.If(mask.get('tasmax'), 1, 0);

  return ee.Feature(null, {
    'date': image.get('system:time_start'),
    'day_above_32': isAbove
  });
});

// Convert to FeatureCollection for analysis
var daysAbove32FC = ee.FeatureCollection(countDaysAbove32);

// Sum the number of days where at least one pixel was above 32°C
var totalDaysAbove32 = daysAbove32FC.aggregate_sum('day_above_32');

print('Total number of days with at least one pixel above 32°C:', totalDaysAbove32);

// Classify risk based on the average yearly extreme heat days
var riskLevel = ee.Algorithms.If(
  totalDaysAbove32.lt(90), 'Low risk',
  ee.Algorithms.If(totalDaysAbove32.lte(180), 'Medium risk', 'High risk')
);

print('Risk Level:', riskLevel);
