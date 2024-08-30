// // The javascript code is the code to download the gfc data from https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/download.html and should be run in https://code.earthengine.google.com/

// Import the Hansen et al. forest loss dataset
var gfc2020 = ee.Image("UMD/hansen/global_forest_change_2023_v1_11");

// Import a shapefile of Sarawak
var sarawak = ee.FeatureCollection("projects/ee-abdullahihussein156/assets/Sarawak");

// Clip the forest loss data to Sarawak
var sarawakForestLoss = gfc2020.clip(sarawak);

// Select the layers we're interested in
var treeCover2000 = sarawakForestLoss.select(['treecover2000']);
var lossYear = sarawakForestLoss.select(['lossyear']);

// Create a mask for areas that had >30% tree cover in 2000
var treeCoverMask = treeCover2000.gte(30);

var maskedLossYear = lossYear.updateMask(treeCoverMask);

// Define visualization parameters
var vizParams = {
    min: 0,
    max: 23,
    palette: ['green', 'red']
};

// Add the layer to the map
Map.centerObject(sarawak, 7);
Map.addLayer(maskedLossYear, vizParams, 'Forest Loss Year');

// Function to create connected components
var createConnectedComponents = function (image) {
    return image.connectedComponents({
        connectedness: ee.Kernel.plus(1),
        maxSize: 256
    });
};

// Create connected components for each year
var connectedComponents = ee.List.sequence(1, 23).map(function (year) {
    var yearNumber = ee.Number(year);
    var yearStr = yearNumber.format('%d');
    var yearMask = maskedLossYear.eq(yearNumber);
    var ccImage = createConnectedComponents(yearMask).select('labels').rename(ee.String('cc_').cat(yearStr));
    return ccImage;
});

// Combine all years into a single multi-band image
var allConnectedComponents = ee.ImageCollection.fromImages(connectedComponents).toBands();

// Rename the bands to remove the index prefix
var bandNames = ee.List.sequence(1, 23).map(function (year) {
    return ee.String('cc_').cat(ee.Number(year).format('%d'));
});

var renamedConnectedComponents = allConnectedComponents.select(allConnectedComponents.bandNames(), bandNames);

// Function to calculate area
var calculateArea = function (feature) {
    return feature.set('area_ha', feature.area().divide(10000));
};

// Function to process each year
var processYear = function (year) {
    var yearNumber = ee.Number(year);
    var yearStr = yearNumber.format('%d'); // Convert year to string for band selection
    var yearImage = renamedConnectedComponents.select(ee.String('cc_').cat(yearStr)).neq(0);
    var yearVectors = yearImage.reduceToVectors({
        geometry: sarawak,
        scale: 30,
        eightConnected: true,
        labelProperty: 'year',
        maxPixels: 1e13
    });

    var yearFeatures = yearVectors.map(calculateArea);

    return yearFeatures.filter(ee.Filter.and(
        ee.Filter.gte('area_ha', 0.18),  // Minimum area of 0.18 ha
        ee.Filter.lte('area_ha', 100)    // Maximum area of 100 ha
    ));
};

// Process all years
var allYears = ee.List.sequence(1, 23);
var allDeforestationEvents = ee.FeatureCollection(allYears.map(function (year) {
    return processYear(year);
})).flatten();

// Stratify by size
var smallEvents = allDeforestationEvents.filter(ee.Filter.lte('area_ha', 10));
var mediumEvents = allDeforestationEvents.filter(ee.Filter.and(
    ee.Filter.gt('area_ha', 10),
    ee.Filter.lte('area_ha', 100)
));
var largeEvents = allDeforestationEvents.filter(ee.Filter.gt('area_ha', 100));

// Sample events
var sampledEvents = smallEvents.randomColumn().sort('random').limit(500)
    .merge(mediumEvents.randomColumn().sort('random').limit(500))
    .merge(largeEvents.randomColumn().sort('random').limit(190));

// Add coordinates to sampled events
var eventsWithCoords = sampledEvents.map(function (feature) {
    var year = ee.Number(feature.get('year')).add(2000);
    var centroid = feature.geometry().centroid();
    return feature.set({
        'latitude': centroid.coordinates().get(1),
        'longitude': centroid.coordinates().get(0),
        'year': year,
        'example_path': ee.String('examples/').cat(ee.Number(centroid.coordinates().get(1)).format('%.8f'))
            .cat('_').cat(ee.Number(centroid.coordinates().get(0)).format('%.8f'))
    });
});

// Export the deforestation events as a CSV
Export.table.toDrive({
    collection: eventsWithCoords,
    description: 'Sarawak_Deforestation_Events',
    fileFormat: 'CSV'
});

