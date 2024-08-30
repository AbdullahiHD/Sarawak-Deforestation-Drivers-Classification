// The javascript code is the code used to pre-process the landsat 8 imagery data and should be run in https://code.earthengine.google.com/

// Load the dataset
var dataset = ee.FeatureCollection('projects/ee-abdullahihussein156/assets/Dataset');

function getComposite(year, geom) {
    var startDate = ee.Date.fromYMD(year, 1, 1);
    var endDate = ee.Date.fromYMD(year, 12, 31);

    var collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(geom)
        .filterDate(startDate, endDate)
        // .filterMetadata('CLOUD_COVER', 'less_than', 50) 
        .filter(ee.Filter.lte('CLOUD_COVER', 50))
        .select(['SR_B4', 'SR_B3', 'SR_B2']);

    var composite = collection.median();

    // Apply more flexible visualization parameters
    var visParams = {
        min: 0,
        max: 30000,
        bands: ['SR_B4', 'SR_B3', 'SR_B2']
    };

    return composite.visualize(visParams).set('year', year);
}

function processOneEvent(feature) {
    // Get the example path and evaluate it to retrieve the actual string value
    var examplePath = feature.get('example_path');
    examplePath.evaluate(function (path) {
        var coords = path.split('_');
        var lat = parseFloat(coords[0].split('/')[1]).toFixed(8);
        var lon = parseFloat(coords[1]).toFixed(8);
        var point = ee.Geometry.Point([parseFloat(lon), parseFloat(lat)]);
        var year = ee.Number(feature.get('year'));

        var folderName = lat.concat('_').concat(lon);
        var taskNameCoords = lat.replace('.', '_') + '_' + lon.replace('.', '_');

        var region = point.buffer(5000);  // ~1128.38m radius for 5 sq km

        ee.List.sequence(1, 4).evaluate(function (years) {
            years.forEach(function (i) {
                var imageYear = year.add(i);
                var image = getComposite(imageYear, point);

                if (image) {  // Check if image is valid
                    // Convert imageYear to a client-side value
                    imageYear.evaluate(function (clientYear) {
                        Export.image.toDrive({
                            image: image,
                            description: taskNameCoords + '_' + clientYear,
                            folder: folderName, // Ensure folderName is the sub-folder you want
                            fileNamePrefix: taskNameCoords + '_' + clientYear,
                            scale: 15,
                            region: region,
                            maxPixels: 1e13,
                            fileFormat: 'GEO_TIFF'
                        });
                    });
                } else {
                    print('Skipping year:', imageYear);
                }
            });
        });

        // Visualize the image on the map
        var compositeImage = getComposite(year.add(1), region);
        if (compositeImage) {
            Map.addLayer(compositeImage, {}, 'Composite ' + year.add(1).getInfo());
            Map.centerObject(region, 12);
        } else {
            print('No composite image available for visualization.');
        }
    });
}

function processBatch(startIndex, batchSize, totalSize) {
    if (startIndex >= totalSize) {
        print('All batches processed');
        return;
    }

    var batch = dataset.toList(batchSize, startIndex);

    batch.evaluate(function (features) {
        print('Processing batch starting at index:', startIndex);
        features.forEach(function (feature) {
            processOneEvent(ee.Feature(feature));
        });

        // Scheduling the next batch
        processBatch(startIndex + batchSize, batchSize, totalSize);
    });
}

// Start processing in batches of 10
dataset.size().evaluate(function (size) {
    processBatch(20, 10, size);
});


// Print the number of features to verify the dataset is loaded correctly
print('Number of features:', dataset.size());
