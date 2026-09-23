// ---- 1. Load and name the bands -------------------------------------------
// Ingestion usually calls them b1 and b2 regardless of what the GeoTIFF says,
// so rename is not optional. Run print(ee.Image(asset).bandNames()) once if you
// want to see what you actually got.
var asset = 'projects/ee-<you>/assets/val_predictions';
var img = ee.Image(asset).rename(['label', 'prob']);

// ---- 2. Decode -------------------------------------------------------------
// prob was quantized into 1..255 with 0 reserved for nodata, so undo that.
// Non wetland pixels are already masked by the --nodata_value=0 at ingestion.
var prob  = img.select('prob').subtract(1).divide(254);
var truth = img.select('label').eq(2);      // band value 2 = converted to developed

// ---- 3. Confusion classes at one operating point ---------------------------
var T = 0.5;                                 // the only number you change
var pred = prob.gte(T);
var cls  = pred.multiply(2).add(truth);      // 0 TN, 1 FN, 2 FP, 3 TP

// ---- 4. Draw ---------------------------------------------------------------
// Drop true negatives. They are 99.4% of the pixels and would bury everything.
var outcomes = cls.updateMask(cls.gt(0));

Map.addLayer(outcomes, {
  min: 1, max: 3,
  palette: ['4477AA',    // 1  false negative, a real conversion the model missed
            'EE6677',    // 2  false positive
            '228833']    // 3  true positive
}, 'outcomes @ p >= ' + T);

Map.addLayer(prob, {min: 0, max: 1,
  palette: ['000004', '51127c', 'b63679', 'fb8861', 'fcfdbf']},
  'predicted probability', false);

Map.setCenter(-81.5, 27.8, 7);


// For statewide view
// counts per sq km instead of individual pixels. 
// var density = cls.eq(2).unmask(0).rename('fp')
//   .addBands(cls.eq(1).unmask(0).rename('fn'))
//   .reduceResolution({reducer: ee.Reducer.sum().unweighted(), maxPixels: 4096})
//   .reproject({crs: 'EPSG:5070', scale: 1000});

// Map.addLayer(density.select('fp'), {min: 0, max: 50,
//   palette: ['ffffcc', 'fd8d3c', 'bd0026']}, 'false positives per km2', false);