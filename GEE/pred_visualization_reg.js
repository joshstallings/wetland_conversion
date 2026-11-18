// Out of fold validation predictions from the hurdle regressor: the share of a
// cell's 2019 wetland developed by 2024. Only cells where some wetland did
// develop are in the image, since that is the population the regressor was
// trained and scored on. Made by GEE/export_cell_predictions.py.

var asset = 'projects/<project>/assets/hurdle_reg_mlp_300m_log_share';

// pred_median is exp(model output), the per cell typical estimate. pred_mean
// multiplies in the fold's smear factor (about 1.58), which gets the average
// right but pushes most single cells over: 67% over predicted versus 50% for
// the median. For a per cell under or over map the median is the fair one.
var ESTIMATE = 'pred_median';

// The error layer is log2(predicted / actual), so 1 is 2x over, minus 1 is 2x
// under. Colors saturate at plus or minus CLIP.
var CLIP = 3;

// Ingestion often names bands b1 to b4 whatever the tif says, so rename. The
// -9999 nodata is masked by --nodata_value at ingestion.
var img    = ee.Image(asset).rename(['actual', 'pred_median', 'pred_mean', 'fold']);
var actual = img.select('actual');
var est    = img.select(ESTIMATE);

var logRatio = est.divide(actual).log().divide(Math.LN2).clamp(-CLIP, CLIP);

// Blue under predicted, white about right, red over predicted.
var DIVERGING = ['2166AC', '67A9CF', 'D1E5F0', 'F7F7F7', 'FDDBC7', 'EF8A62', 'B2182B'];

// Shares run from 0.01 to 1 and most sit under 0.33, so show them on a log10
// scale, minus 2 (1%) to 0 (100%).
var SHARE_VIS = {min: -2, max: 0, palette: ['440154', '3b528b', '21918c', '5ec962', 'fde725']};
Map.addLayer(actual.log10(), SHARE_VIS, 'actual share (log10)', false);
Map.addLayer(est.log10(), SHARE_VIS, ESTIMATE + ' share (log10)', false);

// Two colors only, no magnitude: over or under.
Map.addLayer(logRatio.gt(0), {min: 0, max: 1, palette: ['2166AC', 'B2182B']},
  'under (blue) or over (red)', false);

Map.addLayer(logRatio, {min: -CLIP, max: CLIP, palette: DIVERGING},
  'log2(' + ESTIMATE + ' / actual)');

Map.setCenter(-81.5, 27.8, 7);
