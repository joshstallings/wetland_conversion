// Out of fold validation predictions from a cell classifier: "did any 2019
// wetland in this cell develop by 2024". Made by GEE/export_cell_predictions.py.

// ---- 1. Pick a run ----------------------------------------------------------
// Each run needs its own per fold thresholds. The exporter prints them; these are
// what each fold picked to hit recall 0.5 on its own train cells, so the classes
// below line up with the precision and recall in fold_summary.csv.
var RUNS = {
  clf_3km: {
    asset: 'projects/<project>/assets/hurdle_clf_mlp_3km_bce',
    FOLD_T: {1: 0.7836, 2: 0.8012, 3: 0.7782, 4: 0.7923, 5: 0.7857}
  },
  clf_300m: {
    asset: 'projects/<project>/assets/hurdle_clf_mlp_300m_bce',
    FOLD_T: {1: 0.2005, 2: 0.1739, 3: 0.2196, 4: 0.1708, 5: 0.1751}
  }
};
var RUN = RUNS.clf_300m;

// Set to a number to use one threshold everywhere instead of the per fold ones.
var T_OVERRIDE = null;

// ---- 2. Which outcome classes to draw ----------------------------------------
// Each class is its own layer, so they can also be toggled in the Layers panel
// without rerunning. SHOW sets what starts visible, OPACITY sets transparency
// (0 invisible, 1 solid). TN is about 97% of cells at 300 m and buries
// everything else, so it starts off.
var SHOW    = {TP: true, FP: true, FN: true, TN: false};
var OPACITY = {TP: 1.0,  FP: 1.0,  FN: 1.0,  TN: 0.5};
var COLOR   = {TP: '228833', FP: 'EE6677', FN: '4477AA', TN: 'BBBBBB'};

// ---- 3. Load and decode -------------------------------------------------------
// Ingestion often names bands b1, b2, b3 whatever the tif says, so rename.
// Cells with no 2019 wetland are already masked by --nodata_value=0.
var img   = ee.Image(RUN.asset).rename(['label', 'prob', 'fold']);
var prob  = img.select('prob').subtract(1).divide(254);
var truth = img.select('label').eq(2);
var fold  = img.select('fold');

var thresh;
if (T_OVERRIDE !== null) {
  thresh = ee.Image.constant(T_OVERRIDE);
} else {
  thresh = ee.Image.constant(0).float();
  [1, 2, 3, 4, 5].forEach(function (k) {
    thresh = thresh.where(fold.eq(k), RUN.FOLD_T[k]);
  });
}
var pred = prob.gte(thresh);

var classes = {
  TP: pred.and(truth),
  FP: pred.and(truth.not()),
  FN: pred.not().and(truth),
  TN: pred.not().and(truth.not())
};

// ---- 4. Draw ------------------------------------------------------------------
// Probability goes underneath so the outcome layers sit on top of it.
Map.addLayer(prob, {min: 0, max: 1,
  palette: ['000004', '51127c', 'b63679', 'fb8861', 'fcfdbf']},
  'predicted probability', false);

// Drawn TN first so the rarer classes land on top.
['TN', 'FN', 'FP', 'TP'].forEach(function (name) {
  Map.addLayer(classes[name].selfMask(), {palette: [COLOR[name]]},
    name, SHOW[name], OPACITY[name]);
});

Map.setCenter(-81.5, 27.8, 7);
