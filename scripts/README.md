# scripts

Pipelines for pulling AlphaEarth Foundations embeddings for Florida, joining
them onto the NLCD wetland samples, building a versioned training dataset from
that join, and fitting gradient boosting models against it, plus the shared
config each half depends on. Everything here runs in the `wetlands` mamba env,
from the repo root (`python scripts/<name>.py ...`).

## gee_common.py

Shared config and helpers: the GEE project id, the AlphaEarth dataset id, the
NLCD CRS and 30m pixel grid definitions, and `year_image()` /
`year_native_projection()` for mosaicking one year's AlphaEarth granules over
a region. Not runnable on its own - both pipelines below import from it.

## build_florida_tile_grid_30m.py

Builds the 60km tile grid that `export_alphaearth_30m.py` submits against,
with every tile edge snapped to the real NLCD 30m pixel lattice. Run once (or
whenever `TILE_SIZE_PX_30M` in `gee_common.py` changes) it writes
`data/boundaries/florida_tiles_30m_60km.gpkg`.

## export_alphaearth_30m.py

Mean-reduces AlphaEarth from native 10m to 30m on the NLCD grid, shifts the
result to uint8, and exports one GeoTIFF per (tile, year) to
`gs://jstallings`. Staged commands (`test-direct`, `test-tile`, `submit`,
`status`, `task`) check the math and the GCS export path before committing to
the full statewide batch.

## join_alphaearth_samples.py

Joins the AlphaEarth tiles `export_alphaearth_30m.py` has pulled down onto
`data/processed/wetland_sample_labels_2019_2024.parquet`, producing one row
per wetland pixel with a 2017/2018/2019 embedding stack (`A00_2017`...`A63_2019`)
alongside the label columns. `extract-tile`/`extract-all` pull decoded band
values per (tile, year) into `data/processed/alphaearth_extract/`; `assemble`
inner-joins a tile's extracted years into
`data/processed/alphaearth_wetland_joined/{tile_id}.parquet` once all of them
are done, stamping each row with a `block_id` (~10km grid cell, absolute
origin so it's consistent across tile files) for spatial block
cross-validation later. Both stages skip whatever's already on disk, so
re-running `extract-all` then `assemble` as more tiles finish downloading
just picks up the new ones - `status` shows tif/extract/assemble progress per
tile.

## build_train_pool.py

Turns the joined table into a named, versioned dataset: the spatial folds and
the weighted training pool `gb_train.py` and `eda_dataset.ipynb` read. Every
run needs `--dataset-name NAME`, and output goes to
`data/processed/datasets/NAME/`, so tuning a sampling constant and rerunning
under a new name never touches a dataset already trained against. `folds`
assigns the 1,727 10km blocks, balancing block count, the label=1 and label=2
counts, and total rows all at once (greedy pass, then pairwise swaps), and
writes `block_folds.parquet`. `pool` keeps every label=1 and label=2 pixel,
label=1 duplicated `--pos-row-dup` times over (2 by default), thins label=0
with a keep rate that decays with distance to 2019 development, and writes
`gb_train_pool.parquet`. Each fold gets its own decay constant solved in
closed form so all five contribute the same row budget. `report` prints the
per fold counts, weights and distance quantiles for a named dataset.

The sampling constants worth rerunning for are exposed on the CLI:
`--tau-m`, `--pos-row-dup`, `--pos-weight-mult`, `--rows-per-training-set` on
`pool`, `--k-folds` and `--seed` on `folds`. Whatever a dataset was actually
built with, defaulted or overridden, gets written to
`data/processed/datasets/NAME/config.json`, so the directory is self
documenting.

Both stages are deterministic, so rebuilding the same name reproduces the
same split and the same draw. That depends on two things worth not undoing:
the block aggregate is sorted before the seeded shuffle, and the draw is
keyed on `hash(row, col)` rather than `random()`, which duckdb does not
reproduce across thread counts. Rerun with `--force` to overwrite a dataset
under the same name; otherwise pick a new `--dataset-name` and the old one
stays untouched.

A third subcommand, `features`, builds a shared cache of neighborhood
features straight from the 2019 NLCD raster: local development density at
100m/300m/500m, edge density at 300m, and direction to the nearest
developed pixel, encoded as sine and cosine. Not tied to a dataset name,
one row per wetland sample pixel, written once to
`data/processed/neighborhood_features.parquet` and required before
`gb_common.py` will connect to anything. Pass `--with-neighborhood-features`
to `pool` to carry those columns into a dataset's `gb_train_pool.parquet`
too. See `PLAN.md` for how each feature is built and why.

## block_distance_profiles.py

Groups the 1,727 10km blocks by the shape of their distance to development
distribution. Diagnostic only, nothing here feeds the folds, the pool or the
feature set. `profiles` reduces each block to the quantile function of its
pixels' `dist_to_developed_2019_m`, sampled at 39 knots, and writes
`data/processed/block_distance_profiles.parquet`. Quantile functions rather
than histograms because euclidean distance between two of them is the
2-Wasserstein distance between the distributions, so the k-means in `cluster`
is Wasserstein k-means and its centroids plot back as ECDFs. `cluster` runs
the k sweep, a resampling stability check, and the fit, then writes
`block_distance_clusters.parquet`. `report` prints per cluster conversion
rates and quantiles, and crosstabs clusters against a dataset's folds if you
pass `--dataset-name`.

At k=4 the regimes run from a median of 110m to 7,725m from development, and
conversion to developed falls monotonically across them from 2.19% to 0.006%
while conversion to other, not developed cover stays flat near 0.4%. The
clustering is mostly a rebinning of median distance and says so: `cluster`
prints the level/shape variance split and the ARI against clustering on the
median alone. `--shape-only` strips each block's own median first, which is
what isolates the rest. `eda_dataset.ipynb` reads the two output tables in its
"Block distance regimes" section and builds the figures.

## gb_common.py

Shared config and helpers for the modeling half: feature and target column
definitions, per fold distance normalization, the `iter_folds` generator that
both `gb_train.py` and the notebooks below loop over, and the metric
functions (`best_f1_operating_point`, `calibrate`, `reliability`). Not
runnable on its own, plays the same role for the modeling pipeline that
`gee_common.py` plays for the AlphaEarth export.

## experiments.py

One `ExperimentConfig` per named experiment: which dataset it trains against,
which feature columns the model sees, and its hyperparameter grid, if any.
What used to be a copy-pasted "Experiment N" section in a notebook is one
entry in the `EXPERIMENTS` dict here. Add an entry to try something new,
don't edit one that already has a `results/models/<name>/` directory on disk.

## gb_train.py

Fits and persists gradient boosting models for a named experiment. `baseline`
trains one LightGBM model per fold and writes, per fold, a `.joblib` file and
a slice of `oof_predictions.parquet` (label, raw score, calibrated score,
distance to development), plus `fold_summary.csv` and ROC/PR figures, all
under `results/models/<name>/baseline/`. `grid-search` screens an
experiment's grid against a single held out fold pair and writes
`results/models/<name>/grid_search/`, resumable by default: a `run_NNN`
directory that already has a `metrics.csv` is skipped, pass `--force` to redo
everything. `status` prints what is already on disk for a named experiment.
