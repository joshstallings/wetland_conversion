# scripts

Pipelines for pulling AlphaEarth Foundations embeddings for Florida and
joining them onto the NLCD wetland samples, plus the shared config the GEE
scripts depend on. Everything here runs in the `wetlands` mamba env, from the
repo root (`python scripts/<name>.py ...`).

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
are done. Both stages skip whatever's already on disk, so re-running
`extract-all` then `assemble` as more tiles finish downloading just picks up
the new ones - `status` shows tif/extract/assemble progress per tile.
