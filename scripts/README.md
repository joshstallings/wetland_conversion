# scripts

GEE-side pipelines for pulling AlphaEarth Foundations embeddings for Florida,
plus the shared config they both depend on. Everything here runs in the
`wetlands` mamba env, from the repo root (`python scripts/<name>.py ...`).

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

