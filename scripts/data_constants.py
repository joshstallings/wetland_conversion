"""NLCD grid and class-scheme constants, and the AlphaEarth dataset id.

Plain literals, no third party imports, so anything that just needs one of
these (block_distance_profiles.py chiefly) doesn't have to pull in rasterio,
duckdb, or earthengine-api along with it. Previously these were independently
re-declared in gee_common.py, block_distance_profiles.py, build_train_pool.py,
model_common.py, and wetland_sample_labels.ipynb; this is the one place that
changes if the grid, the class codes, or the buffer ever do.
"""

# Pulled from one of the NLCD .tif headers (rasterio's src.transform): the
# exact 30m pixel lattice NLCD sits on, in the non-standard MRLC "AEA WGS84"
# Albers projection (no clean EPSG code for it, see gee_common.NLCD_CRS_WKT).
# Anything meant to stack with NLCD pixel-for-pixel needs to snap to this.
NLCD_ORIGIN_X, NLCD_ORIGIN_Y, NLCD_PIXEL_M = -2415585, 3314805, 30

# NLCD class codes (see the raster .xml sidecar files for the full legend).
WETLAND_CLASSES = (90, 95)            # Woody Wetlands, Emergent Herbaceous Wetlands
DEVELOPED_CLASSES = (21, 22, 23, 24)  # Developed: Open Space / Low / Medium / High Intensity
NLCD_NODATA = 250

# Meters of padding read beyond Florida's strict bounding box, so the
# distance to development feature can "see" development just across the
# state line rather than treating the window edge as empty space. Has to
# match exactly everywhere it's used: it's what keeps a (row, col) pointing
# at the same pixel in wetland_sample_labels.ipynb's sample table as it does
# in build_train_pool.py's raster reads.
BUFFER_M = 15_000

# AlphaEarth Foundations annual embedding bands, one column per (band, year).
BAND_COLS = [f"A{b:02d}_{year}" for year in (2017, 2018, 2019) for b in range(64)]

# GEE collection id for the AlphaEarth pull, 64 bands (A00-A63), native 10m
# pixels, already tiled by UTM zone at the source.
ALPHAEARTH_DATASET = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
