"""Shared config and helpers for the AlphaEarth GEE pull."""

import ee

from scripts.data_constants import ALPHAEARTH_DATASET, NLCD_ORIGIN_X, NLCD_ORIGIN_Y, NLCD_PIXEL_M

GEE_PROJECT = "wetlands-ml-405421"

# AlphaEarth Foundations annual embedding, 64 bands (A00-A63), native 10m pixels.
# The collection is already tiled by UTM zone at the source, so a region that
# straddles a UTM boundary needs a mosaic() of multiple granules - handled below.
DATASET = ALPHAEARTH_DATASET
N_BANDS = 64

# Same non-standard MRLC "AEA WGS84" Albers definition as the NLCD GeoTIFFs and
# the wetland_sample_labels parquet's x/y columns - pulled straight from one of
# the NLCD .tif headers, no clean EPSG code for it.
NLCD_CRS_WKT = (
    'PROJCS["AEA        WGS84",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],'
    'PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],'
    'PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],'
    'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
)

# NLCD_ORIGIN_X/Y/PIXEL_M (also from that .tif header) live in data_constants.py
# now, shared with block_distance_profiles.py. Anything meant to stack with
# NLCD pixel-for-pixel needs to snap to this grid, not just share NLCD_CRS_WKT.
NLCD_TRANSFORM = [NLCD_PIXEL_M, 0, NLCD_ORIGIN_X, 0, -NLCD_PIXEL_M, NLCD_ORIGIN_Y]

# 60km tiles for the 30m/uint8 export pipeline (export_alphaearth_30m.py) -
# big tiles work fine here since uint8 @ 30m is a small fraction of the data
# per unit area that native float32 @ 10m would be.
TILE_SIZE_PX_30M = 2_000  # 2000 * 30m = 60km, ~256MB/tile/year raw (64 bands, uint8)
TILE_GRID_30M_PATH = "data/boundaries/florida_tiles_30m_60km.gpkg"


def init_ee():
    ee.Initialize(project=GEE_PROJECT)


def year_collection(year: int, region: "ee.Geometry") -> "ee.ImageCollection":
    return (
        ee.ImageCollection(DATASET)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(region)
    )


def year_image(year: int, region: "ee.Geometry") -> "ee.Image":
    """Mosaic the AlphaEarth granules covering `region` for one calendar year."""
    return year_collection(year, region).mosaic().clip(region)


def year_native_projection(year: int, region: "ee.Geometry") -> "ee.Projection":
    """Native 10m projection of one AlphaEarth granule covering `region`.

    mosaic() doesn't leave the result with a usable default projection - the
    granules going into it can be in different native UTM zones, and EE won't
    just pick one - so anything downstream that cares about pixel size
    (reduceResolution, chiefly) needs this passed to setDefaultProjection()
    first. The granule chosen is arbitrary; for a region spanning UTM zones
    the ~1m worst-case offset from using one zone's grid for both is well
    below the 30m target resolution we're aggregating up to.
    """
    return year_collection(year, region).first().select(0).projection()
