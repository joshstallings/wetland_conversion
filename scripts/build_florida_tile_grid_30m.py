"""Build a Florida tile grid snapped to the exact NLCD 30m pixel lattice, for
chunked GEE exports of the mean-reduced AlphaEarth product.

This is a separate grid from florida_tiles_20km.gpkg (which is built in
EPSG:3086 for the native-10m export pipeline and isn't pixel-aligned with
anything). Here every tile edge lands on a real NLCD pixel boundary -
(x - NLCD_ORIGIN_X) is always a whole multiple of 30 - so export_alphaearth_30m.py
can hand each tile's Export.image call an explicit crsTransform + dimensions
and land exactly on the NLCD grid, no half-pixel offset to chase down later.
Trying to do this with region + crsTransform instead runs into a real GEE bug/
quirk: EE appears to mis-derive the pixel bounds from a region far from the
transform's origin, blowing a should-be-5MB request out to several GB.

Tiles are much bigger than the 10m pipeline's: int8 output at 30m is ~144x
less data per unit area than float32 at 10m, so far fewer, larger tiles cover
the state without inflating any single export task.

Run once (or whenever TILE_SIZE_PX_30M changes):
    python scripts/build_florida_tile_grid_30m.py
"""

import geopandas as gpd
from shapely.geometry import box

from gee_common import (
    NLCD_CRS_WKT,
    NLCD_ORIGIN_X,
    NLCD_ORIGIN_Y,
    NLCD_PIXEL_M,
    TILE_GRID_30M_PATH,
    TILE_SIZE_PX_30M,
)

FL_BOUNDARY_PATH = "data/boundaries/cb_2023_us_state_500k"
TILE_SIZE_M = TILE_SIZE_PX_30M * NLCD_PIXEL_M

# Same reasoning as the 10m grid's COASTAL_BUFFER_M: buffer the state polygon
# before selecting tiles so we don't drop a tile that's mostly Gulf/Atlantic
# water but clips a sliver of coastal marsh right at the edge.
COASTAL_BUFFER_M = 2_000


def main():
    states = gpd.read_file(FL_BOUNDARY_PATH)
    florida = states.loc[states["STUSPS"] == "FL"].to_crs(NLCD_CRS_WKT)
    search_area = florida.union_all().buffer(COASTAL_BUFFER_M)

    minx, miny, maxx, maxy = search_area.bounds
    # snap outward to the NLCD lattice, in pixel-tile units from the origin -
    # this is what makes every tile edge a real NLCD pixel edge, not just an
    # arbitrary round number in this CRS
    col0 = int((minx - NLCD_ORIGIN_X) // TILE_SIZE_M)
    col1 = int((maxx - NLCD_ORIGIN_X) // TILE_SIZE_M) + 1
    row0 = int((NLCD_ORIGIN_Y - maxy) // TILE_SIZE_M)  # rows count down - y decreases from the origin
    row1 = int((NLCD_ORIGIN_Y - miny) // TILE_SIZE_M) + 1

    rows = []
    for row in range(row0, row1):
        for col in range(col0, col1):
            x0 = NLCD_ORIGIN_X + col * TILE_SIZE_M
            y1 = NLCD_ORIGIN_Y - row * TILE_SIZE_M
            tile = box(x0, y1 - TILE_SIZE_M, x0 + TILE_SIZE_M, y1)
            if tile.intersects(search_area):
                rows.append({"tile_id": f"r{row:03d}_c{col:03d}", "row": row, "col": col, "geometry": tile})

    grid = gpd.GeoDataFrame(rows, crs=NLCD_CRS_WKT)
    grid.to_file(TILE_GRID_30M_PATH, driver="GPKG")
    print(f"{len(grid)} tiles ({TILE_SIZE_M / 1000:.0f}km each) covering Florida written to {TILE_GRID_30M_PATH}")


if __name__ == "__main__":
    main()
