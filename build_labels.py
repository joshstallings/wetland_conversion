"""
Builds a wetland conversion sample table from two annual NLCD rasters.

Same recipe the original 2019 to 2024 sample used: every pixel NLCD calls
wetland in 2019 inside Florida, labeled by what that pixel became in the
target year. Lives in the repo as a script (the notebook that made the
2019 to 2024 file is gone from the tree) so the 2019 to 2020 file has
provenance.

    python build_labels.py 2020

Writes data/wetland_sample_labels_2019_<target>.parquet
"""

import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import transform as window_transform
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parent
NLCD_DIR = REPO_ROOT / "data" / "NLCD"
BOUNDARY_DIR = REPO_ROOT / "data" / "boundaries"
OUTPUT_DIR = REPO_ROOT / "data"

WETLAND_CLASSES = (90, 95)            # Woody Wetlands, Emergent Herbaceous Wetlands
DEVELOPED_CLASSES = (21, 22, 23, 24)  # Developed Open Space / Low / Medium / High Intensity
NLCD_NODATA = 250

# Padding read beyond Florida's bounding box so the distance to development
# transform can see development just across the state line. This value is what
# fixes the window origin, so it has to stay at 15 km or every (row, col) in the
# output stops pointing at the same pixel as the 2019 to 2024 sample and the
# AlphaEarth join keyed off it.
BUFFER_M = 15_000

# Window origin the 2019 to 2024 sample was built on, recovered from its x, y
# and row, col columns. Asserted below as a grid alignment check.
EXPECTED_WINDOW_OFFSETS = (77955, 106578)  # (row_off, col_off)

STATE_BOUNDARY_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip"

LABEL_NAMES = {
    0: "Remained wetland",
    1: "Converted to developed",
    2: "Converted to other (not developed)",
}


def nlcd_path(year):
    name = f"Annual_NLCD_LndCov_{year}_CU_C1V2"
    return NLCD_DIR / name / f"{name}.tif"


def florida_geometry(raster_crs):
    """Census 1:500k state polygon, cached under data/boundaries, reprojected
    onto the NLCD grid's CRS."""
    shp_dir = BOUNDARY_DIR / "cb_2023_us_state_500k"
    shp = shp_dir / "cb_2023_us_state_500k.shp"
    if not shp.exists():
        shp_dir.mkdir(parents=True, exist_ok=True)
        zip_path = BOUNDARY_DIR / "cb_2023_us_state_500k.zip"
        print("Downloading US state boundaries from the Census Bureau ...")
        urllib.request.urlretrieve(STATE_BOUNDARY_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(shp_dir)

    states = gpd.read_file(shp)
    florida = states.loc[states["NAME"] == "Florida"]
    assert len(florida) == 1, "Expected exactly one Florida polygon"
    return florida.to_crs(raster_crs).geometry.union_all()


def class_mask(lc, classes):
    """Membership test through a 256 entry lookup, much cheaper than np.isin
    on an array this size."""
    lut = np.zeros(256, dtype=bool)
    lut[list(classes)] = True
    return lut[lc]


def distance_to_developed(lc):
    """Meters to the nearest developed pixel. Measured against every developed
    pixel in the window, not just those inside Florida, so pixels near the state
    line are not biased by a political boundary."""
    developed = class_mask(lc, DEVELOPED_CLASSES)
    np.logical_not(developed, out=developed)
    dist = distance_transform_edt(developed, sampling=(30, 30))
    return dist.astype(np.float32)


def main(year_target):
    year_base = 2019
    path_base, path_target = nlcd_path(year_base), nlcd_path(year_target)
    for p in (path_base, path_target):
        assert p.exists(), f"Missing {p}"

    with rasterio.open(path_base) as src:
        florida_geom = florida_geometry(src.crs)
        minx, miny, maxx, maxy = florida_geom.bounds
        window = src.window(
            minx - BUFFER_M, miny - BUFFER_M, maxx + BUFFER_M, maxy + BUFFER_M
        ).round_offsets(op="floor").round_lengths(op="ceil")
        win_transform = window_transform(window, src.transform)
        lc_base = src.read(1, window=window)

    offsets = (int(window.row_off), int(window.col_off))
    assert offsets == EXPECTED_WINDOW_OFFSETS, (
        f"Window origin {offsets} does not match the 2019 to 2024 sample "
        f"{EXPECTED_WINDOW_OFFSETS}; row, col would not be comparable."
    )
    print(f"Window: {window.width:,} x {window.height:,} px at origin {offsets}")

    with rasterio.open(path_target) as src:
        lc_target = src.read(1, window=window)

    florida_pixel_mask = geometry_mask(
        [florida_geom], out_shape=lc_base.shape, transform=win_transform, invert=True
    )
    wetland_mask = class_mask(lc_base, WETLAND_CLASSES) & florida_pixel_mask
    del florida_pixel_mask
    print(f"{year_base} Florida wetland pixels: {int(wetland_mask.sum()):,}")

    t0 = time.time()
    dist_base = distance_to_developed(lc_base)[wetland_mask]
    print(f"{year_base} distance transform: {time.time() - t0:.0f}s")

    t0 = time.time()
    dist_target = distance_to_developed(lc_target)[wetland_mask]
    print(f"{year_target} distance transform: {time.time() - t0:.0f}s")

    rows, cols = np.where(wetland_mask)
    lc_base_sample = lc_base[wetland_mask]
    lc_target_sample = lc_target[wetland_mask]
    del lc_base, lc_target, wetland_mask

    label = np.full(lc_target_sample.shape, 2, dtype=np.int8)  # default: other
    label[class_mask(lc_target_sample, WETLAND_CLASSES)] = 0
    label[class_mask(lc_target_sample, DEVELOPED_CLASSES)] = 1
    label[lc_target_sample == NLCD_NODATA] = -1

    xs = win_transform.c + (cols.astype(np.float64) + 0.5) * win_transform.a
    ys = win_transform.f + (rows.astype(np.float64) + 0.5) * win_transform.e

    sample = pd.DataFrame({
        "row": rows.astype(np.int32),
        "col": cols.astype(np.int32),
        "x": xs.astype(np.float32),  # Albers Conic Equal Area, meters
        "y": ys.astype(np.float32),
        f"lc_{year_base}": lc_base_sample,
        f"lc_{year_target}": lc_target_sample,
        f"dist_to_developed_{year_base}_m": dist_base,
        f"dist_to_developed_{year_target}_m": dist_target,
        "label": label,
    })

    n_excluded = int((sample["label"] == -1).sum())
    if n_excluded:
        print(f"Dropping {n_excluded:,} pixels with nodata in {year_target}")
    sample = sample.loc[sample["label"] != -1].reset_index(drop=True)
    sample["label_name"] = sample["label"].map(LABEL_NAMES).astype("category")

    out_path = OUTPUT_DIR / f"wetland_sample_labels_{year_base}_{year_target}.parquet"
    sample.to_parquet(out_path, index=False)
    print(f"Wrote {len(sample):,} rows to {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")

    counts = sample["label"].value_counts().sort_index()
    for k in (0, 1, 2):
        n = int(counts.get(k, 0))
        print(f"  label {k} {LABEL_NAMES[k]:<36} {n:>12,}  {100 * n / len(sample):6.3f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2020)
