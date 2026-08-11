"""Export AlphaEarth annual embeddings for Florida to GCS, tile by tile,
mean-reduced from native 10m to the 30m grid our NLCD tifs already sit on.

AlphaEarth ships at 10m/64 bands - finer than the 30m NLCD land cover we're
pairing it with, and finer than we can use anyway once it's stacked against
30m labels. reduceResolution() averages each 3x3 block of native 10m pixels
into one 30m output pixel, then reproject() snaps the result onto the exact
same origin and pixel grid as the NLCD tifs (NLCD_TRANSFORM in gee_common.py
is pulled straight from an NLCD .tif header via rasterio, not just a matching
crs+scale) so the two rasters stack with zero further resampling.

The reduceResolution() has to happen directly off the year's mosaic, before
any reproject - AlphaEarth's source granules are tiled by native UTM zone
(Florida spans 16N and 17N) and a mosaic of those only carries correct
per-pixel projection info until something reprojects it. Reproject first and
reduceResolution loses the native 10m grid it needs to average over.

Embedding values live in [-1, 1] as float32. We scale by 127, shift by +128,
and round to uint8 before export - stored = round(127*x) + 128, landing in
[1, 255], decode with x = (stored - 128) / 127. That's a uint8, not the more
obvious int8, because GEE's GeoTIFF writer doesn't trust signed-byte TIFFs
and silently promotes toInt8() output to int16 - doubling the file size
without erroring or warning. uint8 actually holds at 1 byte/pixel, keeping
three years of statewide data in the ~50GB range we're targeting. 0, unused
by the shifted range, is the nodata fill for everything outside Florida.

Usage:
    python scripts/export_alphaearth_30m.py test-direct [--year 2019]
        Small synchronous patch, no GCS/billing involved - confirms the
        reduceResolution/reproject/scale/dtype chain, and checks the output
        actually lands on the same pixel lattice as a real NLCD tif, before
        anything is submitted as a batch task.

    python scripts/export_alphaearth_30m.py test-tile [--year 2019]
        Submits one real tile through the actual GCS export path (bucket
        jstallings, prefix alphaearth_florida_30m/test/), so billing/bucket
        permissions/output get checked before going statewide. Not manifest-
        tracked - check it with `task <id>`, not `status`.

    python scripts/export_alphaearth_30m.py task TASK_ID
        Checks one task by id directly - for test-tile's task, or any other.

    python scripts/export_alphaearth_30m.py submit --years 2017 2018 2019 [--limit N]
        Submits the statewide batch, one task per (tile, year). Safe to
        re-run - tile-years already logged in the manifest are skipped.

    python scripts/export_alphaearth_30m.py status
        Polls task state for everything in the manifest.

    python scripts/export_alphaearth_30m.py verify --years 2017 2018 2019
        Stronger check than status: confirms every expected tile-year in the
        grid was actually submitted (not just what's already in the
        manifest - catches a submit run that got interrupted), flags any
        FAILED/CANCELLED tasks with their error, and, where bucket access is
        available, cross-checks that a file actually exists in GCS for every
        task reporting COMPLETED.
"""

import argparse
import csv
import glob
import os
import time

import ee
import geopandas as gpd
import requests
from pyproj import Transformer

from gee_common import (
    NLCD_CRS_WKT,
    NLCD_ORIGIN_X,
    NLCD_ORIGIN_Y,
    NLCD_PIXEL_M,
    NLCD_TRANSFORM,
    TILE_GRID_30M_PATH,
    TILE_SIZE_PX_30M,
    init_ee,
    year_image,
    year_native_projection,
)

GCS_BUCKET = "jstallings"
GCS_PROJECT = "wetlands-ml-405421"  # project the bucket lives in - happens to match gee_common.GEE_PROJECT now, but kept separate since bucket and EE compute project aren't the same concern
GCS_PREFIX = "alphaearth_florida_30m"
MANIFEST_PATH = "data/AlphaEarth/export_manifest_30m.csv"
TEST_OUT_DIR = "data/AlphaEarth/test"

SCALE_FACTOR = 127  # embeddings are in [-1, 1]; *127 keeps round(127*x) inside a signed byte's range
OFFSET = 128  # shifts round(127*x) from [-127, 127] to [1, 255], leaving 0 free as nodata
NODATA = 0

# Same logic as build_florida_tile_grid_30m.py's COASTAL_BUFFER_M: buffer
# every tile before computing anything so reduceResolution's 3x3 windows have
# real 10m data on every side, including edges shared with a neighboring tile
# (not just the coast). This only affects what feeds the mean - the image
# below is clipped back to the exact, unbuffered Florida polygon before
# export, so no other tile's or state's data reaches the output regardless of
# how generous this is.
COMPUTE_BUFFER_M = 2_000

TEST_LON, TEST_LAT = -80.9, 25.3  # Everglades marsh - actual wetland signal rather than an arbitrary corner of the grid

MANIFEST_FIELDS = ["tile_id", "year", "task_id", "description", "status"]

_to_nlcd = Transformer.from_crs("EPSG:4326", NLCD_CRS_WKT, always_xy=True)


def florida_geometry() -> "ee.Geometry":
    """Exact FL polygon from EE's public TIGER states table - avoids uploading
    the local Census shapefile as an EE asset just for this."""
    states = ee.FeatureCollection("TIGER/2018/States")
    return states.filter(ee.Filter.eq("NAME", "Florida")).geometry()


def load_grid():
    if not os.path.exists(TILE_GRID_30M_PATH):
        raise SystemExit(
            f"{TILE_GRID_30M_PATH} doesn't exist - run "
            "scripts/build_florida_tile_grid_30m.py first"
        )
    return gpd.read_file(TILE_GRID_30M_PATH)  # already in NLCD_CRS_WKT


def tile_transform(bounds) -> list:
    minx, _, _, maxy = bounds
    return [NLCD_PIXEL_M, 0, minx, 0, -NLCD_PIXEL_M, maxy]


def year_tile_image_30m(year: int, tile_bounds, fl_exact: "ee.Geometry") -> "ee.Image":
    """AlphaEarth mosaic for `year` over one tile, mean-reduced to 30m on the
    NLCD grid, shifted to uint8, hard-clipped to Florida with an explicit
    nodata fill. `tile_bounds` is (minx, miny, maxx, maxy) in NLCD_CRS_WKT
    meters, already snapped to the pixel lattice by the tile grid builder."""
    minx, miny, maxx, maxy = tile_bounds
    compute_region = ee.Geometry.Rectangle(
        [minx - COMPUTE_BUFFER_M, miny - COMPUTE_BUFFER_M, maxx + COMPUTE_BUFFER_M, maxy + COMPUTE_BUFFER_M],
        proj=NLCD_CRS_WKT,
        evenOdd=False,
    )
    img10 = year_image(year, compute_region)
    # mosaic() doesn't carry a default projection reduceResolution can use on
    # its own - see year_native_projection's docstring in gee_common.py.
    img10 = img10.setDefaultProjection(year_native_projection(year, compute_region))

    # 30m / 10m = 3, so each output pixel averages a 3x3 block of input
    # pixels - maxPixels=64 (reduceResolution's default) is already well past
    # the 9 we need, spelled out here so the window math is explicit.
    reduced = img10.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=64).reproject(
        ee.Projection(NLCD_CRS_WKT, NLCD_TRANSFORM)
    )
    shifted = reduced.multiply(SCALE_FACTOR).add(OFFSET).round().toUint8()

    # Clip back down to the tile's own exact bounds, not the buffered compute
    # region above. The dimensions+crsTransform export derives its actual
    # pixel size from the image's clipped footprint (EE's internal
    # clipToBoundsAndScale) rather than trusting crsTransform's scale outright
    # - leaving the compute buffer in here silently stretched every pixel
    # past 30m the first time this ran (43.33m, exactly the buffered extent
    # divided by the requested pixel count).
    tile_rect = ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj=NLCD_CRS_WKT, evenOdd=False)
    return shifted.clip(tile_rect).clip(fl_exact).unmask(NODATA)


def cmd_test_direct(args):
    init_ee()
    os.makedirs(TEST_OUT_DIR, exist_ok=True)

    cx, cy = _to_nlcd.transform(TEST_LON, TEST_LAT)
    # snap the patch to the NLCD lattice too, same as a real tile would be
    col = round((cx - NLCD_ORIGIN_X) / NLCD_PIXEL_M)
    row = round((NLCD_ORIGIN_Y - cy) / NLCD_PIXEL_M)
    half_px = 150  # 150*30m = 4.5km half-width, small enough for a sync download
    x0 = NLCD_ORIGIN_X + (col - half_px) * NLCD_PIXEL_M
    y1 = NLCD_ORIGIN_Y - (row - half_px) * NLCD_PIXEL_M
    dims = 2 * half_px
    transform = [NLCD_PIXEL_M, 0, x0, 0, -NLCD_PIXEL_M, y1]

    fl_exact = florida_geometry()
    bounds = (x0, y1 - dims * NLCD_PIXEL_M, x0 + dims * NLCD_PIXEL_M, y1)
    img = year_tile_image_30m(args.year, bounds, fl_exact)

    url = img.getDownloadURL(
        {"crs": NLCD_CRS_WKT, "crsTransform": transform, "dimensions": f"{dims}x{dims}", "format": "GEO_TIFF"}
    )
    out_path = os.path.join(TEST_OUT_DIR, f"test_direct_30m_{args.year}.tif")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"wrote {out_path} ({len(resp.content) / 1e6:.1f} MB)")

    import rasterio

    with rasterio.open(out_path) as src:
        print(f"shape: {src.count} bands, {src.width}x{src.height}, dtype {src.dtypes[0]}")
        print(f"crs: {src.crs}")
        print(f"transform: {src.transform}")
        arr = src.read(1)
        print(f"band A00 min/max: {arr.min()} / {arr.max()} (nodata {NODATA})")
        valid = arr[arr != NODATA]
        if valid.size:
            decoded = (valid.astype("float32") - OFFSET) / SCALE_FACTOR
            print(f"decoded band A00 min/max: {decoded.min():.4f} / {decoded.max():.4f} (should sit in [-1, 1])")

    # confirm this actually lands on the same lattice as a real NLCD tif, not
    # just the same crs - offsets should be whole pixels, ideally zero
    nlcd_tifs = glob.glob("data/NLCD/*/*.tif")
    if nlcd_tifs:
        with rasterio.open(nlcd_tifs[0]) as nlcd:
            dx = (transform[2] - nlcd.transform.c) / NLCD_PIXEL_M
            dy = (transform[5] - nlcd.transform.f) / NLCD_PIXEL_M
            aligned = abs(dx - round(dx)) < 1e-6 and abs(dy - round(dy)) < 1e-6
            print(f"aligned to {nlcd_tifs[0]}: {aligned} (offset {dx:.6f}, {dy:.6f} px)")


def nearest_tile(grid):
    from shapely.geometry import Point

    x, y = _to_nlcd.transform(TEST_LON, TEST_LAT)
    pt = Point(x, y)
    dists = grid.geometry.distance(pt)
    return grid.loc[dists.idxmin()]


def cmd_test_tile(args):
    init_ee()
    grid = load_grid()
    tile = nearest_tile(grid)
    fl_exact = florida_geometry()
    img = year_tile_image_30m(args.year, tile.geometry.bounds, fl_exact)

    description = f"AlphaEarth30m_test_{args.year}_{tile.tile_id}"
    task = ee.batch.Export.image.toCloudStorage(
        image=img,
        description=description,
        bucket=GCS_BUCKET,
        fileNamePrefix=f"{GCS_PREFIX}/test/{description}",
        crs=NLCD_CRS_WKT,
        crsTransform=tile_transform(tile.geometry.bounds),
        dimensions=f"{TILE_SIZE_PX_30M}x{TILE_SIZE_PX_30M}",
        maxPixels=1e8,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True, "noData": NODATA},
    )
    task.start()
    print(f"submitted test export: tile={tile.tile_id} year={args.year}")
    print(f"task id: {task.id}")
    print(f"gs://{GCS_BUCKET}/{GCS_PREFIX}/test/{description}.tif")
    # not manifest-tracked (manifest is only for the resumable submit batch,
    # see cmd_submit), so `status` won't show this - check the task directly
    print(f"poll with: python scripts/export_alphaearth_30m.py task {task.id}")


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH) as f:
        return {(r["tile_id"], r["year"]): r for r in csv.DictReader(f)}


def append_manifest_rows(rows):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    write_header = not os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def cmd_task(args):
    """Check one task by id directly - for test-tile's task, which isn't in
    the manifest, or any other id you have lying around."""
    init_ee()
    statuses = ee.data.getTaskStatus(args.task_id)
    if not statuses or statuses[0].get("state") is None:
        print(f"no such task: {args.task_id}")
        return
    t = statuses[0]
    print(f"{t['description']}: {t['state']}")
    if t.get("error_message"):
        print(f"  error: {t['error_message']}")


def cmd_submit(args):
    init_ee()
    grid = load_grid()
    fl_exact = florida_geometry()
    existing = load_manifest()

    jobs = [(row, year) for _, row in grid.iterrows() for year in args.years]
    jobs = [(row, year) for row, year in jobs if (row.tile_id, str(year)) not in existing]
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"{len(jobs)} tile-years to submit ({len(existing)} already in manifest)")
    submitted = []
    for row, year in jobs:
        img = year_tile_image_30m(year, row.geometry.bounds, fl_exact)
        description = f"AlphaEarth30m_Florida_{year}_{row.tile_id}"
        task = ee.batch.Export.image.toCloudStorage(
            image=img,
            description=description,
            bucket=GCS_BUCKET,
            fileNamePrefix=f"{GCS_PREFIX}/{description}",
            crs=NLCD_CRS_WKT,
            crsTransform=tile_transform(row.geometry.bounds),
            dimensions=f"{TILE_SIZE_PX_30M}x{TILE_SIZE_PX_30M}",
            maxPixels=1e8,
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True, "noData": NODATA},
        )
        task.start()
        submitted.append(
            {
                "tile_id": row.tile_id,
                "year": year,
                "task_id": task.id,
                "description": description,
                "status": "SUBMITTED",
            }
        )
        # GEE throttles task creation - burning through the queue too fast just
        # trades a batch-side wait for a client-side 429
        time.sleep(args.sleep)
        if len(submitted) % 25 == 0:
            print(f"  {len(submitted)}/{len(jobs)} submitted")
            append_manifest_rows(submitted)
            submitted = []

    if submitted:
        append_manifest_rows(submitted)
    print(f"done, {len(jobs)} tasks submitted this run")


def cmd_status(args):
    init_ee()
    manifest = load_manifest()
    if not manifest:
        print("manifest is empty, nothing submitted yet")
        return

    tasks = {t.id: t for t in ee.batch.Task.list()}
    counts = {}
    rows = []
    for row in manifest.values():
        task = tasks.get(row["task_id"])
        state = task.state if task else "UNKNOWN"
        row["status"] = state
        counts[state] = counts.get(state, 0) + 1
        rows.append(row)

    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    for state, n in sorted(counts.items()):
        print(f"{state}: {n}")


def gcs_output_filenames():
    """Every blob under gs://GCS_BUCKET/GCS_PREFIX/, for cross-checking
    against the manifest. Returns None (rather than raising) if we can't
    reach GCS at all - the google-cloud-storage client needs its own
    Application Default Credentials, a separate credential store from
    `earthengine authenticate`, and this environment doesn't have it set up."""
    try:
        from google.cloud import storage
    except ImportError:
        return None
    try:
        client = storage.Client(project=GCS_PROJECT)
        blobs = client.list_blobs(GCS_BUCKET, prefix=f"{GCS_PREFIX}/")
        return [b.name.rsplit("/", 1)[-1] for b in blobs]
    except Exception as e:
        print(f"(couldn't list gs://{GCS_BUCKET}/{GCS_PREFIX}/ - skipping file check: {e})")
        return None


def cmd_verify(args):
    """Stronger than status: status only reports on what's already in the
    manifest, so a submit run that died partway through (network blip,
    ctrl-C, GEE 429) would silently look fine. This instead starts from the
    full expected (tile, year) set and checks both ends - was it submitted at
    all, and did the task actually finish - plus a best-effort check that a
    file exists in the bucket for everything reporting COMPLETED."""
    init_ee()
    grid = load_grid()
    expected = {(row.tile_id, str(year)) for _, row in grid.iterrows() for year in args.years}
    manifest = load_manifest()

    never_submitted = sorted(expected - set(manifest.keys()))

    tasks = {t.id: t for t in ee.batch.Task.list()}
    counts = {}
    failed = []
    rows = list(manifest.values())
    for key, row in manifest.items():
        task = tasks.get(row["task_id"])
        row["status"] = task.state if task else "UNKNOWN"
        if key in expected:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if row["status"] in ("FAILED", "CANCELLED"):
                failed.append((key, task.error_message if task else "task not found"))

    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"expected: {len(expected)} tile-years ({len(grid)} tiles x {len(args.years)} years)")
    print(f"never submitted: {len(never_submitted)}")
    for tile_id, year in never_submitted[:20]:
        print(f"  {tile_id} {year}")
    if len(never_submitted) > 20:
        print(f"  ...and {len(never_submitted) - 20} more")

    for state, n in sorted(counts.items()):
        print(f"{state}: {n}")
    if failed:
        print("failed/cancelled:")
        for (tile_id, year), err in failed:
            print(f"  {tile_id} {year}: {err}")

    filenames = gcs_output_filenames()
    if filenames is None:
        print("run this from an environment with GCS access (gcloud auth application-default login) for a full file check")
        return

    completed = [(key, row) for key, row in manifest.items() if key in expected and row["status"] == "COMPLETED"]
    missing_files = [key for key, row in completed if not any(n.startswith(row["description"]) for n in filenames)]
    print(f"COMPLETED tasks with no matching file in gs://{GCS_BUCKET}/{GCS_PREFIX}/: {len(missing_files)}")
    for tile_id, year in missing_files[:20]:
        print(f"  {tile_id} {year}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_direct = sub.add_parser("test-direct")
    p_direct.add_argument("--year", type=int, default=2019)
    p_direct.set_defaults(func=cmd_test_direct)

    p_tile = sub.add_parser("test-tile")
    p_tile.add_argument("--year", type=int, default=2019)
    p_tile.set_defaults(func=cmd_test_tile)

    p_task = sub.add_parser("task")
    p_task.add_argument("task_id")
    p_task.set_defaults(func=cmd_task)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--years", type=int, nargs="+", default=[2017, 2018, 2019])
    p_submit.add_argument("--limit", type=int, default=None, help="cap tasks submitted this run")
    p_submit.add_argument("--sleep", type=float, default=0.3, help="seconds between task submissions")
    p_submit.set_defaults(func=cmd_submit)

    p_status = sub.add_parser("status")
    p_status.set_defaults(func=cmd_status)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--years", type=int, nargs="+", default=[2017, 2018, 2019])
    p_verify.set_defaults(func=cmd_verify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
