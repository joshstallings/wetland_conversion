"""
Exports one year of AlphaEarth embeddings for Florida as 30 m tiles on the NLCD grid,
matching the 2020 to 2022 tiles in /mnt/disks/data/alphaearth_embeddings.

The original export script for those tiles was lost, so this is rebuilt from the tiles
themselves:

    60 km tiles, 2000 x 2000 px at 30 m, EPSG:5070, named by (row, col) on a fixed grid
    anchored at x = -2415585, y = 3314805 (upper left of the Annual NLCD CONUS raster)
    64 bands A00 to A63, uint8, byte = v * 127.5 + 127.5, nodata 0
    10 m to 30 m by plain area weighted mean, no renormalizing afterward (decoded
    norms sit around 0.96 to 1.0, not exactly 1)

Usage:

    python GEE/export_alphaearth_30m.py --year 2024 --dry-run
    python GEE/export_alphaearth_30m.py --year 2024 --project <ee project> --bucket <bucket>
    gsutil -m cp gs://<bucket>/alphaearth_30m/*2024*.tif /mnt/disks/data/alphaearth_embeddings/
    python GEE/export_alphaearth_30m.py --check /mnt/disks/data/alphaearth_embeddings --year 2024

Provenance: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, calendar year image for --year.
"""

import argparse
import glob
import os
import re

import numpy as np

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
STATES = "data/boundaries/cb_2023_us_state_500k/cb_2023_us_state_500k.shp"
FLORIDA_FIPS = "12"

CRS = "EPSG:5070"
X0, Y0 = -2415585, 3314805
PIXEL = 30
TILE_PX = 2000
TILE_M = TILE_PX * PIXEL

BANDS = [f"A{i:02d}" for i in range(64)]
NAME = "AlphaEarth30m_Florida_{year}_r{r:03d}_c{c:03d}"


def tile_bounds(r, c):
    """(xmin, ymin, xmax, ymax) in EPSG:5070 meters."""
    x = X0 + TILE_M * c
    y = Y0 - TILE_M * r
    return x, y - TILE_M, x + TILE_M, y


def florida_tiles():
    import geopandas as gpd
    from shapely.geometry import box

    states = gpd.read_file(STATES)
    fl = states.loc[states.STATEFP == FLORIDA_FIPS].to_crs(CRS).geometry.union_all()

    xmin, ymin, xmax, ymax = fl.bounds
    c_lo, c_hi = int((xmin - X0) // TILE_M), int((xmax - X0) // TILE_M)
    r_lo, r_hi = int((Y0 - ymax) // TILE_M), int((Y0 - ymin) // TILE_M)
    return [
        (r, c)
        for r in range(r_lo, r_hi + 1)
        for c in range(c_lo, c_hi + 1)
        if fl.intersects(box(*tile_bounds(r, c)))
    ]


def embedding_30m(year, region):
    import ee

    coll = (
        ee.ImageCollection(COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(region)
    )
    if coll.size().getInfo() == 0:
        raise SystemExit(f"no {COLLECTION} images for {year}")

    # Average each source image in its own UTM 10 m projection before mosaicking. A
    # mosaic has no native projection, so reduceResolution on it would average over a
    # resampled grid instead of the real 10 m pixels. Florida straddles UTM 16 and 17.
    def to_30m(img):
        return (
            img.select(BANDS)
            .reduceResolution(ee.Reducer.mean(), maxPixels=64)
            .reproject(crs=CRS, crsTransform=[PIXEL, 0, X0, 0, -PIXEL, Y0])
        )

    emb = coll.map(to_30m).mosaic()
    # toUint8 clamps, so v = 1.0 lands on 255. A real 0 byte only happens at v = -1, and
    # nodata is defined as all 64 bands zero, so the two can't be confused in practice.
    return emb.multiply(127.5).add(127.5).round().toUint8().unmask(0)


def export(year, project, bucket, prefix, tiles):
    import ee

    ee.Initialize(project=project)
    rows, cols = zip(*tiles)
    xmin, _, _, ymax = tile_bounds(min(rows), min(cols))
    _, ymin, xmax, _ = tile_bounds(max(rows), max(cols))
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj=CRS, geodesic=False)
    img = embedding_30m(year, region)

    for r, c in tiles:
        xmin, ymin, xmax, ymax = tile_bounds(r, c)
        name = NAME.format(year=year, r=r, c=c)
        # Inset the region by a meter so EE snaps to exactly 2000 columns and rows and
        # doesn't pick up a sliver pixel from the neighbor tile on an exact edge.
        task = ee.batch.Export.image.toCloudStorage(
            image=img,
            description=name,
            bucket=bucket,
            fileNamePrefix=f"{prefix}/{name}",
            region=ee.Geometry.Rectangle([xmin + 1, ymin + 1, xmax - 1, ymax - 1],
                                         proj=CRS, geodesic=False),
            crs=CRS,
            crsTransform=[PIXEL, 0, xmin, 0, -PIXEL, ymax],
            maxPixels=1e9,
            fileFormat="GeoTIFF",
            formatOptions={"noData": 0},
        )
        task.start()
        print(f"started {name}  {task.id}")


def check(tile_dir, year, ref_year):
    """Compare each new tile against the same cell from an older year already on disk."""
    import rasterio

    new = sorted(glob.glob(os.path.join(tile_dir, f"AlphaEarth30m_Florida_{year}_*.tif")))
    print(f"{len(new)} tiles for {year}")
    bad = 0
    for f in new:
        ref = f.replace(f"_{year}_", f"_{ref_year}_")
        with rasterio.open(f) as s:
            shape, transform, dtype = (s.count, s.height, s.width), s.transform, s.dtypes[0]
            sample = s.read(out_shape=(s.count, 200, 200)).reshape(s.count, -1).T
        problems = []
        if shape != (64, TILE_PX, TILE_PX) or dtype != "uint8":
            problems.append(f"shape {shape} {dtype}")
        if os.path.exists(ref):
            with rasterio.open(ref) as s:
                if s.transform != transform:
                    problems.append("transform differs from reference")
        valid = sample[(sample != 0).any(1)]
        med = np.nan
        if len(valid):
            med = np.median(np.linalg.norm((valid.astype(np.float32) - 127.5) / 127.5, axis=1))
        if problems:
            bad += 1
        print(f"{os.path.basename(f)}  valid {len(valid) / len(sample):.2f}  "
              f"median norm {med:.3f}  {'; '.join(problems)}")
    print(f"{bad} tiles with problems")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--project")
    ap.add_argument("--bucket")
    ap.add_argument("--prefix", default="alphaearth_30m")
    ap.add_argument("--dry-run", action="store_true", help="list tiles, start nothing")
    ap.add_argument("--check", metavar="TILE_DIR", help="verify downloaded tiles")
    ap.add_argument("--ref-year", type=int, default=2022)
    args = ap.parse_args()

    if args.check:
        check(args.check, args.year, args.ref_year)
        return

    tiles = florida_tiles()
    print(f"{len(tiles)} tiles")
    if args.dry_run:
        for r, c in tiles:
            print(NAME.format(year=args.year, r=r, c=c))
        return
    if not (args.project and args.bucket):
        ap.error("--project and --bucket are needed to export")
    export(args.year, args.project, args.bucket, args.prefix, tiles)


if __name__ == "__main__":
    main()
