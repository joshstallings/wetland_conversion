"""Join AlphaEarth 30m embeddings onto the NLCD wetland sample pixels, tile by
tile, as export_alphaearth_30m.py keeps filling in data/AlphaEarth/.

Every wetland sample's x/y (in wetland_sample_labels_2019_2024.parquet) and
every AlphaEarth export tif sit on the exact same 30m Albers lattice
(NLCD_ORIGIN_X/Y, NLCD_PIXEL_M - see gee_common.py), so there's no spatial
join here, just affine-transform inversion: which tile does a sample's x/y
fall in, and which pixel of that tile is it. The tile-membership formula
below is the same one build_florida_tile_grid_30m.py used to lay out the grid
in the first place, so a sample's derived tile_id is guaranteed to name a
real tile - no need to load florida_tiles_30m_60km.gpkg at all.

Two stages, kept separate because they scale differently:

  extract-tile / extract-all
      One (tile_id, year) at a time - opens that tile's tif, pulls the ~64
      band values at every wetland sample pixel inside it, decodes to float32
      [-1, 1], and writes an intermediate parquet under
      data/processed/alphaearth_extract/. This is the expensive step (has to
      read a ~256MB raster per tile-year) and is the only one that needs the
      full 59M-row samples table in memory.

  assemble
      Cheap cross-year merge - once a tile has all of --years extracted,
      stitches the per-year parquets together on (row, col) into one wide
      row per pixel (A00_2017...A63_2019) and writes
      data/processed/alphaearth_wetland_joined/{tile_id}.parquet. Inner join
      on purpose: a pixel that's nodata (no AlphaEarth coverage) in any one
      year drops out entirely rather than leaving holes in the feature stack.

Both stages just skip whatever's already done, so as export_alphaearth_30m.py
finishes more tiles, re-running `extract-all` then `assemble` picks up
whatever's newly available and leaves everything else alone.

Usage:
    python scripts/join_alphaearth_samples.py extract-tile --tile-id r039_c056 --year 2017
    python scripts/join_alphaearth_samples.py extract-all [--years 2017 2018 2019]
    python scripts/join_alphaearth_samples.py assemble [--years 2017 2018 2019]
    python scripts/join_alphaearth_samples.py status
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import rasterio

from export_alphaearth_30m import NODATA, OFFSET, SCALE_FACTOR, local_tile_path
from gee_common import NLCD_ORIGIN_X, NLCD_ORIGIN_Y, NLCD_PIXEL_M, TILE_SIZE_PX_30M

SAMPLES_PATH = "data/processed/wetland_sample_labels_2019_2024.parquet"
EXTRACT_DIR = "data/processed/alphaearth_extract"
JOINED_DIR = "data/processed/alphaearth_wetland_joined"

TILE_SIZE_M = TILE_SIZE_PX_30M * NLCD_PIXEL_M  # 60km - matches build_florida_tile_grid_30m.py

BAND_COLS = [f"A{b:02d}" for b in range(64)]

# lc_2019/lc_2024/dist_to_developed_2024_m describe the post-2019 outcome, not
# the pixel as of the label year - dropped here so they can't leak into what's
# meant to be a feature table. dist_to_developed_2019_m is kept since it's a
# 2019-only quantity, same footing as the embeddings themselves.
LABEL_COLS = ["row", "col", "x", "y", "dist_to_developed_2019_m", "label", "label_name", "tile_row", "tile_col"]

TILE_TIF_RE = re.compile(r"AlphaEarth30m_Florida_(\d{4})_(r\d{3}_c\d{3})\.tif$")
EXTRACT_FILE_RE = re.compile(r"(r\d{3}_c\d{3})_(\d{4})\.parquet$")


def parse_tile_id(tile_id: str) -> tuple[int, int]:
    m = re.match(r"r(\d+)_c(\d+)", tile_id)
    return int(m.group(1)), int(m.group(2))


def load_samples_with_tile_index() -> pd.DataFrame:
    """Wetland samples plus, for every row, which AlphaEarth tile it falls in
    (_grid_row/_grid_col) and its pixel offset within that tile
    (tile_row/tile_col). Kept as int grid indices rather than a formatted
    tile_id string - a full-column string format over 59M rows is real work
    for no benefit, since every caller here already knows which tile_id it
    wants and just needs to filter down to it."""
    df = pd.read_parquet(SAMPLES_PATH)
    x = df["x"].to_numpy(dtype="float64")
    y = df["y"].to_numpy(dtype="float64")

    df["_grid_row"] = np.floor((NLCD_ORIGIN_Y - y) / TILE_SIZE_M).astype(np.int32)
    df["_grid_col"] = np.floor((x - NLCD_ORIGIN_X) / TILE_SIZE_M).astype(np.int32)

    tile_minx = NLCD_ORIGIN_X + df["_grid_col"].to_numpy(dtype="float64") * TILE_SIZE_M
    tile_maxy = NLCD_ORIGIN_Y - df["_grid_row"].to_numpy(dtype="float64") * TILE_SIZE_M
    # samples' x/y are pixel centers (see wetland_sample_labels.ipynb), 15m
    # off any tile or pixel edge, so this floor division never sits close
    # enough to a boundary for float precision to flip it
    df["tile_row"] = np.floor((tile_maxy - y) / NLCD_PIXEL_M).astype(np.int32)
    df["tile_col"] = np.floor((x - tile_minx) / NLCD_PIXEL_M).astype(np.int32)
    return df


def samples_for_tile(samples: pd.DataFrame, tile_id: str) -> pd.DataFrame:
    row, col = parse_tile_id(tile_id)
    return samples[(samples["_grid_row"] == row) & (samples["_grid_col"] == col)]


def local_tile_years() -> list[tuple[str, int]]:
    """(tile_id, year) for every real (non-test) tile-year tif actually on
    disk right now."""
    found = []
    for path in glob.glob("data/AlphaEarth/AlphaEarth30m_Florida_*_r*_c*.tif"):
        m = TILE_TIF_RE.search(os.path.basename(path))
        if m:
            found.append((m.group(2), int(m.group(1))))
    return found


def extract_tile_year(tile_id: str, year: int, samples: pd.DataFrame, force: bool = False):
    """Pull every wetland sample pixel's 64 decoded bands out of one tile's
    tif and write the intermediate parquet. Returns (path, n_rows), or
    (path, None) if it already existed and was left alone."""
    out_path = os.path.join(EXTRACT_DIR, f"{tile_id}_{year}.parquet")
    if os.path.exists(out_path) and not force:
        return out_path, None

    sub = samples_for_tile(samples, tile_id)
    if sub.empty:
        return None, 0

    tif_path = local_tile_path(tile_id, year)
    with rasterio.open(tif_path) as src:
        arr = src.read()  # (64, H, W) uint8 - whole tile at once, ~256MB, cheaper than per-point sampling
        h, w = src.height, src.width

    rows = sub["tile_row"].to_numpy()
    cols = sub["tile_col"].to_numpy()
    in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if not in_bounds.all():
        # shouldn't happen given the tile grid's coastal buffer, but a sample
        # landing outside the raster it was assigned to would silently wrap
        # via fancy indexing otherwise - guard rather than trust it
        print(f"  {tile_id} {year}: dropping {int((~in_bounds).sum())} samples outside the tile's pixel bounds")
        sub, rows, cols = sub.loc[in_bounds], rows[in_bounds], cols[in_bounds]

    raw = arr[:, rows, cols].T  # (n, 64) uint8
    valid = (raw != NODATA).any(axis=1)  # drop only pixels with zero coverage - nodata fills all 64 bands uniformly
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"  {tile_id} {year}: {n_dropped}/{len(valid)} samples had no {year} AlphaEarth coverage here - dropped")

    decoded = (raw.astype("float32") - OFFSET) / SCALE_FACTOR
    out = sub.loc[valid, LABEL_COLS].copy()
    out["tile_id"] = tile_id
    for i, col_name in enumerate(BAND_COLS):
        out[col_name] = decoded[valid, i]

    os.makedirs(EXTRACT_DIR, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path, len(out)


def cmd_extract_tile(args):
    tif_path = local_tile_path(args.tile_id, args.year)
    if not os.path.exists(tif_path):
        raise SystemExit(f"{tif_path} doesn't exist yet - nothing to extract")
    out_path = os.path.join(EXTRACT_DIR, f"{args.tile_id}_{args.year}.parquet")
    if os.path.exists(out_path) and not args.force:
        print(f"{out_path} already exists - pass --force to redo it")
        return

    print(f"loading {SAMPLES_PATH}...")
    samples = load_samples_with_tile_index()
    path, n = extract_tile_year(args.tile_id, args.year, samples, force=args.force)
    if not n:
        print(f"no wetland samples fall inside {args.tile_id} - nothing written")
        return
    print(f"wrote {path} ({n} samples)")

    band_vals = pd.read_parquet(path, columns=BAND_COLS).to_numpy()
    print(f"decoded band range: {band_vals.min():.4f} / {band_vals.max():.4f} (should sit in [-1, 1])")


def cmd_extract_all(args):
    combos = [(t, y) for t, y in local_tile_years() if y in args.years]
    todo = [(t, y) for t, y in combos if args.force or not os.path.exists(os.path.join(EXTRACT_DIR, f"{t}_{y}.parquet"))]
    print(f"{len(combos)} tile-years on disk for years {args.years}, {len(todo)} need extracting")
    if not todo:
        return

    print(f"loading {SAMPLES_PATH}...")
    samples = load_samples_with_tile_index()
    for tile_id, year in todo:
        path, n = extract_tile_year(tile_id, year, samples, force=args.force)
        if n:
            print(f"  {tile_id} {year}: wrote {n} samples")
        else:
            print(f"  {tile_id} {year}: 0 samples (no wetland pixels fall in this tile)")


def extracted_years_by_tile() -> dict[str, set[int]]:
    by_tile: dict[str, set[int]] = {}
    for path in glob.glob(os.path.join(EXTRACT_DIR, "*.parquet")):
        m = EXTRACT_FILE_RE.match(os.path.basename(path))
        if m:
            by_tile.setdefault(m.group(1), set()).add(int(m.group(2)))
    return by_tile


def cmd_assemble(args):
    extracted = extracted_years_by_tile()
    ready = sorted(t for t, years in extracted.items() if set(args.years) <= years)
    print(f"{len(extracted)} tiles have some extraction done, {len(ready)} have all of {args.years}")

    os.makedirs(JOINED_DIR, exist_ok=True)
    n_done = 0
    for tile_id in ready:
        out_path = os.path.join(JOINED_DIR, f"{tile_id}.parquet")
        if os.path.exists(out_path) and not args.force:
            continue

        merged = None
        for year in args.years:
            df = pd.read_parquet(os.path.join(EXTRACT_DIR, f"{tile_id}_{year}.parquet"))
            df = df.rename(columns={b: f"{b}_{year}" for b in BAND_COLS})
            if merged is None:
                merged = df
            else:
                year_cols = ["row", "col"] + [f"{b}_{year}" for b in BAND_COLS]
                merged = merged.merge(df[year_cols], on=["row", "col"], how="inner")

        merged.to_parquet(out_path, index=False)
        print(f"  {tile_id}: wrote {out_path} ({len(merged)} rows)")
        n_done += 1

    print(f"assembled {n_done} tile(s) this run")


def cmd_status(args):
    by_tif = {}
    for tile_id, year in local_tile_years():
        by_tif.setdefault(tile_id, set()).add(year)
    extracted = extracted_years_by_tile()
    assembled = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(JOINED_DIR, "*.parquet"))}

    for tile_id in sorted(set(by_tif) | set(extracted) | assembled):
        print(
            f"{tile_id}: tif years {sorted(by_tif.get(tile_id, []))}, "
            f"extracted {sorted(extracted.get(tile_id, []))}, "
            f"assembled {'yes' if tile_id in assembled else 'no'}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_et = sub.add_parser("extract-tile")
    p_et.add_argument("--tile-id", required=True)
    p_et.add_argument("--year", type=int, required=True)
    p_et.add_argument("--force", action="store_true")
    p_et.set_defaults(func=cmd_extract_tile)

    p_ea = sub.add_parser("extract-all")
    p_ea.add_argument("--years", type=int, nargs="+", default=[2017, 2018, 2019])
    p_ea.add_argument("--force", action="store_true")
    p_ea.set_defaults(func=cmd_extract_all)

    p_asm = sub.add_parser("assemble")
    p_asm.add_argument("--years", type=int, nargs="+", default=[2017, 2018, 2019])
    p_asm.add_argument("--force", action="store_true")
    p_asm.set_defaults(func=cmd_assemble)

    p_st = sub.add_parser("status")
    p_st.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
