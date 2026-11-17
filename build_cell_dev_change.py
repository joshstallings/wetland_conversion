"""
Adds two development context features to an existing cell folder, from Annual
NLCD: the percent of each cell that went from not developed to developed between
two years, and the same percent over a square ring reaching 1 km out from the
cell's edges (the cell itself left out).

    python build_cell_dev_change.py --cell-dir data/cells_2019_2024_300m

Writes dev_change_<start>_<end>_cell_pct.npy and dev_change_<start>_<end>_ring1km_pct.npy
into the cell folder and records them under "added_features" in its manifest.

Both percents are over every pixel position in the cell or ring, water and nodata
included, the same convention as footprint_px. So a coastal cell that is half ocean
tops out near 50. A pixel counts as changed only if it is not developed in the start
year and developed in the end year, with valid data in both. Pixels that go from
developed back to not developed are ignored, and years in between aren't looked at.

These are inputs, not labels. 2017 to 2019 sits entirely before the 2019 to 2024
target window. The ring crosses block edges, so it reads land cover from blocks in
other folds, which is fine since it carries no label information.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import transform as window_transform

from build_arrays import git_commit
from build_labels import (BUFFER_M, DEVELOPED_CLASSES, EXPECTED_WINDOW_OFFSETS, NLCD_NODATA, class_mask,
                          florida_geometry, nlcd_path)

PIXEL_M = 30


def florida_window(path):
    """Same window build_labels reads, so cell_rowcol indexes straight into it."""
    with rasterio.open(path) as src:
        geom = florida_geometry(src.crs)
        minx, miny, maxx, maxy = geom.bounds
        window = src.window(minx - BUFFER_M, miny - BUFFER_M, maxx + BUFFER_M, maxy + BUFFER_M
                            ).round_offsets(op="floor").round_lengths(op="ceil")
        return window, window_transform(window, src.transform)


def read_window(year, window):
    path = nlcd_path(year)
    with rasterio.open(path) as src:
        return src.read(1, window=window)


def cell_extent_px(cell_dir, manifest):
    """Top left pixel and height, width of every cell, in window row, col.

    Width and height aren't stored, but cell_xy is the mean of the footprint pixel
    centers, so they come back out of its offset from the cell's first pixel.
    """
    rc = np.load(cell_dir / "cell_rowcol.npy").astype(np.int64)
    xy = np.load(cell_dir / "cell_xy.npy")
    footprint = np.load(cell_dir / "footprint_px.npy")
    g = manifest["pixel_grid"]
    fx = np.rint(2 * ((xy[:, 0] - g["x_off"]) / PIXEL_M - rc[:, 1]) + 1).astype(np.int64)
    fy = np.rint(2 * ((g["y_off"] - xy[:, 1]) / PIXEL_M - rc[:, 0]) + 1).astype(np.int64)
    if not np.array_equal(fx * fy, footprint):
        raise AssertionError("recovered cell width times height does not match footprint_px")
    return rc[:, 0], rc[:, 1], fy, fx


def rect_sum(S, r0, r1, c0, c1):
    """Sum over rows r0:r1 and cols c0:c1, from a summed area table padded with a zero row and col."""
    return S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-dir", required=True)
    ap.add_argument("--start", type=int, default=2017)
    ap.add_argument("--end", type=int, default=2019)
    ap.add_argument("--ring-m", type=int, default=1000)
    args = ap.parse_args()
    assert args.start < args.end

    cell_dir = Path(args.cell_dir)
    manifest_path = cell_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # 1000 m is 33.3 pixels, so the ring is really 990 m wide
    ring_px = round(args.ring_m / PIXEL_M)

    t0 = time.time()
    window, wt = florida_window(nlcd_path(args.end))
    if (int(window.row_off), int(window.col_off)) != EXPECTED_WINDOW_OFFSETS:
        raise AssertionError(f"window origin {window.row_off, window.col_off} is not {EXPECTED_WINDOW_OFFSETS}")
    # pixel_grid in the cell manifest is pixel centers, the window transform is the corner
    g = manifest["pixel_grid"]
    if (wt.c + PIXEL_M / 2, wt.f - PIXEL_M / 2) != (g["x_off"], g["y_off"]):
        raise AssertionError("cell pixel grid does not line up with the NLCD window")

    lc0 = read_window(args.start, window)
    lc1 = read_window(args.end, window)
    changed = (class_mask(lc1, DEVELOPED_CLASSES) & ~class_mask(lc0, DEVELOPED_CLASSES)
               & (lc0 != NLCD_NODATA) & (lc1 != NLCD_NODATA))
    del lc0, lc1
    print(f"window {changed.shape[0]:,} x {changed.shape[1]:,} px, "
          f"{int(changed.sum()):,} px changed to developed {args.start} to {args.end}, {time.time() - t0:.0f}s")

    # int32 holds it: the whole window is about 7e8 pixels
    S = np.zeros((changed.shape[0] + 1, changed.shape[1] + 1), np.int32)
    np.cumsum(changed, axis=0, dtype=np.int32, out=S[1:, 1:])
    np.cumsum(S[1:, 1:], axis=1, out=S[1:, 1:])
    del changed

    r0, c0, h, w = cell_extent_px(cell_dir, manifest)
    r1, c1 = r0 + h, c0 + w
    R0, C0, R1, C1 = r0 - ring_px, c0 - ring_px, r1 + ring_px, c1 + ring_px
    # the 15 km buffer on the window should keep every ring inside it
    if R0.min() < 0 or C0.min() < 0 or R1.max() > S.shape[0] - 1 or C1.max() > S.shape[1] - 1:
        raise AssertionError("a ring runs off the NLCD window")

    n_cell = rect_sum(S, r0, r1, c0, c1)
    n_ring = rect_sum(S, R0, R1, C0, C1) - n_cell
    cell_pct = (100 * n_cell / (h * w)).astype(np.float32)
    ring_pct = (100 * n_ring / ((h + 2 * ring_px) * (w + 2 * ring_px) - h * w)).astype(np.float32)

    tag = f"dev_change_{args.start}_{args.end}"
    np.save(cell_dir / f"{tag}_cell_pct.npy", cell_pct)
    np.save(cell_dir / f"{tag}_ring1km_pct.npy", ring_pct)

    for name, a in (("cell", cell_pct), ("ring", ring_pct)):
        q = np.percentile(a, [50, 90, 99]).round(2)
        print(f"{name}: share > 0 {np.mean(a > 0):.3f}, median {q[0]}, p90 {q[1]}, p99 {q[2]}, max {a.max():.1f}")

    manifest.setdefault("added_features", {})[tag] = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "script": "build_cell_dev_change.py",
        "nlcd": [nlcd_path(y).name for y in (args.start, args.end)],
        "files": [f"{tag}_cell_pct.npy", f"{tag}_ring1km_pct.npy"],
        "definition": (
            f"percent of pixel positions not developed (21 to 24) in {args.start} and developed in {args.end}, "
            "valid data in both years. Denominator is every pixel position, water and nodata included. "
            f"Ring is a square {ring_px} px ({ring_px * PIXEL_M} m) out from each cell edge, cell excluded, "
            "and crosses block edges."
        ),
        "ring_px": ring_px,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {tag} features for {len(cell_pct):,} cells, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
