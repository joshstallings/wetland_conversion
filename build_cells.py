"""
Aggregates the 30 m pixel arrays from build_arrays.py into square cells of
--cell-size-m meters, for the area version of the task: predict what share of a
cell's 2019 wetland is developed by 2024.

    python build_cells.py --cell-size-m 300
    python build_cells.py --cell-size-m 990 --out-dir data/cells_2019_2024_990m

One row per cell. Target is n_developed / n_wetland, where both counts are over
the 2019 wetland pixels in the cell (the only pixels the source arrays hold), and
label_raw == 2 (converted to something other than developed) counts as not
developed, same as the pixel target. The counts themselves are written too, so a
different denominator is one division away without rebuilding.

Cells never cross a 10 km block edge. Each block gets its own cell grid anchored
at its first pixel row and column, and every cell carries the block_code of the
block it sits in, so folds.assign_folds and arrays.fold_of_code work on cells
unchanged. A cell straddling two blocks would put its pixels in two folds at once.

The 10 km block grid is not on the 30 m pixel grid (its origin is about 15 m off,
and 10000 is not a multiple of 30), so blocks are 333 or 334 pixels wide. Any cell
size that does not divide that leaves a partial strip of cells along the far edge
of each block. footprint_px says how many pixel positions each cell actually covers
inside its block, wetland or not, ignoring the Florida boundary and open water.
Cell sizes of 90, 270, 1110 and 3330 m divide 333 pixels exactly.

Features are the per column mean and std (ddof 0) over a cell's wetland pixels of
all 192 embedding columns, so 384 values per cell. A cell with one wetland pixel
has std 0 everywhere and a target of exactly 0 or 1, so n_wetland belongs in the
trainer as a filter or a sample weight.

Distance to 2019 developed comes through as dist_mean and dist_min, the mean and
min over the same wetland pixels, in raw meters like the source. Not in
feature_columns since it is a scalar the trainer transforms on its own (log1p and
a train fold z score in datasets.py). dist_min is the cell's closest wetland pixel
to existing development.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from build_arrays import git_commit

ARRAY_DIR = "data/arrays_2019_2024"
PIXEL_M = 30
BLOCK_M = 10_000


def block_grid(manifest):
    """(block_row, block_col) per block_code, parsed from ids like b0239_0321."""
    ids = manifest["block_ids"]
    return (np.array([int(b[1:5]) for b in ids], dtype=np.int64),
            np.array([int(b[6:]) for b in ids], dtype=np.int64))


def pixel_grid_affine(xy, rowcol):
    """
    x = x_off + 30 col and y = y_off minus 30 row, checked on every row rather
    than trusted, since the block edge math below runs entirely in row/col.
    """
    col = rowcol[:, 1].astype(np.int64)
    row = rowcol[:, 0].astype(np.int64)
    x_off = int(xy[0, 0]) - PIXEL_M * int(col[0])
    y_off = int(xy[0, 1]) + PIXEL_M * int(row[0])
    if not (np.array_equal(xy[:, 0].astype(np.int64), x_off + PIXEL_M * col)
            and np.array_equal(xy[:, 1].astype(np.int64), y_off - PIXEL_M * row)):
        raise AssertionError("xy is not an exact 30 m affine of rowcol, so the grid assumption is wrong")
    return x_off, y_off


def block_pixel_ranges(xy, block_code, brow, bcol, x_off, y_off):
    """
    Half open pixel col and row ranges each block covers, wetland or not.

    The block origin is not stored anywhere, so it is recovered from the data:
    x minus 10000 block_col has to land in one 10 km window for every pixel. On
    this artifact that window is pinned to 10 m, and pixel x offsets within a block
    only ever take values on a 10 m lattice, so the membership it implies is exact.
    Asserted against every pixel at the end.
    """
    u = xy[:, 0].astype(np.int64) - BLOCK_M * bcol[block_code]
    v = xy[:, 1].astype(np.int64) + BLOCK_M * brow[block_code]
    u0, v0 = int(u.min()), int(v.min())
    if u.max() >= u0 + BLOCK_M or v.max() >= v0 + BLOCK_M:
        raise AssertionError("pixels do not fit one 10 km window per block, block ids are not a 10 km grid")
    # u and v sit on a 10 m lattice. If the data leaves more than 10 m of slack for
    # the origin, some edge pixel positions could belong to either block and the
    # footprint counts stop being exact. The targets and features would still be fine.
    if u0 - (int(u.max()) - BLOCK_M) > 10 or v0 - (int(v.max()) - BLOCK_M) > 10:
        raise AssertionError("block origin is not pinned to 10 m by the data, footprint_px would be ambiguous")

    # Block covers u0 <= x minus 10000 c < u0 + 10000. Solved for col with integer ceil.
    x_lo = u0 + BLOCK_M * bcol - x_off
    col0 = -(-x_lo // PIXEL_M)
    col1 = -(-(x_lo + BLOCK_M) // PIXEL_M)
    # y runs the other way: v0 <= y + 10000 r < v0 + 10000, y = y_off minus 30 row.
    y_hi = y_off + BLOCK_M * brow - v0
    row0 = (y_hi - BLOCK_M) // PIXEL_M + 1
    row1 = y_hi // PIXEL_M + 1
    return col0, col1, row0, row1, (u0, v0)


def cell_keys(rc, code, n, col0, row0, ncx):
    lr = (rc[:, 0].astype(np.int64) - row0[code]) // n
    lc = (rc[:, 1].astype(np.int64) - col0[code]) // n
    return lr * ncx[code] + lc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-size-m", type=int, required=True,
                    help="cell edge in meters, a multiple of 30 so a cell is a whole number of pixels")
    ap.add_argument("--array-dir", default=ARRAY_DIR)
    ap.add_argument("--out-dir", default=None,
                    help="defaults to data/cells_<array dir suffix>_<size>m")
    args = ap.parse_args()

    if args.cell_size_m < PIXEL_M or args.cell_size_m % PIXEL_M:
        lo = max(PIXEL_M, args.cell_size_m // PIXEL_M * PIXEL_M)
        raise SystemExit(f"--cell-size-m has to be a multiple of 30, try {lo} or {lo + PIXEL_M}")
    n = args.cell_size_m // PIXEL_M

    array_dir = Path(args.array_dir)
    suffix = array_dir.name.removeprefix("arrays_")
    out_dir = Path(args.out_dir or f"data/cells_{suffix}_{args.cell_size_m}m")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(array_dir / "manifest.json") as fh:
        src = json.load(fh)
    emb = np.load(array_dir / "emb.npy", mmap_mode="r")
    xy = np.load(array_dir / "xy.npy")
    rowcol = np.load(array_dir / "rowcol.npy")
    block_code = np.load(array_dir / "block_code.npy")
    label_raw = np.load(array_dir / "label_raw.npy")
    dist = np.load(array_dir / "dist.npy")
    n_emb = emb.shape[1]
    if n_emb != len(src["emb_columns"]):
        raise AssertionError("emb.npy column count does not match its manifest")

    brow, bcol = block_grid(src)
    x_off, y_off = pixel_grid_affine(xy, rowcol)
    col0, col1, row0, row1, block_origin = block_pixel_ranges(xy, block_code, brow, bcol, x_off, y_off)
    in_block = ((rowcol[:, 1] >= col0[block_code]) & (rowcol[:, 1] < col1[block_code])
                & (rowcol[:, 0] >= row0[block_code]) & (rowcol[:, 0] < row1[block_code]))
    if not in_block.all():
        raise AssertionError(f"{int((~in_block).sum()):,} pixels fall outside their own block's pixel range")
    del xy, in_block

    width, height = col1 - col0, row1 - row0
    ncx = -(-width // n)
    starts, counts = (np.asarray(src["block_spans"], dtype=np.int64).T)

    # Pass 1 is rowcol only, to size the outputs. At small cell sizes the means
    # and stds run to tens of GB, so they go straight to memmaps.
    n_cells_per_block = np.zeros(len(starts), dtype=np.int64)
    for code, (s, c) in enumerate(zip(starts, counts)):
        n_cells_per_block[code] = np.unique(cell_keys(rowcol[s:s + c], code, n, col0, row0, ncx)).size
    n_cells = int(n_cells_per_block.sum())
    print(f"{args.cell_size_m} m cells ({n}x{n} pixels): {n_cells:,} cells with wetland "
          f"from {len(starts):,} blocks, {time.time() - t0:.0f}s")

    out = {
        "emb_mean": open_memmap(out_dir / "emb_mean.npy", "w+", np.float32, (n_cells, n_emb)),
        "emb_std": open_memmap(out_dir / "emb_std.npy", "w+", np.float32, (n_cells, n_emb)),
    }
    target = np.empty(n_cells, np.float32)
    n_wetland = np.empty(n_cells, np.int32)
    n_developed = np.empty(n_cells, np.int32)
    n_other = np.empty(n_cells, np.int32)
    footprint = np.empty(n_cells, np.int32)
    cell_block = np.empty(n_cells, np.int16)
    cell_rowcol = np.empty((n_cells, 2), np.int32)
    cell_xy = np.empty((n_cells, 2), np.float64)
    dist_mean = np.empty(n_cells, np.float32)
    dist_min = np.empty(n_cells, np.float32)

    off = 0
    for code, (s, c) in enumerate(zip(starts, counts)):
        key = cell_keys(rowcol[s:s + c], code, n, col0, row0, ncx)
        order = np.argsort(key, kind="stable")
        uniq, first, cnt = np.unique(key[order], return_index=True, return_counts=True)
        m = uniq.size
        cell_of_sorted = np.repeat(np.arange(m), cnt)
        sl = slice(off, off + m)

        # One contiguous read per block, then permute in RAM. Accumulate in float64
        # and take the std as a second pass over deviations, not E[x^2] minus E[x]^2.
        e = np.asarray(emb[s:s + c], dtype=np.float32)[order]
        mean = np.add.reduceat(e, first, axis=0, dtype=np.float64) / cnt[:, None]
        dev = e - mean[cell_of_sorted].astype(np.float32)
        var = np.add.reduceat(dev * dev, first, axis=0, dtype=np.float64) / cnt[:, None]
        out["emb_mean"][sl] = mean
        out["emb_std"][sl] = np.sqrt(var)

        lab = label_raw[s:s + c][order]
        n_wetland[sl] = cnt
        n_developed[sl] = np.bincount(cell_of_sorted[lab == 1], minlength=m)
        n_other[sl] = np.bincount(cell_of_sorted[lab == 2], minlength=m)
        target[sl] = n_developed[sl] / cnt

        d = dist[s:s + c][order].astype(np.float64)
        dist_mean[sl] = np.add.reduceat(d, first) / cnt
        dist_min[sl] = np.minimum.reduceat(d, first)

        lr, lc = uniq // ncx[code], uniq % ncx[code]
        fx = np.minimum(n, width[code] - lc * n)
        fy = np.minimum(n, height[code] - lr * n)
        footprint[sl] = fx * fy
        r_start, c_start = row0[code] + lr * n, col0[code] + lc * n
        cell_rowcol[sl] = np.stack([r_start, c_start], axis=1)
        # Mean of the footprint pixels' coordinates, same point convention as xy.npy.
        cell_xy[sl, 0] = x_off + PIXEL_M * (c_start + (fx - 1) / 2)
        cell_xy[sl, 1] = y_off - PIXEL_M * (r_start + (fy - 1) / 2)
        cell_block[sl] = code
        off += m

        if code % 200 == 0:
            print(f"  block {code:>5}/{len(starts)}  {off:>11,} cells  {time.time() - t0:.0f}s")

    if off != n_cells:
        raise AssertionError(f"pass 2 wrote {off:,} cells, pass 1 counted {n_cells:,}")
    if int(n_wetland.sum()) != src["total_rows"]:
        raise AssertionError("cell wetland counts do not add back up to the pixel row count")
    if (n_wetland > footprint).any():
        raise AssertionError("a cell has more wetland pixels than pixel positions, footprint math is off")
    raw = src["label_semantics"]["raw_class_counts"]
    if int(n_developed.sum()) != raw[1] or int(n_other.sum()) != raw[2]:
        raise AssertionError("converted pixel totals do not match the source manifest's raw class counts")
    for a in out.values():
        a.flush()
    del out

    small = {
        "target": target, "n_wetland": n_wetland, "n_developed": n_developed,
        "n_other": n_other, "footprint_px": footprint, "block_code": cell_block,
        "cell_rowcol": cell_rowcol, "cell_xy": cell_xy,
        "dist_mean": dist_mean, "dist_min": dist_min,
    }
    for name, a in small.items():
        np.save(out_dir / f"{name}.npy", a)

    partial = footprint < n * n
    has_dev = n_developed > 0
    q = np.percentile(n_wetland, [0, 10, 25, 50, 75, 90, 100]).astype(int).tolist()
    stats = {
        "n_cells": n_cells,
        "n_partial_cells": int(partial.sum()),
        "n_cells_with_any_developed": int(has_dev.sum()),
        "share_cells_with_any_developed": float(has_dev.mean()),
        "mean_target_unweighted": float(target.mean()),
        "pixel_positive_rate": float(n_developed.sum() / n_wetland.sum()),
        "n_wetland_percentiles_0_10_25_50_75_90_100": q,
        "n_cells_one_wetland_pixel": int((n_wetland == 1).sum()),
    }

    emb_cols = src["emb_columns"]
    manifest = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "cell_size_m": args.cell_size_m,
        "cell_size_px": n,
        "source_array_dir": str(array_dir),
        "source_manifest": {k: src[k] for k in ("built_utc", "git_commit", "source_parquet_path", "total_rows")},
        "emb_columns": emb_cols,
        "feature_columns": [f"mean_{c}" for c in emb_cols] + [f"std_{c}" for c in emb_cols],
        "target": (
            "n_developed / n_wetland. Both count 2019 wetland pixels in the cell. "
            "n_developed is label_raw == 1. label_raw == 2 counts as not developed."
        ),
        "std_ddof": 0,
        "dist": (
            f"dist_mean and dist_min are the mean and min of the source {src['dist_column']} "
            "over the cell's wetland pixels. Raw meters, never normalized."
        ),
        "cell_grid": (
            "per block, anchored at the block's first pixel row and col, so cells "
            "never cross a block edge. block_code indexes source manifest block_ids."
        ),
        "footprint_px": "pixel positions the cell covers inside its block, wetland or not",
        "pixel_grid": {"x_off": x_off, "y_off": y_off, "pixel_m": PIXEL_M},
        "block_origin_window": {"u0": block_origin[0], "v0": block_origin[1], "block_m": BLOCK_M},
        "block_ids": src["block_ids"],
        "arrays": {
            "emb_mean": {"dtype": "float32", "shape": [n_cells, n_emb]},
            "emb_std": {"dtype": "float32", "shape": [n_cells, n_emb]},
            **{k: {"dtype": str(a.dtype), "shape": list(a.shape)} for k, a in small.items()},
        },
        "stats": stats,
        "build_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nwrote {out_dir} in {manifest['build_seconds'] / 60:.1f} min")
    print(f"cells {n_cells:,}  partial {stats['n_partial_cells']:,}  one wetland pixel {stats['n_cells_one_wetland_pixel']:,}")
    print(f"cells with any developed {stats['n_cells_with_any_developed']:,} ({stats['share_cells_with_any_developed']:.2%})")
    print(f"mean target {stats['mean_target_unweighted']:.4%}  pixel positive rate {stats['pixel_positive_rate']:.4%}")
    print(f"n_wetland percentiles 0/10/25/50/75/90/100: {q}")


if __name__ == "__main__":
    main()
