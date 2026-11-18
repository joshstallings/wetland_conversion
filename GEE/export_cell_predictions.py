"""
Stitches the per fold out of fold predictions from a cell model run back onto
the 30 m NLCD grid and writes a GeoTIFF for Google Earth Engine.

    python GEE/export_cell_predictions.py --results results/hurdle_clf_mlp_300m_bce
    python GEE/export_cell_predictions.py --results results/hurdle_reg_mlp_300m_log_share

Every cell sits in exactly one fold's validation set, scored by the one model that
did not train on its block, so the five val_preds.npz files scatter into a
complete map with no train set optimism in it.

The raster is 30 m, not the cell size. build_cells.py restarts the cell grid at
each 10 km block, and 10 km is not a whole number of 300 m or 3 km cells, so each
block ends in a strip of partial cells and no coarse grid lines up with them all.
Cell edges always fall on the 30 m pixel grid, so each cell is painted as a solid
rectangle of 30 m pixels covering its whole footprint, wetland or not.

Classifier runs, uint8, 0 is nodata:

    label  1 = no 2019 wetland in the cell developed by 2024, 2 = some did
    prob   1 to 255 linear over [0, 1], so p = (v - 1) / 254
    fold   1 to 5, the fold whose model scored the cell

The threshold is not baked in. The run picked one per fold at recall 0.5 and the
script prints them as a JS object to paste into pred_visualization_clf.js.

Regression runs, float32, nodata -9999, only cells with n_developed > 0 since
that is the population the hurdle regressor was trained and scored on:

    actual       n_developed / n_wetland
    pred_median  exp(pred), clipped to 1
    pred_mean    exp(pred) times that fold's smear factor, clipped to 1
    fold         1 to 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

# The NLCD rasters and AlphaEarth tiles are Albers Conic Equal Area at lat_0 23,
# lon_0 -96, parallels 29.5 and 45.5, which is EPSG:5070.
CRS = "EPSG:5070"
PIXEL_M = 30
PROB_LEVELS = 254
UINT8_NODATA = 0
FLOAT_NODATA = -9999.0


def cell_geometry(cells_dir, cell_size_m):
    """Upper left pixel (row, col), footprint height and width in pixels, and the
    grid offsets x_off, y_off with x = x_off + 30 col, y = y_off minus 30 row.

    build_cells.py stores the upper left corner and the footprint center but not
    the footprint shape. The shape falls out of the two: the center sits (f - 1)/2
    pixels in from the corner. The offsets come from the full cells, whose shape
    is known, and get checked against every cell through footprint_px.
    """
    rowcol = np.load(cells_dir / "cell_rowcol.npy").astype(np.int64)
    xy = np.load(cells_dir / "cell_xy.npy")
    footprint = np.load(cells_dir / "footprint_px.npy")
    n = cell_size_m // PIXEL_M

    full = footprint == n * n
    x_off = np.unique(xy[full, 0] - PIXEL_M * (rowcol[full, 1] + (n - 1) / 2))
    y_off = np.unique(xy[full, 1] + PIXEL_M * (rowcol[full, 0] + (n - 1) / 2))
    if x_off.size != 1 or y_off.size != 1:
        raise AssertionError(f"full cells disagree on the grid offset: {x_off.size} x, {y_off.size} y")
    x_off, y_off = float(x_off[0]), float(y_off[0])

    fx = 2 * ((xy[:, 0] - x_off) / PIXEL_M - rowcol[:, 1]) + 1
    fy = 2 * ((y_off - xy[:, 1]) / PIXEL_M - rowcol[:, 0]) + 1
    fx_i, fy_i = np.rint(fx).astype(np.int64), np.rint(fy).astype(np.int64)
    if not (np.allclose(fx, fx_i) and np.allclose(fy, fy_i)):
        raise AssertionError("recovered footprint shapes are not whole pixels")
    if not np.array_equal(fx_i * fy_i, footprint):
        raise AssertionError("recovered footprint shapes do not multiply out to footprint_px")
    return rowcol, fy_i, fx_i, x_off, y_off


def paint_cell_ids(rowcol, fy, fx):
    """int32 raster of cell index, minus 1 where no cell. Cells never overlap
    since each block has its own cell grid and blocks do not overlap, but that is
    checked rather than assumed, because a silent overwrite would misplace values."""
    r0, c0 = int(rowcol[:, 0].min()), int(rowcol[:, 1].min())
    height = int((rowcol[:, 0] + fy).max()) - r0
    width = int((rowcol[:, 1] + fx).max()) - c0
    ids = np.full((height, width), -1, dtype=np.int32)
    rr, cc = rowcol[:, 0] - r0, rowcol[:, 1] - c0
    for i in range(len(rowcol)):
        ids[rr[i]:rr[i] + fy[i], cc[i]:cc[i] + fx[i]] = i
    painted = np.bincount(ids[ids >= 0], minlength=len(rowcol))
    if not np.array_equal(painted, fy * fx):
        raise AssertionError("some cells overlap, a later cell overwrote an earlier one")
    return ids, r0, c0


def stitch(results_dir, n_splits, keys):
    """Concatenate the fold files. Returns a dict of arrays plus fold (1 to 5)."""
    parts = {k: [] for k in keys + ["cell_idx", "fold"]}
    for k in range(n_splits):
        with np.load(results_dir / f"fold_{k}" / "val_preds.npz") as z:
            for key in keys + ["cell_idx"]:
                parts[key].append(z[key])
            parts["fold"].append(np.full(len(z["cell_idx"]), k + 1, dtype=np.uint8))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    if np.unique(out["cell_idx"]).size != out["cell_idx"].size:
        raise AssertionError("a cell appears in more than one fold's val set")
    return out


def build_classifier(results_dir, manifest, cells_dir, n_cells):
    s = stitch(results_dir, manifest["n_splits"], ["prob", "any"])
    if s["cell_idx"].size != n_cells:
        raise AssertionError(f"{s['cell_idx'].size:,} of {n_cells:,} cells scored, folds do not partition the cells")

    # The truth rides along in the npz. Checking it against the cells on disk
    # catches a cells directory rebuilt since the run.
    n_dev = np.load(cells_dir / "n_developed.npy")
    if not np.array_equal(n_dev[s["cell_idx"]] > 0, s["any"]):
        raise AssertionError("saved labels do not match n_developed > 0 in the cells directory")

    label = np.zeros(n_cells, np.uint8)
    prob = np.zeros(n_cells, np.uint8)
    fold = np.zeros(n_cells, np.uint8)
    label[s["cell_idx"]] = s["any"].astype(np.uint8) + 1
    prob[s["cell_idx"]] = np.rint(np.clip(s["prob"], 0, 1) * PROB_LEVELS).astype(np.uint8) + 1
    fold[s["cell_idx"]] = s["fold"]

    summary = pd.read_csv(results_dir / "fold_summary.csv").set_index("fold")
    thresholds = {k + 1: float(summary.loc[k, "threshold"]) for k in range(manifest["n_splits"])}

    y = s["any"]
    t = np.array([thresholds[f] for f in range(1, manifest["n_splits"] + 1)])[s["fold"] - 1]
    hit = s["prob"] >= t
    tp, fp = int((hit & y).sum()), int((hit & ~y).sum())
    fn, tn = int((~hit & y).sum()), int((~hit & ~y).sum())
    print(f"\n{y.size:,} cells, {int(y.sum()):,} positive ({y.mean():.2%})")
    print("pooled confusion at each fold's own threshold (recall 0.5 on that fold's train cells):")
    print(f"  TP {tp:>9,}   FP {fp:>9,}\n  FN {fn:>9,}   TN {tn:>9,}")
    print(f"  precision {tp / max(tp + fp, 1):.3f}, recall {tp / max(tp + fn, 1):.3f}")
    print("\npaste into pred_visualization_clf.js:")
    print("var FOLD_T = {" + ", ".join(f"{f}: {v:.4f}" for f, v in thresholds.items()) + "};")

    bands = {"label": label, "prob": prob, "fold": fold}
    return bands, "uint8", UINT8_NODATA, {
        "label_encoding": "0 nodata, 1 no conversion in cell, 2 any 2019 wetland developed by 2024",
        "prob_encoding": f"0 nodata, 1 to 255 linear, p = (v - 1) / {PROB_LEVELS}",
        "fold_encoding": "0 nodata, 1 to 5",
        "fold_thresholds": json.dumps(thresholds),
    }


def build_regression(results_dir, manifest, cells_dir, n_cells):
    s = stitch(results_dir, manifest["n_splits"], ["pred", "target"])
    target = np.load(cells_dir / "target.npy")
    if not np.allclose(target[s["cell_idx"]], s["target"]):
        raise AssertionError("saved targets do not match target.npy in the cells directory")
    n_pos = int((np.load(cells_dir / "n_developed.npy") > 0).sum())
    if s["cell_idx"].size != n_pos:
        raise AssertionError(f"{s['cell_idx'].size:,} cells scored against {n_pos:,} with any development")

    clip = float(manifest["back_transform"]["clip"])
    smear = np.array([manifest["folds"][str(k)]["smear"] for k in range(manifest["n_splits"])])
    median = np.minimum(np.exp(s["pred"].astype(np.float64)), clip)
    mean = np.minimum(np.exp(s["pred"].astype(np.float64)) * smear[s["fold"] - 1], clip)

    bands = {}
    for name, v in [("actual", s["target"]), ("pred_median", median), ("pred_mean", mean), ("fold", s["fold"])]:
        b = np.full(n_cells, FLOAT_NODATA, np.float32)
        b[s["cell_idx"]] = v
        bands[name] = b

    actual = s["target"].astype(np.float64)
    print(f"\n{actual.size:,} cells with any development")
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    print("share quantiles " + "  ".join(f"q{int(p * 100)}" for p in q))
    for name, v in [("actual", actual), ("pred_median", median), ("pred_mean", mean)]:
        print(f"  {name:<12}" + "  ".join(f"{x:.3f}" for x in np.quantile(v, q)))
    for name, v in [("pred_median", median), ("pred_mean", mean)]:
        r = np.log2(v / actual)
        print(f"  log2({name} / actual): median {np.median(r):+.2f}, "
              f"over predicted {(r > 0).mean():.1%}, within 2x {(np.abs(r) <= 1).mean():.1%}")

    return bands, "float32", FLOAT_NODATA, {
        "actual": "n_developed / n_wetland, 2019 wetland pixels in the cell",
        "pred_median": f"exp(pred) clipped to {clip}",
        "pred_mean": f"exp(pred) times fold smear clipped to {clip}, smear {smear.round(4).tolist()}",
        "fold_encoding": "1 to 5",
        "nodata": f"{FLOAT_NODATA}, also every cell with no development",
    }


def write_geotiff(out_path, ids, r0, c0, x_off, y_off, bands, dtype, nodata, tags):
    height, width = ids.shape
    transform = from_origin(x_off + PIXEL_M * c0 - PIXEL_M / 2,
                            y_off - PIXEL_M * r0 + PIXEL_M / 2, PIXEL_M, PIXEL_M)
    profile = dict(
        driver="GTiff", height=height, width=width, count=len(bands), dtype=dtype,
        crs=CRS, transform=transform, nodata=nodata,
        tiled=True, blockxsize=512, blockysize=512, compress="deflate",
        predictor=3 if dtype == "float32" else 2, zlevel=6, BIGTIFF="IF_SAFER",
    )
    valid = ids >= 0
    safe = np.where(valid, ids, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, (name, per_cell) in enumerate(bands.items(), start=1):
            band = np.where(valid, per_cell[safe], per_cell.dtype.type(nodata))
            dst.write(band, i)
            dst.set_band_description(i, name)
            del band
        dst.update_tags(**tags)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True, help="run directory with manifest.json and fold_*/val_preds.npz")
    ap.add_argument("--cells", default=None, help="cells directory, defaults to the one in the run manifest")
    ap.add_argument("--out", default=None, help="defaults to results/GEE_images/<run name>.tif")
    args = ap.parse_args()

    results_dir = Path(args.results)
    with open(results_dir / "manifest.json") as fh:
        manifest = json.load(fh)
    cells_dir = Path(args.cells or manifest["cells"]["path"])
    cell_size_m = int(manifest["cells"]["cell_size_m"])
    # Every run writes the same val_preds.npz layout, so the run name is what
    # tells the tifs apart once they share a folder. It doubles as the asset id.
    out_path = Path(args.out or results_dir.parent / "GEE_images" / f"{results_dir.name}.tif")
    is_clf = "Classifier" in manifest["model"]

    print(f"run {results_dir}  ({manifest['model']})")
    print(f"cells {cells_dir}, {cell_size_m} m, built {manifest['cells']['built_utc']}")

    rowcol, fy, fx, x_off, y_off = cell_geometry(cells_dir, cell_size_m)
    n_cells = len(rowcol)
    if n_cells != manifest["cells"]["n_cells"]:
        raise AssertionError(f"cells directory has {n_cells:,} cells, the run saw {manifest['cells']['n_cells']:,}")

    build = build_classifier if is_clf else build_regression
    bands, dtype, nodata, band_tags = build(results_dir, manifest, cells_dir, n_cells)

    ids, r0, c0 = paint_cell_ids(rowcol, fy, fx)
    tags = dict(
        run_dir=str(results_dir), model=manifest["model"], task=manifest["task"],
        cells_dir=str(cells_dir), cell_size_m=str(cell_size_m),
        cells_git_commit=manifest["cells"]["git_commit"],
        cells_built_utc=manifest["cells"]["built_utc"],
        source_parquet=manifest["cells"]["source_arrays"]["source_parquet_path"],
        seed=str(manifest["seed"]), n_splits=str(manifest["n_splits"]),
        band_names=",".join(bands), **band_tags,
    )
    write_geotiff(out_path, ids, r0, c0, x_off, y_off, bands, dtype, nodata, tags)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nwrote {out_path}  {ids.shape[1]:,} x {ids.shape[0]:,} px at {PIXEL_M} m, "
          f"{len(bands)} bands {dtype}, {size_mb:,.0f} MB")
    print(f"\nupload:  gsutil cp {out_path} gs://<bucket>/")
    print(f"         earthengine upload image --asset_id=projects/<project>/assets/{out_path.stem} \\\n"
          f"             --nodata_value={nodata:g} gs://<bucket>/{out_path.name}")


if __name__ == "__main__":
    main()
