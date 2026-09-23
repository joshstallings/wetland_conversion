"""
Stitches the per fold out of fold predictions back onto the NLCD grid and writes
a GeoTIFF for Google Earth Engine.

Every pixel in the state appears in exactly one fold's validation set, scored by
the one model that did not train on its block, so the five val_preds.npz files
concatenate into a complete statewide map with no train set optimism in it.

The threshold is deliberately not baked in. Band 1 is the truth, band 2 is the
predicted probability, and the confusion categories get derived in Earth Engine,
so the operating point stays a slider instead of a re-upload. That matters here:
focal loss leaves the probabilities uncalibrated, and precision runs from 0.40 at
p >= 0.5 to 0.99 at p >= 0.9.

    python export_val_predictions.py --results results/tcn_2022_2024

Band encoding, both uint8 with 0 as nodata:

    label  1 = stayed wetland or went to something else, 2 = converted to developed
    prob   1 to 255 linear over [0, 1], so p = (v - 1) / 254

In Earth Engine, after uploading the tif to a GCS bucket and ingesting it:

    var img = ee.Image('users/you/val_predictions');
    var p = img.select('prob').subtract(1).divide(254);
    var y = img.select('label').eq(2);
    var t = 0.5;
    var cls = y.and(p.gte(t)).multiply(1)          // true positive
      .add(y.not().and(p.gte(t)).multiply(2))      // false positive
      .add(y.and(p.lt(t)).multiply(3));            // false negative
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import arrays as array_io
from folds import assign_folds
from population_stats import load_population_stats

# The AlphaEarth tiles and the NLCD rasters both carry Albers Conic Equal Area at
# lat_0 23, lon_0 -96, standard parallels 29.5 and 45.5, which is EPSG:5070. The
# AlphaEarth tifs declare a WGS84 datum against 5070's NAD83, a sub meter
# difference that does not matter on a 30 m grid.
CRS = "EPSG:5070"
PIXEL_M = 30

# 0 is reserved as nodata across both bands, so the label is stored one above its
# real value and the probability is quantized into 1 to 255 rather than 0 to 255.
NODATA = 0
PROB_LEVELS = 254


def grid_origin(rowcol, xy, n_sample=200_000, seed=0):
    """Upper left corner of the analysis window in Albers meters, recovered from
    the stored pixel centers.

    Worth deriving rather than hardcoding: the NLCD rasters the window was cut
    from are no longer on disk, so these two arrays are the only remaining record
    of where the grid sits.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(rowcol), min(n_sample, len(rowcol)), replace=False))
    r = rowcol[idx, 0].astype(np.float64)
    c = rowcol[idx, 1].astype(np.float64)
    x = xy[idx, 0].astype(np.float64)
    y = xy[idx, 1].astype(np.float64)

    ox = np.unique(x - (c + 0.5) * PIXEL_M)
    oy = np.unique(y + (r + 0.5) * PIXEL_M)
    if ox.size != 1 or oy.size != 1:
        raise AssertionError(
            f"rowcol and xy disagree on the grid origin: {ox.size} distinct x, "
            f"{oy.size} distinct y. They are not on one affine grid."
        )
    return float(ox[0]), float(oy[0])


def fold_rows(arrays, array_manifest, fold_code, fold_idx):
    """Global row indices for one fold's validation sweep, in the order the
    predictions were written.

    val_chunk_rows does not appear here on purpose. Chunking splits each block's
    span into reads but never reorders them, so the concatenation is the same for
    any chunk size and this does not have to match train.VAL_BATCH_ROWS.
    """
    index = array_io.build_fold_index(arrays, array_manifest, fold_code, fold_idx)
    return np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in index.val_chunks])


def stitch(results_dir, arrays, array_manifest, run_manifest):
    """Per row probability and label over the whole array, assembled from the
    five fold files. Returns (prob float32, label uint8, n_scored)."""
    pop = load_population_stats(run_manifest["population_stats_path"])
    fold_assignment = assign_folds(
        pop["block_row_counts"], run_manifest["n_splits"], run_manifest["seed"]
    )
    fold_code = array_io.fold_of_code(array_manifest, fold_assignment)

    n = array_manifest["total_rows"]
    prob = np.full(n, np.nan, dtype=np.float32)
    seen = np.zeros(n, dtype=bool)

    for fold_idx in range(run_manifest["n_splits"]):
        rows = fold_rows(arrays, array_manifest, fold_code, fold_idx)
        with np.load(results_dir / f"fold_{fold_idx}" / "val_preds.npz") as npz:
            probs, labels = npz["probs"], npz["labels"]

        if len(rows) != len(probs):
            raise AssertionError(
                f"fold {fold_idx}: rebuilt {len(rows):,} val rows against "
                f"{len(probs):,} saved predictions. The fold assignment does not "
                f"match the one the run used, check seed and n_splits."
            )
        # The labels ride along in the npz, so this compares the rebuilt row order
        # against the order the predictions were written in, element by element.
        # Any reordering or off by one dies here rather than showing up as a
        # plausible looking map.
        if not np.array_equal(arrays["label"][rows].astype(np.float32), labels):
            raise AssertionError(
                f"fold {fold_idx}: rebuilt rows do not carry the labels saved beside "
                f"the predictions. The row order is wrong."
            )
        if seen[rows].any():
            raise AssertionError(f"fold {fold_idx} overlaps an earlier fold's val rows")

        prob[rows] = probs
        seen[rows] = True
        print(f"  fold {fold_idx}: {len(rows):>12,} rows, {int(labels.sum()):>7,} positives")

    return prob, np.asarray(arrays["label"]), int(seen.sum())


def write_geotiff(out_path, rowcol, prob, label, origin, run_manifest, scored):
    """Two band uint8 GeoTIFF cropped to the wetland pixels' bounding box."""
    r0, r1 = int(rowcol[:, 0].min()), int(rowcol[:, 0].max())
    c0, c1 = int(rowcol[:, 1].min()), int(rowcol[:, 1].max())
    height, width = r1 - r0 + 1, c1 - c0 + 1
    ox, oy = origin
    transform = from_origin(ox + c0 * PIXEL_M, oy - r0 * PIXEL_M, PIXEL_M, PIXEL_M)

    rr = (rowcol[:, 0] - r0).astype(np.int32)
    cc = (rowcol[:, 1] - c0).astype(np.int32)

    band_label = np.zeros((height, width), dtype=np.uint8)
    band_label[rr, cc] = np.where(scored, label.astype(np.uint8) + 1, NODATA)

    band_prob = np.zeros((height, width), dtype=np.uint8)
    q = np.rint(np.nan_to_num(prob, nan=0.0) * PROB_LEVELS).astype(np.uint8) + 1
    band_prob[rr, cc] = np.where(scored, q, NODATA)

    profile = dict(
        driver="GTiff", height=height, width=width, count=2, dtype="uint8",
        crs=CRS, transform=transform, nodata=NODATA,
        tiled=True, blockxsize=512, blockysize=512,
        compress="deflate", predictor=2, zlevel=6, BIGTIFF="IF_SAFER",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(band_label, 1)
        dst.write(band_prob, 2)
        dst.set_band_description(1, "label")
        dst.set_band_description(2, "prob")
        # A GeoTIFF that says what produced it, since the decode is not guessable
        # and a map without its run is not reproducible.
        dst.update_tags(
            run_dir=str(run_manifest["results_dir"]),
            arrays_path=run_manifest["arrays"]["path"],
            source_parquet=run_manifest["arrays"]["source_parquet_path"],
            git_commit=run_manifest["arrays"]["git_commit"] or "",
            seed=str(run_manifest["seed"]),
            n_splits=str(run_manifest["n_splits"]),
            label_encoding="0 nodata, 1 not converted, 2 converted to developed",
            prob_encoding=f"0 nodata, 1 to 255 linear, p = (v - 1) / {PROB_LEVELS}",
        )
    return height, width


def report(prob, label, scored, thresholds):
    """Confusion counts at a few operating points. Printed only: nothing in the
    tif depends on a threshold."""
    p = prob[scored]
    y = label[scored].astype(bool)
    print(f"\n{len(p):,} scored pixels, {int(y.sum()):,} positives ({y.mean():.4%})")
    print(f"{'thresh':>8} {'TP':>9} {'FP':>10} {'FN':>9} {'precision':>10} {'recall':>8}")
    for t in thresholds:
        hit = p >= t
        tp, fp, fn = int((hit & y).sum()), int((hit & ~y).sum()), int((~hit & y).sum())
        print(f"{t:>8.2f} {tp:>9,} {fp:>10,} {fn:>9,} "
              f"{tp / max(tp + fp, 1):>10.3f} {tp / max(tp + fn, 1):>8.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results/tcn_2022_2024",
                    help="run directory holding manifest.json and fold_*/val_preds.npz")
    ap.add_argument("--arrays", default=None,
                    help="array directory. Defaults to the path in the run manifest")
    ap.add_argument("--population-stats", default=None,
                    help="defaults to the horizon implied by the run's source parquet")
    ap.add_argument("--out", default=None,
                    help="output tif. Defaults to <results>/val_predictions.tif")
    ap.add_argument("--thresholds", default="0.5,0.9,0.99",
                    help="operating points to print, comma separated")
    args = ap.parse_args()

    results_dir = Path(args.results)
    with open(results_dir / "manifest.json") as fh:
        run_manifest = json.load(fh)
    run_manifest["results_dir"] = str(results_dir)

    array_dir = Path(args.arrays or run_manifest["arrays"]["path"])
    if args.population_stats:
        stats_path = args.population_stats
    else:
        horizon = Path(run_manifest["arrays"]["source_parquet_path"]).name.split("joined_")[-1]
        stats_path = f"data/population_stats_{horizon}.json"
    run_manifest["population_stats_path"] = stats_path
    out_path = Path(args.out or results_dir / "val_predictions.tif")

    with open(array_dir / "manifest.json") as fh:
        array_manifest = json.load(fh)
    # Only the three small arrays are needed. load_arrays would mmap the 45 GB
    # embeddings and demand a matching column list for something never read here.
    arrays = {
        "label": np.load(array_dir / "label.npy"),
        "block_code": np.load(array_dir / "block_code.npy"),
    }
    rowcol = np.load(array_dir / "rowcol.npy", mmap_mode="r")
    xy = np.load(array_dir / "xy.npy", mmap_mode="r")

    print(f"run {results_dir}, arrays {array_dir}, stats {stats_path}")
    print(f"seed {run_manifest['seed']}, {run_manifest['n_splits']} folds")

    prob, label, n_scored = stitch(results_dir, arrays, array_manifest, run_manifest)
    scored = np.isfinite(prob)
    if n_scored != array_manifest["total_rows"]:
        raise AssertionError(
            f"{n_scored:,} of {array_manifest['total_rows']:,} rows were scored. The "
            f"folds do not partition the array."
        )

    origin = grid_origin(np.asarray(rowcol), np.asarray(xy))
    print(f"\ngrid origin {origin[0]:,.0f}, {origin[1]:,.0f} in {CRS} at {PIXEL_M} m")

    height, width = write_geotiff(
        out_path, np.asarray(rowcol), prob, label, origin, run_manifest, scored
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path}  {width:,} x {height:,} px, 2 bands uint8, {size_mb:,.0f} MB")

    report(prob, label, scored, [float(t) for t in args.thresholds.split(",")])

    print(f"\nupload:  gsutil cp {out_path} gs://<bucket>/")
    print(f"         earthengine upload image --asset_id=users/<you>/val_predictions \\\n"
          f"             --nodata_value={NODATA} gs://<bucket>/{out_path.name}")


if __name__ == "__main__":
    main()
