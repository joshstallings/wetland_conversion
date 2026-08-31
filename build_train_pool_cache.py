"""
Step 1 of data_pipeline_recipe_path_a.md: builds the negative-undersampled
training pool once, offline, so path A's train.py never touches the full
59M row source parquet directly. Streams the source with
pyarrow.dataset.Dataset.to_batches() so nothing beyond one batch is ever
resident in memory during the scan.

Run once with `python population_stats.py` first to build
data/population_stats.json -- this script reuses its true population counts
rather than rescanning the label column itself. Then run this
(python build_train_pool_cache.py) to (re)build data/train_pool_cache/.
"""

import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from features import FEATURE_COLS, LABEL_COL
from label_utils import binarize_label
from population_stats import POPULATION_STATS_PATH, load_population_stats

SOURCE_PARQUET_PATH = "data/alphaearth_wetland_joined"
OUT_DIR = "data/train_pool_cache"

CACHE_BUILD_SEED = 1  # fixed and logged -- controls which negatives get discarded permanently
NEG_PER_POSITIVE = 10  # target ratio, revisit after seeing baseline precision/recall
BATCH_SIZE = 4096


def build_train_pool_cache(source_parquet_path, out_dir, population_stats_path=POPULATION_STATS_PATH,
                            neg_per_positive=NEG_PER_POSITIVE, cache_build_seed=CACHE_BUILD_SEED,
                            batch_size=BATCH_SIZE):
    population_stats = load_population_stats(population_stats_path)
    total_positive = population_stats["total_positive"]
    total_negative = population_stats["total_negative"]

    target_negatives = neg_per_positive * total_positive
    neg_keep_prob = min(1.0, target_negatives / total_negative)

    dataset = ds.dataset(source_parquet_path, format="parquet")
    rng = np.random.default_rng(cache_build_seed)

    feature_batches, y_batches, block_id_batches = [], [], []
    kept_pos = kept_neg_from_0 = kept_neg_from_2 = 0
    scanned_rows = 0

    columns = FEATURE_COLS + [LABEL_COL, "block_id"]
    print("Streaming full scan: keeping all positives, Bernoulli-subsampling negatives.")
    t0 = time.time()
    n_batches = 0
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        n_batches += 1
        table = pa.Table.from_batches([batch])

        raw_label = table.column(LABEL_COL).to_numpy(zero_copy_only=False)
        scanned_rows += len(raw_label)

        is_pos = raw_label == 1
        is_neg = ~is_pos
        # one Bernoulli draw per batch, not a fixed stride, so kept negatives
        # aren't spatially clustered by row-group order
        neg_draw = rng.random(len(raw_label)) < neg_keep_prob
        keep_mask = is_pos | (is_neg & neg_draw)
        if not np.any(keep_mask):
            continue

        kept_idx = np.nonzero(keep_mask)[0]
        raw_label_kept = raw_label[kept_idx]
        y_batch = binarize_label(raw_label_kept)

        feat_cols_np = [
            table.column(c).to_numpy(zero_copy_only=False)[kept_idx] for c in FEATURE_COLS
        ]
        feat_batch = np.stack(feat_cols_np, axis=1).astype(np.float32)

        block_id_batch = table.column("block_id").to_numpy(zero_copy_only=False)[kept_idx]
        block_id_batch = np.array([str(b) for b in block_id_batch], dtype=object)

        kept_neg_from_0 += int(np.sum(raw_label_kept == 0))
        kept_neg_from_2 += int(np.sum(raw_label_kept == 2))
        kept_pos += int(np.sum(raw_label_kept == 1))

        feature_batches.append(feat_batch)
        y_batches.append(y_batch)
        block_id_batches.append(block_id_batch)

        if n_batches % 500 == 0:
            print(f"..batch {n_batches}: scanned {scanned_rows:,} rows, {time.time() - t0:.0f}s elapsed")

    if scanned_rows != population_stats["total_rows"]:
        print(f"WARNING: scanned row count ({scanned_rows:,}) does not match "
              f"population_stats.json's total_rows ({population_stats['total_rows']:,})")

    all_block_ids = np.concatenate(block_id_batches) if block_id_batches else np.array([], dtype=object)
    unique_block_ids = sorted(set(all_block_ids.tolist()))
    block_id_to_idx = {bid: i for i, bid in enumerate(unique_block_ids)}
    block_id_int = np.array([block_id_to_idx[b] for b in all_block_ids.tolist()], dtype=np.int32)

    X = (np.concatenate(feature_batches, axis=0) if feature_batches
         else np.zeros((0, len(FEATURE_COLS)), dtype=np.float32))
    y = (np.concatenate(y_batches, axis=0).astype(np.int8) if y_batches
         else np.zeros((0,), dtype=np.int8))
    kept_total = len(y)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "block_id_int.npy", block_id_int)
    with open(out_dir / "block_ids.json", "w") as f:
        json.dump(unique_block_ids, f)

    manifest = {
        "source_parquet_path": str(source_parquet_path),
        "row_counts": {
            "total_scanned": scanned_rows,
            "total_kept": kept_total,
            "positives_kept": kept_pos,
            "negatives_kept": kept_total - kept_pos,
            "negatives_kept_from_label_0_wetland": kept_neg_from_0,
            "negatives_kept_from_label_2_other": kept_neg_from_2,
            "true_total_positive_population": total_positive,
            "true_total_negative_population": total_negative,
        },
        "neg_keep_prob": neg_keep_prob,
        "target_neg_per_pos_ratio": neg_per_positive,
        "cache_build_seed": cache_build_seed,
        "feature_cols": FEATURE_COLS,
        "build_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    achieved_ratio = (kept_total - kept_pos) / kept_pos if kept_pos else float("nan")
    print(f"positives kept: {kept_pos:,} (expect {total_positive:,} exactly)")
    print(f"kept negative:positive ratio = {achieved_ratio:.2f} (target {neg_per_positive})")
    print(f"wrote cache to {out_dir}/")

    if kept_pos != total_positive:
        print("WARNING: positives kept does not match the true population positive count -- "
              "either the negative Bernoulli draw is touching positives, or the label "
              "binarization is wrong. Chase this before training anything off the cache.")

    return manifest


def main():
    build_train_pool_cache(SOURCE_PARQUET_PATH, OUT_DIR)


if __name__ == "__main__":
    main()
