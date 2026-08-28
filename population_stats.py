"""
Scans the source parquet data for true population class counts and per-block_id row counts.
models.SimpleLinearModel's pos_weight and fold.assign_fold's both need these.

Reads only label and block_id, skipping the AE embedding cols. 
Run directly (python population_stats.py) to (re)build data/population_stats.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from label_utils import binarize_label

SOURCE_PARQUET_PATH = "data/alphaearth_wetland_joined"
POPULATION_STATS_PATH = "data/population_stats.json"

# A mismatch here means the label binarization is wrong before anything else.
EXPECTED_TOTAL_POSITIVE = 165_908
EXPECTED_TOTAL_NEGATIVE = 59_101_423


def compute_population_stats(source_parquet_path):
    """
    Returns a dict with total_rows, total_positive, total_negative, and
    block_row_counts (dict block_id (int) -> row count, what folds.assign_folds
    needs for its greedy balancer).
    """
    dataset = ds.dataset(source_parquet_path, format="parquet")
    table = dataset.to_table(columns=["label", "block_id"])

    raw_label = table.column("label").to_numpy(zero_copy_only=False)
    block_id = table.column("block_id").to_numpy(zero_copy_only=False).astype(np.int64)

    y = binarize_label(raw_label)
    total_rows = len(raw_label)
    total_positive = int(np.sum(y == 1))
    total_negative = total_rows - total_positive

    unique_blocks, counts = np.unique(block_id, return_counts=True)
    block_row_counts = {int(b): int(c) for b, c in zip(unique_blocks, counts)}

    return {
        "source_parquet_path": str(source_parquet_path),
        "total_rows": total_rows,
        "total_positive": total_positive,
        "total_negative": total_negative,
        "block_row_counts": block_row_counts,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def save_population_stats(stats, out_path):
    """JSON only allows string dict keys, so block_row_counts gets stringified
    on write; load_population_stats casts it back to int on read."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = dict(stats)
    serializable["block_row_counts"] = {
        str(b): c for b, c in stats["block_row_counts"].items()
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_population_stats(path=POPULATION_STATS_PATH):
    """Loads population_stats.json, restoring block_row_counts keys to int."""
    with open(path) as f:
        stats = json.load(f)
    stats["block_row_counts"] = {int(k): int(v) for k, v in stats["block_row_counts"].items()}
    return stats


def main():
    stats = compute_population_stats(SOURCE_PARQUET_PATH)
    save_population_stats(stats, POPULATION_STATS_PATH)

    print(f"total rows: {stats['total_rows']:,}")
    print(f"total positive (label==1, converted to developed): {stats['total_positive']:,}")
    print(f"total negative (label==0 or 2): {stats['total_negative']:,}")
    print(f"unique blocks: {len(stats['block_row_counts']):,}")

    if (stats["total_positive"], stats["total_negative"]) != (EXPECTED_TOTAL_POSITIVE, EXPECTED_TOTAL_NEGATIVE):
        print(
            f"WARNING: counts do not match the expected "
            f"{EXPECTED_TOTAL_POSITIVE:,} positive / {EXPECTED_TOTAL_NEGATIVE:,} negative "
            f"from the recipe. Check the label binarization before trusting anything "
            f"downstream of this file."
        )


if __name__ == "__main__":
    main()
