"""
Spatial CV fold assignment (data_pipeline_recipe.md step 2). Fold
assignment is a runtime random draw from a seed, not a saved file, so which
blocks land in val changes every run unless the same seed is reused.

assign_geographic_folds is the other option: two folds, Florida cut in half
north and south or east and west at SPLIT_CENTER. No seed involved, the same
blocks land on the same side every time.
"""

import numpy as np
from pyproj import Transformer

# Where the geographic splits cut, as (lat, lon): the 2019 wetland pixel median,
# so each half holds 50.0% of the wetland. Latitude is the north and south line,
# longitude the east and west one. Computed once off data/cells_2019_2024_300m
# (block square centers weighted by n_wetland, midpoint between the median block
# and its neighbor so no block center sits on the line) and pinned here rather
# than recomputed, since a 2022 base year artifact has different wetland and
# would move the line between horizons.
#
# This replaced the center of Florida's land area, 28°40′53″N 82°27′36″W. The
# panhandle drags that point west, so its east and west line put 68.5% of the
# wetland and 85.8% of the conversions in the east.
SPLIT_CENTER = (28.2552, -81.6607)
SPLIT_CENTER_DMS = "28°15′19″N, 81°39′39″W"

BLOCK_M = 10_000

# Fold 0 is the first name, fold 1 the second.
GEOGRAPHIC_FOLD_NAMES = {
    "north_south": ("north", "south"),
    "east_west": ("east", "west"),
}


def block_centers_latlon(block_ids, block_origin):
    """
    Center of each block's full 10 km square in degrees, as (lat, lon) arrays in
    block_ids order. block_origin is (u0, v0) from a cell manifest's
    block_origin_window. The squares are on the EPSG:5070 grid, where block col c
    covers u0 + 10000 c <= x < u0 + 10000 (c + 1) and block row r covers
    v0 minus 10000 r <= y < v0 minus 10000 r + 10000, so rows count southward.
    """
    u0, v0 = block_origin
    brow = np.array([int(b[1:5]) for b in block_ids], dtype=np.int64)
    bcol = np.array([int(b[6:]) for b in block_ids], dtype=np.int64)
    x = u0 + BLOCK_M * bcol + BLOCK_M / 2
    y = v0 - BLOCK_M * brow + BLOCK_M / 2
    lon, lat = Transformer.from_crs(5070, 4326, always_xy=True).transform(x, y)
    return np.asarray(lat), np.asarray(lon)


def assign_geographic_folds(block_ids, block_origin, scheme, center=SPLIT_CENTER):
    """
    Two folds split at center, by which side of it each block's square center
    falls on. Fold 0 is north (or east), fold 1 south (or west), per
    GEOGRAPHIC_FOLD_NAMES. A block the line runs through goes whole to the side
    its center is on.

    Compared in latitude and longitude, not EPSG:5070 y and x. Albers grid north
    is about 8° off true north over Florida, since the state sits 14° east of the
    projection's central meridian, so a constant y line through the center would
    be a tilted cut.

    Returns dict block_id (str) -> fold_idx, the same shape as assign_folds, so
    arrays.fold_of_code takes it unchanged.
    """
    if scheme not in GEOGRAPHIC_FOLD_NAMES:
        raise ValueError(f"scheme has to be one of {list(GEOGRAPHIC_FOLD_NAMES)}, got {scheme!r}")
    lat, lon = block_centers_latlon(block_ids, block_origin)
    if scheme == "north_south":
        first = lat >= center[0]
    else:
        first = lon >= center[1]
    return {str(b): 0 if f else 1 for b, f in zip(block_ids, first)}


def assign_folds(block_row_counts, n_splits, seed):
    """
    block_row_counts: dict block_id (str, e.g. "b0239_0323") -> row count for
        that block, from population_stats.json (see population_stats.py).
    n_splits: number of folds.
    seed: fold assignment is a runtime draw from this seed, not a saved file --
        rerun with the same seed to reproduce a split, change it to get a new one.

    Shuffle the blocks, then use greedy size balancing: drop each one into whichever
    fold currently has the fewest total rows.

    Returns dict block_id (str) -> fold_idx.
    """
    block_ids = np.array(list(block_row_counts.keys()), dtype=object)
    row_counts = np.array([block_row_counts[b] for b in block_ids], dtype=np.int64)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(block_ids))
    shuffled_ids = block_ids[order]
    shuffled_counts = row_counts[order]

    fold_row_totals = np.zeros(n_splits, dtype=np.int64)
    assignment = {}
    for bid, cnt in zip(shuffled_ids, shuffled_counts):
        fold = int(np.argmin(fold_row_totals))
        assignment[bid] = fold
        fold_row_totals[fold] += int(cnt)

    return assignment


def log_fold_stats(label, row_fold, fold_assignment, n_splits):
    """
    Prints per fold block count, row count, positive count and positive rate.
    Call this once per (seed, n_splits) right after assign_folds and before
    training: it is the check for whether this seed happened to produce a
    degenerate split, e.g. a fold with almost no positives.

    label: the binary uint8 label array from data/arrays. It is binarized at write
        time, so nothing here calls binarize_label.
    row_fold: per row fold index, from arrays.fold_of_row.

    This used to do its own two column scan of the source parquet, about a minute
    of IO. Off the arrays it is two bincounts over 59M elements.
    """
    row_fold = np.asarray(row_fold)
    n_unassigned = int(np.sum(row_fold < 0))
    if n_unassigned:
        print(f"WARNING: {n_unassigned:,} rows have no fold assigned")

    rows_per_fold = np.bincount(row_fold[row_fold >= 0], minlength=n_splits)
    # Masking first rather than passing weights=label: the mask touches 165,908
    # rows instead of promoting all 59M labels to float64.
    pos_per_fold = np.bincount(row_fold[label == 1], minlength=n_splits)

    blocks_per_fold = np.bincount(
        np.fromiter(fold_assignment.values(), dtype=np.int64, count=len(fold_assignment)),
        minlength=n_splits,
    )

    print(f"{'fold':>4}  {'blocks':>8}  {'rows':>12}  {'positives':>10}  {'pos_rate':>9}")
    for fold in range(n_splits):
        n_rows = int(rows_per_fold[fold])
        n_pos = int(pos_per_fold[fold])
        pos_rate = n_pos / n_rows if n_rows else float("nan")
        print(f"{fold:>4}  {int(blocks_per_fold[fold]):>8}  {n_rows:>12,}  {n_pos:>10,}  {pos_rate:>9.4%}")
        if n_pos == 0:
            print(f"  WARNING: fold {fold} has zero positives, degenerate split, use a different seed.")
