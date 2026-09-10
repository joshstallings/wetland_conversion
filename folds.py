"""
Spatial CV fold assignment (data_pipeline_recipe.md step 2). Fold
assignment is a runtime random draw from a seed, not a saved file, so which
blocks land in val changes every run unless the same seed is reused.
"""

import numpy as np


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
