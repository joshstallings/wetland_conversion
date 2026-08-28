"""
Spatial CV fold assignment (data_pipeline_recipe_path_b.md step 2). Fold
assignment is a runtime random draw from a seed, not a saved file, so which
blocks land in val changes every run unless the same seed is reused.
"""

import numpy as np
import pyarrow.dataset as ds


def assign_folds(block_row_counts, n_splits, seed):
    """
    block_row_counts: dict block_id (int) -> row count for that block, from
        population_stats.json (see population_stats.py).
    n_splits: number of folds.
    seed: fold assignment is a runtime draw from this seed, not a saved file --
        rerun with the same seed to reproduce a split, change it to get a new one.

    Shuffle the blocks, then use greedy size balancing: drop each one into whichever 
    fold currently has the fewest total rows.

    Returns dict block_id (int) -> fold_idx.
    """
    block_ids = np.array(list(block_row_counts.keys()), dtype=np.int64)
    row_counts = np.array([block_row_counts[int(b)] for b in block_ids], dtype=np.int64)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(block_ids))
    shuffled_ids = block_ids[order]
    shuffled_counts = row_counts[order]

    fold_row_totals = np.zeros(n_splits, dtype=np.int64)
    assignment = {}
    for bid, cnt in zip(shuffled_ids, shuffled_counts):
        fold = int(np.argmin(fold_row_totals))
        assignment[int(bid)] = fold
        fold_row_totals[fold] += int(cnt)

    return assignment


def log_fold_stats(source_parquet_path, fold_assignment, n_splits):
    """
    Prints per fold block count, row count, positive count, positive rate.
    Call this once per (seed, n_splits) right after assign_folds and before
    training -- it's the check for whether this seed happened to produce a
    degenerate split (e.g. a fold with almost no positives).

    Does its own narrow (label, block_id only) scan of the source parquet, same
    two columns and same cost as population_stats.py's step 1 scan -- just
    grouped by fold here. population_stats.json only carries the aggregate
    positive count, not a per-block breakdown, so this can't be read off that
    file alone.
    """
    dataset = ds.dataset(source_parquet_path, format="parquet")
    table = dataset.to_table(columns=["label", "block_id"])

    label = table.column("label").to_numpy(zero_copy_only=False)
    block_id = table.column("block_id").to_numpy(zero_copy_only=False).astype(np.int64)
    is_positive = label == 1

    fold_of_row = np.array(
        [fold_assignment.get(int(b), -1) for b in block_id], dtype=np.int64
    )
    n_unassigned = int(np.sum(fold_of_row == -1))
    if n_unassigned:
        print(f"WARNING: {n_unassigned} rows have a block_id missing from fold_assignment")

    block_count_per_fold = {f: 0 for f in range(n_splits)}
    for f in fold_assignment.values():
        block_count_per_fold[f] += 1

    print(f"{'fold':>4}  {'blocks':>8}  {'rows':>10}  {'positives':>10}  {'pos_rate':>9}")
    for fold in range(n_splits):
        in_fold = fold_of_row == fold
        n_rows = int(np.sum(in_fold))
        n_pos = int(np.sum(is_positive[in_fold]))
        pos_rate = n_pos / n_rows if n_rows else float("nan")
        print(f"{fold:>4}  {block_count_per_fold[fold]:>8}  {n_rows:>10}  {n_pos:>10}  {pos_rate:>9.4f}")
        if n_pos == 0:
            print(f"  WARNING: fold {fold} has zero positives -- degenerate split, consider a different seed.")
