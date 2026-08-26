import numpy as np
import pandas as pd


def assign_balanced_spatial_folds(pool_path="data/datasets/data_pool.parquet", n_folds=5, seed=0):
    """Assign each block to one of n_folds spatial CV folds, balancing row count and
    label==1 count across folds instead of a plain random split.

    A pure random block assignment can still leave one fold with far more rows or far
    more positives than another purely by luck, since block size ranges from a handful
    of rows to over 100k and positives cluster unevenly across blocks. This runs a
    greedy longest-processing-time assignment instead: blocks go in order of positives
    first, row count as the tiebreak, each one landing in whichever fold is currently
    furthest behind its target share of rows and positives combined. seed only
    controls how ties among otherwise identical blocks get broken, so fold sizes come
    out close to balanced regardless of seed.

    Returns a block_id -> fold lookup table with the block's n_rows and n_label1
    alongside it, so you can check the per fold totals without a second pass over the
    pool.
    """
    df = pd.read_parquet(pool_path, columns=["block_id", "label"])
    df["is_pos"] = (df["label"] == 1).astype(np.int32)
    blocks = df.groupby("block_id").agg(n_rows=("is_pos", "size"), n_label1=("is_pos", "sum")).reset_index()

    # shuffle first so blocks tied on (n_label1, n_rows) break in a random order
    # rather than by block_id's arbitrary string sort
    blocks = blocks.sample(frac=1, random_state=seed).sort_values(
        ["n_label1", "n_rows"], ascending=[False, False], kind="stable"
    )

    target_rows = max(blocks["n_rows"].sum() / n_folds, 1)
    target_pos = max(blocks["n_label1"].sum() / n_folds, 1)

    fold_rows = np.zeros(n_folds)
    fold_pos = np.zeros(n_folds)
    assigned_fold = np.empty(len(blocks), dtype=int)

    for i, (n_rows, n_label1) in enumerate(zip(blocks["n_rows"], blocks["n_label1"])):
        # how far behind each fold is on its combined share of rows and positives,
        # normalized so the two resources pull on the greedy choice about equally
        # even though raw row counts run in the millions and positives in the
        # thousands. the block goes to whichever fold is furthest behind.
        deficit = (target_rows - fold_rows) / target_rows + (target_pos - fold_pos) / target_pos
        f = np.argmax(deficit)
        assigned_fold[i] = f
        fold_rows[f] += n_rows
        fold_pos[f] += n_label1

    blocks["fold"] = assigned_fold
    return blocks[["block_id", "fold", "n_rows", "n_label1"]].reset_index(drop=True)


def natural_pos_weight_mult(labels_path="data/wetland_sample_labels_2019_2024.parquet"):
    """n_neg / n_pos in the true statewide population, label==2 folded into 0 to
    match data_pool.parquet's binary scheme.

    This is the raw statewide imbalance, not a sample weight on its own. data_pool.parquet
    is already enriched for positives (block selection chases label==1 density), so
    applying this ratio directly to that pool double corrects and overshoots past the
    true rate. Feed it into pos_class_sample_weight along with the pool's own label
    column to get the weight that actually lands on the true statewide rate.
    """
    labels = pd.read_parquet(labels_path, columns=["label"])["label"]
    n_pos = (labels == 1).sum()
    n_neg = (labels != 1).sum()  # label==0 and label==2 both count as negative here
    return n_neg / n_pos


def pos_class_sample_weight(pool_labels, natural_pos_weight_mult):
    """Weight for label==1 rows that undoes the pool's own positive enrichment and
    lands the weighted class balance back on the true statewide rate (weight for
    everything else stays 1).

    Block selection for data_pool.parquet chases label==1 density, so the pool's own
    positive rate is already several times the true statewide rate. Weighting label==1
    rows by natural_pos_weight_mult directly (n_neg/n_pos in the true population) treats
    the pool as if it already matched Florida's real balance and pushes the rare class
    up again on top of that, which lands the weighted positive share around 80 percent
    instead of the true rate. This solves weight_c = true_proportion_c / pool_proportion_c
    for the positive class only, so the correction accounts for how enriched this
    particular pool is rather than reapplying the full population's raw imbalance.
    """
    pool_pos_rate = (pool_labels == 1).mean()
    true_pos_rate = 1 / (1 + natural_pos_weight_mult)
    return (true_pos_rate / (1 - true_pos_rate)) * ((1 - pool_pos_rate) / pool_pos_rate)
