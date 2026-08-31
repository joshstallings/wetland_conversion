"""
Path A dataset classes (data_pipeline_recipe_path_a.md): train reads from the
negative-undersampled cache in data/train_pool_cache/ (built once, offline, by
build_train_pool_cache.py) instead of streaming from the source parquet. Val
still streams from the source parquet at the natural rate, identical to path
B, so StreamingValDataset is imported from datasets.py rather than duplicated
here.
"""

import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from datasets import StreamingValDataset
from folds import assign_folds
from population_stats import load_population_stats


class StratifiedBatchSampler(Sampler):
    """
    Draws pos_frac of every batch from positives, the rest from negatives, so
    every batch sees positives regardless of the pool's true ratio. This is
    path A's positive rebalancing -- path B does it in the loss (pos_weight)
    instead, since it has no fixed-size pool to draw a stratified batch from.
    """

    def __init__(self, y, batch_size, pos_frac=0.25, n_batches=None):
        y = np.asarray(y)
        self.pos_idx = np.flatnonzero(y == 1)
        self.neg_idx = np.flatnonzero(y == 0)
        self.n_pos = max(1, int(batch_size * pos_frac))
        self.n_neg = batch_size - self.n_pos
        self.n_batches = n_batches or len(y) // batch_size

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self.n_batches):
            batch = np.concatenate([
                rng.choice(self.pos_idx, self.n_pos),
                rng.choice(self.neg_idx, self.n_neg, replace=len(self.neg_idx) < self.n_neg),
            ])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.n_batches


def compute_normalization_stats_from_cache(X, row_idx, feature_cols, cols_to_normalize):
    """
    Direct mean/std over this fold's training rows, already sitting in the
    memmapped cache -- no streaming pass needed here, unlike path B, since
    the pool is a fixed array X can already index into. Only normalizes
    cols_to_normalize (features.COLS_TO_NORMALIZE, just
    dist_to_developed_2019_m), matching path B, so the two paths' inputs are
    on the same footing. Rows with NaN/inf in that column (coastal/boundary
    tiles are the likely source) are dropped from the stats rather than left
    to poison them.

    Returns (mean, std), full-length float32 arrays shaped (len(feature_cols),),
    with mean 0 / std 1 (a no-op under (x - mean) / std) on every column not
    in cols_to_normalize.
    """
    norm_idx = [feature_cols.index(c) for c in cols_to_normalize]
    x = np.asarray(X[row_idx][:, norm_idx], dtype=np.float64)

    finite_mask = np.all(np.isfinite(x), axis=1)
    n_bad = int(finite_mask.size - finite_mask.sum())
    if n_bad:
        print(f"normalization pass: dropping {n_bad} rows with NaN/inf feature values")
    x = x[finite_mask]

    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0, ddof=1)
    std[std == 0] = 1.0  # guard a constant column against divide by zero

    mean_full = np.zeros(len(feature_cols), dtype=np.float32)
    std_full = np.ones(len(feature_cols), dtype=np.float32)
    mean_full[norm_idx] = mean.astype(np.float32)
    std_full[norm_idx] = std.astype(np.float32)
    return mean_full, std_full


class ParquetBlockDataset(Dataset):
    """
    Map-style dataset over the cached pool (data/train_pool_cache/), not the
    source parquet -- despite the inherited name, it reads
    X.npy/y.npy/block_id_int.npy rather than scanning parquet.
    """

    def __init__(self, cache_dir, split_block_ids, feature_cols, cols_to_normalize, mean=None, std=None):
        """
        cache_dir: data/train_pool_cache/ from build_train_pool_cache.py.
        split_block_ids: block_id strings belonging to this split (train
            blocks for this fold, from assign_folds).
        feature_cols, cols_to_normalize: features.FEATURE_COLS /
            features.COLS_TO_NORMALIZE.
        mean, std: leave None for the train split -- this constructor then
            computes them from this split's own rows
            (compute_normalization_stats_from_cache). For any other split,
            always pass the train split's stats; normalization must come
            from training rows only, or val's distribution leaks into the
            model's input scale.
        """
        cache_dir = Path(cache_dir)
        self.X = np.load(cache_dir / "X.npy", mmap_mode="r")
        self.y = np.load(cache_dir / "y.npy")
        block_id_int = np.load(cache_dir / "block_id_int.npy")

        with open(cache_dir / "block_ids.json") as f:
            block_id_vocab = json.load(f)  # index == int code used in block_id_int.npy
        block_id_to_int = {b: i for i, b in enumerate(block_id_vocab)}

        # A block can legitimately have zero rows in the cache -- every one of
        # its rows was a negative dropped by the undersample draw, most likely
        # for the smallest blocks (min block size is 1 row). Skip those rather
        # than raising; they contribute nothing to row_idx either way.
        known_block_ids = [b for b in split_block_ids if b in block_id_to_int]
        n_missing = len(split_block_ids) - len(known_block_ids)
        if n_missing:
            print(f"{n_missing} of {len(split_block_ids)} split block_ids have zero rows in "
                  f"the cache, skipping them")
        split_block_ints = np.array([block_id_to_int[b] for b in known_block_ids], dtype=np.int32)
        self.row_idx = np.flatnonzero(np.isin(block_id_int, split_block_ints))

        if mean is None or std is None:
            mean, std = compute_normalization_stats_from_cache(
                self.X, self.row_idx, feature_cols, cols_to_normalize
            )
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __len__(self):
        return len(self.row_idx)

    def __getitem__(self, i):
        row = self.row_idx[i]
        x = (np.asarray(self.X[row], dtype=np.float32) - self.mean) / self.std
        return torch.from_numpy(x), torch.tensor(self.y[row], dtype=torch.float32)


class CachedPoolDataModule(pl.LightningDataModule):
    """
    Train reads from the cached, undersampled pool (ParquetBlockDataset) with
    a StratifiedBatchSampler; val streams from the source parquet at the
    natural rate (datasets.StreamingValDataset), identical to path B. Fold
    assignment reuses folds.assign_folds with population_stats.json's
    block_row_counts, same as path B, so a shared seed produces the same
    split under both.
    """

    def __init__(self, seed, n_splits, fold_idx, feature_cols, label_col, cols_to_normalize,
                 cache_dir, source_parquet_path, population_stats_path="data/population_stats.json",
                 batch_size=256, pos_frac=0.25):
        super().__init__()
        self.seed = seed
        self.n_splits = n_splits
        self.fold_idx = fold_idx
        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.cols_to_normalize = list(cols_to_normalize)
        self.cache_dir = cache_dir
        self.source_parquet_path = source_parquet_path
        self.population_stats_path = population_stats_path
        self.batch_size = batch_size
        self.pos_frac = pos_frac

    def setup(self, stage=None):
        population_stats = load_population_stats(self.population_stats_path)

        self.fold_assignment = assign_folds(
            population_stats["block_row_counts"], self.n_splits, self.seed
        )
        all_blocks = np.array(list(self.fold_assignment.keys()), dtype=object)
        fold_of_block = np.array(list(self.fold_assignment.values()), dtype=np.int64)
        self.val_block_ids = all_blocks[fold_of_block == self.fold_idx].tolist()
        self.train_block_ids = all_blocks[fold_of_block != self.fold_idx].tolist()

        self.train_ds = ParquetBlockDataset(
            self.cache_dir, self.train_block_ids, self.feature_cols, self.cols_to_normalize,
        )
        self.mean, self.std = self.train_ds.mean, self.train_ds.std

        self.val_ds = StreamingValDataset(
            self.source_parquet_path, self.feature_cols, self.label_col,
            self.val_block_ids, self.mean, self.std,
        )
        train_y = self.train_ds.y[self.train_ds.row_idx]
        self.train_batch_sampler = StratifiedBatchSampler(train_y, self.batch_size, self.pos_frac)

    def train_dataloader(self):
        # __getitem__ here is a single memmap row slice, cheap enough that
        # multiprocessing overhead likely isn't worth it -- start at
        # num_workers=0, only raise it if you profile actual IO stall.
        return DataLoader(self.train_ds, batch_sampler=self.train_batch_sampler,
                           num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=0)
