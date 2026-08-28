"""
Dataset classes for the training pipeline.
everything streams straight from the source parquet, nothing is cached or
downsampled to disk. Fold assignment (folds.assign_folds) lives in its own
module since it's also used standalone for the pre-training fold log in
train.py.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

from folds import assign_folds
from label_utils import binarize_label
from population_stats import load_population_stats


def _read_feature_matrix(batch, feature_cols):
    """pyarrow RecordBatch -> float32 (n_rows, n_features) array, column order
    matching feature_cols."""
    return np.stack(
        [batch.column(c).to_numpy(zero_copy_only=False) for c in feature_cols],
        axis=1,
    ).astype(np.float32)


def _block_id_filter(block_ids):
    """pyarrow.dataset filter expression selecting rows whose block_id is in
    block_ids. Shared by the normalization pass and both streaming datasets so
    the predicate pushdown behavior is identical everywhere it's used."""
    return pc.field("block_id").isin(pa.array(np.asarray(block_ids, dtype=np.int64)))


def _to_tensors(x_row, y_row):
    return torch.from_numpy(x_row), torch.tensor(y_row, dtype=torch.float32)


def compute_normalization_stats(source_parquet_path, feature_cols, block_ids, cols_to_normalize,
                                 batch_size=65_536):
    """
    One streaming pass over just this fold's train blocks to get mean/std for
    cols_to_normalize (a subset of feature_cols -- see features.COLS_TO_NORMALIZE:
    dist_to_developed_2019_m is on a meters scale, the AlphaEarth embedding dims
    are already roughly unit scaled and don't need this). Rows with NaN/inf in
    any of cols_to_normalize (coastal/boundary tiles are the likely source) are
    dropped from the running stats rather than left to poison them.

    Always compute this from train rows only and reuse the result for val too
    (never recompute from val) -- val's own statistics would leak its
    distribution into the model's input scale.

    Returns (mean, std), both float32 arrays shaped (len(feature_cols),) so they
    line up with the full feature matrix. Columns not in cols_to_normalize get
    mean 0 / std 1, a no-op under (x - mean) / std.
    """
    dataset = ds.dataset(source_parquet_path, format="parquet")
    filt = _block_id_filter(block_ids)

    n_norm = len(cols_to_normalize)
    count = 0

    mean = np.zeros(n_norm, dtype=np.float64)
    m2 = np.zeros(n_norm, dtype=np.float64)

    for batch in dataset.to_batches(columns=cols_to_normalize, filter=filt, batch_size=batch_size):
        x = _read_feature_matrix(batch, cols_to_normalize).astype(np.float64)

        finite_mask = np.all(np.isfinite(x), axis=1)
        n_bad = int(finite_mask.size - finite_mask.sum())
        if n_bad:
            print(f"normalization pass: dropping {n_bad} rows with NaN/inf feature values")
        x = x[finite_mask]
        if len(x) == 0:
            continue

        batch_count = len(x)
        batch_mean = x.mean(axis=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)

        delta = batch_mean - mean
        total_count = count + batch_count
        mean = mean + delta * (batch_count / total_count)
        m2 = m2 + batch_m2 + delta**2 * count * batch_count / total_count
        count = total_count

    if count < 2:
        raise ValueError("fewer than 2 finite rows seen while computing normalization stats")

    std = np.sqrt(m2 / (count - 1))
    std[std == 0] = 1.0  # guard a constant column against divide by zero

    mean_full = np.zeros(len(feature_cols), dtype=np.float32)
    std_full = np.ones(len(feature_cols), dtype=np.float32)
    norm_idx = [feature_cols.index(c) for c in cols_to_normalize]
    mean_full[norm_idx] = mean.astype(np.float32)
    std_full[norm_idx] = std.astype(np.float32)
    return mean_full, std_full


class StreamingTrainDataset(IterableDataset):
    """
    Streams every row from the source parquet for this fold's train blocks.
    Positive rebalancing happens in the loss (models.SimpleLinearModel's pos_weight),
    not here, so batches are plain shuffled draws that average close to the true 0.28% 
    positive rate; don't expect every batch to contain a positive.

    Approximates a full shuffle with a reservoir-style shuffle buffer (the same
    trick tf.data's shuffle() and WebDataset use) instead of holding the whole
    fold in memory. Rows arrive from parquet roughly in spatial (tile) order, so
    buffer_size has to span more than one source file or batches stay spatially
    clustered. 
    """

    def __init__(self, source_parquet_path, feature_cols, label_col, train_block_ids,
                 mean, std, buffer_size=300_000):
        self.source_parquet_path = source_parquet_path
        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.train_block_ids = list(train_block_ids)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.buffer_size = buffer_size

    def __iter__(self):
        dataset = ds.dataset(self.source_parquet_path, format="parquet")
        filt = _block_id_filter(self.train_block_ids)
        columns = self.feature_cols + [self.label_col, "block_id"]

        rng = np.random.default_rng()
        buffer = []

        # loop through the dataset in batches, only looking at training blocks
        for batch in dataset.to_batches(columns=columns, filter=filt):
            x = (_read_feature_matrix(batch, self.feature_cols) - self.mean) / self.std
            raw_label = batch.column(self.label_col).to_numpy(zero_copy_only=False)
            y = binarize_label(raw_label)

            # then for every row, append to buffer if buffer size not yet reached. 
            for i in range(len(y)):
                row = (x[i], y[i])
                if len(buffer) < self.buffer_size:
                    buffer.append(row)
                else:
                    # if the buffer size is reached in this batch, start randomly
                    # pulling from the buffer
                    slot = int(rng.integers(0, self.buffer_size))
                    yield _to_tensors(*buffer[slot])
                    buffer[slot] = row
        # otherwise the entire batch fit into the buffer,
        # so shuffle it and start yielding values. 
        rng.shuffle(buffer)
        for row in buffer:
            yield _to_tensors(*row)


class StreamingValDataset(IterableDataset):
    """
    Validation stream at the source parquet's true 0.28% positive rate.
    never downsampled, never shuffle-buffered (order doesn't matter, this just
    accumulates predictions to score once). Normalized with the *train* fold's
    mean/std, never its own, or val's distribution leaks into the model's input
    scale. A separate class from StreamingTrainDataset even though both stream
    from the same source, since val never sees pos_weight or any other
    training-only concern.
    """

    def __init__(self, source_parquet_path, feature_cols, label_col, val_block_ids, mean, std):
        self.source_parquet_path = source_parquet_path
        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.val_block_ids = list(val_block_ids)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __iter__(self):
        dataset = ds.dataset(self.source_parquet_path, format="parquet")
        filt = _block_id_filter(self.val_block_ids)
        columns = self.feature_cols + [self.label_col, "block_id"]

        for batch in dataset.to_batches(columns=columns, filter=filt):
            x = (_read_feature_matrix(batch, self.feature_cols) - self.mean) / self.std
            raw_label = batch.column(self.label_col).to_numpy(zero_copy_only=False)
            y = binarize_label(raw_label)

            for i in range(len(y)):
                yield _to_tensors(x[i], y[i])


class SpatialCVDataModule(pl.LightningDataModule):
    """
    Train and val both stream from the source parquet, train through a shuffle
    buffer (StreamingTrainDataset), val at the natural rate
    (StreamingValDataset). No stratified batch sampler here, positive
    rebalancing happens in the loss (models.SimpleLinearModel's pos_weight)
    instead of the batch composition.

    cols_to_normalize is the subset of feature_cols that actually gets
    mean/std normalized (see features.COLS_TO_NORMALIZE); every other column
    passes through compute_normalization_stats unchanged.
    """

    def __init__(self, seed, n_splits, fold_idx, feature_cols, label_col, cols_to_normalize,
                 source_parquet_path, population_stats_path="data/population_stats.json",
                 batch_size=256, buffer_size=300_000):
        super().__init__()
        self.seed = seed
        self.n_splits = n_splits
        self.fold_idx = fold_idx
        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.cols_to_normalize = list(cols_to_normalize)
        self.source_parquet_path = source_parquet_path
        self.population_stats_path = population_stats_path
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def setup(self, stage=None):
        population_stats = load_population_stats(self.population_stats_path)

        self.fold_assignment = assign_folds(
            population_stats["block_row_counts"], self.n_splits, self.seed
        )
        all_blocks = np.array(list(self.fold_assignment.keys()), dtype=np.int64)
        fold_of_block = np.array(list(self.fold_assignment.values()), dtype=np.int64)
        self.val_block_ids = all_blocks[fold_of_block == self.fold_idx].tolist()
        self.train_block_ids = all_blocks[fold_of_block != self.fold_idx].tolist()

        self.mean, self.std = compute_normalization_stats(
            self.source_parquet_path, self.feature_cols, self.train_block_ids, self.cols_to_normalize
        )

        self.train_ds = StreamingTrainDataset(
            self.source_parquet_path, self.feature_cols, self.label_col,
            self.train_block_ids, self.mean, self.std, buffer_size=self.buffer_size,
        )
        self.val_ds = StreamingValDataset(
            self.source_parquet_path, self.feature_cols, self.label_col,
            self.val_block_ids, self.mean, self.std,
        )

    def train_dataloader(self):
        # IterableDataset doesn't support a sampler. num_workers=0: sharding an
        # IterableDataset across worker processes needs get_worker_info() bookkeeping
        # that isn't worth the complexity until an actual IO stall is measured.
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=0)
