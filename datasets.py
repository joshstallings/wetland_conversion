"""
This class to contain datasets and dataloader classes. 
"""

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class StratifiedBatchSampler(Sampler):
    """
    This class ensures we get a certain number of positives per batch. 
    """

    def __init__(self, y, batch_size, pos_frac=0.25, n_batches=None):
        y = np.asarray(y)

        self.pos_idx = np.flatnonzero(y == 1)
        self.other_idx = np.flatnonzero(y == 0)

        self.n_pos = max(1, int(batch_size * pos_frac))
        self.n_other = batch_size - self.n_pos

        self.n_batches = n_batches or len(y) // batch_size

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self.n_batches):
            batch = np.concatenate([
                rng.choice(self.pos_idx, self.n_pos),
                rng.choice(self.other_idx, self.n_other, replace=len(self.other_idx) < self.n_other)
            ])

            rng.shuffle(batch)
            yield batch.tolist()
    
    def __len__(self):
        return self.n_batches


class ParquetBlockDataset(Dataset):
    def __init__(self, block_ids, feature_cols, label_col, parquet_path="data/alphaearth_wetland_joined"):
        self.dataset = ds.dataset(parquet_path, format="parquet")

        filter = pc.field("block_id").isin(pa.array(block_ids))
        # only materialize the rows matching this fold's blocks.
        table = self.dataset.to_table(filter=filter, columns=feature_cols + [label_col, "block_id"])
        self.X = table.select(feature_cols).to_pandas().to_numpy(dtype="float32")
        self.y = table.column(label_col).to_numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)

class SpatialCVDataModule(pl.LightningDataModule):
    def __init__(self, fold_assignments, fold_idx, feature_cols, label_col, parquet_path="data/alphaearth_wetland_joined", batch_size=256, pos_frac=0.25):
        super().__init__()
        self.parquet_path = parquet_path
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.batch_size = batch_size

        all_blocks = np.array(list(fold_assignments.keys()))
        folds = np.array(list(fold_assignments.values()))
        self.val_blocks = all_blocks[folds == fold_idx]
        self.train_blocks = all_blocks[folds != fold_idx]
        self.pos_frac = pos_frac

    def setup(self, stage=None):
        self.train_ds = ParquetBlockDataset(self.train_blocks, self.feature_cols, self.label_col, self.parquet_path)
        self.val_ds = ParquetBlockDataset(self.val_blocks, self.feature_cols, self.label_col, self.parquet_path)

        train_labels = self.train_ds.y
        self.train_batch_sampler = StratifiedBatchSampler(train_labels, self.batch_size, self.pos_frac)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_sampler=self.train_batch_sampler, shuffle=True, num_workers=8, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    
    
