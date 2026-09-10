"""
Datasets, batch samplers and the DataModule for the array pipeline. Everything
reads from data/arrays through arrays.py; nothing here touches the source
parquet.

"""

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, get_worker_info

import arrays as array_io


def dist_norm_stats(arrays, is_train):
    """
    Mean and std of dist_to_developed_2019_m over this fold's train rows.

    Train rows only, always, and the same numbers get reused for val. Computing it
    from val too would leak val's distribution into the model's input scale. This
    is also why the array artifact stores dist in raw meters: normalization stats
    are a fold dependent quantity, so baking them into the file would freeze one
    fold's answer into an artifact shared by all five.

    Replaces the streaming path's filtered parquet scan with two reductions over
    one 237 MB array. Rows that are not finite are excluded and the count is
    printed rather than silently dropped, which is what the old per batch version
    did. build_arrays.py measured zero of them across the whole array, so this
    warning should never fire.
    """
    dist = arrays["dist"]
    finite = np.isfinite(dist)
    n_bad = int(np.sum(is_train & ~finite))
    if n_bad:
        print(f"WARNING: {n_bad:,} train rows have a dist that is not finite, excluded from mean/std")

    mask = is_train & finite
    n = int(mask.sum())
    if n < 2:
        raise ValueError(f"only {n} finite train rows for the dist normalization stats")

    mean = float(np.mean(dist, where=mask, dtype=np.float64))
    std = float(np.std(dist, where=mask, dtype=np.float64))
    if std == 0:
        std = 1.0  # guard a constant column against divide by zero
    return np.float32(mean), np.float32(std)


class GatherDataset:
    """
    Random access rows by index array. __getitem__ takes a whole batch of row
    indices, not one index.

    Returns (emb float16 (B, 192), dist float32 (B,), label float32 (B,)).
    """

    def __init__(self, arrays, dist_mean, dist_std):
        self.emb = arrays["emb"]
        self.dist = arrays["dist"]
        self.label = arrays["label"]
        self.dist_mean = np.float32(dist_mean)
        self.dist_std = np.float32(dist_std)

    def __len__(self):
        return self.emb.shape[0]

    def __getitem__(self, idx):
        emb = self.emb[idx]
        dist = (self.dist[idx] - self.dist_mean) / self.dist_std
        y = self.label[idx].astype(np.float32)
        return torch.from_numpy(emb), torch.from_numpy(dist), torch.from_numpy(y)


class SpanDataset:
    """
    Sequential reads over (start, stop) row spans, for validation and for scoring.
    One item is one span, already batch sized.

    Validation gets its own class because it reads slices rather than gathering.
    The array is block contiguous, so a fold's val rows are a few hundred
    contiguous spans, and a slice of the memmap is one sequential read instead of
    8,192 scattered ones. arrays.build_fold_index chops the spans into chunks.

    np.array() rather than a bare slice because np.load(mmap_mode="r") hands back
    a read only array.
    """

    def __init__(self, arrays, chunks, dist_mean, dist_std):
        self.emb = arrays["emb"]
        self.dist = arrays["dist"]
        self.label = arrays["label"]
        self.chunks = np.asarray(chunks, dtype=np.int64).reshape(-1, 2)
        self.dist_mean = np.float32(dist_mean)
        self.dist_std = np.float32(dist_std)

    def __len__(self):
        return len(self.chunks)

    @property
    def n_rows(self):
        return int((self.chunks[:, 1] - self.chunks[:, 0]).sum())

    def __getitem__(self, i):
        start, stop = self.chunks[i]
        emb = np.array(self.emb[start:stop])
        dist = (self.dist[start:stop] - self.dist_mean) / self.dist_std
        y = self.label[start:stop].astype(np.float32)
        return torch.from_numpy(emb), torch.from_numpy(dist), torch.from_numpy(y)


class StratifiedBatchSampler:
    """
    Composes every batch with a fixed positive to negative ratio, drawing from
    this fold's train rows. Yields int32 index arrays, one per batch.

    Positives are drawn without replacement within an epoch and reshuffled if
    steps_per_epoch asks for more than one pass. Negatives are drawn with
    replacement: at 1,984 per batch out of 47M, the expected number of duplicates
    in a batch is 0.04, and permuting a 47M element array every epoch to avoid
    that is not worth it.
    """

    def __init__(self, train_pos, train_neg, batch_size=2048, pos_per_batch=64,
                 steps_per_epoch=None, seed=0):
        if not 0 < pos_per_batch < batch_size:
            raise ValueError(f"pos_per_batch {pos_per_batch} must be in (0, {batch_size})")
        self.train_pos = train_pos
        self.train_neg = train_neg
        self.batch_size = batch_size
        self.pos_per_batch = pos_per_batch
        self.neg_per_batch = batch_size - pos_per_batch
        self.seed = seed
        self.steps_per_epoch = (
            steps_per_epoch if steps_per_epoch is not None
            else max(train_pos.size // pos_per_batch, 1)
        )
        self._epoch = 0

    @property
    def batch_pos_weight(self):
        """Residual negative to positive ratio still present in a batch after
        stratifying. Not wired into the loss by default, since focal loss is doing
        that job, but it is the right value if you ever pass one."""
        return self.neg_per_batch / self.pos_per_batch

    @property
    def batch_positive_rate(self):
        return self.pos_per_batch / self.batch_size

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        # Seeded off (seed, epoch) so every epoch draws differently and the whole
        # run still reproduces from SEED alone. Lightning calls __iter__ once per
        # epoch, which is what makes the counter safe.
        rng = np.random.default_rng([self.seed, self._epoch])
        self._epoch += 1

        pos_order = rng.permutation(self.train_pos.size)
        cursor = 0
        for _ in range(self.steps_per_epoch):
            if cursor + self.pos_per_batch > pos_order.size:
                pos_order = rng.permutation(self.train_pos.size)
                cursor = 0
            pos = self.train_pos[pos_order[cursor:cursor + self.pos_per_batch]]
            cursor += self.pos_per_batch

            neg = self.train_neg[rng.integers(0, self.train_neg.size, self.neg_per_batch)]

            batch = np.concatenate([pos, neg])
            rng.shuffle(batch)
            yield batch


class SequentialIndexSampler:
    """
    Walks a fixed row index array in order, batch_size at a time. For scoring a
    fitted model over a chosen set of rows, not for training.
    """

    def __init__(self, row_idx, batch_size):
        self.row_idx = np.asarray(row_idx, dtype=np.int32)
        self.batch_size = batch_size

    def __len__(self):
        return (self.row_idx.size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for start in range(0, self.row_idx.size, self.batch_size):
            yield self.row_idx[start:start + self.batch_size]


def _populate_worker(worker_id):
    info = get_worker_info()
    emb = getattr(info.dataset, "emb", None)
    if emb is not None and isinstance(emb, np.memmap):
        emb.reshape(-1).view(np.uint8)[::array_io.PAGE_BYTES].sum()


class ArrayDataModule(pl.LightningDataModule):
    """
    One fold's loaders over the array artifact.
    """

    def __init__(self, arrays, manifest, fold_code, fold_idx, batch_size=2048,
                 pos_per_batch=64, steps_per_epoch=None, val_batch_rows=8192,
                 train_eval_stride=10, num_workers=0, seed=0):
        super().__init__()
        self.arrays = arrays
        self.manifest = manifest
        self.fold_code = fold_code
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.pos_per_batch = pos_per_batch
        self.steps_per_epoch = steps_per_epoch
        self.val_batch_rows = val_batch_rows
        self.train_eval_stride = train_eval_stride
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        # Everything is built for every stage at once, so this is guarded rather
        # than rebuilt per stage: train.py calls setup() itself to read
        # batch_pos_weight before constructing the model, and Trainer.fit would
        # otherwise redo the 0.7 second index build immediately afterwards.
        if getattr(self, "index", None) is not None:
            return

        self.index = array_io.build_fold_index(
            self.arrays, self.manifest, self.fold_code, self.fold_idx,
            val_chunk_rows=self.val_batch_rows,
        )
        self.dist_mean, self.dist_std = dist_norm_stats(self.arrays, self.index.is_train)

        self.gather_ds = GatherDataset(self.arrays, self.dist_mean, self.dist_std)
        self.val_ds = SpanDataset(
            self.arrays, self.index.val_chunks, self.dist_mean, self.dist_std
        )
        self.train_sampler = StratifiedBatchSampler(
            self.index.train_pos, self.index.train_neg,
            batch_size=self.batch_size, pos_per_batch=self.pos_per_batch,
            steps_per_epoch=self.steps_per_epoch, seed=self.seed + self.fold_idx,
        )

        # Scoring the fitted model on its own training blocks needs the natural
        # 0.28% rate, or the train AUPRC is not comparable to val. A strided
        # subsample of the train rows keeps that rate in expectation and cuts 47M
        # rows to 4.7M. Strided rather than random because the rows are block
        # ordered, so a stride stays spatially uniform and reads mostly sequentially.
        train_rows = np.flatnonzero(self.index.is_train).astype(np.int32)
        self.train_eval_idx = train_rows[::self.train_eval_stride]

    @property
    def batch_pos_weight(self):
        """See StratifiedBatchSampler.batch_pos_weight. Unused unless
        train.POS_WEIGHT is set."""
        return self.train_sampler.batch_pos_weight

    def _loader(self, dataset, sampler):
        return DataLoader(
            dataset,
            batch_size=None,  # disables automatic batching: the sampler yields whole batches
            sampler=sampler,
            num_workers=self.num_workers,
            worker_init_fn=_populate_worker if self.num_workers else None,
            persistent_workers=bool(self.num_workers),
        )

    def train_dataloader(self):
        return self._loader(self.gather_ds, self.train_sampler)

    def val_dataloader(self):
        # No sampler: SpanDataset items are already batches, in array order, so the
        # default sequential pass over them is exactly the sweep we want.
        return DataLoader(self.val_ds, batch_size=None, num_workers=0)

    def train_eval_dataloader(self):
        """Natural rate single pass over a subsample of the train rows, for scoring
        the fitted model on its own training data. Deliberately not
        train_dataloader(): those batches are stratified to 1:31, so precision and
        recall off them are not comparable to anything measured on val."""
        return self._loader(
            self.gather_ds, SequentialIndexSampler(self.train_eval_idx, self.val_batch_rows)
        )
