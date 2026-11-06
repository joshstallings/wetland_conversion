"""
Datasets, batch samplers and the DataModule for the array pipeline. Everything
reads from data/arrays through arrays.py; nothing here touches the source
parquet.

"""

import json
import time
from pathlib import Path

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

    def __init__(self, train_pos, train_neg, batch_size=2048, 
                 pos_per_batch=64,steps_per_epoch=None, hard_neg_frac=0.0, seed=0):
        if not 0 < pos_per_batch < batch_size:
            raise ValueError(f"pos_per_batch {pos_per_batch} must be in (0, {batch_size})")
        self.train_pos = train_pos
        self.train_neg = train_neg
        self.batch_size = batch_size
        self.pos_per_batch = pos_per_batch
        self.neg_per_batch = batch_size - pos_per_batch

        self.hard_neg_frac = float(hard_neg_frac)
        self.hard_neg = None

        self.seed = seed
        self.steps_per_epoch = (
            steps_per_epoch if steps_per_epoch is not None
            else max(train_pos.size // pos_per_batch, 1)
        )
        self._epoch = 0

    def set_hard_pool(self, hard_neg):
        """
        Swaps in the row indices that the miner ranked highest. Set 
        to None to go back to all uniform negatives. 
        """
        self.hard_neg = None if hard_neg is None else np.asarray(hard_neg, dtype=np.int32)

    @property
    def hard_per_batch(self):
        if self.hard_neg is None or self.hard_neg.size == 0:
            return 0

        return int(round(self.hard_neg_frac * self.neg_per_batch))

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

        n_hard = self.hard_per_batch
        n_uniform = self.neg_per_batch - n_hard

        pos_order = rng.permutation(self.train_pos.size)
        cursor = 0
        for _ in range(self.steps_per_epoch):
            # If we have gone through all positives, reshuffle them
            if cursor + self.pos_per_batch > pos_order.size:
                pos_order = rng.permutation(self.train_pos.size)
                cursor = 0
            pos = self.train_pos[pos_order[cursor:cursor + self.pos_per_batch]]
            cursor += self.pos_per_batch

            # Get the uniform negatives then the hard negatives.
            neg = self.train_neg[rng.integers(0, self.train_neg.size, n_uniform)]
            if n_hard:
                hard = self.hard_neg[rng.integers(0, self.hard_neg.size, n_hard)]
                neg = np.concat([neg, hard])

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
                 train_eval_stride=10, hard_neg_frac=0.0, num_workers=0, seed=0):
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
        self.hard_neg_frac = hard_neg_frac
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
            steps_per_epoch=self.steps_per_epoch, hard_neg_frac=self.hard_neg_frac, seed=self.seed + self.fold_idx,
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

    def sample_negative_candidates(self, n_candidates, rng):
        """
        Get N random negative indices. 
        """
        pool = self.index.train_neg
        n = min(n_candidates, pool.size)
        take = rng.choice(int(pool.size), size=int(n), replace=False)
        
        return np.sort(pool[take]).astype(np.int32)

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

    def mining_dataloader(self, rows, batch_rows=None):
        """
        Sequential scoring pass over chosen rows. This loader is built then
        thrown away once per mining round. 
        """
        sampler = SequentialIndexSampler(rows, batch_rows or self.val_batch_rows)
        return DataLoader(self.gather_ds, batch_size=None, sampler=sampler)


CELL_DIR = "data/cells_2019_2024_300m"
CELL_SCALARS = ("log1p_dist_mean", "log1p_dist_min", "log1p_n_wetland")


def load_cells(cell_dir=CELL_DIR):
    """
    Reads a build_cells.py output into RAM. Returns (cells, manifest).

    Everything is loaded resident, not mmapped. At 300 m the two embedding arrays
    are 0.9 GB each; at 90 m they are 6.6 GB each, which still fits on this
    machine but is the size to watch if cells get smaller.
    """
    cell_dir = Path(cell_dir)
    with open(cell_dir / "manifest.json") as fh:
        manifest = json.load(fh)
    names = ("emb_mean", "emb_std", "target", "n_wetland", "n_developed",
             "dist_mean", "dist_min", "block_code", "footprint_px", "area_converted")
    cells = {k: np.load(cell_dir / f"{k}.npy") for k in names}

    n = manifest["stats"]["n_cells"]
    for k, a in cells.items():
        if a.shape[0] != n:
            raise ValueError(f"{k}.npy has {a.shape[0]:,} rows, manifest says {n:,}")
    return cells, manifest


class CellDataset:
    """
    Random access cells by index array, same batch at a time convention as
    GatherDataset. Returns (emb_mean (B, 192), emb_std (B, 192), scalars (B, 3),
    target (B,), n_wetland (B,)), all float32 except n_wetland, which stays int32
    so the model can choose whether to weight by it.

    target_key and weight_key pick which cell arrays fill the last two slots. The
    weight should be the denominator of the target, so target times weight is a
    pixel count: n_wetland for the share, footprint_px for area_converted.
    """

    def __init__(self, cells, scalars, target_key="target", weight_key="n_wetland"):
        self.emb_mean = cells["emb_mean"]
        self.emb_std = cells["emb_std"]
        self.target = cells[target_key]
        self.n_wetland = cells[weight_key]
        self.scalars = scalars

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        return (torch.from_numpy(self.emb_mean[idx]), torch.from_numpy(self.emb_std[idx]),
                torch.from_numpy(self.scalars[idx]), torch.from_numpy(self.target[idx]),
                torch.from_numpy(self.n_wetland[idx]))


class ShuffledIndexSampler:
    """
    One pass over row_idx per epoch in a fresh order, batch_size at a time.
    Seeded off (seed, epoch) like StratifiedBatchSampler so a run reproduces from
    its seed. No stratification: about 3% of 300 m cells have any conversion, so a
    1,024 cell batch carries about 30 positives at the natural rate already.
    """

    def __init__(self, row_idx, batch_size, seed=0):
        self.row_idx = np.asarray(row_idx, dtype=np.int64)
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0

    def __len__(self):
        return (self.row_idx.size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self._epoch])
        self._epoch += 1
        order = self.row_idx[rng.permutation(self.row_idx.size)]
        for start in range(0, order.size, self.batch_size):
            # Sorted within the batch so the fancy index walks memory forward.
            yield np.sort(order[start:start + self.batch_size])


class CellDataModule(pl.LightningDataModule):
    """
    One fold's loaders over a cell artifact from build_cells.py.

    fold_code is the same per block array ArrayDataModule takes
    (arrays.fold_of_code), so cell folds are the pixel folds: a cell inherits the
    fold of the 10 km block it sits in, and build_cells.py never lets a cell cross
    a block edge.

    min_wetland_train drops tiny cells from training only. Val always keeps every
    cell in the fold, so val numbers describe the real population and do not move
    when this knob does.

    pos_per_batch None trains on one shuffled pass over the train cells at their
    natural rate. An int switches to StratifiedBatchSampler with "positive" meaning
    any conversion in the cell. That is for the stage one classifier only: on the
    regressor it would inflate the share it learns to predict.

    positive_only keeps only cells with any conversion, in train and in val, for
    the stage two regressor on log share. val_all_idx and val_all_dataloader still
    cover every val cell, so stage two can be scored on the cells stage one sees.

    target_key and weight_key go straight to CellDataset. Folds, filters and
    scalars do not depend on them.
    """

    def __init__(self, cells, manifest, fold_code, fold_idx, batch_size=1024,
                 val_batch_size=8192, min_wetland_train=1, pos_per_batch=None,
                 steps_per_epoch=None, num_workers=0, seed=0, positive_only=False,
                 target_key="target", weight_key="n_wetland"):
        super().__init__()
        self.cells = cells
        self.manifest = manifest
        self.fold_code = np.asarray(fold_code)
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.min_wetland_train = min_wetland_train
        self.pos_per_batch = pos_per_batch
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.seed = seed
        self.positive_only = positive_only
        self.target_key = target_key
        self.weight_key = weight_key

    def setup(self, stage=None):
        if getattr(self, "train_idx", None) is not None:
            return

        c = self.cells
        if len(self.fold_code) != len(self.manifest["block_ids"]):
            raise ValueError("fold_code length does not match the cell manifest's block list")
        cell_fold = self.fold_code[c["block_code"]]
        is_train = cell_fold != self.fold_idx
        keep = c["n_developed"] > 0 if self.positive_only else np.ones(cell_fold.size, dtype=bool)
        self.train_idx = np.flatnonzero(is_train & keep & (c["n_wetland"] >= self.min_wetland_train))
        self.val_idx = np.flatnonzero(~is_train & keep)
        self.val_all_idx = np.flatnonzero(~is_train)

        if np.any(cell_fold[self.train_idx] == self.fold_idx):
            raise AssertionError("train cells include val fold cells, the fold mask was dropped")
        if self.train_idx.size == 0 or self.val_idx.size == 0:
            raise AssertionError(f"fold {self.fold_idx} has an empty train or val split")

        # All three go through log1p before the z score. The distances run 30 m to
        # 110 km, but conversion all but stops past 1.5 km (0.01% of 300 m cells
        # with any conversion), so on raw meters the long tail sets the std and the
        # 30 to 1,500 m range that carries the signal lands in a sliver near the
        # bottom. n_wetland runs 1 to 100 at 300 m and 1 to 1,089 at 990 m, and its
        # effect on the target is roughly logarithmic (figure 3 in
        # notebooks/cell_analysis.ipynb). Stats from the train cells only, same rule
        # as dist_norm_stats.
        raw = np.log1p(np.stack([c["dist_mean"], c["dist_min"], c["n_wetland"]], axis=1).astype(np.float64))
        mu = raw[self.train_idx].mean(axis=0)
        sd = raw[self.train_idx].std(axis=0)
        sd[sd == 0] = 1.0
        self.scalar_mean, self.scalar_std = mu.astype(np.float32), sd.astype(np.float32)
        scalars = ((raw - mu) / sd).astype(np.float32)

        self.ds = CellDataset(c, scalars, self.target_key, self.weight_key)
        if self.positive_only and self.pos_per_batch is not None:
            raise ValueError("pos_per_batch stratifies on any conversion, which positive_only already fixes at 100%")
        if self.pos_per_batch is None:
            if self.steps_per_epoch is not None:
                raise ValueError("steps_per_epoch only applies with pos_per_batch set")
            self.train_sampler = ShuffledIndexSampler(self.train_idx, self.batch_size,
                                                      seed=self.seed + self.fold_idx)
        else:
            is_pos = c["n_developed"][self.train_idx] > 0
            self.train_sampler = StratifiedBatchSampler(
                self.train_idx[is_pos], self.train_idx[~is_pos],
                batch_size=self.batch_size, pos_per_batch=self.pos_per_batch,
                steps_per_epoch=self.steps_per_epoch, seed=self.seed + self.fold_idx,
            )
        self.log_split()

    @property
    def batch_positive_rate(self):
        """Share of train batch cells with any conversion, which is the fold's own
        rate unless pos_per_batch is set."""
        if isinstance(self.train_sampler, StratifiedBatchSampler):
            return self.train_sampler.batch_positive_rate
        return float((self.cells["n_developed"][self.train_idx] > 0).mean())

    def log_split(self):
        """Cell counts and positive rates for this fold, so a val metric always has
        the rate it was computed against sitting next to it."""
        c = self.cells
        print(f"fold {self.fold_idx}  {self.manifest['cell_size_m']} m cells  "
              f"min_wetland_train {self.min_wetland_train}")
        for name, idx in (("train", self.train_idx), ("val", self.val_idx)):
            n_dev, n_wet = int(c["n_developed"][idx].sum()), int(c["n_wetland"][idx].sum())
            print(f"  {name:>5}  cells {idx.size:>10,}  any conversion {(c['n_developed'][idx] > 0).mean():.2%}  "
                  f"mean target {c['target'][idx].mean():.3%}  pixel rate {n_dev / n_wet:.3%}")

    def _loader(self, sampler):
        return DataLoader(self.ds, batch_size=None, sampler=sampler, num_workers=self.num_workers,
                          persistent_workers=bool(self.num_workers))

    def train_dataloader(self):
        return self._loader(self.train_sampler)

    def val_dataloader(self):
        return self._loader(SequentialIndexSampler(self.val_idx, self.val_batch_size))

    def val_all_dataloader(self):
        """Every val cell, converted or not, in val_all_idx order. Same as
        val_dataloader unless positive_only is set."""
        return self._loader(SequentialIndexSampler(self.val_all_idx, self.val_batch_size))

    def train_eval_dataloader(self):
        """Every train cell once, in order, for scoring the fitted model on its own
        training data with the same metrics as val."""
        return self._loader(SequentialIndexSampler(self.train_idx, self.val_batch_size))


BAG_MODES = ("pixels", "cell_mean")


def load_bags(cell_dir, cell_manifest, load_emb=True):
    """
    The build_bags.py index plus the pixel features it points at, reordered once
    into cell order. Returns a dict of ptr (n_cells + 1,), log_dist (n_rows,)
    float32 log1p meters, and emb (n_rows, 192) float16 or None.

    Reordering costs one 22.8 GB resident copy of emb but turns every bag into one
    contiguous slice. Gathering through pix_idx from the original order would
    instead hit about 10 short runs per cell, one per pixel row the cell spans.
    Each block's slice of pix_idx only permutes that block's own rows, so the copy
    goes one block read at a time.

    load_emb False skips the copy, for the cell_mean control, which never reads
    pixel embeddings.
    """
    cell_dir = Path(cell_dir)
    with open(cell_dir / "bags_manifest.json") as fh:
        bag_manifest = json.load(fh)
    if bag_manifest["cells_built_utc"] != cell_manifest["built_utc"]:
        raise AssertionError(
            f"{cell_dir}/pix_idx.npy was built against cells from {bag_manifest['cells_built_utc']}, "
            f"the cells on disk are from {cell_manifest['built_utc']}. Rerun build_bags.py."
        )

    array_dir = Path(cell_manifest["source_array_dir"])
    with open(array_dir / "manifest.json") as fh:
        src = json.load(fh)
    pix_idx = np.load(cell_dir / "pix_idx.npy")
    n_wetland = np.load(cell_dir / "n_wetland.npy")
    ptr = np.concatenate([[0], np.cumsum(n_wetland, dtype=np.int64)])

    # log1p for the same reason the cell scalars get it: the signal sits in the
    # 30 to 1,500 m range and the raw tail out to 110 km would set the std.
    dist = np.load(array_dir / "dist.npy")
    log_dist = np.log1p(dist[pix_idx]).astype(np.float32)
    del dist

    emb = None
    if load_emb:
        t0 = time.time()
        src_emb = np.load(array_dir / "emb.npy", mmap_mode="r")
        emb = np.empty(src_emb.shape, dtype=src_emb.dtype)
        starts, counts = np.asarray(src["block_spans"], dtype=np.int64).T
        for code, (s, c) in enumerate(zip(starts, counts)):
            emb[s:s + c] = np.asarray(src_emb[s:s + c])[pix_idx[s:s + c] - s]
            if code % 400 == 0:
                print(f"  reordering emb into cell order, block {code:>5}/{len(starts)}  {time.time() - t0:.0f}s")
        print(f"emb in cell order: {emb.nbytes / 1e9:.1f} GB in {time.time() - t0:.0f}s")

    return {"ptr": ptr, "log_dist": log_dist, "emb": emb, "manifest": bag_manifest}


class BagDataset:
    """
    Cells as bags of their wetland pixels, a batch of cell indices at a time.

    Returns packed pixels rather than a padded tensor: emb (N, 192), dist (N,),
    bag_id (N,) giving each pixel's cell as a position in the batch, pos (N,) its
    slot within that cell, then scalars (B, 3), target (B,), n_wetland (B,) as in
    CellDataset. Packed because the pixel encoder has BatchNorm, and padding rows
    would leak into its batch statistics. The model pads after encoding.

    cell_mean mode is the control for the averaging question. Each cell becomes a
    bag of one pseudo pixel, its emb_mean and the mean of its pixels' dist
    feature, so the model and training are identical and only the information
    differs.
    """

    def __init__(self, cells, bags, scalars, dist_feat, mode="pixels"):
        if mode not in BAG_MODES:
            raise ValueError(f"mode {mode} is not one of {BAG_MODES}")
        if mode == "pixels" and bags["emb"] is None:
            raise ValueError("pixels mode needs bags loaded with load_emb=True")
        self.mode = mode
        self.ptr = bags["ptr"]
        self.emb = bags["emb"]
        self.dist_feat = dist_feat
        self.emb_mean = cells["emb_mean"]
        self.target = cells["target"]
        self.n_wetland = cells["n_wetland"]
        self.scalars = scalars
        if mode == "cell_mean":
            self.cell_dist_feat = (np.add.reduceat(dist_feat, self.ptr[:-1], dtype=np.float64)
                                   / self.n_wetland).astype(np.float32)

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        b = idx.size
        if self.mode == "cell_mean":
            emb = self.emb_mean[idx]
            dist = self.cell_dist_feat[idx]
            bag_id = np.arange(b, dtype=np.int64)
            pos = np.zeros(b, dtype=np.int64)
        else:
            lengths = self.n_wetland[idx].astype(np.int64)
            first = np.cumsum(lengths) - lengths
            pos = np.arange(lengths.sum(), dtype=np.int64) - np.repeat(first, lengths)
            rows = np.repeat(self.ptr[idx], lengths) + pos
            emb = self.emb[rows]
            dist = self.dist_feat[rows]
            bag_id = np.repeat(np.arange(b, dtype=np.int64), lengths)
        return (torch.from_numpy(emb), torch.from_numpy(dist), torch.from_numpy(bag_id),
                torch.from_numpy(pos), torch.from_numpy(self.scalars[idx]),
                torch.from_numpy(self.target[idx]), torch.from_numpy(self.n_wetland[idx]))


class BagDataModule(CellDataModule):
    """
    CellDataModule with cells served as bags. Folds, min_wetland_train, the cell
    scalars and the samplers all come from the parent unchanged, so val cells and
    their order match the mean and std classifier exactly and their val_preds
    join on cell_idx.

    The per pixel dist feature is log1p meters z scored over the pixels of this
    fold's train cells only, same rule as everywhere else.
    """

    def __init__(self, cells, manifest, bags, fold_code, fold_idx, bag_mode="pixels", **kwargs):
        super().__init__(cells, manifest, fold_code, fold_idx, **kwargs)
        self.bags = bags
        self.bag_mode = bag_mode

    def setup(self, stage=None):
        if isinstance(getattr(self, "ds", None), BagDataset):
            return
        super().setup(stage)

        in_train = np.zeros(self.cells["n_wetland"].size, dtype=bool)
        in_train[self.train_idx] = True
        pix_train = np.repeat(in_train, self.cells["n_wetland"])
        log_dist = self.bags["log_dist"]
        mu = float(np.mean(log_dist, where=pix_train, dtype=np.float64))
        sd = float(np.std(log_dist, where=pix_train, dtype=np.float64)) or 1.0
        self.dist_mean, self.dist_std = np.float32(mu), np.float32(sd)
        dist_feat = ((log_dist - self.dist_mean) / self.dist_std).astype(np.float32)

        # The parent's CellDataset already holds the z scored cell scalars.
        self.ds = BagDataset(self.cells, self.bags, self.ds.scalars, dist_feat, mode=self.bag_mode)
