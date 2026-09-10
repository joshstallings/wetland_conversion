"""
Read side of the array artifact that build_arrays.py writes to data/arrays.
Everything the training pipeline needs from those files goes through here:
opening them, getting them resident, mapping blocks to folds, and turning a fold
assignment into the row indices the sampler draws from.

Nothing here is cached to disk. Every index array below is a few passes over 59M
element arrays, well under a second in total, so persisting them would only
create a second thing that can go stale. The fold assignment itself is a draw
from a seed (folds.assign_folds), so reproducing a split means reusing the seed.

"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from features import EMB_COLS

ARRAY_DIR = "data/arrays"

# emb.npy is 22.76 GB and has to stay mmapped. The other three are small enough
# to read straight into RAM (415 MB for all of them), which is strictly better
# than mmapping: every fold index operation reads label and block_code end to
# end, and dist gets gathered once per batch, so a page fault on any of it is
# pure waste.
MMAPPED = ("emb",)
RESIDENT = ("dist", "label", "block_code")

# Not needed for training, so not opened unless asked for. Cheap to have around
# the first time a result needs a map or a spatial sanity check.
OPTIONAL = ("xy", "rowcol", "label_raw")

PAGE_BYTES = 4096


def load_arrays(array_dir=ARRAY_DIR, optional=()):
    """
    Opens the artifact. Returns (arrays, manifest).

    arrays maps name to array: emb is a read only mmap of float16 (n, 192), and
    dist, label, block_code are real in RAM arrays. Anything named in optional
    (see OPTIONAL) is added as an mmap.

    Refuses to load if the manifest's embedding column list is not exactly
    features.EMB_COLS. A .npy carries shape and dtype and nothing else, so if
    FEATURE_COLS changes and the arrays do not, the only symptom would be a model
    trained on scrambled columns: no error, no clue. This check is the whole
    reason the column order goes in the manifest.
    """
    array_dir = Path(array_dir)
    with open(array_dir / "manifest.json") as fh:
        manifest = json.load(fh)

    stored, expected = manifest["emb_columns"], list(EMB_COLS)
    if stored != expected:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(stored, expected)) if a != b),
            min(len(stored), len(expected)),
        )
        raise ValueError(
            f"{array_dir}/emb.npy was built from a different feature list than "
            f"features.EMB_COLS ({len(stored)} stored columns against {len(expected)} "
            f"expected, first difference at position {first_diff}). Every stored column "
            f"position and every saved row index is wrong. Rebuild with build_arrays.py."
        )

    arrays = {}
    for name in MMAPPED:
        arrays[name] = np.load(array_dir / f"{name}.npy", mmap_mode="r")
    for name in RESIDENT:
        arrays[name] = np.load(array_dir / f"{name}.npy")
    for name in optional:
        if name not in OPTIONAL:
            raise ValueError(f"{name} is not one of the optional arrays {OPTIONAL}")
        arrays[name] = np.load(array_dir / f"{name}.npy", mmap_mode="r")

    n_rows = manifest["total_rows"]
    for name, a in arrays.items():
        if a.shape[0] != n_rows:
            raise ValueError(f"{name}.npy has {a.shape[0]:,} rows, manifest says {n_rows:,}")
    if arrays["emb"].shape[1] != len(expected):
        raise ValueError(f"emb.npy has {arrays['emb'].shape[1]} columns, expected {len(expected)}")
    if len(manifest["block_ids"]) != manifest["n_blocks"]:
        raise ValueError("manifest block_ids length does not match n_blocks")

    # label.npy is written already binarized. If a 2 ever survives, every positive
    # count and every pos_weight downstream of here is wrong.
    label_max = int(arrays["label"].max())
    if label_max > 1:
        raise ValueError(
            f"label.npy holds {label_max}, so it is not binary. It should be label == 1 "
            f"with 0 and 2 both folded to 0, and nothing downstream should binarize it again."
        )

    return arrays, manifest


def populate(arrays, names=MMAPPED, verbose=True):
    """
    Touch one byte per 4 KB page so the pages are in this process's page table
    before the first batch, instead of faulting in one at a time under the
    training loop. See the module docstring for why this is not the same thing as
    the bytes already being cached.

    Call once per process. A DataLoader worker forked from a populated parent
    inherits the mapping; a spawned one does not and has to run this itself.
    """
    for name in names:
        a = arrays.get(name)
        if a is None:
            continue
        t0 = time.time()
        a.reshape(-1).view(np.uint8)[::PAGE_BYTES].sum()
        if verbose:
            gb = a.nbytes / 1e9
            dt = time.time() - t0
            print(f"populate {name}: {gb:.2f} GB in {dt:.1f}s ({gb / max(dt, 1e-9):.2f} GB/s)")


def block_spans(manifest):
    """
    (starts, counts) indexed by block_code, straight off the manifest.

    build_arrays.py derived these with a searchsorted over the block contiguous
    array and checked every one of the 1,727 counts against population_stats.json,
    so they are taken as given here rather than recomputed.
    """
    spans = np.asarray(manifest["block_spans"], dtype=np.int64)
    return spans[:, 0], spans[:, 1]


def fold_of_code(manifest, fold_assignment):
    """
    Length n_blocks array mapping block_code to fold index.

    This is where the two representations of a block meet: the manifest's
    block_ids is in write order, and folds.assign_folds is keyed by the block_id
    string. Line them up wrong and you still get a clean looking 5 way split,
    just of the wrong blocks, which is why the missing key check is loud.
    """
    block_ids = manifest["block_ids"]
    missing = [b for b in block_ids if b not in fold_assignment]
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(block_ids)} blocks have no fold assigned, e.g. "
            f"{missing[:5]}. The fold assignment and the array artifact were built from "
            f"different block sets."
        )
    codes = np.array([fold_assignment[b] for b in block_ids], dtype=np.int8)
    if codes.min() < 0:
        raise ValueError("fold assignment contains a negative fold index")
    return codes


def fold_of_row(arrays, fold_code):
    """
    Per row fold index. A vectorized gather through block_code, instant, and it
    replaces the pyarrow isin predicate pushdown the streaming path needed.
    """
    return fold_code[arrays["block_code"]]


@dataclass
class FoldIndex:
    """
    Everything the loaders need for one fold. train_neg is the big one at 189 MB
    as int32; the rest are small.

    val_chunks is (m, 2) start/stop pairs rather than a row index array. The array
    is block contiguous, so fold k is a few hundred contiguous spans and
    validation reads slices instead of gathering. Chunks never straddle a block
    boundary, which costs one ragged batch per val block and buys a guarantee that
    every val read is sequential.
    """

    fold_idx: int
    train_pos: np.ndarray
    train_neg: np.ndarray
    val_chunks: np.ndarray
    is_train: np.ndarray
    n_train_blocks: int
    n_val_blocks: int

    @property
    def n_train_rows(self):
        return int(self.train_pos.size + self.train_neg.size)

    @property
    def n_val_rows(self):
        return int((self.val_chunks[:, 1] - self.val_chunks[:, 0]).sum())

    @property
    def natural_pos_weight(self):
        """Negative to positive ratio in this fold's train rows, about 356. What the
        loss wanted when batches arrived at the natural rate. Not what it wants once
        batches are stratified: see datasets.StratifiedBatchSampler."""
        return self.train_neg.size / max(self.train_pos.size, 1)


def build_fold_index(arrays, manifest, fold_code, fold_idx, val_chunk_rows=8192):
    """
    Turns a fold assignment into the row indices for fold fold_idx. Returns a
    FoldIndex. Takes well under a second.

    val_chunk_rows is the row count of one validation read. Bigger is cheaper here
    because val is a sequential sweep, unlike the scattered train batches.

    The checks at the end are not decoration. Building the sampler off a global
    positive list and forgetting the fold mask leaks val rows into training, val
    AUPRC comes back excellent, and nothing errors, so it has to be explicit.
    """
    label = arrays["label"]
    row_fold = fold_of_row(arrays, fold_code)
    is_train = row_fold != fold_idx

    is_pos = label == 1
    train_pos = np.flatnonzero(is_train & is_pos).astype(np.int32)
    # label.npy is already binary, so label == 0 is the whole negative class and
    # there is no third value to think about. That is the main thing binarizing at
    # write time buys on this side.
    train_neg = np.flatnonzero(is_train & ~is_pos).astype(np.int32)

    starts, counts = block_spans(manifest)
    val_codes = np.flatnonzero(fold_code == fold_idx)
    chunks = []
    for code in val_codes:
        start, count = int(starts[code]), int(counts[code])
        for offset in range(0, count, val_chunk_rows):
            chunks.append((start + offset, start + min(offset + val_chunk_rows, count)))
    val_chunks = np.asarray(chunks, dtype=np.int64).reshape(-1, 2)

    index = FoldIndex(
        fold_idx=fold_idx,
        train_pos=train_pos,
        train_neg=train_neg,
        val_chunks=val_chunks,
        is_train=is_train,
        n_train_blocks=int((fold_code != fold_idx).sum()),
        n_val_blocks=int(val_codes.size),
    )

    for name, idx in (("train_pos", train_pos), ("train_neg", train_neg)):
        if idx.size and np.any(row_fold[idx] == fold_idx):
            n_leaked = int(np.sum(row_fold[idx] == fold_idx))
            raise AssertionError(
                f"{n_leaked:,} rows in {name} belong to val fold {fold_idx}. The fold "
                f"mask was dropped somewhere and val is leaking into training."
            )
    if index.n_train_rows + index.n_val_rows != manifest["total_rows"]:
        raise AssertionError(
            f"train {index.n_train_rows:,} plus val {index.n_val_rows:,} rows does not "
            f"cover the {manifest['total_rows']:,} row array"
        )
    if train_pos.size == 0:
        raise AssertionError(f"fold {fold_idx} has no training positives")

    return index


def val_positive_count(arrays, index):
    """Positives in this fold's val rows, off the val spans. For logging the rate a
    val metric was computed against, which a bare AUPRC number is meaningless without."""
    label = arrays["label"]
    return int(sum(int(label[s:e].sum()) for s, e in index.val_chunks))
