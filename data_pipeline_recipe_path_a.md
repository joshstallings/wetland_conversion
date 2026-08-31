<title>Path A recipe: cached downsampled pool</title>

# Path A recipe: cached downsampled pool

Implementation recipe for training off a negative-undersampled cache, built
once offline. This is one of two parallel training approaches for this
project; the other, full-data streaming with a weighted loss and no
downsampling, is in `data_pipeline_recipe_path_b.md`. The two are meant to
be run against the same seeded fold split and compared on the same
natural-rate validation stream, so keep the fold-assignment step (step 2)
identical between the two files if you touch it.

Target machine has 16GB RAM and 726GB free disk. Source data is 59.27M rows,
193 features, 1727 spatial blocks (~10km each), true positive rate 0.28%
(165,908 positives, 59,101,423 negatives). Those numbers drive the choices
below, so if the dataset changes size meaningfully, revisit them.

Note on where these numbers come from: `label` in the source parquet is not
binary, it's three classes — 0 "Remained wetland" (58,878,170 rows), 1
"Converted to developed" (165,908 rows), 2 "Converted to other
(non-developed)" (223,253 rows). The positive class for this project's
target is label 1 only. Everything below treats label 0 and label 2 as
negative, i.e. the binary target is "did this wetland convert specifically
to developed land," not "did this wetland change at all." That's a real
target-definition choice, not a fact, and it's the default this recipe
locks in rather than something to silently assume — if the project wants
label-2 rows excluded entirely instead of folded into the negative class,
that changes step 1's kept-row logic and every count that follows from it.
Reuse `label_utils.binarize_label` for this rather than reimplementing it —
it's the same helper the population scan and both of path B's streaming
datasets already use, and it's what caught the original label-2-double-
counts-as-positive bug.

## Design decision this recipe locks in

Fold assignment (block_id -> fold) is a runtime random draw from a seed, not
a saved file, so which blocks land in val changes every run. Validation
therefore can't be served from a fixed cache — it's streamed straight from
the source parquet every run, filtered to whichever blocks land in val for
this seed and fold, at the true 0.28% positive rate. This is what keeps
precision/recall/AUPRC honest, since scoring against a downsampled
validation set inflates precision relative to what the model will see in
production.

Training is different: path A undersamples negatives to a fixed ratio and
caches that pool once, offline. This is fast to iterate on but discards most
negative rows, so its raw output probabilities end up calibrated to the
cache's balanced rate rather than the true ~356:1 rate — step 3 below
includes the standard correction for that. Whether this discarding actually
costs anything on this dataset is an empirical question, not a settled one;
that's what comparing against path B's results answers.

## File layout

```
data/
  train_pool_cache/
    X.npy             # float32, (n_pool_rows, 193), memmap-loadable
    y.npy              # int8,   (n_pool_rows,)
    block_id_int.npy   # int32,  (n_pool_rows,), index into block_ids.json
    block_ids.json      # ordered list, block_id_int i <-> block_ids[i]
    manifest.json       # build params, provenance, counts (see step 1)
results/
  cached_pool/
    fold_{k}/
      metrics.csv, val_preds.npz, loss_curve.png, confusion_matrix.png, pr_curve.png
    manifest.json      # seed, per-fold block/row/positive counts
    fold_summary.csv   # one row per fold: precision, recall, F1, AUPRC
```

## Step 1: build the training pool cache (one-time script, e.g. `build_train_pool_cache.py`)

Runs once, offline, not part of `train.py`. Streams the source dataset with
`pyarrow.dataset.Dataset.to_batches()` so nothing beyond one batch is ever
resident in memory during the scan.

Algorithm:
1. Open the dataset with `ds.dataset(parquet_path, format="parquet")`.
2. Fix a `cache_build_seed` (separate from the per-run fold seed — this one
   controls which negatives get discarded permanently, so it must be fixed
   and logged, not redrawn).
3. Binarize the raw `label` column as you read it with
   `label_utils.binarize_label`: `label == 1` ("Converted to developed") is
   positive, `label == 0` ("Remained wetland") and `label == 2` ("Converted
   to other, non-developed") are both negative. Store this binary value as
   `y`, not the raw 3-valued label — the raw column isn't a valid BCE
   target and treating it as one is exactly the bug that produced a wrong
   positive count earlier (summing raw label values counts each label-2 row
   as 2, which silently inflates the apparent positive count).
4. Iterate `to_batches(columns=feature_cols + [label_col, "block_id"])`. For
   each batch: keep every positive row; keep negative rows with probability
   `neg_keep_prob` computed from the target ratio (default 10 negatives per
   positive — start here, revisit after seeing baseline precision/recall).
   Use a `np.random.default_rng(cache_build_seed)` Bernoulli draw per batch,
   not a fixed stride, so kept negatives aren't spatially clustered by
   row-group order.
5. Append kept rows' feature block, binarized label, and block_id string to
   in-memory lists of arrays as you go. At the default 10:1 ratio that's
   roughly 1.8M rows, under 2GB total, comfortable to hold for the duration
   of this one offline script even on 16GB RAM — there's room to raise the
   ratio well past 10 if step 3's baseline results want more negatives.
   Do not persist per-batch to disk incrementally, just accumulate and
   write once at the end.
6. After the scan: build the sorted unique `block_ids` list, map each kept
   row's string block_id to its integer index, `np.concatenate` the
   feature/label/block_id_int lists, `np.save` the three arrays and the
   `block_ids.json` mapping.
7. Write `manifest.json` with: source parquet path, row counts (total
   scanned, total kept, positives kept, negatives kept, and separately how
   many kept negatives came from label 0 vs label 2, since those are two
   different real-world transitions folded into one training class),
   `neg_keep_prob`, `cache_build_seed`, feature_cols list, NLCD release and
   AlphaEarth embedding version if you have version strings for those,
   build timestamp. This is the provenance record and also where the true
   population counts needed for step 3's intercept correction come from
   (`total scanned` and `positives kept` give you the true positive count
   directly, since every positive is kept; true negative count is `total
   scanned - positives kept`) — though `data/population_stats.json` (path
   B's step 1) already has these same true counts computed once and
   reproducibly, so reuse `population_stats.load_population_stats()` there
   instead of recomputing them from the cache's own scan. This is the
   provenance record — a result trained off this cache without this file
   next to it isn't reproducible.

Sanity check before moving on: print positives kept vs the true 165,908
count (should be exactly that, no subsampling on positives) and confirm the
kept negative rate lands near the target ratio.

## Step 2: seeded fold assignment (runtime, in `folds.py`)

Signature: `assign_folds(block_row_counts, n_splits, seed) -> dict[block_id -> fold_idx]`.
`block_row_counts` is a `{block_id (str): row_count}` dict — reuse
`population_stats.load_population_stats()`'s `block_row_counts` (the same
file path B's step 1 builds) rather than deriving a separate copy from the
cache. One source of truth for row counts, and it's what keeps this
function's output identical to path B's for a shared seed. `block_id` is a
string (e.g. `"b0239_0323"`), the same value stored in `block_ids.json`.

Do not use `sklearn.GroupKFold` for this — it takes no `random_state`, so
"randomizing" it means relying on its undocumented sensitivity to input
order, which is fragile and not really what you want to control. Write it
directly instead:

1. `rng = np.random.default_rng(seed)`, shuffle `block_row_counts`'s keys.
2. Because block size is heavily skewed (min 1 row, max 111,527, mean
   ~34,318), a plain round-robin deal can leave one fold with a
   disproportionate share of rows or positives just by chance. Use greedy
   size balancing instead: for each shuffled block, assign it to whichever
   fold currently has the fewest total rows so far (track a running
   row-count total per fold from `block_row_counts`).
3. Return the block_id -> fold_idx dict.

At the start of every run, after calling this, log per fold: block count,
total row count, positive count, positive rate. That log is your check for
whether a given seed happened to produce a degenerate split (e.g., a fold
that got unlucky and has almost no positives) before you spend time
training on it.

This is the exact same function path B uses (`folds.assign_folds`, already
implemented) — don't reimplement it here, import it. That's what makes a
given seed produce the same split under both paths for a direct comparison.

## Step 3: rewrite `ParquetBlockDataset` for training (map-style, memmap-backed)

Replace the current parquet-scanning constructor. New version:

- Load `X.npy` with `np.load(path, mmap_mode="r")` (not a full read), and
  `y.npy` / `block_id_int.npy` fully into RAM (both are small, one int/int8
  value per pool row, tens of MB at most).
- Constructor takes the set of block_ids belonging to this split (train
  blocks for this fold), maps them to ints via the cache's `block_ids.json`,
  and computes `row_idx = np.flatnonzero(np.isin(block_id_int, split_block_ints))`
  once. This replaces the old `to_table(filter=isin(...))` parquet rescan
  entirely — it's now a vectorized lookup over an in-RAM int array.
- `__getitem__(i)` indexes `X_mm[row_idx[i]]`, normalizes (see below),
  returns it with `y[row_idx[i]]`.
- No pandas anywhere in this class.

Feature normalization: only normalize `features.COLS_TO_NORMALIZE` (just
`dist_to_developed_2019_m`, on a meters scale — plausibly 0 to tens of
thousands), not every feature column. The AlphaEarth embedding dims come
out of GEE already roughly unit-scaled, so normalizing them too is just
extra noise on top of a scale that's already fine; an unnormalized
`dist_to_developed_2019_m`, left as is, would make a linear model
effectively only look at distance. This is the same subset path B's
`compute_normalization_stats` normalizes — match it rather than normalizing
every column, or the two paths' inputs aren't comparable on the same
footing. Compute mean/std once from this fold's training rows only
(`X_mm[row_idx][:, dist_col]` mean/std, done once in the DataModule's
`setup()`, not per `__getitem__`), embed the result into a full-length
`(mean, std)` pair with mean 0 / std 1 (a no-op) on every other column, and
apply `(x - mean) / std` in `__getitem__`. Recompute every fold and every
seed, since which rows are "training" changes each time — don't cache these
stats across folds. Check for NaN/inf in `dist_to_developed_2019_m` before
computing stats (coastal/boundary tiles are the likely source of an
occasional bad value); use `np.nanmean`/`np.nanstd` and impute or drop
rather than letting a single NaN silently poison the whole feature's
normalization.

Intercept correction, worth doing since it's cheap: path A trains on an
artificially balanced population (default 10 negatives per positive instead
of the true ~356:1), so its raw output probabilities are calibrated to that
balanced rate, not the true one. The standard fix for rare-event logistic
regression (King and Zeng 2001) is a closed-form intercept correction: after
fitting, shift the model's bias term by `log(true_neg/true_pos) -
log(sample_neg/sample_pos)`, computed from the cache manifest's true
population counts and its kept counts. This only touches the final linear
layer's bias, apply it once after training before generating predictions.
Skip it if you're only comparing AUPRC/precision/recall on ranked scores
(threshold-independent or fixed-threshold-after-tuning), since ranking is
unaffected by a constant intercept shift; do it if you want raw predicted
probabilities to mean what they claim to mean.

## Step 4: new `StreamingValDataset` for validation (`IterableDataset`, streams from source parquet)

This is a new class, not a modification of `ParquetBlockDataset` —
validation needs iteration, not random access, since it's never
materialized as a fixed array, and it deliberately isn't drawn from the
downsampled cache.

Path B already has exactly this class (`datasets.StreamingValDataset`) and
it doesn't know or care whether train came from a cache or a stream — reuse
it directly rather than writing a second copy. If path A's `datasets`
module doesn't import from path B's, that's the one thing worth sharing
across them.

- Subclass `torch.utils.data.IterableDataset`.
- Constructor takes: source parquet path, `feature_cols`, `label_col`, this
  fold's val block_ids, and the train-fold mean/std computed in step 3 (val
  must be normalized with train statistics, never its own, or you leak val
  distribution info into the model's input scale).
- `__iter__` opens `ds.dataset(...).to_batches(columns=feature_cols + [label_col, "block_id"], filter=pc.field("block_id").isin(pa.array(val_block_ids, type=pa.large_string())))`, binarizes `label_col` with `label_utils.binarize_label` the same way step 1's cache build does (`label == 1` positive, `label == 0` or `label == 2` negative — the raw 3-valued column is not a valid target on its own), and yields one row at a time out of each batch as `(normalized_x, y)` tensors. Batching happens in the `DataLoader` as usual (`IterableDataset` + `DataLoader(batch_size=...)` composes fine, just don't pass a `sampler`/`batch_sampler`, those aren't supported on `IterableDataset`).
- No `__len__` needed; Lightning tolerates that for `IterableDataset`, or estimate one from the fold's row count logged in step 2 if you want a progress bar denominator.

Known cost, not a memory risk: this rescans the relevant slice of the source
parquet every fold. With predicate pushdown on `block_id` this skips most
row groups outside the matching blocks, but running many folds will add up
in wall clock. If that becomes the bottleneck later, the fix is a second
small cache (this one at natural rate, keyed by block_id, same memmap
pattern as step 1) — defer that until you've actually measured it's slow,
don't build it preemptively.

## Step 5: `CachedPoolDataModule`

Named `CachedPoolDataModule`, not `SpatialCVDataModule` — path B already
has a DataModule with a different constructor
(`StreamingDataModule`, `data_pipeline_recipe_path_b.md` step 6), so this
one needs its own name too. `train.py`'s unified entry point (step 6
below) imports both under these distinct names.

Constructor: `(seed, n_splits, fold_idx, feature_cols, label_col, cols_to_normalize, cache_dir, source_parquet_path, population_stats_path="data/population_stats.json", batch_size, pos_frac)`. `cols_to_normalize` is `features.COLS_TO_NORMALIZE`, passed through to step 3's normalization pass.

- `setup()`: load `population_stats.json`, call `assign_folds` from step 2
  with the given seed and its `block_row_counts`, split into this fold's
  train/val block_ids, build the `ParquetBlockDataset` (step 3) from the
  cache for train, compute normalization stats there, build the
  `StreamingValDataset` (step 4) from the source parquet for val, reusing
  those same normalization stats.
- `train_dataloader()`: keep the existing `StratifiedBatchSampler`. `__getitem__`
  here is a single memmap row slice, cheap enough that multiprocessing
  overhead likely isn't worth it, and on a 16GB machine with fork-based
  workers there's no reason to risk duplicating anything. Start at
  `num_workers=0, pin_memory=True`, only raise it if you profile actual IO
  stall.
- `val_dataloader()`: `DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=0)`. No `shuffle` (unsupported and unnecessary for `IterableDataset`), no `persistent_workers`.

## Step 6: `train.py`'s `main()`

`train.py` is shared with path B (`data_pipeline_recipe_path_b.md`), not a
separate script — see that doc's step 7 for the `PIPELINE` selector this
adds `cached_pool` as one branch of. The rest of this step describes that
branch specifically:

- Fix a single `SEED` for the run — the same value used under path B's
  `streaming` branch produces the same fold split, which is what makes the
  two directly comparable (design decision section above).
- Loop over `n_splits` folds for that seed, writing to
  `results/cached_pool/fold_{k}/`. Write `results/cached_pool/manifest.json`
  with the seed and the per-fold block/row/positive counts from step 2's
  logging.
- After all folds finish, build `results/cached_pool/fold_summary.csv`: one
  row per fold with precision, recall, F1, AUPRC, and that fold's true
  positive rate, plus the mean and standard deviation of each metric across
  folds.

## Step 7: before trusting any number out of this pipeline

- Positives kept in the cache should equal 165,908 exactly (step 1). If
  not, the negative-subsample Bernoulli draw is touching positives somehow,
  or the label binarization is wrong, go find the bug before training
  anything.
- Per-fold positive rate (step 2 log) should hover near 0.28% for every
  fold. A fold that's off by a large margin is a sign the greedy balancer
  isn't working as intended, or that a few outsized blocks (the
  111,527-row block, for instance) are dominating one fold.
- A validation AUPRC that looks unreasonably high relative to the sklearn
  baseline is a bug to chase, not a result to report, per usual house rules
  on this project: check the val set really is coming from
  `StreamingValDataset` (natural rate) and not accidentally from the
  downsampled train cache, and check normalization stats were computed from
  train rows only.
- If you've also run path B (`data_pipeline_recipe_path_b.md`) with the
  same `SEED`, compare `results/cached_pool/fold_summary.csv` against
  `results/streaming/fold_summary.csv` on matching fold-index rows, on
  AUPRC specifically (it doesn't depend on the probability calibration
  path A's undersampling has skewed). If the two land far apart, that gap
  is informative and worth understanding before picking one — a likely
  culprit if A looks surprisingly worse is a bug in the intercept
  correction, or having skipped it while still reading raw probabilities
  instead of ranked scores.
