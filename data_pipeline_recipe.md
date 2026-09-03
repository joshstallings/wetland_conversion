<title>Streaming recipe: full-data streaming with weighted loss</title>

# Streaming recipe: full-data streaming with weighted loss

Implementation recipe for training on every row, no undersampling, using a
shuffle-buffer stream and a class-weighted loss instead of a stratified
sampler. This is the project's only training pipeline; an earlier
negative-undersampled cache approach was tried and dropped.

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
that changes step 1's counts and step 5's `pos_weight`.

## Design decision this recipe locks in

Fold assignment (block_id -> fold) is a runtime random draw from a seed, not
a saved file, so which blocks land in val changes every run. Validation is
streamed straight from the source parquet every run, filtered to whichever
blocks land in val for this seed and fold, at the true 0.28% positive rate.
This is what keeps precision/recall/AUPRC honest, since scoring against a
downsampled validation set inflates precision relative to what the model
will see in production.

This recipe applies that same principle to training too: nothing is cached
or downsampled, every row from the source parquet is used, at the cost of
more wall clock per epoch, since every epoch re-reads and decompresses
parquet rather than reusing an in-RAM/memmapped array, and a shuffle buffer
standing in for a true full-epoch permutation.

## File layout

```
data/
  population_stats.json   # true class counts, built once (see step 1)
results/
  streaming/
    fold_{k}/
      metrics.csv, val_preds.npz, loss_curve.png, confusion_matrix.png, pr_curve.png
    manifest.json      # seed, per-fold block/row/positive counts
    fold_summary.csv   # one row per fold: precision, recall, F1, AUPRC
```

## Step 1: compute true population class counts (one-time, cheap)

This recipe's loss weighting needs the true positive/negative counts, and
there's no cache-building pass elsewhere that already touches every row to
get them from. This is a narrow scan, not a data cache: read only `label` and
`block_id` columns from the full source dataset (`dataset.to_table(columns=["label", "block_id"])`), which is fast (a couple hundred milliseconds at this
dataset's size, since it skips the 192 embedding columns entirely).

`label` here is not binary, it's three classes — 0 "Remained wetland", 1
"Converted to developed", 2 "Converted to other (non-developed)". Binarize
it while computing these counts: positive is `label == 1` only; negative is
`label == 0` or `label == 2`. Don't sum the raw label column as a shortcut
for counting positives — a label-2 row would contribute 2 to that sum and
silently inflate the apparent positive count, which is exactly what
happened in an earlier pass over this data (it produced 612,414 "positives"
and a 1.03% rate, both wrong; the real numbers are below).

Save the result to `data/population_stats.json`: total rows, total binary
positives, total binary negatives, per-block_id row counts (the last of
these is also what step 2's greedy fold balancer needs). This file doesn't
need rebuilding unless the source dataset changes.

Sanity check: total positives should read 165,908, total negatives
59,101,423.

## Step 2: seeded fold assignment (runtime, in `folds.py`)

Signature: `assign_folds(block_row_counts, n_splits, seed) -> dict[block_id -> fold_idx]`.
`block_row_counts` is a `{block_id (str): row_count}` dict, built once by
step 1 (`population_stats.json`'s `block_row_counts`) and passed straight
in — this function doesn't re-derive counts from a raw block_id array
itself. `block_id` is a string in the source parquet (e.g. `"b0239_0323"`),
not an int.

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

## Step 3: `StreamingTrainDataset` (`IterableDataset`, streams from source parquet, discards nothing)

- Subclass `torch.utils.data.IterableDataset`.
- Constructor takes: source parquet path, `feature_cols`, `label_col`, this
  fold's train block_ids, per-feature normalization stats, and
  `buffer_size` (default ~300,000 rows — at 193 features x 4 bytes that's
  roughly 230MB, comfortable on 16GB RAM).
- `__iter__` opens a `to_batches(columns=feature_cols + [label_col, "block_id"], filter=pc.field("block_id").isin(pa.array(train_block_ids, type=pa.large_string())))`
  reader over the source dataset (`block_id` is a string column, so filter
  against a string array, not an int one). Binarize `label_col` as each row
  is read — `label == 1` ("Converted to developed") is positive, `label ==
  0` or `label == 2` is negative — before it ever reaches the shuffle
  buffer or the loss; the raw 3-valued column is not a valid BCE target.
  Then run a reservoir-style shuffle buffer, swapped a batch at a time
  rather than one row at a time (profiling showed a python-level per-row
  loop, not the parquet read, was the dominant cost of `__iter__`): top the
  buffer up to `buffer_size` directly from the first incoming batches, then
  for each later batch draw that many buffer slots *without* replacement
  (`rng.choice(buffer_size, size=len(batch), replace=False)`), yield the
  evicted rows, scatter the new batch into those slots — sampling without
  replacement matters here, since a slot drawn twice in one batch would
  duplicate an evicted row and silently drop one of the new ones. If a
  single parquet batch is larger than `buffer_size`, swap it in
  `buffer_size`-sized chunks. At end of stream, permute and drain only the
  slots that actually got filled (matters if the fold is smaller than
  `buffer_size`). This is the same trick tf.data's `shuffle()` and
  WebDataset use to approximate a full shuffle without holding the whole
  dataset in memory. Rows arrive from parquet roughly in spatial (tile)
  order, so check that `buffer_size` is large enough to span more than one
  source file — otherwise batches stay spatially clustered and you've
  reintroduced the autocorrelation problem this project already watches
  for.
- No `StratifiedBatchSampler` here — positive rebalancing happens in the
  loss (step 5's `pos_weight`), not the batch composition, so batches are
  plain shuffled draws and will average close to the true 0.28% positive
  rate; don't expect every batch to contain a positive.
- Normalization stats: compute them once per fold with a single streaming
  pass over just this fold's train blocks using a running mean/var (a
  batched, parallel-variance form of Welford's algorithm — updated a batch
  at a time, not row at a time, same reasoning as the shuffle buffer
  above), before training starts — there's no cached array to compute a
  direct mean/std from here. Only normalize `features.COLS_TO_NORMALIZE`
  (just `dist_to_developed_2019_m`, which is on a meters scale) rather than
  every feature column — the AlphaEarth embedding dims come out of GEE
  already roughly unit-scaled, so normalizing them too is just extra noise
  on top of a scale that's already fine. Embed the result into a
  full-length `(mean, std)` pair shaped `(len(feature_cols),)`, with mean 0
  / std 1 (a no-op under `(x - mean) / std`) on every column not in
  `COLS_TO_NORMALIZE`, so it lines up with the full feature matrix at
  apply time. Check for NaN/inf in `dist_to_developed_2019_m` during this
  pass (coastal/boundary tiles are the likely source of an occasional bad
  value) and drop those rows from the running stats rather than letting a
  single NaN poison them. Reuse these same stats for this fold's
  validation stream too (step 4) — always train-fold statistics, never
  val's own.

## Step 4: `StreamingValDataset` for validation (`IterableDataset`, streams from source parquet)

This is a separate class from `StreamingTrainDataset` above, even though
both stream from the same source — validation doesn't shuffle-buffer (order
doesn't matter when you're just accumulating predictions to score once) and
never sees `pos_weight` or any other training-only concern.

- Subclass `torch.utils.data.IterableDataset`.
- Constructor takes: source parquet path, `feature_cols`, `label_col`, this
  fold's val block_ids, and the train-fold mean/std computed in step 3 (val
  must be normalized with train statistics, never its own, or you leak val
  distribution info into the model's input scale).
- `__iter__` opens `ds.dataset(...).to_batches(columns=feature_cols + [label_col, "block_id"], filter=pc.field("block_id").isin(pa.array(val_block_ids)))`, binarizes `label_col` the same way step 3 does (`label == 1` positive, `label == 0` or `label == 2` negative), and yields one row at a time out of each batch as `(normalized_x, y)` tensors. Batching happens in the `DataLoader` as usual (`IterableDataset` + `DataLoader(batch_size=...)` composes fine, just don't pass a `sampler`/`batch_sampler`, those aren't supported on `IterableDataset`).
- No `__len__` needed; Lightning tolerates that for `IterableDataset`, or estimate one from the fold's row count logged in step 2 if you want a progress bar denominator.

Known cost, not a memory risk: both this and step 3 rescan the relevant
slice of the source parquet every epoch (step 3) or every fold (step 4).
With predicate pushdown on `block_id` this skips most row groups outside
the matching blocks, but running many epochs and many seeds will add up in
wall clock — a known cost of this design, not something to try to
eliminate here.

## Step 5: model change — weighted loss instead of a stratified sampler

`SimpleLinearModel.__init__` needs a `pos_weight` argument, defaulting to
`None` (unweighted), passed through to
`nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))` when set.

Compute `pos_weight` from step 1's true full-population class counts, not
the fold's: `pos_weight = true_negatives / true_positives ≈ 59,101,423 /
165,908 ≈ 356.2`. This is a fixed constant for this dataset, not something
that needs recomputing per fold or per seed. It's a much larger weight than
a naive read of the raw label column would suggest (that read gives ~95.8,
which is wrong for the reason step 1 explains) — a `pos_weight` this large
means single positive examples can dominate a batch's gradient, worth
watching for loss spikes early in training and reducing the learning rate
if so.

## Step 6: `StreamingDataModule`

Named `StreamingDataModule`, not `SpatialCVDataModule`, since it's specific
to the streaming approach this recipe describes.

Constructor: `(seed, n_splits, fold_idx, feature_cols, label_col, cols_to_normalize, source_parquet_path, population_stats_path="data/population_stats.json", batch_size, buffer_size=300_000)`. `pos_weight` lives on the model (step 5), not the DataModule. `cols_to_normalize` is `features.COLS_TO_NORMALIZE`, passed through to step 3's normalization pass.

- `setup()`: load `population_stats.json`, call `assign_folds` from step 2
  with the given seed and its `block_row_counts`, split into this fold's
  train/val block_ids, compute this fold's normalization stats (step 3),
  build the `StreamingTrainDataset` (step 3) for train, build the
  `StreamingValDataset` (step 4) from the source parquet for val, reusing
  those same normalization stats.
- `train_dataloader()`: plain `DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=0)`. No sampler — `IterableDataset` doesn't support one. Leave `num_workers=0`: `IterableDataset` isn't safe under multiple workers without sharding the block list yourself via `torch.utils.data.get_worker_info()`, which isn't worth the complexity until you've actually measured an IO stall that raising workers would fix.
- `val_dataloader()`: `DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=0)`. No `shuffle`, no `persistent_workers`.

## Step 7: `train.py`'s `main()`

`train.py` builds a `StreamingDataModule` (step 6) with `buffer_size` and
writes results under `results/streaming/`. Everything else (the seeded
fold, `models.SimpleLinearModel`, `reporting.py`'s plots and
`write_fold_summary`) is common code the loop calls into.

- Fix a single `SEED` for the run (the design decision section above covers
  why fold assignment is a runtime draw rather than a saved file — pick one
  and log it).
- Loop over `n_splits` folds for that seed, writing to
  `results/streaming/fold_{k}/`. Write `results/streaming/manifest.json`
  with the seed and the per-fold block/row/positive counts from step 2's
  logging.
- Epoch budget: each epoch here re-reads and decompresses this fold's slice
  of the source parquet, so epochs cost meaningfully more wall clock than a
  cached in-RAM dataset would. Start low (2-3) and log the epoch count
  actually run in the manifest.
- After all folds finish, build `results/streaming/fold_summary.csv`: one row
  per fold with precision, recall, F1, AUPRC, that fold's true positive
  rate, and epochs run, plus the mean and standard deviation of each metric
  across folds.

## Step 8: before trusting any number out of this pipeline

- Step 1's counts should read exactly 165,908 positives and 59,101,423
  negatives. If not, something is off in the population scan (check the
  label binarization first) before any model gets near the data.
- Per-fold positive rate (step 2 log) should hover near 0.28% for every
  fold. A fold that's off by a large margin is a sign the greedy balancer
  isn't working as intended, or that a few outsized blocks (the
  111,527-row block, for instance) are dominating one fold.
- If loss isn't decreasing, check `buffer_size` first — too small a buffer relative to how spatially
  ordered the source parquet is means batches are still spatially
  clustered, which both hurts optimization and reintroduces the
  autocorrelation leakage concern this project already watches for.
- A validation AUPRC that looks unreasonably high relative to the sklearn
  baseline is a bug to chase, not a result to report, per usual house rules
  on this project: check normalization stats were computed from train rows
  only, and check `pos_weight` is actually reaching the loss (not silently
  `None`).
