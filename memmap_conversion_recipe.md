# Memmap conversion recipe: parquet to row addressable arrays

Implementation recipe for converting `data/alphaearth_wetland_joined` (76 parquet
files) into a set of flat, row addressable `.npy` arrays that a map style Dataset
can index at random. This replaces the parquet streaming path: once the arrays
exist, `StreamingTrainDataset`'s shuffle buffer and the `IterableDataset`
constraint both go away, and a `StratifiedBatchSampler` can compose every batch
with a fixed positive to negative ratio.

Scope: this recipe covers the conversion, the provenance manifest, the
verification, and how to read the arrays back. The sampler, the fold aware index
construction beyond the basics, and the training loop changes are a separate
recipe.

## Machine and data facts that drive the choices below

Measured on this box, not assumed. If any of these change, revisit the choices.

| fact | value |
|---|---|
| rows | 59,267,331 |
| features | 192 AlphaEarth dims (64 x {2017, 2018, 2019}) plus `dist_to_developed_2019_m` |
| spatial blocks | 1,727 |
| label classes | 0 remained wetland 58,878,170; 1 converted to developed 165,908; 2 converted to other 223,253 |
| positive rate (label 1) | 0.28% |
| RAM | 58 GB total, about 42 GB free |
| disk | 171 GB free, and it is a GCP pd-standard HDD: 27.7 MB/s sequential, 329 random 4 KB IOPS |
| source parquet | 8.7 GB on disk, 76 files, median 625,414 rows per file, max 3,371,240 |
| row groups | 107 total, median 513,372 rows, max 1,048,576 |

The disk is the reason this recipe exists in the shape it does. Random access
against parquet is impossible because the finest seek granularity is a row group,
and one of yours costs 396 MB decompressed at the median. Random access against a
cold array on this disk is also impossible: 329 IOPS means a 2048 row batch of
scattered reads takes about 6 seconds. The whole design is built around getting
the array resident in RAM once and keeping it there.

## Design decisions this recipe locks in

**float16 for the embeddings, float32 for distance, stored separately.** The
193 feature array in float32 is 45.8 GB, which does not fit in 58 GB of RAM
alongside torch with `Swap: 0`. In float16 the 192 embedding dims come to
22.76 GB, which fits with room to spare. Measured on 135M real embedding values:
observed range `[-0.5039, 0.5118]`, max float16 roundtrip absolute error
`1.06e-04`, zero values that are not finite. That error is four orders of
magnitude below the signal, so the cast is free.

`dist_to_developed_2019_m` stays float32 in its own array. It is on a meters
scale reaching past 3,400 m, where float16 spacing is 4 m against a 30 m pixel.
Its own array costs 237 MB, which is nothing.

**The label is binarized at write time, distance is not normalized.** The source
`label` is three class: 0 remained wetland, 1 converted to developed, 2 converted
to other. It is written as `uint8` with 1 staying 1 and both 0 and 2 becoming 0,
so the stored array is binary `{0, 1}` and the target is "did this wetland
convert specifically to developed land," not "did it change at all."

Distance stays in raw meters, never normalized, because normalization stats must
come from each fold's train blocks only. Baking those in would freeze a fold
dependent quantity into the artifact and leak val distribution into the input
scale.

The tradeoff on the label: collapsing 2 into 0 is irreversible without a full
rebuild, and "exclude the 223,253 label 2 rows entirely" is a defensible
alternative target definition, since a wetland that converted to something other
than developed is not obviously a clean negative for this question. If you want
that door left open, also write `label_raw.npy` as the untouched three class
`uint8`. It costs 59 MB against a 24 GB artifact and saves a 30 minute rebuild.
Training reads `label.npy` either way.

**Rows are ordered block contiguous, and that ordering is a contract.** Every
row of a given `block_id` occupies one contiguous span. This makes validation a
sequential sweep instead of 11.9M scattered reads, and it gives the verification
step in step 4 something independent to check against. A `.npy` is a bare blob
with no schema, so the row order and the column order are the two things that can
silently be wrong, and the manifest plus verification exist to catch exactly that.

**Do not write a memmap and then read it with `np.fromfile`.** Reading the
22.76 GB array off this disk cold takes about 14 minutes at 27.7 MB/s, which is
slower than just decoding the 8.7 GB of parquet again from scratch (measured at
1.8 minutes). The array is only worth having if you `mmap` it and let the page
cache hold it across processes. Step 5 covers this.

## File layout

```
data/
  arrays/
    emb.npy          # (59267331, 192) float16   22.76 GB
    dist.npy         # (59267331,)     float32   237 MB
    label.npy        # (59267331,)     uint8     59 MB, binary 0/1, label 2 folded into 0
    label_raw.npy    # (59267331,)     uint8     59 MB, optional, untouched 0/1/2
    block_code.npy   # (59267331,)     int16     119 MB, index into manifest block_ids
    xy.npy           # (59267331, 2)   float32   474 MB, projected x/y for maps
    rowcol.npy       # (59267331, 2)   int32     474 MB, source raster row/col
    manifest.json    # provenance, column order, block spans, ordering contract
```

About 24.1 GB total against 171 GB free. `xy` and `rowcol` are not needed for
training but they are cheap and you will want them the first time a result needs
a map or a spatial sanity check, and regenerating them means another 20 minute
pass.

Suggested modules: `build_arrays.py` for the converter (run once, as
`python build_arrays.py`), and `arrays.py` for the load, populate, and index
helpers that `datasets.py` will import.

## Step 0: metadata pass, no data read

You need the total row count to preallocate, and per file write offsets so each
file's rows land in the right place.

1. `files = sorted(glob("data/alphaearth_wetland_joined/*.parquet"))`. Sort it
   and record the sorted list in the manifest. The row order contract depends on
   this list, so a file added or renamed later invalidates every saved index.
2. For each file, read `pq.ParquetFile(f).metadata.num_rows`. This reads the
   footer only, no column data, and takes milliseconds for all 76.
3. `offsets = np.concatenate([[0], np.cumsum(num_rows)])`. `offsets[i]` is where
   file `i` starts writing. `offsets[-1]` must equal 59,267,331.

Sanity check: assert `offsets[-1]` matches `population_stats.json`'s
`total_rows`. If it does not, the source data changed and everything downstream
of this file needs rebuilding.

## Step 1: preallocate the arrays

Use `np.lib.format.open_memmap(path, mode="w+", dtype=..., shape=...)` rather
than a raw `np.memmap`. It writes a real `.npy` header, so the file is self
describing: `np.load(path, mmap_mode="r")` recovers shape and dtype without you
passing them, which removes one whole class of "I passed the wrong shape and got
garbage" bug.

Creating a 22.76 GB sparse file is instant. The disk cost comes as you write.

## Step 2: convert one file at a time, writing sequentially

For each file `i` in the sorted list, at write offset `off = offsets[i]`:

1. Read the `block_id` column alone. Compute
   `perm = np.argsort(block_id, kind="stable")`. Stable matters: it keeps the
   original parquet row order inside each block, so the ordering is fully
   determined by the file list and nothing else.

2. Sorting inside a file is enough to get globally contiguous blocks, because no
   `block_id` spans more than one parquet file. This is verified: a full scan of
   all 76 files found 0 of 1,727 blocks spanning more than one file. Still
   **keep it as a live assertion in the script**, since it is a property of the
   upstream join rather than a guarantee: hold a running set of blocks already
   written and fail loudly if a later file contains one of them. If it ever
   fires, the block contiguous contract is broken and step 4's verification will
   not save you.

3. Build the embedding block in the layout that is actually fast. Filling a
   `(n_rows, 192)` row major array column by column is a strided write and
   measured 4.1 minutes across the dataset. Filling a `(192, n_rows)` array and
   transposing measured 1.8 minutes for the same result, a bit over 2x faster.
   So: allocate `tmp = np.empty((192, n_rows), dtype=np.float16)`, fill
   `tmp[j] = column_j_as_numpy[perm]` for each of the 192 columns, then assign
   `emb[off:off+n_rows] = tmp.T`.

   Peak transient memory is the arrow table for one file plus `tmp`. On the
   largest file (3,371,240 rows) that is roughly 2.6 GB plus 1.3 GB. Fine against
   42 GB free, but it does mean you should not run this concurrently with a
   training job.

4. Apply the same `perm` to `dist`, `label`, `block_code`, `xy`, `rowcol` and
   write each at `off`. `block_code` is the position of the row's `block_id` in
   the manifest's block list, as `int16` (1,727 blocks fits comfortably).

5. Binarize the label on the way in, and get the rule from exactly one place.
   `label_utils.binarize_label` already owns it, so call that and cast to
   `uint8`: it returns float32, which is what the training loss wants but four
   times the size you want on disk. Do not reimplement the rule inline in the
   converter, or you will have two definitions of the target that can drift.

   Two ways to get this backwards, both silent:

   - `label != 0` sends label 2 to positive, which is the exact opposite of what
     you want. It has to be `label == 1`.
   - Summing the raw label column as a shortcut for counting positives makes
     every label 2 row contribute 2. An earlier pass over this data did that and
     reported 612,414 positives at a 1.03% rate. The real numbers are 165,908
     and 0.28%.

   Both are caught by step 4 check 1, which is why that check is not optional.

6. Accumulate the three class counts **before** binarizing, as a running
   `np.bincount(raw_label, minlength=3)` across files. Once 2 has been folded
   into 0 the artifact no longer records that 223,253 rows were label 2, so the
   manifest becomes the only place that fact survives. Write it there.

7. Count values that are not finite in the embedding columns and in `dist`, and
   accumulate the total. I measured zero across 135M embedding values, so this
   should stay zero. Do not silently drop bad rows the way
   `compute_normalization_stats` currently does per batch. If the count is not
   zero, record the affected global row indices in the manifest so training can
   exclude them by index, which is reproducible, instead of dropping them
   differently on every pass.

8. `flush()` after each file so a crash halfway through leaves a diagnosable
   partial file rather than a mystery.

Expect 20 to 30 minutes end to end. It is write bound: 24.1 GB at about
28 MB/s is roughly 14 minutes on its own, plus 5 minutes to read the parquet
cold, plus decode.

## Step 3: block index and manifest

Once the write is done, derive the block spans from `block_code` in one pass:
`starts = np.searchsorted(block_code, np.arange(n_blocks), side="left")` and
likewise for `"right"`. This works only because the array is block contiguous,
which is a nice side benefit: if the spans come out non contiguous, the ordering
contract is already broken.

Write `manifest.json` with all of:

- source directory, and per file: name, `num_rows`, byte size, mtime
- the sorted file list, in order, as the row order contract
- the 192 embedding column names **in exact write order**, taken from
  `features.FEATURE_COLS` with `dist_to_developed_2019_m` removed
- dtype and shape of every array
- `block_ids` list, where the position in this list is the `block_code` value
- per block `(start, count)`
- label semantics, spelled out: source is three class (0 remained wetland,
  1 converted to developed, 2 converted to other), stored **binarized** as
  `label == 1`, with 0 and 2 both written as 0
- the three class counts as observed before binarizing, `[58878170, 165908,
  223253]`, since the artifact itself can no longer tell you these
- whether `label_raw.npy` was written
- the ordering contract in words: files sorted by name, then stable sort by
  `block_id` within each file
- count of values that are not finite, and the row indices if any
- observed embedding min and max, and the max float16 roundtrip error
- `built_utc` and the repo git commit

The column order entry is the one that matters most. Have `arrays.py` assert on
load that the manifest's embedding column list equals
`[c for c in FEATURE_COLS if c != "dist_to_developed_2019_m"]`, element by
element, and refuse to run otherwise. If `features.py` changes and the arrays do
not, you will train on scrambled features with no error and no clue.

## Step 4: verification, before trusting anything

Run all four. The first three are seconds off the small arrays, the fourth is a
couple of minutes. Checks 1 and 2 have known good target values because I ran the
equivalent scan over the parquet directly (65 seconds, reading only `label` and
`block_id`), so a mismatch means the conversion is wrong, not that the target is
unknown.

1. **Class counts, after binarizing.** `np.bincount(label, minlength=2)` must be
   exactly `[59101423, 165908]`, summing to 59,267,331, and `label.max()` must be
   1 so you know no 2 survived. Those two numbers are also
   `population_stats.json`'s `total_negative` and `total_positive`, which were
   produced independently, so assert against that file rather than against
   literals in the script.

   The negative count is the sum of the two collapsed classes,
   58,878,170 plus 223,253, which is where 59,101,423 comes from. If you get
   58,878,170 negatives you dropped the label 2 rows instead of folding them in;
   if you get 389,161 positives you binarized with `!= 0`. Both are confirmed
   against the source parquet, so a mismatch here is your bug, not an unknown.

2. **Per block row counts.** The `(start, count)` spans must reproduce
   `population_stats.json`'s `block_row_counts` for all 1,727 blocks, exactly.
   Confirmed that the parquet itself reproduces that file exactly, so it is a
   valid target. This is the strongest single check in the recipe: it is an
   independently produced record, and matching it for every block means both the
   row ordering and the per file offsets are almost certainly correct.

3. **Block contiguity.** Assert `block_code` is non decreasing across the whole
   array, and that the number of distinct runs equals 1,727. Cheap and it
   directly tests the contract.

4. **Round trip a sample.** Pick 3 small files and 1 large one. For each, rebuild
   its expected sorted block ordering from the parquet independently, then
   compare against the memmap slice at that file's offset range: `dist`, `label`,
   `block_code`, `rowcol` exactly, and `emb` with `atol=2e-3` to allow the
   float16 cast. This is what catches an off by one offset or a transposed
   column, which checks 1 through 3 can all pass through.

Do not skip 4 because 1 through 3 passed. A column permutation bug leaves all
three of those green.

## Step 5: reading the arrays back

This is where the earlier measurements matter, so it is worth being explicit
about it in `arrays.py`.

**Open with `mmap_mode="r"`, then populate.** Getting bytes into the page cache
and getting pages mapped into your process's page table are two different things.
A fresh process mmapping an already fully cached 22.76 GB file still starts at
5.65 ms per batch with about 1,169 minor page faults per batch, and only decays
toward 0.66 ms as pages fault in one at a time. Collapse that at startup by
touching one byte per 4 KB page:

```
np.load(path, mmap_mode="r").reshape(-1).view(np.uint8)[::4096].sum()
```

Measured at 1.9 seconds for the whole 22.76 GB when the page cache is warm. Every
process needs it, and every DataLoader worker that is not forked from an already
populated parent needs it.

If the page cache is cold, that same populate pass is a 14 minute sequential read.
So warm it once deliberately after a reboot, and expect the first run of the day
to be slow.

**Build the fold indices in RAM, do not cache them.** From `label` and
`block_code`, both tiny:

- map block to fold with `fold_of_row = fold_of_code[block_code]`, where
  `fold_of_code` is a length 1,727 lookup built from `folds.assign_folds`. This
  is a vectorized gather, instant, and it replaces the `pc.field("block_id").isin`
  predicate pushdown entirely.
- `is_train = fold_of_row != k`, then
  `train_pos = np.flatnonzero(is_train & (label == 1)).astype(np.int32)` and
  `train_neg = np.flatnonzero(is_train & (label == 0)).astype(np.int32)`. Since
  `label` is already binary, `label == 0` is the whole negative class and there
  is no third value to think about, which is the main thing binarizing at write
  time buys you here.
- val is the contiguous spans of blocks in fold `k`, so validation sweeps
  sequentially instead of gathering.

The whole thing is a few passes over 59M element arrays, well under a second, so
there is no reason to persist it. `train_neg` is the big one at 189 MB as int32.

Because `label` is already `{0, 1}`, nothing downstream should call
`label_utils.binarize_label` again. It is idempotent so it would not corrupt
anything, but a second call is a signal that someone thinks the stored label is
still three class, and that confusion is worth catching in review.

Both index arrays must be built from train blocks only. Building the sampler off
a global positive list and forgetting the fold mask leaks val rows into training,
val AUPRC comes back looking excellent, and nothing errors.

**In the batch path, do not cast to float32 and do not write into a reserved
column.** Both are slower than the gather they follow. Measured, batch size 2048,
fully warm:

| batch path | ms/batch | batches/s |
|---|---|---|
| gather only, float16 out | 0.47 | 2,136 |
| hand float16 to torch, cast on device | 0.66 | 1,526 |
| reused buffer plus `np.copyto` | 1.94 | 515 |
| `astype(np.float32)` then copy into `(2048, 193)` | 4.63 | 216 |

`astype(np.float32)` on 2048 x 192 float16 costs 1.26 ms, close to 3x the gather
itself. Reserving column 0 of the output for `dist` makes every row copy strided
and costs more again. Keep `dist` as a separate tensor, hand torch the float16
embedding tensor, and do `.to(device, non_blocking=True).float()`. Once the T4 is
usable that also halves the transfer.

Sorting indices within a batch is not worth the code. It helps while pages are
still faulting in and is within noise once warm.

## What changes downstream

| current | after this |
|---|---|
| `StreamingTrainDataset` plus shuffle buffer | deleted, the sampler shuffles |
| `StreamingValDataset` | sequential slice over val block spans |
| `_block_id_filter` and parquet predicate pushdown | deleted, replaced by `fold_of_code[block_code]` |
| `compute_normalization_stats` filtered parquet scan | masked mean and std over one 237 MB array |
| `folds.log_fold_stats` parquet scan | instant, off `label` and `block_code` |
| `population_stats.py` | mostly subsumed by the manifest, but keep it: step 4 checks 1 and 2 use its output as an independent record |
| `label_utils.binarize_label` called per batch in both streams | called once, in the converter, and nowhere else |
| `num_workers=0` forced by `IterableDataset` | free to raise, though 0 or 2 is plenty (see below) |

## Gotchas worth writing down

- The arrays are a derived cache, not source data. Add `data/arrays/` to
  `.gitignore`. The parquet stays the source of truth.
- Any change to `FEATURE_COLS` invalidates the artifact. The manifest assert in
  step 3 is what turns that from a silent wrong answer into a crash.
- The stored label is binary and the label 2 rows are no longer distinguishable
  in `label.npy`. If the target definition ever changes to "exclude wetlands that
  converted to other" rather than "count them as negative," that is a full
  rebuild unless you wrote `label_raw.npy`. The manifest keeps the three class
  counts so at least you can tell how many rows the change would touch: 223,253,
  which is 0.38% of the dataset and 1.35x the entire positive class.
- Do not run the conversion while training. Peak transient is about 4 GB and the
  disk is the bottleneck for both.
- If RAM ever gets tight, the lever is dropping the 2017 and 2018 embedding
  years: 64 dims instead of 192 takes the array from 22.76 GB to 7.6 GB. That is
  a modeling decision, not a plumbing one, so measure the cost in AUPRC before
  reaching for it.
- Page cache is evictable and there is no swap. If something else on the box eats
  20 GB, pages get dropped and each one costs about 3 ms to fault back at
  329 IOPS. Training will quietly fall from 1,500 batches per second to single
  digits rather than fail. If you see that, check `free -g` before debugging the
  model.
- For the current linear baseline, `nn.Linear(193, 1)` forward plus backward plus
  step is 0.57 ms per batch on CPU, so a 0.66 ms data path already keeps up with
  one worker. `num_workers` only starts earning its keep with a real model on the
  GPU, and each worker pays its own populate pass.
