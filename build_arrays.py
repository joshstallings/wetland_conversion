"""
Converts the joined AlphaEarth + NLCD parquet at data/alphaearth_wetland_joined_2019_2024
into flat, row addressable .npy arrays under data/arrays_2019_2024, so a map style Dataset
can index single rows at random instead of streaming whole row groups.

Run once:  python build_arrays.py

Row order contract: files sorted by name, then a stable sort by block_id within
each file. No block_id spans more than one parquet file (verified live below), so
that is enough to put every row of a block in one contiguous span. Contiguity is
what lets validation sweep sequentially and lets the per block spans fall out of a
searchsorted instead of a groupby.

block_code is the position of a block in the manifest's block_ids list, and that
list is in write order, not sorted order. Sorted block_id order is not monotone
with sorted file order in this dataset (it breaks at r039_c055), so coding blocks
by their rank in a sorted list would leave block_code non monotone and quietly
break the searchsorted in block_spans.

A .npy is a bare blob with no schema, so row order and column order are the two
things that can be wrong with no error. The manifest records both and step 4
checks them against data/population_stats.json, which came from an independent
scan of the parquet.

These arrays are a derived cache. The parquet stays the source of truth.
"""

import argparse
import glob
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from features import DIST_COL, EMB_COLS, FEATURE_COLS, LABEL_COL
from label_utils import binarize_label
from population_stats import POPULATION_STATS_PATH, load_population_stats

SOURCE_PARQUET_PATH = "data/alphaearth_wetland_joined_2019_2024"
ARRAY_DIR = "data/arrays_2019_2024"

# EMB_COLS is the column order of emb.npy and lives in features.py so the reader
# side asserts against the same list. Anything that changes FEATURE_COLS
# invalidates the artifact, which is why the list goes in the manifest verbatim.

BLOCK_COL = "block_id"
XY_COLS = ["x", "y"]
ROWCOL_COLS = ["row", "col"]

# float16 roundtrip error is a diagnostic, not a correctness check, so it comes
# off every 16th row rather than a full extra pass over 11.4 billion values. The
# guarantee that nothing overflowed comes from the exhaustive min/max below.
ROUNDTRIP_STRIDE = 16

# Tolerance for step 4's embedding comparison. float16 spacing near 0.5 is about
# 5e-4, so 2e-3 passes a correct cast and still fails a shifted or transposed one.
EMB_ATOL = 2e-3


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def scan_metadata(source_parquet_path):
    """
    Step 0. Footer only read of all 76 files: no column data touched, milliseconds.

    Returns (files, num_rows, offsets, file_meta). offsets[i] is the global row
    where file i starts writing; offsets[-1] is the total row count.
    """
    files = sorted(glob.glob(str(Path(source_parquet_path) / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files under {source_parquet_path}")

    num_rows, file_meta = [], []
    for f in files:
        n = pq.ParquetFile(f).metadata.num_rows
        st = Path(f).stat()
        num_rows.append(n)
        file_meta.append(
            {
                "name": Path(f).name,
                "num_rows": int(n),
                "bytes": int(st.st_size),
                "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
            }
        )

    offsets = np.concatenate([[0], np.cumsum(num_rows)]).astype(np.int64)
    return files, np.asarray(num_rows, dtype=np.int64), offsets, file_meta


def open_arrays(out_dir, n_rows, mode, write_label_raw=True):
    """
    Steps 1 and 5's other half. open_memmap rather than np.memmap so the files get
    a real .npy header and np.load(path, mmap_mode="r") recovers shape and dtype
    without the caller passing them, which removes the whole class of wrong shape
    bugs. mode "w+" preallocates sparse (instant); the disk cost lands on write.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "emb": (np.float16, (n_rows, len(EMB_COLS))),
        "dist": (np.float32, (n_rows,)),
        "label": (np.uint8, (n_rows,)),
        "block_code": (np.int16, (n_rows,)),
        "xy": (np.float32, (n_rows, 2)),
        "rowcol": (np.int32, (n_rows, 2)),
    }
    if write_label_raw:
        spec["label_raw"] = (np.uint8, (n_rows,))

    arrays = {}
    for name, (dtype, shape) in spec.items():
        path = out_dir / f"{name}.npy"
        if mode == "w+":
            arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        else:
            arrays[name] = np.load(path, mmap_mode=mode)
    return arrays


def _block_codes(table, code_of_block, seen_blocks, file_idx):
    """
    Per row global block codes for one file, plus the stable permutation that puts
    the file's rows in ascending block order.

    Sorts on the integer codes rather than the block_id strings: dictionary
    encoding the column first turns a 3.4M element object dtype string sort into an
    int32 sort. Codes are handed out in ascending block_id order within the file,
    so sorting by code is the same ordering as sorting by block_id.

    Asserts no block has already been written by an earlier file. That property
    holds today (a full scan found 0 of 1,727 blocks spanning files) but it is a
    property of the upstream join, not a guarantee, and if it ever breaks the
    contiguity contract is gone and step 4's verification will not catch it.
    """
    bid = table.column(BLOCK_COL)
    if isinstance(bid, pa.ChunkedArray):
        bid = bid.combine_chunks()
    enc = pc.dictionary_encode(bid)
    if isinstance(enc, pa.ChunkedArray):
        enc = enc.combine_chunks()

    dictionary = enc.dictionary.to_pylist()
    indices = enc.indices.to_numpy(zero_copy_only=False)

    for b in sorted(dictionary):
        if b in seen_blocks:
            raise AssertionError(
                f"block {b} appears in file {file_idx} and again in file "
                f"{seen_blocks[b]}. Blocks are assumed not to span parquet files; "
                f"the block contiguous row order contract is broken."
            )
        seen_blocks[b] = file_idx
        code_of_block[b] = len(code_of_block)

    code_of_dict_pos = np.array([code_of_block[b] for b in dictionary], dtype=np.int32)
    row_codes = code_of_dict_pos[indices]
    perm = np.argsort(row_codes, kind="stable")
    return row_codes, perm


def convert(files, offsets, out_dir, write_label_raw=True):
    """
    Step 2. One file at a time, written sequentially at its global offset.

    Returns (arrays, stats, block_ids_in_write_order).
    """
    n_total = int(offsets[-1])
    arrays = open_arrays(out_dir, n_total, mode="w+", write_label_raw=write_label_raw)

    columns = EMB_COLS + [DIST_COL, LABEL_COL, BLOCK_COL] + XY_COLS + ROWCOL_COLS

    code_of_block, seen_blocks = {}, {}
    raw_class_counts = np.zeros(3, dtype=np.int64)
    n_nonfinite_emb = 0
    n_nonfinite_dist = 0
    nonfinite_rows = []
    emb_min, emb_max = np.inf, -np.inf
    max_roundtrip_err = 0.0

    t0 = time.time()
    for i, f in enumerate(files):
        off = int(offsets[i])
        n = int(offsets[i + 1]) - off
        if n == 0:
            continue

        table = pq.read_table(f, columns=columns)
        row_codes, perm = _block_codes(table, code_of_block, seen_blocks, i)

        # Fill (192, n) and transpose on write. Filling an (n, 192) row major array
        # column by column is a strided write and measured 4.1 minutes across the
        # dataset; this way measured 1.8 minutes for the same bytes.
        tmp = np.empty((len(EMB_COLS), n), dtype=np.float16)
        for j, name in enumerate(EMB_COLS):
            col = table.column(name).to_numpy(zero_copy_only=False)[perm]
            tmp[j] = col

            sub = np.abs(col[::ROUNDTRIP_STRIDE] - tmp[j, ::ROUNDTRIP_STRIDE].astype(np.float32))
            sub = sub[np.isfinite(sub)]
            if sub.size:
                max_roundtrip_err = max(max_roundtrip_err, float(sub.max()))

        dist = table.column(DIST_COL).to_numpy(zero_copy_only=False)[perm]
        raw_label = table.column(LABEL_COL).to_numpy(zero_copy_only=False)[perm]

        # Counts before binarizing. Once 2 is folded into 0 the artifact can no
        # longer tell you how many rows were label 2, so the manifest is the only
        # place that fact survives.
        counts = np.bincount(raw_label, minlength=3)
        if len(counts) != 3:
            raise AssertionError(
                f"{Path(f).name}: label column has values outside {{0, 1, 2}} "
                f"(saw up to {int(raw_label.max())})"
            )
        raw_class_counts += counts

        # The rule for the target lives in label_utils and nowhere else. Two silent
        # ways to get this backwards: label != 0 sends label 2 to positive (the
        # opposite of the target), and summing the raw column as a positive count
        # makes every label 2 row contribute 2, which is where an earlier pass got
        # 612,414 positives at 1.03% instead of 165,908 at 0.28%. Check 1 catches both.
        label = binarize_label(raw_label).astype(np.uint8)

        # Do not silently drop rows that are not finite the way the old per batch
        # normalization pass did. Record the global indices instead so training can
        # exclude the same rows every run.
        finite_emb = np.isfinite(tmp).all(axis=0)
        finite_dist = np.isfinite(dist)
        n_nonfinite_emb += int(finite_emb.size - finite_emb.sum())
        n_nonfinite_dist += int(finite_dist.size - finite_dist.sum())
        bad = np.flatnonzero(~(finite_emb & finite_dist))
        if bad.size:
            nonfinite_rows.extend((off + bad).tolist())

        good = tmp if finite_emb.all() else tmp[:, finite_emb]
        if good.size:
            emb_min = min(emb_min, float(good.min()))
            emb_max = max(emb_max, float(good.max()))

        arrays["emb"][off:off + n] = tmp.T
        arrays["dist"][off:off + n] = dist
        arrays["label"][off:off + n] = label
        arrays["block_code"][off:off + n] = row_codes[perm].astype(np.int16)
        arrays["xy"][off:off + n] = np.stack(
            [table.column(c).to_numpy(zero_copy_only=False)[perm] for c in XY_COLS], axis=1
        )
        arrays["rowcol"][off:off + n] = np.stack(
            [table.column(c).to_numpy(zero_copy_only=False)[perm] for c in ROWCOL_COLS], axis=1
        )
        if write_label_raw:
            arrays["label_raw"][off:off + n] = raw_label.astype(np.uint8)

        # Flush per file so a crash halfway leaves a diagnosable partial artifact.
        for a in arrays.values():
            a.flush()

        del tmp, table
        done = off + n
        rate = done / max(time.time() - t0, 1e-9)
        eta = (n_total - done) / max(rate, 1e-9)
        print(
            f"[{i + 1:>2}/{len(files)}] {Path(f).name}  rows {n:>9,}  "
            f"at {off:>11,}  {rate / 1e6:5.2f}M rows/s  eta {eta / 60:5.1f} min"
        )

    if len(code_of_block) >= np.iinfo(np.int16).max:
        raise AssertionError(f"{len(code_of_block)} blocks does not fit in int16 block_code")

    stats = {
        "raw_class_counts": raw_class_counts.tolist(),
        "n_nonfinite_emb": n_nonfinite_emb,
        "n_nonfinite_dist": n_nonfinite_dist,
        "nonfinite_row_indices": nonfinite_rows,
        "emb_min": emb_min,
        "emb_max": emb_max,
        "emb_max_float16_roundtrip_err": max_roundtrip_err,
        "roundtrip_err_row_stride": ROUNDTRIP_STRIDE,
        "convert_seconds": round(time.time() - t0, 1),
    }
    block_ids = [b for b, _ in sorted(code_of_block.items(), key=lambda kv: kv[1])]
    return arrays, stats, block_ids


def block_spans(block_code, n_blocks):
    """
    Step 3. Per block (start, count) in one pass. Only valid because the array is
    block contiguous, which makes it a free check on the contract: if the spans come
    out non contiguous the row ordering is already wrong.
    """
    code = np.asarray(block_code)
    starts = np.searchsorted(code, np.arange(n_blocks), side="left")
    ends = np.searchsorted(code, np.arange(n_blocks), side="right")
    return starts.astype(np.int64), (ends - starts).astype(np.int64)


def write_manifest(out_dir, source_parquet_path, files, file_meta, arrays, stats,
                   block_ids, starts, counts, write_label_raw):
    """
    Everything needed to know what these bytes are, since the .npy files carry
    nothing but shape and dtype.
    """
    manifest = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "source_parquet_path": str(source_parquet_path),
        "source_files": file_meta,
        "file_order": [m["name"] for m in file_meta],
        "row_order_contract": (
            "files sorted by name, then a stable sort by block_id within each file. "
            "Every row of a block occupies one contiguous span. block_code is the "
            "index into block_ids, which is in write order, not sorted block_id order."
        ),
        "total_rows": int(sum(m["num_rows"] for m in file_meta)),
        "emb_columns": list(EMB_COLS),
        "dist_column": DIST_COL,
        "xy_columns": list(XY_COLS),
        "rowcol_columns": list(ROWCOL_COLS),
        "arrays": {
            name: {"dtype": str(a.dtype), "shape": list(a.shape)} for name, a in arrays.items()
        },
        "label_semantics": {
            "source_column": LABEL_COL,
            "source_classes": {
                "0": "remained wetland",
                "1": "converted to developed",
                "2": "converted to other, not developed",
            },
            "stored_rule": "label == 1, so 0 and 2 are both written as 0",
            "stored_dtype": "uint8, binary 0/1",
            "raw_class_counts": stats["raw_class_counts"],
            "label_raw_written": bool(write_label_raw),
            "note": (
                "label.npy is already binary. Nothing downstream should call "
                "label_utils.binarize_label on it again."
            ),
        },
        "distance_units": "meters, raw, never normalized. Fold train stats belong in the trainer.",
        "block_ids": block_ids,
        "block_spans": [[int(s), int(c)] for s, c in zip(starts, counts)],
        "n_blocks": len(block_ids),
        "float16_cast": {
            "emb_min": stats["emb_min"],
            "emb_max": stats["emb_max"],
            "max_roundtrip_abs_err": stats["emb_max_float16_roundtrip_err"],
            "roundtrip_err_row_stride": stats["roundtrip_err_row_stride"],
        },
        "nonfinite": {
            "n_rows_emb": stats["n_nonfinite_emb"],
            "n_rows_dist": stats["n_nonfinite_dist"],
            "row_indices": stats["nonfinite_row_indices"],
        },
        "convert_seconds": stats["convert_seconds"],
    }

    path = Path(out_dir) / "manifest.json"
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def verify(out_dir, source_parquet_path, files, offsets, arrays, manifest,
           population_stats_path=POPULATION_STATS_PATH):
    """
    Step 4. All four checks. Checks 1 and 2 assert against population_stats.json
    rather than literals, because that file was produced by an independent scan of
    the parquet, so a mismatch means the conversion is wrong, not that the target
    is unknown.
    """
    pop = load_population_stats(population_stats_path)
    block_ids = manifest["block_ids"]
    label = arrays["label"]
    block_code = arrays["block_code"]
    failures = []

    # 1. Class counts after binarizing.
    counts = np.bincount(np.asarray(label), minlength=2)
    n_neg, n_pos = int(counts[0]), int(counts[1])
    print(f"check 1  negatives {n_neg:,}  positives {n_pos:,}  max {int(label.max())}")
    if len(counts) != 2 or int(label.max()) > 1:
        failures.append("label.npy holds a value above 1, so a label 2 survived binarizing")
    if n_pos != pop["total_positive"] or n_neg != pop["total_negative"]:
        # 58,878,170 negatives means the label 2 rows got dropped instead of folded
        # in; 389,161 positives means the binarize used != 0 instead of == 1.
        failures.append(
            f"class counts {n_neg:,}/{n_pos:,} do not match population_stats "
            f"{pop['total_negative']:,}/{pop['total_positive']:,}"
        )
    if n_neg + n_pos != pop["total_rows"]:
        failures.append(f"{n_neg + n_pos:,} rows against population_stats {pop['total_rows']:,}")

    # 2. Per block row counts against the independent record. Strongest check here:
    # matching all 1,727 means both the within file ordering and the per file
    # offsets are almost certainly right.
    starts, block_counts = block_spans(block_code, len(block_ids))
    expected = pop["block_row_counts"]
    if set(block_ids) != set(expected):
        failures.append(
            f"block set differs from population_stats: {len(set(block_ids) - set(expected))} "
            f"only in arrays, {len(set(expected) - set(block_ids))} only in population_stats"
        )
    bad_blocks = [
        (b, int(c), expected[b])
        for b, c in zip(block_ids, block_counts)
        if b in expected and int(c) != expected[b]
    ]
    print(f"check 2  {len(block_ids) - len(bad_blocks)}/{len(block_ids)} block row counts match")
    if bad_blocks:
        failures.append(f"{len(bad_blocks)} blocks have the wrong row count, e.g. {bad_blocks[:5]}")

    # 3. Contiguity. Cheap and it tests the contract directly.
    code = np.asarray(block_code)
    n_desc = int(np.sum(np.diff(code) < 0))
    n_runs = 1 + int(np.sum(np.diff(code) != 0))
    print(f"check 3  descents {n_desc}  runs {n_runs} against {len(block_ids)} blocks")
    if n_desc:
        failures.append(f"block_code is not non decreasing ({n_desc} descents)")
    if n_runs != len(block_ids):
        failures.append(f"{n_runs} contiguous runs against {len(block_ids)} blocks")

    # 4. Round trip a sample. This is what catches an off by one offset or a
    # transposed column, all of which pass checks 1 through 3.
    order = np.argsort(offsets[1:] - offsets[:-1])
    sample = sorted(set(order[:3].tolist() + [int(order[-1])]))
    code_of_block = {b: i for i, b in enumerate(block_ids)}
    for i in sample:
        off, n = int(offsets[i]), int(offsets[i + 1] - offsets[i])
        table = pq.read_table(
            files[i], columns=EMB_COLS + [DIST_COL, LABEL_COL, BLOCK_COL] + ROWCOL_COLS
        )
        # Deliberately a different code path from the converter: sort the block_id
        # strings directly and look each code up per row, so a bug in the
        # dictionary encode path shows up as a mismatch here.
        bid = table.column(BLOCK_COL).to_numpy(zero_copy_only=False)
        perm = np.argsort(bid, kind="stable")
        exp_code = np.array([code_of_block[b] for b in bid[perm]], dtype=np.int16)
        exp_dist = table.column(DIST_COL).to_numpy(zero_copy_only=False)[perm]
        exp_label = binarize_label(table.column(LABEL_COL).to_numpy(zero_copy_only=False)[perm])
        exp_rowcol = np.stack(
            [table.column(c).to_numpy(zero_copy_only=False)[perm] for c in ROWCOL_COLS], axis=1
        )

        problems = []
        if not np.array_equal(np.asarray(arrays["block_code"][off:off + n]), exp_code):
            problems.append("block_code")
        if not np.array_equal(np.asarray(arrays["dist"][off:off + n]), exp_dist):
            problems.append("dist")
        if not np.array_equal(np.asarray(arrays["label"][off:off + n]), exp_label.astype(np.uint8)):
            problems.append("label")
        if not np.array_equal(np.asarray(arrays["rowcol"][off:off + n]), exp_rowcol):
            problems.append("rowcol")

        # Chunked so the largest file does not need a 2.6 GB float32 copy of its
        # embeddings alongside the arrow table.
        worst = 0.0
        for s in range(0, n, 200_000):
            e = min(s + 200_000, n)
            got = np.asarray(arrays["emb"][off + s:off + e], dtype=np.float32)
            exp = np.stack(
                [table.column(c).to_numpy(zero_copy_only=False)[perm[s:e]] for c in EMB_COLS],
                axis=1,
            )
            d = np.abs(got - exp)
            d = d[np.isfinite(d)]
            if d.size:
                worst = max(worst, float(d.max()))
        if worst > EMB_ATOL:
            problems.append(f"emb (max abs diff {worst:.2e})")

        print(
            f"check 4  {Path(files[i]).name}  rows {n:>9,}  emb max abs diff {worst:.2e}  "
            f"{'ok' if not problems else 'MISMATCH ' + ', '.join(problems)}"
        )
        if problems:
            failures.append(f"{Path(files[i]).name}: {', '.join(problems)} mismatch")
        del table

    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=SOURCE_PARQUET_PATH)
    ap.add_argument("--out-dir", default=ARRAY_DIR)
    ap.add_argument("--population-stats", default=POPULATION_STATS_PATH)
    ap.add_argument(
        "--skip-label-raw",
        action="store_true",
        help="do not write label_raw.npy. It costs 59 MB against a 24 GB artifact "
             "and is the only thing that saves a full rebuild if the target ever "
             "changes to excluding the label 2 rows rather than counting them negative",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="rerun step 4 against arrays already on disk, no conversion",
    )
    args = ap.parse_args()

    if len(EMB_COLS) != len(FEATURE_COLS) - 1 or DIST_COL in EMB_COLS:
        raise AssertionError(f"{DIST_COL} is not exactly one column of features.FEATURE_COLS")

    files, num_rows, offsets, file_meta = scan_metadata(args.source)
    pop = load_population_stats(args.population_stats)
    print(f"{len(files)} files, {int(offsets[-1]):,} rows, {len(EMB_COLS)} embedding columns")
    if int(offsets[-1]) != pop["total_rows"]:
        raise AssertionError(
            f"parquet has {int(offsets[-1]):,} rows, population_stats.json says "
            f"{pop['total_rows']:,}. The source data changed; rebuild "
            f"population_stats.py first, and rebuild everything downstream of it."
        )

    if args.verify_only:
        with open(Path(args.out_dir) / "manifest.json") as fh:
            manifest = json.load(fh)
        if manifest["file_order"] != [m["name"] for m in file_meta]:
            raise AssertionError("source file list changed since the manifest was written")
        if manifest["emb_columns"] != list(EMB_COLS):
            raise AssertionError(
                "features.FEATURE_COLS changed since the arrays were built. The stored "
                "embedding column order no longer matches, so every saved row index and "
                "every column position is wrong. Rebuild."
            )
        arrays = open_arrays(
            args.out_dir, int(offsets[-1]), mode="r",
            write_label_raw=manifest["label_semantics"]["label_raw_written"],
        )
    else:
        write_label_raw = not args.skip_label_raw
        arrays, stats, block_ids = convert(files, offsets, args.out_dir, write_label_raw)

        starts, counts = block_spans(arrays["block_code"], len(block_ids))
        manifest = write_manifest(
            args.out_dir, args.source, files, file_meta, arrays, stats,
            block_ids, starts, counts, write_label_raw,
        )

        raw = stats["raw_class_counts"]
        print(
            f"\nraw label counts  0 remained wetland {raw[0]:,}  "
            f"1 converted to developed {raw[1]:,}  2 converted to other {raw[2]:,}"
        )
        print(
            f"values not finite  emb rows {stats['n_nonfinite_emb']:,}  "
            f"dist rows {stats['n_nonfinite_dist']:,}"
        )
        print(
            f"emb range [{stats['emb_min']:.4f}, {stats['emb_max']:.4f}]  "
            f"max float16 roundtrip err {stats['emb_max_float16_roundtrip_err']:.2e}"
        )
        print(f"wrote {len(arrays)} arrays plus manifest.json in {stats['convert_seconds'] / 60:.1f} min\n")

    failures = verify(args.out_dir, args.source, files, offsets, arrays, manifest,
                      args.population_stats)
    if failures:
        print("\nVERIFICATION FAILED, do not train on these arrays:")
        for f in failures:
            print(f"  {f}")
        raise SystemExit(1)
    print("\nall four checks passed")


if __name__ == "__main__":
    main()
