"""
Builds an array artifact for a new label horizon without redoing the embeddings.

Only the labels change between data/arrays_2019_2024 and the 2019 to 2020
horizon. The embeddings, the distance feature, the coordinates, and the block
coding are all keyed to 2019, and the row order they sit in is a function of
file order and block_id, both unchanged. So emb.npy, dist.npy, xy.npy,
rowcol.npy, and block_code.npy are hardlinked rather than rewritten, which saves
regenerating a 22.76 GB file byte for byte identical to the one already on disk.

The row order equality is the whole basis for the hardlinks, so it is not
assumed. Every file's recomputed block codes are checked against the linked
block_code.npy, and its (row, col) against the linked rowcol.npy, before any
label is written.

    python relabel_arrays.py

Writes data/arrays_2019_2020/. Hardlinks share inodes with the 2019 to 2024
artifact: editing one edits the other, so treat both as read only.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import build_arrays
from build_arrays import BLOCK_COL, ROWCOL_COLS, _block_codes, block_spans, git_commit
from features import DIST_COL, EMB_COLS, LABEL_COL
from label_utils import binarize_label

SOURCE_PARQUET_PATH = "data/alphaearth_wetland_joined_2019_2020"
BASE_ARRAY_DIR = "data/arrays_2019_2024"
OUT_ARRAY_DIR = "data/arrays_2019_2020"

LINKED = ("emb", "dist", "xy", "rowcol", "block_code")
REBUILT = ("label", "label_raw")


def hardlink(base_dir, out_dir, names):
    for name in names:
        src, dst = Path(base_dir) / f"{name}.npy", Path(out_dir) / f"{name}.npy"
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
        assert dst.stat().st_ino == src.stat().st_ino
        print(f"  linked {name}.npy  {src.stat().st_size / 1e9:6.2f} GB  inode {dst.stat().st_ino}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=SOURCE_PARQUET_PATH)
    ap.add_argument("--base-arrays", default=BASE_ARRAY_DIR)
    ap.add_argument("--out-dir", default=OUT_ARRAY_DIR)
    args = ap.parse_args()

    with open(Path(args.base_arrays) / "manifest.json") as fh:
        base = json.load(fh)

    files, _, offsets, file_meta = build_arrays.scan_metadata(args.source)
    n_total = int(offsets[-1])

    # The linked arrays are only valid for this source if it presents the same
    # files in the same order with the same row counts. Everything else below
    # checks the ordering within a file; this checks the ordering between them.
    assert [m["name"] for m in file_meta] == base["file_order"], "source file list differs"
    assert n_total == base["total_rows"], f"{n_total:,} rows against {base['total_rows']:,}"
    print(f"{len(files)} files, {n_total:,} rows, matching the {args.base_arrays} layout")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    hardlink(args.base_arrays, args.out_dir, LINKED)

    linked_code = np.load(Path(args.out_dir) / "block_code.npy", mmap_mode="r")
    linked_rowcol = np.load(Path(args.out_dir) / "rowcol.npy", mmap_mode="r")

    label_mm = np.lib.format.open_memmap(
        Path(args.out_dir) / "label.npy", mode="w+", dtype=np.uint8, shape=(n_total,))
    label_raw_mm = np.lib.format.open_memmap(
        Path(args.out_dir) / "label_raw.npy", mode="w+", dtype=np.uint8, shape=(n_total,))

    code_of_block, seen_blocks = {}, {}
    raw_class_counts = np.zeros(3, dtype=np.int64)
    t0 = time.time()

    for i, f in enumerate(files):
        off, n = int(offsets[i]), int(offsets[i + 1] - offsets[i])
        if n == 0:
            continue
        table = pq.read_table(f, columns=[LABEL_COL, BLOCK_COL] + ROWCOL_COLS)
        row_codes, perm = _block_codes(table, code_of_block, seen_blocks, i)

        got_code = row_codes[perm].astype(np.int16)
        assert np.array_equal(np.asarray(linked_code[off:off + n]), got_code), (
            f"{Path(f).name}: block_code differs from {args.base_arrays}, so the row "
            f"order is not the same and the hardlinked arrays do not apply"
        )
        got_rowcol = np.stack(
            [table.column(c).to_numpy(zero_copy_only=False)[perm] for c in ROWCOL_COLS], axis=1)
        assert np.array_equal(np.asarray(linked_rowcol[off:off + n]), got_rowcol), (
            f"{Path(f).name}: rowcol differs from {args.base_arrays}"
        )

        raw_label = table.column(LABEL_COL).to_numpy(zero_copy_only=False)[perm]
        counts = np.bincount(raw_label, minlength=3)
        if len(counts) != 3:
            raise AssertionError(
                f"{Path(f).name}: label column has values outside {{0, 1, 2}} "
                f"(saw up to {int(raw_label.max())})")
        raw_class_counts += counts

        label_mm[off:off + n] = binarize_label(raw_label).astype(np.uint8)
        label_raw_mm[off:off + n] = raw_label.astype(np.uint8)
        label_mm.flush()
        label_raw_mm.flush()
        del table
        print(f"[{i + 1:>2}/{len(files)}] {Path(f).name}  rows {n:>9,}  at {off:>11,}", flush=True)

    block_ids = [b for b, _ in sorted(code_of_block.items(), key=lambda kv: kv[1])]
    assert block_ids == base["block_ids"], "block write order differs from the base artifact"
    starts, counts = block_spans(linked_code, len(block_ids))

    arrays = {n: np.load(Path(args.out_dir) / f"{n}.npy", mmap_mode="r")
              for n in LINKED + REBUILT}
    manifest = build_arrays.write_manifest(
        args.out_dir, args.source, files, file_meta, arrays,
        stats={
            "raw_class_counts": raw_class_counts.tolist(),
            # These describe emb.npy and dist.npy, which are the base artifact's
            # bytes, not anything this run produced. Carried over rather than
            # recomputed, which would mean reading 22.76 GB to restate a number.
            "n_nonfinite_emb": base["nonfinite"]["n_rows_emb"],
            "n_nonfinite_dist": base["nonfinite"]["n_rows_dist"],
            "nonfinite_row_indices": base["nonfinite"]["row_indices"],
            "emb_min": base["float16_cast"]["emb_min"],
            "emb_max": base["float16_cast"]["emb_max"],
            "emb_max_float16_roundtrip_err": base["float16_cast"]["max_roundtrip_abs_err"],
            "roundtrip_err_row_stride": base["float16_cast"]["roundtrip_err_row_stride"],
            "convert_seconds": round(time.time() - t0, 1),
        },
        block_ids=block_ids, starts=starts, counts=counts, write_label_raw=True,
    )

    # write_manifest has no notion of a partial rebuild, so the provenance that
    # distinguishes this artifact from a full one gets added after the fact
    manifest["derived_from"] = {
        "base_array_dir": str(args.base_arrays),
        "base_built_utc": base["built_utc"],
        "base_git_commit": base["git_commit"],
        "hardlinked_arrays": list(LINKED),
        "rebuilt_arrays": list(REBUILT),
        "note": (
            "emb, dist, xy, rowcol, and block_code are hardlinks to the base "
            "artifact and share its inodes. float16_cast and nonfinite describe "
            "those base bytes and were copied, not recomputed."
        ),
    }
    manifest["git_commit"] = git_commit()
    with open(Path(args.out_dir) / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    raw = raw_class_counts
    print(f"\n{n_total:,} rows in {time.time() - t0:.0f}s")
    print(f"raw label counts  0 remained wetland {raw[0]:,}  "
          f"1 converted to developed {raw[1]:,}  2 converted to other {raw[2]:,}")
    print(f"binarized  positives {int(np.asarray(label_mm).sum()):,}  "
          f"positive rate {100 * raw[1] / n_total:.4f}%")
    print(f"blocks {len(block_ids):,}")


if __name__ == "__main__":
    main()
