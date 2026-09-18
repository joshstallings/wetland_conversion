"""
Clones the joined AlphaEarth plus NLCD tiles onto a different label horizon.

Everything in data/alphaearth_wetland_joined_2019_2024 except `label` and
`label_name` is keyed to 2019: the 192 embedding bands, the pixel location, the
distance to 2019 development, and the tile and block ids. Swapping the horizon
therefore means rewriting two columns and copying the rest through untouched.

Two things stop this from being a positional overwrite. The join dropped pixels
with no AlphaEarth coverage, so the tiles hold fewer rows than the label file,
and a tile's native row order is its own. Rows are matched on (row, col).

    python relabel_joined.py

Writes data/alphaearth_wetland_joined_2019_2020/
"""

import glob
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_DIR = Path("data/alphaearth_wetland_joined_2019_2024")
DEST_DIR = Path("data/alphaearth_wetland_joined_2019_2020")
LABEL_PARQUET = Path("data/wetland_sample_labels_2019_2020.parquet")


def pixel_key(row, col):
    """One int64 per pixel, so matching is a sort and a searchsorted rather than
    a two column join. col maxes out around 27k, well inside the 32 bit half."""
    return row.astype(np.int64) * (1 << 32) + col.astype(np.int64)


def load_label_lookup(path):
    t = pq.read_table(path, columns=["row", "col", "label"])
    key = pixel_key(t["row"].to_numpy(), t["col"].to_numpy())
    label = t["label"].to_numpy()
    order = np.argsort(key)
    return key[order], label[order]


def main():
    assert SOURCE_DIR.is_dir(), f"Missing {SOURCE_DIR}"
    key_sorted, label_sorted = load_label_lookup(LABEL_PARQUET)
    print(f"label lookup: {len(key_sorted):,} pixels from {LABEL_PARQUET.name}")

    files = sorted(glob.glob(str(SOURCE_DIR / "*.parquet")))
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    class_counts = np.zeros(3, dtype=np.int64)
    t_start = time.time()

    for i, src in enumerate(files, 1):
        pf = pq.ParquetFile(src)
        # match the source's row group layout so the clone reads back the same way
        rg_size = max(pf.metadata.row_group(g).num_rows for g in range(pf.metadata.num_row_groups))
        table = pf.read()

        key = pixel_key(table["row"].to_numpy(), table["col"].to_numpy())
        pos = np.searchsorted(key_sorted, key)
        assert pos.max() < len(key_sorted) and np.array_equal(key_sorted[pos], key), (
            f"{Path(src).name}: some pixels are not in {LABEL_PARQUET.name}"
        )
        label = label_sorted[pos]
        class_counts += np.bincount(label, minlength=3)[:3]

        # reuse the source's dictionary verbatim so the two joined datasets stay
        # schema identical and can be read by the same code without a cast
        name_col = table.column("label_name").combine_chunks()
        values = name_col.dictionary
        by_label = {0: "Remained wetland", 1: "Converted to developed", 2: None}
        lookup = {v: j for j, v in enumerate(values.to_pylist())}
        by_label[2] = next(v for v in values.to_pylist() if v.startswith("Converted to other"))
        idx = np.array([lookup[by_label[k]] for k in (0, 1, 2)], dtype=np.int8)[label]

        table = table.set_column(
            table.schema.get_field_index("label"),
            table.schema.field("label"),
            pa.chunked_array([pa.array(label, type=pa.int8())]),
        )
        table = table.set_column(
            table.schema.get_field_index("label_name"),
            table.schema.field("label_name"),
            pa.chunked_array([pa.DictionaryArray.from_arrays(pa.array(idx, type=pa.int8()), values)]),
        )

        pq.write_table(table, DEST_DIR / Path(src).name, compression="snappy",
                       row_group_size=rg_size)
        written += table.num_rows
        del table
        print(f"  [{i:>2}/{len(files)}] {Path(src).name}  {written:>12,} rows", flush=True)

    print(f"\nWrote {written:,} rows to {DEST_DIR} in {time.time() - t_start:.0f}s")
    print(f"size: {sum(p.stat().st_size for p in DEST_DIR.glob('*.parquet')) / 1e9:.1f} GB")
    for k, n in enumerate(class_counts):
        print(f"  label {k}: {n:>12,}  {100 * n / written:7.4f}%")


if __name__ == "__main__":
    main()
