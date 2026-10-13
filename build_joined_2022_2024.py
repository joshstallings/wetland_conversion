"""
Builds the 2022 to 2024 joined AlphaEarth plus NLCD tiles: pixels NLCD calls
wetland in 2022, labeled by what they are in 2024, with embeddings for 2017
through 2022.

The 2017 to 2019 AlphaEarth tifs are no longer on disk, so those years can only
come out of data/alphaearth_wetland_joined_2019_2024. That limits the population
to pixels wetland in both 2019 and 2022 (58,997,139 of 59,158,854 Florida 2022
wetland pixels). The 161,715 that only became wetland after 2019 are dropped;
getting them back means exporting 2017 to 2019 from GEE again.

2020 to 2022 are read from data/alphaearth_embeddings. Labels and
dist_to_developed_2022_m come from build_labels.py:

    python build_labels.py 2024 --base 2022
    python build_joined_2022_2024.py

row, col, x, y, tile_row, tile_col, tile_id and block_id are copied through, so
blocks line up with the 2019_2024 directory and the fold assignment carries
over. dist_to_developed_2019_m is kept alongside the 2022 one since it predates
the base year and costs nothing.

Writes data/alphaearth_wetland_joined_2022_2024/
"""

import glob
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio

SOURCE_DIR = Path("data/alphaearth_wetland_joined_2019_2024")
LABELS_PATH = Path("data/wetland_sample_labels_2022_2024.parquet")
DEST_DIR = Path("data/alphaearth_wetland_joined_2022_2024")
AE_DIR = Path("data/alphaearth_embeddings")

NEW_YEARS = (2020, 2021, 2022)
N_BANDS = 64
TILE_PX = 2000
PIXEL_M = 30

NEW_DIST_COL = "dist_to_developed_2022_m"

# The tifs ship the embeddings quantized to uint8 with 128 standing in for 0.0.
# Recovered from the existing parquet: code 135 is stored as 0.055118, which is
# 7/127. nodata is code 0 in every band at once, and would decode to -1.0079 if
# it slipped through, so it is checked rather than clipped.
EMB_OFFSET = 128.0
EMB_SCALE = 127.0

# row and col index the same NLCD window in both sample tables (build_labels
# asserts the window origin), so this packs them into one sortable key.
KEY_STRIDE = 1 << 20


def decode(codes):
    return (codes.astype(np.float32) - EMB_OFFSET) / EMB_SCALE


def pixel_key(row, col):
    return row.astype(np.int64) * KEY_STRIDE + col.astype(np.int64)


def load_labels():
    """2022 to 2024 label table sorted by pixel key, as numpy columns."""
    t = pq.read_table(LABELS_PATH, columns=["row", "col", "x", "y", "label", NEW_DIST_COL])
    cols = {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names}
    assert cols["col"].max() < KEY_STRIDE, "col does not fit under KEY_STRIDE"
    key = pixel_key(cols["row"], cols["col"])
    order = np.argsort(key, kind="stable")
    out = {c: v[order] for c, v in cols.items()}
    out["key"] = key[order]
    assert np.all(np.diff(out["key"]) > 0), "duplicate (row, col) in the label table"
    return out


def read_tile(year, tile_id):
    """All 64 bands of one AlphaEarth tile as (64, 2000, 2000) uint8, 256 MB, plus
    its transform for the alignment check."""
    path = AE_DIR / f"AlphaEarth30m_Florida_{year}_{tile_id}.tif"
    with rasterio.open(path) as src:
        assert (src.count, src.width, src.height) == (N_BANDS, TILE_PX, TILE_PX), (
            f"{path.name}: expected {N_BANDS} bands at {TILE_PX}x{TILE_PX}, "
            f"got {src.count} at {src.width}x{src.height}"
        )
        return src.read(), src.transform


def check_alignment(tile_id, transform, x, y, tile_row, tile_col):
    """The old check read band 0 of the 2019 tif, which is gone. This one asks
    the new tif where each stored x, y pixel center lands and requires it to be
    the stored tile_row, tile_col."""
    assert transform.a == PIXEL_M and transform.e == -PIXEL_M, f"{tile_id}: not a 30 m grid"
    got_col = np.floor((x.astype(np.float64) - transform.c) / PIXEL_M).astype(np.intp)
    got_row = np.floor((transform.f - y.astype(np.float64)) / PIXEL_M).astype(np.intp)
    n_bad = int(np.sum((got_row != tile_row) | (got_col != tile_col)))
    if n_bad:
        raise AssertionError(
            f"{tile_id}: {n_bad:,} of {len(x):,} pixels land somewhere other than the "
            f"stored tile_row, tile_col in the new tifs."
        )


def match(table, labels):
    """Indices into table of rows still wetland in 2022, and the matching label
    table indices."""
    key = pixel_key(
        table.column("row").to_numpy(zero_copy_only=False),
        table.column("col").to_numpy(zero_copy_only=False),
    )
    pos = np.searchsorted(labels["key"], key)
    pos = np.minimum(pos, len(labels["key"]) - 1)
    hit = labels["key"][pos] == key
    return np.flatnonzero(hit), pos[hit]


def new_columns(blocks, tile_row, tile_col):
    """(name, float32 array) for every band of every added year, at the given pixels."""
    out = []
    for year, block in zip(NEW_YEARS, blocks):
        gathered = block[:, tile_row, tile_col]  # (64, n)
        nodata = (gathered == 0).all(axis=0)
        if nodata.any():
            raise AssertionError(
                f"{year}: {int(nodata.sum()):,} wetland pixels are nodata in the tif. "
                f"Decide how to handle them before writing a column of -1.0079."
            )
        for b in range(N_BANDS):
            out.append((f"A{b:02d}_{year}", decode(gathered[b])))
    return out


def rebuild_table(table, labels, blocks, transform, tile_id):
    keep, lab_idx = match(table, labels)
    table = table.take(pa.array(keep))

    x = table.column("x").to_numpy(zero_copy_only=False)
    y = table.column("y").to_numpy(zero_copy_only=False)
    if not (np.array_equal(x, labels["x"][lab_idx]) and np.array_equal(y, labels["y"][lab_idx])):
        raise AssertionError(f"{tile_id}: same (row, col) but different x, y between tables")

    # Both horizons label off the 2024 raster, so a pixel that was wetland in 2019
    # and 2022 has to carry the same label in both. Anything else means rows are
    # paired wrong.
    old_label = table.column("label").to_numpy(zero_copy_only=False)
    new_label = labels["label"][lab_idx]
    if not np.array_equal(old_label, new_label):
        raise AssertionError(
            f"{tile_id}: {int(np.sum(old_label != new_label)):,} rows disagree on the 2024 label"
        )

    tile_row = table.column("tile_row").to_numpy(zero_copy_only=False).astype(np.intp)
    tile_col = table.column("tile_col").to_numpy(zero_copy_only=False).astype(np.intp)
    assert tile_row.min() >= 0 and tile_row.max() < TILE_PX, "tile_row outside the tile"
    assert tile_col.min() >= 0 and tile_col.max() < TILE_PX, "tile_col outside the tile"
    check_alignment(tile_id, transform, x, y, tile_row, tile_col)

    insert_at = table.schema.get_field_index("dist_to_developed_2019_m") + 1
    table = table.add_column(
        insert_at, pa.field(NEW_DIST_COL, pa.float32()), pa.array(labels[NEW_DIST_COL][lab_idx])
    )

    # Ahead of block_id so all the embedding columns stay contiguous in the schema.
    insert_at = table.schema.get_field_index("block_id")
    for name, values in new_columns(blocks, tile_row, tile_col):
        table = table.add_column(insert_at, pa.field(name, pa.float32()), pa.array(values))
        insert_at += 1
    return table


def main():
    assert SOURCE_DIR.is_dir(), f"Missing {SOURCE_DIR}"
    assert LABELS_PATH.exists(), f"Missing {LABELS_PATH}, run build_labels.py 2024 --base 2022"
    files = sorted(glob.glob(str(SOURCE_DIR / "*.parquet")))
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    labels = load_labels()
    print(f"{len(labels['key']):,} 2022 wetland pixels in {LABELS_PATH} ({time.time() - t_start:.0f}s)")

    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"{len(files)} tiles, {total_rows:,} 2019 wetland rows in {SOURCE_DIR}")

    read, written = 0, 0
    label_counts = np.zeros(3, dtype=np.int64)
    matched = np.zeros(len(labels["key"]), dtype=bool)

    for i, src in enumerate(files, 1):
        tile_id = Path(src).stem
        pf = pq.ParquetFile(src)

        tiles = [read_tile(y, tile_id) for y in NEW_YEARS]
        blocks = [b for b, _ in tiles]
        transform = tiles[0][1]
        assert all(t == transform for _, t in tiles), f"{tile_id}: transforms differ across years"
        del tiles

        # Row group at a time rather than whole file: the biggest tile is 3.4M
        # rows and holding its old and new embeddings at once is 5 GB of float32.
        writer = None
        dest = DEST_DIR / Path(src).name
        n_out = 0
        try:
            for g in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(g)
                read += table.num_rows
                _, lab_idx = match(table, labels)
                if matched[lab_idx].any():
                    raise AssertionError(f"{tile_id}: a 2022 pixel matched more than one source row")
                matched[lab_idx] = True

                table = rebuild_table(table, labels, blocks, transform, tile_id)
                if writer is None:
                    writer = pq.ParquetWriter(dest, table.schema, compression="snappy")
                writer.write_table(table)
                n_out += table.num_rows
                label_counts += np.bincount(
                    table.column("label").to_numpy(zero_copy_only=False), minlength=3
                )[:3]
                del table
        finally:
            if writer is not None:
                writer.close()
        del blocks

        assert pq.ParquetFile(dest).metadata.num_rows == n_out, f"{dest.name}: row count changed"
        written += n_out

        elapsed = time.time() - t_start
        eta = (total_rows - read) * elapsed / max(read, 1)
        print(
            f"  [{i:>2}/{len(files)}] {Path(src).name}  {n_out:>9,} kept of "
            f"{pf.metadata.num_rows:>9,}  {written:>12,} written  eta {eta / 60:5.1f} min",
            flush=True,
        )

    size = sum(p.stat().st_size for p in DEST_DIR.glob("*.parquet")) / 1e9
    print(f"\nWrote {written:,} rows to {DEST_DIR} in {(time.time() - t_start) / 60:.1f} min, {size:.1f} GB")
    print(f"dropped {read - written:,} rows not wetland in 2022")
    print(f"dropped {int((~matched).sum()):,} 2022 wetland pixels with no 2017 to 2019 embeddings")
    print(
        f"labels  0 remained wetland {label_counts[0]:,}  1 converted to developed "
        f"{label_counts[1]:,}  2 converted to other {label_counts[2]:,}"
    )


if __name__ == "__main__":
    main()
