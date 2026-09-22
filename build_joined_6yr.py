"""
Extends the joined AlphaEarth plus NLCD tiles from three years of embeddings to six.

data/alphaearth_wetland_joined_2019_2024 already holds A00..A63 for 2017, 2018
and 2019 against the 2019 wetland mask and the 2024 label. This adds 2020, 2021
and 2022 from the same AlphaEarth tiles, so the features run 2017 to 2022 while
the population and the label stay exactly what they were.

Nothing is recomputed. row, col, x, y, dist_to_developed_2019_m, label,
label_name, tile_row, tile_col, tile_id and block_id are copied through, which
keeps the new directory row for row comparable with the 2019_2024 and 2019_2020
ones and lets the same population_stats and fold assignment apply.

    python build_joined_6yr.py

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
DEST_DIR = Path("data/alphaearth_wetland_joined_2022_2024")
AE_DIR = Path("data/alphaearth_florida/alphaearth_florida_30m")

NEW_YEARS = (2020, 2021, 2022)
N_BANDS = 64
TILE_PX = 2000

# The tifs ship the embeddings quantized to uint8 with 128 standing in for 0.0.
# Recovered from the existing parquet: code 135 is stored as 0.055118, which is
# 7/127. nodata is code 0 in every band at once, and would decode to -1.0079 if
# it slipped through, so it is checked rather than clipped.
EMB_OFFSET = 128.0
EMB_SCALE = 127.0

# Cheap per tile guard that the tile_row, tile_col pair in the parquet still
# points at the pixel it did when the original join ran. Band 0 of the 2019 tif
# has to reproduce the stored A00_2019 exactly, since both come from the same
# uint8 code.
ALIGNMENT_CHECK_YEAR = 2019
ALIGNMENT_CHECK_BAND = 0


def decode(codes):
    return (codes.astype(np.float32) - EMB_OFFSET) / EMB_SCALE


def read_tile(year, tile_id):
    """All 64 bands of one AlphaEarth tile as (64, 2000, 2000) uint8, 256 MB."""
    path = AE_DIR / f"AlphaEarth30m_Florida_{year}_{tile_id}.tif"
    with rasterio.open(path) as src:
        assert (src.count, src.width, src.height) == (N_BANDS, TILE_PX, TILE_PX), (
            f"{path.name}: expected {N_BANDS} bands at {TILE_PX}x{TILE_PX}, "
            f"got {src.count} at {src.width}x{src.height}"
        )
        return src.read()


def check_alignment(tile_id, tile_row, tile_col, stored_a00_2019):
    """Fails loudly if the stored pixel indices no longer land on the pixel the
    original join used."""
    path = AE_DIR / f"AlphaEarth30m_Florida_{ALIGNMENT_CHECK_YEAR}_{tile_id}.tif"
    with rasterio.open(path) as src:
        band = src.read(ALIGNMENT_CHECK_BAND + 1)
    got = decode(band[tile_row, tile_col])
    if not np.array_equal(got, stored_a00_2019):
        n_bad = int(np.sum(got != stored_a00_2019))
        raise AssertionError(
            f"{tile_id}: {n_bad:,} of {len(got):,} pixels disagree with the stored "
            f"A{ALIGNMENT_CHECK_BAND:02d}_{ALIGNMENT_CHECK_YEAR}. tile_row, tile_col "
            f"no longer index the tile the original join used."
        )


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


def extend_table(table, blocks):
    """Source table plus the new year columns, inserted ahead of block_id so all
    the embedding columns stay contiguous in the schema."""
    tile_row = table.column("tile_row").to_numpy(zero_copy_only=False).astype(np.intp)
    tile_col = table.column("tile_col").to_numpy(zero_copy_only=False).astype(np.intp)
    assert tile_row.min() >= 0 and tile_row.max() < TILE_PX, "tile_row outside the tile"
    assert tile_col.min() >= 0 and tile_col.max() < TILE_PX, "tile_col outside the tile"

    insert_at = table.schema.get_field_index("block_id")
    for name, values in new_columns(blocks, tile_row, tile_col):
        table = table.add_column(insert_at, pa.field(name, pa.float32()), pa.array(values))
        insert_at += 1
    return table


def main():
    assert SOURCE_DIR.is_dir(), f"Missing {SOURCE_DIR}"
    files = sorted(glob.glob(str(SOURCE_DIR / "*.parquet")))
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"{len(files)} tiles, {total_rows:,} rows, adding {len(NEW_YEARS) * N_BANDS} columns")

    written = 0
    t_start = time.time()

    for i, src in enumerate(files, 1):
        tile_id = Path(src).stem
        pf = pq.ParquetFile(src)
        n_rows = pf.metadata.num_rows

        first = pf.read_row_group(0, columns=["tile_row", "tile_col", "A00_2019"])
        check_alignment(
            tile_id,
            first.column("tile_row").to_numpy(zero_copy_only=False).astype(np.intp),
            first.column("tile_col").to_numpy(zero_copy_only=False).astype(np.intp),
            first.column("A00_2019").to_numpy(zero_copy_only=False),
        )
        del first

        blocks = [read_tile(y, tile_id) for y in NEW_YEARS]

        # Row group at a time rather than whole file: the biggest tile is 3.4M
        # rows and holding its old and new embeddings at once is 5 GB of float32.
        writer = None
        dest = DEST_DIR / Path(src).name
        try:
            for g in range(pf.metadata.num_row_groups):
                table = extend_table(pf.read_row_group(g), blocks)
                if writer is None:
                    writer = pq.ParquetWriter(dest, table.schema, compression="snappy")
                writer.write_table(table)
                written += table.num_rows
                del table
        finally:
            if writer is not None:
                writer.close()
        del blocks

        assert pq.ParquetFile(dest).metadata.num_rows == n_rows, f"{dest.name}: row count changed"

        elapsed = time.time() - t_start
        eta = (total_rows - written) * elapsed / max(written, 1)
        print(
            f"  [{i:>2}/{len(files)}] {Path(src).name}  {n_rows:>9,} rows  "
            f"{written:>12,} done  eta {eta / 60:5.1f} min",
            flush=True,
        )

    size = sum(p.stat().st_size for p in DEST_DIR.glob("*.parquet")) / 1e9
    print(f"\nWrote {written:,} rows to {DEST_DIR} in {(time.time() - t_start) / 60:.1f} min")
    print(f"size: {size:.1f} GB")


if __name__ == "__main__":
    main()
