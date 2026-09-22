"""
Feature and label column names for the joined AlphaEarth + NLCD parquets under
data/.

A00..A63 for each year are the AlphaEarth embedding dims for the years leading
up to the label snapshot. dist_to_developed_2019_m is the one non embedding
feature and is on a meters scale, not unit scaled like the embedding dims, so it
needs its own normalization pass wherever it's used.

Two column sets are in play. YEARS_2017_2019 is what
data/alphaearth_wetland_joined_2019_2024 and _2019_2020 hold, and stays the
module default so nothing already built has to change. YEARS_2017_2022 is the
six year set in data/alphaearth_wetland_joined_2022_2024. Same rows, same label,
three more years of embeddings.
"""

YEARS_2017_2019 = (2017, 2018, 2019)
YEARS_2017_2022 = (2017, 2018, 2019, 2020, 2021, 2022)

N_EMBEDDING_DIMS = 64

DIST_COL = "dist_to_developed_2019_m"

# Raw column is three valued: 0 remained wetland, 1 converted to developed,
# 2 converted to other (non-developed). The binary target folds 0 and 2 into
# negative -- see label_utils.binarize_label for where that happens.
LABEL_COL = "label"


def emb_cols(years=YEARS_2017_2019):
    """A00..A63 for each year, year major, which is the order they sit in the
    parquet schema and the order emb.npy stores them in."""
    return [f"A{d:02d}_{y}" for y in years for d in range(N_EMBEDDING_DIMS)]


def feature_cols(years=YEARS_2017_2019):
    """dist first, then the embedding dims."""
    return [DIST_COL] + emb_cols(years)


FEATURE_COLS = feature_cols()

# Only the meters-scale column needs mean/std normalization -- the AlphaEarth
# embedding dims come out of GEE already roughly unit scaled, so normalizing
# them too would just be extra noise on top of a scale that's already fine.
COLS_TO_NORMALIZE = [DIST_COL]

# The array artifact (build_arrays.py) splits the features into two files: the
# embedding dims as float16 and the one meters scale column as float32. This is
# the single definition of that split. arrays.load_arrays asserts the stored
# manifest's column list equals the column list it was handed, element by
# element, so changing either invalidates data/arrays and crashes on load
# instead of quietly training on scrambled columns.
EMB_COLS = emb_cols()
