"""
Feature and label column names for the joined AlphaEarth + NLCD parquet at
data/alphaearth_wetland_joined. 

A00..A63 x {2017, 2018, 2019} are the AlphaEarth embedding dims for each of the
three years leading up to the 2019 label snapshot. dist_to_developed_2019_m is
the one non-embedding feature and is on a meters scale, not unit scaled like the
embedding dims, so it needs its own normalization pass wherever it's used.
"""

FEATURE_COLS = [
    "dist_to_developed_2019_m",
    "A00_2017", "A01_2017", "A02_2017", "A03_2017", "A04_2017", "A05_2017", "A06_2017", "A07_2017",
    "A08_2017", "A09_2017", "A10_2017", "A11_2017", "A12_2017", "A13_2017", "A14_2017", "A15_2017",
    "A16_2017", "A17_2017", "A18_2017", "A19_2017", "A20_2017", "A21_2017", "A22_2017", "A23_2017",
    "A24_2017", "A25_2017", "A26_2017", "A27_2017", "A28_2017", "A29_2017", "A30_2017", "A31_2017",
    "A32_2017", "A33_2017", "A34_2017", "A35_2017", "A36_2017", "A37_2017", "A38_2017", "A39_2017",
    "A40_2017", "A41_2017", "A42_2017", "A43_2017", "A44_2017", "A45_2017", "A46_2017", "A47_2017",
    "A48_2017", "A49_2017", "A50_2017", "A51_2017", "A52_2017", "A53_2017", "A54_2017", "A55_2017",
    "A56_2017", "A57_2017", "A58_2017", "A59_2017", "A60_2017", "A61_2017", "A62_2017", "A63_2017",
    "A00_2018", "A01_2018", "A02_2018", "A03_2018", "A04_2018", "A05_2018", "A06_2018", "A07_2018",
    "A08_2018", "A09_2018", "A10_2018", "A11_2018", "A12_2018", "A13_2018", "A14_2018", "A15_2018",
    "A16_2018", "A17_2018", "A18_2018", "A19_2018", "A20_2018", "A21_2018", "A22_2018", "A23_2018",
    "A24_2018", "A25_2018", "A26_2018", "A27_2018", "A28_2018", "A29_2018", "A30_2018", "A31_2018",
    "A32_2018", "A33_2018", "A34_2018", "A35_2018", "A36_2018", "A37_2018", "A38_2018", "A39_2018",
    "A40_2018", "A41_2018", "A42_2018", "A43_2018", "A44_2018", "A45_2018", "A46_2018", "A47_2018",
    "A48_2018", "A49_2018", "A50_2018", "A51_2018", "A52_2018", "A53_2018", "A54_2018", "A55_2018",
    "A56_2018", "A57_2018", "A58_2018", "A59_2018", "A60_2018", "A61_2018", "A62_2018", "A63_2018",
    "A00_2019", "A01_2019", "A02_2019", "A03_2019", "A04_2019", "A05_2019", "A06_2019", "A07_2019",
    "A08_2019", "A09_2019", "A10_2019", "A11_2019", "A12_2019", "A13_2019", "A14_2019", "A15_2019",
    "A16_2019", "A17_2019", "A18_2019", "A19_2019", "A20_2019", "A21_2019", "A22_2019", "A23_2019",
    "A24_2019", "A25_2019", "A26_2019", "A27_2019", "A28_2019", "A29_2019", "A30_2019", "A31_2019",
    "A32_2019", "A33_2019", "A34_2019", "A35_2019", "A36_2019", "A37_2019", "A38_2019", "A39_2019",
    "A40_2019", "A41_2019", "A42_2019", "A43_2019", "A44_2019", "A45_2019", "A46_2019", "A47_2019",
    "A48_2019", "A49_2019", "A50_2019", "A51_2019", "A52_2019", "A53_2019", "A54_2019", "A55_2019",
    "A56_2019", "A57_2019", "A58_2019", "A59_2019", "A60_2019", "A61_2019", "A62_2019", "A63_2019",
]

# Only the meters-scale column needs mean/std normalization -- the AlphaEarth
# embedding dims come out of GEE already roughly unit scaled, so normalizing
# them too would just be extra noise on top of a scale that's already fine.
COLS_TO_NORMALIZE = ["dist_to_developed_2019_m"]

# Raw column is three valued: 0 remained wetland, 1 converted to developed,
# 2 converted to other (non-developed). The binary target folds 0 and 2 into
# negative -- see label_utils.binarize_label for where that happens.
LABEL_COL = "label"
