"""Shared config and helpers for the modeling pipeline: feature and
target column definitions, per fold distance normalization, the iter_folds
generator, and the metric functions that scripts/model/train.py,
eda_dataset.ipynb, and model_analysis.ipynb all need against one chosen
dataset (see scripts/dataset/build_train_pool.py for what a dataset is).
Not runnable on its own, mirrors the role gee_common.py plays for the
AlphaEarth export pipeline.
"""

import json
import os

from typing import Callable

import duckdb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve

from scripts.data_constants import BAND_COLS

JOINED_GLOB = "data/processed/alphaearth_wetland_joined/*.parquet"
DATASETS_DIR = "data/processed/datasets"

# Built once by `python -m scripts.dataset.build_train_pool features`, not tied to
# any one dataset: the local development density, edge density, and
# direction-to-nearest-development columns, one row per wetland sample
# pixel. connect_raw() joins this onto `samples` unconditionally, same as
# the AlphaEarth bands, so it needs to exist before anything in this module
# runs. See build_train_pool.py's `features` subcommand for how each column
# is built.
NEIGHBORHOOD_FEATURES_PATH = "data/processed/neighborhood_features.parquet"
NEIGHBORHOOD_FEATURE_COLS = [
    "frac_dev_100m", "frac_dev_300m", "frac_dev_500m", "edge_density_300m", "dev_dir_sin", "dev_dir_cos",
]

K_FOLDS = 5
DIST_COL = "dist_to_developed_2019_m"
DIST_NORM_COL = "dist_log_z"
TARGET_COL = "label_bin"
# BAND_COLS comes from data_constants.py, shared with build_train_pool.py.
FEATURE_COLS = [DIST_NORM_COL] + BAND_COLS
DIST_ONLY_FEATURE_COLS = [DIST_NORM_COL]
# Distance, the neighborhood features, and every AlphaEarth band together.
EXTENDED_FEATURE_COLS = [DIST_NORM_COL] + NEIGHBORHOOD_FEATURE_COLS + BAND_COLS
# Distance and the neighborhood features, no AlphaEarth bands: isolates what
# the raster derived features add on their own, without embeddings in the mix.
DIST_PLUS_NEIGHBORHOOD_FEATURE_COLS = [DIST_NORM_COL] + NEIGHBORHOOD_FEATURE_COLS

M2_PER_ACRE = 4046.8564224
PIXEL_ACRES = 30 * 30 / M2_PER_ACRE  # one 30m NLCD pixel, in acres

SUMMARY_COLS = ["avg_precision", "pr_auc", "accuracy", "roc_auc", "precision", "recall", "f1", "best_threshold"]
CALIB_COLS = ["brier", "pred_acres", "actual_acres", "acres_error_pct", "acres_error_isotonic_pct"]

TEST_BATCH_ROWS = 200_000

# Slide sized figure defaults: legible once dropped into a deck, not just
# readable in a notebook cell.
SLIDE_RC = {
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
}

# Colorblind separable palette, one fixed color per fold so the same fold
# means the same color across every figure in eda_dataset.ipynb and
# model_analysis.ipynb.
FOLD_COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 3: "#eda100", 4: "#e87ba4"}
LABEL_COLORS = {
    "Remained wetland": "#2a78d6",
    "Converted to other (non-developed)": "#1baf7a",
    "Converted to developed": "#eb6834",
}

# Okabe-Ito, for the block distance clusters in eda_dataset.ipynb. Clusters
# are numbered by centroid distance, so index 0 is the regime closest to
# development and the ramp reads roughly near to far.
CLUSTER_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#666666",
]


def dataset_dir(dataset_name: str) -> str:
    return os.path.join(DATASETS_DIR, dataset_name)


def load_dataset_config(dataset_name: str) -> dict:
    """The config.json build_train_pool.py writes alongside a dataset: every
    sampling constant (TAU_M, POS_ROW_DUP, POS_WEIGHT_MULT, ...) actually used
    to build it. Read this instead of assuming build_train_pool.py's module
    defaults, since a dataset can have been built with an override."""
    with open(os.path.join(dataset_dir(dataset_name), "config.json")) as f:
        return json.load(f)


def connect_raw() -> duckdb.DuckDBPyConnection:
    """Open a connection with just the `samples` view, the full joined table
    plus the neighborhood features, no fold join. For EDA that doesn't
    depend on any one generated dataset."""
    if not os.path.exists(NEIGHBORHOOD_FEATURES_PATH):
        raise SystemExit(f"{NEIGHBORHOOD_FEATURES_PATH} not found, run "
                         f"`python -m scripts.dataset.build_train_pool features` first")
    con = duckdb.connect()
    con.execute(f"""
        CREATE OR REPLACE VIEW samples AS
        SELECT s.*, nf.* EXCLUDE ("row", "col")
        FROM read_parquet('{JOINED_GLOB}') s
        LEFT JOIN read_parquet('{NEIGHBORHOOD_FEATURES_PATH}') nf
            ON s."row" = nf."row" AND s."col" = nf."col"
    """)
    return con


def connect_samples(dataset_name: str) -> tuple[duckdb.DuckDBPyConnection, pd.DataFrame]:
    """Open a connection with `samples` (the full joined table) and
    `samples_fold` (joined against dataset_name's block assignment) views
    registered. Returns (con, block_folds)."""
    con = connect_raw()
    block_folds = pd.read_parquet(os.path.join(dataset_dir(dataset_name), "block_folds.parquet"))
    con.register("block_fold", block_folds[["block_id", "fold"]])
    con.execute("""
        CREATE OR REPLACE VIEW samples_fold AS
        SELECT s.*, bf.fold FROM samples s JOIN block_fold bf USING (block_id)
    """)
    return con, block_folds


def load_train_pool(dataset_name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(dataset_dir(dataset_name), "train_pool.parquet"))


def fit_dist_normalizer(con: duckdb.DuckDBPyConnection, fold_id: int) -> tuple[float, float]:
    stats = con.execute(f"""
        SELECT AVG(LN(1 + {DIST_COL})) AS log_mean,
               STDDEV(LN(1 + {DIST_COL})) AS log_std
        FROM samples_fold
        WHERE fold != {fold_id}
    """).df()
    return stats["log_mean"][0], stats["log_std"][0]


def fit_dist_normalizer_excluding(con: duckdb.DuckDBPyConnection, excluded_folds: list[int]) -> tuple[float, float]:
    fold_list = ", ".join(str(f) for f in excluded_folds)
    stats = con.execute(f"""
        SELECT AVG(LN(1 + {DIST_COL})) AS log_mean,
               STDDEV(LN(1 + {DIST_COL})) AS log_std
        FROM samples_fold
        WHERE fold NOT IN ({fold_list})
    """).df()
    return stats["log_mean"][0], stats["log_std"][0]


def add_normalized_dist(df: pd.DataFrame, log_mean: float, log_std: float) -> pd.DataFrame:
    df = df.copy()
    df[DIST_NORM_COL] = (np.log1p(df[DIST_COL]) - log_mean) / log_std
    return df


def natural_weights(df: pd.DataFrame, pos_weight_mult: float, pos_row_dup: int) -> np.ndarray:
    """Pool weights with the positive boost taken back out.

    label=0 already carries 1/p and label=2 sat at p=1, so only label=1 needs
    dividing: each of its pos_row_dup copies drops from pos_weight_mult /
    pos_row_dup down to 1 / pos_row_dup, and the copies sum to the one row
    that actually exists. Weighted this way the pool reconstructs the real
    population, which is what lets an isotonic fitted on it read out in true
    probability units.
    """
    w = df["sample_weight"].to_numpy(dtype=float).copy()
    w[df["label"].to_numpy() == 1] /= pos_weight_mult
    return w


def calibrate(p: np.ndarray, pos_weight_mult: float) -> np.ndarray:
    """Undo the positive upweight analytically. Exact for the sampling
    design, but it cannot fix the model's own miscalibration, so treat it as
    a baseline reading rather than the thing accuracy claims get hung on."""
    return p / (p + pos_weight_mult * (1 - p))


def best_f1_operating_point(precision, recall, thresholds):
    """precision_recall_curve returns one more precision/recall pair than it
    does thresholds, the curve's closed endpoint at precision 1, recall 0
    with no threshold behind it, so that pair is dropped before searching."""
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) > 0,
    )
    best_idx = np.argmax(f1)
    return thresholds[best_idx], precision[best_idx], recall[best_idx], f1[best_idx]


def reliability(y_true: np.ndarray, p_cal: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed versus predicted conversion rate, in equal count bins of
    predicted risk. Equal count rather than equal width: at well under 1%
    base rate almost every pixel lands in the bottom width bin, which tells
    you nothing."""
    order = np.argsort(p_cal)
    return pd.DataFrame([
        {"bin": i, "n": len(idx), "pred_mean": float(p_cal[idx].mean()), "obs_rate": float(y_true[idx].mean())}
        for i, idx in enumerate(np.array_split(order, n_bins))
    ])


def predict_fold(con: duckdb.DuckDBPyConnection, model, fold_id: int, log_mean: float, log_std: float,
                 feature_cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Score a whole natural fold in batches, so nothing holds a multi GB
    fold in memory at once. Returns (y_true, y_score)."""
    feature_cols = feature_cols or FEATURE_COLS
    reader = con.execute(f"""
        SELECT {DIST_COL}, {", ".join(BAND_COLS)}, {", ".join(NEIGHBORHOOD_FEATURE_COLS)},
               CASE WHEN label = 1 THEN 1 ELSE 0 END AS {TARGET_COL}
        FROM samples_fold
        WHERE fold = {fold_id}
    """).fetch_record_batch(TEST_BATCH_ROWS)

    y_true, y_score = [], []
    for batch in reader:
        df = batch.to_pandas()
        df[DIST_NORM_COL] = (np.log1p(df[DIST_COL]) - log_mean) / log_std
        y_score.append(model.predict_proba(df[feature_cols])[:, 1])
        y_true.append(df[TARGET_COL].to_numpy())
    return np.concatenate(y_true), np.concatenate(y_score)


def predict_fold_with_dist(con: duckdb.DuckDBPyConnection, model, fold_id: int, log_mean: float, log_std: float,
                           feature_cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same streamed pass as predict_fold, plus the raw distance per row, for
    the decile breakdown in model_analysis.ipynb."""
    feature_cols = feature_cols or FEATURE_COLS
    reader = con.execute(f"""
        SELECT {DIST_COL}, {", ".join(BAND_COLS)}, {", ".join(NEIGHBORHOOD_FEATURE_COLS)},
               CASE WHEN label = 1 THEN 1 ELSE 0 END AS {TARGET_COL}
        FROM samples_fold
        WHERE fold = {fold_id}
    """).fetch_record_batch(TEST_BATCH_ROWS)

    y_true, y_score, dist = [], [], []
    for batch in reader:
        df = batch.to_pandas()
        df[DIST_NORM_COL] = (np.log1p(df[DIST_COL]) - log_mean) / log_std
        y_score.append(model.predict_proba(df[feature_cols])[:, 1])
        y_true.append(df[TARGET_COL].to_numpy())
        dist.append(df[DIST_COL].to_numpy())
    return np.concatenate(y_true), np.concatenate(y_score), np.concatenate(dist)


def natural_fold_sample(con: duckdb.DuckDBPyConnection, fold_id: int, n_rows: int,
                        log_mean: float, log_std: float) -> pd.DataFrame:
    """Uniform subsample of one natural fold, for anything that needs a real
    frame instead of a stream (early stopping validation, mainly).

    Uniform rather than stratified on purpose: early stopping should see the
    same base rate the test fold has, so thinning positives and negatives
    together is the point.
    """
    n_fold = con.execute(f"SELECT COUNT(*) FROM samples_fold WHERE fold = {fold_id}").fetchone()[0]
    frac = min(1.0, n_rows / n_fold)
    df = con.execute(f"""
        SELECT {DIST_COL}, {", ".join(BAND_COLS)}, {", ".join(NEIGHBORHOOD_FEATURE_COLS)},
               CASE WHEN label = 1 THEN 1 ELSE 0 END AS {TARGET_COL}
        FROM samples_fold
        WHERE fold = {fold_id}
          AND (hash("row", "col", 99) % 1073741824) / 1073741824.0 < {frac}
    """).df()
    df[DIST_NORM_COL] = (np.log1p(df[DIST_COL]) - log_mean) / log_std
    return df


def sample_fold_with_xy(con: duckdb.DuckDBPyConnection, fold_id: int, n_rows: int,
                        log_mean: float, log_std: float) -> pd.DataFrame:
    """Deterministic subsample of one held out fold, kept with x/y for the
    spatial diagnostic maps in model_analysis.ipynb.

    Hash based on (row, col) rather than duckdb's SAMPLE, so rerunning this
    always draws the same pixels. Seed 21 keeps this draw independent of the
    seed 99 draw natural_fold_sample uses.
    """
    n_fold = con.execute(f"SELECT COUNT(*) FROM samples_fold WHERE fold = {fold_id}").fetchone()[0]
    frac = min(1.0, n_rows / n_fold)
    df = con.execute(f"""
        SELECT x, y, label, {DIST_COL}, {", ".join(BAND_COLS)}, {", ".join(NEIGHBORHOOD_FEATURE_COLS)}
        FROM samples_fold
        WHERE fold = {fold_id}
          AND (hash("row", "col", 21) % 1073741824) / 1073741824.0 < {frac}
    """).df()
    df[DIST_NORM_COL] = (np.log1p(df[DIST_COL]) - log_mean) / log_std
    df[TARGET_COL] = (df["label"] == 1).astype(int)
    return df


def decile_precision_recall(oof: pd.DataFrame, dist_edges: list[float], dist_labels: list[str],
                            k_folds: int = K_FOLDS) -> pd.DataFrame:
    """Precision and recall by distance-to-development decile, one row per
    (fold, decile). Pulled out of model_analysis.ipynb so a non LightGBM
    experiment gets the same breakdown for free: this only touches oof's
    label/y_score/DIST_COL columns, nothing model specific.

    Each fold classifies at its own best F1 threshold, recovered fresh from
    its out of fold precision_recall_curve rather than refit, the same
    operating point fold_summary's precision/recall columns already report.
    Bins with no positives to recall, or no predicted positives to score
    precision on, come back as NaN rather than a misleading zero.
    """
    rows = []
    for fold_id in range(k_folds):
        sub = oof[oof["fold"] == fold_id]
        y_test = sub["label"].to_numpy()
        proba = sub["y_score"].to_numpy()
        dist = sub[DIST_COL].to_numpy()

        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, proba)
        best_threshold, _, _, _ = best_f1_operating_point(precision_curve, recall_curve, pr_thresholds)
        y_pred = proba >= best_threshold

        bin_idx = np.digitize(dist, dist_edges)
        for b in range(len(dist_labels)):
            in_bin = bin_idx == b
            tp = int(np.sum(y_pred[in_bin] & (y_test[in_bin] == 1)))
            fp = int(np.sum(y_pred[in_bin] & (y_test[in_bin] == 0)))
            fn = int(np.sum(~y_pred[in_bin] & (y_test[in_bin] == 1)))
            rows.append({
                "fold": fold_id, "decile": b, "label": dist_labels[b],
                "n": int(in_bin.sum()), "n_pos": tp + fn,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def spatial_fold_predictions(con: duckdb.DuckDBPyConnection, score_fns: dict[int, Callable[[pd.DataFrame], np.ndarray]],
                             fold_summary: pd.DataFrame, dataset_config: dict, k_folds: int = K_FOLDS,
                             rows_per_fold: int = 100_000) -> pd.DataFrame:
    """Deterministic spatial subsample of every fold, scored out of fold and
    tagged true/false positive/negative against each fold's own best F1
    threshold. Also pulled out of model_analysis.ipynb.

    score_fns[fold_id] is the one model specific piece: a callable from a
    feature frame to a raw, uncalibrated positive class score. For a
    LightGBM fold model that's `lambda df: model.predict_proba(df[cols])[:, 1]`;
    a DL model plugs in its own batched inference the same way. Everything
    after that (the sample_fold_with_xy draw, calibrate(), thresholding at
    fold_summary's best_threshold, the outcome tagging) is the same either
    way, since it only ever sees the score, not what produced it.
    """
    frames = []
    for fold_id in range(k_folds):
        log_mean, log_std = fit_dist_normalizer(con, fold_id)
        df = sample_fold_with_xy(con, fold_id, rows_per_fold, log_mean, log_std)
        proba = score_fns[fold_id](df)
        df["score"] = calibrate(proba, dataset_config["pos_weight_mult"])
        df["y_pred"] = (df["score"] >= fold_summary.loc[fold_id, "best_threshold"]).astype(int)
        frames.append(df[["x", "y", "label", TARGET_COL, "score", "y_pred"]])
        del df, proba

    predictions = pd.concat(frames, ignore_index=True)
    # tag each pixel once so a caller plotting multiple error panels doesn't
    # recompute this per panel
    predictions["outcome"] = np.select(
        [
            (predictions[TARGET_COL] == 1) & (predictions["y_pred"] == 1),
            (predictions[TARGET_COL] == 0) & (predictions["y_pred"] == 1),
            (predictions[TARGET_COL] == 1) & (predictions["y_pred"] == 0),
        ],
        ["true_positive", "false_positive", "false_negative"],
        default="true_negative",
    )
    return predictions


def iter_folds(con: duckdb.DuckDBPyConnection, train_pool: pd.DataFrame,
               pos_weight_mult: float, pos_row_dup: int, k: int = K_FOLDS):
    """Yield (fold_id, X_train, y_train, w_train, predict_test, make_calibrator).

    X_train carries every column train_pool has, not just FEATURE_COLS,
    since a dataset built with --with-neighborhood-features carries columns
    beyond that fixed list. Callers slice down to whatever their experiment
    actually wants (see DIST_ONLY_FEATURE_COLS, EXTENDED_FEATURE_COLS)
    before fit, same for predict_test/make_calibrator's feature_cols
    argument. Nothing about the fold split, the weighting, or the nested
    calibration split depends on which columns end up in front of the
    model.

    predict_test(model, feature_cols=None) streams the held out fold and
    hands back (y_true, y_score), so no caller ever holds the full test frame
    in memory.

    make_calibrator(model_factory, feature_cols=None) fits an isotonic on a
    nested split: an inner model trains on 3 of the 4 training folds, and the
    calibrator is fitted on its scores over the 4th. Fold k is never touched,
    so the test fold stays clean and the outer model still trains on all 4.
    """
    for fold_id in range(k):
        log_mean, log_std = fit_dist_normalizer(con, fold_id)
        train_fold = train_pool[train_pool["fold"] != fold_id].pipe(add_normalized_dist, log_mean, log_std)
        w_train = train_fold["sample_weight"] / train_fold["sample_weight"].mean()

        calib_fold = (fold_id + 1) % k
        inner_train = train_fold[train_fold["fold"] != calib_fold]
        calib_frame = train_fold[train_fold["fold"] == calib_fold]

        def predict_test(model, feature_cols=None, _fold=fold_id, _mean=log_mean, _std=log_std):
            return predict_fold(con, model, _fold, _mean, _std, feature_cols)

        def make_calibrator(model_factory, feature_cols=None, _inner=inner_train, _cal=calib_frame):
            feature_cols = feature_cols or FEATURE_COLS
            inner = model_factory()
            w_inner = _inner["sample_weight"] / _inner["sample_weight"].mean()
            inner.fit(_inner[feature_cols], _inner[TARGET_COL], sample_weight=w_inner)
            scores = inner.predict_proba(_cal[feature_cols])[:, 1]
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(scores, _cal[TARGET_COL], sample_weight=natural_weights(_cal, pos_weight_mult, pos_row_dup))
            return iso

        yield (
            fold_id,
            train_fold, train_fold[TARGET_COL], w_train,
            predict_test, make_calibrator,
        )
