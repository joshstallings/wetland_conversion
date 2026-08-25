"""Build the 5 spatial folds and the gradient boosting training pool from the
joined AlphaEarth/NLCD table. Every run is named: output goes to
data/processed/datasets/<dataset-name>/, so tuning a sampling constant and
rerunning under a new name never touches a dataset you already trained
against. Two stages, both deterministic, so a rerun of the same name
reproduces the same split and the same draw:

  folds
      Assigns the 1,727 10km blocks to K_FOLDS folds, balancing three things
      at once: block count, rare class counts, and total row count. Balancing
      on label=1 alone (what gradient_boosting.ipynb used to do inline) got
      the positive counts exact but left folds 0 and 3 holding twice the rows
      of the others, so test fold prevalence swung from 0.19% to 0.43% and
      the per fold AP numbers were not comparable to each other. A greedy
      pass gets it roughly right, then pairwise swaps between folds close the
      rest of the gap. Writes
      data/processed/datasets/<dataset-name>/block_folds.parquet. Pass
      --cluster-file and --cluster to restrict the block universe to one
      block_distance_profiles.py cluster before assigning folds, e.g. to
      build a dataset out of just the near-development regime (cluster 0).

  pool
      Draws the training pool. Every label=1 and label=2 pixel is kept,
      label=1 duplicated pos-row-dup times over, and label=0 is kept with a
      probability that decays with distance to 2019 development so the pool
      concentrates where conversion actually happens. Each fold gets its own
      decay constant A_k, solved in closed form so every fold contributes the
      same row budget no matter where its distance distribution happens to
      sit. Pass --no-thinning to skip the decay draw entirely and keep every
      label=0 pixel instead, useful once the block universe (via
      --cluster-file/--cluster) is already small enough that there's nothing
      left to thin. Requires a fold assignment already built under the same
      dataset-name, with the same --cluster-file/--cluster if any. Writes
      data/processed/datasets/<dataset-name>/train_pool.parquet. Pass
      --with-neighborhood-features to add the columns `features` below
      builds to the draw.

  features
      Builds neighborhood features straight from the 2019 NLCD raster: the
      share of nearby pixels already developed at 100m/300m/500m, a local
      density of developed/not developed edges at 300m, and the direction
      (as sine and cosine) toward the nearest developed pixel. Not tied to
      any one dataset-name, these are fixed properties of a pixel's
      surroundings, the same for every dataset built from the same raster.
      Writes data/processed/neighborhood_features.parquet, one row per
      wetland sample pixel. See _local_density/_edge_mask/
      _direction_to_nearest_developed below for the reasoning behind each
      feature and the approximations made to keep this affordable on a
      668 million pixel raster.

Every constant a dataset was actually built with, defaulted or overridden,
gets written to data/processed/datasets/<dataset-name>/config.json, so the
directory is self documenting: no need to cross reference a notebook cell or
shell history to know what produced it.

On weighting: label=0 rows carry 1/p, so in expectation they reconstruct the
true label=0 population, and label=2 sits at p=1 and weight 1, so it
represents exactly itself. That leaves the pos-weight-mult boost on label=1
as the only place the pool departs from reality, which is what makes the
model's scores recoverable back to real conversion probabilities with
p_true = p / (p + pos_weight_mult * (1 - p)). scripts/model/model_common.py's
calibrate() applies that correction before anything gets reported in
probability units. Left unset, pos-weight-mult is derived rather than hand
picked: n_neg_natural / n_pos_natural, the ratio that gives label=1 and
label!=1 equal total weight in the loss (natural_pos_weight_mult). Pass
--pos-weight-mult to pin a specific value instead.

Usage:
    python -m scripts.dataset.build_train_pool features [--force]
    python -m scripts.dataset.build_train_pool folds --dataset-name NAME [--force]
        [--k-folds 5] [--seed 0] [--cluster-file PATH --cluster N]
    python -m scripts.dataset.build_train_pool pool --dataset-name NAME [--force]
        [--tau-m 200.0] [--pos-row-dup 2] [--pos-weight-mult VALUE]
        [--rows-per-training-set 2500000] [--draw-seed 42] [--no-thinning]
        [--with-neighborhood-features] [--cluster-file PATH --cluster N]
    python -m scripts.dataset.build_train_pool report --dataset-name NAME
"""

import argparse
import gc
import json
import os

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import transform as window_transform
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, uniform_filter

from scripts.data_constants import ALPHAEARTH_DATASET, BAND_COLS, BUFFER_M, DEVELOPED_CLASSES

JOINED_GLOB = "data/processed/alphaearth_wetland_joined/*.parquet"
DATASETS_DIR = "data/processed/datasets"

# Inputs for the features command. Kept separate from the rest of this
# file's constants since nothing else here touches the raw NLCD raster.
NLCD_2019_PATH = "data/NLCD/Annual_NLCD_LndCov_2019_CU_C1V2/Annual_NLCD_LndCov_2019_CU_C1V2.tif"
STATE_BOUNDARY_SHP = "data/boundaries/cb_2023_us_state_500k/cb_2023_us_state_500k.shp"
SAMPLE_LABELS_PATH = "data/processed/wetland_sample_labels_2019_2024.parquet"
NEIGHBORHOOD_FEATURES_PATH = "data/processed/neighborhood_features.parquet"

# BUFFER_M and DEVELOPED_CLASSES come from data_constants.py, shared with
# wetland_sample_labels.ipynb: BUFFER_M has to match exactly, it's what keeps
# a row, col pair pointing at the same pixel here as it does in the sample
# table built there.
PIXEL_M = 30.0

NEIGHBORHOOD_RADII_M = {"frac_dev_100m": 100.0, "frac_dev_300m": 300.0, "frac_dev_500m": 500.0}
EDGE_DENSITY_RADIUS_M = 300.0
NEIGHBORHOOD_FEATURE_COLS = list(NEIGHBORHOOD_RADII_M) + ["edge_density_300m", "dev_dir_sin", "dev_dir_cos"]

K_FOLDS = 5

# 2.5M rows in front of the model per CV iteration, the default. Each
# iteration trains on k_folds - 1 folds, so rows_per_fold is what each fold
# has to contribute: rows_per_training_set // (k_folds - 1).
ROWS_PER_TRAINING_SET = 2_500_000

# Negative keep rate is P_FLOOR + A_k * exp(-dist / TAU_M). 200m is picked off
# the label=1 distance profile: converters sit at a median of 30 to 60m and a
# p90 under 250m, so the enrichment is spent almost entirely where conversion
# is actually possible. Past about 1km the rate is indistinguishable from
# P_FLOOR, which is deliberate, that far out nothing converts and the 1/p
# weight carries the population back anyway. Not exposed on the CLI, unlike
# TAU_M: it is a floor against zero probability, not a modeling choice worth
# sweeping per dataset.
TAU_M = 200.0
P_FLOOR = 0.01

# The two levers are separate on purpose. Duplicating rows relaxes
# min_data_in_leaf, which counts rows and ignores weights, so trees can split
# more finely around converters. The weight is what actually moves the loss,
# since LightGBM accumulates weighted gradients and hessians.
POS_ROW_DUP = 2

# None means derive it from the data: n_neg_natural / n_pos_natural, the
# scale_pos_weight ratio that makes label=1 and label!=1 carry equal total
# weight in the loss (see natural_pos_weight_mult). That ratio comes out
# around 355 on this data, since conversions are under 0.3% of samples, so
# it pulls the model much harder toward the positive class than a hand
# picked constant like the old default of 10.0 did. Pass --pos-weight-mult
# explicitly to pin a specific value instead, e.g. to reproduce a dataset
# built before this was auto derived.
POS_WEIGHT_MULT = None

SEED = 0
DRAW_SEED = 42

# Swap refinement settings. It converges in about 6 sweeps on the real table;
# the cap is only there so a pathological input cannot spin forever.
MAX_SWAP_SWEEPS = 60
SWAP_CANDIDATES = 220

DIST_COL = "dist_to_developed_2019_m"
# BAND_COLS comes from data_constants.py, shared with model_common.py.

# Positives first: label=1 balance is what drives the variance in AP across
# folds. Rows and label=2 matter too but they are worth less than getting the
# converter count even.
BALANCE_COLS = ["n1", "n2", "n"]
BALANCE_WEIGHTS = np.array([1.0, 0.6, 0.6])

QUANTILES = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]

M2_PER_ACRE = 4046.8564224
PIXEL_ACRES = 30 * 30 / M2_PER_ACRE


def dataset_dir(dataset_name: str) -> str:
    return os.path.join(DATASETS_DIR, dataset_name)


def config_path(dataset_name: str) -> str:
    return os.path.join(dataset_dir(dataset_name), "config.json")


def load_config(dataset_name: str) -> dict:
    path = config_path(dataset_name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_config(dataset_name: str, updates: dict) -> None:
    """Merges updates into the dataset's config.json rather than overwriting
    it, since folds and pool each contribute their own half of the picture
    and both need to survive in the same file."""
    cfg = load_config(dataset_name)
    cfg.update(updates)
    os.makedirs(dataset_dir(dataset_name), exist_ok=True)
    with open(config_path(dataset_name), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


def connect(with_neighborhood_features: bool = False,
            block_filter: pd.DataFrame | None = None) -> duckdb.DuckDBPyConnection:
    """samples view over the joined tiles. Pass block_filter (a block_id
    column, see load_cluster_blocks) to restrict it to one
    block_distance_profiles.py cluster before anything downstream runs off
    it, e.g. building a dataset out of just the near-development regime."""
    con = duckdb.connect()
    filter_join = ""
    if block_filter is not None:
        con.register("block_filter", block_filter[["block_id"]])
        filter_join = "JOIN block_filter bf USING (block_id)"
    if with_neighborhood_features:
        if not os.path.exists(NEIGHBORHOOD_FEATURES_PATH):
            raise SystemExit(f"{NEIGHBORHOOD_FEATURES_PATH} not found, run `build_train_pool.py features` first")
        con.execute(f"""
            CREATE OR REPLACE VIEW samples AS
            SELECT s.*, nf.* EXCLUDE ("row", "col")
            FROM read_parquet('{JOINED_GLOB}') s
            LEFT JOIN read_parquet('{NEIGHBORHOOD_FEATURES_PATH}') nf
                ON s."row" = nf."row" AND s."col" = nf."col"
            {filter_join}
        """)
    else:
        con.execute(f"""
            CREATE OR REPLACE VIEW samples AS
            SELECT * FROM read_parquet('{JOINED_GLOB}') s
            {filter_join}
        """)
    return con


def load_cluster_blocks(cluster_file: str, cluster: int) -> pd.DataFrame:
    """block_id set for one block_distance_profiles.py cluster, e.g. cluster 0,
    the regime closest to development by that script's centroid ordering."""
    if not os.path.exists(cluster_file):
        raise SystemExit(f"{cluster_file} not found, run "
                         f"`python -m scripts.dataset.block_distance_profiles cluster` first")
    clusters = pd.read_parquet(cluster_file)
    blocks = clusters.loc[clusters["cluster"] == cluster, ["block_id"]]
    if blocks.empty:
        raise SystemExit(f"no blocks with cluster == {cluster} in {cluster_file}")
    return blocks


def attach_folds(con: duckdb.DuckDBPyConnection, block_folds_path: str) -> pd.DataFrame:
    """Register a dataset's block_folds.parquet and expose the samples_fold view."""
    if not os.path.exists(block_folds_path):
        raise SystemExit(f"{block_folds_path} not found, run `build_train_pool.py folds` for this dataset first")
    folds = pd.read_parquet(block_folds_path)
    con.register("block_fold", folds[["block_id", "fold"]])
    con.execute("""
        CREATE OR REPLACE VIEW samples_fold AS
        SELECT s.*, bf.fold FROM samples s JOIN block_fold bf USING (block_id)
    """)
    return folds


def block_stats(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    # ORDER BY matters for more than tidiness: a parallel GROUP BY hands back
    # blocks in whatever order the threads finish in, and the seeded shuffle
    # below would then be permuting a different frame every run. Sorting here is
    # what makes the whole fold assignment reproducible.
    return con.execute("""
        SELECT block_id,
               COUNT(*) AS n,
               SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS n1,
               SUM(CASE WHEN label = 2 THEN 1 ELSE 0 END) AS n2
        FROM samples
        GROUP BY block_id
        ORDER BY block_id
    """).df()


def _imbalance(totals: np.ndarray, target: np.ndarray) -> float:
    """Weighted sum of squared relative deviation from the per fold target."""
    return float((BALANCE_WEIGHTS * ((totals - target) / target) ** 2).sum())


def assign_folds(stats: pd.DataFrame, k: int = K_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """Split blocks into k folds, evening out block count, rare class counts and rows.

    Block count is a hard quota; the rest is a soft cost the swap phase drives
    down. Returns stats with a `fold` column, in a shuffled row order.
    """
    rng = np.random.default_rng(seed)
    # Shuffle first so the 427 blocks with no converters, which all tie on n1,
    # don't get placed in parquet scan order.
    stats = stats.sample(frac=1, random_state=seed).reset_index(drop=True)

    metrics = stats[BALANCE_COLS].to_numpy(dtype=float)
    target = metrics.sum(axis=0) / k
    quota = np.full(k, len(stats) // k)
    quota[: len(stats) % k] += 1

    # Place the blocks that eat the biggest share of a target first. They are the
    # ones with nowhere to go once the folds start filling up.
    order = np.argsort(-(metrics / target * BALANCE_WEIGHTS).sum(axis=1), kind="mergesort")

    totals = np.zeros((k, len(BALANCE_COLS)))
    placed = np.zeros(k, dtype=int)
    fold = np.empty(len(stats), dtype=int)
    for i in order:
        open_folds = np.flatnonzero(placed < quota)
        cost = (BALANCE_WEIGHTS * ((totals[open_folds] + metrics[i] - target) / target) ** 2).sum(axis=1)
        j = int(open_folds[int(np.argmin(cost))])
        fold[i] = j
        totals[j] += metrics[i]
        placed[j] += 1

    # Greedy alone lands badly (rows come out 75% over on one fold and 64% under
    # on another) because the quota fills before the small blocks arrive to
    # correct it. Swapping pairs between folds fixes that and preserves the quota
    # for free, since a swap moves one block each way.
    best = _imbalance(totals, target)
    for sweep in range(MAX_SWAP_SWEEPS):
        improved = False
        for a in rng.permutation(len(stats)):
            for b in rng.choice(len(stats), SWAP_CANDIDATES, replace=False):
                fa, fb = fold[a], fold[b]
                if fa == fb:
                    continue
                trial = totals.copy()
                delta = metrics[a] - metrics[b]
                trial[fa] -= delta
                trial[fb] += delta
                score = _imbalance(trial, target)
                if score < best - 1e-12:
                    totals = trial
                    fold[a], fold[b] = fb, fa
                    best = score
                    improved = True
        if not improved:
            break
    print(f"  swap refinement: {sweep + 1} sweeps, imbalance {best:.3e}")

    stats["fold"] = fold
    return stats


def fold_summary(stats: pd.DataFrame) -> pd.DataFrame:
    out = stats.groupby("fold").agg(blocks=("block_id", "size"), rows=("n", "sum"),
                                    n1=("n1", "sum"), n2=("n2", "sum")).reset_index()
    for col, src in (("rows", "rows"), ("n1", "n1"), ("n2", "n2")):
        out[f"{col}_pct_dev"] = (out[src] / out[src].mean() - 1) * 100
    return out


def cmd_folds(args):
    if (args.cluster_file is None) != (args.cluster is None):
        raise SystemExit("--cluster-file and --cluster must be passed together")

    ddir = dataset_dir(args.dataset_name)
    block_folds_path = os.path.join(ddir, "block_folds.parquet")
    if os.path.exists(block_folds_path) and not args.force:
        print(f"{block_folds_path} exists, use --force to rebuild")
        return

    block_filter = None
    if args.cluster is not None:
        block_filter = load_cluster_blocks(args.cluster_file, args.cluster)
        print(f"restricting to cluster {args.cluster} ({len(block_filter)} blocks) from {args.cluster_file}")

    con = connect(block_filter=block_filter)
    print("aggregating blocks...")
    stats = block_stats(con)
    print(f"  {len(stats)} blocks, {stats['n'].sum():,} rows, "
          f"{stats['n1'].sum():,} label=1, {stats['n2'].sum():,} label=2")

    stats = assign_folds(stats, k=args.k_folds, seed=args.seed)
    summary = fold_summary(stats)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    for col in ("rows", "n1", "n2"):
        worst = summary[f"{col}_pct_dev"].abs().max()
        assert worst < 1.0, f"{col} is {worst:.2f}% off target, expected under 1%"
    assert stats["block_id"].is_unique, "a block landed in more than one fold"

    os.makedirs(ddir, exist_ok=True)
    stats.sort_values("block_id").to_parquet(block_folds_path, index=False)
    print(f"wrote {block_folds_path}")

    save_config(args.dataset_name, {
        "dataset_name": args.dataset_name,
        "k_folds": args.k_folds,
        "seed": args.seed,
        "nlcd_release": os.path.basename(os.path.dirname(NLCD_2019_PATH)),
        "alphaearth_dataset": ALPHAEARTH_DATASET,
        "cluster_file": args.cluster_file,
        "cluster": args.cluster,
    })


def fold_calibration(con: duckdb.DuckDBPyConnection, tau_m: float, p_floor: float,
                     pos_row_dup: int, rows_per_fold: int) -> pd.DataFrame:
    """Solve the per fold decay constant A_k so every fold contributes rows_per_fold.

    Closed form because p never gets near 1 on this data (it tops out around
    0.17 at the 30m minimum distance), so the LEAST(1.0, ...) clamp in the draw
    never actually bites and the expected count is linear in A_k. The assert
    below is what guarantees that stays true.
    """
    agg = con.execute(f"""
        SELECT fold,
               COUNT(*) AS n,
               SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS n1,
               SUM(CASE WHEN label = 2 THEN 1 ELSE 0 END) AS n2,
               SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS n0,
               AVG(CASE WHEN label = 0 THEN EXP(-{DIST_COL} / {tau_m}) END) AS mean_decay
        FROM samples_fold
        GROUP BY fold
        ORDER BY fold
    """).df()

    agg["budget0"] = rows_per_fold - pos_row_dup * agg["n1"] - agg["n2"]
    agg["A"] = (agg["budget0"] / agg["n0"] - p_floor) / agg["mean_decay"]

    assert (agg["budget0"] > 0).all(), "rare classes alone overflow the per fold budget"
    assert (agg["A"] > 0).all(), "P_FLOOR alone already overshoots the budget, lower it"
    p_max = p_floor + agg["A"] * np.exp(-30.0 / tau_m)
    assert (p_max < 1.0).all(), f"keep rate saturates (max {p_max.max():.3f}), closed form is invalid"
    return agg


def natural_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per fold natural label counts. The light half of what fold_calibration
    computes, no decay constant to solve, for --no-thinning's straight
    keep-everything draw."""
    return con.execute("""
        SELECT fold, COUNT(*) AS n,
               SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS n1,
               SUM(CASE WHEN label = 2 THEN 1 ELSE 0 END) AS n2,
               SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS n0
        FROM samples_fold
        GROUP BY fold
        ORDER BY fold
    """).df()


def natural_pos_weight_mult(fold_calib: pd.DataFrame) -> float:
    """n_neg_natural / n_pos_natural before any subsampling, i.e. the classic
    scale_pos_weight ratio: weighting each label=1 row by this much makes the
    positive and negative classes carry equal total weight in the loss, so
    the model trains as if conversions were as common as everything else.
    label=2 (converted, but not to development) counts on the negative side
    here, same as it does in label_bin.

    Reuses fold_calibration's per fold n0/n1/n2 counts rather than a fresh
    query, since cmd_pool already has them on hand.
    """
    n_pos = fold_calib["n1"].sum()
    n_neg = fold_calib["n0"].sum() + fold_calib["n2"].sum()
    return n_neg / n_pos


def cmd_pool(args):
    if (args.cluster_file is None) != (args.cluster is None):
        raise SystemExit("--cluster-file and --cluster must be passed together")

    ddir = dataset_dir(args.dataset_name)
    block_folds_path = os.path.join(ddir, "block_folds.parquet")
    train_pool_path = os.path.join(ddir, "train_pool.parquet")

    if os.path.exists(train_pool_path) and not args.force:
        print(f"{train_pool_path} exists, use --force to rebuild")
        return

    cfg = load_config(args.dataset_name)
    if "k_folds" not in cfg:
        raise SystemExit(f"no fold assignment for dataset '{args.dataset_name}', "
                         f"run `folds --dataset-name {args.dataset_name}` first")
    k_folds = cfg["k_folds"]
    rows_per_fold = args.rows_per_training_set // (k_folds - 1)

    block_filter = None
    if args.cluster is not None:
        block_filter = load_cluster_blocks(args.cluster_file, args.cluster)
        print(f"restricting to cluster {args.cluster} ({len(block_filter)} blocks) from {args.cluster_file}")

    con = connect(with_neighborhood_features=args.with_neighborhood_features, block_filter=block_filter)
    attach_folds(con, block_folds_path)

    # --no-thinning skips the exponential distance decay draw and keeps every
    # label=0 pixel, so there's no per fold decay constant A to solve for.
    if args.no_thinning:
        calib = natural_counts(con)
        p_include_expr = "1.0"
        join_calib = ""
    else:
        calib = fold_calibration(con, args.tau_m, P_FLOOR, args.pos_row_dup, rows_per_fold)
        con.register("fold_calib", calib[["fold", "A"]])
        p_include_expr = (f"CASE WHEN s.label IN (1, 2) THEN 1.0 ELSE "
                          f"LEAST(1.0, {P_FLOOR} + c.A * EXP(-s.{DIST_COL} / {args.tau_m})) END")
        join_calib = "JOIN fold_calib c USING (fold)"
    print(calib.to_string(index=False, float_format=lambda v: f"{v:,.5f}"))

    pos_weight_mult = args.pos_weight_mult
    if pos_weight_mult is None:
        pos_weight_mult = natural_pos_weight_mult(calib)
        print(f"pos_weight_mult not given, derived {pos_weight_mult:,.2f} from natural class counts")

    pos_weight = pos_weight_mult / args.pos_row_dup
    bands = ", ".join(BAND_COLS)
    carried = f'"row", "col", block_id, fold, x, y, {DIST_COL}, label, label_bin, p_include, sample_weight'
    if args.with_neighborhood_features:
        carried += ", " + ", ".join(NEIGHBORHOOD_FEATURE_COLS)

    # random() is not reproducible across thread counts, so the draw hangs off a
    # hash of the pixel key instead. (row, col) is unique statewide.
    dup_selects = "\n            UNION ALL\n            ".join(
        f"SELECT {carried}, TRUE AS is_dup, {bands} FROM drawn WHERE label = 1"
        for _ in range(args.pos_row_dup - 1)
    )
    extra_union = f"\n            UNION ALL\n            {dup_selects}" if dup_selects else ""

    os.makedirs(ddir, exist_ok=True)
    con.execute(f"""
        COPY (
            WITH scored AS (
                SELECT s.*,
                       {p_include_expr} AS p_include,
                       (hash(s."row", s."col", {args.draw_seed}) % 1073741824) / 1073741824.0 AS u
                FROM samples_fold s
                {join_calib}
            ),
            drawn AS (
                SELECT *,
                       CASE WHEN label = 1 THEN {pos_weight} ELSE 1.0 / p_include END AS sample_weight,
                       CASE WHEN label = 1 THEN 1 ELSE 0 END AS label_bin
                FROM scored
                WHERE u < p_include
            )
            SELECT {carried}, FALSE AS is_dup, {bands} FROM drawn{extra_union}
        ) TO '{train_pool_path}' (FORMAT PARQUET)
    """)

    counts = con.execute(f"""
        SELECT fold, COUNT(*) AS pool_rows,
               SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS n1,
               SUM(CASE WHEN label = 2 THEN 1 ELSE 0 END) AS n2,
               SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS n0,
               SUM(sample_weight) AS sum_w
        FROM read_parquet('{train_pool_path}')
        GROUP BY fold ORDER BY fold
    """).df()
    print(counts.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    print(f"wrote {train_pool_path} ({counts['pool_rows'].sum():,} rows)")

    save_config(args.dataset_name, {
        "dataset_name": args.dataset_name,
        "no_thinning": args.no_thinning,
        # moot once --no-thinning skips the decay draw; null them out rather
        # than record defaults that were never actually used
        "tau_m": None if args.no_thinning else args.tau_m,
        "p_floor": None if args.no_thinning else P_FLOOR,
        "rows_per_training_set": None if args.no_thinning else args.rows_per_training_set,
        "rows_per_fold": None if args.no_thinning else rows_per_fold,
        "draw_seed": None if args.no_thinning else args.draw_seed,
        "pos_row_dup": args.pos_row_dup,
        "pos_weight_mult": pos_weight_mult,
        "with_neighborhood_features": args.with_neighborhood_features,
        "cluster_file": args.cluster_file,
        "cluster": args.cluster,
        "nlcd_release": os.path.basename(os.path.dirname(NLCD_2019_PATH)),
        "alphaearth_dataset": ALPHAEARTH_DATASET,
    })


def florida_window():
    """Reproduces wetland_sample_labels.ipynb's exact window over the 2019
    NLCD raster: Florida's bounding box plus a 15km buffer. Has to match
    that notebook exactly, since that is what keeps a row, col pair here
    pointing at the same pixel it points at in the sample table. Returns
    the rasterio Window and its affine transform."""
    us_states = gpd.read_file(STATE_BOUNDARY_SHP)
    florida = us_states.loc[us_states["NAME"] == "Florida"]
    with rasterio.open(NLCD_2019_PATH) as src:
        florida_proj = florida.to_crs(src.crs)
        florida_geom = florida_proj.geometry.union_all()
        minx, miny, maxx, maxy = florida_geom.bounds
        buffered_bounds = (minx - BUFFER_M, miny - BUFFER_M, maxx + BUFFER_M, maxy + BUFFER_M)
        window = src.window(*buffered_bounds).round_offsets(op="floor").round_lengths(op="ceil")
        win_transform = window_transform(window, src.transform)
    return window, win_transform


def _local_density(mask: np.ndarray, radius_m: float) -> np.ndarray:
    """Share of True pixels in a square neighborhood approximating the
    requested radius. A square box filter, not a true circle: uniform_filter
    is a couple of cheap passes over the array, while an exact circular
    average would need a slower convolution, and at this pixel count that
    cost is not worth it for what is still just a proxy feature. Box side is
    the requested radius rounded to the nearest whole 30m pixel, doubled."""
    r_px = max(1, round(radius_m / PIXEL_M))
    return uniform_filter(mask.astype(np.float32), size=2 * r_px + 1, mode="nearest")


def _edge_mask(developed: np.ndarray) -> np.ndarray:
    """Pixels sitting on the boundary between developed and not developed
    land: present in a slightly grown version of the mask but not in a
    slightly shrunk one, the standard morphological way to find edges."""
    return binary_dilation(developed) & ~binary_erosion(developed)


def _direction_to_nearest_developed(developed: np.ndarray, win_transform,
                                    rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sine and cosine of the compass bearing from each sample pixel toward
    the nearest 2019 developed pixel. A raw angle is a bad feature here.
    359 degrees and 1 degree point almost the same direction but look
    nothing alike as numbers, sine and cosine avoid that wraparound.

    Only the location of the nearest developed pixel is needed, the
    distance itself already exists as dist_to_developed_2019_m, so
    return_distances is turned off, which skips building a second array the
    size of the whole window that nothing here would use.
    """
    indices = distance_transform_edt(
        ~developed, sampling=(PIXEL_M, PIXEL_M), return_distances=False, return_indices=True,
    )
    nearest_row = indices[0][rows, cols]
    nearest_col = indices[1][rows, cols]
    del indices
    gc.collect()

    dx_m = (nearest_col.astype(np.float64) - cols) * win_transform.a
    dy_m = (nearest_row.astype(np.float64) - rows) * win_transform.e
    theta = np.arctan2(dy_m, dx_m)
    return np.sin(theta).astype(np.float32), np.cos(theta).astype(np.float32)


def cmd_features(args):
    if os.path.exists(NEIGHBORHOOD_FEATURES_PATH) and not args.force:
        print(f"{NEIGHBORHOOD_FEATURES_PATH} exists, use --force to rebuild")
        return

    sample = pd.read_parquet(SAMPLE_LABELS_PATH, columns=["row", "col"])
    rows = sample["row"].to_numpy()
    cols = sample["col"].to_numpy()
    print(f"{len(sample):,} sample pixels, reading the 2019 NLCD window...")

    window, win_transform = florida_window()
    with rasterio.open(NLCD_2019_PATH) as src:
        lc_2019 = src.read(1, window=window)
    print(f"window shape {lc_2019.shape}, {lc_2019.size:,} pixels")

    # unclipped to Florida, same as dist_to_developed_2019_m: development
    # just across the state line still counts, which is the whole point of
    # BUFFER_M
    developed = np.isin(lc_2019, DEVELOPED_CLASSES)
    del lc_2019
    gc.collect()

    out = {"row": rows.astype(np.int32), "col": cols.astype(np.int32)}

    for col_name, radius_m in NEIGHBORHOOD_RADII_M.items():
        print(f"  {col_name} ...")
        density = _local_density(developed, radius_m)
        out[col_name] = density[rows, cols].astype(np.float32)
        del density
        gc.collect()

    print("  edge_density_300m ...")
    edges = _edge_mask(developed)
    edge_density = _local_density(edges, EDGE_DENSITY_RADIUS_M)
    out["edge_density_300m"] = edge_density[rows, cols].astype(np.float32)
    del edges, edge_density
    gc.collect()

    print("  dev_dir_sin, dev_dir_cos ...")
    out["dev_dir_sin"], out["dev_dir_cos"] = _direction_to_nearest_developed(developed, win_transform, rows, cols)
    del developed
    gc.collect()

    os.makedirs(os.path.dirname(NEIGHBORHOOD_FEATURES_PATH), exist_ok=True)
    pd.DataFrame(out).to_parquet(NEIGHBORHOOD_FEATURES_PATH, index=False)
    print(f"wrote {NEIGHBORHOOD_FEATURES_PATH} ({len(sample):,} rows, "
          f"{len(NEIGHBORHOOD_FEATURE_COLS)} feature columns)")


def _quantiles(con, table: str, where: str, tag: str) -> pd.DataFrame:
    cols = ", ".join(f"QUANTILE_CONT({DIST_COL}, {q}) AS p{int(q * 100)}" for q in QUANTILES)
    df = con.execute(f"""
        SELECT fold, {cols}, AVG({DIST_COL}) AS mean
        FROM {table} WHERE {where}
        GROUP BY fold ORDER BY fold
    """).df()
    df.insert(1, "dist_set", tag)
    return df


def cmd_report(args):
    ddir = dataset_dir(args.dataset_name)
    block_folds_path = os.path.join(ddir, "block_folds.parquet")
    train_pool_path = os.path.join(ddir, "train_pool.parquet")

    cfg = load_config(args.dataset_name)
    print(f"== dataset '{args.dataset_name}'")
    print(json.dumps(cfg, indent=2, sort_keys=True))

    con = connect()
    folds = attach_folds(con, block_folds_path)
    con.execute(f"CREATE OR REPLACE VIEW pool AS SELECT * FROM read_parquet('{train_pool_path}')")

    natural = con.execute("""
        SELECT fold, COUNT(*) AS nat_rows,
               SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS nat_n0,
               SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS nat_n1,
               SUM(CASE WHEN label = 2 THEN 1 ELSE 0 END) AS nat_n2
        FROM samples_fold GROUP BY fold ORDER BY fold
    """).df()

    pool = con.execute("""
        SELECT fold, COUNT(*) AS pool_rows,
               SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS pool_n0,
               SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS pool_n1,
               SUM(CASE WHEN label = 2 THEN 1 ELSE 0 END) AS pool_n2,
               SUM(sample_weight) AS sum_w,
               SUM(sample_weight * sample_weight) AS sum_w2,
               SUM(CASE WHEN label = 1 THEN sample_weight ELSE 0 END) AS pos_w
        FROM pool GROUP BY fold ORDER BY fold
    """).df()

    rep = natural.merge(pool, on="fold").merge(
        folds.groupby("fold").size().rename("blocks").reset_index(), on="fold")
    rep["mean_weight"] = rep["sum_w"] / rep["pool_rows"]
    # The row share is what a printout of the pool looks like; the weight share
    # is what the loss actually sees. They differ by more than an order of
    # magnitude here, so both get reported.
    rep["pos_share_rows_pct"] = 100 * rep["pool_n1"] / rep["pool_rows"]
    rep["pos_share_weight_pct"] = 100 * rep["pos_w"] / rep["sum_w"]
    rep["kish_ess"] = rep["sum_w"] ** 2 / rep["sum_w2"]
    rep["nat_prevalence_pct"] = 100 * rep["nat_n1"] / rep["nat_rows"]
    rep["converted_acres"] = rep["nat_n1"] * PIXEL_ACRES

    cols = ["fold", "blocks", "nat_rows", "nat_n0", "nat_n1", "nat_n2", "nat_prevalence_pct",
            "pool_rows", "pool_n0", "pool_n1", "pool_n2", "mean_weight",
            "pos_share_rows_pct", "pos_share_weight_pct", "kish_ess", "converted_acres"]
    print("\n== per fold counts")
    print(rep[cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    dist = pd.concat([
        _quantiles(con, "samples_fold", "label = 0", "natural_label0"),
        _quantiles(con, "pool", "label = 0", "pool_label0"),
        _quantiles(con, "samples_fold", "label = 1", "natural_label1"),
        _quantiles(con, "samples_fold", "label = 2", "natural_label2"),
    ])
    print(f"\n== {DIST_COL} (metres)")
    print(dist.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ft = sub.add_parser("features")
    p_ft.add_argument("--force", action="store_true")
    p_ft.set_defaults(func=cmd_features)

    p_f = sub.add_parser("folds")
    p_f.add_argument("--dataset-name", required=True)
    p_f.add_argument("--force", action="store_true")
    p_f.add_argument("--k-folds", type=int, default=K_FOLDS)
    p_f.add_argument("--seed", type=int, default=SEED)
    p_f.add_argument("--cluster-file", default=None,
                     help="block_distance_profiles.py cluster assignment parquet; pass with "
                          "--cluster to restrict the block universe to one cluster")
    p_f.add_argument("--cluster", type=int, default=None,
                     help="cluster id to restrict to, e.g. 0 for the near-development regime")
    p_f.set_defaults(func=cmd_folds)

    p_p = sub.add_parser("pool")
    p_p.add_argument("--dataset-name", required=True)
    p_p.add_argument("--force", action="store_true")
    p_p.add_argument("--tau-m", type=float, default=TAU_M)
    p_p.add_argument("--pos-row-dup", type=int, default=POS_ROW_DUP)
    p_p.add_argument("--pos-weight-mult", type=float, default=POS_WEIGHT_MULT,
                     help="fixed multiplier for label=1 rows; omit to derive it from the "
                          "natural class balance instead (see natural_pos_weight_mult)")
    p_p.add_argument("--rows-per-training-set", type=int, default=ROWS_PER_TRAINING_SET)
    p_p.add_argument("--draw-seed", type=int, default=DRAW_SEED)
    p_p.add_argument("--with-neighborhood-features", action="store_true")
    p_p.add_argument("--no-thinning", action="store_true",
                     help="keep every label=0 pixel instead of the exponential distance decay draw")
    p_p.add_argument("--cluster-file", default=None,
                     help="block_distance_profiles.py cluster assignment parquet, same file used "
                          "for `folds`; pass with --cluster to restrict the block universe")
    p_p.add_argument("--cluster", type=int, default=None,
                     help="cluster id to restrict to, must match what `folds` was built with")
    p_p.set_defaults(func=cmd_pool)

    p_r = sub.add_parser("report")
    p_r.add_argument("--dataset-name", required=True)
    p_r.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
