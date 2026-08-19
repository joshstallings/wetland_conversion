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
      data/processed/datasets/<dataset-name>/block_folds.parquet.

  pool
      Draws the training pool. Every label=1 and label=2 pixel is kept,
      label=1 duplicated pos-row-dup times over, and label=0 is kept with a
      probability that decays with distance to 2019 development so the pool
      concentrates where conversion actually happens. Each fold gets its own
      decay constant A_k, solved in closed form so every fold contributes the
      same row budget no matter where its distance distribution happens to
      sit. Requires a fold assignment already built under the same
      dataset-name. Writes
      data/processed/datasets/<dataset-name>/gb_train_pool.parquet.

Every constant a dataset was actually built with, defaulted or overridden,
gets written to data/processed/datasets/<dataset-name>/config.json, so the
directory is self documenting: no need to cross reference a notebook cell or
shell history to know what produced it.

On weighting: label=0 rows carry 1/p, so in expectation they reconstruct the
true label=0 population, and label=2 sits at p=1 and weight 1, so it
represents exactly itself. That leaves the deliberate pos-weight-mult boost
on label=1 as the only place the pool departs from reality, which is what
makes the model's scores recoverable back to real conversion probabilities
with p_true = p / (p + pos_weight_mult * (1 - p)). scripts/gb_common.py's
calibrate() applies that correction before anything gets reported in
probability units.

Usage:
    python scripts/build_train_pool.py folds --dataset-name NAME [--force]
        [--k-folds 5] [--seed 0]
    python scripts/build_train_pool.py pool --dataset-name NAME [--force]
        [--tau-m 200.0] [--pos-row-dup 2] [--pos-weight-mult 10.0]
        [--rows-per-training-set 2500000] [--draw-seed 42]
    python scripts/build_train_pool.py report --dataset-name NAME
"""

import argparse
import json
import os

import duckdb
import numpy as np
import pandas as pd

JOINED_GLOB = "data/processed/alphaearth_wetland_joined/*.parquet"
DATASETS_DIR = "data/processed/datasets"

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
# since LightGBM accumulates weighted gradients and hessians: at the defaults
# below it takes effective positive prevalence from 0.28% natural to about
# 2.7%.
POS_ROW_DUP = 2
POS_WEIGHT_MULT = 10.0

SEED = 0
DRAW_SEED = 42

# Swap refinement settings. It converges in about 6 sweeps on the real table;
# the cap is only there so a pathological input cannot spin forever.
MAX_SWAP_SWEEPS = 60
SWAP_CANDIDATES = 220

DIST_COL = "dist_to_developed_2019_m"
BAND_COLS = [f"A{b:02d}_{year}" for year in (2017, 2018, 2019) for b in range(64)]

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


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW samples AS SELECT * FROM read_parquet('{JOINED_GLOB}')")
    return con


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
    ddir = dataset_dir(args.dataset_name)
    block_folds_path = os.path.join(ddir, "block_folds.parquet")
    if os.path.exists(block_folds_path) and not args.force:
        print(f"{block_folds_path} exists, use --force to rebuild")
        return

    con = connect()
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


def cmd_pool(args):
    ddir = dataset_dir(args.dataset_name)
    block_folds_path = os.path.join(ddir, "block_folds.parquet")
    train_pool_path = os.path.join(ddir, "gb_train_pool.parquet")

    if os.path.exists(train_pool_path) and not args.force:
        print(f"{train_pool_path} exists, use --force to rebuild")
        return

    cfg = load_config(args.dataset_name)
    if "k_folds" not in cfg:
        raise SystemExit(f"no fold assignment for dataset '{args.dataset_name}', "
                         f"run `folds --dataset-name {args.dataset_name}` first")
    k_folds = cfg["k_folds"]
    rows_per_fold = args.rows_per_training_set // (k_folds - 1)

    con = connect()
    attach_folds(con, block_folds_path)

    calib = fold_calibration(con, args.tau_m, P_FLOOR, args.pos_row_dup, rows_per_fold)
    print(calib.to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    con.register("fold_calib", calib[["fold", "A"]])

    pos_weight = args.pos_weight_mult / args.pos_row_dup
    bands = ", ".join(BAND_COLS)
    carried = f'"row", "col", block_id, fold, x, y, {DIST_COL}, label, label_bin, p_include, sample_weight'

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
                       CASE WHEN s.label IN (1, 2) THEN 1.0
                            ELSE LEAST(1.0, {P_FLOOR} + c.A * EXP(-s.{DIST_COL} / {args.tau_m}))
                       END AS p_include,
                       (hash(s."row", s."col", {args.draw_seed}) % 1073741824) / 1073741824.0 AS u
                FROM samples_fold s
                JOIN fold_calib c USING (fold)
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
        "tau_m": args.tau_m,
        "p_floor": P_FLOOR,
        "pos_row_dup": args.pos_row_dup,
        "pos_weight_mult": args.pos_weight_mult,
        "rows_per_training_set": args.rows_per_training_set,
        "rows_per_fold": rows_per_fold,
        "draw_seed": args.draw_seed,
    })


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
    train_pool_path = os.path.join(ddir, "gb_train_pool.parquet")

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

    p_f = sub.add_parser("folds")
    p_f.add_argument("--dataset-name", required=True)
    p_f.add_argument("--force", action="store_true")
    p_f.add_argument("--k-folds", type=int, default=K_FOLDS)
    p_f.add_argument("--seed", type=int, default=SEED)
    p_f.set_defaults(func=cmd_folds)

    p_p = sub.add_parser("pool")
    p_p.add_argument("--dataset-name", required=True)
    p_p.add_argument("--force", action="store_true")
    p_p.add_argument("--tau-m", type=float, default=TAU_M)
    p_p.add_argument("--pos-row-dup", type=int, default=POS_ROW_DUP)
    p_p.add_argument("--pos-weight-mult", type=float, default=POS_WEIGHT_MULT)
    p_p.add_argument("--rows-per-training-set", type=int, default=ROWS_PER_TRAINING_SET)
    p_p.add_argument("--draw-seed", type=int, default=DRAW_SEED)
    p_p.set_defaults(func=cmd_pool)

    p_r = sub.add_parser("report")
    p_r.add_argument("--dataset-name", required=True)
    p_r.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
