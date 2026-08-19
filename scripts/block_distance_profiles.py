"""Group the 1,727 10km blocks by the shape of their distance to development
distribution. Diagnostic only: nothing here feeds the folds, the training pool,
or the feature set.

dist_to_developed_2019_m is the strongest single feature in every experiment so
far, and the whole label=0 thinning in build_train_pool.py is an exponential
decay in it. But it has only ever been looked at per pixel or per fold. The
question this answers is whether Florida wetland falls into a few distinct
distance regimes or is one continuum.

  profiles
      One duckdb pass over the joined tiles, reducing each block to the
      quantile function of its pixels' distance to 2019 development, sampled
      at P_GRID. Writes data/processed/block_distance_profiles.parquet.

  cluster
      K-means over those quantile functions, plus the k sweep and stability
      check that say how many regimes are actually there. Writes
      data/processed/block_distance_clusters.parquet and the sweep csv
      alongside it.

  report
      Per cluster block counts, conversion rates and distance quantiles.
      Pass --dataset-name to also crosstab clusters against that dataset's
      CV folds.

On the representation: a block is stored as a quantile function rather than a
normalized histogram because Euclidean distance between quantile functions is
the 2-Wasserstein distance between the underlying distributions. K-means on
that vector is therefore Wasserstein k-means, which is the geometry that
belongs on an ordered support. Two blocks with the same profile shape shifted
200m apart come out close, where bin-wise Euclidean on histograms would call
them nearly orthogonal. It also keeps the centroids interpretable: a centroid
is itself a valid quantile function and plots straight back as an ECDF.

The caveat that has to be checked rather than assumed: a distribution family
that is mostly a location shift in log space will cluster on its median and
nothing else, and k-means will happily hand back "clusters" that are just bins
of median distance. So cluster decomposes each profile into level, spread and
shape, reports the share of variance level alone accounts for, and offers
--shape-only as the run that isolates whatever is left.

Usage:
    python scripts/block_distance_profiles.py profiles [--force]
    python scripts/block_distance_profiles.py cluster [--k 6] [--force]
        [--shape-only] [--min-block-pixels 500] [--seed 0]
    python scripts/block_distance_profiles.py report [--shape-only]
        [--dataset-name NAME]
"""

import argparse
import os

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score

JOINED_GLOB = "data/processed/alphaearth_wetland_joined/*.parquet"
DATASETS_DIR = "data/processed/datasets"
PROFILES_PATH = "data/processed/block_distance_profiles.parquet"

DIST_COL = "dist_to_developed_2019_m"

# Same 10km grid and absolute NLCD origin join_alphaearth_samples.py stamps
# block_id with. Repeated rather than imported because that module pulls in
# rasterio and the earth engine client just to be loaded.
BLOCK_SIZE_M = 10_000
NLCD_ORIGIN_X, NLCD_ORIGIN_Y = -2415585, 3314805

# The probability grid a block's profile is sampled on. Trimmed off both ends:
# below 0.025 and above 0.975 the quantile is set by a handful of pixels even
# in a big block, so those knots are mostly noise about where the single
# nearest and single farthest pixel happened to land.
P_GRID = np.round(np.arange(0.025, 0.9751, 0.025), 4)
PROFILE_COLS = [f"q{int(round(p * 1000)):04d}" for p in P_GRID]

# Blocks under this many pixels are not fit against. The 1st percentile of
# block size is 97 pixels and the smallest block has 1, which QUANTILE_CONT
# turns into a flat profile at zero spread that would drag a centroid. 62
# blocks fall below 500, together 0.021% of all rows. They still get assigned,
# to the nearest fitted centroid, flagged is_small.
MIN_BLOCK_PIXELS = 500

K_RANGE = range(2, 13)
# Picked off the sweep: k=2 wins silhouette outright but only says "near and
# far", and past 5 the subsample ARI starts falling apart. k=4 is the best
# silhouette of the non trivial options (0.460) and the most stable (0.944).
DEFAULT_K = 4
SEED = 0
N_INIT = 20

# Stability check: refit on random subsamples and compare each fit's labels
# back to the full fit on the blocks they share. A k whose ARI collapses is
# not real structure no matter what silhouette says about it.
STABILITY_RUNS = 20
STABILITY_FRAC = 0.8

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


def clusters_path(shape_only: bool) -> str:
    suffix = "_shape" if shape_only else ""
    return f"data/processed/block_distance_clusters{suffix}.parquet"


def sweep_path(shape_only: bool) -> str:
    suffix = "_shape" if shape_only else ""
    return f"data/processed/block_distance_cluster_sweep{suffix}.csv"


def connect() -> duckdb.DuckDBPyConnection:
    """Plain view over the joined tiles.

    Deliberately not gb_common.connect_raw(): that one LEFT JOINs the 388MB
    neighborhood features table unconditionally, and none of those columns are
    wanted here. Straight off the glob, parquet column pruning reads 5 columns
    out of 203.
    """
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW samples AS SELECT * FROM read_parquet('{JOINED_GLOB}')")
    return con


def block_centers(block_id: pd.Series) -> pd.DataFrame:
    """Recover each block's grid cell center from its id.

    The center of the 10km cell, not the centroid of the wetland pixels in it,
    so the map draws a clean lattice. add_block_id floors into this same grid
    off the absolute NLCD origin, and y runs north to south.
    """
    parts = block_id.str.extract(r"^b(\d{4})_(\d{4})$").astype(int)
    return pd.DataFrame({
        "block_row": parts[0].to_numpy(),
        "block_col": parts[1].to_numpy(),
        "x_center": NLCD_ORIGIN_X + (parts[1].to_numpy() + 0.5) * BLOCK_SIZE_M,
        "y_center": NLCD_ORIGIN_Y - (parts[0].to_numpy() + 0.5) * BLOCK_SIZE_M,
    }, index=block_id.index)


def cmd_profiles(args):
    if os.path.exists(PROFILES_PATH) and not args.force:
        print(f"{PROFILES_PATH} exists, use --force to rebuild")
        return

    con = connect()
    p_list = "[" + ", ".join(f"{p:.4f}" for p in P_GRID) + "]"

    # ORDER BY for the same reason build_train_pool.block_stats has it: a
    # parallel GROUP BY hands back groups in thread completion order, and the
    # seeded k-means below would then be fed a differently ordered frame every
    # run.
    print("aggregating block distance profiles...")
    df = con.execute(f"""
        SELECT block_id,
               COUNT(*) AS n,
               SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS n1,
               SUM(CASE WHEN label = 2 THEN 1 ELSE 0 END) AS n2,
               AVG(x) AS x_mean,
               AVG(y) AS y_mean,
               AVG({DIST_COL}) AS mean_dist_m,
               MIN({DIST_COL}) AS min_dist_m,
               MAX({DIST_COL}) AS max_dist_m,
               COUNT(*) - COUNT({DIST_COL}) AS n_null_dist,
               QUANTILE_CONT({DIST_COL}, {p_list}) AS qs
        FROM samples
        GROUP BY block_id
        ORDER BY block_id
    """).df()

    q = pd.DataFrame(np.vstack(df.pop("qs").to_numpy()), columns=PROFILE_COLS, index=df.index)
    df = pd.concat([df, block_centers(df["block_id"]), q], axis=1)

    assert df["block_id"].is_unique, "a block_id came back twice"
    assert (df["n_null_dist"] == 0).all(), "null distance in the joined table"
    assert np.isfinite(df[PROFILE_COLS].to_numpy()).all(), "non finite quantile"
    # Wetland pixels are by definition not developed in 2019, so no pixel can
    # sit closer than one 30m cell to development.
    assert df["min_dist_m"].min() >= 30.0, f"distance below one pixel: {df['min_dist_m'].min()}"
    monotone = np.diff(df[PROFILE_COLS].to_numpy(), axis=1) >= -1e-6
    assert monotone.all(), "quantile function is not nondecreasing"

    print(f"  {len(df)} blocks, {df['n'].sum():,} rows, "
          f"{int(df['n1'].sum()):,} label=1, {int(df['n2'].sum()):,} label=2")
    print(f"  block size: min {df['n'].min():,}, median {int(df['n'].median()):,}, max {df['n'].max():,}")
    print(f"  {(df['n'] < MIN_BLOCK_PIXELS).sum()} blocks under {MIN_BLOCK_PIXELS} pixels "
          f"({df.loc[df['n'] < MIN_BLOCK_PIXELS, 'n'].sum() / df['n'].sum() * 100:.3f}% of rows)")

    _check_against_folds(df)

    os.makedirs(os.path.dirname(PROFILES_PATH), exist_ok=True)
    df.to_parquet(PROFILES_PATH, index=False)
    print(f"wrote {PROFILES_PATH}")


def _check_against_folds(df: pd.DataFrame) -> None:
    """Cross check the aggregate against whatever block_folds.parquet is on disk.

    Free end to end test: `folds` groups the same table by the same key, so if
    n, n1, n2 disagree then one of the two is reading something different.
    """
    for name in sorted(os.listdir(DATASETS_DIR)) if os.path.isdir(DATASETS_DIR) else []:
        path = os.path.join(DATASETS_DIR, name, "block_folds.parquet")
        if not os.path.exists(path):
            continue
        ref = pd.read_parquet(path)
        merged = df.merge(ref, on="block_id", how="outer", suffixes=("", "_ref"), indicator=True)
        assert (merged["_merge"] == "both").all(), f"block_id set differs from {path}"
        for col in ("n", "n1", "n2"):
            assert np.allclose(merged[col], merged[f"{col}_ref"]), f"{col} differs from {path}"
        print(f"  matches {path} ({len(ref)} blocks)")


def load_profiles() -> pd.DataFrame:
    if not os.path.exists(PROFILES_PATH):
        raise SystemExit(f"{PROFILES_PATH} not found, run `block_distance_profiles.py profiles` first")
    return pd.read_parquet(PROFILES_PATH)


def profile_matrix(df: pd.DataFrame, shape_only: bool) -> np.ndarray:
    """Quantile functions in log10 metres, one row per block.

    QUANTILE_CONT is monotone equivariant so the log can be taken after the
    aggregation. Under --shape-only each row has its own median subtracted off,
    which strips the level and leaves whatever the profile says beyond "how far
    is this block from development on average".
    """
    x = np.log10(df[PROFILE_COLS].to_numpy(dtype=np.float64))
    if shape_only:
        x = x - np.median(x, axis=1, keepdims=True)
    return x


def variance_decomposition(x: np.ndarray) -> dict:
    """Split the between block variation into level, shape, and their covariance.

    Writing each profile as level plus shape, x_ij = L_i + s_ij, the sum of
    squares about the per knot mean is level + shape + a cross term. The cross
    term is not zero here and is worth printing rather than hiding: it comes
    out negative, meaning blocks farther from development have slightly
    tighter profiles in log space, so level and shape partly cancel.
    """
    level = np.median(x, axis=1, keepdims=True)
    shape = x - level
    lc = level - level.mean()
    sc = shape - shape.mean(axis=0)
    total = float(((x - x.mean(axis=0)) ** 2).sum())
    level_ss = float((lc ** 2).sum() * x.shape[1])
    shape_ss = float((sc ** 2).sum())
    cross_ss = float(2 * (lc * sc).sum())
    return {
        "total": total,
        "level_share": level_ss / total,
        "shape_share": shape_ss / total,
        "cross_share": cross_ss / total,
    }


def fit_kmeans(x: np.ndarray, k: int, seed: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(x)


def sweep(x: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in K_RANGE:
        km = fit_kmeans(x, k, seed)
        labels = km.labels_
        aris = []
        for run in range(STABILITY_RUNS):
            idx = rng.choice(len(x), int(len(x) * STABILITY_FRAC), replace=False)
            sub = fit_kmeans(x[idx], k, seed + 1 + run)
            aris.append(adjusted_rand_score(labels[idx], sub.labels_))
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(x, labels),
            "calinski_harabasz": calinski_harabasz_score(x, labels),
            "ari_mean": float(np.mean(aris)),
            "ari_std": float(np.std(aris)),
        })
        print(f"  k={k:2d}  silhouette {rows[-1]['silhouette']:.3f}  "
              f"ARI {rows[-1]['ari_mean']:.3f} +/- {rows[-1]['ari_std']:.3f}  "
              f"inertia {km.inertia_:,.1f}")
    return pd.DataFrame(rows)


def cmd_cluster(args):
    out_path = clusters_path(args.shape_only)
    if os.path.exists(out_path) and not args.force:
        print(f"{out_path} exists, use --force to rebuild")
        return

    df = load_profiles()
    x = profile_matrix(df, args.shape_only)

    decomp = variance_decomposition(profile_matrix(df, shape_only=False))
    print("profile variance across blocks: "
          f"level {decomp['level_share'] * 100:.1f}%, shape {decomp['shape_share'] * 100:.1f}%, "
          f"level/shape covariance {decomp['cross_share'] * 100:+.1f}%")

    fit_mask = (df["n"] >= args.min_block_pixels).to_numpy()
    print(f"fitting on {fit_mask.sum()} blocks, assigning the other {(~fit_mask).sum()}")

    print(f"k sweep ({'shape only' if args.shape_only else 'full profile'}):")
    sw = sweep(x[fit_mask], args.seed)
    sw.to_csv(sweep_path(args.shape_only), index=False)
    print(f"wrote {sweep_path(args.shape_only)}")

    km = fit_kmeans(x[fit_mask], args.k, args.seed)

    # The question the whole analysis has to answer: is this clustering
    # anything more than binning blocks on their median distance? Cluster the
    # levels alone at the same k and see how much of the partition that
    # reproduces. An ARI near 1 means the 39 knot profile bought nothing.
    level_only = np.median(profile_matrix(df, shape_only=False), axis=1)[fit_mask].reshape(-1, 1)
    ari_level = adjusted_rand_score(km.labels_, fit_kmeans(level_only, args.k, args.seed).labels_)
    print(f"ARI against clustering on median distance alone: {ari_level:.3f}")

    # Relabel by centroid median distance so cluster 0 is always the regime
    # closest to development. Without this the numbering is whatever k-means
    # init happened to produce and every figure's legend shuffles between runs.
    order = np.argsort(np.median(km.cluster_centers_, axis=1))
    remap = np.empty(args.k, dtype=int)
    remap[order] = np.arange(args.k)
    centers = km.cluster_centers_[order]

    full = profile_matrix(df, shape_only=False)
    d = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
    out = pd.DataFrame({
        "block_id": df["block_id"],
        "cluster": d.argmin(axis=1).astype(np.int8),
        "is_small": ~fit_mask,
        "dist_to_centroid": d.min(axis=1),
        "level": np.median(full, axis=1),
        "spread": np.log10(df["q0750"].to_numpy()) - np.log10(df["q0250"].to_numpy()),
    })
    # The fitted blocks keep k-means' own assignment; recomputing nearest
    # centroid should agree but there is no reason to let a tie flip one.
    out.loc[fit_mask, "cluster"] = remap[km.labels_].astype(np.int8)

    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path} (k={args.k})")
    print(out.groupby("cluster").size().rename("blocks").to_string())


def cmd_report(args):
    df = load_profiles().merge(pd.read_parquet(clusters_path(args.shape_only)), on="block_id")

    rep = df.groupby("cluster").agg(
        blocks=("block_id", "size"),
        small=("is_small", "sum"),
        rows=("n", "sum"),
        n1=("n1", "sum"),
        n2=("n2", "sum"),
    ).reset_index()
    rep["pct_label1"] = rep["n1"] / rep["rows"] * 100
    rep["pct_label2"] = rep["n2"] / rep["rows"] * 100
    # Pixel weighted so a cluster's quantiles describe its pixels, not its
    # blocks: block size ranges over five orders of magnitude here.
    for q in QUANTILES:
        col = f"q{int(round(q * 1000)):04d}"
        w = df[col] * df["n"]
        rep[f"p{int(q * 100)}_m"] = (w.groupby(df["cluster"]).sum() / rep.set_index("cluster")["rows"]).to_numpy()

    print(rep.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    if args.dataset_name:
        folds = pd.read_parquet(os.path.join(DATASETS_DIR, args.dataset_name, "block_folds.parquet"))
        df = df.merge(folds[["block_id", "fold"]], on="block_id")
        print(f"\nblocks per cluster per fold ({args.dataset_name}):")
        print(pd.crosstab(df["cluster"], df["fold"]).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profiles", help="aggregate per block distance quantile functions")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_profiles)

    p = sub.add_parser("cluster", help="k sweep and k-means over the profiles")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--min-block-pixels", type=int, default=MIN_BLOCK_PIXELS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--shape-only", action="store_true",
                   help="subtract each block's own median first, clustering shape without level")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_cluster)

    p = sub.add_parser("report", help="per cluster counts, conversion rates and quantiles")
    p.add_argument("--shape-only", action="store_true")
    p.add_argument("--dataset-name", default=None, help="crosstab clusters against this dataset's folds")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
