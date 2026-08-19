"""Fit and persist gradient boosting models for a named experiment (see
scripts/experiments.py), and screen hyperparameters against one held out
fold. Everything an experiment produces lives under
results/models/<experiment_name>/, and a fold's model, once fit, is never
implicitly retrained: baseline writes one joblib file per fold, plus the out
of fold predictions every downstream analysis needs, so
model_analysis.ipynb only ever loads from disk.

  baseline
      One LightGBM classifier per fold, default hyperparameters unless an
      experiment overrides model_params. Each fold trains on that fold's
      weighted training pool and gets scored, streamed, against its own
      natural, unweighted held out fold, so the reported metrics reflect the
      real base rate, not the oversampled one. Writes, per fold,
      model_fold{k}.joblib and one row of oof_predictions.parquet (label,
      raw score, calibrated score, distance to development), plus
      fold_summary.csv, roc_by_fold.png, pr_by_fold.png, and config.json
      across all folds.

  grid-search
      Screens an experiment's grid (scripts/experiments.py) against a single
      fold pair: fold 0 held out as the screening test, fold 1 as an early
      stopping validation set pulled from the natural distribution, the rest
      as training data. A full grid across a full k fold CV would be far too
      slow, this narrows the search, it does not by itself justify
      generalization: retrain the winning combination with `baseline` once a
      promising region is found. Resumable by default: a run_NNN directory
      with a metrics.csv already in it is skipped, so a long sweep can be
      restarted after an interruption without redoing finished work. Pass
      --force to redo every run anyway.

  status
      Prints what is already on disk for an experiment: whether baseline has
      run and how many fold models it left behind, and how many grid runs
      are complete.

Usage:
    python scripts/gb_train.py baseline --experiment NAME [--force]
    python scripts/gb_train.py grid-search --experiment NAME [--force]
    python scripts/gb_train.py status --experiment NAME
"""

import argparse
import itertools
import json
import os
from dataclasses import asdict

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import experiments
import gb_common


def make_model(cfg: experiments.ExperimentConfig, **overrides):
    """A fresh, untrained model for this experiment. Only lightgbm is wired
    up; a second model_type is a new branch here, not a restructure."""
    params = {**cfg.model_params, **overrides}
    if cfg.model_type == "lightgbm":
        return lgb.LGBMClassifier(**params)
    raise ValueError(f"unknown model_type '{cfg.model_type}'")


def _binary_metrics(y_test: np.ndarray, proba: np.ndarray, pos_weight_mult: float) -> tuple[dict, dict]:
    """Rank based metrics and the best F1 operating point, off the raw score.
    Both calibration corrections are monotone, so this would give identical
    numbers off either one, and the raw score is what the model emits."""
    avg_prec = average_precision_score(y_test, proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, proba)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    best_threshold, best_p, best_r, best_f1 = gb_common.best_f1_operating_point(precision, recall, pr_thresholds)
    accuracy = accuracy_score(y_test, proba >= best_threshold)

    metrics = {
        "avg_precision": avg_prec, "pr_auc": pr_auc, "accuracy": accuracy, "roc_auc": roc_auc,
        "precision": best_p, "recall": best_r, "f1": best_f1,
        # calibrated, so this threshold means the same thing wherever it's read back
        "best_threshold": float(gb_common.calibrate(np.array([best_threshold]), pos_weight_mult)[0]),
    }
    curves = {"fpr": fpr, "tpr": tpr, "recall_curve": recall, "precision_curve": precision}
    return metrics, curves


def cmd_baseline(args):
    cfg = experiments.get(args.experiment)
    baseline_dir = os.path.join(cfg.results_dir, "baseline")
    if os.path.exists(baseline_dir) and not args.force:
        print(f"{baseline_dir} exists, use --force to rebuild")
        return
    os.makedirs(baseline_dir, exist_ok=True)

    dcfg = gb_common.load_dataset_config(cfg.dataset_name)
    pos_weight_mult = dcfg["pos_weight_mult"]
    pos_row_dup = dcfg["pos_row_dup"]
    k_folds = dcfg.get("k_folds", gb_common.K_FOLDS)

    con, _ = gb_common.connect_samples(cfg.dataset_name)
    train_pool = gb_common.load_train_pool(cfg.dataset_name)
    model_factory = lambda: make_model(cfg)  # noqa: E731, reused per fold by iter_folds' calibrator

    fold_metrics = []
    roc_curves, pr_curves = {}, {}
    oof_frames = []

    for fold_id, X_train, y_train, w_train, _predict_test, make_calibrator in gb_common.iter_folds(
        con, train_pool, pos_weight_mult, pos_row_dup, k=k_folds,
    ):
        model = make_model(cfg)
        model.fit(X_train[cfg.feature_cols], y_train, sample_weight=w_train)
        joblib.dump(model, os.path.join(baseline_dir, f"model_fold{fold_id}.joblib"))

        # one streamed pass over the held out fold covers both the metrics
        # below and the decile analysis model_analysis.ipynb does later, so
        # oof_predictions.parquet carries distance too and that notebook
        # never has to re-score the fold from scratch
        log_mean, log_std = gb_common.fit_dist_normalizer(con, fold_id)
        y_test, proba, dist = gb_common.predict_fold_with_dist(
            con, model, fold_id, log_mean, log_std, cfg.feature_cols)
        p_cal = gb_common.calibrate(proba, pos_weight_mult)
        p_iso = make_calibrator(model_factory, cfg.feature_cols).predict(proba)

        metrics, curves = _binary_metrics(y_test, proba, pos_weight_mult)
        pred_acres = float(p_cal.sum() * gb_common.PIXEL_ACRES)
        actual_acres = float(y_test.sum() * gb_common.PIXEL_ACRES)
        fold_metrics.append({
            "fold": fold_id, **metrics,
            "prevalence": float(y_test.mean()),
            "brier": float(np.mean((p_cal - y_test) ** 2)),
            "pred_acres": pred_acres, "actual_acres": actual_acres,
            "acres_error_pct": 100 * (pred_acres / actual_acres - 1),
            "acres_error_isotonic_pct": 100 * (p_iso.sum() / y_test.sum() - 1),
        })
        roc_curves[fold_id] = (curves["fpr"], curves["tpr"])
        pr_curves[fold_id] = (curves["recall_curve"], curves["precision_curve"])
        oof_frames.append(pd.DataFrame({
            "fold": np.full(len(y_test), fold_id, dtype=np.int8),
            "label": y_test.astype(np.int8),
            gb_common.DIST_COL: dist.astype(np.float32),
            "y_score": proba.astype(np.float32),
            "p_calibrated": p_cal.astype(np.float32),
        }))

        print(f"fold {fold_id}: AP={metrics['avg_precision']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
              f"ROC-AUC={metrics['roc_auc']:.4f}  acres {pred_acres:,.0f} predicted vs "
              f"{actual_acres:,.0f} actual ({100 * (pred_acres / actual_acres - 1):+.1f}%)")
        del model, X_train, y_train, w_train, y_test, proba, dist, p_cal, p_iso

    fold_summary = pd.DataFrame(fold_metrics).set_index("fold")[gb_common.SUMMARY_COLS + gb_common.CALIB_COLS]
    fold_summary = pd.concat([fold_summary, fold_summary.agg(["mean", "std"])])
    fold_summary.to_csv(os.path.join(baseline_dir, "fold_summary.csv"))
    print(fold_summary.to_string(float_format=lambda v: f"{v:,.4f}"))

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold_id, (fpr, tpr) in roc_curves.items():
        ax.plot(fpr, tpr, label=f"fold {fold_id} (AUC {fold_summary.loc[fold_id, 'roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="no skill")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC by fold: {args.experiment}")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(baseline_dir, "roc_by_fold.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold_id, (recall, precision) in pr_curves.items():
        ax.plot(recall, precision, label=f"fold {fold_id} (AP {fold_summary.loc[fold_id, 'avg_precision']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision and recall by fold: {args.experiment}")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(baseline_dir, "pr_by_fold.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.concat(oof_frames, ignore_index=True).to_parquet(
        os.path.join(baseline_dir, "oof_predictions.parquet"), index=False)

    with open(os.path.join(baseline_dir, "config.json"), "w") as f:
        json.dump({**asdict(cfg), "k_folds": k_folds}, f, indent=2, sort_keys=True)
    print(f"wrote {baseline_dir}")


def _save_run_plots(run_dir: str, run_id: int, screen_fold: int, curves: dict,
                    avg_prec: float, roc_auc: float, title_suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(curves["fpr"], curves["tpr"], label=f"run {run_id:03d} (AUC {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="no skill")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC, run {run_id:03d}, fold {screen_fold}{title_suffix}")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(run_dir, "roc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(curves["recall_curve"], curves["precision_curve"], label=f"run {run_id:03d} (AP {avg_prec:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision and recall, run {run_id:03d}, fold {screen_fold}{title_suffix}")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(run_dir, "pr.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _read_best_iteration(run_dir: str) -> int | None:
    params_path = os.path.join(run_dir, "params.txt")
    if not os.path.exists(params_path):
        return None
    for line in open(params_path):
        if line.startswith("best_iteration="):
            return int(line.strip().split("=")[1])
    return None


def cmd_grid_search(args):
    cfg = experiments.get(args.experiment)
    if cfg.grid is None:
        raise SystemExit(f"experiment '{args.experiment}' has no grid configured in scripts/experiments.py")

    grid_dir = os.path.join(cfg.results_dir, "grid_search")
    os.makedirs(grid_dir, exist_ok=True)

    dcfg = gb_common.load_dataset_config(cfg.dataset_name)
    pos_weight_mult = dcfg["pos_weight_mult"]
    k_folds = dcfg.get("k_folds", gb_common.K_FOLDS)

    con, _ = gb_common.connect_samples(cfg.dataset_name)
    train_pool = gb_common.load_train_pool(cfg.dataset_name)

    # fold 0 held out as the screening test, fold 1 as an early stopping
    # validation set, the rest as training data
    screen_fold = 0
    val_fold = (screen_fold + 1) % k_folds
    train_folds = [f for f in range(k_folds) if f not in (screen_fold, val_fold)]
    val_rows = 1_000_000

    log_mean, log_std = gb_common.fit_dist_normalizer_excluding(con, [screen_fold, val_fold])
    fit_frame = train_pool[train_pool["fold"].isin(train_folds)].pipe(gb_common.add_normalized_dist, log_mean, log_std)
    X_fit = fit_frame[cfg.feature_cols]
    y_fit = fit_frame[gb_common.TARGET_COL]
    w_fit = fit_frame["sample_weight"] / fit_frame["sample_weight"].mean()

    val_frame = gb_common.natural_fold_sample(con, val_fold, val_rows, log_mean, log_std)
    X_val = val_frame[cfg.feature_cols]
    y_val = val_frame[gb_common.TARGET_COL]

    grid_keys = list(cfg.grid.keys())
    grid_combos = list(itertools.product(*cfg.grid.values()))
    random_state = cfg.model_params.get("random_state", 0)

    summary_path = os.path.join(grid_dir, "grid_search_summary.csv")
    grid_summary_rows = []

    for run_id, combo in enumerate(grid_combos):
        params = dict(zip(grid_keys, combo))
        run_dir = os.path.join(grid_dir, f"run_{run_id:03d}")
        os.makedirs(run_dir, exist_ok=True)
        metrics_path = os.path.join(run_dir, "metrics.csv")

        if os.path.exists(metrics_path) and not args.force:
            row = pd.read_csv(metrics_path).iloc[0].to_dict()
            grid_summary_rows.append({"run_id": run_id, **params, "best_iteration": _read_best_iteration(run_dir), **row})
            print(f"run {run_id:03d}: already done, skipping")
            pd.DataFrame(grid_summary_rows).to_csv(summary_path, index=False)
            continue

        try:
            model = lgb.LGBMClassifier(
                random_state=random_state, verbosity=-1,
                n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
                num_leaves=params["num_leaves"], feature_fraction=params["feature_fraction"],
            )
            model.fit(
                X_fit, y_fit, sample_weight=w_fit,
                eval_X=X_val, eval_y=y_val,
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(
                    stopping_rounds=params["early_stopping_rounds"], first_metric_only=True, verbose=False)],
            )

            y_test, proba = gb_common.predict_fold(con, model, screen_fold, log_mean, log_std, cfg.feature_cols)
            metrics, curves = _binary_metrics(y_test, proba, pos_weight_mult)

            with open(os.path.join(run_dir, "params.txt"), "w") as f:
                for k, v in params.items():
                    f.write(f"{k}={v}\n")
                f.write(f"random_state={random_state}\n")
                f.write(f"screen_fold={screen_fold}\n")
                f.write(f"val_fold={val_fold}\n")
                f.write(f"train_folds={','.join(map(str, train_folds))}\n")
                f.write(f"best_iteration={model.best_iteration_}\n")

            pd.DataFrame([metrics])[gb_common.SUMMARY_COLS].to_csv(metrics_path, index=False)
            _save_run_plots(run_dir, run_id, screen_fold, curves, metrics["avg_precision"], metrics["roc_auc"])

            grid_summary_rows.append({"run_id": run_id, **params, "best_iteration": model.best_iteration_, **metrics})
            print(f"run {run_id:03d}: AP={metrics['avg_precision']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
                  f"ROC-AUC={metrics['roc_auc']:.3f}  trees={model.best_iteration_}")
            del model, proba
        except Exception as e:
            with open(os.path.join(run_dir, "error.txt"), "w") as f:
                f.write(str(e))
            print(f"run {run_id:03d}: FAILED, {e}")
            grid_summary_rows.append({"run_id": run_id, **params, "best_iteration": None, "error": str(e)})

        pd.DataFrame(grid_summary_rows).to_csv(summary_path, index=False)

    summary = pd.DataFrame(grid_summary_rows)
    if "avg_precision" in summary.columns:
        summary = summary.sort_values("avg_precision", ascending=False)
    print(summary.head(10).to_string(index=False))


def cmd_status(args):
    cfg = experiments.get(args.experiment)
    baseline_dir = os.path.join(cfg.results_dir, "baseline")
    grid_dir = os.path.join(cfg.results_dir, "grid_search")

    print(f"experiment '{args.experiment}' -> {cfg.results_dir}")
    print(f"  dataset: {cfg.dataset_name}")
    print(f"  feature_cols: {len(cfg.feature_cols)} columns")

    if os.path.exists(os.path.join(baseline_dir, "fold_summary.csv")):
        n_models = sum(
            os.path.exists(os.path.join(baseline_dir, f"model_fold{k}.joblib"))
            for k in range(gb_common.K_FOLDS)
        )
        print(f"  baseline: done ({n_models}/{gb_common.K_FOLDS} fold models on disk)")
    else:
        print("  baseline: not run")

    if cfg.grid is None:
        print("  grid_search: not configured for this experiment")
    elif os.path.exists(grid_dir):
        n_combos = len(list(itertools.product(*cfg.grid.values())))
        n_done = sum(
            os.path.exists(os.path.join(grid_dir, f"run_{i:03d}", "metrics.csv"))
            for i in range(n_combos)
        )
        print(f"  grid_search: {n_done}/{n_combos} runs complete")
    else:
        print("  grid_search: not run")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_b = sub.add_parser("baseline")
    p_b.add_argument("--experiment", required=True)
    p_b.add_argument("--force", action="store_true")
    p_b.set_defaults(func=cmd_baseline)

    p_g = sub.add_parser("grid-search")
    p_g.add_argument("--experiment", required=True)
    p_g.add_argument("--force", action="store_true")
    p_g.set_defaults(func=cmd_grid_search)

    p_s = sub.add_parser("status")
    p_s.add_argument("--experiment", required=True)
    p_s.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
