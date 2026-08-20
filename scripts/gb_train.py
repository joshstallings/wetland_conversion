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

  grid-finalize
      Refits one grid-search run's winning combination (params.txt in its
      run_NNN directory) on the exact same fit frame it was screened on,
      writes model.joblib into that run directory (skipped if one is already
      there, pass --force to redo), then writes run_NNN/model_analysis/:
      precision/recall/accuracy and a PR curve against its own training
      data, plus the same three metrics streamed against every fold's
      natural population. Folds 2/3/4 fed the fit (through a subsampled,
      weighted slice, not this full natural population) and fold 1 set
      early stopping, so only fold 0 is a genuine generalization check;
      still worth running `baseline` on the winning combo for a real CV
      estimate.

  status
      Prints what is already on disk for an experiment: whether baseline has
      run and how many fold models it left behind, and how many grid runs
      are complete.

Usage:
    python scripts/gb_train.py baseline --experiment NAME [--force]
    python scripts/gb_train.py grid-search --experiment NAME [--force]
    python scripts/gb_train.py grid-finalize --experiment NAME [--run run_NNN]
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
    precision_score,
    recall_score,
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


def grid_split(cfg: experiments.ExperimentConfig, con, train_pool: pd.DataFrame, k_folds: int):
    """The fold split grid-search screens against: fold 0 held out as the
    screening test, fold 1 as an early stopping validation set pulled from
    the natural distribution, the rest as training data. Factored out of
    cmd_grid_search so grid-finalize can rebuild the exact same training
    frame a winning run was fit on, rather than a fresh, differently shuffled
    one.

    Returns (X_fit, y_fit, w_fit, X_val, y_val, screen_fold, val_fold,
    train_folds, log_mean, log_std).
    """
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

    return X_fit, y_fit, w_fit, X_val, y_val, screen_fold, val_fold, train_folds, log_mean, log_std


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

    X_fit, y_fit, w_fit, X_val, y_val, screen_fold, val_fold, train_folds, log_mean, log_std = grid_split(
        cfg, con, train_pool, k_folds)

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
            # early_stopping_rounds drives the callback below, not a
            # LGBMClassifier constructor arg, so it's the one grid key held
            # back; everything else in the combo is passed straight through,
            # which is what lets the grid sweep params like min_child_samples
            # or boosting_type without a new hardcoded line here each time
            model_kwargs = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
            model = lgb.LGBMClassifier(random_state=random_state, verbosity=-1, **model_kwargs)
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


def _read_params_txt(run_dir: str) -> dict:
    """params.txt as cmd_grid_search writes it: every grid key plus a
    handful of run metadata (random_state, screen_fold, val_fold,
    train_folds, best_iteration), one `key=value` per line, values still
    strings. grid-finalize sorts those back into fit kwargs vs metadata."""
    params = {}
    for line in open(os.path.join(run_dir, "params.txt")):
        k, v = line.strip().split("=", 1)
        params[k] = v
    return params


def _coerce_grid_value(v: str):
    """params.txt only ever holds what an experiment's grid dict can hold:
    ints, floats, or plain strings like boosting_type's 'gbdt'/'goss'. A
    float parse that comes out integral and had no decimal point in the
    original text round trips ints; anything else that fails to parse is a
    string param, passed through as-is."""
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v else f
    except ValueError:
        return v


def cmd_grid_finalize(args):
    cfg = experiments.get(args.experiment)
    grid_dir = os.path.join(cfg.results_dir, "grid_search")

    run_id = args.run
    if run_id is None:
        summary = pd.read_csv(os.path.join(grid_dir, "grid_search_summary.csv"))
        run_id = f"run_{int(summary.sort_values('avg_precision', ascending=False).iloc[0]['run_id']):03d}"
    run_dir = os.path.join(grid_dir, run_id)
    if not os.path.exists(run_dir):
        raise SystemExit(f"{run_dir} not found")

    dcfg = gb_common.load_dataset_config(cfg.dataset_name)
    k_folds = dcfg.get("k_folds", gb_common.K_FOLDS)
    con, _ = gb_common.connect_samples(cfg.dataset_name)
    train_pool = gb_common.load_train_pool(cfg.dataset_name)

    # rebuild the exact frame this run was screened on, so a refit (if one
    # is needed below) reproduces it, and so the natural fold scoring further
    # down normalizes distance the same way training did
    X_fit, y_fit, w_fit, X_val, y_val, screen_fold, val_fold, train_folds, log_mean, log_std = grid_split(
        cfg, con, train_pool, k_folds)

    model_path = os.path.join(run_dir, "model.joblib")
    if os.path.exists(model_path) and not args.force:
        model = joblib.load(model_path)
        print(f"{model_path} already on disk, loading rather than refitting (--force to redo)")
    else:
        raw = _read_params_txt(run_dir)
        meta_keys = {"random_state", "screen_fold", "val_fold", "train_folds", "best_iteration"}
        params = {k: _coerce_grid_value(v) for k, v in raw.items() if k not in meta_keys}
        random_state = int(raw["random_state"])

        model_kwargs = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        model = lgb.LGBMClassifier(random_state=random_state, verbosity=-1, **model_kwargs)
        model.fit(
            X_fit, y_fit, sample_weight=w_fit,
            eval_X=X_val, eval_y=y_val,
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(
                stopping_rounds=params["early_stopping_rounds"], first_metric_only=True, verbose=False)],
        )
        recorded_best_iter = int(raw["best_iteration"])
        if model.best_iteration_ != recorded_best_iter:
            print(f"warning: refit best_iteration {model.best_iteration_} != recorded {recorded_best_iter}, "
                  f"the grid search screen wasn't exactly reproducible on this refit")
        joblib.dump(model, model_path)
        print(f"wrote {model_path}")

    analysis_dir = os.path.join(run_dir, "model_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # own training data, i.e. the weighted, duplicated fit frame itself, not
    # a held out fold - this is a memorization check, not a generalization
    # one, and the model's implicit 0.5 decision boundary is on that same
    # heavily upweighted/duplicated pool, not the natural class balance
    proba_fit = model.predict_proba(X_fit)[:, 1]
    pred_fit = (proba_fit >= 0.5).astype(int)
    train_metrics = pd.DataFrame([{
        "n_rows": len(y_fit),
        "n_positive": int(y_fit.sum()),
        "precision": precision_score(y_fit, pred_fit),
        "recall": recall_score(y_fit, pred_fit),
        "accuracy": accuracy_score(y_fit, pred_fit),
    }])
    metrics_path = os.path.join(analysis_dir, "train_metrics.csv")
    train_metrics.to_csv(metrics_path, index=False)
    print(train_metrics.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"wrote {metrics_path}")

    precision_curve, recall_curve, _ = precision_recall_curve(y_fit, proba_fit)
    pr_auc = auc(recall_curve, precision_curve)
    avg_prec = average_precision_score(y_fit, proba_fit)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall_curve, precision_curve, color="#0072B2", label=f"AP {avg_prec:.3f}, PR-AUC {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision and recall on training data: {args.experiment} {run_id}")
    ax.legend(loc="upper right")
    pr_plot_path = os.path.join(analysis_dir, "train_pr_curve.png")
    fig.savefig(pr_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pr_plot_path}")

    # every fold's natural, unweighted population, same rows baseline scores
    # against. Folds 2/3/4 fed the fit (through a subsampled, weighted slice,
    # not this full natural population) and fold 1 set early stopping, so
    # only fold 0 is genuinely held out; the rest are a fit check on data
    # partly or fully seen, not a fair read on generalization
    role_by_fold = {screen_fold: "test (held out)", val_fold: "val (early stopping)"}
    role_by_fold.update({f: "train" for f in train_folds})
    fold_rows = []
    fold_pr_curves = {}
    for fold_id in range(k_folds):
        y_true, y_score = gb_common.predict_fold(con, model, fold_id, log_mean, log_std, cfg.feature_cols)
        pred = (y_score >= 0.5).astype(int)
        row = {
            "fold": fold_id,
            "role": role_by_fold[fold_id],
            "n_rows": len(y_true),
            "n_positive": int(y_true.sum()),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "accuracy": accuracy_score(y_true, pred),
        }
        fold_rows.append(row)
        print(f"fold {fold_id} ({row['role']}): precision={row['precision']:.4f}  "
              f"recall={row['recall']:.4f}  accuracy={row['accuracy']:.4f}")

        # curve needs the full score, not the 0.5 cutoff above, so it's
        # built off the same y_true/y_score rather than the row's hard preds
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_score)
        fold_pr_curves[fold_id] = (r_curve, p_curve, average_precision_score(y_true, y_score))
        del y_true, y_score, pred

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics_path = os.path.join(analysis_dir, "fold_metrics.csv")
    fold_metrics.to_csv(fold_metrics_path, index=False)
    print(f"wrote {fold_metrics_path}")

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold_id, (r_curve, p_curve, avg_prec_fold) in fold_pr_curves.items():
        ax.plot(r_curve, p_curve, color=gb_common.FOLD_COLORS[fold_id],
                label=f"fold {fold_id} ({role_by_fold[fold_id]}, AP {avg_prec_fold:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision and recall by fold, natural data: {args.experiment} {run_id}")
    ax.legend(loc="upper right", fontsize="small")
    fold_pr_path = os.path.join(analysis_dir, "fold_pr_curve.png")
    fig.savefig(fold_pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fold_pr_path}")


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

    p_gf = sub.add_parser("grid-finalize")
    p_gf.add_argument("--experiment", required=True)
    p_gf.add_argument("--run", default=None,
                      help="run_NNN under that experiment's grid_search/; default is whichever run "
                           "has the highest avg_precision in grid_search_summary.csv")
    p_gf.add_argument("--force", action="store_true", help="refit even if model.joblib is already on disk")
    p_gf.set_defaults(func=cmd_grid_finalize)

    p_s = sub.add_parser("status")
    p_s.add_argument("--experiment", required=True)
    p_s.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
