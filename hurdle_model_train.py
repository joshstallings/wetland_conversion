"""
Both stages of the hurdle model, spatially blocked CV over a build_cells.py
artifact:

    python hurdle_model_train.py classify
    python hurdle_model_train.py regress

classify is stage one: a cell classifier (MODEL_CLS, a models.Cell*Classifier)
for whether a cell has any 2019 wetland developed by 2024 (n_developed > 0).
Conversion to other, not developed, land counts as negative.

regress is stage two: models.CellTemporalConvolutionalLogShare on the natural
log of n_developed / n_wetland, trained and scored only on cells with
n_developed > 0. Its fold summary carries a constant and an OLS on the three
cell scalars as baselines. Nearly half of converting cells have exactly one
developed pixel, so their target is just -log(n_wetland), and the OLS is what
shows whether the embeddings add anything past cell size and distance.

Run order from scratch is the same as regress_train.py:
    python population_stats.py
    python build_arrays.py
    python build_cells.py --cell-size-m 300
    python hurdle_model_train.py classify
    python hurdle_model_train.py regress

Folds are the regress_train.py folds, same assign_folds call, same SEED and
N_SPLITS, for both stages. That is what lets each fold's val AUPRC sit next to
the regressor's "any conversion" AUPRC on the exact same cells, and what lets
the two stages be combined per cell on cell_idx. REGRESSOR_DIR names the run to compare against;
its val cell_idx is checked against ours fold by fold before anything trains.

Final epoch weights are scored, no early stopping, for the same reason as the
regressor: picking the epoch off val_loss lets the val fold choose the model.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import average_precision_score

import arrays as array_io
import datasets
import models
import reporting
from features import YEARS_2017_2019
from population_stats import load_population_stats
from folds import assign_folds

CELL_DIR = "data/cells_2019_2024_300m"
POPULATION_STATS_PATH = "data/population_stats_2019_2024.json"
N_YEARS = len(YEARS_2017_2019)

# CellTemporalConvolutionalClassifier or CellMultiLayerPerceptronClassifier. Both
# take the same batches, so nothing else in this file changes between them.
MODEL_CLS = models.CellMultiLayerPerceptronClassifier

RESULTS_DIR = Path("results/hurdle_clf_mlp_300m")

# The regressor run whose any conversion AUPRC this has to beat. Best of the 300 m
# runs at about 0.27 mean val AUPRC. None skips the comparison.
REGRESSOR_DIR = Path("results/cell_tcn_300m_binomial_loss_unweighted")

SEED = 0
N_SPLITS = 5

MAX_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 0.0

# Focal loss is the default. GAMMA=None falls back to nn.BCEWithLogitsLoss.
GAMMA = 2.0
ALPHA = None  # None to disable alpha. Must be None when GAMMA is None.

# No pos_weight, same reasoning as classify_train.py. If POS_PER_BATCH is set,
# data_module.train_sampler.batch_pos_weight is the value an ablation should pass.
POS_WEIGHT = None

BATCH_SIZE = 1024
# None trains at each fold's natural rate, 2.7% to 3.5% of cells with any
# conversion, about 28 to 36 per 1,024 batch. Set an int to stratify. Moving it
# off None moves the sigmoid off the natural prior, which matters once this
# feeds the hurdle product, not for AUPRC.
POS_PER_BATCH = None

# Only used with POS_PER_BATCH set. None is one pass over the fold's train
# positives per epoch, about 450 batches at 64 per batch.
STEPS_PER_EPOCH = None

VAL_BATCH_SIZE = 8192

# 1 keeps every train cell. Val always keeps every cell regardless.
MIN_WETLAND_TRAIN = 1

# The confusion matrix threshold is the one that reaches this recall on the
# fold's train cells, then applied unchanged to val.
RECALL_TARGET = 0.5

NUM_WORKERS = 0

LOSS_NAME = "focal loss" if GAMMA is not None else "BCE loss"
CLASS_LABELS = ("no conversion", "any conversion")

# Stage two. MIN_WETLAND_TRAIN, VAL_BATCH_SIZE and NUM_WORKERS above are shared.
REG_RESULTS_DIR = Path("results/hurdle_reg_tcn_300m")

# A stage one run to check val cell_idx against fold by fold, so the two can be
# joined later. None skips it. Every hurdle_clf run so far was trained on cells
# built 2026-10-01, before the current artifact, so pointing at one fails until
# stage one is rerun.
REG_CLASSIFIER_DIR = None

# About 29k converting train cells per fold, so 1024 would be only 28 steps an epoch.
REG_BATCH_SIZE = 256
REG_MAX_EPOCHS = 30
REG_LR = 1e-3
REG_WEIGHT_DECAY = 1e-4
REG_LOSS_NAME = "MSE on log share"


def load_regressor_fold(fold_idx, val_idx):
    """The regressor's val predictions for this fold, after checking they cover
    the same cells in the same order. A mismatch means the two runs did not
    share folds, and the comparison would be meaningless."""
    z = np.load(REGRESSOR_DIR / f"fold_{fold_idx}" / "val_preds.npz")
    if not np.array_equal(z["cell_idx"], val_idx):
        raise AssertionError(
            f"{REGRESSOR_DIR} fold {fold_idx} val cells differ from this run's. "
            f"Check SEED, N_SPLITS and CELL_DIR against its manifest."
        )
    return z["pred"]


def run_classify_fold(fold_idx, data_module, fold_dir, reg_pred=None):
    """
    Trains one fold for MAX_EPOCHS, writes its plots and predictions into
    fold_dir. Returns (fold score dict, list of n_wetland bin rows).
    """
    model = MODEL_CLS(
        n_years=N_YEARS, lr=LR, weight_decay=WEIGHT_DECAY,
        gamma=GAMMA, alpha=ALPHA, pos_weight=POS_WEIGHT,
    )

    fold_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=str(fold_dir.parent), name=f"fold_{fold_idx}")
    # monitor=None keeps only the most recent epoch, which after fit is the final one.
    checkpoint = ModelCheckpoint(monitor=None, save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint],
        logger=logger,
        enable_progress_bar=True,
        accelerator="auto",
    )
    trainer.fit(model, datamodule=data_module)
    epochs_run = trainer.current_epoch

    final_model = MODEL_CLS.load_from_checkpoint(
        checkpoint.best_model_path
    )

    prob, target, n_wet = reporting.get_cell_predictions(final_model, data_module.val_dataloader())
    labels = target > 0
    # target and cell_idx ride along so stage two can join on them later.
    np.savez(fold_dir / "val_preds.npz", prob=prob, any=labels, target=target,
             n_wetland=n_wet, cell_idx=data_module.val_idx)

    train_prob, train_target, train_n_wet = reporting.get_cell_predictions(
        final_model, data_module.train_eval_dataloader()
    )
    train_labels = train_target > 0
    np.savez(fold_dir / "train_preds.npz", prob=train_prob, any=train_labels,
             target=train_target, n_wetland=train_n_wet, cell_idx=data_module.train_idx)

    metrics_csvs = sorted(fold_dir.glob("**/metrics.csv"))
    if metrics_csvs:
        reporting.plot_loss_curve(
            metrics_csvs[-1], fold_dir / "loss_curve.png",
            f"Fold {fold_idx}: training and validation loss", ylabel=LOSS_NAME,
        )

    threshold = reporting.threshold_at_recall(train_labels, train_prob, RECALL_TARGET)
    reporting.plot_confusion_matrix(
        labels, prob, threshold=threshold, out_path=fold_dir / "confusion_matrix.png",
        title=f"Fold {fold_idx}: any conversion per cell, cut at {RECALL_TARGET:.0%} train recall",
        display_labels=CLASS_LABELS,
    )
    reporting.plot_pr_curve_train_val(
        train_labels, train_prob, labels, prob, out_path=fold_dir / "pr_curve.png",
        title=f"Fold {fold_idx}: precision recall, any conversion in the cell",
    )

    fold_score = reporting.score_fold(labels, prob, threshold=threshold)
    train_score = reporting.score_fold(train_labels, train_prob, threshold=threshold)
    fold_score.update({f"train_{k}": v for k, v in train_score.items()})
    fold_score[f"precision_at_recall_{RECALL_TARGET}"] = reporting.precision_at_recall(
        labels, prob, RECALL_TARGET
    )

    bin_rows = [{"fold": fold_idx, "model": "classifier", **r}
                for r in reporting.auprc_by_n_wetland(labels, prob, n_wet)]
    if reg_pred is not None:
        fold_score["regressor_auprc"] = float(average_precision_score(labels, reg_pred))
        fold_score[f"regressor_precision_at_recall_{RECALL_TARGET}"] = reporting.precision_at_recall(
            labels, reg_pred, RECALL_TARGET
        )
        bin_rows += [{"fold": fold_idx, "model": "regressor", **r}
                     for r in reporting.auprc_by_n_wetland(labels, reg_pred, n_wet)]

    fold_score.update({
        "fold": fold_idx,
        "threshold": threshold,
        "epochs_run": epochs_run,
        "checkpoint_path": checkpoint.best_model_path,
        "n_train_cells": int(data_module.train_idx.size),
        "n_val_cells": int(data_module.val_idx.size),
        "n_val_positive_cells": int(labels.sum()),
        "batch_positive_rate": data_module.batch_positive_rate,
        "gamma": GAMMA,
        "alpha": ALPHA,
        "pos_weight": POS_WEIGHT,
        "scalar_mean": data_module.scalar_mean.tolist(),
        "scalar_std": data_module.scalar_std.tolist(),
    })
    return fold_score, bin_rows


def load_checked_cells():
    """The cell artifact, its manifest, fold_code and the provenance block both
    stages write into their run manifest, after the label horizon checks."""
    cells, cell_manifest = datasets.load_cells(CELL_DIR)
    src = cell_manifest["source_manifest"]
    stats = cell_manifest["stats"]
    print(
        f"cells: {stats['n_cells']:,} at {cell_manifest['cell_size_m']} m, built "
        f"{cell_manifest['built_utc']} at commit {(cell_manifest['git_commit'] or '?')[:9]}, "
        f"from {cell_manifest['source_array_dir']} built {src['built_utc']}"
    )
    print(
        f"{stats['n_cells_with_any_developed']:,} cells "
        f"({stats['share_cells_with_any_developed']:.2%}) have any conversion"
    )

    population_stats = load_population_stats(POPULATION_STATS_PATH)
    # Same three checks as regress_train.py: row count, positive count and source
    # parquet together pin the label horizon.
    if population_stats["total_rows"] != src["total_rows"]:
        raise AssertionError(
            f"{POPULATION_STATS_PATH} has {population_stats['total_rows']:,} rows, the "
            f"cells' source arrays have {src['total_rows']:,}. One of them is stale."
        )
    if population_stats["total_positive"] != int(cells["n_developed"].sum()):
        raise AssertionError(
            f"{POPULATION_STATS_PATH} has {population_stats['total_positive']:,} positives, "
            f"the cells hold {int(cells['n_developed'].sum()):,}. Different label horizons."
        )
    if population_stats["source_parquet_path"] != src["source_parquet_path"]:
        raise AssertionError(
            f"{POPULATION_STATS_PATH} was scanned from {population_stats['source_parquet_path']} "
            f"but the cells come from {src['source_parquet_path']}."
        )

    fold_assignment = assign_folds(population_stats["block_row_counts"], N_SPLITS, SEED)
    fold_code = array_io.fold_of_code(cell_manifest, fold_assignment)
    print(f"\n=== seed {SEED}, {N_SPLITS} folds ===")

    provenance = {
        "path": str(CELL_DIR),
        "cell_size_m": cell_manifest["cell_size_m"],
        "built_utc": cell_manifest["built_utc"],
        "git_commit": cell_manifest["git_commit"],
        "n_cells": stats["n_cells"],
        "share_cells_with_any_developed": stats["share_cells_with_any_developed"],
        "source_arrays": src,
    }
    return cells, cell_manifest, fold_code, provenance


def check_cells_match(run_dir, cell_manifest):
    """Matching cell_idx alone would pass on a rebuilt artifact with the same grid
    but different labels, so the build time has to match too. Returns the other
    run's manifest."""
    with open(run_dir / "manifest.json") as fh:
        other = json.load(fh)
    if other["cells"]["built_utc"] != cell_manifest["built_utc"]:
        raise AssertionError(
            f"{run_dir} was trained on cells built {other['cells']['built_utc']}, "
            f"this run reads cells built {cell_manifest['built_utc']}."
        )
    return other


def run_classify(cells, cell_manifest, fold_code, provenance):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    regressor = None
    if REGRESSOR_DIR is not None:
        reg_manifest = check_cells_match(REGRESSOR_DIR, cell_manifest)
        regressor = {
            "path": str(REGRESSOR_DIR),
            "loss": reg_manifest.get("loss"),
            "weight_by_n_wetland": reg_manifest.get("weight_by_n_wetland"),
            "seed": reg_manifest.get("seed"),
            "n_splits": reg_manifest.get("n_splits"),
        }

    run_manifest = {
        "task": "any 2019 wetland developed by 2024, per cell (n_developed > 0)",
        "model": MODEL_CLS.__name__,
        "loss": LOSS_NAME,
        "gamma": GAMMA,
        "alpha": ALPHA,
        "pos_weight": POS_WEIGHT,
        "seed": SEED,
        "n_splits": N_SPLITS,
        "max_epochs": MAX_EPOCHS,
        "checkpoint_scored": "final epoch, no early stopping",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "pos_per_batch": POS_PER_BATCH,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "min_wetland_train": MIN_WETLAND_TRAIN,
        "recall_target_for_threshold": RECALL_TARGET,
        "scalars": list(datasets.CELL_SCALARS),
        "cells": provenance,
        "regressor_compared": regressor,
        "folds": {},
    }

    all_fold_rows, all_bin_rows = [], []
    for fold_idx in range(N_SPLITS):
        print(f"\n--- fold {fold_idx} ---")
        data_module = datasets.CellDataModule(
            cells, cell_manifest, fold_code, fold_idx,
            batch_size=BATCH_SIZE, val_batch_size=VAL_BATCH_SIZE,
            min_wetland_train=MIN_WETLAND_TRAIN, pos_per_batch=POS_PER_BATCH,
            steps_per_epoch=STEPS_PER_EPOCH, num_workers=NUM_WORKERS, seed=SEED,
        )
        data_module.setup()

        # Before training, so a fold mismatch costs seconds, not a fold of training.
        reg_pred = load_regressor_fold(fold_idx, data_module.val_idx) if regressor else None

        train_rate = float((cells["n_developed"][data_module.train_idx] > 0).mean())
        print(
            f"batches: {len(data_module.train_sampler):,} per epoch at {BATCH_SIZE} cells, "
            f"{data_module.batch_positive_rate:.2%} positive, against {train_rate:.2%} in "
            f"the fold's train cells. loss: {LOSS_NAME}, gamma {GAMMA}, alpha {ALPHA}, "
            f"pos_weight {POS_WEIGHT}"
        )

        fold_score, bin_rows = run_classify_fold(
            fold_idx, data_module, RESULTS_DIR / f"fold_{fold_idx}", reg_pred
        )
        all_fold_rows.append(fold_score)
        all_bin_rows += bin_rows
        run_manifest["folds"][fold_idx] = {
            "n_train_cells": fold_score["n_train_cells"],
            "n_val_cells": fold_score["n_val_cells"],
            "val_positive_rate": fold_score["positive_rate"],
            "train_positive_rate": fold_score["train_positive_rate"],
            "batch_positive_rate": fold_score["batch_positive_rate"],
            "threshold": fold_score["threshold"],
            "epochs_run": fold_score["epochs_run"],
            "scalar_mean": fold_score["scalar_mean"],
            "scalar_std": fold_score["scalar_std"],
        }

    with open(RESULTS_DIR / "manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    fold_dirs = [RESULTS_DIR / f"fold_{i}" for i in range(N_SPLITS)]
    metrics_csvs = [c for d in fold_dirs for c in sorted(d.glob("**/metrics.csv"))[-1:]]
    if metrics_csvs:
        reporting.plot_mean_loss_curve(
            metrics_csvs, RESULTS_DIR / "loss_curve_mean.png",
            f"Mean training and validation loss over {N_SPLITS} spatially blocked folds",
            ylabel=LOSS_NAME,
        )

    fold_preds = []
    for d in fold_dirs:
        tr, va = np.load(d / "train_preds.npz"), np.load(d / "val_preds.npz")
        fold_preds.append((tr["any"], tr["prob"], va["any"], va["prob"]))
    reporting.plot_mean_pr_curve(
        fold_preds, RESULTS_DIR / "pr_curve_mean.png",
        f"Mean precision recall over {N_SPLITS} folds, any conversion in the cell",
        chance_unit="cells",
    )

    print("\n=== fold summary ===")
    print(f"precision, recall and f1 are at each fold's {RECALL_TARGET:.0%} train recall threshold")
    df = reporting.write_fold_summary(all_fold_rows, RESULTS_DIR / "fold_summary.csv")

    if regressor:
        p_col = f"precision_at_recall_{RECALL_TARGET}"
        cols = ["fold", "positive_rate", "auprc", "regressor_auprc", p_col, f"regressor_{p_col}"]
        print(f"\n=== classifier against {REGRESSOR_DIR.name}, val ===")
        print(df[cols].to_string(index=False))
        print(df[cols[1:]].agg(["mean", "std"]).to_string())
        df[cols].to_csv(RESULTS_DIR / "comparison.csv", index=False)

    bins = pd.DataFrame(all_bin_rows)
    bins.to_csv(RESULTS_DIR / "auprc_by_n_wetland.csv", index=False)
    print("\n=== val AUPRC by n_wetland bin, mean across folds ===")
    print(bins.groupby(["n_wetland_bin", "model"], sort=False)[["n_cells", "positive_rate", "auprc"]]
          .mean().unstack("model").to_string())

    best_fold = max(all_fold_rows, key=lambda row: row["auprc"])
    shutil.copy2(Path(best_fold["checkpoint_path"]), RESULTS_DIR / "best_model.ckpt")


def load_classifier_fold(fold_idx, val_all_idx):
    """Checks a stage one fold scored the same val cells, in the same order, as
    this fold's val_all_preds.npz. Stage one keeps every val cell, so it lines up
    with val_all_idx, not the converting only val_idx."""
    z = np.load(REG_CLASSIFIER_DIR / f"fold_{fold_idx}" / "val_preds.npz")
    if not np.array_equal(z["cell_idx"], val_all_idx):
        raise AssertionError(
            f"{REG_CLASSIFIER_DIR} fold {fold_idx} val cells differ from this run's. "
            f"Check SEED, N_SPLITS and CELL_DIR against its manifest."
        )


def baseline_scores(data_module):
    """
    A constant (train mean log share) and an OLS of log share on the three z
    scored cell scalars, both fit on the fold's train cells and scored on its val
    cells, with their own smearing factors. Returns score columns prefixed by
    baseline name, plus the OLS coefficients, intercept first.
    """
    c = data_module.cells
    scalars = data_module.ds.scalars
    tr, va = data_module.train_idx, data_module.val_idx
    t_tr, t_va = c["target"][tr], c["target"][va]
    y_tr = np.log(t_tr.astype(np.float64))

    x_tr = np.column_stack([np.ones(tr.size), scalars[tr]])
    x_va = np.column_stack([np.ones(va.size), scalars[va]])
    coef, *_ = np.linalg.lstsq(x_tr, y_tr, rcond=None)

    out = {"ols_scalars_coef": coef.tolist()}
    for name, p_tr, p_va in (
        ("constant", np.full(tr.size, y_tr.mean()), np.full(va.size, y_tr.mean())),
        ("ols_scalars", x_tr @ coef, x_va @ coef),
    ):
        smear = reporting.smearing_factor(p_tr, t_tr)
        score = reporting.score_log_share(p_va, t_va, c["n_wetland"][va], smear)
        out.update({f"{name}_{k}": score[k] for k in ("mse", "mae", "r2", "spearman", "pred_over_actual")})
    return out


def run_regress_fold(fold_idx, data_module, fold_dir):
    """
    Trains one stage two fold for REG_MAX_EPOCHS, writes its plots and predictions
    into fold_dir, returns a score dict. data_module must be set up with
    positive_only.
    """
    model = models.CellTemporalConvolutionalLogShare(
        n_years=N_YEARS, lr=REG_LR, weight_decay=REG_WEIGHT_DECAY,
    )

    fold_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=str(fold_dir.parent), name=f"fold_{fold_idx}")
    # monitor=None keeps only the most recent epoch, which after fit is the final one.
    checkpoint = ModelCheckpoint(monitor=None, save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=REG_MAX_EPOCHS,
        callbacks=[checkpoint],
        logger=logger,
        enable_progress_bar=True,
        accelerator="auto",
    )
    trainer.fit(model, datamodule=data_module)
    epochs_run = trainer.current_epoch

    final_model = models.CellTemporalConvolutionalLogShare.load_from_checkpoint(
        checkpoint.best_model_path
    )

    # Train first: the smearing factor comes off the train cells and gets applied
    # unchanged to val.
    train_pred, train_target, train_n_wet = reporting.get_cell_predictions(
        final_model, data_module.train_eval_dataloader()
    )
    smear = reporting.smearing_factor(train_pred, train_target)
    np.savez(fold_dir / "train_preds.npz", pred=train_pred, target=train_target,
             n_wetland=train_n_wet, cell_idx=data_module.train_idx, smear=smear)

    pred, target, n_wet = reporting.get_cell_predictions(final_model, data_module.val_dataloader())
    np.savez(fold_dir / "val_preds.npz", pred=pred, target=target, n_wetland=n_wet,
             cell_idx=data_module.val_idx, smear=smear)

    # Every val cell, including the ones that did not convert, which is what the
    # hurdle product p(any) * exp(pred) * smear gets scored on. target is 0 on
    # most of these, so there is no log of it here.
    all_pred, all_target, all_n_wet = reporting.get_cell_predictions(
        final_model, data_module.val_all_dataloader()
    )
    np.savez(fold_dir / "val_all_preds.npz", pred=all_pred, target=all_target,
             n_wetland=all_n_wet, cell_idx=data_module.val_all_idx, smear=smear)

    metrics_csvs = sorted(fold_dir.glob("**/metrics.csv"))
    if metrics_csvs:
        reporting.plot_loss_curve(
            metrics_csvs[-1], fold_dir / "loss_curve.png",
            f"Fold {fold_idx}: training and validation loss", ylabel=REG_LOSS_NAME,
        )
        reporting.plot_log_share_val_metrics(metrics_csvs[-1:], fold_dir / "val_metrics.png",
                                             title_suffix=f", fold {fold_idx}")

    reporting.plot_log_share_hexbin(
        pred, target, fold_dir / "pred_vs_actual.png",
        f"Fold {fold_idx}: predicted against actual log share, converting cells",
    )
    calibration = reporting.plot_log_share_calibration(
        pred, target, fold_dir / "calibration.png", fold_dir / "calibration_ratio.png",
        f"Fold {fold_idx}: predicted against actual log share",
        f"Fold {fold_idx}: actual over predicted share, by bin",
    )
    calibration.to_csv(fold_dir / "calibration.csv", index=False)

    fold_score = reporting.score_log_share(pred, target, n_wet, smear)
    train_score = reporting.score_log_share(train_pred, train_target, train_n_wet, smear)
    fold_score.update({f"train_{k}": v for k, v in train_score.items()})
    fold_score.update(baseline_scores(data_module))
    fold_score.update({
        "fold": fold_idx,
        "epochs_run": epochs_run,
        "checkpoint_path": checkpoint.best_model_path,
        "n_train_cells": int(data_module.train_idx.size),
        "n_val_cells": int(data_module.val_idx.size),
        "n_val_all_cells": int(data_module.val_all_idx.size),
        "scalar_mean": data_module.scalar_mean.tolist(),
        "scalar_std": data_module.scalar_std.tolist(),
    })
    return fold_score


def run_regress(cells, cell_manifest, fold_code, provenance):
    REG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    classifier = None
    if REG_CLASSIFIER_DIR is not None:
        clf_manifest = check_cells_match(REG_CLASSIFIER_DIR, cell_manifest)
        classifier = {
            "path": str(REG_CLASSIFIER_DIR),
            "model": clf_manifest.get("model"),
            "seed": clf_manifest.get("seed"),
            "n_splits": clf_manifest.get("n_splits"),
        }

    run_manifest = {
        "task": "log(n_developed / n_wetland), per cell, cells with n_developed > 0 only",
        "model": models.CellTemporalConvolutionalLogShare.__name__,
        "loss": REG_LOSS_NAME,
        "weight_by_n_wetland": False,
        "back_transform": "exp(pred) * smear, smear is Duan's factor off each fold's train cells",
        "seed": SEED,
        "n_splits": N_SPLITS,
        "max_epochs": REG_MAX_EPOCHS,
        "checkpoint_scored": "final epoch, no early stopping",
        "lr": REG_LR,
        "weight_decay": REG_WEIGHT_DECAY,
        "batch_size": REG_BATCH_SIZE,
        "min_wetland_train": MIN_WETLAND_TRAIN,
        "scalars": list(datasets.CELL_SCALARS),
        "baselines": "constant train mean log share, OLS on the z scored cell scalars",
        "cells": provenance,
        "classifier_checked": classifier,
        "folds": {},
    }

    all_fold_rows = []
    for fold_idx in range(N_SPLITS):
        print(f"\n--- fold {fold_idx} ---")
        data_module = datasets.CellDataModule(
            cells, cell_manifest, fold_code, fold_idx,
            batch_size=REG_BATCH_SIZE, val_batch_size=VAL_BATCH_SIZE,
            min_wetland_train=MIN_WETLAND_TRAIN, num_workers=NUM_WORKERS, seed=SEED,
            positive_only=True,
        )
        data_module.setup()

        if classifier:
            load_classifier_fold(fold_idx, data_module.val_all_idx)

        print(f"batches: {len(data_module.train_sampler):,} per epoch at {REG_BATCH_SIZE} cells. "
              f"loss: {REG_LOSS_NAME}, every converting cell counts once")

        fold_score = run_regress_fold(fold_idx, data_module, REG_RESULTS_DIR / f"fold_{fold_idx}")
        all_fold_rows.append(fold_score)
        run_manifest["folds"][fold_idx] = {
            "n_train_cells": fold_score["n_train_cells"],
            "n_val_cells": fold_score["n_val_cells"],
            "n_val_all_cells": fold_score["n_val_all_cells"],
            "smear": fold_score["smear"],
            "ols_scalars_coef": fold_score["ols_scalars_coef"],
            "epochs_run": fold_score["epochs_run"],
            "scalar_mean": fold_score["scalar_mean"],
            "scalar_std": fold_score["scalar_std"],
        }

    with open(REG_RESULTS_DIR / "manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    fold_dirs = [REG_RESULTS_DIR / f"fold_{i}" for i in range(N_SPLITS)]
    metrics_csvs = [c for d in fold_dirs for c in sorted(d.glob("**/metrics.csv"))[-1:]]
    if metrics_csvs:
        reporting.plot_mean_loss_curve(
            metrics_csvs, REG_RESULTS_DIR / "loss_curve_mean.png",
            f"Mean training and validation loss over {N_SPLITS} spatially blocked folds",
            ylabel=REG_LOSS_NAME,
        )
        reporting.plot_log_share_val_metrics(metrics_csvs, REG_RESULTS_DIR / "val_metrics_mean.png",
                                             title_suffix=f", {N_SPLITS} fold mean")

    val_pred_all, val_target_all = [], []
    for d in fold_dirs:
        va = np.load(d / "val_preds.npz")
        val_pred_all.append(va["pred"])
        val_target_all.append(va["target"])
    # The folds partition the converting cells, so this scores each one once.
    val_pred_all, val_target_all = np.concatenate(val_pred_all), np.concatenate(val_target_all)
    reporting.plot_log_share_hexbin(
        val_pred_all, val_target_all, REG_RESULTS_DIR / "pred_vs_actual_pooled.png",
        "Predicted against actual log share, converting cells, all folds",
    )
    calibration = reporting.plot_log_share_calibration(
        val_pred_all, val_target_all,
        REG_RESULTS_DIR / "calibration_pooled.png", REG_RESULTS_DIR / "calibration_pooled_ratio.png",
        "Predicted against actual log share, all folds",
        "Actual over predicted share by bin, all folds",
    )
    calibration.to_csv(REG_RESULTS_DIR / "calibration_pooled.csv", index=False)
    reporting.write_train_log_share_calibration_pooled(fold_dirs, REG_RESULTS_DIR)

    print("\n=== fold summary, val cells with any conversion (positive rate 100% by construction) ===")
    print("errors in natural log of the share. constant_ and ols_scalars_ are the baselines")
    reporting.write_regression_summary(
        all_fold_rows, REG_RESULTS_DIR / "fold_summary.csv",
        base=("mse", "constant_mse", "ols_scalars_mse", "mae", "ols_scalars_mae", "r2",
              "ols_scalars_r2", "spearman", "ols_scalars_spearman", "pred_over_actual",
              "ols_scalars_pred_over_actual", "smear", "n_cells"),
    )

    best_fold = min(all_fold_rows, key=lambda row: row["mse"])
    shutil.copy2(Path(best_fold["checkpoint_path"]), REG_RESULTS_DIR / "best_model.ckpt")


def main():
    parser = argparse.ArgumentParser(description="Train one stage of the cell hurdle model.")
    parser.add_argument("stage", choices=("classify", "regress"),
                        help="classify: any conversion in the cell. regress: log share "
                             "converted, on converting cells only.")
    args = parser.parse_args()

    pl.seed_everything(SEED, workers=True)
    cells, cell_manifest, fold_code, provenance = load_checked_cells()
    if args.stage == "classify":
        run_classify(cells, cell_manifest, fold_code, provenance)
    else:
        run_regress(cells, cell_manifest, fold_code, provenance)

if __name__ == "__main__":
    main()
