"""
Both stages of the hurdle model, spatially blocked CV over a build_cells.py
artifact:

    python hurdle_model_train.py classify
    python hurdle_model_train.py regress

classify is stage one: a cell classifier (MODEL_CLS, a models.Cell*Classifier)
for whether a cell has any 2019 wetland developed by 2024 (n_developed > 0).
Conversion to other, not developed, land counts as negative.

regress is stage two: a log share regressor (REG_MODEL, the TCN or its MLP
baseline) on the natural log of one of two per cell targets, picked by
REG_TARGET, trained and scored
only on cells with n_developed > 0. Its fold summary carries a constant and an
OLS on the three cell scalars as baselines.

    "area": area_converted, the percent of the whole 300 m cell footprint that
    was 2019 wetland and is developed by 2024 (100 n_developed / footprint_px).
    "share": n_developed / n_wetland, the share of the cell's 2019 wetland
    developed by 2024.

44% of converting cells have exactly one developed pixel. On area that puts
them at log(1%) = 0 on a full 100 pixel cell, a hard floor, and n_wetland caps
the target at 100 n_wetland / footprint_px. On share their target is just
-log(n_wetland). Either way cell size carries a lot of it, so the OLS on
log1p(n_wetland) and distance is what shows whether the embeddings add anything
past cell size and distance. The two targets correlate at only r = 0.55 on
converting cells, so their errors are not on the same scale.

Run order from scratch is the same as regress_train.py:
    python population_stats.py
    python build_arrays.py
    python build_cells.py --cell-size-m 300
    python hurdle_model_train.py classify
    python hurdle_model_train.py regress

Folds are the regress_train.py folds, same assign_folds call, same SEED and
N_SPLITS, for both stages. That is what lets the two stages be combined per
cell on cell_idx. FOLD_SCHEME swaps them for a two fold north and south or east
and west split of the state, see folds.assign_geographic_folds.

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

import arrays as array_io
import datasets
import models
import reporting
from features import YEARS_2017_2019
from population_stats import load_population_stats
from folds import SPLIT_CENTER, GEOGRAPHIC_FOLD_NAMES, assign_folds, assign_geographic_folds

CELL_DIR = "data/cells_2019_2024_300m"
POPULATION_STATS_PATH = "data/population_stats_2019_2024.json"
N_YEARS = len(YEARS_2017_2019)

# CellTemporalConvolutionalClassifier or CellMultiLayerPerceptronClassifier. Both
# take the same batches, so nothing else in this file changes between them.
MODEL_CLS = models.CellMultiLayerPerceptronClassifier

RESULTS_DIR = Path("results/hurdle_clf_mlp_300m_l2_512")

SEED = 0

# "blocked" is the 5 fold random draw of 10 km blocks, balanced on wetland
# pixels. "north_south" and "east_west" cut Florida in two at the 2019 wetland
# median (folds.SPLIT_CENTER), train on one half and validate on the other, so
# fold 0 validates on the north (or east) and fold 1 on the south (or west). Each
# half holds half the wetland. East and west then convert at about the same rate,
# but the south still converts at about twice the north's rate, so a north and
# south run mixes a shift in base rate into the transfer test.
FOLD_SCHEME = "east_west"
BLOCKED_N_SPLITS = 5
N_SPLITS = BLOCKED_N_SPLITS if FOLD_SCHEME == "blocked" else 2

# Geographic runs get their own results dirs so they never land on top of a
# blocked run.
if FOLD_SCHEME != "blocked":
    RESULTS_DIR = RESULTS_DIR.with_name(f"{RESULTS_DIR.name}_{FOLD_SCHEME}")

# Written into both stages' run manifests.
FOLD_SPEC = (
    {"scheme": "blocked", "n_splits": N_SPLITS, "seed": SEED}
    if FOLD_SCHEME == "blocked" else
    {"scheme": FOLD_SCHEME, "n_splits": N_SPLITS, "center_lat_lon": list(SPLIT_CENTER),
     "fold_names": list(GEOGRAPHIC_FOLD_NAMES[FOLD_SCHEME]),
     "center": "2019 wetland pixel median, half the wetland on each side",
     "block_side": "each 10 km block goes whole to the side its square center is on"}
)

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
# "area" or "share", see REG_TARGETS.
REG_TARGET = "area"

# "tcn" or "mlp". Both take the same batches and constructor arguments, and the
# key goes in the results path so the two never write over each other.
REG_MODEL = "mlp"
REG_MODELS = {
    "tcn": models.CellTemporalConvolutionalLogShare,
    "mlp": models.CellMultiLayerPerceptronLogShare,
}
REG_MODEL_CLS = REG_MODELS[REG_MODEL]

# target_key and weight_key fill the y and weight slots of each batch. The weight
# is the target's denominator, so the predicted over actual converted pixels stays
# in pixels. label goes in plot titles, quantity on axes. scale names the back
# transformed outputs, to_percent takes the target to percent (area_converted
# already is one), and natural_quantity labels the percent axes.
REG_TARGETS = {
    "area": {
        "target_key": "area_converted",
        "weight_key": "footprint_px",
        "label": "log percent area",
        "quantity": "percent of cell area converted",
        "weight_label": "weighted by cell footprint",
        "results_suffix": "log_area",
        "scale": "area",
        "to_percent": 1.0,
        "natural_quantity": "cell area converted",
        "units": "percentage points of the 300 m cell footprint",
        "cap": "100 * n_wetland / footprint_px",
    },
    "share": {
        "target_key": "target",
        "weight_key": "n_wetland",
        "label": "log share",
        "quantity": "share of wetland converted",
        "weight_label": "weighted by wetland pixels",
        "results_suffix": "log_share",
        "scale": "share",
        "to_percent": 100.0,
        "natural_quantity": "share of wetland converted",
        "units": "percentage points of the cell's 2019 wetland",
        "cap": "1.0",
    },
}
REG = REG_TARGETS[REG_TARGET]
REG_TARGET_KEY = REG["target_key"]
REG_WEIGHT_KEY = REG["weight_key"]
REG_RESULTS_DIR = Path(f"results/hurdle_reg_{REG_MODEL}_3km_{REG['results_suffix']}")
if FOLD_SCHEME != "blocked":
    REG_RESULTS_DIR = REG_RESULTS_DIR.with_name(f"{REG_RESULTS_DIR.name}_{FOLD_SCHEME}")

# A stage one run to check val cell_idx against fold by fold, so the two can be
# joined later. None skips it. Every hurdle_clf run so far was trained on cells
# built 2026-10-01, before the current artifact, so pointing at one fails until
# stage one is rerun.
REG_CLASSIFIER_DIR = None

# About 29k converting train cells per fold, so 1024 would be only 28 steps an epoch.
REG_BATCH_SIZE = 256
REG_MAX_EPOCHS = 15
REG_LR = 1e-3
REG_WEIGHT_DECAY = 0
REG_LOSS_NAME = f"MSE on {REG['label']}"

# Equal count bins on pred, shared by the log and back transformed calibration
# plots so bin k is the same cells in both.
CALIBRATION_BINS = 20


def target_cap(cells, idx):
    """Most the target can be for each cell in idx. A smeared prediction can run
    past it (exp(pred) of 0.7 times a smear of 1.6 is over 100% share), so back
    transformed predictions get clipped here before scoring."""
    if REG_TARGET == "share":
        return np.ones(idx.size)
    return 100.0 * cells["n_wetland"][idx].astype(np.float64) / cells["footprint_px"][idx]


def run_classify_fold(fold_idx, data_module, fold_dir):
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


def assign_scheme_folds(scheme, population_stats, cell_manifest):
    """dict block_id to fold for one FOLD_SCHEME value. Takes the scheme as an
    argument so the fold notebook can draw all three off one load."""
    if scheme == "blocked":
        return assign_folds(population_stats["block_row_counts"], BLOCKED_N_SPLITS, SEED)
    origin = cell_manifest["block_origin_window"]
    if origin["block_m"] != 10_000:
        raise AssertionError(f"block grid is {origin['block_m']} m, folds.BLOCK_M assumes 10 km")
    return assign_geographic_folds(cell_manifest["block_ids"], (origin["u0"], origin["v0"]), scheme)


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

    fold_assignment = assign_scheme_folds(FOLD_SCHEME, population_stats, cell_manifest)
    fold_code = array_io.fold_of_code(cell_manifest, fold_assignment)
    if FOLD_SCHEME == "blocked":
        print(f"\n=== seed {SEED}, {N_SPLITS} folds ===")
    else:
        names = GEOGRAPHIC_FOLD_NAMES[FOLD_SCHEME]
        print(f"\n=== {FOLD_SCHEME} at {SPLIT_CENTER[0]:.4f}, {SPLIT_CENTER[1]:.4f}: fold 0 "
              f"validates on the {names[0]} ({int((fold_code == 0).sum())} blocks), fold 1 on "
              f"the {names[1]} ({int((fold_code == 1).sum())} blocks) ===")

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
    # Manifests from before FOLD_SCHEME existed were all blocked.
    other_scheme = other.get("folds_spec", {"scheme": "blocked"})["scheme"]
    if other_scheme != FOLD_SCHEME:
        raise AssertionError(f"{run_dir} used {other_scheme} folds, this run uses {FOLD_SCHEME}.")
    return other


def run_classify(cells, cell_manifest, fold_code, provenance):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "task": "any 2019 wetland developed by 2024, per cell (n_developed > 0)",
        "model": MODEL_CLS.__name__,
        "loss": LOSS_NAME,
        "gamma": GAMMA,
        "alpha": ALPHA,
        "pos_weight": POS_WEIGHT,
        "seed": SEED,
        "folds_spec": FOLD_SPEC,
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

        train_rate = float((cells["n_developed"][data_module.train_idx] > 0).mean())
        print(
            f"batches: {len(data_module.train_sampler):,} per epoch at {BATCH_SIZE} cells, "
            f"{data_module.batch_positive_rate:.2%} positive, against {train_rate:.2%} in "
            f"the fold's train cells. loss: {LOSS_NAME}, gamma {GAMMA}, alpha {ALPHA}, "
            f"pos_weight {POS_WEIGHT}"
        )

        fold_score, bin_rows = run_classify_fold(
            fold_idx, data_module, RESULTS_DIR / f"fold_{fold_idx}"
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
    reporting.write_fold_summary(all_fold_rows, RESULTS_DIR / "fold_summary.csv")

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
    A constant (train mean log target) and an OLS of log target on the three z
    scored cell scalars, both fit on the fold's train cells and scored on its val
    cells, with their own smearing factors. Returns score columns prefixed by
    baseline name plus the OLS coefficients, intercept first, and each baseline's
    (train pred, val pred, smear) for the metric tables.
    """
    c = data_module.cells
    scalars = data_module.ds.scalars
    tr, va = data_module.train_idx, data_module.val_idx
    t_tr, t_va = c[REG_TARGET_KEY][tr], c[REG_TARGET_KEY][va]
    y_tr = np.log(t_tr.astype(np.float64))

    x_tr = np.column_stack([np.ones(tr.size), scalars[tr]])
    x_va = np.column_stack([np.ones(va.size), scalars[va]])
    coef, *_ = np.linalg.lstsq(x_tr, y_tr, rcond=None)

    out = {"ols_scalars_coef": coef.tolist()}
    preds = {}
    for name, p_tr, p_va in (
        ("constant", np.full(tr.size, y_tr.mean()), np.full(va.size, y_tr.mean())),
        ("ols_scalars", x_tr @ coef, x_va @ coef),
    ):
        smear = reporting.smearing_factor(p_tr, t_tr)
        score = reporting.score_log_share(p_va, t_va, c[REG_WEIGHT_KEY][va], smear)
        out.update({f"{name}_{k}": score[k] for k in ("mse", "mae", "r2", "spearman", "pred_over_actual")})
        preds[name] = (p_tr, p_va, smear)
    return out, preds


def run_regress_fold(fold_idx, data_module, fold_dir):
    """
    Trains one stage two fold for REG_MAX_EPOCHS, writes its plots and predictions
    into fold_dir. data_module must be set up with positive_only.

    Returns (score dict, log scale metric rows, back transformed metric rows, val
    predictions). The rows cover the model and both baselines on train and val.
    The val predictions are {model: (pred, target, smear, cap)}, for scoring
    pooled across folds.
    """
    model = REG_MODEL_CLS(
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

    final_model = REG_MODEL_CLS.load_from_checkpoint(
        checkpoint.best_model_path
    )

    # Train first: the smearing factor comes off the train cells and gets applied
    # unchanged to val.
    # The third slot is REG_WEIGHT_KEY, and gets saved under that name, so the npz
    # says whether it holds footprint_px or n_wetland.
    train_pred, train_target, train_w = reporting.get_cell_predictions(
        final_model, data_module.train_eval_dataloader()
    )
    smear = reporting.smearing_factor(train_pred, train_target)
    np.savez(fold_dir / "train_preds.npz", pred=train_pred, target=train_target,
             cell_idx=data_module.train_idx, smear=smear, **{REG_WEIGHT_KEY: train_w})

    pred, target, w = reporting.get_cell_predictions(final_model, data_module.val_dataloader())
    np.savez(fold_dir / "val_preds.npz", pred=pred, target=target,
             cell_idx=data_module.val_idx, smear=smear, **{REG_WEIGHT_KEY: w})

    # Every val cell, including the ones that did not convert, which is what the
    # hurdle product p(any) * exp(pred) * smear gets scored on. target is 0 on
    # most of these, so there is no log of it here.
    all_pred, all_target, all_w = reporting.get_cell_predictions(
        final_model, data_module.val_all_dataloader()
    )
    np.savez(fold_dir / "val_all_preds.npz", pred=all_pred, target=all_target,
             cell_idx=data_module.val_all_idx, smear=smear, **{REG_WEIGHT_KEY: all_w})

    metrics_csvs = sorted(fold_dir.glob("**/metrics.csv"))
    if metrics_csvs:
        reporting.plot_loss_curve(
            metrics_csvs[-1], fold_dir / "loss_curve.png",
            f"Fold {fold_idx}: training and validation loss", ylabel=REG_LOSS_NAME,
        )
        reporting.plot_log_share_val_metrics(metrics_csvs[-1:], fold_dir / "val_metrics.png",
                                             title_suffix=f", fold {fold_idx}", quantity=REG["quantity"],
                                             weight_label=REG["weight_label"])

    reporting.plot_log_share_hexbin(
        pred, target, fold_dir / "pred_vs_actual.png",
        f"Fold {fold_idx}: predicted against actual {REG['label']}, converting cells",
        quantity=REG["quantity"],
    )
    calibration = reporting.plot_log_share_calibration(
        pred, target, fold_dir / "calibration_log.png", fold_dir / "calibration_log_ratio.png",
        f"Fold {fold_idx}: predicted against actual {REG['label']}",
        f"Fold {fold_idx}: actual over predicted {REG['label'].removeprefix('log ')}, by bin",
        n_bins=CALIBRATION_BINS, quantity=REG["quantity"],
    )
    calibration.to_csv(fold_dir / "calibration_log.csv", index=False)

    scale = REG["scale"]
    cap_tr = target_cap(data_module.cells, data_module.train_idx)
    cap_va = target_cap(data_module.cells, data_module.val_idx)
    calibration = reporting.plot_natural_calibration(
        pred, target, smear, cap_va, REG["to_percent"],
        fold_dir / f"calibration_{scale}.png", fold_dir / f"calibration_{scale}_ratio.png",
        f"Fold {fold_idx}: predicted against actual {REG['natural_quantity']}",
        f"Fold {fold_idx}: actual over predicted {REG['natural_quantity']}, by bin",
        n_bins=CALIBRATION_BINS, quantity=REG["natural_quantity"],
    )
    calibration.to_csv(fold_dir / f"calibration_{scale}.csv", index=False)

    fold_score = reporting.score_log_share(pred, target, w, smear)
    train_score = reporting.score_log_share(train_pred, train_target, train_w, smear)
    fold_score.update({f"train_{k}": v for k, v in train_score.items()})
    baselines, baseline_preds = baseline_scores(data_module)
    fold_score.update(baselines)

    c = data_module.cells
    t_tr = c[REG_TARGET_KEY][data_module.train_idx]
    t_va = c[REG_TARGET_KEY][data_module.val_idx]
    preds = {REG_MODEL: (train_pred, pred, smear), **baseline_preds}
    log_rows, natural_rows, val_preds = [], [], {}
    for name, (p_tr, p_va, s) in preds.items():
        for split, p, t, cap in (("train", p_tr, t_tr, cap_tr), ("val", p_va, t_va, cap_va)):
            key = {"split": split, "fold": fold_idx, "model": name}
            log_rows.append({**key, **reporting.score_log_scale(p, t)})
            natural_rows.append({**key, **reporting.score_natural_scale(p, t, s, cap, REG["to_percent"])})
        val_preds[name] = (p_va, t_va, s, cap_va)
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
    return fold_score, log_rows, natural_rows, val_preds


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
        "task": f"log({REG_TARGET_KEY}), per cell, cells with n_developed > 0 only",
        "reg_target": REG_TARGET,
        "target_key": REG_TARGET_KEY,
        "target_definition": cell_manifest[REG_TARGET_KEY],
        "weight_key": REG_WEIGHT_KEY,
        "model": REG_MODEL_CLS.__name__,
        "loss": REG_LOSS_NAME,
        "weight_by_n_wetland": False,
        "back_transform": {
            "smear": "Duan's factor, mean of exp(log target minus pred) over each fold's "
                     "train cells, applied unchanged to that fold's val cells",
            "median_estimate": "exp(pred), scores mae and median_ae",
            "mean_estimate": "exp(pred) * smear, scores rmse, r2, mean_ratio and the "
                             f"calibration_{REG['scale']} plots",
            "clip": REG["cap"],
            "units": REG["units"],
            "calibration_bins": f"{CALIBRATION_BINS} equal count bins sorted on pred, same "
                                "cells in the log and back transformed versions",
            "pooled_rows": "every val cell across folds scored once, each with its own fold's smear",
        },
        "seed": SEED,
        "folds_spec": FOLD_SPEC,
        "n_splits": N_SPLITS,
        "max_epochs": REG_MAX_EPOCHS,
        "checkpoint_scored": "final epoch, no early stopping",
        "lr": REG_LR,
        "weight_decay": REG_WEIGHT_DECAY,
        "batch_size": REG_BATCH_SIZE,
        "min_wetland_train": MIN_WETLAND_TRAIN,
        "scalars": list(datasets.CELL_SCALARS),
        "baselines": "constant train mean log target, OLS on the z scored cell scalars",
        "cells": provenance,
        "classifier_checked": classifier,
        "folds": {},
    }

    all_fold_rows, all_log_rows, all_natural_rows = [], [], []
    pooled_val = {}
    for fold_idx in range(N_SPLITS):
        print(f"\n--- fold {fold_idx} ---")
        data_module = datasets.CellDataModule(
            cells, cell_manifest, fold_code, fold_idx,
            batch_size=REG_BATCH_SIZE, val_batch_size=VAL_BATCH_SIZE,
            min_wetland_train=MIN_WETLAND_TRAIN, num_workers=NUM_WORKERS, seed=SEED,
            positive_only=True, target_key=REG_TARGET_KEY, weight_key=REG_WEIGHT_KEY,
        )
        data_module.setup()

        if classifier:
            load_classifier_fold(fold_idx, data_module.val_all_idx)

        print(f"batches: {len(data_module.train_sampler):,} per epoch at {REG_BATCH_SIZE} cells. "
              f"loss: {REG_LOSS_NAME}, every converting cell counts once")

        fold_score, log_rows, natural_rows, val_preds = run_regress_fold(
            fold_idx, data_module, REG_RESULTS_DIR / f"fold_{fold_idx}"
        )
        all_fold_rows.append(fold_score)
        all_log_rows += log_rows
        all_natural_rows += natural_rows
        for name, arrays in val_preds.items():
            pooled_val.setdefault(name, []).append(arrays)
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
                                             title_suffix=f", {N_SPLITS} fold mean", quantity=REG["quantity"],
                                             weight_label=REG["weight_label"])

    # The folds partition the converting cells, so pooling scores each one once.
    # Smear is spread to one value per cell since every fold fit its own.
    pooled = {}
    for name, per_fold in pooled_val.items():
        pooled[name] = (
            np.concatenate([p for p, _, _, _ in per_fold]),
            np.concatenate([t for _, t, _, _ in per_fold]),
            np.concatenate([np.full(p.size, s) for p, _, s, _ in per_fold]),
            np.concatenate([cap for _, _, _, cap in per_fold]),
        )
        key = {"split": "val", "fold": "pooled", "model": name}
        p, t, s, cap = pooled[name]
        all_log_rows.append({**key, **reporting.score_log_scale(p, t)})
        all_natural_rows.append({**key, **reporting.score_natural_scale(p, t, s, cap, REG["to_percent"])})

    val_pred_all, val_target_all, val_smear_all, val_cap_all = pooled[REG_MODEL]
    reporting.plot_log_share_hexbin(
        val_pred_all, val_target_all, REG_RESULTS_DIR / "pred_vs_actual_pooled.png",
        f"Predicted against actual {REG['label']}, converting cells, all folds",
        quantity=REG["quantity"],
    )
    calibration = reporting.plot_log_share_calibration(
        val_pred_all, val_target_all,
        REG_RESULTS_DIR / "calibration_log_pooled.png", REG_RESULTS_DIR / "calibration_log_pooled_ratio.png",
        f"Predicted against actual {REG['label']}, all folds",
        f"Actual over predicted {REG['label'].removeprefix('log ')} by bin, all folds",
        n_bins=CALIBRATION_BINS, quantity=REG["quantity"],
    )
    calibration.to_csv(REG_RESULTS_DIR / "calibration_log_pooled.csv", index=False)
    scale = REG["scale"]
    calibration = reporting.plot_natural_calibration(
        val_pred_all, val_target_all, val_smear_all, val_cap_all, REG["to_percent"],
        REG_RESULTS_DIR / f"calibration_{scale}_pooled.png",
        REG_RESULTS_DIR / f"calibration_{scale}_pooled_ratio.png",
        f"Predicted against actual {REG['natural_quantity']}, all folds",
        f"Actual over predicted {REG['natural_quantity']} by bin, all folds",
        n_bins=CALIBRATION_BINS, quantity=REG["natural_quantity"],
    )
    calibration.to_csv(REG_RESULTS_DIR / f"calibration_{scale}_pooled.csv", index=False)
    reporting.write_train_log_share_calibration_pooled(fold_dirs, REG_RESULTS_DIR, quantity=REG["quantity"])

    print("\n=== fold summary, val cells with any conversion (positive rate 100% by construction) ===")
    print(f"errors in natural log of {REG['quantity']}. constant_ and ols_scalars_ are the baselines")
    reporting.write_regression_summary(
        all_fold_rows, REG_RESULTS_DIR / "fold_summary.csv",
        base=("mse", "constant_mse", "ols_scalars_mse", "mae", "ols_scalars_mae", "r2",
              "ols_scalars_r2", "spearman", "ols_scalars_spearman", "pred_over_actual",
              "ols_scalars_pred_over_actual", "smear", "n_cells"),
    )

    log_table = reporting.write_metric_table(all_log_rows, REG_RESULTS_DIR / "metrics_log.csv")
    natural_table = reporting.write_metric_table(all_natural_rows, REG_RESULTS_DIR / f"metrics_{scale}.csv")
    reporting.print_metric_table(log_table, f"val, natural log of {REG['quantity']}")
    reporting.print_metric_table(
        natural_table,
        f"val, {REG['natural_quantity']} in {REG['units']}. mae and median_ae on exp(pred), "
        f"the rest on exp(pred) * smear clipped at {REG['cap']}",
    )

    best_fold = min(all_fold_rows, key=lambda row: row["mse"])
    shutil.copy2(Path(best_fold["checkpoint_path"]), REG_RESULTS_DIR / "best_model.ckpt")


def main():
    parser = argparse.ArgumentParser(description="Train one stage of the cell hurdle model.")
    parser.add_argument("stage", choices=("classify", "regress"),
                        help="classify: any conversion in the cell. regress: log percent "
                             "area or log share converted (REG_TARGET), on converting cells only.")
    args = parser.parse_args()

    pl.seed_everything(SEED, workers=True)
    cells, cell_manifest, fold_code, provenance = load_checked_cells()
    if args.stage == "classify":
        run_classify(cells, cell_manifest, fold_code, provenance)
    else:
        run_regress(cells, cell_manifest, fold_code, provenance)

if __name__ == "__main__":
    main()
