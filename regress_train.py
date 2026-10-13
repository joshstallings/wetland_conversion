"""
Training entry point for the cell task: spatially blocked CV of
models.CellTemporalConvolutionalNetwork over a build_cells.py artifact,
predicting the share of each cell's 2019 wetland developed by 2024.

Run order from scratch:
    python population_stats.py              # fold assignment comes off its block counts
    python build_arrays.py                  # data/arrays_2019_2024
    python build_cells.py --cell-size-m 300 # data/cells_2019_2024_300m
    python regress_train.py

Folds are the classify_train.py folds: same assign_folds call on the same
population stats with the same seed, and every cell inherits the fold of its
10 km block. So a fold here holds out the same ground as the pixel run with the
same SEED and N_SPLITS.

No early stopping and no best epoch pick. Every fold trains MAX_EPOCHS and the
final epoch's weights are what get scored. Choosing the epoch by val_loss would
let the val fold pick the model, and the val numbers would then be slightly
optimistic.
"""

import json
from pathlib import Path
import shutil

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import arrays as array_io
import datasets
import models
import reporting
from features import YEARS_2017_2019
from folds import assign_folds
from population_stats import load_population_stats

CELL_DIR = "data/cells_2019_2024_120m"
POPULATION_STATS_PATH = "data/population_stats_2019_2024.json"
N_YEARS = len(YEARS_2017_2019)

SEED = 0
N_SPLITS = 5

MAX_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4

# False: every cell counts once in the loss, including the 1 pixel cells whose
# target can only be 0 or 1. val_mae_weighted is logged either way.
WEIGHT_BY_N_WETLAND = True

# True: BCE on the logit with the share as a soft target, which is the binomial
# log likelihood once WEIGHT_BY_N_WETLAND is on. False: MSE on the sigmoid output.
BINOMIAL_LOSS = False
LOSS_NAME = "binomial log loss" if BINOMIAL_LOSS else "MSE"

# The two MSE runs keep the directory names they were written under.
RESULTS_DIR = Path(
    "results/cell_tcn_120m"
)

BATCH_SIZE = 1024
VAL_BATCH_SIZE = 8192

# 1 keeps every train cell. Val always keeps every cell regardless.
MIN_WETLAND_TRAIN = 1

NUM_WORKERS = 0


def run_fold(fold_idx, data_module, fold_dir):
    """
    Trains one fold for MAX_EPOCHS, writes its plots and predictions into
    fold_dir, returns a score dict. data_module must already be set up.
    """
    model = models.CellTemporalConvolutionalNetwork(
        n_years=N_YEARS, lr=LR, weight_decay=WEIGHT_DECAY,
        weight_by_n_wetland=WEIGHT_BY_N_WETLAND, binomial_loss=BINOMIAL_LOSS,
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

    final_model = models.CellTemporalConvolutionalNetwork.load_from_checkpoint(checkpoint.best_model_path)

    pred, target, n_wet = reporting.get_cell_predictions(final_model, data_module.val_dataloader())
    # cell_idx indexes the cell artifact's arrays (cell_xy, block_code and so on),
    # so a fold's predictions can be mapped without rebuilding the split.
    np.savez(fold_dir / "val_preds.npz", pred=pred, target=target, n_wetland=n_wet,
             cell_idx=data_module.val_idx)

    # Every train cell scored by the same final model, so train and val sit on the
    # same population and the gap between them is a real overfit signal.
    train_pred, train_target, train_n_wet = reporting.get_cell_predictions(
        final_model, data_module.train_eval_dataloader()
    )
    np.savez(fold_dir / "train_preds.npz", pred=train_pred, target=train_target,
             n_wetland=train_n_wet, cell_idx=data_module.train_idx)

    metrics_csvs = sorted(fold_dir.glob("**/metrics.csv"))
    if metrics_csvs:
        reporting.plot_loss_curve(
            metrics_csvs[-1], fold_dir / "loss_curve.png",
            f"Fold {fold_idx}: training and validation loss", ylabel=LOSS_NAME,
        )
        reporting.plot_cell_val_metrics(metrics_csvs[-1:], fold_dir / "val_metrics.png",
                                        title_suffix=f", fold {fold_idx}")

    reporting.plot_pr_curve_train_val(
        train_target > 0, train_pred, target > 0, pred, out_path=fold_dir / "pr_curve.png",
        title=f"Fold {fold_idx}: precision recall, any conversion in the cell",
    )
    calibration = reporting.plot_cell_calibration(
        pred, target, fold_dir / "calibration.png", fold_dir / "calibration_ratio.png",
        f"Fold {fold_idx}: predicted against actual share converted",
        f"Fold {fold_idx}: actual over predicted share converted, by bin",
    )
    calibration.to_csv(fold_dir / "calibration.csv", index=False)

    fold_score = reporting.score_cells(pred, target, n_wet)
    train_score = reporting.score_cells(train_pred, train_target, train_n_wet)
    fold_score.update({f"train_{k}": v for k, v in train_score.items()})
    fold_score.update({
        "fold": fold_idx,
        "epochs_run": epochs_run,
        "checkpoint_path": checkpoint.best_model_path,
        "n_train_cells": int(data_module.train_idx.size),
        "n_val_cells": int(data_module.val_idx.size),
        "weight_by_n_wetland": WEIGHT_BY_N_WETLAND,
        "binomial_loss": BINOMIAL_LOSS,
        "min_wetland_train": MIN_WETLAND_TRAIN,
        "scalar_mean": data_module.scalar_mean.tolist(),
        "scalar_std": data_module.scalar_std.tolist(),
    })
    return fold_score


def main():
    pl.seed_everything(SEED, workers=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cells, cell_manifest = datasets.load_cells(CELL_DIR)
    src = cell_manifest["source_manifest"]
    stats = cell_manifest["stats"]
    print(
        f"cells: {stats['n_cells']:,} at {cell_manifest['cell_size_m']} m, built "
        f"{cell_manifest['built_utc']} at commit {(cell_manifest['git_commit'] or '?')[:9]}, "
        f"from {cell_manifest['source_array_dir']} built {src['built_utc']}"
    )
    print(
        f"{stats['share_cells_with_any_developed']:.2%} of cells have any conversion, "
        f"mean target {stats['mean_target_unweighted']:.3%}, pixel rate {stats['pixel_positive_rate']:.3%}"
    )

    population_stats = load_population_stats(POPULATION_STATS_PATH)
    # The same three checks classify_train.py runs against the arrays, here against
    # the arrays the cells were built from. Row count alone does not pin the label
    # horizon, and positives alone do not tell 2019_2024 from 2022_2024.
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

    run_manifest = {
        "task": "share of 2019 wetland developed by 2024, per cell",
        "model": "CellTemporalConvolutionalNetwork",
        "loss": "BCE on logit, share as soft target" if BINOMIAL_LOSS else "MSE on sigmoid output",
        "binomial_loss": BINOMIAL_LOSS,
        "seed": SEED,
        "n_splits": N_SPLITS,
        "max_epochs": MAX_EPOCHS,
        "checkpoint_scored": "final epoch, no early stopping",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "weight_by_n_wetland": WEIGHT_BY_N_WETLAND,
        "batch_size": BATCH_SIZE,
        "min_wetland_train": MIN_WETLAND_TRAIN,
        "scalars": list(datasets.CELL_SCALARS),
        # Which artifact this result came from. Cell size changes the positive rate
        # from 0.82% at 90 m to 10.87% at 990 m, so a number without it is not
        # interpretable, let alone reproducible.
        "cells": {
            "path": str(CELL_DIR),
            "cell_size_m": cell_manifest["cell_size_m"],
            "built_utc": cell_manifest["built_utc"],
            "git_commit": cell_manifest["git_commit"],
            "n_cells": stats["n_cells"],
            "share_cells_with_any_developed": stats["share_cells_with_any_developed"],
            "source_arrays": src,
        },
        "folds": {},
    }

    all_fold_rows = []
    for fold_idx in range(N_SPLITS):
        print(f"\n--- fold {fold_idx} ---")
        data_module = datasets.CellDataModule(
            cells, cell_manifest, fold_code, fold_idx,
            batch_size=BATCH_SIZE, val_batch_size=VAL_BATCH_SIZE,
            min_wetland_train=MIN_WETLAND_TRAIN, num_workers=NUM_WORKERS, seed=SEED,
        )
        data_module.setup()
        print(f"batches: {len(data_module.train_sampler):,} per epoch at {BATCH_SIZE} cells. "
              f"loss: {LOSS_NAME}, weight_by_n_wetland {WEIGHT_BY_N_WETLAND}")

        fold_score = run_fold(fold_idx, data_module, RESULTS_DIR / f"fold_{fold_idx}")
        all_fold_rows.append(fold_score)
        run_manifest["folds"][fold_idx] = {
            "n_train_cells": fold_score["n_train_cells"],
            "n_val_cells": fold_score["n_val_cells"],
            "val_positive_rate": fold_score["positive_rate"],
            "train_positive_rate": fold_score["train_positive_rate"],
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
        reporting.plot_cell_val_metrics(metrics_csvs, RESULTS_DIR / "val_metrics_mean.png",
                                        title_suffix=f", {N_SPLITS} fold mean")

    fold_preds, val_pred_all, val_target_all = [], [], []
    for d in fold_dirs:
        tr, va = np.load(d / "train_preds.npz"), np.load(d / "val_preds.npz")
        fold_preds.append((tr["target"] > 0, tr["pred"], va["target"] > 0, va["pred"]))
        val_pred_all.append(va["pred"])
        val_target_all.append(va["target"])
    reporting.plot_mean_pr_curve(
        fold_preds, RESULTS_DIR / "pr_curve_mean.png",
        f"Mean precision recall over {N_SPLITS} folds, any conversion in the cell",
        chance_unit="cells",
    )
    # The folds partition the cells, so pooling their val predictions scores
    # every cell exactly once.
    calibration = reporting.plot_cell_calibration(
        np.concatenate(val_pred_all), np.concatenate(val_target_all),
        RESULTS_DIR / "calibration_pooled.png", RESULTS_DIR / "calibration_pooled_ratio.png",
        "Predicted against actual share converted, all folds",
        "Actual over predicted share converted by bin, all folds",
    )
    calibration.to_csv(RESULTS_DIR / "calibration_pooled.csv", index=False)
    reporting.write_train_calibration_pooled(fold_dirs, RESULTS_DIR)

    print("\n=== fold summary ===")
    reporting.write_regression_summary(all_fold_rows, RESULTS_DIR / "fold_summary.csv")

    # Same convention as classify_train.py, lowest val MSE instead of highest AUPRC.
    best_fold = min(all_fold_rows, key=lambda row: row["mse"])
    shutil.copy2(Path(best_fold["checkpoint_path"]), RESULTS_DIR / "best_model.ckpt")


if __name__ == "__main__":
    main()
