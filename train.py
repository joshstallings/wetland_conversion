"""
Training entry point for the streaming pipeline (data_pipeline_recipe.md):
train and val both stream from the source parquet, fold logging/model/reporting
are common code the loop below calls into.

Run once with `python population_stats.py` first to build
data/population_stats.json.
"""

import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import datasets
import models
import reporting
from features import COLS_TO_NORMALIZE, FEATURE_COLS, LABEL_COL
from folds import assign_folds, log_fold_stats
from population_stats import POPULATION_STATS_PATH, load_population_stats

SOURCE_PARQUET_PATH = "data/alphaearth_wetland_joined"

SEED = 0
N_SPLITS = 5

RESULTS_DIR = Path("results/streaming")
MAX_EPOCHS = 15
BATCH_SIZE = 2048
LR = 8e-3
BUFFER_SIZE = 300_000


def build_data_module(fold_idx):
    return datasets.StreamingDataModule(
        seed=SEED, n_splits=N_SPLITS, fold_idx=fold_idx,
        feature_cols=FEATURE_COLS, label_col=LABEL_COL, cols_to_normalize=COLS_TO_NORMALIZE,
        source_parquet_path=SOURCE_PARQUET_PATH, population_stats_path=POPULATION_STATS_PATH,
        batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE,
    )


def run_fold(fold_idx, pos_weight, fold_dir):
    """Trains one fold, writes its plots and val_preds.npz into fold_dir,
    returns (score_dict, data_module) -- the data_module is returned so main()
    can log its block counts into the manifest without recomputing them."""
    data_module = build_data_module(fold_idx)

    model = models.SimpleLinearModel(len(FEATURE_COLS), lr=LR, pos_weight=pos_weight, gamma=2.0)

    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_name = f"fold_{fold_idx}"
    logger = CSVLogger(save_dir=str(fold_dir.parent), name=fold_name)
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=1e-4)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[early_stop, checkpoint],
        logger=logger,
        enable_progress_bar=True,
        accelerator="auto",
    )

    trainer.fit(model, datamodule=data_module)
    epochs_run = trainer.current_epoch + 1

    best_model = models.SimpleLinearModel.load_from_checkpoint(checkpoint.best_model_path)

    probs, labels = reporting.get_val_predictions(best_model, data_module.val_dataloader())
    np.savez(fold_dir / "val_preds.npz", probs=probs, labels=labels)

    # Same model, scored on its own training blocks at their true positive rate
    # (train_eval_dataloader, not train_dataloader -- see its docstring) so the
    # train and val PR curves land on the same scale and the gap between them
    # is a real overfit signal, not an artifact of a rebalanced sample.
    train_probs, train_labels = reporting.get_val_predictions(best_model, data_module.train_eval_dataloader())
    np.savez(fold_dir / "train_preds.npz", probs=train_probs, labels=train_labels)

    metrics_csvs = sorted(fold_dir.glob("**/metrics.csv"))
    if metrics_csvs:
        reporting.plot_loss_curve(
            metrics_csvs[-1], fold_dir / "loss_curve.png",
            f"fold {fold_idx}: training loss",
        )

    reporting.plot_confusion_matrix(
        labels, probs, threshold=0.5, out_path=fold_dir / "confusion_matrix.png",
        title=f"fold {fold_idx}: confusion matrix",
    )
    reporting.plot_pr_curve_train_val(
        train_labels, train_probs, labels, probs, out_path=fold_dir / "pr_curve.png",
        title=f"fold {fold_idx}: train vs validation precision-recall curve",
    )

    fold_score = reporting.score_fold(labels, probs)
    train_score = reporting.score_fold(train_labels, train_probs)
    fold_score.update({f"train_{k}": v for k, v in train_score.items()})
    fold_score.update({"fold": fold_idx, "epochs_run": epochs_run})
    return fold_score, data_module


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    population_stats = load_population_stats(POPULATION_STATS_PATH)

    # Fixed constant for this dataset -- true_negatives / true_positives from
    # the full population scan, passed straight to the loss (pos_weight).
    pos_weight = population_stats["total_negative"] / population_stats["total_positive"]
    print(f"pos_weight = {pos_weight:.1f} (from true population counts, "
          f"{population_stats['total_negative']:,} neg / {population_stats['total_positive']:,} pos)")

    # This assign_folds call is only for the pre-training log below -- each
    # fold's DataModule.setup() calls it again independently, which is fine
    # since it's a cheap in-memory computation over ~1700 blocks, not a data
    # scan.
    fold_assignment = assign_folds(population_stats["block_row_counts"], N_SPLITS, SEED)
    print(f"=== seed {SEED} ===")
    log_fold_stats(SOURCE_PARQUET_PATH, fold_assignment, N_SPLITS)

    manifest = {"seed": SEED, "n_splits": N_SPLITS, "max_epochs": MAX_EPOCHS, "folds": {}}
    all_fold_rows = []
    for fold_idx in range(N_SPLITS):
        print(f"\n--- fold {fold_idx} ---")
        fold_dir = RESULTS_DIR / f"fold_{fold_idx}"
        fold_score, data_module = run_fold(fold_idx, pos_weight, fold_dir)
        all_fold_rows.append(fold_score)
        manifest["folds"][fold_idx] = {
            "n_train_blocks": len(data_module.train_block_ids),
            "n_val_blocks": len(data_module.val_block_ids),
            "epochs_run": fold_score["epochs_run"],
            "positive_rate": fold_score["positive_rate"],
        }

    with open(RESULTS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== fold summary ===")
    reporting.write_fold_summary(all_fold_rows, RESULTS_DIR / "fold_summary.csv")


if __name__ == "__main__":
    main()
