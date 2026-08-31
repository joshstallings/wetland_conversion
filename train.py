"""
Shared training entry point for both pipelines
(data_pipeline_recipe_path_a.md, data_pipeline_recipe_path_b.md): --pipeline
selects which DataModule gets built and where results land, everything else
(fold logging, model, reporting) is common code both branches call into.

Run once with `python population_stats.py` first to build
data/population_stats.json. The cached_pool branch additionally needs
`python build_train_pool_cache.py` first to build data/train_pool_cache/.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import cached_pool_datasets
import datasets
import models
import reporting
from features import COLS_TO_NORMALIZE, FEATURE_COLS, LABEL_COL
from folds import assign_folds, log_fold_stats
from population_stats import POPULATION_STATS_PATH, load_population_stats

SOURCE_PARQUET_PATH = "data/alphaearth_wetland_joined"
CACHE_DIR = "data/train_pool_cache"

SEED = 0
N_SPLITS = 5

# Per pipeline knobs -- streaming rescans/decompresses parquet every epoch so
# epochs cost real wall clock, hence fewer of them at a bigger batch size;
# cached_pool's pool is a small in-RAM array, so more epochs are cheap.
PIPELINE_CONFIGS = {
    "streaming": {
        "results_dir": Path("results/streaming"),
        "max_epochs": 15,
        "batch_size": 2048,
        "lr": 8e-3,
    },
    "cached_pool": {
        "results_dir": Path("results/cached_pool"),
        "max_epochs": 15,
        "batch_size": 2048,
        "lr": 8e-3,
        "pos_frac": 0.25,
    },
}


def build_data_module(pipeline, config, fold_idx):
    if pipeline == "streaming":
        return datasets.StreamingDataModule(
            seed=SEED, n_splits=N_SPLITS, fold_idx=fold_idx,
            feature_cols=FEATURE_COLS, label_col=LABEL_COL, cols_to_normalize=COLS_TO_NORMALIZE,
            source_parquet_path=SOURCE_PARQUET_PATH, population_stats_path=POPULATION_STATS_PATH,
            batch_size=config["batch_size"], buffer_size=300_000,
        )
    if pipeline == "cached_pool":
        return cached_pool_datasets.CachedPoolDataModule(
            seed=SEED, n_splits=N_SPLITS, fold_idx=fold_idx,
            feature_cols=FEATURE_COLS, label_col=LABEL_COL, cols_to_normalize=COLS_TO_NORMALIZE,
            cache_dir=CACHE_DIR, source_parquet_path=SOURCE_PARQUET_PATH,
            population_stats_path=POPULATION_STATS_PATH,
            batch_size=config["batch_size"], pos_frac=config["pos_frac"],
        )
    raise ValueError(f"unknown pipeline {pipeline!r}")


def run_fold(pipeline, config, fold_idx, pos_weight, cache_manifest, fold_dir):
    """Trains one fold, writes its plots and val_preds.npz into fold_dir,
    returns (score_dict, data_module) -- the data_module is returned so main()
    can log its block counts into the manifest without recomputing them."""
    data_module = build_data_module(pipeline, config, fold_idx)

    # streaming rebalances positives in the loss (pos_weight); cached_pool
    # rebalances in the batch (StratifiedBatchSampler), so its loss stays
    # unweighted or the two corrections would compound.
    model_pos_weight = pos_weight if pipeline == "streaming" else None
    model = models.SimpleLinearModel(len(FEATURE_COLS), lr=config["lr"], pos_weight=model_pos_weight)

    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_name = f"fold_{fold_idx}"
    logger = CSVLogger(save_dir=str(fold_dir.parent), name=fold_name)
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=1e-4)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        callbacks=[early_stop, checkpoint],
        logger=logger,
        enable_progress_bar=True,
        accelerator="auto",
    )

    trainer.fit(model, datamodule=data_module)
    epochs_run = trainer.current_epoch + 1

    best_model = models.SimpleLinearModel.load_from_checkpoint(checkpoint.best_model_path)

    if pipeline == "cached_pool":
        # cached_pool trained on an artificially balanced sample, so its raw
        # probabilities are calibrated to that sample's rate, not the true
        # one -- correct before scoring, since reporting.score_fold reads
        # probs at a fixed 0.5 threshold.
        row_counts = cache_manifest["row_counts"]
        models.apply_intercept_correction(
            best_model,
            true_negative=row_counts["true_total_negative_population"],
            true_positive=row_counts["true_total_positive_population"],
            sample_negative=row_counts["negatives_kept"],
            sample_positive=row_counts["positives_kept"],
        )

    probs, labels = reporting.get_val_predictions(best_model, data_module.val_dataloader())
    np.savez(fold_dir / "val_preds.npz", probs=probs, labels=labels)

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
    reporting.plot_pr_curve(
        labels, probs, out_path=fold_dir / "pr_curve.png",
        title=f"fold {fold_idx}: precision-recall curve",
    )

    fold_score = reporting.score_fold(labels, probs)
    fold_score.update({"fold": fold_idx, "epochs_run": epochs_run})
    return fold_score, data_module


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", choices=list(PIPELINE_CONFIGS), default="streaming")
    return p.parse_args()


def main():
    args = parse_args()
    pipeline = args.pipeline
    config = PIPELINE_CONFIGS[pipeline]
    results_dir = config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    population_stats = load_population_stats(POPULATION_STATS_PATH)

    # Fixed constant for this dataset -- true_negatives / true_positives from
    # the full population scan. Only reaches the loss under the streaming
    # branch (see run_fold).
    pos_weight = population_stats["total_negative"] / population_stats["total_positive"]
    print(f"pipeline = {pipeline!r}")
    print(f"pos_weight = {pos_weight:.1f} (from true population counts, "
          f"{population_stats['total_negative']:,} neg / {population_stats['total_positive']:,} pos)")

    cache_manifest = None
    if pipeline == "cached_pool":
        with open(Path(CACHE_DIR) / "manifest.json") as f:
            cache_manifest = json.load(f)

    # This assign_folds call is only for the pre-training log below -- each
    # fold's DataModule.setup() calls it again independently, which is fine
    # since it's a cheap in-memory computation over ~1700 blocks, not a data
    # scan.
    fold_assignment = assign_folds(population_stats["block_row_counts"], N_SPLITS, SEED)
    print(f"=== seed {SEED} ===")
    log_fold_stats(SOURCE_PARQUET_PATH, fold_assignment, N_SPLITS)

    manifest = {
        "pipeline": pipeline, "seed": SEED, "n_splits": N_SPLITS,
        "max_epochs": config["max_epochs"], "folds": {},
    }
    all_fold_rows = []
    for fold_idx in range(N_SPLITS):
        print(f"\n--- fold {fold_idx} ---")
        fold_dir = results_dir / f"fold_{fold_idx}"
        fold_score, data_module = run_fold(pipeline, config, fold_idx, pos_weight, cache_manifest, fold_dir)
        all_fold_rows.append(fold_score)
        manifest["folds"][fold_idx] = {
            "n_train_blocks": len(data_module.train_block_ids),
            "n_val_blocks": len(data_module.val_block_ids),
            "epochs_run": fold_score["epochs_run"],
            "positive_rate": fold_score["positive_rate"],
        }

    with open(results_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== fold summary ===")
    reporting.write_fold_summary(all_fold_rows, results_dir / "fold_summary.csv")


if __name__ == "__main__":
    main()
