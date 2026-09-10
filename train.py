"""
Training entry point for the array pipeline: 5 fold spatially blocked CV of a
linear baseline over data/arrays.

Run order from scratch:
    python population_stats.py    # data/population_stats.json, the independent record
    python build_arrays.py        # data/arrays, about 10 minutes, verifies itself
    python train.py
"""

import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import arrays as array_io
import datasets
import models
import reporting
from features import FEATURE_COLS
from folds import assign_folds, log_fold_stats
from population_stats import POPULATION_STATS_PATH, load_population_stats

ARRAY_DIR = array_io.ARRAY_DIR
RESULTS_DIR = Path("results/mlp_batch_size_256")

SEED = 0
N_SPLITS = 5

MAX_EPOCHS = 15
LR = 1e-3

# Focal loss is the default. GAMMA=None falls back to nn.BCEWithLogitsLoss.
GAMMA = 2.0

# No pos_weight: focal loss already down weights the easy negatives and
# POS_PER_BATCH is a second mechanism on the same imbalance. Set this to a fold's
# data_module.batch_pos_weight (31) only as a deliberate ablation.
POS_WEIGHT = None

BATCH_SIZE = 256
# 64 in 2048 is a 1 in 32 positive rate, against 1 in 356 in the population. This
# is the number to move first if the model is starved of positives or, in the
# other direction, if it starts overfitting the 165,908 positives it now sees
# many times per epoch.
POS_PER_BATCH = 8

# None means one pass over this fold's train positives per epoch, about 2,100
# batches. An epoch is a choice now, not a pass over the data.
STEPS_PER_EPOCH = None

# Val is a sequential sweep over contiguous block spans, so a bigger read is
# cheaper here than in the scattered train batches.
VAL_BATCH_ROWS = 8192

# Score the fitted model on every 10th train row: 4.7M rows instead of 47M, at
# the same natural positive rate, which is what makes the train AUPRC comparable
# to val.
TRAIN_EVAL_STRIDE = 10

# The linear model is 0.57 ms per batch against a 0.62 ms data path, so one
# process keeps up. Raising this only earns its keep with a real model on a GPU,
# and every worker pays its own 1.4 second populate pass.
NUM_WORKERS = 0


def run_fold(fold_idx, data_module, fold_dir):
    """
    Trains one fold, writes its plots and predictions into fold_dir, returns a
    score dict. data_module must already be set up, since run_fold reads its fold
    index and normalization stats.
    """
    index = data_module.index
    model = models.MultiLayerPerceptron(
        len(FEATURE_COLS), lr=LR, pos_weight=POS_WEIGHT, gamma=GAMMA
    )

    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_name = f"fold_{fold_idx}"
    logger = CSVLogger(save_dir=str(fold_dir.parent), name=fold_name)
    # early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=1e-4)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint],
        logger=logger,
        enable_progress_bar=True,
        accelerator="auto",
    )

    trainer.fit(model, datamodule=data_module)
    # No + 1: Lightning increments current_epoch at the end of each epoch, so it
    # already equals the number run, whether fit ended on max_epochs or on
    # EarlyStopping.
    epochs_run = trainer.current_epoch

    best_model = models.MultiLayerPerceptron.load_from_checkpoint(checkpoint.best_model_path)

    probs, labels = reporting.get_val_predictions(best_model, data_module.val_dataloader())
    np.savez(fold_dir / "val_preds.npz", probs=probs, labels=labels)

    # Same model, scored on its own training blocks at their true positive rate
    # (train_eval_dataloader, not train_dataloader -- see its docstring) so the
    # train and val PR curves land on the same scale and the gap between them is a
    # real overfit signal, not an artifact of the stratified batch composition.
    train_probs, train_labels = reporting.get_val_predictions(
        best_model, data_module.train_eval_dataloader()
    )
    np.savez(fold_dir / "train_preds.npz", probs=train_probs, labels=train_labels)

    metrics_csvs = sorted(fold_dir.glob("**/metrics.csv"))
    if metrics_csvs:
        reporting.plot_loss_curve(
            metrics_csvs[-1], fold_dir / "loss_curve.png", f"fold {fold_idx}: training loss",
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
    fold_score.update({
        "fold": fold_idx,
        "epochs_run": epochs_run,
        "n_train_blocks": index.n_train_blocks,
        "n_val_blocks": index.n_val_blocks,
        "n_train_rows": index.n_train_rows,
        "n_val_rows": index.n_val_rows,
        "train_positives": int(index.train_pos.size),
        "gamma": GAMMA,
        "pos_weight": POS_WEIGHT,
        # Recorded even when unused: it is the value an ablation should pass.
        "batch_pos_weight_available": data_module.batch_pos_weight,
        "dist_mean_m": float(data_module.dist_mean),
        "dist_std_m": float(data_module.dist_std),
    })
    return fold_score


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Opened once for the whole run, not per fold. populate() is the difference
    # between 0.62 ms and 5.65 ms per batch: see the arrays module docstring.
    arrays, manifest = array_io.load_arrays(ARRAY_DIR)
    array_io.populate(arrays)
    print(
        f"arrays: {manifest['total_rows']:,} rows, {manifest['n_blocks']} blocks, built "
        f"{manifest['built_utc']} at commit {(manifest['git_commit'] or '?')[:9]}"
    )
    raw_counts = manifest["label_semantics"]["raw_class_counts"]
    print(
        f"source label classes: {raw_counts[0]:,} remained wetland, {raw_counts[1]:,} "
        f"converted to developed, {raw_counts[2]:,} converted to other. Stored binary as "
        f"label == 1, so the target is conversion specifically to developed land."
    )

    population_stats = load_population_stats(POPULATION_STATS_PATH)
    if population_stats["total_rows"] != manifest["total_rows"]:
        raise AssertionError(
            f"population_stats.json has {population_stats['total_rows']:,} rows against "
            f"the artifact's {manifest['total_rows']:,}. One of them is stale."
        )

    fold_assignment = assign_folds(population_stats["block_row_counts"], N_SPLITS, SEED)
    fold_code = array_io.fold_of_code(manifest, fold_assignment)
    print(f"\n=== seed {SEED}, {N_SPLITS} folds ===")
    log_fold_stats(
        arrays["label"], array_io.fold_of_row(arrays, fold_code), fold_assignment, N_SPLITS
    )

    run_manifest = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "max_epochs": MAX_EPOCHS,
        "lr": LR,
        "gamma": GAMMA,
        "pos_weight": POS_WEIGHT,
        "batch_size": BATCH_SIZE,
        "pos_per_batch": POS_PER_BATCH,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "train_eval_stride": TRAIN_EVAL_STRIDE,
        # Which artifact this result came from. A number without this is not
        # reproducible: the arrays are as much an input as the parquet, and they
        # get rebuilt.
        "arrays": {
            "path": str(ARRAY_DIR),
            "built_utc": manifest["built_utc"],
            "git_commit": manifest["git_commit"],
            "source_parquet_path": manifest["source_parquet_path"],
            "total_rows": manifest["total_rows"],
            "n_blocks": manifest["n_blocks"],
            "raw_class_counts": raw_counts,
            "emb_dtype": manifest["arrays"]["emb"]["dtype"],
            "float16_cast": manifest["float16_cast"],
        },
        "folds": {},
    }

    all_fold_rows = []
    for fold_idx in range(N_SPLITS):
        print(f"\n--- fold {fold_idx} ---")
        data_module = datasets.ArrayDataModule(
            arrays, manifest, fold_code, fold_idx,
            batch_size=BATCH_SIZE, pos_per_batch=POS_PER_BATCH,
            steps_per_epoch=STEPS_PER_EPOCH, val_batch_rows=VAL_BATCH_ROWS,
            train_eval_stride=TRAIN_EVAL_STRIDE, num_workers=NUM_WORKERS, seed=SEED,
        )
        # Called here rather than left to Trainer.fit because pos_weight comes off
        # the batch composition, which is not known until the sampler exists.
        data_module.setup()

        index = data_module.index
        n_val_pos = array_io.val_positive_count(arrays, index)
        print(
            f"train {index.n_train_rows:,} rows in {index.n_train_blocks} blocks "
            f"({index.train_pos.size:,} positives), val {index.n_val_rows:,} rows in "
            f"{index.n_val_blocks} blocks ({n_val_pos:,} positives, "
            f"{n_val_pos / index.n_val_rows:.4%})"
        )
        print(
            f"batches: {len(data_module.train_sampler):,} per epoch at {BATCH_SIZE} rows, "
            f"{data_module.train_sampler.batch_positive_rate:.2%} positive, "
            f"against {index.train_pos.size / index.n_train_rows:.4%} in the fold. "
            f"loss: focal gamma {GAMMA}, pos_weight {POS_WEIGHT}"
        )
        print(f"dist normalization from train blocks only: mean {data_module.dist_mean:.1f} m, "
              f"std {data_module.dist_std:.1f} m")

        fold_score = run_fold(fold_idx, data_module, RESULTS_DIR / f"fold_{fold_idx}")
        all_fold_rows.append(fold_score)
        run_manifest["folds"][fold_idx] = {
            "n_train_blocks": index.n_train_blocks,
            "n_val_blocks": index.n_val_blocks,
            "n_train_rows": index.n_train_rows,
            "n_val_rows": index.n_val_rows,
            "train_positives": int(index.train_pos.size),
            "val_positives": n_val_pos,
            "epochs_run": fold_score["epochs_run"],
            "positive_rate": fold_score["positive_rate"],
            "dist_mean_m": fold_score["dist_mean_m"],
            "dist_std_m": fold_score["dist_std_m"],
        }

    with open(RESULTS_DIR / "manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    print("\n=== fold summary ===")
    reporting.write_fold_summary(all_fold_rows, RESULTS_DIR / "fold_summary.csv")


if __name__ == "__main__":
    main()
