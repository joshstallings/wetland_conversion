"""
Does the 300 m cell classifier need all 64 AlphaEarth dimensions, and how much
of it is distance to developed land? Three runs of
models.CellMultiLayerPerceptronClassifier on input subsets:

    python train_on_top_dims.py                 # the three below
    python train_on_top_dims.py all_dims_no_scalars top10_no_scalars

    all_dims_no_scalars    all 64 dims, no cell scalars
    top10_no_scalars       the 10 top dims, no cell scalars
    top10_with_distance    the 10 top dims plus log1p dist_mean and dist_min
    full_reference         all 64 dims plus all three scalars, the stock model.
                           Not run unless named, for a baseline on the current
                           cell artifact.

TOP_DIMS is the 300 m top 10 from notebooks/model_weight_analysis.ipynb. Each
dim keeps both its mean_AXX and std_AXX column for every year, so a dim is 6
inputs and the top 10 is 60. log1p_n_wetland is left out of all three runs:
it is cell size, not distance, and the point is what the embeddings carry on
their own.

The top 10 came from models trained on these same 5 folds. If the ranking used
permutation drops on the val cells, the val folds helped pick the inputs, so the
top 10 runs are slightly optimistic against all_dims_no_scalars.

Everything else matches results/hurdle_clf_mlp_300m and the weight analysis run
the top 10 came from: 5 spatially blocked folds (seed 0), focal loss at gamma 2,
no pos_weight, natural positive rate batches of 1,024, 20 epochs, final epoch
scored, (256, 64) head.

A condition whose fold_summary.csv already exists is skipped. Delete that
directory to rerun it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import datasets
import hurdle_model_train as hmt
import models
import reporting

RESULTS_DIR = Path("results/model_trained_on_most_important_weights")

# load_checked_cells reads these module globals when it runs, so pinning them
# here keeps a flip of FOLD_SCHEME or CELL_DIR in hurdle_model_train.py from
# moving this run.
hmt.FOLD_SCHEME = "blocked"
hmt.CELL_DIR = "data/cells_2019_2024_300m"
N_SPLITS = hmt.BLOCKED_N_SPLITS

TOP_DIMS = ("A36", "A18", "A39", "A49", "A23", "A05", "A35", "A07", "A10", "A24")

# Indices into datasets.CELL_SCALARS: log1p_dist_mean, log1p_dist_min, log1p_n_wetland.
DIST_COLS = (0, 1)
ALL_SCALAR_COLS = (0, 1, 2)

# dims None means all 64.
CONDITIONS = {
    "all_dims_no_scalars": {"dims": None, "scalar_cols": (),
                            "label": "all 64 AlphaEarth dims, no distance"},
    "top10_no_scalars": {"dims": TOP_DIMS, "scalar_cols": (),
                         "label": "top 10 AlphaEarth dims, no distance"},
    "top10_with_distance": {"dims": TOP_DIMS, "scalar_cols": DIST_COLS,
                            "label": "top 10 AlphaEarth dims plus distance"},
    "full_reference": {"dims": None, "scalar_cols": ALL_SCALAR_COLS,
                       "label": "all 64 dims, distance, wetland area"},
}
DEFAULT_CONDITIONS = ("all_dims_no_scalars", "top10_no_scalars", "top10_with_distance")

SEED = 0
MAX_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 0.0
DROPOUT = 0.3
HIDDEN_DIMS = (256, 64)
GAMMA = 2.0
ALPHA = None
POS_WEIGHT = None
BATCH_SIZE = 1024
POS_PER_BATCH = None
VAL_BATCH_SIZE = 8192
MIN_WETLAND_TRAIN = 1
RECALL_TARGET = 0.5
NUM_WORKERS = 0
LOSS_NAME = "focal loss"


def subset_embeddings(cells, cell_manifest, dims):
    """cells with emb_mean and emb_std cut to dims in every year, same year major
    order the model's view(n_years, emb_dim) expects. Columns are found by name
    off the manifest rather than by arithmetic, so a reordered build fails loudly."""
    if dims is None:
        return cells, 64
    cols = list(cell_manifest["emb_columns"])
    years = sorted({c.split("_")[1] for c in cols})
    if len(years) != hmt.N_YEARS:
        raise AssertionError(f"manifest has years {years}, the model expects {hmt.N_YEARS}")
    idx = np.array([cols.index(f"{d}_{y}") for y in years for d in dims])
    out = dict(cells)
    out["emb_mean"] = np.ascontiguousarray(cells["emb_mean"][:, idx])
    out["emb_std"] = np.ascontiguousarray(cells["emb_std"][:, idx])
    return out, len(dims)


def run_fold(cond, emb_dim, fold_idx, data_module, fold_dir):
    n_scalars = len(cond["scalar_cols"])
    model = models.CellMultiLayerPerceptronClassifier(
        emb_dim=emb_dim, n_years=hmt.N_YEARS, n_scalars=n_scalars, dropout=DROPOUT,
        lr=LR, weight_decay=WEIGHT_DECAY, gamma=GAMMA, alpha=ALPHA,
        pos_weight=POS_WEIGHT, hidden_dims=HIDDEN_DIMS,
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=str(fold_dir.parent), name=fold_dir.name)
    # monitor=None keeps only the most recent epoch, which after fit is the final one.
    checkpoint = ModelCheckpoint(monitor=None, save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS, callbacks=[checkpoint], logger=logger,
        enable_progress_bar=True, accelerator="auto",
    )
    trainer.fit(model, datamodule=data_module)

    final_model = models.CellMultiLayerPerceptronClassifier.load_from_checkpoint(
        checkpoint.best_model_path
    )
    prob, target, n_wet = reporting.get_cell_predictions(final_model, data_module.val_dataloader())
    labels = target > 0
    np.savez(fold_dir / "val_preds.npz", prob=prob, any=labels, target=target,
             n_wetland=n_wet, cell_idx=data_module.val_idx)
    train_prob, train_target, train_n_wet = reporting.get_cell_predictions(
        final_model, data_module.train_eval_dataloader()
    )
    train_labels = train_target > 0
    np.savez(fold_dir / "train_preds.npz", prob=train_prob, any=train_labels,
             target=train_target, n_wetland=train_n_wet, cell_idx=data_module.train_idx)

    threshold = reporting.threshold_at_recall(train_labels, train_prob, RECALL_TARGET)
    score = reporting.score_fold(labels, prob, threshold=threshold)
    train_score = reporting.score_fold(train_labels, train_prob, threshold=threshold)
    score.update({f"train_{k}": v for k, v in train_score.items()})
    score[f"precision_at_recall_{RECALL_TARGET}"] = reporting.precision_at_recall(
        labels, prob, RECALL_TARGET
    )
    score.update({
        "fold": fold_idx,
        "threshold": threshold,
        "epochs_run": trainer.current_epoch,
        "checkpoint_path": checkpoint.best_model_path,
        "n_params": sum(p.numel() for p in final_model.parameters()),
        "n_train_cells": int(data_module.train_idx.size),
        "n_val_cells": int(data_module.val_idx.size),
        "n_val_positive_cells": int(labels.sum()),
        "batch_positive_rate": data_module.batch_positive_rate,
    })
    return score


def run_condition(name, cells, cell_manifest, fold_code, provenance):
    cond = CONDITIONS[name]
    cond_dir = RESULTS_DIR / name
    cond_dir.mkdir(parents=True, exist_ok=True)
    sub_cells, emb_dim = subset_embeddings(cells, cell_manifest, cond["dims"])
    scalar_cols = list(cond["scalar_cols"])

    rows = []
    for fold_idx in range(N_SPLITS):
        print(f"\n--- {cond['label']}, fold {fold_idx} ---")
        # Reseeded per condition and fold so a result does not depend on which
        # conditions ran before it.
        pl.seed_everything(SEED, workers=True)
        data_module = datasets.CellDataModule(
            sub_cells, cell_manifest, fold_code, fold_idx,
            batch_size=BATCH_SIZE, val_batch_size=VAL_BATCH_SIZE,
            min_wetland_train=MIN_WETLAND_TRAIN, pos_per_batch=POS_PER_BATCH,
            num_workers=NUM_WORKERS, seed=SEED,
        )
        data_module.setup()
        # setup z scores all three scalars off the train cells. Dropping columns
        # afterwards leaves the kept ones scaled exactly as in the stock model.
        # setup returns early on the second call inside fit, so this sticks.
        data_module.ds.scalars = np.ascontiguousarray(data_module.ds.scalars[:, scalar_cols])
        rows.append(run_fold(cond, emb_dim, fold_idx, data_module, cond_dir / f"fold_{fold_idx}"))

    fold_dirs = [cond_dir / f"fold_{i}" for i in range(N_SPLITS)]
    metrics_csvs = [c for d in fold_dirs for c in sorted(d.glob("**/metrics.csv"))[-1:]]
    reporting.plot_mean_loss_curve(
        metrics_csvs, cond_dir / "loss_curve_mean.png",
        f"Loss, cell MLP on {cond['label']}", ylabel=LOSS_NAME,
    )
    fold_preds = []
    for d in fold_dirs:
        tr, va = np.load(d / "train_preds.npz"), np.load(d / "val_preds.npz")
        fold_preds.append((tr["any"], tr["prob"], va["any"], va["prob"]))
    reporting.plot_mean_pr_curve(
        fold_preds, cond_dir / "pr_curve_mean.png",
        f"Cell MLP on {cond['label']}",
        chance_unit="cells",
    )

    manifest = {
        "task": "any 2019 wetland developed by 2024, per cell (n_developed > 0)",
        "condition": name,
        "description": cond["label"],
        "alphaearth_dims": list(cond["dims"]) if cond["dims"] else "all 64",
        "emb_columns_per_dim": "mean and std, years 2017 to 2019",
        "n_emb_inputs": 2 * hmt.N_YEARS * emb_dim,
        "scalars": [datasets.CELL_SCALARS[i] for i in scalar_cols],
        "model": models.CellMultiLayerPerceptronClassifier.__name__,
        "hidden_dims": list(HIDDEN_DIMS),
        "n_params": rows[0]["n_params"],
        "dropout": DROPOUT,
        "loss": LOSS_NAME, "gamma": GAMMA, "alpha": ALPHA, "pos_weight": POS_WEIGHT,
        "seed": SEED,
        "folds_spec": hmt.FOLD_SPEC,
        "max_epochs": MAX_EPOCHS,
        "checkpoint_scored": "final epoch, no early stopping",
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE, "pos_per_batch": POS_PER_BATCH,
        "min_wetland_train": MIN_WETLAND_TRAIN,
        "recall_target_for_threshold": RECALL_TARGET,
        "top_dims_source": "notebooks/model_weight_analysis.ipynb, 300 m cell MLP",
        "cells": provenance,
    }
    with open(cond_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Written last: its presence is what marks the condition as done.
    reporting.write_fold_summary(rows, cond_dir / "fold_summary.csv")


def summary_row(name):
    df = pd.read_csv(RESULTS_DIR / name / "fold_summary.csv")
    row = {"condition": name, "n_params": int(df["n_params"].iloc[0]),
           "val_positive_rate_mean": df["positive_rate"].mean()}
    for col in ("auprc", "train_auprc", f"precision_at_recall_{RECALL_TARGET}",
                "precision", "recall", "f1"):
        row[f"{col}_mean"] = df[col].mean()
        row[f"{col}_std"] = df[col].std(ddof=1)
    row["train_minus_val_auprc"] = row["train_auprc_mean"] - row["auprc_mean"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("conditions", nargs="*",
                        help=f"any of {', '.join(CONDITIONS)}. Defaults to {', '.join(DEFAULT_CONDITIONS)}")
    args = parser.parse_args()
    # Checked here rather than with choices=, since argparse in this env rejects
    # an empty positional list against choices.
    unknown = [c for c in args.conditions if c not in CONDITIONS]
    if unknown:
        parser.error(f"unknown condition {', '.join(unknown)}, choose from {', '.join(CONDITIONS)}")
    conditions = args.conditions or list(DEFAULT_CONDITIONS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(SEED, workers=True)
    cells, cell_manifest, fold_code, provenance = hmt.load_checked_cells()

    for name in conditions:
        if (RESULTS_DIR / name / "fold_summary.csv").exists():
            print(f"\n{name} already done, skipping")
            continue
        run_condition(name, cells, cell_manifest, fold_code, provenance)

    done = [n for n in CONDITIONS if (RESULTS_DIR / n / "fold_summary.csv").exists()]
    summary = pd.DataFrame([summary_row(n) for n in done])
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    print(f"\n=== val AUPRC by input set, mean of {N_SPLITS} blocked folds ===")
    print(f"val positive rate {summary['val_positive_rate_mean'].iloc[0]:.2%} of cells. "
          f"precision, recall and f1 at each fold's {RECALL_TARGET:.0%} train recall threshold")
    show = ["condition", "n_params", "auprc_mean", "auprc_std", "train_auprc_mean",
            "train_minus_val_auprc", f"precision_at_recall_{RECALL_TARGET}_mean",
            "precision_mean", "recall_mean"]
    print(summary[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
