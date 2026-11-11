"""
Width and depth sweep for the stage one cell classifier,
models.CellMultiLayerPerceptronClassifier, on the east and west folds:

    python hurdle_mlp_architecture.py              # 300 m cells
    python hurdle_mlp_architecture.py --cells 3km  # 3 km cells

--cells picks data/cells_2019_2024_<size> and writes to
results/hurdle_mlp_<size>_architecture, so the two sizes never share a results
directory and never skip each other's runs. Every hyperparameter stays the same
across sizes. At 3 km there are about 24k cells instead of millions, so 15
epochs at batch 1024 is only about 280 optimizer steps.

Same task, data, loss and fold handling as hurdle_model_train.py classify: any
2019 wetland developed by 2024 per cell, focal loss at gamma 2, natural positive
rate batches, final epoch scored. The only thing that changes between runs is
hidden_dims. [256, 64, 1] is the architecture the earlier hurdle_clf runs used;
the grid holds every hidden layer at one width so a row of the summary changes
depth only and a column changes width only.

Dropout 0.3 sits after every hidden layer, so a deeper net also carries more
total dropout. A depth effect here is partly a regularization effect.

Each architecture writes to RESULTS_DIR/<name>/: per fold loss and PR curves,
train and val predictions, the two fold mean curves, fold_summary.csv and a
manifest. RESULTS_DIR/architecture_summary.csv has one row per architecture.
An architecture whose fold_summary.csv already exists is skipped, so a crashed
sweep picks up where it stopped. Delete that directory to rerun it.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path

import datasets
import hurdle_model_train as hmt
import models
import reporting
from folds import GEOGRAPHIC_FOLD_NAMES, SPLIT_CENTER

CELL_SIZES = {"300m": "300 m cells", "1km": "1 km cells", "3km": "3 km cells"}
# Set by configure() from --cells. Defaults to 300 m so importing this module
# from a notebook still points at the original sweep.
CELL_SIZE = "300m"
CELL_LABEL = CELL_SIZES[CELL_SIZE]
RESULTS_DIR = Path("results/hurdle_mlp_300m_architecture")


def configure(cell_size):
    """Points the sweep and hmt.load_checked_cells at one cell size. hmt reads
    CELL_DIR when load_checked_cells runs, so setting it here is enough."""
    global CELL_SIZE, CELL_LABEL, RESULTS_DIR
    CELL_SIZE, CELL_LABEL = cell_size, CELL_SIZES[cell_size]
    RESULTS_DIR = Path(f"results/hurdle_mlp_{cell_size}_architecture")
    hmt.CELL_DIR = f"data/cells_2019_2024_{cell_size}"

# Pinned here rather than read off hmt.FOLD_SCHEME, which gets flipped between
# runs. load_checked_cells reads the module global when called, so setting it
# before the call is enough.
FOLD_SCHEME = "east_west"
N_SPLITS = 2
hmt.FOLD_SCHEME = FOLD_SCHEME
FOLD_NAMES = GEOGRAPHIC_FOLD_NAMES[FOLD_SCHEME]

SEED = 0
MAX_EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 0.0
DROPOUT = 0.3
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

BASELINE = (256, 64)
WIDTHS = (256, 512, 1024)
DEPTHS = (1, 2, 3, 4)
ARCHITECTURES = [BASELINE] + [(w,) * d for w in WIDTHS for d in DEPTHS]

# Okabe and Ito, one per width, baseline in black.
WIDTH_COLORS = {256: "#0072B2", 512: "#D55E00", 1024: "#009E73"}
BASELINE_COLOR = "#000000"


def arch_name(hidden_dims):
    return "h_" + "_".join(str(h) for h in hidden_dims)


def arch_label(hidden_dims):
    return "[" + ", ".join(str(h) for h in hidden_dims) + ", 1]"


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def run_fold(hidden_dims, fold_idx, data_module, fold_dir):
    """Trains one architecture on one fold and writes its curves and predictions.
    Returns the fold score dict."""
    model = models.CellMultiLayerPerceptronClassifier(
        n_years=hmt.N_YEARS, dropout=DROPOUT, lr=LR, weight_decay=WEIGHT_DECAY,
        gamma=GAMMA, alpha=ALPHA, pos_weight=POS_WEIGHT, hidden_dims=hidden_dims,
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

    val_region = FOLD_NAMES[fold_idx]
    metrics_csvs = sorted(fold_dir.glob("**/metrics.csv"))
    if metrics_csvs:
        reporting.plot_loss_curve(
            metrics_csvs[-1], fold_dir / "loss_curve.png",
            f"MLP {arch_label(hidden_dims)}, {CELL_LABEL}, validated on the {val_region}", ylabel=LOSS_NAME,
        )
    reporting.plot_pr_curve_train_val(
        train_labels, train_prob, labels, prob, out_path=fold_dir / "pr_curve.png",
        title=f"MLP {arch_label(hidden_dims)}, any conversion, {CELL_LABEL}, validated on the {val_region}",
    )

    threshold = reporting.threshold_at_recall(train_labels, train_prob, RECALL_TARGET)
    score = reporting.score_fold(labels, prob, threshold=threshold)
    train_score = reporting.score_fold(train_labels, train_prob, threshold=threshold)
    score.update({f"train_{k}": v for k, v in train_score.items()})
    score[f"precision_at_recall_{RECALL_TARGET}"] = reporting.precision_at_recall(
        labels, prob, RECALL_TARGET
    )
    score.update({
        "fold": fold_idx,
        "val_region": val_region,
        "threshold": threshold,
        "epochs_run": trainer.current_epoch,
        "checkpoint_path": checkpoint.best_model_path,
        "n_params": n_params(final_model),
        "n_train_cells": int(data_module.train_idx.size),
        "n_val_cells": int(data_module.val_idx.size),
        "n_val_positive_cells": int(labels.sum()),
        "batch_positive_rate": data_module.batch_positive_rate,
    })
    return score


def run_architecture(hidden_dims, cells, cell_manifest, fold_code, provenance):
    arch_dir = RESULTS_DIR / arch_name(hidden_dims)
    arch_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold_idx in range(N_SPLITS):
        print(f"\n--- {arch_label(hidden_dims)}, fold {fold_idx}, validating on the "
              f"{FOLD_NAMES[fold_idx]} ---")
        # Reseeded per architecture and fold so a result does not depend on where
        # the architecture falls in the sweep or whether earlier ones were skipped.
        pl.seed_everything(SEED, workers=True)
        data_module = datasets.CellDataModule(
            cells, cell_manifest, fold_code, fold_idx,
            batch_size=BATCH_SIZE, val_batch_size=VAL_BATCH_SIZE,
            min_wetland_train=MIN_WETLAND_TRAIN, pos_per_batch=POS_PER_BATCH,
            num_workers=NUM_WORKERS, seed=SEED,
        )
        data_module.setup()
        rows.append(run_fold(hidden_dims, fold_idx, data_module, arch_dir / f"fold_{fold_idx}"))

    fold_dirs = [arch_dir / f"fold_{i}" for i in range(N_SPLITS)]
    metrics_csvs = [c for d in fold_dirs for c in sorted(d.glob("**/metrics.csv"))[-1:]]
    if metrics_csvs:
        reporting.plot_mean_loss_curve(
            metrics_csvs, arch_dir / "loss_curve_mean.png",
            f"MLP {arch_label(hidden_dims)}, {CELL_LABEL}, loss over east and west folds", ylabel=LOSS_NAME,
        )
    fold_preds = []
    for d in fold_dirs:
        tr, va = np.load(d / "train_preds.npz"), np.load(d / "val_preds.npz")
        fold_preds.append((tr["any"], tr["prob"], va["any"], va["prob"]))
    reporting.plot_mean_pr_curve(
        fold_preds, arch_dir / "pr_curve_mean.png",
        f"MLP {arch_label(hidden_dims)}, any conversion, {CELL_LABEL}, east and west folds",
        chance_unit="cells",
    )

    manifest = {
        "task": "any 2019 wetland developed by 2024, per cell (n_developed > 0)",
        "model": models.CellMultiLayerPerceptronClassifier.__name__,
        "architecture": arch_label(hidden_dims),
        "hidden_dims": list(hidden_dims),
        "n_params": rows[0]["n_params"],
        "dropout": DROPOUT,
        "loss": LOSS_NAME, "gamma": GAMMA, "alpha": ALPHA, "pos_weight": POS_WEIGHT,
        "seed": SEED,
        "folds_spec": {"scheme": FOLD_SCHEME, "n_splits": N_SPLITS,
                       "center_lat_lon": list(SPLIT_CENTER), "fold_names": list(FOLD_NAMES)},
        "max_epochs": MAX_EPOCHS,
        "checkpoint_scored": "final epoch, no early stopping",
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE, "pos_per_batch": POS_PER_BATCH,
        "min_wetland_train": MIN_WETLAND_TRAIN,
        "recall_target_for_threshold": RECALL_TARGET,
        "scalars": list(datasets.CELL_SCALARS),
        "cells": provenance,
    }
    with open(arch_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Written last: its presence is what marks the architecture as done.
    reporting.write_fold_summary(rows, arch_dir / "fold_summary.csv")


def summary_row(hidden_dims):
    df = pd.read_csv(RESULTS_DIR / arch_name(hidden_dims) / "fold_summary.csv")
    row = {
        "architecture": arch_label(hidden_dims),
        "width": hidden_dims[0] if len(set(hidden_dims)) == 1 else None,
        "depth": len(hidden_dims),
        "n_params": int(df["n_params"].iloc[0]),
    }
    for _, r in df.iterrows():
        region = r["val_region"]
        row[f"val_auprc_{region}"] = r["auprc"]
        row[f"val_positive_rate_{region}"] = r["positive_rate"]
    for col in ("auprc", "train_auprc", f"precision_at_recall_{RECALL_TARGET}",
                "precision", "recall", "f1"):
        row[f"{col}_mean"] = df[col].mean()
    row["train_minus_val_auprc"] = row["train_auprc_mean"] - row["auprc_mean"]
    return row


def plot_auprc_by_size(summary, out_path):
    """Mean val AUPRC against parameter count, one line per width running depth 1
    to 4, baseline as a lone marker."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for width, color in WIDTH_COLORS.items():
        g = summary[summary["width"] == width].sort_values("depth")
        if g.empty:
            continue
        ax.plot(g["n_params"], g["auprc_mean"], color=color, linewidth=2, marker="o",
                markersize=8, label=f"width {width}")
        for _, r in g.iterrows():
            d = int(r["depth"])
            ax.annotate(f"{d} layer{'s' if d > 1 else ''}", (r["n_params"], r["auprc_mean"]),
                        textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=9, color="#333333")
    base = summary[summary["architecture"] == arch_label(BASELINE)]
    ax.plot(base["n_params"], base["auprc_mean"], linestyle="none", marker="D", markersize=9,
            color=BASELINE_COLOR, label=f"baseline {arch_label(BASELINE)}")

    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log scale)", fontsize=13)
    ax.set_ylabel("validation AUPRC, mean of east and west folds", fontsize=12)
    ax.set_title(f"Conversion classifier on {CELL_LABEL}: validation AUPRC by MLP width and depth",
                 fontsize=13)
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=11, frameon=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", choices=sorted(CELL_SIZES), default="300m")
    configure(parser.parse_args().cells)
    print(f"cells from {hmt.CELL_DIR}, results to {RESULTS_DIR}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cells, cell_manifest, fold_code, provenance = hmt.load_checked_cells()

    for hidden_dims in ARCHITECTURES:
        if (RESULTS_DIR / arch_name(hidden_dims) / "fold_summary.csv").exists():
            print(f"\n{arch_label(hidden_dims)} already done, skipping")
            continue
        run_architecture(hidden_dims, cells, cell_manifest, fold_code, provenance)

    summary = pd.DataFrame([summary_row(h) for h in ARCHITECTURES])
    summary.to_csv(RESULTS_DIR / "architecture_summary.csv", index=False)
    plot_auprc_by_size(summary, RESULTS_DIR / "auprc_by_architecture.png")

    print("\n=== validation AUPRC by architecture, east and west folds ===")
    rate_cols = [c for c in summary.columns if c.startswith("val_positive_rate_")]
    print("val positive rates: " + ", ".join(
        f"{c.removeprefix('val_positive_rate_')} {summary[c].iloc[0]:.2%}" for c in rate_cols))
    show = ["architecture", "n_params"] + [c for c in summary.columns if c.startswith("val_auprc_")] \
        + ["auprc_mean", "train_auprc_mean", "train_minus_val_auprc",
           f"precision_at_recall_{RECALL_TARGET}_mean"]
    print(summary[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
