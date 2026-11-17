"""
Stage one classifier (any 2019 wetland developed by 2024, n_developed > 0) on
300 m cells, with the 2017 to 2019 development change features from
build_cell_dev_change.py added to the cell scalars.

    python classify_dev_change.py

Settings match the 300 m BCE run in results/hurdle_clf_mlp_300m_bce (MLP, BCE, no
pos_weight, natural batch rate, 20 epochs, final epoch scored, blocked 5 fold
split, seed 0), so the only thing that moves is the two added scalars. That run
is read back as the reference and checked cell for cell against this run's
validation folds.

Two logistic regressions ride along per fold, one on the original 3 scalars and
one on all 5, to show what the new features carry on their own with no
embeddings.
"""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import arrays as array_io
import datasets
import models
import reporting
from features import YEARS_2017_2019
from hurdle_model_train import assign_scheme_folds
from population_stats import load_population_stats

CELL_DIR = Path("data/cells_2019_2024_300m")
POPULATION_STATS_PATH = "data/population_stats_2019_2024.json"
RESULTS_DIR = Path("results/mlp_new_features")
REFERENCE_DIR = Path("results/hurdle_clf_mlp_300m_bce")
DEV_CHANGE_TAG = "dev_change_2017_2019"

N_YEARS = len(YEARS_2017_2019)
MODEL_CLS = models.CellMultiLayerPerceptronClassifier
SEED = 0
FOLD_SCHEME = "blocked"
N_SPLITS = 5
MAX_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 0.0
GAMMA = None  # BCE, to match the reference run
ALPHA = None
POS_WEIGHT = None
BATCH_SIZE = 1024
VAL_BATCH_SIZE = 8192
MIN_WETLAND_TRAIN = 1
RECALL_TARGET = 0.5
# Same cut as the validation map in notebooks/clf_prediction_analysis.ipynb, for
# hit and false alarm counts. Not a tuned operating point, probabilities are uncalibrated.
MAP_CUTOFF = 0.15
LOSS_NAME = "BCE loss"
CLASS_LABELS = ("no conversion", "any conversion")

# Order matters: the first three are datasets.CELL_SCALARS, so the logistic
# baseline on those is the old scalar set exactly.
SCALAR_SOURCES = (
    ("log1p_dist_mean", "dist_mean"),
    ("log1p_dist_min", "dist_min"),
    ("log1p_n_wetland", "n_wetland"),
    ("log1p_dev_change_cell_pct", f"{DEV_CHANGE_TAG}_cell_pct"),
    ("log1p_dev_change_ring1km_pct", f"{DEV_CHANGE_TAG}_ring1km_pct"),
)
SCALARS = tuple(name for name, _ in SCALAR_SOURCES)
N_OLD_SCALARS = len(datasets.CELL_SCALARS)
assert SCALARS[:N_OLD_SCALARS] == datasets.CELL_SCALARS

COMPARE_COLORS = {
    "logistic, 3 scalars": "#c9c8c4",
    "logistic, 5 scalars": "#8a8984",
    "MLP, without change features": "#009E73",
    "MLP, with change features": reporting.VAL_COLOR,
}


class DevChangeCellDataModule(datasets.CellDataModule):
    """CellDataModule with the scalars swapped for SCALAR_SOURCES. Folds, samplers
    and filters come from the parent unchanged, so train and val cells are the
    reference run's."""

    def setup(self, stage=None):
        if getattr(self, "train_idx", None) is not None:
            return
        super().setup(stage)
        c = self.cells
        # Change percents get log1p too: 94% of cells are exactly 0 and the rest run
        # out to 99, so on the raw scale a handful of cleared parcels set the std.
        raw = np.log1p(np.stack([c[k] for _, k in SCALAR_SOURCES], axis=1).astype(np.float64))
        mu = raw[self.train_idx].mean(axis=0)
        sd = raw[self.train_idx].std(axis=0)
        sd[sd == 0] = 1.0
        self.scalar_mean, self.scalar_std = mu.astype(np.float32), sd.astype(np.float32)
        scalars = ((raw - mu) / sd).astype(np.float32)
        self.ds = datasets.CellDataset(c, scalars, self.target_key, self.weight_key)


def load_cells_with_dev_change():
    cells, manifest = datasets.load_cells(CELL_DIR)
    added = manifest.get("added_features", {}).get(DEV_CHANGE_TAG)
    if added is None:
        raise SystemExit(f"{CELL_DIR} has no {DEV_CHANGE_TAG} features, run build_cell_dev_change.py first")
    for _, k in SCALAR_SOURCES[N_OLD_SCALARS:]:
        cells[k] = np.load(CELL_DIR / f"{k}.npy")
        if cells[k].shape[0] != manifest["stats"]["n_cells"]:
            raise AssertionError(f"{k}.npy does not have one row per cell")

    # Same label horizon checks as hurdle_model_train.load_checked_cells.
    src = manifest["source_manifest"]
    pop = load_population_stats(POPULATION_STATS_PATH)
    if pop["total_rows"] != src["total_rows"] or pop["total_positive"] != int(cells["n_developed"].sum()):
        raise AssertionError("population stats and cells disagree on rows or positives, one is stale")
    fold_code = array_io.fold_of_code(manifest, assign_scheme_folds(FOLD_SCHEME, pop, manifest))
    return cells, manifest, fold_code, added


def logistic_val_prob(data_module, n_scalars):
    """Val probabilities from a logistic regression on the first n_scalars z scored
    scalars, fit on the fold's train cells."""
    x = data_module.ds.scalars[:, :n_scalars]
    y = data_module.cells["n_developed"] > 0
    lr = LogisticRegression(max_iter=1000).fit(x[data_module.train_idx], y[data_module.train_idx])
    return lr.predict_proba(x[data_module.val_idx])[:, 1]


def run_fold(fold_idx, data_module, fold_dir):
    model = MODEL_CLS(n_years=N_YEARS, n_scalars=len(SCALARS), lr=LR, weight_decay=WEIGHT_DECAY,
                      gamma=GAMMA, alpha=ALPHA, pos_weight=POS_WEIGHT)
    fold_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=str(fold_dir.parent), name=f"fold_{fold_idx}")
    checkpoint = ModelCheckpoint(monitor=None, save_top_k=1)
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, callbacks=[checkpoint], logger=logger,
                         enable_progress_bar=False, accelerator="auto")
    trainer.fit(model, datamodule=data_module)
    final_model = MODEL_CLS.load_from_checkpoint(checkpoint.best_model_path)

    prob, target, n_wet = reporting.get_cell_predictions(final_model, data_module.val_dataloader())
    labels = target > 0
    np.savez(fold_dir / "val_preds.npz", prob=prob, any=labels, target=target,
             n_wetland=n_wet, cell_idx=data_module.val_idx)
    train_prob, train_target, train_n_wet = reporting.get_cell_predictions(
        final_model, data_module.train_eval_dataloader())
    train_labels = train_target > 0
    np.savez(fold_dir / "train_preds.npz", prob=train_prob, any=train_labels, target=train_target,
             n_wetland=train_n_wet, cell_idx=data_module.train_idx)

    metrics_csv = sorted(fold_dir.glob("**/metrics.csv"))[-1]
    reporting.plot_loss_curve(metrics_csv, fold_dir / "loss_curve.png",
                              f"Fold {fold_idx}: training and validation loss", ylabel=LOSS_NAME)
    reporting.plot_pr_curve_train_val(train_labels, train_prob, labels, prob, out_path=fold_dir / "pr_curve.png",
                                      title=f"Fold {fold_idx}: precision recall, any conversion in the cell")
    threshold = reporting.threshold_at_recall(train_labels, train_prob, RECALL_TARGET)
    reporting.plot_confusion_matrix(
        labels, prob, threshold=threshold, out_path=fold_dir / "confusion_matrix.png",
        title=f"Fold {fold_idx}: any conversion per cell, cut at {RECALL_TARGET:.0%} train recall",
        display_labels=CLASS_LABELS)

    score = reporting.score_fold(labels, prob, threshold=threshold)
    score.update({f"train_{k}": v for k, v in reporting.score_fold(train_labels, train_prob, threshold).items()})
    score[f"precision_at_recall_{RECALL_TARGET}"] = reporting.precision_at_recall(labels, prob, RECALL_TARGET)
    score.update({
        "fold": fold_idx, "threshold": threshold, "epochs_run": trainer.current_epoch,
        "checkpoint_path": checkpoint.best_model_path,
        "n_train_cells": int(data_module.train_idx.size), "n_val_cells": int(data_module.val_idx.size),
        "n_val_positive_cells": int(labels.sum()),
        "scalar_mean": data_module.scalar_mean.tolist(), "scalar_std": data_module.scalar_std.tolist(),
    })
    bins = [{"fold": fold_idx, **r} for r in reporting.auprc_by_n_wetland(labels, prob, n_wet)]
    return score, bins, labels, prob


def compare_row(name, fold, labels, prob):
    flag = prob > MAP_CUTOFF
    return {
        "model": name, "fold": fold, "n_val_cells": labels.size, "positive_rate": labels.mean(),
        "auprc": average_precision_score(labels, prob),
        f"precision_at_recall_{RECALL_TARGET}": reporting.precision_at_recall(labels, prob, RECALL_TARGET),
        f"hits_above_{MAP_CUTOFF}": int((flag & labels).sum()),
        f"false_alarms_above_{MAP_CUTOFF}": int((flag & ~labels).sum()),
    }


def plot_auprc_comparison(comp, out_path):
    per_fold = comp[comp["fold"] != "pooled"]
    folds = sorted(per_fold["fold"].unique())
    names = list(COMPARE_COLORS)
    width = 0.8 / len(names)
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, name in enumerate(names):
        s = per_fold[per_fold["model"] == name].set_index("fold").loc[folds]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, s["auprc"], width=width,
               color=COMPARE_COLORS[name], label=name, zorder=2)
    rate = per_fold[per_fold["model"] == names[-1]].set_index("fold").loc[folds, "positive_rate"]
    ax.hlines(rate, x - 0.4, x + 0.4, color="#52514e", lw=2, ls="--",
              label="Positive rate (random ranking)", zorder=3)
    ax.set_xticks(x, [f"Fold {f}" for f in folds])
    ax.set_ylim(0, max(0.7, per_fold["auprc"].max() * 1.35))
    ax.set_xlabel("Validation fold (held out 10 km blocks)", fontsize=14)
    ax.set_ylabel("AUPRC", fontsize=14)
    ax.set_title("Validation AUPRC with and without recent development features, 300 m cells", fontsize=15)
    ax.tick_params(labelsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=3, fontsize=11, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    pl.seed_everything(SEED, workers=True)
    cells, manifest, fold_code, added = load_cells_with_dev_change()

    ref_manifest = json.loads((REFERENCE_DIR / "manifest.json").read_text())
    if ref_manifest["cells"]["built_utc"] != manifest["built_utc"]:
        raise AssertionError(f"{REFERENCE_DIR} was trained on different cells, the comparison would not line up")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "task": "any 2019 wetland developed by 2024, per cell (n_developed > 0)",
        "model": MODEL_CLS.__name__, "loss": LOSS_NAME, "gamma": GAMMA, "alpha": ALPHA,
        "pos_weight": POS_WEIGHT, "seed": SEED,
        "folds_spec": {"scheme": FOLD_SCHEME, "n_splits": N_SPLITS, "seed": SEED,
                       "assignment": "greedy into the fold with fewest wetland pixels"},
        "max_epochs": MAX_EPOCHS, "checkpoint_scored": "final epoch, no early stopping",
        "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "pos_per_batch": None,
        "min_wetland_train": MIN_WETLAND_TRAIN, "recall_target_for_threshold": RECALL_TARGET,
        "scalars": list(SCALARS),
        "cells": {"path": str(CELL_DIR), "cell_size_m": manifest["cell_size_m"],
                  "built_utc": manifest["built_utc"], "git_commit": manifest["git_commit"],
                  "n_cells": manifest["stats"]["n_cells"],
                  "share_cells_with_any_developed": manifest["stats"]["share_cells_with_any_developed"],
                  "source_arrays": manifest["source_manifest"], DEV_CHANGE_TAG: added},
        "reference_run": str(REFERENCE_DIR),
        "folds": {},
    }

    scores, bins, comp_rows, fold_preds = [], [], [], []
    pooled = {name: ([], []) for name in COMPARE_COLORS}
    for fold_idx in range(N_SPLITS):
        print(f"\n--- fold {fold_idx} ---")
        dm = DevChangeCellDataModule(cells, manifest, fold_code, fold_idx, batch_size=BATCH_SIZE,
                                     val_batch_size=VAL_BATCH_SIZE, min_wetland_train=MIN_WETLAND_TRAIN,
                                     num_workers=0, seed=SEED)
        dm.setup()

        ref = np.load(REFERENCE_DIR / f"fold_{fold_idx}" / "val_preds.npz")
        if not np.array_equal(ref["cell_idx"], dm.val_idx):
            raise AssertionError(f"fold {fold_idx} val cells differ from {REFERENCE_DIR}")

        score, fold_bins, labels, prob = run_fold(fold_idx, dm, RESULTS_DIR / f"fold_{fold_idx}")
        scores.append(score)
        bins += fold_bins
        tr = np.load(RESULTS_DIR / f"fold_{fold_idx}" / "train_preds.npz")
        fold_preds.append((tr["any"], tr["prob"], labels, prob))

        for name, p in (("logistic, 3 scalars", logistic_val_prob(dm, N_OLD_SCALARS)),
                        ("logistic, 5 scalars", logistic_val_prob(dm, len(SCALARS))),
                        ("MLP, without change features", ref["prob"]),
                        ("MLP, with change features", prob)):
            comp_rows.append(compare_row(name, fold_idx, labels, p))
            pooled[name][0].append(labels)
            pooled[name][1].append(p)
        print(pd.DataFrame([r for r in comp_rows if r["fold"] == fold_idx]).round(4).to_string(index=False))

        run_manifest["folds"][fold_idx] = {k: score[k] for k in (
            "n_train_cells", "n_val_cells", "n_val_positive_cells", "threshold", "epochs_run",
            "scalar_mean", "scalar_std")}

    (RESULTS_DIR / "manifest.json").write_text(json.dumps(run_manifest, indent=2))

    # each cell is in exactly one val fold, so pooling scores every cell once
    for name, (ls, ps) in pooled.items():
        comp_rows.append(compare_row(name, "pooled", np.concatenate(ls), np.concatenate(ps)))
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(RESULTS_DIR / "comparison.csv", index=False)

    fold_dirs = [RESULTS_DIR / f"fold_{i}" for i in range(N_SPLITS)]
    reporting.plot_mean_loss_curve([sorted(d.glob("**/metrics.csv"))[-1] for d in fold_dirs],
                                   RESULTS_DIR / "loss_curve_mean.png",
                                   f"Mean training and validation loss over {N_SPLITS} spatially blocked folds",
                                   ylabel=LOSS_NAME)
    reporting.plot_mean_pr_curve(fold_preds, RESULTS_DIR / "pr_curve_mean.png",
                                 f"Mean precision recall over {N_SPLITS} folds, any conversion in the cell",
                                 chance_unit="cells")
    plot_auprc_comparison(comp, RESULTS_DIR / "auprc_comparison.png")

    print("\n=== fold summary, MLP with change features ===")
    print(f"precision, recall and f1 are at each fold's {RECALL_TARGET:.0%} train recall threshold")
    reporting.write_fold_summary(scores, RESULTS_DIR / "fold_summary.csv")
    pd.DataFrame(bins).to_csv(RESULTS_DIR / "auprc_by_n_wetland.csv", index=False)

    metric_cols = ["auprc", f"precision_at_recall_{RECALL_TARGET}",
                   f"hits_above_{MAP_CUTOFF}", f"false_alarms_above_{MAP_CUTOFF}"]
    per_fold = comp[comp["fold"] != "pooled"]
    summary = per_fold.groupby("model", sort=False)[metric_cols[:2]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.join(comp[comp["fold"] == "pooled"].set_index("model")[metric_cols].add_prefix("pooled_"))
    summary.to_csv(RESULTS_DIR / "summary.csv")
    rate = comp.loc[comp["fold"] == "pooled", "positive_rate"].iloc[0]
    print(f"\n=== validation summary, positive rate {rate:.2%} of {int(comp.loc[comp['fold'] == 'pooled', 'n_val_cells'].iloc[0]):,} cells ===")
    print(summary.round(4).to_string())

    best = max(scores, key=lambda r: r["auprc"])
    shutil.copy2(Path(best["checkpoint_path"]), RESULTS_DIR / "best_model.ckpt")


if __name__ == "__main__":
    main()
