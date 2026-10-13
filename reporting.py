"""
Results reporting: per-fold plots (loss curve, confusion matrix, PR curve) and
a fold-level summary CSV.

train.py calls into this rather than building these inline.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

# Colorblind safe pair, reused across every figure in here so train and val mean
# the same thing from plot to plot.
TRAIN_COLOR = "#D55E00"
VAL_COLOR = "#0072B2"


def get_val_predictions(model, dataloader):
    """Runs model over dataloader once, no grad. Returns (probs, labels) as numpy
    arrays.

    Batches are (emb, dist, y) from datasets.py: float16 embeddings and float32
    distance, kept separate until the model assembles them on device.

    Called after trainer.fit(), on a model loaded straight from checkpoint --
    unlike inside fit()/validate(), there's no Trainer here to move batches onto
    the model's device, so it has to be done by hand or this breaks the moment
    the accelerator isn't cpu.
    """
    model.eval()
    device = next(model.parameters()).device
    all_probs, all_labels = [], []
    with torch.no_grad():
        for emb, dist, y in dataloader:
            logits = model(emb.to(device), dist.to(device)).squeeze(1)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(y)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def plot_loss_curve(metrics_csv_path, out_path, title, ylabel="Focal loss"):
    """metrics_csv_path: the metrics.csv written by pytorch_lightning's CSVLogger."""
    df = pd.read_csv(metrics_csv_path)
    fig, ax = plt.subplots()
    if "train_loss" in df.columns:
        train_df = df.dropna(subset=["train_loss"])
        ax.plot(train_df["epoch"], train_df["train_loss"], label="train loss")
    if "val_loss" in df.columns:
        val_df = df.dropna(subset=["val_loss"])
        ax.plot(val_df["epoch"], val_df["val_loss"], label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(labels, probs, threshold, out_path, title):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        cm, display_labels=["not developed", "developed"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    return cm


def plot_pr_curve_train_val(train_labels, train_probs, val_labels, val_probs, out_path, title):
    """Train and val PR curves on one axes, so the overfit gap is a single
    glance instead of two separate PNGs you'd have to hold in your head
    together. Train sits at the true population positive rate here too (see
    each DataModule's train_eval_dataloader), so the two AUPRCs are directly
    comparable, not apples to oranges."""
    train_precision, train_recall, _ = precision_recall_curve(train_labels, train_probs)
    train_ap = average_precision_score(train_labels, train_probs)
    val_precision, val_recall, _ = precision_recall_curve(val_labels, val_probs)
    val_ap = average_precision_score(val_labels, val_probs)

    fig, ax = plt.subplots()
    ax.plot(val_recall, val_precision, color="#0072B2", label=f"validation (AUPRC = {val_ap:.3f})")
    ax.plot(train_recall, train_precision, color="#D55E00", linestyle="--",
            label=f"train (AUPRC = {train_ap:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)
    return train_ap, val_ap


def score_fold(labels, probs, threshold=0.5):
    """Precision/recall/F1/AUPRC/positive rate for one fold's val predictions.
    Report all of these together, not accuracy alone -- conversion is rare
    (~0.28% positive rate), so accuracy is close to meaningless on its own."""
    preds = (probs >= threshold).astype(int)
    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auprc": average_precision_score(labels, probs),
        "positive_rate": float(np.mean(labels)),
    }


def precision_at_recall(labels, probs, recall_target):
    """Precision interpolated at a fixed recall, same interpolation
    plot_mean_pr_curve uses for its recall grid (recall comes back descending
    from precision_recall_curve, np.interp wants it ascending)."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    return float(np.interp(recall_target, recall[::-1], precision[::-1]))


def write_fold_summary(rows, out_path):
    """
    rows: list of dicts, one per fold, each with at least fold, precision,
    recall, f1, auprc.

    Writes one row per fold plus the mean/std across folds for each metric --
    the usual spread you'd expect from k-fold CV.
    """
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    metric_cols = ["precision", "recall", "f1", "auprc"]
    train_metric_cols = [f"train_{c}" for c in metric_cols if f"train_{c}" in df.columns]
    metric_cols = metric_cols + train_metric_cols
    print(df)
    print("\nmean, std across folds:")
    print(df[metric_cols].agg(["mean", "std"]))
    return df


def _mean_std(stacked):
    """Mean and standard deviation down axis 0. ddof=1 for the usual k-fold
    sample std, but a single fold gets a flat zero ribbon instead of nan."""
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
    return mean, std


def plot_mean_loss_curve(metrics_csv_paths, out_path, title, ylabel="focal loss"):
    """Mean train and val loss per epoch across folds, with a plus or minus one
    standard deviation ribbon.

    metrics_csv_paths: one CSVLogger metrics.csv per fold. Lightning writes train
    and val on separate rows of the same epoch, so each is pulled out and grouped
    by epoch rather than read off a single row.

    Folds that stopped early would otherwise shrink the sample under the right
    hand end of the ribbon, so everything is truncated to the shortest fold and
    the ribbon is computed off a constant number of folds throughout.

    Do not read the vertical gap between the two curves as overfitting. train_loss
    is logged on the stratified batches while val_loss is a sweep at the natural rate 
    near 1 in 356, so the two sit on different scales by construction. 
    What is comparable is the shape of each over epochs and the fold to fold spread. 
    The PR curves are where the overfit gap is real, since both sides of those are scored at 
    the population positive rate.
    """
    train_series, val_series = [], []
    for path in metrics_csv_paths:
        df = pd.read_csv(path)
        for col, bucket in (("train_loss", train_series), ("val_loss", val_series)):
            if col in df.columns:
                s = df.dropna(subset=[col]).groupby("epoch")[col].mean().sort_index()
                if not s.empty:
                    bucket.append(s)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for series, color, label in (
        (train_series, TRAIN_COLOR, "train"),
        (val_series, VAL_COLOR, "validation"),
    ):
        if not series:
            continue
        n_epochs = min(len(s) for s in series)
        stacked = np.vstack([s.to_numpy()[:n_epochs] for s in series])
        epochs = series[0].index.to_numpy()[:n_epochs]
        mean, std = _mean_std(stacked)
        ax.plot(epochs, mean, color=color, linewidth=2,
                label=f"{label}, mean of {stacked.shape[0]} folds")
        # Unlabeled: the line's legend entry already says what the band is.
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)

    ax.set_xlabel("epoch", fontsize=13)
    ax.set_ylabel(f"{ylabel} (band is plus or minus 1 std across folds)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=12, frameon=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_ratio_summary(rows, out_path):
    """
    rows: one dict per (ratio, fold), each with at least ratio, fold,
    precision, recall, f1, auprc.

    Writes the full per fold per ratio table, plus mean/std grouped by ratio
    -- the k-fold spread from write_fold_summary, but kept separate per ratio
    instead of pooled across all of them.
    """
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    metric_cols = [c for c in ["precision", "recall", "f1", "auprc", "precision_at_recall"]
                   if c in df.columns]
    print(df)
    print("\nmean, std across folds, by ratio:")
    print(df.groupby("ratio")[metric_cols].agg(["mean", "std"]))
    return df


def _set_ratio_axis(ax, ratios):
    """Log x-axis with a tick at every ratio actually tested. Log scale's
    default ticks land on round powers of ten, which would miss most of a
    sweep like 1, 5, 10, 50, 356."""
    ax.set_xscale("log")
    ratios = sorted(set(ratios))
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"1:{r:.0f}" for r in ratios])
    ax.minorticks_off()
    ax.set_xlabel("pos:neg ratio in training batch", fontsize=13)


def plot_auprc_vs_ratio(rows, out_path, title):
    """Validation AUPRC vs. the realized pos:neg ratio in the training batch,
    fold mean with a plus or minus one std error bar.

    rows: one dict per (ratio, fold) with at least "ratio" and "auprc" (val
    AUPRC from score_fold). Use the realized ratio (batch_pos_weight off the
    sampler), not the requested one -- rounding pos_per_batch to an integer
    shifts the natural-rate point noticeably (1:356 target rounds to a
    realized 1:340 at batch size 2048).
    """
    df = pd.DataFrame(rows)
    grouped = df.groupby("ratio")["auprc"].agg(["mean", "std"]).sort_index()
    grouped["std"] = grouped["std"].fillna(0.0)  # a single fold has no spread

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                fmt="o-", color=VAL_COLOR, capsize=4, linewidth=2, markersize=7)
    _set_ratio_axis(ax, grouped.index)
    ax.set_ylabel("validation AUPRC (mean plus or minus 1 std across folds)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=11)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves_by_ratio(ratio_fold_preds, out_path, title, n_recall_points=201):
    """One mean validation PR curve per ratio, overlaid on shared axes, so the
    crossover between ratios is visible directly instead of buried in five
    separate AUPRC numbers.

    ratio_fold_preds: dict of realized ratio -> list of (train_labels,
    train_probs, val_labels, val_probs) tuples, one tuple per fold. Same fold
    mean interpolation as plot_mean_pr_curve, run once per ratio.
    """
    recall_grid = np.linspace(0.0, 1.0, n_recall_points)
    # Colorblind safe categorical palette, low to high ratio light to dark.
    palette = ["#009E73", "#56B4E9", "#0072B2", "#E69F00", "#D55E00", "#CC79A7"]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    positive_rates = []
    for color, (ratio, fold_preds) in zip(palette, sorted(ratio_fold_preds.items())):
        interpolated, aps = [], []
        for _, _, val_labels, val_probs in fold_preds:
            precision, recall, _ = precision_recall_curve(val_labels, val_probs)
            interpolated.append(np.interp(recall_grid, recall[::-1], precision[::-1]))
            aps.append(average_precision_score(val_labels, val_probs))
            positive_rates.append(float(np.mean(val_labels)))

        mean, _ = _mean_std(np.vstack(interpolated))
        ap_mean = float(np.mean(aps))
        ax.plot(recall_grid, mean, color=color, linewidth=2,
                label=f"1:{ratio:.0f}, AUPRC {ap_mean:.3f}")

    # A flat line at the positive rate is what a coin flip scores here.
    chance = float(np.mean(positive_rates))
    ax.axhline(chance, color="#666666", linewidth=1, linestyle=":",
               label=f"chance, {chance:.3%} of pixels converted")

    ax.set_xlabel("recall", fontsize=13)
    ax.set_ylabel("precision", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11, frameon=False, title="training batch ratio")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_precision_at_recall_vs_ratio(rows, recall_target, out_path, title):
    """Precision at a fixed validation recall target vs. ratio, fold mean with
    a plus or minus one std error bar.

    Where plot_auprc_vs_ratio asks which ratio has the best overall tradeoff,
    this asks the sharper question: at the recall we've decided we need,
    which ratio buys the fewest false positives.

    rows: one dict per (ratio, fold) with at least "ratio" and
    "precision_at_recall".
    """
    df = pd.DataFrame(rows)
    grouped = df.groupby("ratio")["precision_at_recall"].agg(["mean", "std"]).sort_index()
    grouped["std"] = grouped["std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                fmt="o-", color="#CC79A7", capsize=4, linewidth=2, markersize=7)
    _set_ratio_axis(ax, grouped.index)
    ax.set_ylabel(f"precision at {recall_target:.0%} recall\n(mean plus or minus 1 std across folds)",
                  fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=11)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mean_pr_curve(fold_preds, out_path, title, n_recall_points=201, chance_unit="pixels"):
    """Mean train and val precision recall curve across folds, with a plus or
    minus one standard deviation ribbon.

    fold_preds: one (train_labels, train_probs, val_labels, val_probs) tuple per
    fold, matching the argument order of plot_pr_curve_train_val.

    Folds do not share recall values, so each fold's precision is interpolated
    onto a common recall grid before averaging. Precision against recall is
    really a step function, so the interpolation smooths it slightly; the point
    of this figure is fold to fold spread, not the exact shape of any one curve.

    Legend AUPRCs are the mean and std of the per fold average_precision_score,
    computed on the raw predictions, not read off the interpolated mean curve.
    """
    recall_grid = np.linspace(0.0, 1.0, n_recall_points)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    positive_rates = []
    for idx_labels, idx_probs, color, style, label in (
        (0, 1, TRAIN_COLOR, "--", "train"),
        (2, 3, VAL_COLOR, "-", "validation"),
    ):
        interpolated, aps = [], []
        for fold in fold_preds:
            labels, probs = fold[idx_labels], fold[idx_probs]
            precision, recall, _ = precision_recall_curve(labels, probs)
            # precision_recall_curve hands back recall descending; np.interp needs
            # it ascending.
            interpolated.append(np.interp(recall_grid, recall[::-1], precision[::-1]))
            aps.append(average_precision_score(labels, probs))

        stacked = np.vstack(interpolated)
        mean, std = _mean_std(stacked)
        ap_mean = float(np.mean(aps))
        ap_std = float(np.std(aps, ddof=1)) if len(aps) > 1 else 0.0
        ax.plot(recall_grid, mean, color=color, linestyle=style, linewidth=2,
                label=f"{label}, AUPRC {ap_mean:.3f} plus or minus {ap_std:.3f}")
        # Unlabeled: the line's legend entry already says what the band is.
        ax.fill_between(recall_grid, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                        color=color, alpha=0.2, linewidth=0)
        positive_rates.append(float(np.mean(np.concatenate([f[idx_labels] for f in fold_preds]))))

    # A flat line at the positive rate is what a coin flip scores here. Without
    # it an AUPRC of 0.09 reads as terrible rather than as 30x chance.
    chance = float(np.mean(positive_rates))
    ax.axhline(chance, color="#666666", linewidth=1, linestyle=":",
               label=f"chance, {chance:.3%} of {chance_unit} converted")

    ax.set_xlabel("recall", fontsize=13)
    ax.set_ylabel("precision (band is plus or minus 1 std across folds)", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=12, frameon=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# Regression side, for the cell task in regress_train.py. Predictions there are a
# share of the cell's 2019 wetland developed by 2024, not a probability, so the
# threshold metrics above do not apply. "Any conversion" (target > 0) is the
# binary view used wherever a PR curve or AUPRC shows up.

WEIGHTED_COLOR = "#009E73"
REFERENCE_COLOR = "#666666"


def get_cell_predictions(model, dataloader):
    """Runs a CellTemporalConvolutionalNetwork over dataloader once, no grad.
    Returns (pred, target, n_wetland) as numpy arrays, in loader order. Batches
    are the 5 tuples datasets.CellDataModule yields. Same device caveat as
    get_val_predictions."""
    model.eval()
    device = next(model.parameters()).device
    preds, targets, n_wets = [], [], []
    with torch.no_grad():
        for emb_mean, emb_std, scalars, y, n_wetland in dataloader:
            pred = model(emb_mean.to(device), emb_std.to(device), scalars.to(device))
            preds.append(pred.cpu())
            targets.append(y)
            n_wets.append(n_wetland)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy(), torch.cat(n_wets).numpy()


def score_cells(pred, target, n_wetland):
    """
    Everything a regression fold gets judged on. Unweighted metrics count every
    cell once; weighted ones count every wetland pixel once.

    zero_mse and zero_mae are what predicting 0 for every cell would score. With
    about 97% of 300 m cells at exactly 0, MSE and MAE look small for any model,
    so they only mean something next to this baseline.
    """
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    w = n_wetland.astype(np.float64)
    err = pred - target
    any_dev = target > 0
    mse = float(np.mean(err ** 2))
    # Clipped because pred comes in as a float32 probability, which rounds to
    # exactly 0 or 1 once the logit is far enough out. Binomial log likelihood per
    # pixel, so it lines up with val_log_loss_weighted in training.
    p = np.clip(pred, 1e-7, 1 - 1e-7)
    log_loss = -(target * np.log(p) + (1 - target) * np.log(1 - p))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mse_weighted": float(np.average(err ** 2, weights=w)),
        "log_loss_weighted": float(np.average(log_loss, weights=w)),
        "mae": float(np.mean(np.abs(err))),
        "mae_weighted": float(np.average(np.abs(err), weights=w)),
        "zero_mse": float(np.mean(target ** 2)),
        "zero_mae": float(np.mean(target)),
        "auprc": float(average_precision_score(any_dev, pred)) if any_dev.any() else float("nan"),
        "positive_rate": float(any_dev.mean()),
        "pred_over_actual": float((pred * w).sum() / max((target * w).sum(), 1.0)),
        "mean_pred": float(pred.mean()),
        "mean_target": float(target.mean()),
        "n_cells": int(target.size),
    }


def write_regression_summary(rows, out_path):
    """One row per fold plus the mean and std across folds, printed. The CSV keeps
    every column score_cells and the training script put in."""
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    base = ["mse", "zero_mse", "mse_weighted", "log_loss_weighted", "mae", "zero_mae",
            "mae_weighted", "auprc", "positive_rate", "pred_over_actual"]
    cols = [c for c in base + [f"train_{c}" for c in base] if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df[["fold"] + cols])
        print("\nmean, std across folds:")
        print(df[cols].agg(["mean", "std"]))
    return df


def _stack_epochs(metrics_csv_paths, col):
    """(epochs, stacked) for one logged column across folds, stacked shaped
    (n_folds, n_epochs) and truncated to the shortest fold. (None, None) if no
    fold logged it."""
    series = []
    for path in metrics_csv_paths:
        df = pd.read_csv(path)
        if col in df.columns:
            s = df.dropna(subset=[col]).groupby("epoch")[col].mean().sort_index()
            if not s.empty:
                series.append(s)
    if not series:
        return None, None
    n = min(len(s) for s in series)
    return series[0].index.to_numpy()[:n], np.vstack([s.to_numpy()[:n] for s in series])


def _plot_band(ax, epochs, stacked, color, label, linestyle="-"):
    mean, std = _mean_std(stacked)
    if stacked.shape[0] > 1:
        label = f"{label}, mean of {stacked.shape[0]} folds"
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
    ax.plot(epochs, mean, color=color, linewidth=2, linestyle=linestyle, label=label)


def plot_cell_val_metrics(metrics_csv_paths, out_path, title_suffix=""):
    """
    Validation MAE, AUPRC on any conversion, and predicted over actual converted
    pixels, per epoch. One metrics.csv gives a single fold's curves; several give
    the fold mean with a plus or minus one std band.

    AUPRC is drawn against the share of val cells with any conversion, which is
    what a random ranking scores. Predicted over actual gets a line at 1, where
    the model's total converted area matches the real one.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    ax = axes[0]
    for col, color, label in (("val_mae", VAL_COLOR, "every cell counts once"),
                              ("val_mae_weighted", WEIGHTED_COLOR, "weighted by wetland pixels")):
        epochs, stacked = _stack_epochs(metrics_csv_paths, col)
        if stacked is not None:
            _plot_band(ax, epochs, stacked, color, label)
    ax.set_ylabel("Mean absolute error (fraction of wetland)", fontsize=12)
    ax.set_title(f"Validation MAE{title_suffix}", fontsize=14)

    ax = axes[1]
    epochs, stacked = _stack_epochs(metrics_csv_paths, "val_auprc")
    if stacked is not None:
        _plot_band(ax, epochs, stacked, VAL_COLOR, "validation AUPRC")
    epochs, rate = _stack_epochs(metrics_csv_paths, "val_pos_rate")
    if rate is not None:
        ax.axhline(float(rate.mean()), color=REFERENCE_COLOR, linestyle=":", linewidth=1.5,
                   label=f"chance, {rate.mean():.2%} of cells with any conversion")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("AUPRC, any conversion in the cell", fontsize=12)
    ax.set_title(f"Validation AUPRC{title_suffix}", fontsize=14)

    ax = axes[2]
    epochs, stacked = _stack_epochs(metrics_csv_paths, "val_pred_over_actual")
    if stacked is not None:
        _plot_band(ax, epochs, stacked, VAL_COLOR, "validation")
    ax.axhline(1.0, color=REFERENCE_COLOR, linestyle=":", linewidth=1.5,
               label="predicted total equals actual")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Predicted over actual converted pixels", fontsize=12)
    ax.set_title(f"Total converted area{title_suffix}", fontsize=14)

    for ax in axes:
        ax.set_xlabel("Epoch", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# Outcome classes for the per bin mix columns in the calibration CSV.
OUTCOME_BINS = (
    ("no conversion", None),
    ("under 10%", (0.0, 0.10)),
    ("10 to 50%", (0.10, 0.50)),
    ("50 to 100%", (0.50, 1.0)),
)


def plot_cell_calibration(pred, target, out_path, ratio_out_path, title, ratio_title, n_bins=20,
                          label="validation", color=VAL_COLOR):
    """
    Cells sorted by predicted share into n_bins equal count bins, two figures.

    out_path: mean actual against mean predicted per bin, both axes log. On linear
    axes almost every bin sits under 1% predicted and collapses into the corner,
    so the top bin alone decides how the plot looks; on log axes a bin that is
    off by 10x is visibly off by 10x however small its values are.
    ratio_out_path: mean actual over mean predicted per bin, with a line at 1.
    Above the line the model underpredicts that bin.

    Bin means hide the mix inside a bin (mostly zeros plus a few cells that lost
    everything), so the CSV keeps the share of each bin in each OUTCOME_BINS class.

    Predictions saturate at the sigmoid's floor for many cells, so bin means are
    floored at 1e-10 percent before the log. Returns the per bin table, which is
    also what goes in the CSV next to the PNGs.
    """
    order = np.argsort(pred, kind="stable")
    bins = np.array_split(order, n_bins)
    rows = []
    for i, b in enumerate(bins):
        t, p = target[b], pred[b]
        row = {
            "bin": i + 1,
            "n_cells": b.size,
            "mean_pred_pct": 100 * float(p.mean()),
            "mean_actual_pct": 100 * float(t.mean()),
            "actual_p90_pct": 100 * float(np.percentile(t, 90)),
        }
        for name, rng in OUTCOME_BINS:
            m = (t == 0) if rng is None else ((t > rng[0]) & (t <= rng[1]))
            row[f"share_{name.replace(' ', '_').replace('%', 'pct')}"] = float(m.mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    floor = 1e-10
    mp = table["mean_pred_pct"].clip(lower=floor)
    ma = table["mean_actual_pct"].clip(lower=floor)
    table["actual_over_pred"] = ma / mp

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    lo = min(mp.min(), ma.min()) / 3
    hi = max(mp.max(), ma.max()) * 3
    ax.plot([lo, hi], [lo, hi], color=REFERENCE_COLOR, linestyle=":", linewidth=1.5,
            label="prediction matches actual")
    ax.plot(mp, ma, "o-", color=color, linewidth=2, markersize=6,
            label=f"{label}, {n_bins} bins of {len(pred) // n_bins:,} cells")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Mean predicted share converted (percent, log)", fontsize=12)
    ax.set_ylabel("Mean actual share converted (percent, log)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    x = table["bin"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    ax.axhline(1.0, color=REFERENCE_COLOR, linestyle=":", linewidth=1.5, label="prediction matches actual")
    ax.plot(x, table["actual_over_pred"], "o-", color=color, linewidth=2, markersize=6,
            label=f"{label}, actual over predicted")
    ax.set_yscale("log")
    ax.set_xticks(x[::2] if len(x) > 10 else x)
    ax.set_xlabel("Prediction bin, lowest to highest", fontsize=12)
    ax.set_ylabel("Mean actual over mean predicted (log)", fontsize=12)
    ax.set_title(ratio_title, fontsize=14)
    ax.legend(fontsize=11, frameon=False)
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ratio_out_path, bbox_inches="tight")
    plt.close(fig)
    return table


def write_train_calibration_pooled(fold_dirs, results_dir):
    """
    Calibration of each fold's final model on its own training cells, pooled
    over folds, from the train_preds.npz every fold writes. Writes
    training_calibration_pooled.png, its _ratio.png and .csv into results_dir.

    Unlike the val pool, this does not score each cell once. A cell sits in the
    train set of every fold but its own, so with 5 folds it shows up 4 times,
    scored by 4 different models. Read it as the average in sample fit of the
    fold models, next to calibration_pooled.png for the out of sample one.
    """
    preds, targets = [], []
    for d in fold_dirs:
        tr = np.load(Path(d) / "train_preds.npz")
        preds.append(tr["pred"])
        targets.append(tr["target"])
    results_dir = Path(results_dir)
    table = plot_cell_calibration(
        np.concatenate(preds), np.concatenate(targets),
        results_dir / "training_calibration_pooled.png",
        results_dir / "training_calibration_pooled_ratio.png",
        "Predicted against actual share converted, training cells, all folds",
        "Actual over predicted share converted by bin, training cells, all folds",
        label="training", color=TRAIN_COLOR,
    )
    table.to_csv(results_dir / "training_calibration_pooled.csv", index=False)
    return table
