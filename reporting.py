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


def get_val_predictions(model, val_dataloader):
    """Runs model over val_dataloader once, no grad. Returns (probs, labels) as
    numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in val_dataloader:
            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits)
            all_probs.append(probs)
            all_labels.append(y)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def plot_loss_curve(metrics_csv_path, out_path, title):
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
    ax.set_ylabel("BCE loss")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(labels, probs, threshold, out_path, title):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        cm, display_labels=["not developed", "converted to developed"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    return cm


def plot_pr_curve(labels, probs, out_path, title):
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AUPRC = {ap:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)
    return ap


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
    print(df)
    print("\nmean, std across folds:")
    print(df[metric_cols].agg(["mean", "std"]))
    return df
