
import pyarrow.dataset as ds
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score,
)
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


from pathlib import Path


import datasets
import models

PARQUET_DIR = "data/alphaearth_wetland_joined"
FEATURE_COLS = ['dist_to_developed_2019_m', 'A00_2017', 'A01_2017', 'A02_2017', 'A03_2017', 'A04_2017', 'A05_2017', 'A06_2017', 'A07_2017', 'A08_2017', 'A09_2017', 'A10_2017', 
                'A11_2017', 'A12_2017', 'A13_2017', 'A14_2017', 'A15_2017', 'A16_2017', 'A17_2017', 'A18_2017', 'A19_2017', 'A20_2017', 'A21_2017', 'A22_2017', 'A23_2017', 
                'A24_2017', 'A25_2017', 'A26_2017', 'A27_2017', 'A28_2017', 'A29_2017', 'A30_2017', 'A31_2017', 'A32_2017', 'A33_2017', 'A34_2017', 'A35_2017', 'A36_2017', 
                'A37_2017', 'A38_2017', 'A39_2017', 'A40_2017', 'A41_2017', 'A42_2017', 'A43_2017', 'A44_2017', 'A45_2017', 'A46_2017', 'A47_2017', 'A48_2017', 'A49_2017', 
                'A50_2017', 'A51_2017', 'A52_2017', 'A53_2017', 'A54_2017', 'A55_2017', 'A56_2017', 'A57_2017', 'A58_2017', 'A59_2017', 'A60_2017', 'A61_2017', 'A62_2017', 
                'A63_2017', 'A00_2018', 'A01_2018', 'A02_2018', 'A03_2018', 'A04_2018', 'A05_2018', 'A06_2018', 'A07_2018', 'A08_2018', 'A09_2018', 'A10_2018', 'A11_2018', 
                'A12_2018', 'A13_2018', 'A14_2018', 'A15_2018', 'A16_2018', 'A17_2018', 'A18_2018', 'A19_2018', 'A20_2018', 'A21_2018', 'A22_2018', 'A23_2018', 'A24_2018', 
                'A25_2018', 'A26_2018', 'A27_2018', 'A28_2018', 'A29_2018', 'A30_2018', 'A31_2018', 'A32_2018', 'A33_2018', 'A34_2018', 'A35_2018', 'A36_2018', 'A37_2018', 
                'A38_2018', 'A39_2018', 'A40_2018', 'A41_2018', 'A42_2018', 'A43_2018', 'A44_2018', 'A45_2018', 'A46_2018', 'A47_2018', 'A48_2018', 'A49_2018', 'A50_2018', 
                'A51_2018', 'A52_2018', 'A53_2018', 'A54_2018', 'A55_2018', 'A56_2018', 'A57_2018', 'A58_2018', 'A59_2018', 'A60_2018', 'A61_2018', 'A62_2018', 'A63_2018', 
                'A00_2019', 'A01_2019', 'A02_2019', 'A03_2019', 'A04_2019', 'A05_2019', 'A06_2019', 'A07_2019', 'A08_2019', 'A09_2019', 'A10_2019', 'A11_2019', 'A12_2019', 
                'A13_2019', 'A14_2019', 'A15_2019', 'A16_2019', 'A17_2019', 'A18_2019', 'A19_2019', 'A20_2019', 'A21_2019', 'A22_2019', 'A23_2019', 'A24_2019', 'A25_2019', 
                'A26_2019', 'A27_2019', 'A28_2019', 'A29_2019', 'A30_2019', 'A31_2019', 'A32_2019', 'A33_2019', 'A34_2019', 'A35_2019', 'A36_2019', 'A37_2019', 'A38_2019', 
                'A39_2019', 'A40_2019', 'A41_2019', 'A42_2019', 'A43_2019', 'A44_2019', 'A45_2019', 'A46_2019', 'A47_2019', 'A48_2019', 'A49_2019', 'A50_2019', 'A51_2019', 
                'A52_2019', 'A53_2019', 'A54_2019', 'A55_2019', 'A56_2019', 'A57_2019', 'A58_2019', 'A59_2019', 'A60_2019', 'A61_2019', 'A62_2019', 'A63_2019']

LABEL_COL = 'label'

def get_fold_assignments(parquet_dir="data/alphaearth_wetland_joined"):
    """
    Reads the directory of joined AE embeddings and partitions the block_id column
    into 5 folds for spatial CV. 
    """
    # load dataset then get list of unique block ids.
    dataset = ds.dataset(parquet_dir, format="parquet")
    block_table = dataset.to_table(columns=["block_id"])
    block_ids = block_table.column("block_id").to_pandas()
    unique_blocks = np.array(block_ids.unique()) 

    # create fold assignments.
    gkf = GroupKFold(n_splits=5)
    # GroupKFold needs X and groups of matching length
    # so we are going to fake X since we only care about the groups
    fold_assignments = {}
    dummy_X = np.zeros(len(unique_blocks))
    for fold_idx, (_, val_idx) in enumerate(gkf.split(dummy_X, groups=unique_blocks)):
        for block in unique_blocks[val_idx]:
            fold_assignments[block] = fold_idx


    return fold_assignments


def report_metrics(save_dir="results/"):
    """
    Reports training and validation metrics, seperately for each trained model.
    Reports confusion matrices, loss curves over epochs, PR-AUC curve showing precision on y-axis and recall on x-axis. 
    """
    save_dir = Path(save_dir)
    fold_dirs = sorted(save_dir.glob("fold_*"))

    summary_rows = []
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name

        # --- loss curve, from the CSVLogger's metrics.csv ---
        metrics_csvs = list(fold_dir.glob("**/metrics.csv"))
        if not metrics_csvs:
            print(f"[{fold_name}] no metrics.csv found, skipping loss curve")
        else:
            df = pd.read_csv(metrics_csvs[0])
            fig, ax = plt.subplots()
            if "train_loss" in df.columns:
                train_df = df.dropna(subset=["train_loss"])
                ax.plot(train_df["epoch"], train_df["train_loss"], label="train_loss")
            if "val_loss" in df.columns:
                val_df = df.dropna(subset=["val_loss"])
                ax.plot(val_df["epoch"], val_df["val_loss"], label="val_loss")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.set_title(f"{fold_name} loss curve")
            ax.legend()
            fig.savefig(fold_dir / "loss_curve.png")
            plt.close(fig)

        # --- confusion matrix + PR curve, from saved val predictions ---
        pred_path = fold_dir / "val_preds.npz"
        if not pred_path.exists():
            print(f"[{fold_name}] no val_preds.npz found, skipping confusion matrix / PR curve")
            continue

        data = np.load(pred_path)
        probs, labels = data["probs"], data["labels"]
        preds = (probs >= 0.5).astype(int)

        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"]).plot(ax=ax, cmap="Blues")
        ax.set_title(f"{fold_name} confusion matrix")
        fig.savefig(fold_dir / "confusion_matrix.png")
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f"AP = {ap:.3f}")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title(f"{fold_name} precision-recall curve")
        ax.legend()
        fig.savefig(fold_dir / "pr_curve.png")
        plt.close(fig)

        summary_rows.append({
            "fold": fold_name,
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "average_precision": ap,
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(save_dir / "summary_metrics.csv", index=False)
        print(summary_df)
        print("\nmean across folds:")
        print(summary_df.drop(columns="fold").mean(numeric_only=True))
        return summary_df
    else:
        print("no fold results found to summarize")
        return None


def get_val_predictions(model, data_module):
    """
    Run trained module over the validation set and collect probs + true labels.
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in data_module.val_dataloader():
            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits)
            all_probs.append(probs)
            all_labels.append(y)

    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def main():
    save_dir = Path("results/")
    save_dir.mkdir(parents=True, exist_ok=True)

    fold_assignments = get_fold_assignments(PARQUET_DIR)
    folds = set(fold_assignments.values())

    results = []
    for fold in folds:
        print(f"ON FOLD {fold}")
        data_module = datasets.SpatialCVDataModule(fold_assignments, fold, FEATURE_COLS, LABEL_COL)
        model = models.SimpleLinearModel(len(FEATURE_COLS))

        fold_name = f"fold_{fold}"
        logger = CSVLogger(save_dir=str(save_dir), name=fold_name)  # writes metrics.csv per epoch

        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=1e-4)
        checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
        trainer = pl.Trainer(max_epochs=15, callbacks=[early_stop, checkpoint], logger=logger, enable_progress_bar=True, accelerator="auto")

        trainer.fit(model, datamodule=data_module)

        best_model = models.SimpleLinearModel.load_from_checkpoint(checkpoint.best_model_path)
        val_metrics = trainer.validate(best_model, datamodule=data_module)
        results.append(val_metrics)

        probs, labels = get_val_predictions(best_model, data_module)
        fold_dir = save_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez(fold_dir / "val_preds.npz", probs=probs, labels=labels)

    print("REPORTING METRICS")
    report_metrics()


if __name__ == "__main__":
    main()