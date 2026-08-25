""" This file to define a simple 1 layer linear neural network model, and in the future more complex models. """

import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from scripts.model import model_common


class StratifiedBatchSampler(Sampler):
    """ This class ensures we get a certain number of positives in a batch. Specifically, we want to ensure label=1 
    instances make it, and label=0 but near development make it in. """

    def __init__(self, y, dist_norm, batch_size, pos_frac=0.1, near_dev_frac=0.1,
                 near_dev_z=-1.0, n_batches=None):
        
        y, dist_norm = np.asarray(y), np.asarray(dist_norm)
        self.pos_idx = np.flatnonzero(y == 1)

        # near_dev_z=-1 means '1 std closer than average'
        self.near_dev_idx = np.flatnonzero((y==0) & (dist_norm <= near_dev_z)) 
        self.other_idx = np.flatnonzero((y==0) & (dist_norm > near_dev_z))

        self.n_pos = max(1, int(batch_size * pos_frac))
        self.n_near = max(1, int(batch_size * near_dev_frac))
        self.n_other = batch_size - self.n_pos - self.n_near

        self.n_batches = n_batches or len(y) // batch_size # batch size.

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self.n_batches):
            batch = np.concatenate([
                rng.choice(self.pos_idx, self.n_pos),
                rng.choice(self.near_dev_idx, self.n_near),
                rng.choice(self.other_idx, self.n_other, replace=len(self.other_idx) < self.n_other)
            ])

            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.n_batches


class SimpleLinearModel(pl.LightningModule):
    def __init__(self, n_features: int):
        super().__init__()

        # Omitting sigmoid in this section
        # If you compute sigmoid and BCE seperately, extreme logits break it
        # Say you get a logit as +30, sigmoid(30) is 1 and then log(1-1) is -inf 
        # which turns to NaN. By omitting sigmoid, I use BCEWithLogitsLoss which computes loss
        # directly from the logit. 
        self.model = nn.Sequential(
            nn.Linear(n_features, 1)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        logits = self(x).squeeze(1) # remove dim 1 at idx 1 from tensor
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = (loss * w).sum() / w.sum()
        # on_step=False, on_epoch=True: one averaged row per epoch in the CSV log,
        # not one row per batch, so the loss curve plot reads cleanly
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        logits = self(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = (loss * w).sum() / w.sum()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(X.to_numpy(dtype="float32"))
            p = torch.sigmoid(self(x).squeeze(-1)).numpy()

        return np.stack([1 - p, p], axis=1)

    def fit(self, X, y, sample_weight=None, eval_X=None, eval_y=None, max_epochs=20, batch_size=4096,
            run_dir=None, run_name=None, title=None):
        """
        Wrapper function that conforms to sklearn's fit method so trian.py's cmd_baseline
        method can call this function seamlessly without knowing that pl.Trainer is under the hood.

        eval_X/eval_y, if given, are scored every epoch as validation loss (unweighted, since
        they're expected to already be a natural, unweighted sample, same as LightGBM's eval_X/
        eval_y in train.py's grid_split). run_dir/run_name/title control where the loss curve
        (and the CSV log behind it) get written; without run_dir nothing is saved to disk, only
        trained.
        """

        x_t = torch.as_tensor(X.to_numpy(dtype="float32"))
        y_t = torch.as_tensor(y.to_numpy(dtype="float32"))
        w_t = torch.as_tensor(
            np.asarray(sample_weight, dtype="float32") if sample_weight is not None else np.ones(len(y), dtype="float32" )
        )
        dist_norm = X[model_common.DIST_NORM_COL].to_numpy()

        batch_sampler = StratifiedBatchSampler(y.to_numpy(), dist_norm, batch_size)

        # num_workers=0, persistent_workers=True if dataset larger than memory
        loader = DataLoader(TensorDataset(x_t, y_t, w_t), batch_sampler=batch_sampler)

        val_loader = None
        if eval_X is not None and eval_y is not None:
            ex_t = torch.as_tensor(eval_X.to_numpy(dtype="float32"))
            ey_t = torch.as_tensor(eval_y.to_numpy(dtype="float32"))
            ew_t = torch.ones(len(eval_y), dtype=torch.float32)
            # no stratified sampling here, just plain batches: this is a loss readout, not a
            # training signal, so it should see the natural class balance as-is
            val_loader = DataLoader(TensorDataset(ex_t, ey_t, ew_t), batch_size=batch_size, shuffle=False)

        # explicit CSVLogger instead of logger=True: pins the metrics.csv this run writes to a
        # known path under run_dir rather than an auto-incrementing lightning_logs/version_N at
        # the repo root that nothing points back to
        csv_logger = CSVLogger(save_dir=run_dir or "lightning_logs", name="loss_logs", version=run_name)

        # put accelerator="mps" devices=1
        trainer = pl.Trainer(max_epochs=max_epochs, enable_progress_bar=True, logger=csv_logger,
                             enable_checkpointing=False)
        if val_loader is not None:
            trainer.fit(self, train_dataloaders=loader, val_dataloaders=val_loader)
        else:
            trainer.fit(self, loader)

        if run_dir is not None:
            _plot_loss_curve(csv_logger.log_dir, os.path.join(run_dir, f"loss_curve_{run_name}.png"), title)

        return self


def _plot_loss_curve(log_dir: str, save_path: str, title: str | None) -> None:
    """Read Lightning's own metrics.csv back out of log_dir and plot train_loss
    against val_loss by epoch. Both columns are logged on_epoch only, so each
    metric has one row per epoch, on different rows since Lightning writes a
    training epoch and a validation epoch as separate log entries; groupby
    collapses that back to one row per epoch.
    """
    metrics = pd.read_csv(os.path.join(log_dir, "metrics.csv"))
    by_epoch = metrics.groupby("epoch").mean(numeric_only=True)

    with plt.rc_context(model_common.SLIDE_RC):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        if "train_loss" in by_epoch:
            ax.plot(by_epoch.index, by_epoch["train_loss"], label="train", color=model_common.FOLD_COLORS[0])
        if "val_loss" in by_epoch and by_epoch["val_loss"].notna().any():
            ax.plot(by_epoch.index, by_epoch["val_loss"], label="val", color=model_common.FOLD_COLORS[1])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (weighted BCE)")
        ax.set_title(title or "Train and validation loss")
        ax.legend(loc="upper right")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {save_path}")


def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()