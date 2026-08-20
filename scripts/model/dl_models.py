""" This file to define a simple 1 layer linear neural network model, and in the future more complex models. """

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, TensorDataset
import numpy as np
from torch.optim import Adam
import pytorch_lightning as pl 

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

        self.n_batches = n_batches or len(y) # batch size.

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
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        logits = self(x).squeeze(1) 
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = (loss * w).sum() / w.sum()
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(X.to_numpy(dtype="float32"))
            p = torch.sigmoid(self(x).squeeze(-1)).numpy()

        return np.stack([1 - p, p], axis=1)

    def fit(self, X, y, sample_weight=None, max_epochs=20, batch_size=4096):
        """ 
        Wrapper function that conforms to sklearn's fit method so trian.py's cmd_baseline
        method can call this function seamlessly without knowing that pl.Trainer is under the hood. 
        """

        x_t = torch.as_tensor(X.to_numpy(dtype="float32"))
        y_t = torch.as_tensor(y.to_numpy(dtype="float32"))
        w_t = torch.as_tensor(
            np.asarray(sample_weight, dtype="float32") if sample_weight is not None else np.ones(len(y), dtype="float32" )
        )
        dist_norm = X[model_common.DIST_NORM_COL].to_numpy()

        batch_sampler = StratifiedBatchSampler(y.to_numpy(), dist_norm, batch_size)
        loader = DataLoader(TensorDataset(x_t, y_t, w_t), batch_sampler=batch_sampler, num_workers=9, persistent_workers=True)

        trainer = pl.Trainer(max_epochs=max_epochs, enable_progress_bar=False, logger=False, enable_checkpointing=False)
        trainer.fit(self, loader)

        return self

def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()