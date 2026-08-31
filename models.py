"""
This file will contain classes for architectures used in this project,
starting with a basic linear model.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score


class SimpleLinearModel(pl.LightningModule):
    def __init__(self, n_features, lr=1e-3, threshold=0.5, pos_weight=None):
        """
        pos_weight: scalar weight on the positive class in BCEWithLogitsLoss.
            Needed here because training sees every row with no undersampling
            so positives are ~356:1 outnumbered. Defaults to None (unweighted). 
            Compute it from the true population counts (population_stats.load_population_stats),
            not a fold's counts. 
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.model = nn.Sequential(
            nn.Linear(n_features, 1)
        )
        loss_pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=loss_pos_weight)

        self.train_precision = BinaryPrecision(threshold=threshold)
        self.train_recall = BinaryRecall(threshold=threshold)
        self.train_f1 = BinaryF1Score(threshold=threshold)

        self.val_precision = BinaryPrecision(threshold=threshold)
        self.val_recall = BinaryRecall(threshold=threshold)
        self.val_f1 = BinaryF1Score(threshold=threshold)



    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1) # rmv dim 1 at idx 1 from tensor
        loss = self.loss_fn(logits, y)

        probs = torch.sigmoid(logits)
        self.train_precision(probs, y.int())
        self.train_recall(probs, y.int())
        self.train_f1(probs, y.int())

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train_recall", self.train_recall, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.loss_fn(logits, y)

        probs = torch.sigmoid(logits)
        self.val_precision(probs, y.int())
        self.val_recall(probs, y.int())
        self.val_f1(probs, y.int())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision, on_epoch=True)
        self.log("val_recall", self.val_recall, on_epoch=True)
        self.log("val_f1", self.val_f1, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


def apply_intercept_correction(model, true_negative, true_positive, sample_negative, sample_positive):
    """
    King and Zeng (2001) closed-form intercept correction for rare-event
    logistic regression trained on an artificially balanced sample -- path
    A's negative-undersampled cache, per data_pipeline_recipe_path_a.md step
    3. Shifts the final linear layer's bias in place by
    log(true_negative/true_positive) - log(sample_negative/sample_positive)
    so raw predicted probabilities are calibrated to the true population
    rate instead of the cache's balanced rate. Only touches the bias term;
    apply it once after training, before generating predictions.

    true_negative, true_positive: counts from the full population
        (population_stats.json). sample_negative, sample_positive: counts
        actually kept in the cache (train_pool_cache/manifest.json).
    """
    correction = math.log(true_negative / true_positive) - math.log(sample_negative / sample_positive)
    with torch.no_grad():
        model.model[-1].bias += correction
    return model
