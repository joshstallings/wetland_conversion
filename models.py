"""
This file will contain classes for architectures used in this project,
starting with a basic linear model.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score


class BinaryFocalLoss(nn.Module):
    """
    Binary focal loss with logits.

    FL(p_t) = alpha_t * (1 - p_t)^gamma * BCE(p_t)

    - gamma: focusing parameter. gamma=0 reduces to (weighted) BCE.
      Higher gamma down-weights easy, well-classified examples more.
    - alpha: optional scalar in [0, 1] weighting the positive class
      (alpha for y=1, 1-alpha for y=0). Leave as None to skip this
      and rely solely on pos_weight, exactly like BCEWithLogitsLoss.
    - pos_weight: same semantics as nn.BCEWithLogitsLoss's pos_weight —
      a scalar multiplier on the positive class term, useful when
      alpha alone isn't enough to compensate a severe imbalance.
    
    """
    def __init__(self, gamma=2.0, alpha=None, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

        self.register_buffer("pos_weight", pos_weight) # moves to to_device() automatically
        self.pos_weight = pos_weight


    def forward(self, logits, targets):
        targets = targets.float()

        # per element BCE 
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )

        # p_t is predicted prob of true class
        p = torch.sigmoid(logits)
        p_t = p * targets + (1-p) * (1 - targets)

        modulating_factor = (1.0 - p_t).clamp(min=0.0) ** self.gamma
        loss = modulating_factor * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "max":
            return loss.max()
        return loss # for self.reduction == "none"



class SimpleLinearModel(pl.LightningModule):
    def __init__(self, n_features, lr=1e-3, threshold=0.5, pos_weight=None, gamma=None, alpha=None):
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

        # be wary of passing poss_weight AND gamma
        # not necessarily wrong but is a form of "double counting" the upweighting
        loss_pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None
        if gamma is not None:
            self.loss_fn = BinaryFocalLoss(gamma=gamma, alpha=alpha, pos_weight=loss_pos_weight)
        else:
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
