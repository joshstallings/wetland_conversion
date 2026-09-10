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

        # No plain assignment after this: self.pos_weight = pos_weight would shadow
        # the buffer with a bare tensor that does not move with .to(device), which
        # only shows up as a device mismatch once you are off the CPU.
        self.register_buffer("pos_weight", pos_weight)


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
    def __init__(self, n_features, lr=1e-3, threshold=0.5, pos_weight=None, gamma=2.0, alpha=None):
        """
        gamma: focal loss focusing parameter, and the default path. gamma=None
            falls back to nn.BCEWithLogitsLoss; gamma=0.0 is the same thing routed
            through BinaryFocalLoss.

        pos_weight: scalar weight on the positive class, default None. Focal loss
            is already handling the imbalance through (1 - p_t)^gamma, and the
            batches are stratified on top of that, so a pos_weight would be a third
            mechanism aimed at the same thing. Kept for ablations. If you do pass
            one, use ArrayDataModule.batch_pos_weight (31 at 64 positives in 2048),
            not the population ratio of 356, which would overshoot by 11.5x. Under
            focal loss alpha is the better knob anyway, being bounded in [0, 1].

        Batches arrive as (emb, dist, y): float16 embeddings, float32 distance in
        train fold normalized units, float32 binary label. forward assembles the
        193 column design matrix on device, see _assemble.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.model = nn.Sequential(
            nn.Linear(n_features, 1)
        )

        # be wary of passing pos_weight AND gamma
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



    def _assemble(self, emb, dist):
        """
        The (B, 193) design matrix from the two tensors a batch arrives as.

        dist goes in column 0 so the column order matches features.FEATURE_COLS,
        whose first entry is dist_to_developed_2019_m. That keeps the fitted
        coefficients readable straight against the feature list.

        The float16 to float32 cast lives here, after the device transfer, rather
        than in the dataset: on the GPU it is free, while doing it CPU side costs
        1.26 ms per batch against a 0.62 ms gather. On a CPU accelerator it costs
        the same wherever you put it, so this is one of the places the T4 pays off.
        """
        return torch.cat([dist.unsqueeze(1), emb.float()], dim=1)

    def forward(self, emb, dist):
        return self.model(self._assemble(emb, dist))

    def training_step(self, batch, batch_idx):
        emb, dist, y = batch
        logits = self(emb, dist).squeeze(1) # rmv dim 1 at idx 1 from tensor
        loss = self.loss_fn(logits, y)

        # These come off stratified batches at a 1:31 positive rate while their val
        # counterparts come off data at 1:356, so train_precision will look far
        # better than val_precision for reasons that have nothing to do with fit.
        # Read them as "is it learning anything", and compare train against val
        # through the natural rate train_eval pass in train.py.
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
        emb, dist, y = batch
        logits = self(emb, dist).squeeze(1)
        loss = self.loss_fn(logits, y)

        probs = torch.sigmoid(logits)
        self.val_precision(probs, y.int())
        self.val_recall(probs, y.int())
        self.val_f1(probs, y.int())

        # batch_size is passed explicitly because val batches are ragged: chunks
        # never straddle a block boundary, so every val block ends in a short
        # batch. Without it Lightning weights each batch equally in the epoch mean
        # and a 40 row tail counts as much as an 8,192 row batch, which moves the
        # val_loss that EarlyStopping is watching.
        n = y.numel()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=n)
        self.log("val_precision", self.val_precision, on_epoch=True, batch_size=n)
        self.log("val_recall", self.val_recall, on_epoch=True, batch_size=n)
        self.log("val_f1", self.val_f1, on_epoch=True, batch_size=n)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self, n_features, dropout=0.3, lr=1e-3, threshold=0.5,
                 pos_weight=None, gamma=2.0, alpha=None, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.loss_fn = BinaryFocalLoss(gamma=gamma, alpha=alpha, pos_weight=pos_weight)

        self.train_precision = BinaryPrecision(threshold=threshold)
        self.train_recall = BinaryRecall(threshold=threshold)
        self.train_f1 = BinaryF1Score(threshold=threshold)

        self.val_precision = BinaryPrecision(threshold=threshold)
        self.val_recall = BinaryRecall(threshold=threshold)
        self.val_f1 = BinaryF1Score(threshold=threshold)

    def _assemble(self, emb, dist):
        return torch.cat([dist.unsqueeze(1), emb.float()], dim=1)

    def forward(self, emb, dist):
        return self.model(self._assemble(emb, dist))

    def training_step(self, batch, batch_idx):
        emb, dist, y = batch
        logits = self(emb, dist).squeeze(1)
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
        emb, dist, y = batch
        logits = self(emb, dist).squeeze(1)
        loss = self.loss_fn(logits, y)

        probs = torch.sigmoid(logits)
        self.val_precision(probs, y.int())
        self.val_recall(probs, y.int())
        self.val_f1(probs, y.int())

        n = y.numel()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=n)
        self.log("val_precision", self.val_precision, on_epoch=True, batch_size=n)
        self.log("val_recall", self.val_recall, on_epoch=True, batch_size=n)
        self.log("val_f1", self.val_f1, on_epoch=True, batch_size=n)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)