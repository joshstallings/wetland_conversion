"""
This file will contain classes for architectures used in this project,
starting with a basic linear model.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics import MeanMetric, SumMetric
from torchmetrics.classification import (BinaryAveragePrecision, BinaryF1Score, BinaryPrecision,
                                         BinaryRecall)


def tcn_dilation(n_years, dilation=None):
    """
    Two kernel 2 convs with no padding give a receptive field of 2 + dilation
    and an output length of n_years - 1 - dilation. Setting dilation to
    n_years - 2 is the point where the receptive field covers the whole
    sequence and the length collapses to 1: the two are mutually exclusive
    without padding, so there is nothing to trade off, it is the only
    setting that sees all the years. At n_years 3 it works out to 1, which
    is what the three year runs were trained with.
    """
    if dilation is None:
        dilation = max(1, n_years - 2)
    if n_years - 1 - dilation < 1:
        raise ValueError(
            f"dilation {dilation} leaves no sequence length at n_years {n_years}. "
            f"The second conv needs n_years - 1 - dilation >= 1."
        )
    return dilation


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


def binary_loss(gamma, alpha=None, pos_weight=None):
    """
    Focal loss, or plain nn.BCEWithLogitsLoss when gamma is None. alpha only
    exists in the focal form, so passing it with gamma None is an error rather
    than a knob that silently does nothing.
    """
    if pos_weight is not None and not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
    if gamma is None:
        if alpha is not None:
            raise ValueError("alpha is a focal loss parameter, it has no effect with gamma None")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return BinaryFocalLoss(gamma=gamma, alpha=alpha, pos_weight=pos_weight)


def _cell_tcn_block(emb_dim, hidden_dim, kernel_size, dilation):
    # 2 * emb_dim in: means and stds share a year as one set of channels.
    return nn.Sequential(nn.Conv1d(2 * emb_dim, hidden_dim, kernel_size=kernel_size),
                         nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                         nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                                   dilation=dilation),
                         nn.BatchNorm1d(hidden_dim), nn.ReLU())


def _cell_sequence(emb_mean, emb_std, n_years, emb_dim):
    """(B, n_years * emb_dim) year major means and stds to the (B, 2 * emb_dim,
    n_years) layout Conv1d wants, means first."""
    for name, t in (("emb_mean", emb_mean), ("emb_std", emb_std)):
        if t.shape[1] != n_years * emb_dim:
            raise ValueError(
                f"{name} has {t.shape[1]} columns against n_years {n_years} times "
                f"emb_dim {emb_dim}. Pass the n_years the cells were built with."
            )
    mean = emb_mean.float().view(-1, n_years, emb_dim)
    std = emb_std.float().view(-1, n_years, emb_dim)
    return torch.cat([mean, std], dim=2).transpose(1, 2)


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

        self.loss_fn = binary_loss(gamma, alpha, pos_weight)

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
    
class TemporalConvolutionalNetwork(pl.LightningModule):
    def __init__(self, emb_dim=64, n_years=3, hidden_dim=128, dropout=0.3, lr=1e-3,
                 threshold=0.5, pos_weight=None, gamma=2.0, alpha=None, weight_decay=0.0,
                 kernel_size=2, dilation=None):
        super().__init__()
        dilation = tcn_dilation(n_years, dilation)
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        self.tcn_block = nn.Sequential(nn.Conv1d(emb_dim, hidden_dim, kernel_size=kernel_size),
                                       nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                       nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                                                 dilation=dilation),
                                       nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.model = nn.Sequential(
            nn.Linear(hidden_dim+1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.loss_fn = binary_loss(gamma, alpha, pos_weight)
        self.train_precision = BinaryPrecision(threshold=threshold)
        self.train_recall = BinaryRecall(threshold=threshold)
        self.train_f1 = BinaryF1Score(threshold=threshold)

        self.val_precision = BinaryPrecision(threshold=threshold)
        self.val_recall = BinaryRecall(threshold=threshold)
        self.val_f1 = BinaryF1Score(threshold=threshold)

    def forward(self, emb, dist):
        # emb arrives flat, (B, n_years * emb_dim), laid out year major because
        # that is the order features.emb_cols generates. Reshape to (B, year, dim)
        # then transpose, bc Conv1D takes (channels, length).
        emb = emb.float()
        n_years, emb_dim = self.hparams.n_years, self.hparams.emb_dim
        if emb.shape[1] != n_years * emb_dim:
            # view would happily fold the extra years into the batch dimension and
            # only blow up later at the cat in the head, so catch it here.
            raise ValueError(
                f"emb has {emb.shape[1]} columns against n_years {n_years} times "
                f"emb_dim {emb_dim}. Pass the n_years the arrays were built with."
            )
        reshaped_emb = emb.view(-1, n_years, emb_dim)
        transpose_emb = reshaped_emb.transpose(1, 2)
        tcn_out = self.tcn_block(transpose_emb)

        aap_output = self.pool(tcn_out).squeeze(-1)

        head_input = torch.cat([dist.unsqueeze(1), aap_output], dim=1)
        return self.model(head_input)


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

class CellTemporalConvolutionalNetwork(pl.LightningModule):
    """
    The pixel TCN reworked for cells: predicts the share of a cell's 2019 wetland
    that is developed by 2024, from per cell embedding means and stds.

    Batches arrive as (emb_mean, emb_std, scalars, y, n_wetland) from
    datasets.CellDataModule. emb_mean and emb_std are each (B, n_years * emb_dim),
    year major. They go into the conv as 2 * emb_dim channels per year, means
    first, so the sequence is still n_years long and the dilation logic carries
    over unchanged.

    Output is sigmoid(logit), so predictions stay in [0, 1]. Loss is MSE on that,
    or with binomial_loss set, BCE on the logit with the share y as a soft target.
    The binomial version is the log likelihood of n_developed out of n_wetland
    coin flips once it is weighted by n_wetland. It also drops the p(1 - p) factor
    MSE picks up through the sigmoid, which mutes converted cells the model scores
    near 0.

    Either loss is weighted by n_wetland when weight_by_n_wetland is set.
    Unweighted, a 1 pixel cell that is either 0 or 1 counts as much as a 100 pixel
    cell, and those small cells carry most of the variance in the target.

    val_mse_weighted and val_log_loss_weighted are logged the same way under every
    flag combination, so runs can be compared on them. val_loss is whatever the
    run trained on and is not comparable across flags.
    """

    def __init__(self, emb_dim=64, n_years=3, n_scalars=3, hidden_dim=128, dropout=0.3,
                 lr=1e-3, weight_decay=0.0, kernel_size=2, dilation=None,
                 weight_by_n_wetland=True, binomial_loss=False):
        super().__init__()
        dilation = tcn_dilation(n_years, dilation)
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        self.tcn_block = _cell_tcn_block(emb_dim, hidden_dim, kernel_size, dilation)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.model = nn.Sequential(
            nn.Linear(hidden_dim + n_scalars, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )


        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_mse_weighted = MeanMetric()
        self.val_log_loss_weighted = MeanMetric()
        self.val_mae = MeanMetric()
        self.val_mae_weighted = MeanMetric()

        self.val_auprc = BinaryAveragePrecision()
        self.val_pos_rate = MeanMetric()

        self.val_pred_pixels = SumMetric()
        self.val_true_pixels = SumMetric()

    def logits(self, emb_mean, emb_std, scalars):
        x = _cell_sequence(emb_mean, emb_std, self.hparams.n_years, self.hparams.emb_dim)
        pooled = self.pool(self.tcn_block(x)).squeeze(-1)
        return self.model(torch.cat([scalars, pooled], dim=1)).squeeze(1)

    def forward(self, emb_mean, emb_std, scalars):
        # Probabilities, not logits, so reporting.get_cell_predictions and the
        # checkpoints trained before binomial_loss existed read the same way.
        return torch.sigmoid(self.logits(emb_mean, emb_std, scalars))

    def _weights(self, n_wetland):
        if self.hparams.weight_by_n_wetland:
            return n_wetland.float()
        return torch.ones_like(n_wetland, dtype=torch.float32)

    def _per_cell_loss(self, logits, y):
        # The logits form of BCE, not sigmoid then log: a saturated sigmoid would
        # hand log a 0. A perfect prediction still scores y log y plus the same
        # for 1 - y here, not 0, but that floor carries no gradient.
        if self.hparams.binomial_loss:
            return F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        return (torch.sigmoid(logits) - y) ** 2

    def training_step(self, batch, batch_idx):
        emb_mean, emb_std, scalars, y, n_wetland = batch
        logits = self.logits(emb_mean, emb_std, scalars)
        w = self._weights(n_wetland)
        per_cell = self._per_cell_loss(logits, y)
        loss = (w * per_cell).sum() / w.sum()

        self.train_loss(per_cell.detach(), weight=w)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        emb_mean, emb_std, scalars, y, n_wetland = batch
        logits = self.logits(emb_mean, emb_std, scalars)
        pred = torch.sigmoid(logits)
        w = self._weights(n_wetland)
        n_w = n_wetland.float()
        err = pred - y

        self.val_loss(self._per_cell_loss(logits, y), weight=w)
        self.val_mse_weighted(err ** 2, weight=n_w)
        self.val_log_loss_weighted(
            F.binary_cross_entropy_with_logits(logits, y, reduction="none"), weight=n_w
        )
        self.val_mae(err.abs())
        self.val_mae_weighted(err.abs(), weight=n_w)
        any_dev = (y > 0).int()
        self.val_auprc(pred, any_dev)
        self.val_pos_rate(any_dev.float())
        self.val_pred_pixels(pred * n_w)
        self.val_true_pixels(y * n_w)

        self.log("val_loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.log("val_mse_weighted", self.val_mse_weighted, on_epoch=True)
        self.log("val_log_loss_weighted", self.val_log_loss_weighted, on_epoch=True)
        self.log("val_mae", self.val_mae, on_epoch=True)
        self.log("val_mae_weighted", self.val_mae_weighted, on_epoch=True)
        self.log("val_auprc", self.val_auprc, on_epoch=True, prog_bar=True)
        self.log("val_pos_rate", self.val_pos_rate, on_epoch=True)

    def on_validation_epoch_end(self):
        true = self.val_true_pixels.compute()
        ratio = self.val_pred_pixels.compute() / true.clamp(min=1.0)
        self.log("val_pred_over_actual", ratio)
        self.val_pred_pixels.reset()
        self.val_true_pixels.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class CellTemporalConvolutionalLogShare(CellTemporalConvolutionalNetwork):
    """
    Stage two of the hurdle model: the natural log of the share of a cell's 2019
    wetland developed by 2024, given that some of it was. Same trunk, head and
    batches as CellTemporalConvolutionalNetwork, with no sigmoid on the output and
    MSE in log space, every cell counted once.

    Only feed it cells with n_developed > 0 (CellDataModule with positive_only),
    otherwise log(y) is -inf and the loss goes with it.

    forward returns the predicted log share. exp of it is a geometric mean, which
    sits below the mean share, so val_pred_over_actual reads low until a smearing
    factor is applied (reporting.smearing_factor, done after training).
    """

    def __init__(self, emb_dim=64, n_years=3, n_scalars=3, hidden_dim=128, dropout=0.3,
                 lr=1e-3, weight_decay=0.0, kernel_size=2, dilation=None):
        super().__init__(emb_dim=emb_dim, n_years=n_years, n_scalars=n_scalars,
                         hidden_dim=hidden_dim, dropout=dropout, lr=lr,
                         weight_decay=weight_decay, kernel_size=kernel_size, dilation=dilation,
                         weight_by_n_wetland=False, binomial_loss=False)

    def forward(self, emb_mean, emb_std, scalars):
        return self.logits(emb_mean, emb_std, scalars)

    def training_step(self, batch, batch_idx):
        emb_mean, emb_std, scalars, y, _ = batch
        loss = ((self.logits(emb_mean, emb_std, scalars) - torch.log(y)) ** 2).mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=y.numel())
        return loss

    def validation_step(self, batch, batch_idx):
        emb_mean, emb_std, scalars, y, n_wetland = batch
        pred = self.logits(emb_mean, emb_std, scalars)
        n_w = n_wetland.float()
        err = pred - torch.log(y)

        self.val_loss(err ** 2)
        self.val_mae(err.abs())
        self.val_mae_weighted(err.abs(), weight=n_w)
        self.val_pred_pixels(torch.exp(pred) * n_w)
        self.val_true_pixels(y * n_w)

        self.log("val_loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.val_mae, on_epoch=True)
        self.log("val_mae_weighted", self.val_mae_weighted, on_epoch=True)


class CellTemporalConvolutionalClassifier(pl.LightningModule):
    """
    Stage one of the hurdle model: does a cell see any 2019 wetland developed by
    2024. Same trunk, inputs and batches as CellTemporalConvolutionalNetwork, so
    both stages read one cell artifact through one CellDataModule.

    y arrives as the share and is binarized here as y > 0. n_wetland comes along
    in the batch and is ignored: whether a cell converts at all is one event per
    cell however big the cell is.

    Loss is focal by default, plain BCE with gamma None. Neither leaves sigmoid
    output calibrated to the natural rate once batches are stratified, and focal
    loss does not even at the natural rate, so the probabilities rank cells but
    are not yet the p(any) a hurdle product needs.
    """

    def __init__(self, emb_dim=64, n_years=3, n_scalars=3, hidden_dim=128, dropout=0.3,
                 lr=1e-3, weight_decay=0.0, kernel_size=2, dilation=None,
                 gamma=2.0, alpha=None, pos_weight=None):
        super().__init__()
        dilation = tcn_dilation(n_years, dilation)
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        self.tcn_block = _cell_tcn_block(emb_dim, hidden_dim, kernel_size, dilation)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.model = nn.Sequential(
            nn.Linear(hidden_dim + n_scalars, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.loss_fn = binary_loss(gamma, alpha, pos_weight)

        self.val_auprc = BinaryAveragePrecision()
        self.val_pos_rate = MeanMetric()

    def logits(self, emb_mean, emb_std, scalars):
        x = _cell_sequence(emb_mean, emb_std, self.hparams.n_years, self.hparams.emb_dim)
        pooled = self.pool(self.tcn_block(x)).squeeze(-1)
        return self.model(torch.cat([scalars, pooled], dim=1)).squeeze(1)

    def forward(self, emb_mean, emb_std, scalars):
        # Probabilities, so reporting.get_cell_predictions reads this the same way
        # it reads the regressor.
        return torch.sigmoid(self.logits(emb_mean, emb_std, scalars))

    def training_step(self, batch, batch_idx):
        emb_mean, emb_std, scalars, y, _ = batch
        any_dev = (y > 0).float()
        loss = self.loss_fn(self.logits(emb_mean, emb_std, scalars), any_dev)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=y.numel())
        return loss

    def validation_step(self, batch, batch_idx):
        emb_mean, emb_std, scalars, y, _ = batch
        any_dev = (y > 0).float()
        logits = self.logits(emb_mean, emb_std, scalars)
        loss = self.loss_fn(logits, any_dev)

        self.val_auprc(torch.sigmoid(logits), any_dev.int())
        self.val_pos_rate(any_dev)

        # Every val sweep ends in a short batch, so batch_size keeps the epoch mean
        # weighted by cells rather than by batches.
        n = y.numel()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=n)
        self.log("val_auprc", self.val_auprc, on_epoch=True, prog_bar=True)
        self.log("val_pos_rate", self.val_pos_rate, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class CellMultiLayerPerceptronClassifier(pl.LightningModule):
    """
    Stage one hurdle classifier with no temporal structure: year means, year stds
    and the cell scalars go in as one flat vector. Baseline for
    CellTemporalConvolutionalClassifier, same batches and same loss, so the gap
    between the two is what the convolution over years buys.
    """

    def __init__(self, emb_dim=64, n_years=3, n_scalars=3, dropout=0.3, lr=1e-3,
                 weight_decay=0.0, gamma=2.0, alpha=None, pos_weight=None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        n_features = 2 * n_years * emb_dim + n_scalars
        self.model = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.loss_fn = binary_loss(gamma, alpha, pos_weight)

        self.val_auprc = BinaryAveragePrecision()
        self.val_pos_rate = MeanMetric()

    def logits(self, emb_mean, emb_std, scalars):
        # _cell_sequence for its column count check, then flattened straight back.
        x = _cell_sequence(emb_mean, emb_std, self.hparams.n_years, self.hparams.emb_dim)
        return self.model(torch.cat([scalars, x.flatten(1)], dim=1)).squeeze(1)

    def forward(self, emb_mean, emb_std, scalars):
        return torch.sigmoid(self.logits(emb_mean, emb_std, scalars))

    def training_step(self, batch, batch_idx):
        emb_mean, emb_std, scalars, y, _ = batch
        any_dev = (y > 0).float()
        loss = self.loss_fn(self.logits(emb_mean, emb_std, scalars), any_dev)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=y.numel())
        return loss

    def validation_step(self, batch, batch_idx):
        emb_mean, emb_std, scalars, y, _ = batch
        any_dev = (y > 0).float()
        logits = self.logits(emb_mean, emb_std, scalars)
        loss = self.loss_fn(logits, any_dev)

        self.val_auprc(torch.sigmoid(logits), any_dev.int())
        self.val_pos_rate(any_dev)

        n = y.numel()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=n)
        self.log("val_auprc", self.val_auprc, on_epoch=True, prog_bar=True)
        self.log("val_pos_rate", self.val_pos_rate, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


MIL_POOLINGS = ("attention", "max", "mean")


class CellAttentionMILClassifier(pl.LightningModule):
    """
    Stage one of the hurdle model as multiple instance learning: a cell is a bag
    of its 30 m wetland pixels, and the label is whether any of them is developed
    by 2024. Only the cell label is used, never which pixel converted.

    Each pixel goes through the same TCN trunk as TemporalConvolutionalNetwork
    plus its own dist, then the pixels are pooled to one vector per cell and the
    head sees that next to the cell scalars. log1p_n_wetland stays in the scalars
    because attention and mean pooling both hide how big the bag was.

    pooling is gated attention (Ilse, Tomczak and Welling 2018) by default. max
    matches the label's "any pixel" logic more literally but sends gradient to one
    pixel per cell, and mean is the closest thing to the averaging this is meant
    to get away from, so both are ablations.

    Batches are packed pixels from datasets.BagDataset. The encoder runs on the
    packed (N, ...) tensor so its BatchNorm never sees padding, and only the
    encoded pixels get padded to (B, max bag, hidden).
    """

    def __init__(self, emb_dim=64, n_years=3, n_scalars=3, hidden_dim=128, attn_dim=64,
                 dropout=0.3, lr=1e-3, weight_decay=0.0, kernel_size=2, dilation=None,
                 pooling="attention", gamma=2.0, alpha=None, pos_weight=None):
        super().__init__()
        if pooling not in MIL_POOLINGS:
            raise ValueError(f"pooling {pooling} is not one of {MIL_POOLINGS}")
        dilation = tcn_dilation(n_years, dilation)
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        self.tcn_block = nn.Sequential(nn.Conv1d(emb_dim, hidden_dim, kernel_size=kernel_size),
                                       nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                       nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                                                 dilation=dilation),
                                       nn.BatchNorm1d(hidden_dim), nn.ReLU())
        # One layer after the dist concat so a pixel's distance can interact with
        # its embedding before pooling, rather than only ever being averaged in.
        self.instance = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.ReLU())

        self.attn_v = nn.Linear(hidden_dim, attn_dim)
        self.attn_u = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

        self.model = nn.Sequential(
            nn.Linear(hidden_dim + n_scalars, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.loss_fn = binary_loss(gamma, alpha, pos_weight)

        self.val_auprc = BinaryAveragePrecision()
        self.val_pos_rate = MeanMetric()

    def encode(self, emb, dist):
        """(N, n_years * emb_dim) year major pixels and (N,) dist to (N, hidden_dim)."""
        n_years, emb_dim = self.hparams.n_years, self.hparams.emb_dim
        if emb.shape[1] != n_years * emb_dim:
            raise ValueError(
                f"emb has {emb.shape[1]} columns against n_years {n_years} times "
                f"emb_dim {emb_dim}. Pass the n_years the arrays were built with."
            )
        x = emb.float().view(-1, n_years, emb_dim).transpose(1, 2)
        h = self.tcn_block(x).mean(dim=2)
        return self.instance(torch.cat([h, dist.float().unsqueeze(1)], dim=1))

    def pool(self, h, bag_id, pos, n_bags):
        """Pooled (B, hidden_dim) and the (B, max bag) attention weights, which are
        None unless pooling is attention. Padded slots get weight exactly 0."""
        n_slots = int(pos.max()) + 1
        padded = h.new_zeros(n_bags, n_slots, h.shape[1])
        padded[bag_id, pos] = h
        mask = torch.zeros(n_bags, n_slots, dtype=torch.bool, device=h.device)
        mask[bag_id, pos] = True

        if self.hparams.pooling == "max":
            return padded.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1), None
        if self.hparams.pooling == "mean":
            return padded.sum(dim=1) / mask.sum(dim=1, keepdim=True), None

        # Scored on the packed pixels, then scattered into a -inf grid, so padding
        # never goes through the attention layers. Every bag has at least one
        # pixel, so no row of the softmax is all -inf.
        a = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))).squeeze(1)
        scores = h.new_full((n_bags, n_slots), float("-inf"))
        scores[bag_id, pos] = a
        weights = torch.softmax(scores, dim=1)
        return (weights.unsqueeze(-1) * padded).sum(dim=1), weights

    def logits(self, emb, dist, bag_id, pos, scalars, return_attention=False):
        pooled, weights = self.pool(self.encode(emb, dist), bag_id, pos, scalars.shape[0])
        out = self.model(torch.cat([scalars, pooled], dim=1)).squeeze(1)
        return (out, weights) if return_attention else out

    def forward(self, emb, dist, bag_id, pos, scalars):
        return torch.sigmoid(self.logits(emb, dist, bag_id, pos, scalars))

    def training_step(self, batch, batch_idx):
        emb, dist, bag_id, pos, scalars, y, _ = batch
        any_dev = (y > 0).float()
        loss = self.loss_fn(self.logits(emb, dist, bag_id, pos, scalars), any_dev)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=y.numel())
        return loss

    def validation_step(self, batch, batch_idx):
        emb, dist, bag_id, pos, scalars, y, _ = batch
        any_dev = (y > 0).float()
        logits = self.logits(emb, dist, bag_id, pos, scalars)
        loss = self.loss_fn(logits, any_dev)

        self.val_auprc(torch.sigmoid(logits), any_dev.int())
        self.val_pos_rate(any_dev)

        n = y.numel()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=n)
        self.log("val_auprc", self.val_auprc, on_epoch=True, prog_bar=True)
        self.log("val_pos_rate", self.val_pos_rate, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
