"""Feature standardization transforms (standalone reimplementation).

``StandardScalar`` standardizes named tensor channels to zero mean / unit std,
with the (mu, sd) fit from data and stored as buffers so they bake into a
checkpoint. Reimplemented from the internal zeus transform so the student model
runs without zeus; the fit/apply math is identical.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl


def _nanstd(x: torch.Tensor, dim: int) -> torch.Tensor:
    mean = torch.nanmean(x, dim=dim, keepdim=True)
    var = torch.nanmean((x - mean) ** 2, dim=dim)
    return torch.sqrt(var)


class BaseTransform(pl.LightningModule):
    def __init__(self, keys=None, ckpt_path=None, **kwargs):
        super().__init__()
        self.checkpoint_path = ckpt_path
        self.keys = keys

    def forward(self, sample):
        return sample

    def backward(self, y):
        return y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)


class StandardScalar(BaseTransform):
    """Per-channel (mean, std) standardization along ``dim`` for named keys."""

    def __init__(self, keys, dim: int = 1, **kwargs):
        super().__init__(keys=keys, **kwargs)
        self.dim = dim
        self.mu = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(d), requires_grad=False)
            for k, d in self.keys.items()
        })
        self.sd = nn.ParameterDict({
            k: nn.Parameter(torch.ones(d), requires_grad=False)
            for k, d in self.keys.items()
        })

    def forward(self, sample):
        for k in sample.keys():
            if k not in self.keys:
                continue
            sample[k] = sample[k].swapaxes(-1, self.dim)
            sample[k] = (sample[k] - self.mu[k]) / self.sd[k]
            sample[k] = sample[k].swapaxes(-1, self.dim)
        return sample

    def backward(self, batch):
        y = {}
        for k in batch.keys():
            x = batch[k].swapaxes(-1, self.dim)
            x = x * self.sd[k] + self.mu[k]
            y[k] = x.swapaxes(-1, self.dim)
        return y

    def training_step(self, batch, batch_idx):
        """Fit (mu, sd) per channel from a large concatenated batch."""
        for k in self.keys:
            x = batch[k]
            x = x.swapaxes(0, self.dim)
            x = x.reshape(x.shape[0], -1)
            self.mu[k] = nn.Parameter(torch.nanmean(x, dim=1), requires_grad=False)
            self.sd[k] = nn.Parameter(_nanstd(x, dim=1), requires_grad=False)
