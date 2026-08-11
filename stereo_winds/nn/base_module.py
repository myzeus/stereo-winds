"""Minimal Lightning base module for the wind student (standalone).

Reimplements the pieces of the internal zeus ``BaseLightningModule`` the student
actually uses — optimizer, a ``transform`` hook, a ``mode`` property, data-driven
transform fitting, and ``training/validation_step`` delegating to ``step`` — so
the model trains and runs inference without zeus. The wandb / media / video
logging of the original is intentionally omitted (telemetry, not required).
"""
from __future__ import annotations

import itertools

import torch
from pytorch_lightning import LightningModule


class BaseLightningModule(LightningModule):
    def __init__(self, learning_rate: float = 1e-3, transform=None,
                 task: str = "rad", log_media_step: int = 200,
                 log_metrics_step: int = 50, **kwargs):
        super().__init__()
        self.transform = transform
        if not self.transform:
            self.transform = lambda x: x
        self.learning_rate = learning_rate
        self.task = task
        self.log_media_step = log_media_step
        self.log_metrics_step = log_metrics_step

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-6,
            betas=(0.5, 0.9))

    def get_trainer(self):
        try:
            return getattr(self, "trainer", None)
        except RuntimeError:  # not attached to a trainer
            return None

    @property
    def mode(self):
        trainer = self.get_trainer()
        if trainer is None:
            return None
        if getattr(trainer, "sanity_checking", False):
            return None
        if getattr(trainer, "testing", False):
            return "test"
        if getattr(trainer, "predicting", False):
            return "predict"
        if getattr(trainer, "evaluating", False):
            return "eval"
        if getattr(trainer, "training", False):
            return "train"
        return None

    def prepare_data_transformation(self, dataloader, n_batches: int = 1000):
        """Fit the transform on up to ``n_batches`` of the training data."""
        if self.transform and hasattr(self.transform, "training_step"):
            batches = itertools.islice(dataloader, n_batches)
            samples = {self.task: torch.cat([s[self.task] for s in batches], 0)}
            self.transform.training_step(samples, 0)

    def step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)
