"""Lightning training module for the single-satellite wind student.

Distills the cross-satellite stereo retrieval (teacher) into a lightweight
pixelwise model.  Loss is a masked, optionally robustified, heteroscedastic
Gaussian negative log-likelihood on u, v (m/s) and cloud-top height (km),
supervised only where the teacher's quality flag is good.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps as plt_cmaps
import numpy as np
import torch
from pytorch_lightning import LightningModule

from .student_model import PixelwiseWindStudent

logger = logging.getLogger(__name__)


def _huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    ax = x.abs()
    quad = ax.clamp(max=delta)
    lin = ax - quad
    return 0.5 * quad.pow(2) / delta + lin


def heteroscedastic_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
    mode: str = "gaussian",
    delta: float = 5.0,
) -> torch.Tensor:
    """Masked, weighted heteroscedastic NLL.

    gaussian:      0.5 * (exp(-s) * (t-mu)^2 + s)
    huber_learned: exp(-s) * huber(t-mu, delta) + 0.5 * s

    Returns a scalar.  If no pixel is masked-in, returns 0 connected to the
    graph (so the optimizer step stays well-defined).
    """
    if not mask.any():
        return mean.sum() * 0.0
    diff = target - mean
    if mode == "huber_learned":
        data_term = torch.exp(-logvar) * _huber(diff, delta)
        nll = data_term + 0.5 * logvar
    else:
        nll = 0.5 * (torch.exp(-logvar) * diff.pow(2) + logvar)
    nll_m = nll[mask]
    if weight is None:
        return nll_m.mean()
    w = weight[mask]
    return (w * nll_m).sum() / w.sum().clamp(min=1.0)


def vector_nll(
    target_u: torch.Tensor,
    target_v: torch.Tensor,
    mean_u: torch.Tensor,
    mean_v: torch.Tensor,
    logvar_uv: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Heteroscedastic bivariate-Gaussian NLL for the wind vector (u, v).

    Treats (u, v) as a single 2-D Gaussian with isotropic covariance
    ``sigma_uv^2 * I`` and a *single* predicted log-variance per pixel.
    The data term equals the squared RMSVD residual scaled by 1 / (2 sigma^2),
    so minimizing this loss is directly aligned with the RMSVD metric.

    NLL = 0.5 * exp(-logvar_uv) * ( (u - u_pred)^2 + (v - v_pred)^2 ) + logvar_uv

    (Drops the bivariate-Gaussian constant ``log(2 pi)``.  Note the ``+ logvar``
    — not ``+ 0.5 * logvar`` — because the log-determinant of a 2-D isotropic
    covariance is ``2 * logvar``.)

    All quantities should be in the *same* (physical) units — typically m/s.
    Returns the masked, optionally weighted mean.  Empty mask returns a
    zero scalar still connected to the graph.
    """
    if not mask.any():
        return mean_u.sum() * 0.0
    du = target_u - mean_u
    dv = target_v - mean_v
    err_sq = du.pow(2) + dv.pow(2)
    nll = 0.5 * torch.exp(-logvar_uv) * err_sq + logvar_uv
    nll_m = nll[mask]
    if weight is None:
        return nll_m.mean()
    w = weight[mask]
    return (w * nll_m).sum() / w.sum().clamp(min=1.0)


def huber_uv(
    target_u: torch.Tensor,
    target_v: torch.Tensor,
    mean_u: torch.Tensor,
    mean_v: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
    delta: float = 10.0,
) -> torch.Tensor:
    """Masked, weighted Huber (smooth-L1) loss on the wind vector (u, v).

    A robust *point* loss with NO learned variance term.  Unlike
    ``vector_nll`` / ``heteroscedastic_nll`` there is no ``exp(-logvar)``
    weighting, so the model cannot lower the loss by predicting near the
    conditional mean and inflating its uncertainty.  Removing that
    variance-hedging escape hatch is what suppresses the regression-to-mean
    (jet under-prediction) seen with the heteroscedastic NLL, and mirrors the
    Huber objective used to fine-tune the stereo teacher.

    ``delta`` is in the same (physical) units as the inputs — m/s.  Keep it
    LARGE (~10-15): Huber goes linear beyond ``delta``, so a small delta
    down-weights large residuals and treats jets as outliers, which *worsens*
    their under-prediction.  A large delta stays quadratic (L2-like) across
    the real wind range and only softens genuine outliers / noisy teacher
    labels.

    Returns a scalar.  Empty mask returns 0 connected to the graph.
    """
    if not mask.any():
        return mean_u.sum() * 0.0
    hb = _huber(target_u - mean_u, delta) + _huber(target_v - mean_v, delta)
    hb_m = hb[mask]
    if weight is None:
        return hb_m.mean()
    w = weight[mask]
    return (w * hb_m).sum() / w.sum().clamp(min=1.0)


class StudentWindsModule(LightningModule):
    """Train ``PixelwiseWindStudent`` against stereo teacher labels."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        n_layers: int = 4,
        context: bool = True,
        wind_cap: float | None = 150.0,
        learning_rate: float = 3e-4,
        w_u: float = 1.0,
        w_v: float = 1.0,
        w_h: float = 1.0,
        nll_mode: str = "gaussian",
        huber_delta: float = 5.0,
        use_teacher_sigma: bool = False,
        sigma_floor: float = 1.0,
        log_media_step: int = 100,
        flow_bands: list | None = None,
        rad_bands: list | None = None,
        rad_stats: list | None = None,
    ):
        super().__init__()
        # flow_bands/rad_bands/rad_stats are recorded for inference metadata
        # (input channel order + per-band radiance z-score), not used in forward.
        self.save_hyperparameters()
        self.model = PixelwiseWindStudent(
            in_channels=in_channels, hidden=hidden, n_layers=n_layers,
            context=context, wind_cap=wind_cap,
        )
        self.learning_rate = learning_rate
        self.w_u, self.w_v, self.w_h = w_u, w_v, w_h
        self.nll_mode = nll_mode
        self.huber_delta = huber_delta
        self.use_teacher_sigma = use_teacher_sigma
        self.sigma_floor = sigma_floor
        self.log_media_step = log_media_step

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x)

    def _effective_weight(self, batch, key, default_sigma_key):
        """Per-pixel loss weight, optionally down-weighting high-teacher-sigma."""
        w = batch["weight"]
        if self.use_teacher_sigma and default_sigma_key in batch:
            s = batch[default_sigma_key]
            w = w / (1.0 + (s / self.sigma_floor).pow(2))
        return w

    def _step(self, batch, stage: str) -> torch.Tensor:
        out = self.forward(batch["x"])
        mask = batch["mask"]

        nll_u = heteroscedastic_nll(
            batch["u_target"], out["u_mean"], out["u_logvar"], mask,
            self._effective_weight(batch, "u", "sigma_u_ms"),
            self.nll_mode, self.huber_delta,
        )
        nll_v = heteroscedastic_nll(
            batch["v_target"], out["v_mean"], out["v_logvar"], mask,
            self._effective_weight(batch, "v", "sigma_v_ms"),
            self.nll_mode, self.huber_delta,
        )
        nll_h = heteroscedastic_nll(
            batch["h_target_km"], out["h_mean"], out["h_logvar"], mask,
            self._effective_weight(batch, "h", "sigma_h_km"),
            self.nll_mode, self.huber_delta,
        )
        loss = self.w_u * nll_u + self.w_v * nll_v + self.w_h * nll_h

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/nll_u", nll_u)
        self.log(f"{stage}/nll_v", nll_v)
        self.log(f"{stage}/nll_h", nll_h)
        self._log_metrics(batch, out, mask, stage)

        if (stage == "train" and self.global_step % self.log_media_step == 0):
            self._log_images(batch, out, mask)
        return loss

    @torch.no_grad()
    def _log_metrics(self, batch, out, mask, stage):
        if not mask.any():
            return
        du = (out["u_mean"] - batch["u_target"])[mask]
        dv = (out["v_mean"] - batch["v_target"])[mask]
        rmsvd = torch.sqrt((du.pow(2) + dv.pow(2)).mean())
        dh_m = (out["h_mean"] - batch["h_target_km"])[mask] * 1000.0
        h_rmse = torch.sqrt(dh_m.pow(2).mean())
        self.log(f"{stage}/rmsvd", rmsvd, prog_bar=True)
        self.log(f"{stage}/h_rmse_m", h_rmse, prog_bar=True)
        self.log(f"{stage}/mask_frac", mask.float().mean())
        # Calibration: fraction of u-errors within +/-1 predicted sigma (~0.68)
        sigma_u = torch.exp(0.5 * out["u_logvar"])[mask]
        self.log(f"{stage}/calib_u", (du.abs() <= sigma_u).float().mean())

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def on_before_optimizer_step(self, optimizer):
        for p in self.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _make_grid(tiles: list[np.ndarray], border: int = 2) -> np.ndarray:
        padded = []
        for i, t in enumerate(tiles):
            if i > 0:
                padded.append(np.ones((t.shape[0], border, *t.shape[2:]), dtype=t.dtype))
            padded.append(t)
        return np.concatenate(padded, axis=1)

    def _wandb_experiment(self):
        """Return the wandb run among the active loggers, or None."""
        for lg in getattr(self, "loggers", None) or [self.logger]:
            exp = getattr(lg, "experiment", None)
            if exp is not None and hasattr(exp, "log") and \
                    type(exp).__module__.split(".")[0] == "wandb":
                return exp
        return None

    @torch.no_grad()
    def _log_images(self, batch, out, mask, n_img: int = 6):
        exp = self._wandb_experiment()
        if exp is None:
            return
        import wandb

        def cmap(arr, name, vmin, vmax):
            cm = plt_cmaps[name]
            n = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
            return (cm(np.nan_to_num(n))[..., :3] * 255).astype(np.uint8)

        B = min(batch["x"].shape[0], n_img)
        rows = {k: [] for k in ("u_teach", "u_pred", "v_teach", "v_pred", "h_teach", "h_pred")}
        for i in range(B):
            m = mask[i].cpu().numpy()
            def masked(a):
                return np.where(m, a, np.nan)
            ut = masked(batch["u_target"][i].cpu().numpy())
            vt = masked(batch["v_target"][i].cpu().numpy())
            ht = masked(batch["h_target_km"][i].cpu().numpy())
            up = masked(out["u_mean"][i].cpu().numpy())
            vp = masked(out["v_mean"][i].cpu().numpy())
            hp = masked(out["h_mean"][i].cpu().numpy())
            rows["u_teach"].append(cmap(ut, "RdBu_r", -40, 40))
            rows["u_pred"].append(cmap(up, "RdBu_r", -40, 40))
            rows["v_teach"].append(cmap(vt, "RdBu_r", -40, 40))
            rows["v_pred"].append(cmap(vp, "RdBu_r", -40, 40))
            rows["h_teach"].append(cmap(ht, "turbo", 0, 18))
            rows["h_pred"].append(cmap(hp, "turbo", 0, 18))

        logs = {f"train/images/{k}": wandb.Image(self._make_grid(v)) for k, v in rows.items()}
        exp.log(logs, step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.trainer.estimated_stepping_batches), eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
