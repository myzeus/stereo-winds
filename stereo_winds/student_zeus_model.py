"""Single-satellite wind student as a zeus ``BaseLightningModule``.

Leverages the team's training tooling: a zeus ``StandardScalar`` transform
standardizes the raw radiance channels per band (fit via
``prepare_data_transformation`` before ``fit`` and baked into the checkpoint),
the base module handles optimizer/logging/wandb, and the pixelwise net + masked
heteroscedastic NLL come from the existing student modules.

Batch contract (from ``StudentXBatchDataset``): keys ``flow`` (B,4*n_flow,H,W),
``rad`` (B,n_rad,H,W, RAW), ``geom`` (B,3,H,W), ``u``/``v``/``h_km`` (B,H,W),
``mask`` (B,H,W bool), ``weight`` (B,H,W).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

_zeus = str(Path(__file__).resolve().parent.parent / "zeus")
if _zeus not in sys.path:
    sys.path.insert(0, _zeus)

from zeus.models.base_lightning_module import BaseLightningModule
from zeus.datasets.transform import StandardScalar

from .student_model import PixelwiseWindStudent, UNetWindStudent
from .student_module import heteroscedastic_nll, vector_nll
from .student_dataset import FLOW_SCALE, PIXEL_SCALE_NORM

logger = logging.getLogger(__name__)

# Temporal flow offset (s) for forward / backward pairs.  Same value used by
# the data generator (`compute_scene_times` returns ±600 s for ABI 10-min).
DT_SEC = 600.0


class StudentWindsModel(BaseLightningModule):
    """Pixelwise wind/height student trained against stereo teacher labels."""

    def __init__(
        self,
        n_flow_bands: int,
        n_rad_bands: int,
        hidden: int = 128,
        n_layers: int = 4,
        context: bool = True,
        w_u: float = 1.0,
        w_v: float = 1.0,
        w_h: float = 3.0,
        nll_mode: str = "gaussian",
        huber_delta: float = 5.0,
        wind_loss: str = "gaussian",
        logvar_init_offset: float = 5.0,
        trunk: str = "pixelwise",
        unet_base_channels: int = 32,
        unet_n_levels: int = 3,
        rad_time_frames: int = 1,
        learning_rate: float = 3e-4,
        log_media_step: int = 200,
        **kwargs,
    ):
        # Radiance is standardized per band×frame by the zeus StandardScalar
        # transform (task="rad"); flows/geometry are pre-scaled in the dataset.
        # When rad_time_frames=3, the rad tensor is (B, 3*n_rad, H, W) with
        # per-band [t-, t0, t+] interleaved — StandardScalar fits one
        # (mu, sd) pair per channel, so it adapts to each (band, frame) slot.
        if rad_time_frames not in (1, 3):
            raise ValueError(
                f"rad_time_frames must be 1 or 3, got {rad_time_frames}")
        n_rad_channels = n_rad_bands * rad_time_frames
        transform = (StandardScalar({"rad": n_rad_channels}, dim=1)
                     if n_rad_channels else None)
        super().__init__(learning_rate=learning_rate, transform=transform,
                         log_media_step=log_media_step)
        self.save_hyperparameters()
        self.task = "rad"
        self.w_u, self.w_v, self.w_h = w_u, w_v, w_h
        self.nll_mode, self.huber_delta = nll_mode, huber_delta
        # wind_loss: "gaussian" → per-component NLL on z-scored u,v
        #            "vector"   → bivariate Gaussian NLL on (u,v) in PHYSICAL m/s
        #                         (data term equals squared RMSVD scaled by 1/2σ²)
        # logvar_init_offset shifts the model's joint logvar at init so the
        # initial variance estimate sits at a sane wind-error scale (~e^5 ≈ 150 m²/s² ⇒ σ ≈ 12 m/s).
        if wind_loss not in ("gaussian", "vector"):
            raise ValueError(f"wind_loss must be 'gaussian' or 'vector', got {wind_loss!r}")
        self.wind_loss = wind_loss
        self.logvar_init_offset = logvar_init_offset

        in_channels = 4 * n_flow_bands + n_rad_channels + 3
        self.rad_time_frames = rad_time_frames
        wlm = "joint" if wind_loss == "vector" else "per_component"
        if trunk == "pixelwise":
            self.net = PixelwiseWindStudent(
                in_channels=in_channels, hidden=hidden, n_layers=n_layers,
                context=context, wind_logvar_mode=wlm,
            )
        elif trunk == "unet":
            self.net = UNetWindStudent(
                in_channels=in_channels,
                base_channels=unet_base_channels,
                n_levels=unet_n_levels,
                wind_logvar_mode=wlm,
            )
        else:
            raise ValueError(f"trunk must be 'pixelwise' or 'unet', got {trunk!r}")
        self.trunk_name = trunk

        # Target standardization (u m/s, v m/s, h km).  Fit from training data
        # via ``fit_target_stats`` so the NLL is dimensionless and balanced
        # across targets — applying loss in raw m/s vs km biases gradients.
        # Default mu=0, sd=1 is identity (safe before fitting).
        self.register_buffer("target_mu", torch.zeros(3))
        self.register_buffer("target_sd", torch.ones(3))

    def _physics_baseline(self, flow, geom):
        """Per-band parallax-free wind estimate, averaged across flow bands.

        From the WLS design matrix:  flow_fwd = p + V·dt, flow_back = p − V·dt
        ⇒ V = (flow_fwd − flow_back) / (2·dt) (pixels/s, with p_u and p_v
        cancelled). Multiplied by per-pixel ground scale (m/pixel) it gives
        wind in m/s.  We average across the flow bands as the baseline; the
        network's job is then to predict the residual correction.

        Returns (u_base, v_base) in m/s, each shape (B, H, W).
        """
        # Channel order per band: [back_u, back_v, fwd_u, fwd_v]; strides of 4.
        flow_pix = flow * FLOW_SCALE                      # (B, 4*n_flow, H, W)
        back_u, back_v = flow_pix[:, 0::4], flow_pix[:, 1::4]
        fwd_u, fwd_v = flow_pix[:, 2::4], flow_pix[:, 3::4]
        dx_m = geom[:, 0:1] * PIXEL_SCALE_NORM            # (B, 1, H, W)
        dy_m = geom[:, 1:2] * PIXEL_SCALE_NORM
        u_band = (fwd_u - back_u) * dx_m / (2.0 * DT_SEC)  # m/s, (B, n_flow, H, W)
        v_band = (fwd_v - back_v) * dy_m / (2.0 * DT_SEC)
        return u_band.mean(dim=1), v_band.mean(dim=1)     # (B, H, W) each

    def forward(self, flow, rad, geom):
        # Physics baseline for u/v in m/s, computed from the raw inputs.
        u_base, v_base = self._physics_baseline(flow, geom)
        # Convert baseline to the same z-score space the network predicts in,
        # so we can add it directly to the network's mean output.
        u_base_z = (u_base - self.target_mu[0]) / self.target_sd[0]
        v_base_z = (v_base - self.target_mu[1]) / self.target_sd[1]

        if self.transform is not None and rad.shape[1] > 0:
            rad = self.transform.forward({"rad": rad})["rad"]
        delta = self.net(torch.cat([flow, rad, geom], dim=1))

        # Network predicts the RESIDUAL on top of the physics prior (for u/v);
        # height has no flow-based baseline so the net carries it alone.
        out = {
            "u_mean": delta["u_mean"] + u_base_z,
            "v_mean": delta["v_mean"] + v_base_z,
            "h_mean": delta["h_mean"],
            "h_logvar": delta["h_logvar"],
        }
        if self.wind_loss == "vector":
            out["uv_logvar"] = delta["uv_logvar"]
        else:
            out["u_logvar"] = delta["u_logvar"]
            out["v_logvar"] = delta["v_logvar"]
        return out

    @torch.no_grad()
    def fit_target_stats(self, dataloader, n_batches: int = 50) -> None:
        """Estimate per-target (mean, std) on masked pixels from the train loader."""
        import itertools
        sums = torch.zeros(3, dtype=torch.float64)
        sumsq = torch.zeros(3, dtype=torch.float64)
        counts = torch.zeros(3, dtype=torch.int64)
        keys = ("u", "v", "h_km")
        for batch in itertools.islice(dataloader, n_batches):
            m = batch["mask"]
            for i, k in enumerate(keys):
                y = batch[k][m].double()
                y = y[torch.isfinite(y)]
                sums[i] += y.sum()
                sumsq[i] += y.pow(2).sum()
                counts[i] += y.numel()
        c = counts.clamp(min=1).double()
        mu = (sums / c).float()
        var = (sumsq / c - (sums / c).pow(2)).clamp(min=1e-8).float()
        sd = var.sqrt()
        self.target_mu.copy_(mu)
        self.target_sd.copy_(sd)
        print(f"target_mu (u, v, h_km): {mu.tolist()}")
        print(f"target_sd (u, v, h_km): {sd.tolist()}")

    @torch.no_grad()
    def predict(self, flow, rad, geom):
        """Forward + denormalize to physical units (m/s, km).  Use at inference.

        The output keys (``u_mean``, ``v_mean``, ``h_mean``, ``u_logvar``,
        ``v_logvar``, ``h_logvar``) are stable regardless of the wind-loss
        mode — downstream consumers don't need to branch.  In vector mode,
        ``u_logvar`` and ``v_logvar`` are *both* the (already-physical) joint
        wind logvar, so ``sigma_u == sigma_v`` per pixel.
        """
        out = self(flow, rad, geom)
        mu, sd = self.target_mu, self.target_sd
        log_sd = torch.log(sd)
        h_logvar_phys = out["h_logvar"] + 2 * log_sd[2]
        result = {
            "u_mean": out["u_mean"] * sd[0] + mu[0],
            "v_mean": out["v_mean"] * sd[1] + mu[1],
            "h_mean": out["h_mean"] * sd[2] + mu[2],
            "h_logvar": h_logvar_phys,
        }
        if self.wind_loss == "vector":
            # Model's uv_logvar is trained in PHYSICAL m/s already (via the
            # init offset).  Surface it as both u_logvar and v_logvar so
            # downstream sigma-consumers see a consistent dict shape.
            uv_logvar_phys = out["uv_logvar"] + self.logvar_init_offset
            result["u_logvar"] = uv_logvar_phys
            result["v_logvar"] = uv_logvar_phys
        else:
            result["u_logvar"] = out["u_logvar"] + 2 * log_sd[0]
            result["v_logvar"] = out["v_logvar"] + 2 * log_sd[1]
        return result

    def step(self, batch, batch_idx):
        out = self(batch["flow"], batch["rad"], batch["geom"])
        mask, w = batch["mask"], batch["weight"]
        bs = batch["u"].shape[0]
        mu, sd = self.target_mu, self.target_sd
        m = self.mode or "train"

        # Height: always heteroscedastic Gaussian/Huber in z-space.
        h_n = (batch["h_km"] - mu[2]) / sd[2]
        nll_h = heteroscedastic_nll(
            h_n, out["h_mean"], out["h_logvar"], mask, w,
            self.nll_mode, self.huber_delta,
        )

        if self.wind_loss == "vector":
            # Compute wind residual in PHYSICAL m/s so the data term is RMSVD-aligned.
            u_p = out["u_mean"] * sd[0] + mu[0]
            v_p = out["v_mean"] * sd[1] + mu[1]
            logvar_uv = out["uv_logvar"] + self.logvar_init_offset
            nll_wind = vector_nll(
                batch["u"], batch["v"], u_p, v_p, logvar_uv, mask, w,
            )
            # Vector NLL replaces u + v components; weight by (w_u + w_v) / 2 so
            # defaults keep the wind-side budget similar to the per-component path.
            loss = 0.5 * (self.w_u + self.w_v) * nll_wind + self.w_h * nll_h
            self.log(f"{m}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            self.log(f"{m}/nll_wind", nll_wind, batch_size=bs)
            self.log(f"{m}/nll_h", nll_h, batch_size=bs)
        else:
            # Per-component Gaussian / huber on z-scored targets.
            u_n = (batch["u"] - mu[0]) / sd[0]
            v_n = (batch["v"] - mu[1]) / sd[1]
            nll_u = heteroscedastic_nll(u_n, out["u_mean"], out["u_logvar"], mask, w, self.nll_mode, self.huber_delta)
            nll_v = heteroscedastic_nll(v_n, out["v_mean"], out["v_logvar"], mask, w, self.nll_mode, self.huber_delta)
            loss = self.w_u * nll_u + self.w_v * nll_v + self.w_h * nll_h
            self.log(f"{m}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            self.log(f"{m}/nll_u", nll_u, batch_size=bs)
            self.log(f"{m}/nll_v", nll_v, batch_size=bs)
            self.log(f"{m}/nll_h", nll_h, batch_size=bs)

        with torch.no_grad():
            if mask.any():
                # Denormalize predictions for physical-unit metrics.
                u_p = out["u_mean"] * sd[0] + mu[0]
                v_p = out["v_mean"] * sd[1] + mu[1]
                h_p_km = out["h_mean"] * sd[2] + mu[2]
                du = (u_p - batch["u"])[mask]
                dv = (v_p - batch["v"])[mask]
                dh_m = (h_p_km - batch["h_km"])[mask] * 1000.0
                self.log(f"{m}/rmsvd", torch.sqrt((du.pow(2) + dv.pow(2)).mean()), prog_bar=True, batch_size=bs)
                self.log(f"{m}/h_rmse_m", torch.sqrt(dh_m.pow(2).mean()), prog_bar=True, batch_size=bs)
                if self.wind_loss == "vector":
                    # Joint calibration: fraction of err-vector magnitudes within
                    # the predicted σ.  For a 2-D isotropic Gaussian the
                    # reference is the Rayleigh CDF at 1σ ≈ 0.393, NOT 0.68.
                    logvar_uv = out["uv_logvar"] + self.logvar_init_offset
                    s_uv = torch.exp(0.5 * logvar_uv)[mask]
                    err_mag = torch.sqrt(du.pow(2) + dv.pow(2))
                    self.log(f"{m}/calib_uv", (err_mag <= s_uv).float().mean(), batch_size=bs)
                else:
                    u_n = (batch["u"] - mu[0]) / sd[0]
                    u_resid = (out["u_mean"] - u_n)[mask]
                    su = torch.exp(0.5 * out["u_logvar"])[mask]
                    self.log(f"{m}/calib_u", (u_resid.abs() <= su).float().mean(), batch_size=bs)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-6)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, self.trainer.estimated_stepping_batches), eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
