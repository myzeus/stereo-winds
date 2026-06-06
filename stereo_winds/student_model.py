"""Lightweight single-satellite wind student model.

A pixelwise MLP (implemented as a stack of 1x1 convolutions) that maps a
per-pixel feature vector — channel-wise temporal optical flows + radiances +
viewing geometry — to wind (u, v), cloud-top height, and a per-pixel
heteroscedastic uncertainty for each.

This is the *student* in a distillation setup: the cross-satellite stereo
retrieval (which needs two overlapping satellites) is the *teacher* that
supplies dense u/v/h labels in the overlap region; the student learns to
reproduce them from inputs available on a single satellite's full disk, so it
can be applied globally to any geostationary satellite.

Because the model is essentially pixelwise (1x1 convs, with an optional couple
of small 3x3 context convs), it is satellite-agnostic and tiles trivially over
a full disk at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Height is predicted in kilometres so the regression and NLL terms stay O(1)
# and comparable to the m/s wind terms.  Teacher QA bounds height to [0, 20] km.
H_MAX_KM = 20.0


def _hetero_head_outputs(
    raw: torch.Tensor,
    wind_logvar_mode: str,
    logvar_min: float,
    logvar_max: float,
    predict_chi2: bool = False,
) -> dict[str, torch.Tensor]:
    """Slice raw (B, n_out, H, W) into the heteroscedastic head dict.

    Same key contract as ``PixelwiseWindStudent.forward`` so the LightningModule
    doesn't care which trunk produced ``raw``.

    When ``predict_chi2``, the last channel is the student's prediction of
    log(teacher_chi²) — distilling the WLS-solver residual so the student can
    emit its own QA-gate at inference (no need for ``eval_from_parquet --qa-from``).
    """
    out = {"u_mean": raw[:, 0], "v_mean": raw[:, 1], "h_mean": raw[:, 2]}
    if wind_logvar_mode == "per_component":
        out["u_logvar"] = raw[:, 3].clamp(logvar_min, logvar_max)
        out["v_logvar"] = raw[:, 4].clamp(logvar_min, logvar_max)
        out["h_logvar"] = raw[:, 5].clamp(logvar_min, logvar_max)
        next_idx = 6
    else:  # joint
        out["uv_logvar"] = raw[:, 3].clamp(logvar_min, logvar_max)
        out["h_logvar"] = raw[:, 4].clamp(logvar_min, logvar_max)
        next_idx = 5
    if predict_chi2:
        out["log_chi2"] = raw[:, next_idx]
    return out


def head_n_out(wind_logvar_mode: str, predict_chi2: bool = False) -> int:
    """Total channels in the heteroscedastic head."""
    base = 6 if wind_logvar_mode == "per_component" else 5
    return base + (1 if predict_chi2 else 0)


class PixelwiseWindStudent(nn.Module):
    """Per-pixel MLP with a heteroscedastic (mean + log-variance) head.

    Parameters
    ----------
    in_channels : number of input feature channels (see ``student_dataset``;
        28 with radiance, 23 without).  The training script must pass the
        dataset's ``in_channels`` so the two cannot drift.
    hidden : width of the hidden 1x1-conv layers
    n_layers : number of hidden layers (>= 1)
    context : if True, prepend two depth-preserving 3x3 convs to give the
        model a small local receptive field; if False, the model is strictly
        pixelwise (1x1 only) and tiles with zero halo.
    wind_cap : soft cap on |u|, |v| in m/s via ``cap * tanh(raw / cap)``;
        None disables the cap.  Keeps early-training predictions sane.
    logvar_min, logvar_max : clamp range for predicted log-variances.
    wind_logvar_mode : ``"per_component"`` (default) → 6-channel head with
        separate ``u_logvar`` and ``v_logvar`` (used by the per-component
        Gaussian/Huber NLL).  ``"joint"`` → 5-channel head with a single
        shared ``uv_logvar`` for the wind vector, paired with the
        bivariate (vector) NLL.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        n_layers: int = 4,
        context: bool = True,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
        wind_logvar_mode: str = "per_component",
        predict_chi2: bool = False,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if wind_logvar_mode not in ("per_component", "joint"):
            raise ValueError(
                f"wind_logvar_mode must be 'per_component' or 'joint', "
                f"got {wind_logvar_mode!r}")
        self.in_channels = in_channels
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.wind_logvar_mode = wind_logvar_mode
        self.predict_chi2 = predict_chi2

        layers: list[nn.Module] = []
        if context:
            layers += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GELU(),
            ]

        # Pixelwise trunk: 1x1 convs == an MLP applied at every pixel.  No
        # spatial normalization — that would couple pixels and break the
        # zero-halo tiling property (inputs are already normalized upstream).
        layers += [nn.Conv2d(in_channels, hidden, 1), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Conv2d(hidden, hidden, 1), nn.GELU()]
        self.trunk = nn.Sequential(*layers)

        # per_component: 6 outputs (u_mean, v_mean, h_mean, u_logvar, v_logvar, h_logvar)
        # joint:         5 outputs (u_mean, v_mean, h_mean, uv_logvar, h_logvar)
        # + 1 extra log_chi2 channel when predict_chi2.
        self.n_out = head_n_out(wind_logvar_mode, predict_chi2)
        self.head = nn.Conv2d(hidden, self.n_out, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W) feature tensor (already normalized by the
            dataset; NaNs replaced with 0).

        Returns
        -------
        dict with keys u_mean, v_mean, h_mean and a logvar set that depends on
        ``wind_logvar_mode``:
            per_component → u_logvar, v_logvar, h_logvar
            joint         → uv_logvar (shared across u,v), h_logvar
        Each is (B, H, W).  ALL means are predicted in STANDARDIZED units
        (z-score) — the wrapping LightningModule normalizes the targets and
        denormalizes predictions for inference.  Unbounded (no tanh/sigmoid)
        so gradients flow freely through the O(1) standardized range.
        """
        raw = self.head(self.trunk(x))
        return _hetero_head_outputs(
            raw, self.wind_logvar_mode, self.logvar_min, self.logvar_max,
            predict_chi2=self.predict_chi2,
        )


def _double_conv(in_ch: int, out_ch: int, gn_groups: int = 8) -> nn.Sequential:
    """Two 3x3 conv + GroupNorm + GELU blocks — building block for U-Net."""
    g = min(gn_groups, out_ch)
    while out_ch % g != 0 and g > 1:
        g -= 1
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(g, out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(g, out_ch),
        nn.GELU(),
    )


class UNetWindStudent(nn.Module):
    """U-Net trunk + pixelwise heteroscedastic head for the wind student.

    Same input/output contract as :class:`PixelwiseWindStudent` — the wrapping
    ``StudentWindsModel`` doesn't need to know which trunk it has.  The U-Net
    gives the model a meaningful spatial receptive field (cloud morphology,
    inter-band gradient patterns) which a pixelwise trunk can't access.

    Architecture: 3-level encoder–decoder with concat skip connections,
    GroupNorm + GELU.  Default sizing ~150-200 K params for a
    base_channels=32 setup — a 3× bump over the pixelwise design.

    Receptive field at the head is roughly (2^n_levels) × kernel + skip — for
    n_levels=3 and 3x3 kernels, ~30 px in each direction.  Inference tiles
    therefore need a halo of ~32 px to avoid edge artifacts.

    Parameters
    ----------
    in_channels : int — input feature width (4·n_flow + n_rad + 3 geom).
    base_channels : int — channels in the first encoder block; widths
        double per encoder level.
    n_levels : int — number of encoder downsampling steps (default 3 →
        bottleneck at 1/8 spatial).
    logvar_min, logvar_max : clamp range for predicted log-variances.
    wind_logvar_mode : same as PixelwiseWindStudent.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        n_levels: int = 3,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
        wind_logvar_mode: str = "per_component",
        predict_chi2: bool = False,
    ):
        super().__init__()
        if wind_logvar_mode not in ("per_component", "joint"):
            raise ValueError(
                f"wind_logvar_mode must be 'per_component' or 'joint', "
                f"got {wind_logvar_mode!r}")
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")

        self.in_channels = in_channels
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.wind_logvar_mode = wind_logvar_mode
        self.predict_chi2 = predict_chi2
        self.n_levels = n_levels

        # Channel widths per level: [base, 2*base, 4*base, ...]
        widths = [base_channels * (2 ** i) for i in range(n_levels + 1)]

        # Encoder path
        self.encoders = nn.ModuleList()
        prev = in_channels
        for w in widths[:-1]:
            self.encoders.append(_double_conv(prev, w))
            prev = w
        # Bottleneck
        self.bottleneck = _double_conv(widths[-2], widths[-1])

        # Decoder path (in reverse order; takes concat of upsampled + skip)
        self.decoders = nn.ModuleList()
        for i in range(n_levels - 1, -1, -1):
            up_ch = widths[i + 1]
            skip_ch = widths[i]
            self.decoders.append(_double_conv(up_ch + skip_ch, skip_ch))

        # per_component: 6 outputs; joint: 5 outputs (+1 for chi²).
        self.n_out = head_n_out(wind_logvar_mode, predict_chi2)
        self.head = nn.Conv2d(widths[0], self.n_out, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        skips = []
        h = x
        # Encoder: each block, store skip, then pool
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)
        # Bottleneck
        h = self.bottleneck(h)
        # Decoder: upsample, concat skip (in reverse), conv
        for dec, skip in zip(self.decoders, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)
        raw = self.head(h)
        return _hetero_head_outputs(
            raw, self.wind_logvar_mode, self.logvar_min, self.logvar_max,
            predict_chi2=self.predict_chi2,
        )
