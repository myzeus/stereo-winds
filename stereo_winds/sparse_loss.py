"""Sparse loss functions for training with partial labels.

Used for EarthCARE height supervision where only a thin track of pixels
has ground truth — labels cover ~0.2% of a 512x512 patch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseL1Loss(nn.Module):
    """L1 loss computed only at masked (labeled) pixels."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred, target : (...) matching shapes
        mask : (...) boolean, True where labels exist

        Returns
        -------
        Scalar mean absolute error over labeled pixels, or 0 if none.
        """
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return torch.abs(pred[mask] - target[mask]).mean()


class SparseCloudContainmentLoss(nn.Module):
    """Loss that is zero when predicted height is inside any cloud layer.

    For each labeled pixel, checks whether the predicted height falls
    between the base and top of any valid cloud layer.  If inside a
    cloud, the loss is zero.  If outside, the loss is a Huber penalty
    on the distance to the nearest layer boundary.

    This lets the model track anywhere within a cloud without bias
    toward the top, center, or base.

    Parameters
    ----------
    delta : Huber threshold in the same units as inputs (default 1.0 km
        when heights are passed in km).
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,
        layer_tops: torch.Tensor,
        layer_bases: torch.Tensor,
        layer_mask: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : (B, H, W) predicted heights
        layer_tops : (B, H, W, K) top height of each cloud layer
        layer_bases : (B, H, W, K) base height of each cloud layer
        layer_mask : (B, H, W, K) bool, which layers are valid
        pixel_mask : (B, H, W) bool, which pixels have any label

        Returns
        -------
        Scalar loss over labeled pixels, or 0 if none.
        """
        if not pixel_mask.any():
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        h = pred.unsqueeze(-1)  # (B, H, W, 1)
        INF = torch.tensor(float("inf"), device=pred.device)

        # Distance above top (positive if above cloud) or below base
        above_top = torch.where(layer_mask, torch.clamp(h - layer_tops, min=0), INF)
        below_base = torch.where(layer_mask, torch.clamp(layer_bases - h, min=0), INF)

        # If inside any layer, both above_top and below_base are 0 for that layer
        # Outside distance = min over layers of max(above_top, below_base)
        # But simpler: distance to nearest boundary
        dist_to_top = torch.where(layer_mask, torch.abs(h - layer_tops), INF)
        dist_to_base = torch.where(layer_mask, torch.abs(h - layer_bases), INF)

        # Check if inside any layer: h >= base AND h <= top
        inside = layer_mask & (h >= layer_bases) & (h <= layer_tops)  # (B,H,W,K)
        inside_any = inside.any(dim=-1)  # (B,H,W)

        # For pixels outside all layers: distance to nearest boundary
        all_dists = torch.minimum(dist_to_top, dist_to_base)  # (B,H,W,K)
        nearest_boundary = all_dists.min(dim=-1).values  # (B,H,W)

        # Zero loss inside cloud, Huber loss outside
        gap = torch.where(inside_any, torch.zeros_like(nearest_boundary), nearest_boundary)

        d = gap[pixel_mask]
        quadratic = torch.clamp(d, max=self.delta)
        linear = d - quadratic
        loss = 0.5 * quadratic.pow(2) / self.delta + linear
        return loss.mean()


class SparseHuberLoss(nn.Module):
    """Huber (smooth L1) loss at masked pixels.

    More robust than L1 to EarthCARE cloud-edge outliers where the
    active sensor footprint doesn't perfectly match the GOES pixel.

    Parameters
    ----------
    delta : threshold between L1 and L2 regimes (default 1.0, in same
        units as pred/target — use km if heights are in km).
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        diff = pred[mask] - target[mask]
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic.pow(2) / self.delta + linear
        return loss.mean()


class SparseWindLoss(nn.Module):
    """Match predicted (u, v) to a sonde wind profile, optionally
    interpolated to the predicted height.

    For each labeled pixel, two matching modes:

    - **closest**: pick the sonde level whose geopotential height is closest
      to ``h_pred`` and compute loss against its (u, v). Gradient w.r.t. h_pred
      is zero (the chosen level is selected by argmin).

    - **interp** (default): linearly interpolate (u, v) between the bracketing
      sonde levels (one above and one below ``h_pred``) at exactly h_pred.
      The interpolation weight is differentiable in h_pred, so gradient flows
      through the height too — the model learns "if I want to match this
      wind, I need to predict the right height." Pixels without bracketing
      levels are skipped.

    Parameters
    ----------
    loss_type : 'huber' or 'rmsvd' (default 'huber')
        - 'huber': mean of Huber(du) + Huber(dv) at labeled pixels
        - 'rmsvd': sqrt(mean(du² + dv²)) at labeled pixels
    delta : Huber threshold in m/s (default 5.0). Only used for 'huber'.
    require_bracket : if True, only compute loss at pixels where the sonde
        profile has a level above and below h_pred (default True)
    match_mode : 'closest' or 'interp' (default 'interp')
    """

    def __init__(
        self,
        delta: float = 5.0,
        require_bracket: bool = True,
        loss_type: str = "huber",
        match_mode: str = "interp",
    ):
        super().__init__()
        self.delta = delta
        self.require_bracket = require_bracket
        if loss_type not in ("huber", "rmsvd"):
            raise ValueError(f"loss_type must be 'huber' or 'rmsvd', got {loss_type}")
        if match_mode not in ("closest", "interp"):
            raise ValueError(f"match_mode must be 'closest' or 'interp', got {match_mode}")
        self.loss_type = loss_type
        self.match_mode = match_mode

    def forward(
        self,
        h_pred: torch.Tensor,
        u_pred: torch.Tensor,
        v_pred: torch.Tensor,
        sonde_h: torch.Tensor,
        sonde_u: torch.Tensor,
        sonde_v: torch.Tensor,
        sonde_mask: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_pred : (B, H, W) predicted height in meters
        u_pred, v_pred : (B, H, W) predicted wind in m/s
        sonde_h : (B, H, W, K) sonde heights at K levels (m)
        sonde_u, sonde_v : (B, H, W, K) sonde winds at those levels (m/s)
        sonde_mask : (B, H, W, K) bool, which levels are valid
        pixel_mask : (B, H, W) bool, which pixels have any sonde label

        Returns
        -------
        Scalar Huber loss in m/s, or 0 if no labels.
        """
        if not pixel_mask.any():
            return torch.tensor(0.0, device=h_pred.device, dtype=h_pred.dtype)

        h_p = h_pred.unsqueeze(-1)  # (B, H, W, 1)
        INF = torch.tensor(float("inf"), device=h_pred.device)

        if self.match_mode == "interp":
            # Linear interpolation between bracketing sonde levels.
            # NaN-safe: replace NaN sonde values with 0 before all arithmetic
            # so non-labeled pixels don't propagate NaN through gradients.
            # The eff_mask filters them out at loss time anyway.
            sonde_h_safe = torch.nan_to_num(sonde_h, nan=0.0)
            sonde_u_safe = torch.nan_to_num(sonde_u, nan=0.0)
            sonde_v_safe = torch.nan_to_num(sonde_v, nan=0.0)

            below = sonde_mask & (sonde_h_safe <= h_p)
            above = sonde_mask & (sonde_h_safe > h_p)
            has_bracket = below.any(dim=-1) & above.any(dim=-1)

            # Below: pick the level with the largest height (closest from below)
            neg_inf = torch.full_like(sonde_h_safe, float("-inf"))
            pos_inf = torch.full_like(sonde_h_safe, float("inf"))
            h_below = torch.where(below, sonde_h_safe, neg_inf)
            below_idx = h_below.argmax(dim=-1, keepdim=True)
            h_above = torch.where(above, sonde_h_safe, pos_inf)
            above_idx = h_above.argmin(dim=-1, keepdim=True)

            hb = sonde_h_safe.gather(-1, below_idx).squeeze(-1)
            ha = sonde_h_safe.gather(-1, above_idx).squeeze(-1)
            ub = sonde_u_safe.gather(-1, below_idx).squeeze(-1)
            ua = sonde_u_safe.gather(-1, above_idx).squeeze(-1)
            vb = sonde_v_safe.gather(-1, below_idx).squeeze(-1)
            va = sonde_v_safe.gather(-1, above_idx).squeeze(-1)

            # Robust denominator: clamp to a large positive value to avoid
            # tiny/zero divisions producing huge or NaN gradients.
            denom = (ha - hb).clamp(min=1.0)
            w = ((h_pred - hb) / denom).clamp(0.0, 1.0)
            u_match = (1.0 - w) * ub + w * ua
            v_match = (1.0 - w) * vb + w * va

            eff_mask = pixel_mask & has_bracket
        else:
            # Closest level (no gradient through h_pred)
            dist = torch.where(sonde_mask, torch.abs(sonde_h - h_p), INF)
            closest_lev = dist.argmin(dim=-1)
            idx = closest_lev.unsqueeze(-1)
            u_match = sonde_u.gather(-1, idx).squeeze(-1)
            v_match = sonde_v.gather(-1, idx).squeeze(-1)
            valid_match = sonde_mask.gather(-1, idx).squeeze(-1)
            eff_mask = pixel_mask & valid_match
            if self.require_bracket:
                above = sonde_mask & (sonde_h > h_p)
                below = sonde_mask & (sonde_h < h_p)
                has_bracket = above.any(dim=-1) & below.any(dim=-1)
                eff_mask = eff_mask & has_bracket

        if not eff_mask.any():
            return torch.tensor(0.0, device=h_pred.device, dtype=h_pred.dtype)

        du = u_pred[eff_mask] - u_match[eff_mask]
        dv = v_pred[eff_mask] - v_match[eff_mask]

        if self.loss_type == "rmsvd":
            # sqrt(mean(du² + dv²)) — matches the validation metric.
            # Add a small epsilon for gradient stability when du, dv → 0.
            sq = du.pow(2) + dv.pow(2)
            return torch.sqrt(sq.mean() + 1e-8)

        # Default: Huber on each component, summed and averaged
        def huber(x, delta):
            ax = torch.abs(x)
            quad = torch.clamp(ax, max=delta)
            lin = ax - quad
            return 0.5 * quad.pow(2) / delta + lin

        loss = (huber(du, self.delta) + huber(dv, self.delta)).mean()
        return loss
