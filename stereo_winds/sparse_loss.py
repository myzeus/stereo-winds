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
