"""Validation metrics for stereo wind retrievals."""

from __future__ import annotations

import numpy as np


def rmsvd(u_pred: np.ndarray, v_pred: np.ndarray,
          u_true: np.ndarray, v_true: np.ndarray) -> float:
    """Root-mean-square vector difference.

    RMSVD = sqrt(mean((u_p - u_t)^2 + (v_p - v_t)^2))
    """
    valid = (
        np.isfinite(u_pred) & np.isfinite(v_pred)
        & np.isfinite(u_true) & np.isfinite(v_true)
    )
    if valid.sum() == 0:
        return np.nan
    du = u_pred[valid] - u_true[valid]
    dv = v_pred[valid] - v_true[valid]
    return float(np.sqrt(np.mean(du**2 + dv**2)))


def speed_bias(u_pred: np.ndarray, v_pred: np.ndarray,
               u_true: np.ndarray, v_true: np.ndarray) -> float:
    """Mean speed bias (predicted - true)."""
    valid = (
        np.isfinite(u_pred) & np.isfinite(v_pred)
        & np.isfinite(u_true) & np.isfinite(v_true)
    )
    if valid.sum() == 0:
        return np.nan
    spd_pred = np.sqrt(u_pred[valid]**2 + v_pred[valid]**2)
    spd_true = np.sqrt(u_true[valid]**2 + v_true[valid]**2)
    return float(np.mean(spd_pred - spd_true))


def height_rmse(h_pred: np.ndarray, h_true: np.ndarray) -> float:
    """Root-mean-square error in height."""
    valid = np.isfinite(h_pred) & np.isfinite(h_true)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((h_pred[valid] - h_true[valid])**2)))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return np.nan
    return float(np.corrcoef(a[valid].ravel(), b[valid].ravel())[0, 1])
