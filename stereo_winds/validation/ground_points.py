"""Ground-point validation: h ~ 0, wind ~ 0 pixels as truth references."""

from __future__ import annotations

import numpy as np

from .metrics import height_rmse, rmsvd


def identify_ground_points(
    h: np.ndarray,
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    quality_flag: np.ndarray,
    h_threshold: float = 500.0,
    wind_threshold: float = 2.0,
) -> np.ndarray:
    """Identify clear-sky ground pixels where h ~ 0 and wind ~ 0.

    Returns boolean mask of candidate ground points.
    """
    valid = quality_flag > 0
    low_h = np.abs(h) < h_threshold
    low_wind = np.sqrt(u_wind**2 + v_wind**2) < wind_threshold
    return valid & low_h & low_wind


def ground_point_statistics(
    h: np.ndarray,
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    quality_flag: np.ndarray,
    h_threshold: float = 500.0,
    wind_threshold: float = 2.0,
) -> dict[str, float]:
    """Compute validation statistics at ground-truth points.

    Ground points are clear-sky pixels where retrieved h ~ 0 and wind ~ 0.
    At these points, the "true" values are h=0, u=0, v=0.
    """
    mask = identify_ground_points(h, u_wind, v_wind, quality_flag, h_threshold, wind_threshold)
    n_points = int(mask.sum())

    if n_points == 0:
        return {"n_points": 0, "h_rmse": np.nan, "wind_rmsvd": np.nan}

    h_ground = h[mask]
    u_ground = u_wind[mask]
    v_ground = v_wind[mask]

    return {
        "n_points": n_points,
        "h_rmse": height_rmse(h_ground, np.zeros_like(h_ground)),
        "h_mean_bias": float(np.mean(h_ground)),
        "wind_rmsvd": rmsvd(u_ground, v_ground, np.zeros_like(u_ground), np.zeros_like(v_ground)),
        "wind_mean_speed": float(np.mean(np.sqrt(u_ground**2 + v_ground**2))),
    }
