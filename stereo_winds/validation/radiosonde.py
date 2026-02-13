"""Radiosonde (IGRA) comparison for stereo wind validation."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

from .metrics import rmsvd, speed_bias, height_rmse


def collocate_with_sonde(
    ds: xr.Dataset,
    sonde_ds: xr.Dataset,
    max_distance_km: float = 15.0,
    max_pressure_diff_hpa: float = 50.0,
) -> dict[str, np.ndarray]:
    """Collocate stereo wind retrievals with radiosonde profiles.

    Parameters
    ----------
    ds : stereo wind Dataset with u_wind, v_wind, cloud_top_height, lat, lon
    sonde_ds : IGRA Dataset with u, v, geopotential_height, geometry
    max_distance_km : max horizontal collocation distance
    max_pressure_diff_hpa : max pressure level matching tolerance

    Returns
    -------
    dict with matched arrays: u_stereo, v_stereo, h_stereo, u_sonde, v_sonde, h_sonde
    """
    # Placeholder — real implementation needs spatial matching
    # between the gridded stereo product and point sonde locations
    return {
        "u_stereo": np.array([]),
        "v_stereo": np.array([]),
        "h_stereo": np.array([]),
        "u_sonde": np.array([]),
        "v_sonde": np.array([]),
        "h_sonde": np.array([]),
    }


def sonde_validation_statistics(matched: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute validation statistics from collocated sonde data."""
    if len(matched["u_stereo"]) == 0:
        return {"n_matches": 0, "rmsvd": np.nan, "speed_bias": np.nan, "h_rmse": np.nan}

    return {
        "n_matches": len(matched["u_stereo"]),
        "rmsvd": rmsvd(
            matched["u_stereo"], matched["v_stereo"],
            matched["u_sonde"], matched["v_sonde"],
        ),
        "speed_bias": speed_bias(
            matched["u_stereo"], matched["v_stereo"],
            matched["u_sonde"], matched["v_sonde"],
        ),
        "h_rmse": height_rmse(matched["h_stereo"], matched["h_sonde"]),
    }
