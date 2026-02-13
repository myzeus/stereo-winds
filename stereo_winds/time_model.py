"""ABI/AHI per-pixel scan time models.

ABI Mode 6 scans full disk in ~600s, north-to-south in 22 swaths.
Initial implementation uses a linear model sufficient for the dominant
scan-to-scan time offset signal.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import xarray as xr

from .config import SatelliteConfig

# ABI Mode 6 full-disk scan duration (seconds)
ABI_FULL_DISK_DURATION = 600.0

# AHI full-disk scan duration (seconds)
AHI_FULL_DISK_DURATION = 600.0


def abi_pixel_times(
    row: np.ndarray,
    sat: SatelliteConfig,
    scan_duration: float = ABI_FULL_DISK_DURATION,
) -> np.ndarray:
    """Compute per-pixel time offsets within a single ABI full-disk scan.

    Linear model: offset = (row / n_rows) * scan_duration.
    ABI scans north-to-south, so row 0 (north) is earliest.

    Parameters
    ----------
    row : ndarray
        Row indices (0-based).
    sat : SatelliteConfig
        Satellite configuration (for grid dimensions).
    scan_duration : float
        Total scan duration in seconds.

    Returns
    -------
    offsets : ndarray
        Time offset in seconds from scan start for each row.
    """
    return (np.asarray(row, dtype=np.float64) / sat.n_rows) * scan_duration


def ahi_pixel_times(
    row: np.ndarray,
    sat: SatelliteConfig,
    scan_duration: float = AHI_FULL_DISK_DURATION,
) -> np.ndarray:
    """Compute per-pixel time offsets within a single AHI full-disk scan.

    AHI also scans north-to-south, same linear model as ABI.
    """
    return (np.asarray(row, dtype=np.float64) / sat.n_rows) * scan_duration


def read_time_bounds(nc_path: str) -> tuple[float, float]:
    """Read time bounds from ABI L1b netCDF metadata.

    Returns (t_start, t_end) as seconds since 2000-01-01 12:00:00.
    """
    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    try:
        t_start = float(ds["time_bounds"].values[0])
        t_end = float(ds["time_bounds"].values[1])
    finally:
        ds.close()
    return t_start, t_end


def compute_scene_times(
    t0: datetime,
    dt_minutes: float,
    sat_a: SatelliteConfig,
    sat_b: SatelliteConfig,
) -> dict[str, float]:
    """Compute time offsets (seconds) for each of the 5 scenes relative to t0.

    For same-satellite temporal pairs, dt = ±dt_minutes * 60.
    For cross-satellite pairs, the offset also includes scan-phase difference.

    Returns dict with keys: A_minus, A0, A_plus, B_minus, B_plus.
    Values are global time offsets in seconds (not per-pixel).
    """
    dt_sec = dt_minutes * 60.0
    return {
        "A_minus": -dt_sec,
        "A0": 0.0,
        "A_plus": dt_sec,
        "B_minus": -dt_sec,
        "B_plus": dt_sec,
    }
