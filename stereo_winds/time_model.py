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
    """Compute nominal time offsets (seconds) for the 5 scenes relative to t0.

    Scalar nominal offsets only (±dt_minutes * 60); scan phase is ignored.
    Prefer :func:`compute_scene_dt_fields` (per-pixel, actual scan times)
    when per-scene timing info is available.

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


_UNIX_EPOCH = np.datetime64("1970-01-01T00:00:00")

_abi_lut_cache: dict[str, np.ndarray] = {}


def load_abi_time_lut(nc_path: str) -> np.ndarray:
    """Load a Carr et al. ABI time-model LUT (per-pixel scan-time offsets).

    The supplement of Carr et al. (2020, RS 12:3779) provides per-scan-mode
    netCDFs with ``FD_pixel_times`` stored as (cols, rows) timedeltas
    relative to the L1b product start time ("add the product start time to
    the LUT values"). GOES-East Mode 6A is 'ABI-Timeline05B_Mode 6A_*.nc'.
    Returns (rows, cols) float64 seconds. Values are Band-2 referenced;
    per-band detector offsets are < 1 s and ignored.

    The linear N->S model deviates from the true swath pattern by up to
    ~±45 s (row) plus ~12 s within-row spread — irrelevant when both stereo
    partners share the model (errors cancel in the pairs) but directly
    biasing cross-instrument dt when the partner has exact pixel times
    (e.g. FCI).
    """
    key = str(nc_path)
    if key not in _abi_lut_cache:
        ds = xr.open_dataset(key, decode_timedelta=True)
        try:
            vals = ds["FD_pixel_times"].values
        finally:
            ds.close()
        lut = vals.astype("timedelta64[ms]").astype(np.float64).T / 1e3
        _abi_lut_cache[key] = lut
    return _abi_lut_cache[key]


def _to_unix(t: datetime) -> float:
    """Datetime (naive = UTC) → float Unix seconds."""
    return float(
        (np.datetime64(t.replace(tzinfo=None)) - _UNIX_EPOCH) / np.timedelta64(1, "s")
    )


def _scene_row_times(
    info: dict, sat: SatelliteConfig, rows: np.ndarray
) -> np.ndarray:
    """Per-row acquisition time (Unix seconds) via the linear scan model.

    Anchored on the scene's actual observation start/end when available
    (``info["t_start"]``/``info["t_end"]``), else the nominal time and the
    default full-disk duration.
    """
    t_start = info.get("t_start") or info["t_nominal"]
    t_end = info.get("t_end")
    if t_end is not None and info.get("t_start") is not None:
        duration = (t_end - t_start).total_seconds()
    else:
        duration = ABI_FULL_DISK_DURATION
    return _to_unix(t_start) + abi_pixel_times(rows, sat, duration)


def _sample_b_grid(
    field_b: np.ndarray,
    col_b: np.ndarray,
    row_b: np.ndarray,
    fill: float,
) -> np.ndarray:
    """Nearest-neighbor sample a B-grid field at the remap LUT positions.

    Invalid LUT entries (NaN / out of bounds) and non-finite samples get
    ``fill``. Nearest-neighbor is plenty for time (sub-second variation
    between adjacent pixels).
    """
    rr = np.rint(row_b)
    cc = np.rint(col_b)
    ok = (
        np.isfinite(rr) & np.isfinite(cc)
        & (rr >= 0) & (rr < field_b.shape[0])
        & (cc >= 0) & (cc < field_b.shape[1])
    )
    rr_i = np.where(ok, rr, 0).astype(np.intp)
    cc_i = np.where(ok, cc, 0).astype(np.intp)
    out = field_b[rr_i, cc_i]
    return np.where(ok & np.isfinite(out), out, fill)


def compute_scene_dt_fields(
    time_info: dict[str, dict],
    sat_a: SatelliteConfig,
    sat_b: SatelliteConfig,
    col_b: np.ndarray,
    row_b: np.ndarray,
    abi_time_lut: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-pixel (H, W) time offsets (seconds) of each scene relative to A0.

    This is what the solver's velocity columns should multiply: the actual
    acquisition-time difference between the pixel's observation in scene k
    and its observation in A0, on satellite A's grid.

    - A scenes use the linear row-scan model anchored on actual observation
      bounds; the scan phase cancels between A scenes (same instrument),
      leaving the (row-dependent, if durations differ) start-time offset.
    - B scenes use the satellite's native per-pixel acquisition time field
      (``time_info[k]["pixel_time"]``, e.g. FCI's index_map→time LUT) sampled
      through the remap LUT; without one, the linear model is evaluated at
      the remapped row (``row_b``), which captures the cross-instrument
      scan-phase difference to the accuracy of the linear model.
    - Pixels the B satellite cannot see fall back to the nominal offset so
      the design matrix stays finite (their disparities are NaN-masked
      anyway).
    - ``abi_time_lut``: optional (H, W) per-pixel scan-time LUT for the A
      satellite (see :func:`load_abi_time_lut`). Replaces the linear model
      for the A0 reference of the cross-satellite pairs — the temporal
      A pairs are unaffected (any shared scan model cancels there).

    Parameters
    ----------
    time_info : per-scene dict from ``load_stereo_scenes(return_times=True)``
        with keys "t_nominal", "t_start", "t_end", "pixel_time".
    sat_a, sat_b : satellite configs (grid dimensions, scan model)
    col_b, row_b : (H, W) remap LUT (A-grid pixel → B-grid position)

    Returns
    -------
    dict with keys A_minus, A_plus, B_minus, B_plus of (H, W) float64
    seconds relative to the A0 pixel time.
    """
    H, W = sat_a.n_rows, sat_a.n_cols
    rows_a = np.arange(H, dtype=np.float64)[:, None]  # (H, 1)

    t_a0 = _scene_row_times(time_info["A0"], sat_a, rows_a)  # (H, 1)
    t_a0_nom = _to_unix(time_info["A0"]["t_nominal"])

    # Per-pixel A0 reference for the cross-satellite pairs: LUT if given
    # (exact swath pattern), else the linear row model.
    if abi_time_lut is not None:
        if abi_time_lut.shape != (H, W):
            raise ValueError(
                f"abi_time_lut shape {abi_time_lut.shape} != grid ({H}, {W})")
        a0_start = time_info["A0"].get("t_start") or time_info["A0"]["t_nominal"]
        t_a0_ref = _to_unix(a0_start) + abi_time_lut  # (H, W)
    else:
        t_a0_ref = t_a0  # (H, 1), broadcasts

    out: dict[str, np.ndarray] = {}

    # Temporal pairs (same instrument): row phase cancels up to duration
    # differences; (H, 1) broadcast keeps that exactness cheaply.
    for name in ("A_minus", "A_plus"):
        t_k = _scene_row_times(time_info[name], sat_a, rows_a)
        out[name] = np.broadcast_to((t_k - t_a0), (H, W)).astype(np.float64)

    # Cross-satellite pairs: full per-pixel field on A's grid.
    for name in ("B_minus", "B_plus"):
        info = time_info[name]
        t_k_nom = _to_unix(info["t_nominal"])
        fill_dt = t_k_nom - t_a0_nom

        pt_b = info.get("pixel_time")
        if pt_b is None:
            # Linear model on the B grid, evaluated at the remapped rows
            t_start = info.get("t_start") or info["t_nominal"]
            t_end = info.get("t_end")
            duration = (
                (t_end - t_start).total_seconds()
                if (t_end is not None and info.get("t_start") is not None)
                else ABI_FULL_DISK_DURATION
            )
            pt_b_rows = (
                _to_unix(t_start)
                + abi_pixel_times(np.arange(sat_b.n_rows, dtype=np.float64),
                                  sat_b, duration)
            )
            pt_b = np.broadcast_to(pt_b_rows[:, None], (sat_b.n_rows, sat_b.n_cols))

        t_k = _sample_b_grid(np.asarray(pt_b, dtype=np.float64), col_b, row_b,
                             fill=np.nan)
        dt = t_k - t_a0_ref  # (H, W) − (H, W) or (H, 1)
        out[name] = np.where(np.isfinite(dt), dt, fill_dt).astype(np.float64)

    return out
