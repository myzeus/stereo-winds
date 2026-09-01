"""Comparison with sparse Carr et al. stereo AMV product."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .metrics import rmsvd, speed_bias, height_rmse


def load_carr_data(filepath: str) -> dict[str, np.ndarray]:
    """Load a Carr et al. stereo-AMV retrieval NetCDF into numpy arrays.

    Parameters
    ----------
    filepath : str
        Path to a Carr retrieval NetCDF (``V_3D``/``H_3D``/``sig_H_3D``/``DQF_3D``).

    Returns
    -------
    dict
        ``lat``, ``lon``, ``u``, ``v``, ``h``, ``sig_h``, ``spd`` arrays plus a
        ``good`` boolean mask (DQF clean, finite, 0 <= h <= 20000 m) and the
        ``n_total`` / ``n_good`` counts.
    """
    try:
        ds = xr.open_dataset(filepath, engine="h5netcdf")
    except Exception:
        ds = xr.open_dataset(filepath)
    lat = ds["lat"].values
    lon = ds["lon"].values
    u = ds["V_3D"].values[:, 0]
    v = ds["V_3D"].values[:, 1]
    h = ds["H_3D"].values
    sig_h = ds["sig_H_3D"].values
    dqf = ds["DQF_3D"].values
    ds.close()

    spd = np.sqrt(u**2 + v**2)
    good = (
        (dqf == 0)
        & np.isfinite(h) & np.isfinite(u) & np.isfinite(v)
        & (h >= 0) & (h <= 20000)
    )
    return dict(
        lat=lat, lon=lon, u=u, v=v, h=h, sig_h=sig_h, spd=spd,
        good=good, n_total=len(lat), n_good=int(good.sum()),
    )


def match_dense_to_sparse(
    ds_dense: xr.Dataset,
    amv_u: np.ndarray,
    amv_v: np.ndarray,
    amv_h: np.ndarray,
    amv_row: np.ndarray,
    amv_col: np.ndarray,
) -> dict[str, np.ndarray]:
    """Match dense stereo retrievals to sparse AMV target locations.

    Parameters
    ----------
    ds_dense : Dense stereo wind Dataset
    amv_u, amv_v : Sparse AMV wind components (m/s)
    amv_h : Sparse AMV cloud-top heights (m)
    amv_row, amv_col : Pixel locations of AMV targets

    Returns
    -------
    dict with matched arrays
    """
    u_dense = ds_dense["u_wind"].values
    v_dense = ds_dense["v_wind"].values
    h_dense = ds_dense["cloud_top_height"].values

    # Extract dense values at AMV locations
    row_idx = np.clip(amv_row.astype(int), 0, u_dense.shape[0] - 1)
    col_idx = np.clip(amv_col.astype(int), 0, u_dense.shape[1] - 1)

    u_at_amv = u_dense[row_idx, col_idx]
    v_at_amv = v_dense[row_idx, col_idx]
    h_at_amv = h_dense[row_idx, col_idx]

    # Filter to valid matches
    valid = np.isfinite(u_at_amv) & np.isfinite(amv_u)

    return {
        "u_dense": u_at_amv[valid],
        "v_dense": v_at_amv[valid],
        "h_dense": h_at_amv[valid],
        "u_amv": amv_u[valid],
        "v_amv": amv_v[valid],
        "h_amv": amv_h[valid],
    }


def amv_validation_statistics(matched: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute validation statistics from dense vs sparse AMV comparison."""
    if len(matched.get("u_dense", [])) == 0:
        return {"n_matches": 0, "rmsvd": np.nan, "speed_bias": np.nan, "h_rmse": np.nan}

    return {
        "n_matches": len(matched["u_dense"]),
        "rmsvd": rmsvd(
            matched["u_dense"], matched["v_dense"],
            matched["u_amv"], matched["v_amv"],
        ),
        "speed_bias": speed_bias(
            matched["u_dense"], matched["v_dense"],
            matched["u_amv"], matched["v_amv"],
        ),
        "h_rmse": height_rmse(matched["h_dense"], matched["h_amv"]),
    }
