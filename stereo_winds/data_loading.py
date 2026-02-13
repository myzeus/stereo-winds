"""Native fixed-grid data loading for ABI and AHI.

Loads satellite imagery in native scanning-angle coordinates (radians),
preserving the fixed-grid projection needed for parallax retrieval.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .config import SatelliteConfig, SATELLITE_CONFIGS


def _sat_config_from_nc(ds: xr.Dataset, satellite_id: str) -> SatelliteConfig:
    """Build a SatelliteConfig from ABI L1b netCDF projection metadata."""
    proj = ds["goes_imager_projection"]
    attrs = proj.attrs

    sub_lon = float(attrs["longitude_of_projection_origin"])
    sat_height = float(attrs["perspective_point_height"])
    semi_major = float(attrs["semi_major_axis"])
    semi_minor = float(attrs["semi_minor_axis"])
    sweep = str(attrs["sweep_angle_axis"])

    # Derive scale/offset from actual coordinate arrays.
    # ABI L1b stores x and y as explicit coordinate arrays (radians),
    # not via scale_factor/add_offset attributes.
    x_vals = ds["x"].values.astype(np.float64)
    y_vals = ds["y"].values.astype(np.float64)
    n_rows, n_cols = ds["Rad"].shape

    scale_x = float((x_vals[-1] - x_vals[0]) / (len(x_vals) - 1)) if len(x_vals) > 1 else 5.6e-05
    scale_y = float((y_vals[-1] - y_vals[0]) / (len(y_vals) - 1)) if len(y_vals) > 1 else -5.6e-05
    x_offset = float(x_vals[0])
    y_offset = float(y_vals[0])

    return SatelliteConfig(
        satellite_id=satellite_id,
        sub_lon_deg=sub_lon,
        satellite_height_m=sat_height,
        semi_major_m=semi_major,
        semi_minor_m=semi_minor,
        sweep=sweep,
        scale_x=scale_x,
        scale_y=scale_y,
        x_offset=x_offset,
        y_offset=y_offset,
        n_rows=n_rows,
        n_cols=n_cols,
    )


def load_native_abi(
    nc_path: str | Path,
    satellite_id: str = "goes16",
) -> tuple[np.ndarray, SatelliteConfig]:
    """Load ABI L1b radiance in native fixed-grid coordinates.

    Parameters
    ----------
    nc_path : path to ABI L1b netCDF file
    satellite_id : satellite identifier for config

    Returns
    -------
    data : (n_rows, n_cols) float32 array of radiance values
    sat_config : SatelliteConfig populated from file metadata
    """
    ds = xr.open_dataset(str(nc_path), engine="h5netcdf")
    try:
        data = ds["Rad"].values.astype(np.float32)
        sat_config = _sat_config_from_nc(ds, satellite_id)
    finally:
        ds.close()

    return data, sat_config


def download_abi(
    t: dt.datetime,
    band: str,
    satellite: str = "goes16",
    cache_dir: str | Path | None = None,
) -> list[Path]:
    """Download ABI L1b files for a given time using zeus GOES source.

    Returns list of local file paths.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "zeus"))
    from zeus.datasets.sources.goes import GOES
    from zeus.datasets.core.base import DataSourceConfig

    config = DataSourceConfig(cache_dir=cache_dir)
    source = GOES(
        config=config,
        satellite=satellite,
        product="ABI-L1b-RadF",
        bands=[band],
    )
    return source.download(t)


def download_ahi(
    t: dt.datetime,
    band: str,
    satellite: str = "himawari8",
    cache_dir: str | Path | None = None,
) -> list[Path]:
    """Download AHI L1b files for a given time using zeus AHI source."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "zeus"))
    from zeus.datasets.sources.ahi import AHI
    from zeus.datasets.core.base import DataSourceConfig

    config = DataSourceConfig(cache_dir=cache_dir)
    source = AHI(
        config=config,
        satellite=satellite,
        bands=[band],
    )
    return source.download(t)


def load_stereo_scenes(
    t0: dt.datetime,
    dt_minutes: float,
    sat_a_id: str,
    sat_b_id: str,
    band_a: str = "C14",
    band_b: str | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, tuple[np.ndarray, SatelliteConfig]]:
    """Load all 5 scenes for a stereo retrieval.

    Scenes: A_minus, A0, A_plus, B_minus, B_plus.

    Parameters
    ----------
    t0 : center time
    dt_minutes : time offset for temporal pairs
    sat_a_id, sat_b_id : satellite identifiers (e.g., "goes16", "goes18")
    band_a : band for satellite A
    band_b : band for satellite B (defaults to band_a)
    cache_dir : local cache directory

    Returns dict mapping scene name to (data_2d, SatelliteConfig).
    """
    if band_b is None:
        band_b = band_a

    delta = dt.timedelta(minutes=dt_minutes)
    times = {
        "A_minus": t0 - delta,
        "A0": t0,
        "A_plus": t0 + delta,
        "B_minus": t0 - delta,
        "B_plus": t0 + delta,
    }

    scenes = {}
    for name, t in times.items():
        is_sat_b = name.startswith("B")
        sat_id = sat_b_id if is_sat_b else sat_a_id
        band = band_b if is_sat_b else band_a

        # Download
        if "goes" in sat_id:
            files = download_abi(t, band, sat_id, cache_dir)
        else:
            files = download_ahi(t, band, sat_id, cache_dir)

        if not files:
            raise FileNotFoundError(
                f"No data found for {sat_id} at {t} band {band}"
            )

        # Load the first matching file
        data, config = load_native_abi(files[0], sat_id)
        scenes[name] = (data, config)

    return scenes
