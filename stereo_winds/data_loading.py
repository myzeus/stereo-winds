"""Native fixed-grid data loading for ABI and AHI.

Loads satellite imagery in native scanning-angle coordinates (radians),
preserving the fixed-grid projection needed for parallax retrieval.
Supports both direct netCDF loading and zeus GOES/AHI data sources.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .config import SatelliteConfig, SATELLITE_CONFIGS

logger = logging.getLogger(__name__)

# Ensure zeus is on the import path
_zeus_path = str(Path(__file__).resolve().parent.parent / "zeus")
if _zeus_path not in sys.path:
    sys.path.insert(0, _zeus_path)


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
    try:
        ds = xr.open_dataset(str(nc_path), engine="h5netcdf")
    except Exception:
        ds = xr.open_dataset(str(nc_path), engine="netcdf4")
    try:
        data = ds["Rad"].values.astype(np.float32)
        sat_config = _sat_config_from_nc(ds, satellite_id)
    finally:
        ds.close()

    return data, sat_config


def _sat_config_from_satpy(ds: xr.Dataset, satellite_id: str) -> SatelliteConfig:
    """Build a SatelliteConfig from a satpy-returned xr.Dataset.

    Satpy returns x/y in meters (scanning_angle * satellite_height) with y
    ascending (south→north). We convert to radians and set y_offset to the
    north edge to match native ABI convention (row 0 = north).
    """
    rad_attrs = ds["Rad"].attrs
    orbital = json.loads(rad_attrs["orbital_parameters"])

    sat_height = float(orbital["projection_altitude"])
    sub_lon = float(orbital["satellite_nominal_longitude"])

    x_vals = ds["x"].values.astype(np.float64)
    y_vals = ds["y"].values.astype(np.float64)
    n_rows = len(y_vals)
    n_cols = len(x_vals)

    # Convert meters → radians
    scale_x = float((x_vals[-1] - x_vals[0]) / (n_cols - 1) / sat_height)
    x_offset = float(x_vals[0] / sat_height)

    # y ascending in satpy (south→north); after flip, row 0 = north edge
    scale_y = -float((y_vals[-1] - y_vals[0]) / (n_rows - 1) / sat_height)
    y_offset = float(y_vals[-1] / sat_height)  # north edge

    return SatelliteConfig(
        satellite_id=satellite_id,
        sub_lon_deg=sub_lon,
        satellite_height_m=sat_height,
        sweep="x",  # ABI always x-sweep
        scale_x=scale_x,
        scale_y=scale_y,
        x_offset=x_offset,
        y_offset=y_offset,
        n_rows=n_rows,
        n_cols=n_cols,
    )


def _make_goes_source(
    satellite: str,
    band: str,
    cache_dir: str | Path | None = None,
    product: str = "ABI-L1b-RadF",
):
    """Create a zeus GOES data source instance."""
    from zeus.datasets.sources.goes import GOES
    from zeus.datasets.core.base import DataSourceConfig

    config = DataSourceConfig(cache_dir=cache_dir)
    return GOES(
        config=config,
        satellite=satellite,
        product=product,
        bands=[band],
    )


def _ensure_band_downloaded(source, t: dt.datetime, band: str) -> None:
    """Ensure band-specific files are downloaded.

    Zeus local_files() matches by time only, so if another band's files
    are cached for the same time, it skips downloading. This function
    checks that the requested band is actually present and downloads
    if not.
    """
    t = source._snap_time(t)
    cache_dir_t = source._get_cache_dir(t)
    cache_dir_t.mkdir(parents=True, exist_ok=True)

    # Check if band-specific files already exist locally
    band_pattern = f"*{band}*_s{t:%Y%j%H%M}*.nc"
    existing = list(cache_dir_t.glob(band_pattern))
    if existing:
        return

    # Band not cached — download from S3
    remote_files = source._get_remote_files(t)
    if not remote_files:
        raise ValueError(f"No remote files for {band} at {t}")

    for remote in remote_files:
        local = cache_dir_t / Path(remote).name
        if not local.exists():
            logger.info("Downloading %s", Path(remote).name)
            source._download_file(remote, local)


def load_goes_scene(
    t: dt.datetime,
    band: str,
    satellite: str = "goes16",
    cache_dir: str | Path | None = None,
    product: str = "ABI-L1b-RadF",
    stream: bool = False,
) -> tuple[np.ndarray, SatelliteConfig]:
    """Load a GOES ABI scene using zeus, returning native fixed-grid data.

    Uses GOES.data_at_time() for time-snapping, S3 download or streaming,
    and satpy-based loading. Converts satpy output back to native fixed-grid
    coordinates needed by the stereo solver.

    Parameters
    ----------
    t : target datetime (snapped to nearest valid scan time)
    band : ABI band name (e.g., "C14", "C04")
    satellite : satellite identifier ("goes16", "goes18", "goes19")
    cache_dir : local cache directory for downloaded files
    product : ABI product type ("ABI-L1b-RadF", "ABI-L1b-RadC", "ABI-L1b-RadM")
    stream : if True, read directly from S3 without caching to disk

    Returns
    -------
    data : (n_rows, n_cols) float32 array, row 0 = north
    sat_config : SatelliteConfig with scanning-angle coordinates in radians
    """
    source = _make_goes_source(satellite, band, cache_dir, product)
    logger.info("Loading %s %s at %s via zeus%s", satellite, band, t,
                " (streaming)" if stream else "")

    if not stream:
        # Ensure band-specific files are downloaded (zeus cache is not band-aware)
        _ensure_band_downloaded(source, t, band)

    ds = source.data_at_time(t, download=not stream)

    # Extract 2D array: squeeze time and band dims
    data = ds["Rad"].values[0, 0, :, :].astype(np.float32)

    # Flip y: satpy returns south→north, we need north→south (row 0 = north)
    data = data[::-1]

    sat_config = _sat_config_from_satpy(ds, satellite)
    logger.info(
        "  %s: %dx%d, sub_lon=%.2f°",
        satellite, sat_config.n_rows, sat_config.n_cols, sat_config.sub_lon_deg,
    )
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
    source = _make_goes_source(satellite, band, cache_dir)
    return source.download(t)


def download_ahi(
    t: dt.datetime,
    band: str,
    satellite: str = "himawari8",
    cache_dir: str | Path | None = None,
) -> list[Path]:
    """Download AHI L1b files for a given time using zeus AHI source."""
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
    product: str = "ABI-L1b-RadF",
    stream: bool = False,
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
    product : ABI product type (e.g., "ABI-L1b-RadF")

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

        if "goes" in sat_id:
            data, config = load_goes_scene(t, band, sat_id, cache_dir, product=product, stream=stream)
        elif "himawari" in sat_id:
            files = download_ahi(t, band, sat_id, cache_dir)
            if not files:
                raise FileNotFoundError(
                    f"No data found for {sat_id} at {t} band {band}"
                )
            data, config = load_native_abi(files[0], sat_id)
        else:
            raise ValueError(f"Unknown satellite type: {sat_id}")

        scenes[name] = (data, config)

    return scenes
