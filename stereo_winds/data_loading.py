"""Native fixed-grid data loading for ABI, AHI, and FCI.

Loads satellite imagery in native scanning-angle coordinates (radians),
preserving the fixed-grid projection needed for parallax retrieval.
Supports both direct netCDF loading and zeus GOES/AHI/FCI data sources.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .config import ABI_TO_FCI_BAND, MTG_I1_CONFIG, SatelliteConfig, SATELLITE_CONFIGS

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


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """f×f block-mean downsample (NaN-aware; all-NaN blocks stay NaN)."""
    H, W = a.shape
    a = a[: H - H % f, : W - W % f]
    a = a.reshape(H // f, f, W // f, f)
    with np.errstate(invalid="ignore"):
        return np.nanmean(a, axis=(1, 3)).astype(np.float32)


def _coarsen_to_canonical(
    data: np.ndarray, cfg: SatelliteConfig, satellite_id: str
) -> tuple[np.ndarray, SatelliteConfig, int]:
    """Coarsen a finer-than-canonical band to the registry's 2-km grid.

    High-resolution bands (ABI C02 at 0.5 km, FCI vis at 1 km) are
    block-mean downsampled so every band lives on the canonical grid that
    the remap LUT, parallax vectors, and solver use. Scale doubles per
    factor and the offset moves to the first block's center.

    Returns (data, cfg, factor); factor == 1 means no-op.
    """
    canonical = SATELLITE_CONFIGS.get(satellite_id)
    if canonical is None or cfg.n_rows <= canonical.n_rows:
        return data, cfg, 1
    f = int(round(cfg.n_rows / canonical.n_rows))
    if f <= 1 or cfg.n_rows % canonical.n_rows:
        return data, cfg, 1
    logger.info("  coarsening %dx%d -> %dx%d (factor %d)",
                cfg.n_rows, cfg.n_cols, canonical.n_rows, canonical.n_cols, f)
    data = _block_mean(data, f)
    from dataclasses import replace
    cfg = replace(
        cfg,
        scale_x=cfg.scale_x * f,
        scale_y=cfg.scale_y * f,
        x_offset=cfg.x_offset + (f - 1) / 2.0 * cfg.scale_x,
        y_offset=cfg.y_offset + (f - 1) / 2.0 * cfg.scale_y,
        n_rows=cfg.n_rows // f,
        n_cols=cfg.n_cols // f,
    )
    return data, cfg, f


def _orbital_params(rad_attrs: dict) -> dict:
    """Decode satpy 'orbital_parameters' (JSON string or dict)."""
    orbital = rad_attrs["orbital_parameters"]
    if isinstance(orbital, str):
        orbital = json.loads(orbital)
    return orbital


def _sat_config_from_satpy(
    ds: xr.Dataset, satellite_id: str, sweep: str = "x"
) -> SatelliteConfig:
    """Build a SatelliteConfig from a satpy-returned xr.Dataset.

    Satpy returns x/y in meters (scanning_angle * satellite_height) with y
    ascending (south→north). We convert to radians and set y_offset to the
    north edge to match native ABI convention (row 0 = north).
    """
    rad_attrs = ds["Rad"].attrs
    orbital = _orbital_params(rad_attrs)

    sat_height = float(orbital["projection_altitude"])
    # Reader-dependent key: abi_l1b uses satellite_nominal_longitude,
    # fci_l1c_nc defines the fixed grid at projection_longitude.
    for key in ("projection_longitude", "satellite_nominal_longitude"):
        if key in orbital:
            sub_lon = float(orbital[key])
            break
    else:
        raise KeyError(f"No sub-satellite longitude in orbital_parameters: {orbital}")

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
        sweep=sweep,
        scale_x=scale_x,
        scale_y=scale_y,
        x_offset=x_offset,
        y_offset=y_offset,
        n_rows=n_rows,
        n_cols=n_cols,
    )


def _scene_time_bounds(ds: xr.Dataset) -> tuple[dt.datetime | None, dt.datetime | None]:
    """Actual observation start/end from satpy Rad attrs, if present."""
    attrs = ds["Rad"].attrs

    def _get(*keys):
        for k in keys:
            v = attrs.get(k)
            if v is None:
                continue
            if isinstance(v, dt.datetime):
                return v
            try:
                return dt.datetime.fromisoformat(str(v))
            except ValueError:
                continue
        return None

    start = _get("observation_start_time", "start_time", "time_coverage_start")
    end = _get("observation_end_time", "end_time", "time_coverage_end")
    return start, end


def _make_goes_source(
    satellite: str,
    band: str,
    cache_dir: str | Path | None = None,
    product: str = "ABI-L1b-RadF",
):
    """Create a standalone public-S3 GOES ABI reader."""
    from stereo_winds.readers.goes import GOES

    return GOES(
        satellite=satellite,
        product=product,
        bands=[band],
        cache_dir=cache_dir,
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
    return_aux: bool = False,
):
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
    return_aux : if True, also return an aux dict with the actual
        observation "t_start"/"t_end" (datetime or None)

    Returns
    -------
    data : (n_rows, n_cols) float32 array, row 0 = north
    sat_config : SatelliteConfig with scanning-angle coordinates in radians
    aux : dict (only if return_aux)
    """
    source = _make_goes_source(satellite, band, cache_dir, product)
    logger.info("Loading %s %s at %s%s", satellite, band, t,
                " (streaming)" if stream else "")

    if not stream and hasattr(source, "_get_remote_files"):
        # Legacy zeus source: cache is not band-aware, so force the band.
        # The standalone reader self-downloads the correct band.
        _ensure_band_downloaded(source, t, band)

    ds = source.data_at_time(t, download=not stream)

    # Extract 2D array: squeeze time and band dims
    data = ds["Rad"].values[0, 0, :, :].astype(np.float32)

    # Flip y: satpy returns south→north, we need north→south (row 0 = north)
    data = data[::-1]

    sat_config = _sat_config_from_satpy(ds, satellite)
    # High-res VIS bands (e.g. C02 0.5 km) -> canonical 2 km grid
    data, sat_config, _ = _coarsen_to_canonical(data, sat_config, satellite)
    logger.info(
        "  %s: %dx%d, sub_lon=%.2f°",
        satellite, sat_config.n_rows, sat_config.n_cols, sat_config.sub_lon_deg,
    )
    if return_aux:
        t_start, t_end = _scene_time_bounds(ds)
        return data, sat_config, {"t_start": t_start, "t_end": t_end,
                                  "pixel_time": None}
    return data, sat_config


_UNIX_EPOCH = np.datetime64("1970-01-01T00:00:00")


def _pixel_time_to_unix(arr: np.ndarray, attrs: dict) -> np.ndarray:
    """Convert an FCI per-pixel time array to float64 Unix seconds.

    Handles datetime64 arrays and numeric arrays with CF "since"-style
    units. Bare "s"/"seconds" units are FCI's IDPF convention: seconds
    since 2000-01-01T00:00:00 UTC.
    """
    if np.issubdtype(arr.dtype, np.datetime64):
        return (arr - _UNIX_EPOCH) / np.timedelta64(1, "s")

    arr = arr.astype(np.float64)
    units = str(attrs.get("units", "")).strip()
    m = re.search(r"since\s+([0-9T:\- .]+)", units)
    if m:
        epoch_str = m.group(1).strip().replace(" ", "T").rstrip("Z")
        epoch = np.datetime64(epoch_str)
    else:
        # FCI L1c time LUT: seconds since J2000 epoch
        epoch = np.datetime64("2000-01-01T00:00:00")
        if units not in ("s", "seconds", "sec"):
            logger.warning(
                "pixel_time units %r not recognized; assuming seconds since "
                "2000-01-01T00:00:00 UTC", units)
    offset = float((epoch - _UNIX_EPOCH) / np.timedelta64(1, "s"))
    return arr + offset


def load_fci_scene(
    t: dt.datetime,
    band: str,
    satellite: str = "mtg-i1",
    cache_dir: str | Path | None = None,
    return_aux: bool = False,
):
    """Load an MTG FCI L1c scene using zeus, in native fixed-grid coordinates.

    The retrieval band is given as the ABI name (e.g. "C13") and translated
    to the FCI channel via ``ABI_TO_FCI_BAND``; a native FCI name (e.g.
    "ir_105") is also accepted.

    Parameters
    ----------
    t : target datetime (snapped to the 10-min repeat cycle)
    band : ABI band name ("C13") or FCI channel name ("ir_105")
    satellite : satellite identifier ("mtg-i1")
    cache_dir : local cache directory for downloaded chunk files
    return_aux : if True, also return an aux dict with actual observation
        "t_start"/"t_end" and the per-pixel acquisition time field
        "pixel_time" ((H, W) float64 Unix seconds, row 0 = north)

    Returns
    -------
    data : (n_rows, n_cols) float32 array, row 0 = north
    sat_config : SatelliteConfig with scanning-angle coordinates in radians
    aux : dict (only if return_aux)
    """
    import dask
    from zeus.datasets.core.base import DataSourceConfig
    from zeus.datasets.sources.mtg_fci import FCI, BANDS_ALL

    fci_band = band if band in BANDS_ALL else ABI_TO_FCI_BAND.get(band)
    if fci_band is None:
        raise ValueError(
            f"No FCI equivalent for band {band!r}. "
            f"ABI bands with a twin: {sorted(ABI_TO_FCI_BAND)}"
        )

    source = FCI(
        config=DataSourceConfig(cache_dir=cache_dir),
        bands=[fci_band],
    )
    logger.info("Loading %s %s (from %s) at %s via zeus", satellite, fci_band, band, t)

    # data_at_time reads synchronously and returns a fully materialized
    # dataset (threaded HDF5 reads segfault; handles close with the Scene).
    ds = source.data_at_time(t, include_pixel_times=return_aux)

    # Extract 2D array: squeeze time and band dims; flip to row 0 = north
    data = ds["Rad"].values[0, 0, :, :].astype(np.float32)[::-1]

    sat_config = _sat_config_from_satpy(ds, satellite, sweep=MTG_I1_CONFIG.sweep)
    # High-res VIS bands (1 km) -> canonical 2 km grid
    data, sat_config, coarsen_f = _coarsen_to_canonical(data, sat_config, satellite)
    # fci_l1c_nc exposes the projection sweep via the area; if satpy carried
    # it through in the attrs, prefer the file's value.
    proj = ds["Rad"].attrs.get("mtg_geos_projection", None)
    if isinstance(proj, dict) and "sweep_angle_axis" in proj:
        sat_config.sweep = str(proj["sweep_angle_axis"])

    logger.info(
        "  %s: %dx%d, sub_lon=%.2f°, sweep=%s",
        satellite, sat_config.n_rows, sat_config.n_cols,
        sat_config.sub_lon_deg, sat_config.sweep,
    )

    if return_aux:
        pt = ds["pixel_time"].values[0, 0, :, :]
        pt = _pixel_time_to_unix(pt, ds["pixel_time"].attrs)[::-1]
        # Mask fill values: anything > 1 day from the nominal time is junk
        t_unix = (np.datetime64(t.replace(tzinfo=None)) - _UNIX_EPOCH) / np.timedelta64(1, "s")
        pt = np.where(np.abs(pt - t_unix) < 86400.0, pt, np.nan)
        if coarsen_f > 1:
            # Subsample is fine for time (adjacent-pixel variation << 1 s)
            pt = pt[::coarsen_f, ::coarsen_f][:sat_config.n_rows, :sat_config.n_cols]
        t_start, t_end = _scene_time_bounds(ds)
        return data, sat_config, {"t_start": t_start, "t_end": t_end,
                                  "pixel_time": pt}
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
    """Download AHI L1b files. Not available in the standalone build."""
    raise NotImplementedError(
        "Himawari AHI loading is not included in the standalone stereo-winds "
        "build. GOES-R ABI (readers.goes) and MTG FCI (readers.fci) are "
        "supported; add an AHI reader to enable this path.")


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
    return_times: bool = False,
):
    """Load all 5 scenes for a stereo retrieval.

    Scenes: A_minus, A0, A_plus, B_minus, B_plus.

    Parameters
    ----------
    t0 : center time
    dt_minutes : time offset for temporal pairs
    sat_a_id, sat_b_id : satellite identifiers (e.g., "goes16", "mtg-i1")
    band_a : band for satellite A
    band_b : band for satellite B (defaults to band_a; for an FCI
        satellite B an ABI band name is translated via ABI_TO_FCI_BAND)
    cache_dir : local cache directory
    product : ABI product type (e.g., "ABI-L1b-RadF")
    return_times : if True, also return a per-scene timing dict with keys
        "t_nominal" (requested datetime), "t_start"/"t_end" (actual
        observation bounds, or None), and "pixel_time" ((H, W) float64 Unix
        seconds on the scene's native grid, or None). Used to build
        per-pixel dt fields for the solver.

    Returns dict mapping scene name to (data_2d, SatelliteConfig); with
    return_times, returns (scenes, time_info).
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
    time_info = {}
    for name, t in times.items():
        is_sat_b = name.startswith("B")
        sat_id = sat_b_id if is_sat_b else sat_a_id
        band = band_b if is_sat_b else band_a
        aux = {"t_start": None, "t_end": None, "pixel_time": None}

        if "goes" in sat_id:
            out = load_goes_scene(t, band, sat_id, cache_dir, product=product,
                                  stream=stream, return_aux=return_times)
            data, config = out[0], out[1]
            if return_times:
                aux = out[2]
        elif "himawari" in sat_id:
            files = download_ahi(t, band, sat_id, cache_dir)
            if not files:
                raise FileNotFoundError(
                    f"No data found for {sat_id} at {t} band {band}"
                )
            data, config = load_native_abi(files[0], sat_id)
        elif "mtg" in sat_id:
            out = load_fci_scene(t, band, sat_id, cache_dir,
                                 return_aux=return_times)
            data, config = out[0], out[1]
            if return_times:
                aux = out[2]
        else:
            raise ValueError(f"Unknown satellite type: {sat_id}")

        scenes[name] = (data, config)
        time_info[name] = {"t_nominal": t, **aux}

    if return_times:
        return scenes, time_info
    return scenes
