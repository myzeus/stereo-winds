"""ERA5 reanalysis access and sampling for stereo-wind validation.

Thin wrapper over the zeus ``ERA5`` reader (public ``arco-era5`` GCS zarr) plus
the nearest-grid / nearest-level-by-height sampling recipe used to compare
retrieved cloud-top winds against reanalysis.

Public surface
--------------
- ``load_era5_for_stereo(stereo_ds)`` — single-scene ERA5 for the stereo scene's
  time, with ``u_component_of_wind``, ``v_component_of_wind`` and
  ``geometric_height`` on ``(level, lat, lon)``.  This is the contract
  ``scripts/eval_at_sondes.py`` imports (it also unblocks that script).
- ``load_era5_single_time(reader, time, lat_bbox, lon_bbox)`` — one analysis time,
  optionally bbox-subset and ``.load()``-ed, for batched per-scene use.
- ``sample_era5(era5_1t, lats, lons, heights)`` — vectorized sampler returning the
  ``(u, v)`` of the nearest grid cell and the level closest in geometric height.
- ``open_era5_reader(levels)`` — construct the underlying zeus reader once.
- ``resolve_sat_config(stereo_ds)`` — infer the ``SatelliteConfig`` for a stereo
  product (defaults to GOES-19).

Longitude is normalized to ``[-180, 180)`` so it matches IGRA / GOES geodetic
longitudes; ERA5 native longitude is ``[0, 360)``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Standard gravity used to convert ERA5 geopotential (m^2 s^-2) to geometric
# height (m).  ERA5 reports geopotential, not geopotential height.
G0 = 9.80665

# Pressure levels (hPa) retained for cloud-top matching.  Dense enough through
# the troposphere/lower stratosphere where cloud tops live; trimming the full
# 37-level set keeps the per-scene GCS read small.
DEFAULT_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50]

_WIND_VARS = ["u_component_of_wind", "v_component_of_wind", "geopotential"]


def open_era5_reader(levels=DEFAULT_LEVELS):
    """Construct an ERA5 reader over the public ARCO-ERA5 zarr.

    Not included in the standalone build. The rest of this module (matching,
    interpolation, metrics) works on any ERA5 xarray dataset you supply.
    """
    raise NotImplementedError(
        "The bundled ERA5 reader is not part of the standalone stereo-winds "
        "build. Pass your own ERA5 xarray.Dataset to the matching/metric "
        "helpers in this module instead.")


def _normalize_lon(ds: xr.Dataset) -> xr.Dataset:
    """Rename longitude to ``lon`` in [-180, 180), sorted ascending."""
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lon = ds[lon_name].values
    if np.nanmax(lon) > 180.0:
        ds = ds.assign_coords({lon_name: ((lon + 180.0) % 360.0) - 180.0})
        ds = ds.sortby(lon_name)
    return ds


def _standardize(ds: xr.Dataset) -> xr.Dataset:
    """Canonicalize coord names to lat/lon and add ``geometric_height``."""
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    if "geopotential" in ds and "geometric_height" not in ds:
        ds = ds.assign(geometric_height=ds["geopotential"] / G0)
    return ds


def _subset_bbox(ds: xr.Dataset, lat_bbox, lon_bbox) -> xr.Dataset:
    """Slice to a lat/lon box, honoring descending-latitude ERA5 grids."""
    if lat_bbox is not None:
        lo, hi = sorted(lat_bbox)
        lat = ds["lat"].values
        ds = ds.sel(lat=slice(hi, lo)) if lat[0] > lat[-1] else ds.sel(lat=slice(lo, hi))
    if lon_bbox is not None:
        lo, hi = sorted(lon_bbox)
        ds = ds.sel(lon=slice(lo, hi))
    return ds


def load_era5_single_time(reader, time, lat_bbox=None, lon_bbox=None) -> xr.Dataset:
    """Load one ERA5 analysis time, standardized, bbox-subset and ``.load()``-ed.

    Parameters
    ----------
    reader : zeus ERA5
        From :func:`open_era5_reader`.
    time : datetime-like
        Analysis time; the reader selects the nearest available hour.
    lat_bbox, lon_bbox : (float, float), optional
        Inclusive bounds; longitudes in [-180, 180).
    """
    t = pd.Timestamp(np.asarray(time).astype("datetime64[ns]").item())
    ds = reader.read_analysis(t.to_pydatetime())
    ds = ds[[v for v in _WIND_VARS if v in ds]]
    ds = _normalize_lon(ds)
    ds = _standardize(ds)
    ds = _subset_bbox(ds, lat_bbox, lon_bbox)
    return ds.load()


def load_era5_for_stereo(stereo_ds, levels=DEFAULT_LEVELS,
                         lat_bbox=None, lon_bbox=None) -> xr.Dataset:
    """Single-scene ERA5 for a stereo retrieval's time (eval_at_sondes contract)."""
    reader = open_era5_reader(levels)
    t0 = np.asarray(stereo_ds.time.values).astype("datetime64[ns]").flat[0]
    return load_era5_single_time(reader, t0, lat_bbox=lat_bbox, lon_bbox=lon_bbox)


def sample_era5(era5_1t: xr.Dataset, lats, lons, heights):
    """Sample ERA5 ``(u, v)`` at nearest grid cell and nearest level by height.

    Parameters
    ----------
    era5_1t : xr.Dataset
        Single analysis time with ``u_component_of_wind``,
        ``v_component_of_wind`` and ``geometric_height`` on ``(level, lat, lon)``.
    lats, lons, heights : array-like
        Target latitudes (deg), longitudes (deg, [-180, 180)) and geometric
        heights (m).  Broadcast to a common shape.

    Returns
    -------
    (u, v) : np.ndarray, np.ndarray
        Same shape as the broadcast inputs; NaN where an input is non-finite.
    """
    u_all = np.asarray(era5_1t["u_component_of_wind"].values)  # (level, lat, lon)
    v_all = np.asarray(era5_1t["v_component_of_wind"].values)
    h_all = np.asarray(era5_1t["geometric_height"].values)
    elat = np.asarray(era5_1t["lat"].values)
    elon = np.asarray(era5_1t["lon"].values)

    lats = np.atleast_1d(np.asarray(lats, dtype=np.float64))
    lons = np.atleast_1d(np.asarray(lons, dtype=np.float64))
    heights = np.atleast_1d(np.asarray(heights, dtype=np.float64))
    lats, lons, heights = np.broadcast_arrays(lats, lons, heights)
    shape = lats.shape
    lats, lons, heights = lats.ravel(), lons.ravel(), heights.ravel()

    out_u = np.full(lats.shape, np.nan)
    out_v = np.full(lats.shape, np.nan)
    for i in range(lats.size):
        if not (np.isfinite(lats[i]) and np.isfinite(lons[i]) and np.isfinite(heights[i])):
            continue
        ai = int(np.argmin(np.abs(elat - lats[i])))
        oi = int(np.argmin(np.abs(elon - lons[i])))
        hcol = h_all[:, ai, oi]
        if not np.any(np.isfinite(hcol)):
            continue
        li = int(np.nanargmin(np.abs(hcol - heights[i])))
        out_u[i] = u_all[li, ai, oi]
        out_v[i] = v_all[li, ai, oi]
    return out_u.reshape(shape), out_v.reshape(shape)


def resolve_sat_config(stereo_ds):
    """Infer the SatelliteConfig for a stereo product (defaults to GOES-19)."""
    from stereo_winds.config import GOES16_CONFIG, GOES19_CONFIG

    hay = " ".join(
        str(stereo_ds.attrs.get(k, "")) for k in ("satellite", "sat_a", "title")
    ).lower()
    if "16" in hay or "goes-16" in hay or "goes16" in hay:
        return GOES16_CONFIG
    return GOES19_CONFIG
