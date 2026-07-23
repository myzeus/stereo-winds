"""Read an EarthCARE CPR L1b (``CPR_NOM_1B``) granule into a vertical
reflectivity curtain for plotting alongside stereo/student cloud-top winds.

The ESA/JAXA CPR nominal L1b product stores radar reflectivity factor as a
2-D field ``(nray, nbin)`` in *linear* units (mm^6 m^-3) under
``ScienceData/Data/radarReflectivityFactor``, with a matching per-ray bin
height grid ``ScienceData/Geo/binHeight`` (metres, top-of-atmosphere first).
This module converts to dBZ, regrids every ray onto a common height axis, and
returns an :class:`xarray.Dataset` with an along-track distance coordinate.

Self-contained (only h5py/numpy/xarray) so it runs without the zeus EarthCARE
reader, which does not expose the 2-D reflectivity field.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

FILL = 9.96921e36
_REFL = "ScienceData/Data/radarReflectivityFactor"
_BINH = "ScienceData/Geo/binHeight"
_LAT = "ScienceData/Geo/latitude"
_LON = "ScienceData/Geo/longitude"
_SURF = "ScienceData/Geo/surfaceElevation"


def _great_circle_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cumulative along-track great-circle distance (km), starting at 0."""
    lat0, lon0 = np.radians(lat[:-1]), np.radians(lon[:-1])
    lat1, lon1 = np.radians(lat[1:]), np.radians(lon[1:])
    dlat = lat1 - lat0
    dlon = lon1 - lon0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    seg = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _read_sensing_times(h) -> tuple[np.datetime64, np.datetime64] | None:
    """Best-effort frame start/stop from HeaderData (ISO byte strings)."""
    base = "HeaderData/VariableProductHeader/MainProductHeader"
    try:
        s = h[f"{base}/sensingStartTime"][()]
        e = h[f"{base}/sensingStopTime"][()]
        s = s.decode() if isinstance(s, bytes) else str(s)
        e = e.decode() if isinstance(e, bytes) else str(e)
        # format e.g. '2025-11-07T20:57:19Z' or with fractional seconds
        return np.datetime64(s.rstrip("Z")[:19]), np.datetime64(e.rstrip("Z")[:19])
    except Exception:
        return None


def load_cpr_curtain(
    path: str | Path,
    height_top_km: float = 18.0,
    height_res_m: float = 100.0,
    dbz_floor: float = -30.0,
    clutter_buffer_km: float = 0.5,
    lat_range: tuple[float, float] | None = None,
) -> xr.Dataset:
    """Load a CPR_NOM_1B granule as a regridded reflectivity curtain.

    Parameters
    ----------
    path : str or Path
        Local ``.h5`` granule.
    height_top_km : float
        Top of the common height axis (km).
    height_res_m : float
        Vertical resolution of the common height axis (m).
    dbz_floor : float
        Reflectivities below this (dBZ) are set to NaN (below CPR sensitivity /
        clear air), so the curtain shows only meaningful echo.
    clutter_buffer_km : float
        Blank reflectivity within this height of the surface (per ray) to
        remove the CPR surface/ground-clutter return.
    lat_range : (lo, hi), optional
        Keep only rays with latitude in this range (crop the frame to the
        segment of interest).

    Returns
    -------
    xarray.Dataset with dims ``(along_track, height)``:
        ``reflectivity`` (dBZ), coords ``lat``/``lon``/``time``/``distance_km``
        (along_track) and ``height`` (km), plus ``surface_km`` (along_track).
    """
    import h5py

    with h5py.File(str(path), "r") as h:
        Z = h[_REFL][:].astype(np.float64)          # (nray, nbin) mm^6 m^-3
        binh = h[_BINH][:].astype(np.float64)        # (nray, nbin) m, TOA-first
        lat = h[_LAT][:].astype(np.float64)
        lon = h[_LON][:].astype(np.float64)
        surf = h[_SURF][:].astype(np.float64) if _SURF in h else np.zeros_like(lat)
        span = _read_sensing_times(h)

    # crop to latitude segment
    if lat_range is not None:
        lo, hi = sorted(lat_range)
        keep = (lat >= lo) & (lat <= hi)
        Z, binh, lat, lon, surf = Z[keep], binh[keep], lat[keep], lon[keep], surf[keep]

    nray = lat.size
    # linear reflectivity -> dBZ, masking fills and non-positive values
    Z[np.abs(Z) > 1e30] = np.nan
    Z[Z <= 0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        dbz = 10.0 * np.log10(Z)
    dbz[dbz < dbz_floor] = np.nan
    binh[np.abs(binh) > 1e30] = np.nan

    # common height axis (ascending, km)
    hgrid_m = np.arange(0.0, height_top_km * 1000.0 + 1e-6, height_res_m)
    nh = hgrid_m.size
    refl = np.full((nray, nh), np.nan, dtype=np.float32)
    for i in range(nray):
        hb = binh[i]
        zb = dbz[i]
        good = np.isfinite(hb)
        if good.sum() < 2:
            continue
        hh = hb[good]
        zz = zb[good]
        order = np.argsort(hh)          # ascending height for np.interp
        hh, zz = hh[order], zz[order]
        # interp only within the measured range; leave NaN outside
        refl[i] = np.interp(hgrid_m, hh, zz, left=np.nan, right=np.nan)

    # blank surface/ground clutter: everything within clutter_buffer_km of the
    # surface elevation (per ray) — over ocean this is the bright 0-0.5 km stripe
    if clutter_buffer_km > 0:
        clutter = hgrid_m[None, :] < (surf + clutter_buffer_km * 1000.0)[:, None]
        refl[clutter] = np.nan

    dist_km = _great_circle_km(lat, lon)

    # per-ray UTC time from a linear ramp between frame start/stop
    if span is not None and nray > 1:
        t0, t1 = span
        frac = np.linspace(0.0, 1.0, nray)
        times = t0 + (frac * (t1 - t0).astype("timedelta64[s]").astype(float)).astype("timedelta64[s]")
    else:
        times = np.full(nray, np.datetime64("NaT"), dtype="datetime64[s]")

    ds = xr.Dataset(
        {
            "reflectivity": (("along_track", "height"), refl),
            "surface_km": ("along_track", (surf / 1000.0).astype(np.float32)),
        },
        coords={
            "lat": ("along_track", lat.astype(np.float32)),
            "lon": ("along_track", lon.astype(np.float32)),
            "time": ("along_track", times),
            "distance_km": ("along_track", dist_km.astype(np.float32)),
            "height": ("height", (hgrid_m / 1000.0).astype(np.float32)),
        },
    )
    ds["reflectivity"].attrs = {"long_name": "CPR radar reflectivity factor",
                                "units": "dBZ", "source": "EarthCARE CPR_NOM_1B"}
    ds["height"].attrs = {"units": "km", "long_name": "height above ellipsoid"}
    ds.attrs["granule"] = Path(path).name
    if span is not None:
        ds.attrs["frame_start"] = str(span[0])
        ds.attrs["frame_stop"] = str(span[1])
    return ds
