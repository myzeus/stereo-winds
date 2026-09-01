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

# ATLID L1b (ATL_NOM_1B): dataset names vary slightly between processor
# baselines, so the reader probes the file tree for the first match instead of
# hardcoding a path.
_ATL_PATTERNS = {
    "mie": ("mie_attenuated_backscatter", "mie_attenuated_bsc"),
    "rayleigh": ("rayleigh_attenuated_backscatter", "rayleigh_attenuated_bsc"),
    "crosspolar": ("crosspolar_attenuated_backscatter", "cross_polar"),
    "height": ("sample_altitude", "ellipsoid_height", "height"),
    "lat": ("ellipsoid_latitude", "sensor_latitude", "latitude"),
    "lon": ("ellipsoid_longitude", "sensor_longitude", "longitude"),
}
# Ancillary fields sit beside the real ones under the same stem ("..._random_error",
# "mie_relative_backscatter"); never match those, nor the scalar header coordinates.
_ATL_EXCLUDE = ("_error", "relative_backscatter", "proportionality",
                "headerdata", "_raw")


def _great_circle_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cumulative along-track great-circle distance (km), starting at 0."""
    lat0, lon0 = np.radians(lat[:-1]), np.radians(lon[:-1])
    lat1, lon1 = np.radians(lat[1:]), np.radians(lon[1:])
    dlat = lat1 - lat0
    dlon = lon1 - lon0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    seg = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _parse_iso(raw) -> np.datetime64:
    """Parse an EarthCARE header timestamp.

    The products write ``UTC=2025-10-10T08:53:26`` -- the ``UTC=`` prefix has to
    come off before the ISO parse, otherwise the whole read silently fails and
    every ray ends up NaT.
    """
    s = raw.decode() if isinstance(raw, bytes) else str(raw)
    s = s.strip()
    if "=" in s[:8]:
        s = s.split("=", 1)[1]
    s = s.rstrip("Z").strip()
    return np.datetime64(s[:26])


def _read_sensing_times(h) -> tuple[np.datetime64, np.datetime64] | None:
    """Frame sensing start/stop from HeaderData."""
    base = "HeaderData/VariableProductHeader/MainProductHeader"
    try:
        return (_parse_iso(h[f"{base}/sensingStartTime"][()]),
                _parse_iso(h[f"{base}/sensingStopTime"][()]))
    except Exception:
        return None


def _read_per_ray_times(h, nray: int) -> np.ndarray | None:
    """True per-ray UTC times from a ``seconds since <epoch>`` dataset.

    Preferred over interpolating between frame start/stop: the figure keys a
    GOES scan time off the overpass time of a *cropped* segment, so a per-ray
    time is what makes that match exact.
    """
    import re

    import h5py

    cands: list[str] = []

    def visit(name, obj):
        if (isinstance(obj, h5py.Dataset) and obj.shape == (nray,)
                and "time" in name.lower() and "flag" not in name.lower()):
            cands.append(name)

    h.visititems(visit)
    for name in sorted(cands, key=len):
        units = h[name].attrs.get("units", b"")
        units = units.decode() if isinstance(units, bytes) else str(units)
        m = re.search(r"seconds since\s+(\d{4})-(\d{1,2})-(\d{1,2})", units)
        if not m:
            continue
        y, mo, d = (int(g) for g in m.groups())
        epoch = np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}", "us")
        secs = np.asarray(h[name][:], dtype=np.float64)
        secs[np.abs(secs) > 1e30] = np.nan
        if not np.isfinite(secs).any():
            continue
        out = np.full(nray, np.datetime64("NaT"), dtype="datetime64[us]")
        ok = np.isfinite(secs)
        out[ok] = epoch + (secs[ok] * 1e6).astype("timedelta64[us]")
        return out
    return None


def _ray_times(h, nray: int, span) -> np.ndarray:
    """Per-ray times: real dataset if present, else a frame start/stop ramp."""
    t = _read_per_ray_times(h, nray)
    if t is not None:
        return t
    if span is not None and nray > 1:
        t0, t1 = span
        frac = np.linspace(0.0, 1.0, nray)
        dur = (t1 - t0) / np.timedelta64(1, "s")
        return t0.astype("datetime64[us]") + (frac * dur * 1e6).astype("timedelta64[us]")
    return np.full(nray, np.datetime64("NaT"), dtype="datetime64[us]")


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
        times_all = _ray_times(h, lat.size, span)

    # crop to latitude segment
    if lat_range is not None:
        lo, hi = sorted(lat_range)
        keep = (lat >= lo) & (lat <= hi)
        Z, binh, lat, lon, surf = Z[keep], binh[keep], lat[keep], lon[keep], surf[keep]
        times_all = times_all[keep]

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
    times = times_all

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


# ---------------------------------------------------------------------------
# ATLID (lidar) curtain
# ---------------------------------------------------------------------------
def h5_tree(path: str | Path, max_items: int = 400) -> list[str]:
    """List `name  shape  dtype` for every dataset in an HDF5 file.

    Used to pin down product layout when a new EarthCARE product type is
    introduced (names drift between processor baselines).
    """
    import h5py

    out: list[str] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and len(out) < max_items:
            out.append(f"{name}  {obj.shape}  {obj.dtype}")

    with h5py.File(str(path), "r") as h:
        h.visititems(visit)
    return out


def _find_dataset(h, patterns, ndim=None):
    """Best dataset whose path matches `patterns`, honouring rank and shape.

    Ranked by pattern order first (so a specific name like ``ellipsoid_latitude``
    beats the generic ``latitude``), then by path depth/length, so the plain
    field wins over any longer ancillary sharing its stem.  `ndim` restricts to
    arrays of that rank -- without it a scalar header coordinate can shadow the
    real per-ray field.
    """
    import h5py

    allowed = None if ndim is None else set(np.atleast_1d(ndim).tolist())
    hits: list[tuple[int, int, str]] = []

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        low = name.lower()
        if any(x in low for x in _ATL_EXCLUDE):
            return
        if allowed is not None and obj.ndim not in allowed:
            return
        for rank, pat in enumerate(patterns):
            if pat.lower() in low:
                hits.append((rank, len(name), name))
                return

    h.visititems(visit)
    if not hits:
        return None
    hits.sort()
    return hits[0][2]


def load_atlid_curtain(
    path: str | Path,
    height_top_km: float = 18.0,
    height_res_m: float = 100.0,
    channel: str = "mie",
    lat_range: tuple[float, float] | None = None,
    smooth_rays: int = 5,
) -> xr.Dataset:
    """Load an ATLID ``ATL_NOM_1B`` granule as an attenuated-backscatter curtain.

    ATLID is the reason a semi-transparent cirrus layer is visible at all: the
    355 nm lidar detects tenuous ice that the 94 GHz radar underestimates or
    misses entirely, so the pair of curtains shows the *same* cloud as
    optically thin (lidar-bright, radar-weak) versus opaque (radar-strong,
    lidar-extinguished).

    Parameters
    ----------
    channel : {"mie", "rayleigh", "crosspolar"}
        Which attenuated-backscatter channel to return as ``backscatter``.
    smooth_rays : int
        Boxcar-average this many along-track rays to lift the tenuous-cirrus
        signal out of shot noise (ATLID single-shot SNR is low for thin ice).

    Returns
    -------
    xarray.Dataset with dims ``(along_track, height)``: ``backscatter``
    (m^-1 sr^-1), coords ``lat``/``lon``/``distance_km``/``height`` (km).
    """
    import h5py

    with h5py.File(str(path), "r") as h:
        k_bsc = _find_dataset(h, _ATL_PATTERNS[channel], ndim=2)
        k_hgt = _find_dataset(h, _ATL_PATTERNS["height"], ndim=[1, 2])
        k_lat = _find_dataset(h, _ATL_PATTERNS["lat"], ndim=1)
        k_lon = _find_dataset(h, _ATL_PATTERNS["lon"], ndim=1)
        if not all([k_bsc, k_hgt, k_lat, k_lon]):
            raise KeyError(
                f"ATLID layout not recognised in {Path(path).name}; found "
                f"bsc={k_bsc} hgt={k_hgt} lat={k_lat} lon={k_lon}. "
                "Use h5_tree() to inspect.")
        bsc = h[k_bsc][:].astype(np.float64)
        hgt = h[k_hgt][:].astype(np.float64)
        lat = h[k_lat][:].astype(np.float64).ravel()
        lon = h[k_lon][:].astype(np.float64).ravel()
        span = _read_sensing_times(h)
        times_all = _ray_times(h, lat.size, span)

    bsc[np.abs(bsc) > 1e30] = np.nan
    hgt[np.abs(hgt) > 1e30] = np.nan
    # height may be a shared 1-D axis or a per-ray 2-D grid
    if hgt.ndim == 1:
        hgt = np.broadcast_to(hgt, bsc.shape).copy()

    if lat_range is not None:
        lo, hi = sorted(lat_range)
        keep = (lat >= lo) & (lat <= hi)
        bsc, hgt, lat, lon = bsc[keep], hgt[keep], lat[keep], lon[keep]
        times_all = times_all[keep]

    nray = lat.size
    hgrid_m = np.arange(0.0, height_top_km * 1000.0 + 1e-6, height_res_m)
    out = np.full((nray, hgrid_m.size), np.nan, dtype=np.float32)
    for i in range(nray):
        hh, bb = hgt[i], bsc[i]
        good = np.isfinite(hh)
        if good.sum() < 2:
            continue
        hh, bb = hh[good], bb[good]
        order = np.argsort(hh)
        out[i] = np.interp(hgrid_m, hh[order], bb[order], left=np.nan, right=np.nan)

    if smooth_rays and smooth_rays > 1:
        k = int(smooth_rays)
        pad = np.pad(out, ((k // 2, k // 2), (0, 0)), mode="edge")
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        for j in range(k):
            seg = pad[j:j + nray]
            m = np.isfinite(seg)
            acc[m] += seg[m]
            cnt[m] += 1
        out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

    dist_km = _great_circle_km(lat, lon)
    times = times_all

    ds = xr.Dataset(
        {"backscatter": (("along_track", "height"), out)},
        coords={
            "lat": ("along_track", lat.astype(np.float32)),
            "lon": ("along_track", lon.astype(np.float32)),
            "time": ("along_track", times),
            "distance_km": ("along_track", dist_km.astype(np.float32)),
            "height": ("height", (hgrid_m / 1000.0).astype(np.float32)),
        },
    )
    ds["backscatter"].attrs = {
        "long_name": f"ATLID {channel} attenuated backscatter",
        "units": "m-1 sr-1", "source": "EarthCARE ATL_NOM_1B"}
    ds.attrs["granule"] = Path(path).name
    if span is not None:
        ds.attrs["frame_start"] = str(span[0])
        ds.attrs["frame_stop"] = str(span[1])
    return ds
