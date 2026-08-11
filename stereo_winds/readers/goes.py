"""Minimal public-S3 GOES ABI L1b reader (no satpy, no auth).

Reads GOES-16/17/18/19 ABI Level-1b radiance from NOAA's public S3 buckets
(``noaa-goes16`` ...) and returns a dataset shaped exactly like the internal
satpy-based reader stereo-winds was built against: ``Rad`` as ``(time, band,
y, x)`` oriented south->north, ``x``/``y`` in **meters** (scan angle x
perspective height), and ``Rad.attrs["orbital_parameters"]`` carrying
``projection_altitude`` and ``satellite_nominal_longitude``.

This is the standalone replacement for the zeus GOES data source. Only the
native fixed-grid, single-band path used by the stereo solver is implemented
(no reprojection / HEALPix / multi-band resampling).

Requires ``s3fs`` (and an xarray NetCDF backend: ``h5netcdf`` or ``netCDF4``).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

PRODUCT_TIMESTEPS = {"ABI-L1b-RadF": 10, "ABI-L1b-RadC": 5, "ABI-L1b-RadM": 1}
_GNUM = {"goes16": "16", "goes17": "17", "goes18": "18", "goes19": "19"}


def _abi_to_satpy_like(raw: xr.Dataset) -> xr.Dataset:
    """Reshape a raw ABI L1b NetCDF into the satpy-style dataset the solver
    pipeline consumes (see module docstring for the exact contract)."""
    proj = raw["goes_imager_projection"].attrs
    height = float(proj["perspective_point_height"])
    sub_lon = float(proj["longitude_of_projection_origin"])

    x_rad = raw["x"].values.astype(np.float64)          # west->east, increasing
    y_rad = raw["y"].values.astype(np.float64)          # north->south, decreasing
    rad = np.asarray(raw["Rad"].values, dtype=np.float32)  # (y, x), row 0 = north

    # Flip to south->north (y ascending) to match the satpy convention.
    y_rad_asc = y_rad[::-1]
    rad_sn = rad[::-1, :]

    x_m = x_rad * height
    y_m = y_rad_asc * height

    Rad = xr.DataArray(
        rad_sn[None, None, :, :],
        dims=("time", "band", "y", "x"),
        coords={"x": ("x", x_m), "y": ("y", y_m)},
        name="Rad",
    )
    Rad.attrs["orbital_parameters"] = {
        "projection_altitude": height,
        "satellite_nominal_longitude": sub_lon,
        "projection_longitude": sub_lon,
    }
    for k in ("time_coverage_start", "time_coverage_end"):
        if k in raw.attrs:
            Rad.attrs[k] = raw.attrs[k]
    ds = xr.Dataset({"Rad": Rad})
    ds.attrs["sweep_angle_axis"] = proj.get("sweep_angle_axis", "x")
    return ds


class GOES:
    """Public-S3 GOES ABI L1b reader (drop-in for the zeus GOES source).

    Parameters
    ----------
    satellite : "goes16" | "goes17" | "goes18" | "goes19"
    product : "ABI-L1b-RadF" (full disk), "ABI-L1b-RadC" (CONUS), "ABI-L1b-RadM"
    bands : list with a single ABI band, e.g. ``["C14"]``
    cache_dir : local download cache (default ~/.cache/stereo_winds)
    """

    def __init__(self, satellite="goes16", product="ABI-L1b-RadF",
                 bands=None, cache_dir=None):
        self.satellite = satellite
        self.product = product
        self.bands = list(bands) if bands else ["C13"]
        self.cache_dir = Path(cache_dir) if cache_dir else (
            Path.home() / ".cache" / "stereo_winds")
        self.bucket = f"noaa-{satellite}"
        self.step = PRODUCT_TIMESTEPS.get(product, 10)
        self._fs = None

    @property
    def fs(self):
        if self._fs is None:
            import s3fs
            self._fs = s3fs.S3FileSystem(anon=True)
        return self._fs

    def _snap_time(self, t: dt.datetime) -> dt.datetime:
        return t.replace(minute=(t.minute // self.step) * self.step,
                         second=0, microsecond=0)

    def _find_key(self, t: dt.datetime, band: str) -> str:
        t = self._snap_time(t)
        gnum = _GNUM[self.satellite]
        # RadC/RadM scans start +1 minute past the nominal slot.
        ts = t + dt.timedelta(minutes=1) if self.product != "ABI-L1b-RadF" else t
        prefix = f"{self.bucket}/{self.product}/{t:%Y/%j/%H}"
        pattern = f"{prefix}/OR_{self.product}-M*{band}_G{gnum}_s{ts:%Y%j%H%M}*"
        matches = self.fs.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No ABI file on S3: {pattern}")
        return matches[0]

    def data_at_time(self, t: dt.datetime, download: bool = True, **_) -> xr.Dataset:
        """Return the ABI scene at (snapped) time ``t`` for ``self.bands[0]``."""
        band = self.bands[0]
        key = self._find_key(t, band)
        if download:
            local = self.cache_dir / self.satellite / Path(key).name
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                self.fs.get(key, str(local))
            raw = xr.open_dataset(local)
        else:
            raw = xr.open_dataset(self.fs.open(key))
        try:
            return _abi_to_satpy_like(raw)
        finally:
            raw.close()

    def __repr__(self):
        return (f"GOES(satellite={self.satellite!r}, product={self.product!r}, "
                f"bands={self.bands!r})")
