"""NetCDF/zarr output formatting for stereo wind retrievals.

Produces CF-1.8 compliant datasets with grid_mapping metadata
for the geostationary fixed-grid projection.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from .config import SatelliteConfig, StereoPairConfig
from .navigation import pixel_to_scanning_angle


def create_output_dataset(
    solution: dict[str, np.ndarray],
    sat_config: SatelliteConfig,
    t0: datetime,
    pair_config: StereoPairConfig | None = None,
) -> xr.Dataset:
    """Create an xarray Dataset from solver output.

    Parameters
    ----------
    solution : dict from solve_stereo_winds()
    sat_config : reference satellite configuration (A)
    t0 : center time of the retrieval
    pair_config : optional pipeline configuration for metadata
    """
    n_rows, n_cols = sat_config.n_rows, sat_config.n_cols

    # Coordinate arrays: scanning angles in radians
    cols = np.arange(n_cols)
    rows = np.arange(n_rows)
    x_rad = cols * sat_config.scale_x + sat_config.x_offset
    y_rad = rows * sat_config.scale_y + sat_config.y_offset

    # Grid mapping variable
    grid_mapping_attrs = {
        "grid_mapping_name": "geostationary",
        "longitude_of_projection_origin": sat_config.sub_lon_deg,
        "perspective_point_height": sat_config.satellite_height_m,
        "semi_major_axis": sat_config.semi_major_m,
        "semi_minor_axis": sat_config.semi_minor_m,
        "sweep_angle_axis": sat_config.sweep,
        "latitude_of_projection_origin": 0.0,
    }

    encoding_opts = {"zlib": True, "complevel": 4, "dtype": "float32"}

    def _make_var(data, long_name, units, standard_name=None):
        attrs = {"long_name": long_name, "units": units, "grid_mapping": "goes_imager_projection"}
        if standard_name:
            attrs["standard_name"] = standard_name
        return xr.Variable(["y", "x"], data.astype(np.float32), attrs=attrs)

    data_vars = {
        "u_wind": _make_var(
            solution.get("u_wind", solution.get("V_u", np.zeros((n_rows, n_cols)))),
            "Eastward Wind Component", "m s-1", "eastward_wind",
        ),
        "v_wind": _make_var(
            solution.get("v_wind", solution.get("V_v", np.zeros((n_rows, n_cols)))),
            "Northward Wind Component", "m s-1", "northward_wind",
        ),
        "cloud_top_height": _make_var(
            solution.get("h", np.zeros((n_rows, n_cols))),
            # The solver returns the height of the tracked radiative feature,
            # which sits below the geometric cloud top by an amount that grows
            # with the tracer's transparency, hence the long_name. The CF
            # standard_name stays cloud_top_altitude deliberately: CF has no
            # term for a feature-tracked height, and downstream tooling keys
            # off this one.
            "Feature-Tracked Height", "m", "cloud_top_altitude",
        ),
        "sigma_u": _make_var(
            solution.get("sigma_u", np.zeros((n_rows, n_cols))),
            "Formal Uncertainty in Eastward Wind", "m s-1",
        ),
        "sigma_v": _make_var(
            solution.get("sigma_v", np.zeros((n_rows, n_cols))),
            "Formal Uncertainty in Northward Wind", "m s-1",
        ),
        "sigma_h": _make_var(
            solution.get("sigma_h", np.zeros((n_rows, n_cols))),
            "Formal Uncertainty in Feature-Tracked Height", "m",
        ),
        "chi_squared": _make_var(
            solution.get("chi2", np.zeros((n_rows, n_cols))),
            "Chi-Squared Residual", "1",
        ),
        "position_correction_u": _make_var(
            solution.get("p_u", np.zeros((n_rows, n_cols))),
            "Position Correction (E-W)", "pixel",
        ),
        "position_correction_v": _make_var(
            solution.get("p_v", np.zeros((n_rows, n_cols))),
            "Position Correction (N-S)", "pixel",
        ),
        "quality_flag": xr.Variable(
            ["y", "x"],
            solution.get("quality_flag", np.zeros((n_rows, n_cols), dtype=np.float32)).astype(np.float32),
            attrs={
                "long_name": "Quality Assurance Flag",
                "units": "1",
                "flag_values": np.array([0, 1, 2], dtype=np.float32),
                "flag_meanings": "no_retrieval low_quality high_quality",
                "valid_range": np.array([0, 2], dtype=np.float32),
                "grid_mapping": "goes_imager_projection",
                "comment": "Higher values = stricter QC. Level 1 for general use; 2 for validation.",
            },
        ),
        "goes_imager_projection": xr.Variable(
            [], np.int32(0), attrs=grid_mapping_attrs,
        ),
    }

    coords = {
        "x": xr.Variable(
            ["x"], x_rad,
            attrs={
                "long_name": "GOES fixed grid projection x-coordinate",
                "units": "rad",
                "axis": "X",
            },
        ),
        "y": xr.Variable(
            ["y"], y_rad,
            attrs={
                "long_name": "GOES fixed grid projection y-coordinate",
                "units": "rad",
                "axis": "Y",
            },
        ),
        "time": xr.Variable(
            [], np.datetime64(t0),
            attrs={"long_name": "reference time", "axis": "T"},
        ),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs = {
        "Conventions": "CF-1.8",
        "title": "Stereo Wind Retrieval",
        "institution": "stereo-winds",
        "source": f"Cross-satellite stereo retrieval ({sat_config.satellite_id})",
        "history": f"Created {datetime.now().isoformat()}",
    }

    if pair_config is not None:
        ds.attrs["satellite_a"] = pair_config.sat_a.satellite_id
        ds.attrs["satellite_b"] = pair_config.sat_b.satellite_id
        ds.attrs["band"] = pair_config.band
        ds.attrs["dt_minutes"] = pair_config.dt_minutes
        ds.attrs["product"] = pair_config.product

    return ds


def write_netcdf(ds: xr.Dataset, path: str | Path) -> None:
    """Write dataset to CF-compliant NetCDF4 with compression."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    encoding = {}
    for var in ds.data_vars:
        if ds[var].dtype in (np.float32, np.float64):
            encoding[var] = {"zlib": True, "complevel": 4}

    try:
        ds.to_netcdf(str(path), engine="h5netcdf", encoding=encoding)
    except Exception:
        ds.to_netcdf(str(path), engine="netcdf4", encoding=encoding)


def write_zarr(ds: xr.Dataset, path: str | Path) -> None:
    """Write dataset to zarr store."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(str(path), mode="w")
