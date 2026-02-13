"""Cross-satellite image remapping via look-up tables.

Builds a LUT that maps each pixel in satellite A's grid to the corresponding
pixel in satellite B's grid, by inverse-projecting through (lat, lon) at h=0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from .config import SatelliteConfig
from .navigation import (
    compute_zenith_angle,
    fixed_grid_to_geodetic,
    geodetic_to_fixed_grid,
    pixel_to_scanning_angle,
    scanning_angle_to_pixel,
)


def build_remap_lut(
    sat_a: SatelliteConfig, sat_b: SatelliteConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Build a look-up table mapping A's pixel grid into B's pixel grid.

    For each pixel (col_a, row_a) in A's fixed grid:
      1. Convert to scanning angles
      2. Inverse-project to geodetic (lat, lon) at h = 0
      3. Forward-project into B's scanning angles
      4. Convert to continuous (col_b, row_b)

    Returns
    -------
    col_b, row_b : ndarray of shape (n_rows_a, n_cols_a)
        Continuous pixel coordinates in B's grid.  NaN where B cannot see
        the point (off-limb or behind Earth).
    """
    rows_a = np.arange(sat_a.n_rows)
    cols_a = np.arange(sat_a.n_cols)
    col_grid, row_grid = np.meshgrid(cols_a, rows_a)

    # A pixels → scanning angles → geodetic
    x_a, y_a = pixel_to_scanning_angle(col_grid, row_grid, sat_a)
    lat, lon = fixed_grid_to_geodetic(x_a, y_a, sat_a)

    # Geodetic → B scanning angles → B pixels
    x_b, y_b = geodetic_to_fixed_grid(lat, lon, sat_b, h_m=0.0)
    col_b, row_b = scanning_angle_to_pixel(x_b, y_b, sat_b)

    return col_b, row_b


def save_remap_lut(
    col_b: np.ndarray, row_b: np.ndarray, path: str | Path
) -> None:
    """Cache a remap LUT to disk as compressed npz."""
    np.savez_compressed(str(path), col_b=col_b, row_b=row_b)


def load_remap_lut(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a cached remap LUT from disk."""
    data = np.load(str(path))
    return data["col_b"], data["row_b"]


def remap_image(
    image_b: np.ndarray,
    col_b: np.ndarray,
    row_b: np.ndarray,
    order: int = 1,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Remap an image from B's native grid onto A's grid using a LUT.

    Uses bilinear interpolation (order=1) by default.

    Parameters
    ----------
    image_b : (n_rows_b, n_cols_b) array
        Image in B's native grid.
    col_b, row_b : (n_rows_a, n_cols_a) arrays
        LUT mapping A's pixels into B's pixel coordinates.
    order : int
        Interpolation order (0=nearest, 1=bilinear).
    fill_value : float
        Value for out-of-bounds or NaN LUT entries.
    """
    # Mask invalid LUT entries
    valid = np.isfinite(col_b) & np.isfinite(row_b)
    in_bounds = (
        valid
        & (col_b >= 0)
        & (col_b <= image_b.shape[1] - 1)
        & (row_b >= 0)
        & (row_b <= image_b.shape[0] - 1)
    )

    # map_coordinates expects (row, col) coordinate arrays
    coords = np.array(
        [
            np.where(in_bounds, row_b, 0),
            np.where(in_bounds, col_b, 0),
        ]
    )

    remapped = map_coordinates(
        image_b.astype(np.float64),
        coords,
        order=order,
        mode="constant",
        cval=fill_value,
    )

    remapped[~in_bounds] = fill_value
    return remapped


def compute_valid_mask(
    col_b: np.ndarray,
    row_b: np.ndarray,
    sat_a: SatelliteConfig,
    sat_b: SatelliteConfig,
    max_zenith: float = 70.0,
) -> np.ndarray:
    """Mask pixels where either satellite's zenith angle exceeds threshold.

    Returns a boolean array (True = valid) of shape (n_rows_a, n_cols_a).
    """
    rows_a = np.arange(sat_a.n_rows)
    cols_a = np.arange(sat_a.n_cols)
    col_grid, row_grid = np.meshgrid(cols_a, rows_a)

    x_a, y_a = pixel_to_scanning_angle(col_grid, row_grid, sat_a)
    lat, lon = fixed_grid_to_geodetic(x_a, y_a, sat_a)

    valid_earth = np.isfinite(lat)
    zen_a = np.full_like(lat, 90.0)
    zen_b = np.full_like(lat, 90.0)
    zen_a[valid_earth] = compute_zenith_angle(lat[valid_earth], lon[valid_earth], sat_a)
    zen_b[valid_earth] = compute_zenith_angle(lat[valid_earth], lon[valid_earth], sat_b)

    # Also require valid LUT entries
    lut_valid = np.isfinite(col_b) & np.isfinite(row_b)

    return lut_valid & (zen_a < max_zenith) & (zen_b < max_zenith)
