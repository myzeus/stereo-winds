"""Geostationary projection math following GOES-R PUG Section 4.2.8.

Pure numpy implementation. All functions are vectorized over (H, W) arrays.
Handles both x-sweep (ABI) and y-sweep (AHI) instruments.
"""

from __future__ import annotations

import numpy as np

from .config import SatelliteConfig


def pixel_to_scanning_angle(
    col: np.ndarray, row: np.ndarray, sat: SatelliteConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pixel (col, row) to scanning angles (x, y) in radians."""
    x = col * sat.scale_x + sat.x_offset
    y = row * sat.scale_y + sat.y_offset
    return x, y


def scanning_angle_to_pixel(
    x: np.ndarray, y: np.ndarray, sat: SatelliteConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Convert scanning angles (x, y) in radians to pixel (col, row)."""
    col = (x - sat.x_offset) / sat.scale_x
    row = (y - sat.y_offset) / sat.scale_y
    return col, row


def fixed_grid_to_geodetic(
    x: np.ndarray, y: np.ndarray, sat: SatelliteConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse projection: scanning angles → geodetic (lat_deg, lon_deg).

    Returns NaN for off-Earth pixels (negative discriminant).
    """
    a = sat.semi_major_m
    b = sat.semi_minor_m
    H = sat.satellite_height_m + a  # distance from Earth center

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # For y-sweep (AHI), swap x and y before computing
    if sat.sweep == "y":
        x, y = y, x

    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)

    a2 = a * a
    b2 = b * b

    # Quadratic coefficients for ray–ellipsoid intersection
    a_coeff = sin_x**2 + cos_x**2 * (cos_y**2 + (a2 / b2) * sin_y**2)
    b_coeff = -2.0 * H * cos_x * cos_y
    c_coeff = H * H - a2

    discriminant = b_coeff**2 - 4.0 * a_coeff * c_coeff

    # Off-Earth pixels → NaN
    valid = discriminant >= 0
    disc_safe = np.where(valid, discriminant, 0.0)

    r_s = (-b_coeff - np.sqrt(disc_safe)) / (2.0 * a_coeff)

    sx = r_s * cos_x * cos_y
    sy = -r_s * sin_x
    sz = r_s * cos_x * sin_y

    lat_rad = np.arctan((a2 / b2) * sz / np.sqrt((H - sx) ** 2 + sy**2))
    lon_rad = np.radians(sat.sub_lon_deg) - np.arctan2(sy, H - sx)

    lat_deg = np.where(valid, np.degrees(lat_rad), np.nan)
    lon_deg = np.where(valid, np.degrees(lon_rad), np.nan)

    return lat_deg, lon_deg


def geodetic_to_ecef(
    lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: np.ndarray | float = 0.0,
    semi_major: float = 6378137.0, semi_minor: float = 6356752.31414,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic (lat, lon, height) to ECEF (X, Y, Z)."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    h = np.asarray(h_m, dtype=np.float64)

    a = semi_major
    b = semi_minor
    e2 = 1.0 - (b / a) ** 2  # first eccentricity squared

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat**2)  # prime vertical radius

    X = (N + h) * cos_lat * np.cos(lon)
    Y = (N + h) * cos_lat * np.sin(lon)
    Z = (N * (1.0 - e2) + h) * sin_lat

    return X, Y, Z


def geodetic_to_fixed_grid(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    sat: SatelliteConfig,
    h_m: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward projection: geodetic (lat, lon, h) → scanning angles (x, y).

    When h_m > 0, the point is elevated along the geodetic normal.
    Returns NaN for points not visible from the satellite.

    Uses the PUG s-vector convention where:
        s_x = H - p_x  (H is satellite distance from Earth center)
        s_y = -p_y
        s_z = p_z
    with (p_x, p_y, p_z) in the rotated frame (x along sub_lon).
    """
    a = sat.semi_major_m
    b = sat.semi_minor_m
    H = sat.satellite_height_m + a  # distance from Earth center

    # Compute ECEF position of the point (possibly elevated)
    px, py, pz = geodetic_to_ecef(lat_deg, lon_deg, h_m, a, b)

    # Rotate ECEF to satellite-centered frame (x-axis along sub_lon)
    sub_lon_rad = np.radians(sat.sub_lon_deg)
    cos_lon = np.cos(sub_lon_rad)
    sin_lon = np.sin(sub_lon_rad)

    p_x = cos_lon * px + sin_lon * py
    p_y = -sin_lon * px + cos_lon * py
    p_z = pz

    # PUG s-vector (consistent with inverse projection formulas)
    s_x = H - p_x
    s_y = -p_y
    s_z = p_z

    # Visibility check via surface-normal dot product:
    # Point is visible iff the outward normal at the surface point has a
    # positive dot product with the satellite direction.  For the ellipsoid
    # at h ≈ 0 this reduces exactly to  p_x > a² / H.
    visible = p_x * H > a * a

    # Extract scanning angles (x-sweep convention):
    #   s_x = r_s * cos(x) * cos(y)
    #   s_y = -r_s * sin(x)
    #   s_z = r_s * cos(x) * sin(y)
    x_angle = np.arctan2(-s_y, np.sqrt(s_x**2 + s_z**2))
    y_angle = np.arctan2(s_z, s_x)

    # For y-sweep (AHI), swap the angles
    if sat.sweep == "y":
        x_angle, y_angle = y_angle, x_angle

    x_angle = np.where(visible, x_angle, np.nan)
    y_angle = np.where(visible, y_angle, np.nan)

    return x_angle, y_angle


def compute_zenith_angle(
    lat_deg: np.ndarray, lon_deg: np.ndarray, sat: SatelliteConfig
) -> np.ndarray:
    """Compute satellite zenith angle in degrees for given geodetic points."""
    a = sat.semi_major_m
    b = sat.semi_minor_m
    H = sat.satellite_height_m + a

    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))

    # Geocentric latitude
    lat_gc = np.arctan((b / a) ** 2 * np.tan(lat))
    cos_dlon = np.cos(lon - np.radians(sat.sub_lon_deg))

    # Distance from Earth center at the geocentric latitude
    r_earth = a / np.sqrt(
        1.0 + ((a / b) ** 2 - 1.0) * np.sin(lat_gc) ** 2
    )

    # Zenith angle via the law of cosines
    r_sat = H
    cos_gamma = np.cos(lat_gc) * cos_dlon  # angle subtended at Earth center
    d2 = r_earth**2 + r_sat**2 - 2.0 * r_earth * r_sat * cos_gamma
    d = np.sqrt(np.maximum(d2, 0.0))

    # Sine of zenith angle via sine rule
    sin_zenith = (r_sat / d) * np.sqrt(1.0 - cos_gamma**2)
    zenith = np.degrees(np.arcsin(np.clip(sin_zenith, -1.0, 1.0)))

    return zenith


_pixel_scale_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def compute_pixel_scale(
    sat: SatelliteConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pixel ground distances for the full fixed grid.

    Each pixel on the geostationary grid subtends the same scanning angle
    increment, but the corresponding ground distance varies with viewing
    geometry. At nadir the pixel footprint is ~2 km (for 2 km bands), but
    it grows toward the limb.

    Method: evaluate the geostationary projection at half-pixel offsets in
    each direction, convert to ECEF, and compute the 3-D distance.

    Results are cached by satellite_id since the grid geometry is fixed.

    Parameters
    ----------
    sat : SatelliteConfig with grid dimensions and projection parameters

    Returns
    -------
    dx_m : (n_rows, n_cols) east-west ground distance per pixel (meters)
    dy_m : (n_rows, n_cols) north-south ground distance per pixel (meters)
    NaN for off-Earth pixels.
    """
    cache_key = f"{sat.satellite_id}_{sat.n_rows}_{sat.n_cols}_{sat.scale_x}"
    if cache_key in _pixel_scale_cache:
        return _pixel_scale_cache[cache_key]
    cols = np.arange(sat.n_cols, dtype=np.float64)
    rows = np.arange(sat.n_rows, dtype=np.float64)
    col2d, row2d = np.meshgrid(cols, rows)

    x_center = col2d * sat.scale_x + sat.x_offset
    y_center = row2d * sat.scale_y + sat.y_offset

    half_sx = sat.scale_x / 2.0
    half_sy = sat.scale_y / 2.0

    a = sat.semi_major_m
    b = sat.semi_minor_m

    # --- dx: east-west ground distance per pixel ---
    lat_l, lon_l = fixed_grid_to_geodetic(x_center - half_sx, y_center, sat)
    lat_r, lon_r = fixed_grid_to_geodetic(x_center + half_sx, y_center, sat)
    xl, yl, zl = geodetic_to_ecef(lat_l, lon_l, 0.0, a, b)
    xr, yr, zr = geodetic_to_ecef(lat_r, lon_r, 0.0, a, b)
    dx_m = np.sqrt((xr - xl)**2 + (yr - yl)**2 + (zr - zl)**2)

    # --- dy: north-south ground distance per pixel ---
    lat_t, lon_t = fixed_grid_to_geodetic(x_center, y_center - half_sy, sat)
    lat_b, lon_b = fixed_grid_to_geodetic(x_center, y_center + half_sy, sat)
    xt, yt, zt = geodetic_to_ecef(lat_t, lon_t, 0.0, a, b)
    xb, yb, zb = geodetic_to_ecef(lat_b, lon_b, 0.0, a, b)
    dy_m = np.sqrt((xb - xt)**2 + (yb - yt)**2 + (zb - zt)**2)

    _pixel_scale_cache[cache_key] = (dx_m, dy_m)
    return dx_m, dy_m
