"""Five-state geometric solver for stereo wind retrieval.

Implements Carr et al. 2020 Equation 1:
    delta(t_n) = h * w_hat_n + p_vec + V_vec * (t_n - t_0)

State vector x = [h, p_u, p_v, V_u, V_v]  (5 unknowns)

Design matrix H (8x5) per pixel constructed from:
  - Parallax vectors w_hat for cross-satellite scenes
  - Time offsets dt for each scene
  - 4 pairs x 2 components = 8 observations
"""

from __future__ import annotations

import logging

import numpy as np

from .config import SatelliteConfig
from .navigation import (
    fixed_grid_to_geodetic,
    geodetic_to_fixed_grid,
    pixel_to_scanning_angle,
    scanning_angle_to_pixel,
)

logger = logging.getLogger(__name__)


def compute_parallax_vectors(
    sat_a: SatelliteConfig,
    sat_b: SatelliteConfig,
    dh: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pixel parallax sensitivity vectors.

    For each pixel in A's grid, computes the displacement (in A's pixel
    coordinates) caused by a height change `dh` as seen from B relative to A.

    Parameters
    ----------
    sat_a, sat_b : SatelliteConfig
    dh : height perturbation in meters for finite differences

    Returns
    -------
    w_u, w_v : (n_rows, n_cols) arrays
        Parallax sensitivity in pixels per meter of height.
    """
    rows = np.arange(sat_a.n_rows)
    cols = np.arange(sat_a.n_cols)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # Inverse-project A's pixels to (lat, lon) at h = 0
    x_a, y_a = pixel_to_scanning_angle(col_grid, row_grid, sat_a)
    lat, lon = fixed_grid_to_geodetic(x_a, y_a, sat_a)

    # Project (lat, lon, 0) and (lat, lon, dh) into B's grid
    xb_0, yb_0 = geodetic_to_fixed_grid(lat, lon, sat_b, h_m=0.0)
    xb_h, yb_h = geodetic_to_fixed_grid(lat, lon, sat_b, h_m=dh)

    # Convert B scanning angles to A pixel coordinates via the remap LUT
    # Actually, we want the displacement in A's pixel space.
    # Project into A's grid at h=0 and h=dh, then difference with B.
    xa_0, ya_0 = geodetic_to_fixed_grid(lat, lon, sat_a, h_m=0.0)
    xa_h, ya_h = geodetic_to_fixed_grid(lat, lon, sat_a, h_m=dh)

    # Parallax in scanning angles: displacement in B minus displacement in A
    # (what the solver sees as additional displacement from height)
    par_x = (xb_h - xb_0) - (xa_h - xa_0)
    par_y = (yb_h - yb_0) - (ya_h - ya_0)

    # Convert from scanning angle difference to pixel difference
    # Δpixel = Δangle / scale
    w_u = par_x / sat_a.scale_x / dh  # pixels per meter
    w_v = par_y / sat_a.scale_y / dh  # pixels per meter

    return w_u, w_v


def compute_parallax_vectors_at_h(
    sat_a: SatelliteConfig,
    sat_b: SatelliteConfig,
    h_m: np.ndarray | float,
    dh: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pixel parallax sensitivity vectors evaluated at height h_m.

    Same as ``compute_parallax_vectors`` but the finite-difference is taken
    around *h_m* instead of h=0, capturing the nonlinear dependence of
    parallax geometry on cloud-top height.

    Parameters
    ----------
    sat_a, sat_b : SatelliteConfig
    h_m : (n_rows, n_cols) array or scalar
        Height in meters at which to evaluate the parallax.
    dh : float
        Height perturbation in meters for finite differences.

    Returns
    -------
    w_u, w_v : (n_rows, n_cols) arrays
        Parallax sensitivity in pixels per meter of height.
    """
    rows = np.arange(sat_a.n_rows)
    cols = np.arange(sat_a.n_cols)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # Inverse-project A's pixels to (lat, lon) at h = 0
    # (the pixel grid is defined at h=0; we evaluate parallax at h_m)
    x_a, y_a = pixel_to_scanning_angle(col_grid, row_grid, sat_a)
    lat, lon = fixed_grid_to_geodetic(x_a, y_a, sat_a)

    h_m = np.broadcast_to(np.asarray(h_m, dtype=np.float64),
                           (sat_a.n_rows, sat_a.n_cols))

    # Project (lat, lon) at h_m and h_m+dh into both satellites' grids
    xb_lo, yb_lo = geodetic_to_fixed_grid(lat, lon, sat_b, h_m=h_m)
    xb_hi, yb_hi = geodetic_to_fixed_grid(lat, lon, sat_b, h_m=h_m + dh)

    xa_lo, ya_lo = geodetic_to_fixed_grid(lat, lon, sat_a, h_m=h_m)
    xa_hi, ya_hi = geodetic_to_fixed_grid(lat, lon, sat_a, h_m=h_m + dh)

    # Differential parallax at height h_m
    par_x = (xb_hi - xb_lo) - (xa_hi - xa_lo)
    par_y = (yb_hi - yb_lo) - (ya_hi - ya_lo)

    w_u = par_x / sat_a.scale_x / dh
    w_v = par_y / sat_a.scale_y / dh

    return w_u, w_v


def build_design_matrix(
    w_u: np.ndarray,
    w_v: np.ndarray,
    dt_a_minus: float | np.ndarray,
    dt_a_plus: float | np.ndarray,
    dt_b_minus: float | np.ndarray,
    dt_b_plus: float | np.ndarray,
) -> np.ndarray:
    """Build the (H, W, 8, 5) design matrix for the 5-state solver.

    Rows of H per pixel:
      0: A_minus u  →  [0,      1, 0, dt_am, 0    ]
      1: A_minus v  →  [0,      0, 1, 0,     dt_am]
      2: A_plus  u  →  [0,      1, 0, dt_ap, 0    ]
      3: A_plus  v  →  [0,      0, 1, 0,     dt_ap]
      4: B_minus u  →  [w_u,    1, 0, dt_bm, 0    ]
      5: B_minus v  →  [w_v,    0, 1, 0,     dt_bm]
      6: B_plus  u  →  [w_u,    1, 0, dt_bp, 0    ]
      7: B_plus  v  →  [w_v,    0, 1, 0,     dt_bp]

    State vector: [h, p_u, p_v, V_u, V_v]
    """
    H_px, W_px = w_u.shape
    H_mat = np.zeros((H_px, W_px, 8, 5), dtype=np.float64)

    # Broadcast time offsets to (H, W) if scalar
    dt_am = np.broadcast_to(np.asarray(dt_a_minus, dtype=np.float64), (H_px, W_px))
    dt_ap = np.broadcast_to(np.asarray(dt_a_plus, dtype=np.float64), (H_px, W_px))
    dt_bm = np.broadcast_to(np.asarray(dt_b_minus, dtype=np.float64), (H_px, W_px))
    dt_bp = np.broadcast_to(np.asarray(dt_b_plus, dtype=np.float64), (H_px, W_px))

    # Row 0: A_minus u
    H_mat[:, :, 0, 1] = 1.0
    H_mat[:, :, 0, 3] = dt_am

    # Row 1: A_minus v
    H_mat[:, :, 1, 2] = 1.0
    H_mat[:, :, 1, 4] = dt_am

    # Row 2: A_plus u
    H_mat[:, :, 2, 1] = 1.0
    H_mat[:, :, 2, 3] = dt_ap

    # Row 3: A_plus v
    H_mat[:, :, 3, 2] = 1.0
    H_mat[:, :, 3, 4] = dt_ap

    # Row 4: B_minus u
    H_mat[:, :, 4, 0] = w_u
    H_mat[:, :, 4, 1] = 1.0
    H_mat[:, :, 4, 3] = dt_bm

    # Row 5: B_minus v
    H_mat[:, :, 5, 0] = w_v
    H_mat[:, :, 5, 2] = 1.0
    H_mat[:, :, 5, 4] = dt_bm

    # Row 6: B_plus u
    H_mat[:, :, 6, 0] = w_u
    H_mat[:, :, 6, 1] = 1.0
    H_mat[:, :, 6, 3] = dt_bp

    # Row 7: B_plus v
    H_mat[:, :, 7, 0] = w_v
    H_mat[:, :, 7, 2] = 1.0
    H_mat[:, :, 7, 4] = dt_bp

    return H_mat


def _solve_wls(
    H_matrix: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    regularization: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass WLS solve (internal helper).

    Returns (x, chi2, sigma_h, sigma_u, sigma_v).
    """
    H_px, W_px = H_matrix.shape[:2]

    sqrt_W = np.sqrt(W)
    H_w = H_matrix * sqrt_W[..., np.newaxis]
    y_w = y * sqrt_W

    # Column scaling for stability
    col_norms = np.sqrt(np.einsum("...ki,...ki->...i", H_w, H_w))
    col_norms = np.where(col_norms > 0, col_norms, 1.0)

    H_scaled = H_w / col_norms[:, :, np.newaxis, :]

    A = np.einsum("...ki,...kj->...ij", H_scaled, H_scaled)
    b = np.einsum("...ki,...k->...i", H_scaled, y_w)

    reg = np.eye(5, dtype=np.float64) * regularization
    A = A + reg

    x_scaled = np.linalg.solve(A, b[..., np.newaxis])[..., 0]
    x = x_scaled / col_norms

    # Residuals
    y_pred = np.einsum("...ij,...j->...i", H_matrix, x)
    residual = y - y_pred
    chi2 = np.sum((residual**2) * W, axis=-1)

    # Formal uncertainties
    try:
        C_scaled = np.linalg.inv(A)
        inv_norms = 1.0 / col_norms
        C = C_scaled * inv_norms[:, :, :, np.newaxis] * inv_norms[:, :, np.newaxis, :]
        sigma_h = np.sqrt(np.maximum(C[:, :, 0, 0], 0.0))
        sigma_u = np.sqrt(np.maximum(C[:, :, 3, 3], 0.0))
        sigma_v = np.sqrt(np.maximum(C[:, :, 4, 4], 0.0))
    except np.linalg.LinAlgError:
        sigma_h = np.full((H_px, W_px), np.nan)
        sigma_u = np.full((H_px, W_px), np.nan)
        sigma_v = np.full((H_px, W_px), np.nan)

    return x, chi2, sigma_h, sigma_u, sigma_v


def solve_stereo_winds(
    disparities: dict[str, np.ndarray],
    H_matrix: np.ndarray,
    weights: np.ndarray | None = None,
    regularization: float = 1e-10,
    sat_a: SatelliteConfig | None = None,
    sat_b: SatelliteConfig | None = None,
    n_iter: int = 1,
) -> dict[str, np.ndarray]:
    """Solve the 5-state weighted least squares system per pixel.

    Parameters
    ----------
    disparities : dict with keys D1..D4, each (2, H, W)
        D1: A0→A_minus, D2: A0→A_plus, D3: A0→B_minus, D4: A0→B_plus
    H_matrix : (H, W, 8, 5) design matrix
    weights : (H, W, 8) or None
        Per-observation weights.  None = equal weights.
    regularization : float
        Tikhonov regularization added to diagonal of normal matrix.
    sat_a, sat_b : SatelliteConfig or None
        Satellite configs needed for iterative solve (n_iter > 1).
        When None and n_iter > 1, falls back to a single-pass solve.
    n_iter : int
        Number of iterations.  At each iteration after the first, parallax
        vectors are recomputed at the current height estimate, capturing
        the nonlinear dependence of parallax geometry on height.

    Returns
    -------
    solution : dict with keys h, p_u, p_v, V_u, V_v, chi2, sigma_h,
               sigma_u, sigma_v, quality_flag, delta_h_history
    """
    H_matrix = H_matrix.copy()
    H_px, W_px = H_matrix.shape[:2]

    # Stack 8 observations
    y = np.zeros((H_px, W_px, 8), dtype=np.float64)
    for i, key in enumerate(["D1", "D2", "D3", "D4"]):
        flow = disparities[key]
        y[:, :, 2 * i] = flow[0]
        y[:, :, 2 * i + 1] = flow[1]

    if weights is None:
        W = np.ones((H_px, W_px, 8), dtype=np.float64)
    else:
        W = weights

    can_iterate = n_iter > 1 and sat_a is not None and sat_b is not None
    actual_iters = n_iter if can_iterate else 1

    delta_h_history = []
    h_prev = None

    for iteration in range(actual_iters):
        x, chi2, sigma_h, sigma_u, sigma_v = _solve_wls(
            H_matrix, y, W, regularization,
        )
        h = x[:, :, 0]

        if h_prev is not None:
            delta_h = np.abs(h - h_prev)
            valid_delta = np.isfinite(delta_h) & (h > 0) & (h < 20000)
            if valid_delta.any():
                med_delta = float(np.median(delta_h[valid_delta]))
            else:
                med_delta = 0.0
            delta_h_history.append(med_delta)

            # Log stats for high clouds
            high_mask = valid_delta & (h > 8000)
            if high_mask.any():
                mean_high = float(np.mean(delta_h[high_mask]))
                logger.info(
                    "Iter %d: median |Δh| = %.1f m, "
                    "high cloud (>8km) mean |Δh| = %.1f m",
                    iteration, med_delta, mean_high,
                )

            if med_delta < 50.0:
                logger.info(
                    "Converged at iteration %d (median |Δh| = %.1f m)",
                    iteration, med_delta,
                )
                break

        h_prev = h.copy()

        # Update parallax vectors in H_matrix for next iteration
        if can_iterate and iteration < actual_iters - 1:
            h_clipped = np.clip(np.where(np.isfinite(h), h, 0.0), 0.0, 20000.0)
            w_u_new, w_v_new = compute_parallax_vectors_at_h(
                sat_a, sat_b, h_clipped,
            )
            H_matrix[:, :, 4, 0] = w_u_new
            H_matrix[:, :, 5, 0] = w_v_new
            H_matrix[:, :, 6, 0] = w_u_new
            H_matrix[:, :, 7, 0] = w_v_new

    # Extract state variables
    h = x[:, :, 0]
    p_u = x[:, :, 1]
    p_v = x[:, :, 2]
    V_u = x[:, :, 3]
    V_v = x[:, :, 4]

    # Quality flags
    quality_flag = np.ones((H_px, W_px), dtype=np.float64)
    quality_flag[h < 0] = 0.0
    quality_flag[h > 20000] = 0.0
    quality_flag[~np.isfinite(h)] = 0.0

    return {
        "h": h,
        "p_u": p_u,
        "p_v": p_v,
        "V_u": V_u,
        "V_v": V_v,
        "chi2": chi2,
        "sigma_h": sigma_h,
        "sigma_u": sigma_u,
        "sigma_v": sigma_v,
        "quality_flag": quality_flag,
        "delta_h_history": delta_h_history,
    }


def estimate_cross_sat_offset(
    disparities: dict[str, np.ndarray],
    w_u: np.ndarray,
    w_v: np.ndarray,
    temporal_motion_threshold: float = 1.0,
    sigma: float = 200.0,
) -> tuple[float, float]:
    """Estimate systematic RAFT offset in cross-satellite flows.

    At clear-sky pixels (low temporal motion, h ≈ 0), the measured parallax
    (cross-sat minus temporal) should be zero. Any non-zero residual is a
    systematic RAFT offset that biases height estimates.

    Parameters
    ----------
    disparities : dict with keys D1..D4, each (2, H, W)
    w_u, w_v : parallax sensitivity arrays
    temporal_motion_threshold : max temporal flow magnitude for "clear sky"
    sigma : Gaussian smoothing sigma (not used, kept for future spatial field)

    Returns
    -------
    offset_u, offset_v : scalar offsets to subtract from D3/D4
    """
    D1, D2, D3, D4 = (disparities[k] for k in ["D1", "D2", "D3", "D4"])

    # Temporal motion magnitude as cloud proxy
    temporal_mag = np.sqrt(D1[0]**2 + D1[1]**2 + D2[0]**2 + D2[1]**2)

    # Measured parallax = mean cross-sat minus mean temporal
    par_u = ((D3[0] + D4[0]) / 2) - ((D1[0] + D2[0]) / 2)
    par_v = ((D3[1] + D4[1]) / 2) - ((D1[1] + D2[1]) / 2)

    valid = (
        np.isfinite(par_u) & np.isfinite(par_v)
        & np.isfinite(temporal_mag)
        & (np.abs(w_u) > 1e-7)
    )
    clear_sky = valid & (temporal_mag < temporal_motion_threshold)

    n_clear = int(clear_sky.sum())
    if n_clear < 100:
        return 0.0, 0.0

    offset_u = float(np.median(par_u[clear_sky]))
    offset_v = float(np.median(par_v[clear_sky]))

    return offset_u, offset_v


def correct_cross_sat_offset(
    disparities: dict[str, np.ndarray],
    offset_u: float,
    offset_v: float,
) -> dict[str, np.ndarray]:
    """Subtract systematic offset from cross-satellite flows D3 and D4.

    Returns a new disparities dict with corrected D3/D4 (D1/D2 unchanged).
    """
    corrected = {}
    corrected["D1"] = disparities["D1"]
    corrected["D2"] = disparities["D2"]

    D3_corr = disparities["D3"].copy()
    D4_corr = disparities["D4"].copy()
    D3_corr[0] = D3_corr[0] - offset_u
    D3_corr[1] = D3_corr[1] - offset_v
    D4_corr[0] = D4_corr[0] - offset_u
    D4_corr[1] = D4_corr[1] - offset_v

    corrected["D3"] = D3_corr
    corrected["D4"] = D4_corr

    return corrected


def pixels_to_wind_ms(
    V_u: np.ndarray,
    V_v: np.ndarray,
    sat: SatelliteConfig,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pixel velocity to wind speed in m/s.

    At nadir, pixel scale = |satellite_height * scale_factor|.
    This is an approximation; for full accuracy, compute the local pixel
    footprint size as a function of viewing angle.

    Parameters
    ----------
    V_u, V_v : pixel velocity (pixels per dt_seconds)
    sat : SatelliteConfig
    dt_seconds : the time baseline used in the velocity estimate

    Returns
    -------
    u_ms, v_ms : wind components in m/s (east, north)
    """
    # Pixel scale in meters at nadir
    px_scale_x = abs(sat.satellite_height_m * sat.scale_x)
    px_scale_y = abs(sat.satellite_height_m * sat.scale_y)

    u_ms = V_u * px_scale_x / dt_seconds
    v_ms = V_v * px_scale_y / dt_seconds

    return u_ms, v_ms
