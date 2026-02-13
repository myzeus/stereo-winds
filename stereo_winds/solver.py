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

import numpy as np

from .config import SatelliteConfig
from .navigation import (
    fixed_grid_to_geodetic,
    geodetic_to_fixed_grid,
    pixel_to_scanning_angle,
    scanning_angle_to_pixel,
)


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


def solve_stereo_winds(
    disparities: dict[str, np.ndarray],
    H_matrix: np.ndarray,
    weights: np.ndarray | None = None,
    regularization: float = 1e-10,
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

    Returns
    -------
    solution : dict with keys h, p_u, p_v, V_u, V_v, chi2, sigma_h,
               sigma_u, sigma_v, quality_flag
    """
    H_px, W_px = H_matrix.shape[:2]

    # Stack 8 observations: [D1_u, D1_v, D2_u, D2_v, D3_u, D3_v, D4_u, D4_v]
    y = np.zeros((H_px, W_px, 8), dtype=np.float64)
    for i, key in enumerate(["D1", "D2", "D3", "D4"]):
        flow = disparities[key]  # (2, H, W)
        y[:, :, 2 * i] = flow[0]      # u component
        y[:, :, 2 * i + 1] = flow[1]  # v component

    H = H_matrix  # (H, W, 8, 5)

    # Weight matrix (diagonal)
    if weights is None:
        W = np.ones((H_px, W_px, 8), dtype=np.float64)
    else:
        W = weights

    # Apply weights by scaling H and y
    sqrt_W = np.sqrt(W)
    H_w = H * sqrt_W[..., np.newaxis]  # (H, W, 8, 5)
    y_w = y * sqrt_W  # (H, W, 8)

    # Solve via SVD-based lstsq for numerical stability.
    # The design matrix has extreme dynamic range (parallax ~1e-6 vs
    # time offsets ~600), making the normal equations ill-conditioned.
    # Reshape to (N_pixels, 8, 5) for vectorized lstsq.
    H_flat = H_w.reshape(-1, 8, 5)   # (N, 8, 5)
    y_flat = y_w.reshape(-1, 8, 1)   # (N, 8, 1)

    # np.linalg.lstsq on batched arrays: solve each pixel independently
    # lstsq doesn't support batched dims, so use the normal equations
    # with a column-scaled approach for stability.
    #
    # Column scaling: normalize each column of H to unit norm to improve
    # conditioning, then rescale the solution.
    col_norms = np.sqrt(np.einsum("...ki,...ki->...i", H_w, H_w))  # (H, W, 5)
    col_norms = np.where(col_norms > 0, col_norms, 1.0)

    H_scaled = H_w / col_norms[:, :, np.newaxis, :]  # (H, W, 8, 5)

    # Normal equations with scaled H
    A = np.einsum("...ki,...kj->...ij", H_scaled, H_scaled)  # (H, W, 5, 5)
    b = np.einsum("...ki,...k->...i", H_scaled, y_w)  # (H, W, 5)

    # Small regularization on the scaled system
    reg = np.eye(5, dtype=np.float64) * regularization
    A = A + reg

    # Solve the scaled system
    x_scaled = np.linalg.solve(A, b[..., np.newaxis])[..., 0]  # (H, W, 5)

    # Unscale the solution
    x = x_scaled / col_norms  # (H, W, 5)

    # Extract state variables
    h = x[:, :, 0]
    p_u = x[:, :, 1]
    p_v = x[:, :, 2]
    V_u = x[:, :, 3]
    V_v = x[:, :, 4]

    # Residuals
    y_pred = np.einsum("...ij,...j->...i", H, x)  # (H, W, 8)
    residual = y - y_pred
    weighted_resid_sq = (residual**2) * W
    chi2 = np.sum(weighted_resid_sq, axis=-1)  # (H, W), DOF = 8 - 5 = 3

    # Formal uncertainties from covariance: C = D^{-1} A_s^{-1} D^{-1}
    # where D = diag(col_norms) and A_s is the scaled normal matrix
    try:
        C_scaled = np.linalg.inv(A)  # (H, W, 5, 5)
        # Unscale: C[i,j] = C_scaled[i,j] / (col_norms[i] * col_norms[j])
        inv_norms = 1.0 / col_norms
        C = C_scaled * inv_norms[:, :, :, np.newaxis] * inv_norms[:, :, np.newaxis, :]
        sigma_h = np.sqrt(np.maximum(C[:, :, 0, 0], 0.0))
        sigma_u = np.sqrt(np.maximum(C[:, :, 3, 3], 0.0))
        sigma_v = np.sqrt(np.maximum(C[:, :, 4, 4], 0.0))
    except np.linalg.LinAlgError:
        sigma_h = np.full((H_px, W_px), np.nan)
        sigma_u = np.full((H_px, W_px), np.nan)
        sigma_v = np.full((H_px, W_px), np.nan)

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
    }


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
