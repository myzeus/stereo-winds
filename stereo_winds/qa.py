"""Quality assurance flags for stereo wind retrievals.

Computes multi-level per-pixel QA flags based on retrieval diagnostics
(chi-squared residual, formal uncertainties, height gradient).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel


def height_gradient(h_2d: np.ndarray) -> np.ndarray:
    """Sobel height-gradient magnitude (m/pixel).

    Parameters
    ----------
    h_2d : 2-D height field (meters). NaN pixels are filled with 0
        before computing the gradient.

    Returns
    -------
    Gradient magnitude array, same shape as *h_2d*.
    """
    h_filled = np.where(np.isfinite(h_2d), h_2d, 0.0)
    gx = sobel(h_filled, axis=1)
    gy = sobel(h_filled, axis=0)
    return np.sqrt(gx**2 + gy**2) / 8.0


def compute_qa_flag(
    solution: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    chi2_threshold: float = 10.0,
) -> np.ndarray:
    """Compute a multi-level quality assurance flag.

    Levels (higher = stricter):

    * **0 — no_retrieval**: invalid height, out of zenith range, or failed solve
    * **1 — low_quality**: basic validity + chi2 ≤ *chi2_threshold* + finite winds
    * **2 — high_quality**: level 1 + chi2 ≤ 0.2, height-dependent sigma_h,
      height gradient ≤ 3000 m/pixel, speed ≤ 100 m/s

    Parameters
    ----------
    solution : dict from solve_stereo_winds(), must also contain
        ``u_wind`` and ``v_wind`` (m/s).
    valid_mask : boolean array (True = within valid zenith angle).
    chi2_threshold : loose chi-squared cutoff for level 1 (default 10.0).

    Returns
    -------
    float32 array with integer values 0, 1, or 2.
    """
    h = solution["h"]
    chi2 = solution["chi2"]
    sigma_h = solution["sigma_h"]
    u = solution["u_wind"]
    v = solution["v_wind"]
    speed = np.sqrt(u**2 + v**2)

    qa = np.zeros(h.shape, dtype=np.float32)

    # Level 1: current pipeline QC
    level1 = (
        np.isfinite(h) & (h >= 0) & (h <= 20000)
        & valid_mask
        & np.isfinite(u) & np.isfinite(v)
        & (chi2 <= chi2_threshold)
    )
    qa[level1] = 1.0

    # Level 2: strict QC (matches triple-collocation validation filters)
    sigh_thresh = np.where(
        h < 3000, 1000.0,
        np.where(h < 7000, 2000.0, 1000.0),
    )
    hgrad = height_gradient(h)
    level2 = level1 & (
        (chi2 <= 0.2)
        & (sigma_h <= sigh_thresh)
        & (hgrad <= 3000.0)
        & (speed <= 100.0)
    )
    qa[level2] = 2.0

    return qa
