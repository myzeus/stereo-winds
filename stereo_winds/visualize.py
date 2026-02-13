"""Debug visualizations for each stereo wind pipeline stage.

Matplotlib-based plotting with cartopy geostationary projections
for proper fixed-grid rendering with coastlines.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import SatelliteConfig
from .navigation import pixel_to_scanning_angle


def _get_geo_crs(sat: SatelliteConfig) -> Any:
    """Get a cartopy Geostationary CRS for the given satellite."""
    import cartopy.crs as ccrs

    return ccrs.Geostationary(
        central_longitude=sat.sub_lon_deg,
        satellite_height=sat.satellite_height_m,
        sweep_axis=sat.sweep,
    )


def _fixed_grid_extent(sat: SatelliteConfig) -> list[float]:
    """Compute axis extent in scanning angle coordinates (radians * H)."""
    # Convert to cartopy's expected units (meters from projection center)
    H = sat.satellite_height_m
    x_min = sat.x_offset * H
    x_max = (sat.x_offset + sat.scale_x * sat.n_cols) * H
    y_max = sat.y_offset * H
    y_min = (sat.y_offset + sat.scale_y * sat.n_rows) * H
    return [x_min, x_max, y_min, y_max]


def _add_coastlines(ax: Any, sat: SatelliteConfig) -> None:
    """Add coastlines and borders to a geostationary projection axis."""
    import cartopy.feature as cfeature

    ax.coastlines(resolution="50m", color="white", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor="white", linewidth=0.3)


def _image_extent(sat: SatelliteConfig) -> list[float]:
    """Image extent in projection coordinates for imshow."""
    H = sat.satellite_height_m
    x0 = sat.x_offset * H
    x1 = (sat.x_offset + sat.scale_x * (sat.n_cols - 1)) * H
    y0 = (sat.y_offset + sat.scale_y * (sat.n_rows - 1)) * H
    y1 = sat.y_offset * H
    return [x0, x1, y0, y1]


# -------------------------------------------------------------------------
# Remapping verification
# -------------------------------------------------------------------------


def plot_remap_comparison(
    a0: np.ndarray,
    b_original: np.ndarray,
    b_remapped: np.ndarray,
    sat_a: SatelliteConfig,
    sat_b: SatelliteConfig,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "Remap Verification",
) -> plt.Figure:
    """3-panel comparison: A0, B original, B remapped onto A's grid.

    Plus a 4th panel showing A0 - B_remapped (parallax signal).
    """
    geo_a = _get_geo_crs(sat_a)
    geo_b = _get_geo_crs(sat_b)
    ext_a = _image_extent(sat_a)
    ext_b = _image_extent(sat_b)

    fig, axes = plt.subplots(
        2, 2, figsize=(16, 14),
        subplot_kw={"projection": geo_a},
    )
    # Override projection for B panel
    axes[0, 1].remove()
    axes[0, 1] = fig.add_subplot(2, 2, 2, projection=geo_b)

    # Panel 1: A0
    ax = axes[0, 0]
    ax.imshow(a0, origin="upper", extent=ext_a, cmap="gray", vmin=vmin, vmax=vmax)
    _add_coastlines(ax, sat_a)
    ax.set_title(f"A0 ({sat_a.satellite_id})")

    # Panel 2: B original
    ax = axes[0, 1]
    ax.imshow(b_original, origin="upper", extent=ext_b, cmap="gray", vmin=vmin, vmax=vmax)
    _add_coastlines(ax, sat_b)
    ax.set_title(f"B original ({sat_b.satellite_id})")

    # Panel 3: B remapped onto A
    ax = axes[1, 0]
    ax.imshow(b_remapped, origin="upper", extent=ext_a, cmap="gray", vmin=vmin, vmax=vmax)
    _add_coastlines(ax, sat_a)
    ax.set_title("B remapped onto A's grid")

    # Panel 4: Difference (parallax signal)
    ax = axes[1, 1]
    diff = a0.astype(np.float64) - b_remapped.astype(np.float64)
    d_abs = np.nanpercentile(np.abs(diff), 99)
    im = ax.imshow(
        diff, origin="upper", extent=ext_a, cmap="RdBu_r",
        vmin=-d_abs, vmax=d_abs,
    )
    _add_coastlines(ax, sat_a)
    ax.set_title("A0 - B_remapped (parallax)")
    plt.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_remap_lut(
    col_b: np.ndarray,
    row_b: np.ndarray,
    sat_a: SatelliteConfig,
    step: int = 100,
) -> plt.Figure:
    """Plot the remap LUT as a displacement vector field."""
    rows_a = np.arange(sat_a.n_rows)
    cols_a = np.arange(sat_a.n_cols)
    col_grid, row_grid = np.meshgrid(cols_a, rows_a)

    # Subsample
    r = row_grid[::step, ::step]
    c = col_grid[::step, ::step]
    cb = col_b[::step, ::step]
    rb = row_b[::step, ::step]

    du = cb - c
    dv = rb - r
    valid = np.isfinite(du) & np.isfinite(dv)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.quiver(
        c[valid], r[valid], du[valid], -dv[valid],
        np.sqrt(du[valid] ** 2 + dv[valid] ** 2),
        cmap="viridis", scale_units="xy",
    )
    ax.set_xlim(0, sat_a.n_cols)
    ax.set_ylim(sat_a.n_rows, 0)
    ax.set_aspect("equal")
    ax.set_title("Remap LUT: A→B pixel displacement")
    ax.set_xlabel("Column (A)")
    ax.set_ylabel("Row (A)")
    fig.tight_layout()
    return fig


def plot_zenith_angle_mask(
    valid_mask: np.ndarray,
    sat_a: SatelliteConfig,
) -> plt.Figure:
    """Show which pixels pass the limb-angle threshold."""
    geo = _get_geo_crs(sat_a)
    ext = _image_extent(sat_a)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": geo})
    ax.imshow(
        valid_mask.astype(float), origin="upper", extent=ext,
        cmap="RdYlGn", vmin=0, vmax=1,
    )
    _add_coastlines(ax, sat_a)
    ax.set_title("Valid pixel mask (green = valid)")
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Flow field verification
# -------------------------------------------------------------------------


def plot_flow_field(
    flow: np.ndarray,
    sat: SatelliteConfig,
    title: str = "Optical Flow",
    vmax: float | None = None,
) -> plt.Figure:
    """Plot a single flow field as du, dv components + magnitude.

    Parameters
    ----------
    flow : (2, H, W) array of (du, dv) pixel displacements.
    """
    du, dv = flow[0], flow[1]
    mag = np.sqrt(du**2 + dv**2)

    if vmax is None:
        vmax = float(np.nanpercentile(mag, 99))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(du, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("du (pixels)")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)

    im1 = axes[1].imshow(dv, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("dv (pixels)")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)

    im2 = axes[2].imshow(mag, cmap="viridis", vmin=0, vmax=vmax)
    axes[2].set_title("magnitude (pixels)")
    plt.colorbar(im2, ax=axes[2], shrink=0.7)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_all_disparities(
    disparities: dict[str, np.ndarray],
    sat_a: SatelliteConfig,
) -> plt.Figure:
    """4x2 grid of all disparity pairs: (du, dv) for D1..D4.

    Expected keys: D1 (A0→A_minus), D2 (A0→A_plus),
                   D3 (A0→B_minus), D4 (A0→B_plus).
    """
    labels = [
        ("D1", "A0 → A_minus (temporal bwd)"),
        ("D2", "A0 → A_plus (temporal fwd)"),
        ("D3", "A0 → B_minus (cross-sat bwd)"),
        ("D4", "A0 → B_plus (cross-sat fwd)"),
    ]

    # Find global vmax
    all_mag = []
    for key, _ in labels:
        if key in disparities:
            f = disparities[key]
            all_mag.append(np.nanpercentile(np.sqrt(f[0] ** 2 + f[1] ** 2), 99))
    vmax = max(all_mag) if all_mag else 10.0

    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    for i, (key, label) in enumerate(labels):
        if key not in disparities:
            axes[i, 0].set_visible(False)
            axes[i, 1].set_visible(False)
            continue
        flow = disparities[key]
        im0 = axes[i, 0].imshow(flow[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[i, 0].set_title(f"{label} — du")
        plt.colorbar(im0, ax=axes[i, 0], shrink=0.7)

        im1 = axes[i, 1].imshow(flow[1], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[i, 1].set_title(f"{label} — dv")
        plt.colorbar(im1, ax=axes[i, 1], shrink=0.7)

    fig.suptitle("All Disparity Fields", fontsize=14)
    fig.tight_layout()
    return fig


def plot_flow_difference(
    flow_temporal: np.ndarray,
    flow_cross: np.ndarray,
    title: str = "Flow Difference (parallax isolation)",
) -> plt.Figure:
    """Difference between cross-satellite and temporal flow fields."""
    diff = flow_cross - flow_temporal
    du, dv = diff[0], diff[1]
    mag = np.sqrt(du**2 + dv**2)
    vmax = float(np.nanpercentile(mag, 99))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(du, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Δdu (parallax u)")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)

    im1 = axes[1].imshow(dv, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Δdv (parallax v)")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)

    im2 = axes[2].imshow(mag, cmap="hot", vmin=0, vmax=vmax)
    axes[2].set_title("Parallax magnitude")
    plt.colorbar(im2, ax=axes[2], shrink=0.7)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Solver output verification
# -------------------------------------------------------------------------


def plot_solver_results(
    solution: dict[str, np.ndarray],
    sat_a: SatelliteConfig,
) -> plt.Figure:
    """Multi-panel figure of solver outputs.

    Expected keys: h, u_wind, v_wind, p_u, p_v, chi2, quality_flag.
    """
    geo = _get_geo_crs(sat_a)
    ext = _image_extent(sat_a)

    fig, axes = plt.subplots(
        2, 3, figsize=(20, 12),
        subplot_kw={"projection": geo},
    )

    # 1. Cloud-top height
    ax = axes[0, 0]
    h = solution.get("h", np.zeros((sat_a.n_rows, sat_a.n_cols)))
    im = ax.imshow(h, origin="upper", extent=ext, cmap="terrain", vmin=0, vmax=15000)
    _add_coastlines(ax, sat_a)
    ax.set_title("Cloud-top Height (m)")
    plt.colorbar(im, ax=ax, shrink=0.6)

    # 2. Wind speed
    ax = axes[0, 1]
    u = solution.get("u_wind", np.zeros_like(h))
    v = solution.get("v_wind", np.zeros_like(h))
    spd = np.sqrt(u**2 + v**2)
    im = ax.imshow(spd, origin="upper", extent=ext, cmap="magma", vmin=0, vmax=40)
    _add_coastlines(ax, sat_a)
    ax.set_title("Wind Speed (m/s)")
    plt.colorbar(im, ax=ax, shrink=0.6)

    # 3. Wind vectors over height
    ax = axes[0, 2]
    im = ax.imshow(h, origin="upper", extent=ext, cmap="terrain", vmin=0, vmax=15000, alpha=0.5)
    _add_coastlines(ax, sat_a)
    # Quiver (subsampled)
    step = max(sat_a.n_rows // 50, 1)
    rows = np.arange(0, sat_a.n_rows, step)
    cols = np.arange(0, sat_a.n_cols, step)
    cg, rg = np.meshgrid(cols, rows)
    x_q, y_q = pixel_to_scanning_angle(cg, rg, sat_a)
    u_sub = u[::step, ::step]
    v_sub = v[::step, ::step]
    ax.quiver(
        x_q * sat_a.satellite_height_m,
        y_q * sat_a.satellite_height_m,
        u_sub, -v_sub, scale=500, width=0.002, color="k", alpha=0.6,
    )
    ax.set_title("Wind Vectors")

    # 4. Position correction
    ax = axes[1, 0]
    p_u = solution.get("p_u", np.zeros_like(h))
    p_v = solution.get("p_v", np.zeros_like(h))
    p_mag = np.sqrt(p_u**2 + p_v**2)
    im = ax.imshow(p_mag, origin="upper", extent=ext, cmap="viridis", vmin=0, vmax=3)
    _add_coastlines(ax, sat_a)
    ax.set_title("Position Correction (px)")
    plt.colorbar(im, ax=ax, shrink=0.6)

    # 5. Chi-squared
    ax = axes[1, 1]
    chi2 = solution.get("chi2", np.zeros_like(h))
    im = ax.imshow(chi2, origin="upper", extent=ext, cmap="hot_r", vmin=0, vmax=10)
    _add_coastlines(ax, sat_a)
    ax.set_title("Chi-squared Residual")
    plt.colorbar(im, ax=ax, shrink=0.6)

    # 6. Quality flag
    ax = axes[1, 2]
    qf = solution.get("quality_flag", np.ones_like(h))
    im = ax.imshow(qf, origin="upper", extent=ext, cmap="RdYlGn", vmin=0, vmax=1)
    _add_coastlines(ax, sat_a)
    ax.set_title("Quality Flag (1=good)")
    plt.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle("Stereo Wind Retrieval Results", fontsize=14)
    fig.tight_layout()
    return fig


def plot_height_histogram(
    h: np.ndarray,
    quality_flag: np.ndarray | None = None,
) -> plt.Figure:
    """Histogram of retrieved cloud-top heights."""
    fig, ax = plt.subplots(figsize=(8, 5))

    h_valid = h[np.isfinite(h)]
    if quality_flag is not None:
        h_valid = h[np.isfinite(h) & (quality_flag > 0)]

    ax.hist(h_valid.ravel(), bins=100, range=(0, 18000), color="steelblue", alpha=0.8)
    ax.set_xlabel("Cloud-top Height (m)")
    ax.set_ylabel("Count")
    ax.set_title("Height Distribution (QC-passed pixels)")
    fig.tight_layout()
    return fig


def plot_parallax_vectors(
    w_u: np.ndarray,
    w_v: np.ndarray,
    sat_a: SatelliteConfig,
    step: int = 100,
) -> plt.Figure:
    """Quiver plot of parallax sensitivity field w_hat."""
    rows = np.arange(0, sat_a.n_rows, step)
    cols = np.arange(0, sat_a.n_cols, step)
    cg, rg = np.meshgrid(cols, rows)

    wu_sub = w_u[::step, ::step]
    wv_sub = w_v[::step, ::step]
    valid = np.isfinite(wu_sub) & np.isfinite(wv_sub)
    mag = np.sqrt(wu_sub**2 + wv_sub**2)

    fig, ax = plt.subplots(figsize=(10, 10))
    q = ax.quiver(
        cg[valid], rg[valid], wu_sub[valid], -wv_sub[valid],
        mag[valid], cmap="plasma",
    )
    ax.set_xlim(0, sat_a.n_cols)
    ax.set_ylim(sat_a.n_rows, 0)
    ax.set_aspect("equal")
    ax.set_title("Parallax Sensitivity w_hat (px/m)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.colorbar(q, ax=ax, label="magnitude (px/m)")
    fig.tight_layout()
    return fig
