"""Figure 1 for the stereo-winds paper: parallax concept + footprint overlap.

Produces a two-panel PNG:

  (a) A 2-D cross-section schematic of two geostationary satellites (A, B)
      viewing a cloud top, illustrating the parallax geometry and naming the
      retrieval state variables ``[h, p_u, p_v, V_u, V_v]`` and the parallax
      sensitivity vector ``w_hat``, following the observation model of
      Carr et al. (2020):  delta(t_n) = h * w_hat_n + p + V * (t_n - t_0).

  (b) Real GOES-19 (sub_lon -75 deg) + GOES-18 (sub_lon -137 deg) C14 full-disk
      imagery on a shared map, with each satellite's usable footprint outlined
      and the stereo-overlap region (both zenith angles below a threshold)
      highlighted.

Usage
-----
    python scripts/fig_parallax_concept.py \
        --time 2025-07-01T18:00 --cache-dir ./cache --out fig1_parallax_concept.png

Panel (a) needs no data; panel (b) downloads two ABI L1b full-disk files from
the public NOAA S3 buckets (no credentials required).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Visual styling
COLOR_A = "#e8743b"  # satellite A (GOES-19) — orange
COLOR_B = "#19a7ce"  # satellite B (GOES-18) — cyan/blue
COLOR_CLOUD = "#f2f2f2"
COLOR_GROUND = "#7a5c3e"
COLOR_PARALLAX = "#c0392b"
COLOR_WIND = "#2e7d32"


# ---------------------------------------------------------------------------
# Panel (a): parallax geometry schematic
# ---------------------------------------------------------------------------

def _draw_satellite(ax, x, y, color, scale=0.45):
    """Draw a simple satellite glyph (body + two solar panels) centered at (x, y)."""
    body_w, body_h = 0.45 * scale, 0.55 * scale
    ax.add_patch(
        plt.Rectangle(
            (x - body_w / 2, y - body_h / 2), body_w, body_h,
            facecolor=color, edgecolor="black", linewidth=1.0, zorder=6,
        )
    )
    panel_w, panel_h = 0.55 * scale, 0.28 * scale
    for sign in (-1, 1):
        px = x + sign * (body_w / 2 + panel_w / 2)
        ax.add_patch(
            plt.Rectangle(
                (px - panel_w / 2, y - panel_h / 2), panel_w, panel_h,
                facecolor="#27408b", edgecolor="black", linewidth=0.8, zorder=6,
            )
        )
        ax.plot([x + sign * body_w / 2, px - panel_w / 2], [y, y],
                color="black", linewidth=0.8, zorder=6)


def _draw_cloud(ax, cx, cy, scale=1.0, alpha=1.0, edge=True):
    """Draw a fluffy cloud as a cluster of overlapping ellipses centered at (cx, cy)."""
    blobs = [
        (-0.45, -0.05, 0.55, 0.40),
        (0.0, 0.10, 0.70, 0.55),
        (0.45, -0.05, 0.55, 0.40),
        (-0.15, -0.18, 0.55, 0.38),
        (0.20, -0.18, 0.55, 0.38),
    ]
    ec = "#9aa0a6" if edge else "none"
    for dx, dy, w, h in blobs:
        ax.add_patch(
            Ellipse(
                (cx + dx * scale, cy + dy * scale), w * scale, h * scale,
                facecolor=COLOR_CLOUD, edgecolor=ec, linewidth=0.8,
                alpha=alpha, zorder=4,
            )
        )


def draw_geometry_schematic(ax):
    """Render the 2-D parallax-geometry schematic onto ``ax``.

    Pure matplotlib, no data. Two satellites view a cloud at height ``h``; the
    cloud's apparent ground position differs between the two views, and that
    gap is the measured parallax disparity. Wind moves the cloud between scan
    epochs t0-, t0, t0+.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("auto")
    ax.axis("off")

    # --- Earth surface: a shallow dome (large-radius arc) ------------------
    R, cx0, cy0 = 51.5, 5.0, -50.0  # circle giving a subtle curve, top at y~1.5
    xs = np.linspace(0, 10, 400)
    ys = cy0 + np.sqrt(R**2 - (xs - cx0) ** 2)
    y_ground = float(cy0 + np.sqrt(R**2 - (5.0 - cx0) ** 2))  # ground level at x=5
    ax.fill_between(xs, 0, ys, facecolor=COLOR_GROUND, alpha=0.22, zorder=0)
    ax.plot(xs, ys, color=COLOR_GROUND, linewidth=2.0, zorder=1)
    ax.text(0.4, y_ground - 0.55, "Earth surface", color=COLOR_GROUND,
            fontsize=10, style="italic", zorder=2)

    # --- Geometry: satellites, cloud, lines of sight -----------------------
    sat_a = np.array([2.4, 9.1])
    sat_b = np.array([7.6, 9.1])
    cloud = np.array([5.0, 4.3])
    cloud_ground = np.array([5.0, y_ground])

    def ground_intercept(sat):
        """Where the sat->cloud ray, extended, meets the ground level."""
        d = cloud - sat
        s = (y_ground - sat[1]) / d[1]
        return np.array([sat[0] + s * d[0], y_ground])

    gi_a = ground_intercept(sat_a)  # A's apparent ground position (to the right)
    gi_b = ground_intercept(sat_b)  # B's apparent ground position (to the left)

    # Lines of sight: solid sat->cloud, dashed cloud->ground intercept
    for sat, gi, color in ((sat_a, gi_a, COLOR_A), (sat_b, gi_b, COLOR_B)):
        ax.plot([sat[0], cloud[0]], [sat[1], cloud[1]], color=color,
                linewidth=2.0, zorder=3)
        ax.plot([cloud[0], gi[0]], [cloud[1], gi[1]], color=color,
                linewidth=1.6, linestyle=(0, (4, 3)), zorder=3)
        ax.plot([gi[0]], [gi[1]], marker="v", color=color, markersize=9,
                zorder=5)

    # Satellites
    _draw_satellite(ax, sat_a[0], sat_a[1], COLOR_A)
    _draw_satellite(ax, sat_b[0], sat_b[1], COLOR_B)
    ax.text(sat_a[0], sat_a[1] + 0.55, "Satellite A\n(GOES-19)", color=COLOR_A,
            fontsize=10, fontweight="bold", ha="center", va="bottom")
    ax.text(sat_b[0], sat_b[1] + 0.55, "Satellite B\n(GOES-18)", color=COLOR_B,
            fontsize=10, fontweight="bold", ha="center", va="bottom")

    # --- Wind: cloud ghosts at t0-, t0+ and a velocity arrow ---------------
    wind = np.array([1.25, 0.0])  # horizontal displacement over one time step
    _draw_cloud(ax, cloud[0] - wind[0], cloud[1], scale=0.78, alpha=0.30, edge=False)
    _draw_cloud(ax, cloud[0] + wind[0], cloud[1], scale=0.78, alpha=0.30, edge=False)
    _draw_cloud(ax, cloud[0], cloud[1], scale=0.95, alpha=1.0)
    for dx, lab in ((-wind[0], r"$t_0^{-}$"), (0.0, r"$t_0$"), (wind[0], r"$t_0^{+}$")):
        ax.text(cloud[0] + dx, cloud[1] + 0.95, lab, fontsize=10, ha="center",
                color="#555555")
    ax.add_patch(FancyArrowPatch(
        (cloud[0] + 0.55, cloud[1]), (cloud[0] + wind[0] + 0.35, cloud[1]),
        arrowstyle="-|>", mutation_scale=16, color=COLOR_WIND, linewidth=2.2,
        zorder=5))
    ax.text(cloud[0] + wind[0] / 2 + 0.45, cloud[1] + 0.30,
            r"$\vec{V}=(V_u, V_v)$", color=COLOR_WIND, fontsize=11,
            fontweight="bold", ha="center")

    # --- Height h ----------------------------------------------------------
    hx = 5.0
    ax.add_patch(FancyArrowPatch(
        (hx, y_ground), (hx, cloud[1] - 0.18), arrowstyle="<|-|>",
        mutation_scale=12, color="black", linewidth=1.4, zorder=5))
    ax.plot([cloud_ground[0]], [cloud_ground[1]], marker="o", color="black",
            markersize=5, zorder=5)
    ax.text(hx + 0.18, (y_ground + cloud[1]) / 2, r"$h$", fontsize=15,
            fontweight="bold", va="center")

    # --- Parallax displacement between the two apparent ground positions ---
    y_par = y_ground - 0.95
    ax.add_patch(FancyArrowPatch(
        (gi_b[0], y_par), (gi_a[0], y_par), arrowstyle="<|-|>",
        mutation_scale=14, color=COLOR_PARALLAX, linewidth=2.0, zorder=5))
    for gi in (gi_a, gi_b):
        ax.plot([gi[0], gi[0]], [y_ground, y_par], color=COLOR_PARALLAX,
                linewidth=0.8, linestyle=":", zorder=4)
    ax.text((gi_a[0] + gi_b[0]) / 2, y_par - 0.32,
            "apparent parallax disparity  " + r"$\delta \propto h\,\hat{w}$",
            color=COLOR_PARALLAX, fontsize=10.5, fontweight="bold", ha="center",
            va="top")

    # Parallax sensitivity direction w_hat (unit vector along the disparity axis)
    wmid = (gi_a[0] + gi_b[0]) / 2
    ax.add_patch(FancyArrowPatch(
        (wmid, y_ground + 0.30), (gi_a[0] - 0.05, y_ground + 0.30),
        arrowstyle="-|>", mutation_scale=13, color=COLOR_PARALLAX,
        linewidth=1.6, zorder=5))
    ax.text(wmid + 0.05, y_ground + 0.55, r"$\hat{w}$", color=COLOR_PARALLAX,
            fontsize=13, fontweight="bold", ha="center")

    # --- Co-registration offset p at an apparent ground position -----------
    ax.add_patch(FancyArrowPatch(
        (gi_a[0], y_ground - 0.02), (gi_a[0] + 0.85, y_ground - 0.02),
        arrowstyle="-|>", mutation_scale=12, color="#6a1b9a", linewidth=1.8,
        zorder=6))
    ax.text(gi_a[0] + 0.95, y_ground - 0.02, r"$\vec{p}=(p_u, p_v)$",
            color="#6a1b9a", fontsize=10.5, fontweight="bold", ha="left",
            va="center")

    # --- Observation model box (top-center, clear of the satellites) -------
    bx0, bw = 2.85, 4.30
    ax.add_patch(FancyBboxPatch(
        (bx0, 8.50), bw, 1.15, boxstyle="round,pad=0.12,rounding_size=0.12",
        facecolor="#fbf7e8", edgecolor="#999999", linewidth=1.0, zorder=7))
    bxc = bx0 + bw / 2
    ax.text(bxc, 9.27,
            r"$\delta(t_n) = h\,\hat{w}_n + \vec{p} + \vec{V}\,(t_n - t_0)$",
            fontsize=12.5, ha="center", va="center", zorder=8)
    ax.text(bxc, 8.74, "Carr et al. (2020), Eq. 1", fontsize=8.5,
            style="italic", color="#777777", ha="center", va="center", zorder=8)

    ax.text(0.1, 0.35, "(not to scale)", fontsize=8, style="italic",
            color="#999999")
    ax.set_title("(a)  Cross-satellite parallax geometry", fontsize=13,
                 fontweight="bold", loc="left")


# ---------------------------------------------------------------------------
# Panel (b): real footprint overlap
# ---------------------------------------------------------------------------

def _to_brightness(rad):
    """Convert ABI radiance to an inverted, percentile-clipped grayscale image."""
    rad = rad.astype(np.float32)
    finite = np.isfinite(rad)
    if not finite.any():
        return rad
    lo, hi = np.nanpercentile(rad[finite], [1, 99])
    bright = 1.0 - np.clip((rad - lo) / (hi - lo + 1e-9), 0, 1)
    bright[~finite] = np.nan  # keep off-disk pixels transparent
    return bright


def _imshow_extent(cfg):
    """imshow extent (meters in the geostationary plane) for a full-disk grid."""
    H = cfg.satellite_height_m
    x0 = cfg.x_offset * H
    x1 = (cfg.x_offset + cfg.scale_x * (cfg.n_cols - 1)) * H
    y_top = cfg.y_offset * H
    y_bot = (cfg.y_offset + cfg.scale_y * (cfg.n_rows - 1)) * H
    return [x0, x1, y_bot, y_top]


def load_pair(t, cache_dir, band="C14"):
    """Download and load GOES-19 and GOES-18 full-disk imagery for time ``t``.

    Returns a list of ``(brightness_image, SatelliteConfig)`` tuples.
    """
    from stereo_winds.data_loading import load_goes_scene

    out = []
    for sat in ("goes19", "goes18"):
        logger.info("Loading %s %s at %s ...", sat, band, t)
        data, cfg = load_goes_scene(t, band, satellite=sat, cache_dir=cache_dir)
        out.append((_to_brightness(data), cfg))
    return out


def draw_overlap_panel(ax, pair, theta_max=80.0, theta_stereo=None):
    """Render the footprint-overlap panel onto cartopy GeoAxes ``ax``.

    Parameters
    ----------
    ax : cartopy GeoAxes in PlateCarree projection.
    pair : list of (brightness_image, SatelliteConfig) for [GOES-19, GOES-18].
    theta_max : satellite zenith-angle threshold (deg) defining each footprint.
    theta_stereo : optional tighter threshold; if set, draws a second contour
        and uses it for the highlighted stereo-overlap region.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    from stereo_winds.navigation import compute_zenith_angle

    extent = [-178, -22, -68, 68]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    gray = plt.cm.gray.copy()
    gray.set_bad(alpha=0.0)

    (img19, cfg19), (img18, cfg18) = pair

    # Imagery: each disk through its own geostationary transform.
    for img, cfg in ((img18, cfg18), (img19, cfg19)):  # B under A
        geo = ccrs.Geostationary(
            central_longitude=cfg.sub_lon_deg,
            satellite_height=cfg.satellite_height_m,
            sweep_axis=cfg.sweep,
        )
        ax.imshow(np.ma.masked_invalid(img), origin="upper",
                  extent=_imshow_extent(cfg), transform=geo, cmap=gray,
                  vmin=0, vmax=1, zorder=1)

    ax.add_feature(cfeature.COASTLINE, edgecolor="#ffd23f", linewidth=0.5,
                   zorder=3)
    ax.add_feature(cfeature.BORDERS, edgecolor="#ffd23f", linewidth=0.25,
                   zorder=3)

    # Footprint / overlap regions on a lat-lon mesh.
    lons = np.linspace(extent[0], extent[1], 561)
    lats = np.linspace(extent[2], extent[3], 401)
    LON, LAT = np.meshgrid(lons, lats)
    z19 = compute_zenith_angle(LAT, LON, cfg19)
    z18 = compute_zenith_angle(LAT, LON, cfg18)

    pc = ccrs.PlateCarree()
    cs19 = ax.contour(LON, LAT, z19, levels=[theta_max], colors=[COLOR_A],
                      linewidths=2.2, transform=pc, zorder=4)
    cs18 = ax.contour(LON, LAT, z18, levels=[theta_max], colors=[COLOR_B],
                      linewidths=2.2, transform=pc, zorder=4)

    thr = theta_stereo if theta_stereo is not None else theta_max
    overlap = ((z19 <= thr) & (z18 <= thr)).astype(float)
    ax.contourf(LON, LAT, overlap, levels=[0.5, 1.5], colors=["#f4d03f"],
                alpha=0.28, transform=pc, zorder=2)
    ax.contour(LON, LAT, overlap, levels=[0.5], colors=["#b7950b"],
               linewidths=1.4, transform=pc, zorder=4)
    if theta_stereo is not None:
        ax.contour(LON, LAT, z19, levels=[theta_stereo], colors=[COLOR_A],
                   linewidths=1.0, linestyles="--", transform=pc, zorder=4)
        ax.contour(LON, LAT, z18, levels=[theta_stereo], colors=[COLOR_B],
                   linewidths=1.0, linestyles="--", transform=pc, zorder=4)

    # Sub-satellite points.
    for cfg, color, name in ((cfg19, COLOR_A, "GOES-19"),
                             (cfg18, COLOR_B, "GOES-18")):
        ax.plot(cfg.sub_lon_deg, 0, marker="*", markersize=15, color=color,
                markeredgecolor="black", markeredgewidth=0.6, transform=pc,
                zorder=6)
        ax.text(cfg.sub_lon_deg, -5.5, name, color=color, fontsize=9.5,
                fontweight="bold", ha="center", transform=pc, zorder=6)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                      alpha=0.4, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False

    # Legend.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], color=COLOR_A, lw=2.2,
               label=f"GOES-19 footprint ({theta_max:.0f}° zenith)"),
        Line2D([0], [0], color=COLOR_B, lw=2.2,
               label=f"GOES-18 footprint ({theta_max:.0f}° zenith)"),
        Patch(facecolor="#f4d03f", alpha=0.45, edgecolor="#b7950b",
              label="stereo overlap"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.9)

    ax.set_title("(b)  GOES-19 + GOES-18 imagery and stereo overlap",
                 fontsize=13, fontweight="bold", loc="left")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", default="2025-07-01T18:00",
                        help="UTC time (ISO) for the imagery panel.")
    parser.add_argument("--cache-dir", default="./cache",
                        help="Cache directory for downloaded ABI files.")
    parser.add_argument("--out", default="fig1_parallax_concept.png",
                        help="Output PNG path.")
    parser.add_argument("--band", default="C14", help="ABI band for imagery.")
    parser.add_argument("--theta-max", type=float, default=80.0,
                        help="Zenith-angle threshold (deg) defining footprints.")
    parser.add_argument("--theta-stereo", type=float, default=None,
                        help="Optional tighter zenith threshold for the "
                             "highlighted stereo-overlap region.")
    parser.add_argument("--no-imagery", action="store_true",
                        help="Skip panel (b); render only the schematic.")
    args = parser.parse_args()

    t = dt.datetime.fromisoformat(args.time)

    if args.no_imagery:
        fig, ax_a = plt.subplots(figsize=(8, 7))
        draw_geometry_schematic(ax_a)
    else:
        import cartopy.crs as ccrs
        fig = plt.figure(figsize=(16, 7))
        ax_a = fig.add_subplot(1, 2, 1)
        draw_geometry_schematic(ax_a)
        ax_b = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        pair = load_pair(t, args.cache_dir, band=args.band)
        draw_overlap_panel(ax_b, pair, theta_max=args.theta_max,
                           theta_stereo=args.theta_stereo)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    logger.info("Saved figure to %s", args.out)


if __name__ == "__main__":
    main()
