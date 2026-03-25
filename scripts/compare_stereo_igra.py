"""Compare stereo wind retrievals with IGRA radiosonde data.

Collocates stereo winds from ArrayLake (zeus-ai/geo-stereo-winds) with
IGRA 2024 radiosonde profiles (zeus-ai/obs-xvec) for January 2024,
across all available ABI bands.

Matching follows Carr et al.: nearest sonde level by height, requiring
bracketing above/below, within 2 km (above 500 hPa) or 1 km (below).
"""

import os

import arraylake
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import sobel

from stereo_winds.config import GOES16_CONFIG
from stereo_winds.navigation import (
    geodetic_to_fixed_grid,
    scanning_angle_to_pixel,
)
from stereo_winds.validation.metrics import (
    correlation,
    height_rmse,
    rmsvd,
    speed_bias,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

BANDS = ["C04", "C08", "C09", "C10", "C12", "C14"]
MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04"]
SAT = GOES16_CONFIG

# Minimum cloud-top height (m) per band. C04 (0.47 um visible) sees
# clear-sky surface returns that masquerade as low features; require
# heights above 8 km to keep only genuine cloud signals.
MIN_HEIGHT = {"C04": 8000, "C08": 2000, "C09": 2000, "C10": 2000, "C12": 1000, "C14": 1000}

STEREO_TIMES = pd.date_range("2024-01-01", "2024-04-30", freq="12h")


def load_stereo(band, times):
    """Load stereo winds for specific times from ArrayLake."""
    client = arraylake.Client()
    repo = client.get_repo("zeus-ai/geo-stereo-winds")
    session = repo.readonly_session("main")
    datasets = []
    for month in MONTHS:
        group = f"goes16_goes18/{band}/{month}"
        ds = xr.open_zarr(session.store, group=group, consolidated=False)
        datasets.append(ds)
    combined = xr.concat(datasets, dim="time")
    avail = combined.time.values
    use = np.isin(avail, times.values)
    return combined.isel(time=use)


def load_igra(times):
    """Load IGRA 2024 data at the given times."""
    client = arraylake.Client()
    repo = client.get_repo("zeus-ai/obs-xvec")
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, group="igra/2024", consolidated=False)
    return ds.sel(time=times)


def station_pixel_coords(lats, lons):
    """Convert station lat/lon to GOES pixel row/col.

    Returns (rows, cols, valid_mask) where valid_mask is True for
    stations visible on the GOES-16 grid.
    """
    x_fg, y_fg = geodetic_to_fixed_grid(lats, lons, SAT)
    cols_f, rows_f = scanning_angle_to_pixel(x_fg, y_fg, SAT)

    on_grid = np.isfinite(cols_f) & np.isfinite(rows_f)
    cols_i = np.where(on_grid, np.round(cols_f).astype(int), -1)
    rows_i = np.where(on_grid, np.round(rows_f).astype(int), -1)
    on_grid &= (rows_i >= 0) & (rows_i < 5424) & (cols_i >= 0) & (cols_i < 5424)

    return rows_i, cols_i, on_grid


def _height_gradient(h_2d):
    """Sobel height gradient magnitude (m/pixel) on a 2D field."""
    h_filled = np.where(np.isfinite(h_2d), h_2d, 0.0)
    gx = sobel(h_filled, axis=1)
    gy = sobel(h_filled, axis=0)
    return np.sqrt(gx**2 + gy**2) / 8.0


def collocate(stereo_ds, igra_ds, min_height=0, box_half=2,
              sigma_h_max_low=1000, sigma_h_max_mid=2000,
              sigma_h_max_high=1000,
              h_grad_max=3000, chi2_max=0.2):
    """Collocate stereo winds with IGRA profiles across all shared times.

    Uses a (2*box_half+1)^2 neighborhood median and height-gradient
    filtering to reduce single-pixel noise and cloud-edge contamination.

    Parameters
    ----------
    stereo_ds : xr.Dataset with dims (time, y, x)
    igra_ds : xr.Dataset with dims (pressure, time, geometry)
    min_height : float
        Minimum cloud top height (m) to accept.
    box_half : int
        Half-width of the neighborhood box (default 2 -> 5x5).
    sigma_h_max : float
        Maximum formal height uncertainty (m).
    h_grad_max : float
        Maximum height gradient (m/pixel) to reject cloud edges.
    chi2_max : float
        Maximum chi-squared residual.

    Returns
    -------
    dict of matched arrays
    """
    empty = {k: np.array([]) for k in [
        "u_stereo", "v_stereo", "h_stereo",
        "u_sonde", "v_sonde", "h_sonde",
        "lat", "lon", "pressure_matched",
        "h_diff", "sigma_h_matched", "chi2_matched",
        "shear",
    ]}

    lats = igra_ds["lat"].values
    lons = igra_ds["lon"].values
    rows, cols, on_grid = station_pixel_coords(lats, lons)
    grid_idx = np.where(on_grid)[0]

    if len(grid_idx) == 0:
        return empty

    r, c = rows[grid_idx], cols[grid_idx]
    n_times = len(stereo_ds.time)
    n_sta = len(grid_idx)
    pressure = igra_ds["pressure"].values

    # ----- Neighborhood median extraction -----
    # Pixel offsets for the box
    offsets = np.arange(-box_half, box_half + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()  # (n_neighbors,)
    n_nb = len(dy)

    # Neighbor row/col for each station: (n_sta, n_neighbors)
    r_nb = np.clip(r[:, None] + dy[None, :], 0, 5423)
    c_nb = np.clip(c[:, None] + dx[None, :], 0, 5423)

    # Load full grids
    u_grid = stereo_ds["u_wind"].values       # (n_times, 5424, 5424)
    v_grid = stereo_ds["v_wind"].values
    h_grid = stereo_ds["cloud_top_height"].values
    chi2_grid = stereo_ds["chi_squared"].values
    qf_grid = stereo_ds["quality_flag"].values
    sigh_grid = stereo_ds["sigma_h"].values

    # Compute height gradient per time step
    hgrad_grid = np.stack([_height_gradient(h_grid[t]) for t in range(n_times)])

    # Extract neighborhoods: (n_times, n_sta, n_neighbors)
    u_nb = u_grid[:, r_nb, c_nb]
    v_nb = v_grid[:, r_nb, c_nb]
    h_nb = h_grid[:, r_nb, c_nb]
    chi2_nb = chi2_grid[:, r_nb, c_nb]
    qf_nb = qf_grid[:, r_nb, c_nb]
    sigh_nb = sigh_grid[:, r_nb, c_nb]
    hgrad_nb = hgrad_grid[:, r_nb, c_nb]

    # Per-pixel QA mask within neighborhoods
    spd_nb = np.sqrt(u_nb**2 + v_nb**2)
    sigh_thresh = np.where(
        h_nb < 3000, sigma_h_max_low,
        np.where(h_nb < 7000, sigma_h_max_mid, sigma_h_max_high),
    )
    qa_nb = (
        np.isfinite(u_nb) & np.isfinite(h_nb)
        & (qf_nb > 0.5)
        & (chi2_nb <= chi2_max)
        & (sigh_nb <= sigh_thresh)
        & (hgrad_nb <= h_grad_max)
        & (spd_nb <= 100)
        & (h_nb >= min_height) & (h_nb <= 20000)
    )

    # Mask out bad pixels, then nanmedian
    u_nb = np.where(qa_nb, u_nb, np.nan)
    v_nb = np.where(qa_nb, v_nb, np.nan)
    h_nb = np.where(qa_nb, h_nb, np.nan)
    chi2_nb = np.where(qa_nb, chi2_nb, np.nan)
    sigh_nb = np.where(qa_nb, sigh_nb, np.nan)

    with np.errstate(all="ignore"):
        u_s = np.nanmedian(u_nb, axis=2)      # (n_times, n_sta)
        v_s = np.nanmedian(v_nb, axis=2)
        h_s = np.nanmedian(h_nb, axis=2)
        chi2_s = np.nanmedian(chi2_nb, axis=2)
        sigh_s = np.nanmedian(sigh_nb, axis=2)

    # Count valid neighbors — require at least 3 of 25
    n_valid = np.sum(qa_nb, axis=2)
    qa = np.isfinite(u_s) & np.isfinite(h_s) & (n_valid >= 3)

    # ----- IGRA matching (same as before) -----
    sonde_u = igra_ds["u"].values[:, :, grid_idx]      # (n_plevs, n_times, n_sta)
    sonde_v = igra_ds["v"].values[:, :, grid_idx]
    sonde_h = igra_ds["geopotential_height"].values[:, :, grid_idx]

    sonde_valid = (
        np.isfinite(sonde_h)
        & np.isfinite(sonde_u)
        & np.isfinite(sonde_v)
    )

    h_diff_all = np.abs(sonde_h - h_s[np.newaxis, :, :])
    h_diff_all = np.where(sonde_valid, h_diff_all, np.inf)

    best_lev = np.argmin(h_diff_all, axis=0)
    ti, si = np.meshgrid(np.arange(n_times), np.arange(n_sta), indexing="ij")

    u_sonde_match = sonde_u[best_lev, ti, si]
    v_sonde_match = sonde_v[best_lev, ti, si]
    h_sonde_match = sonde_h[best_lev, ti, si]
    p_sonde_match = pressure[best_lev]
    min_h_diff = h_diff_all[best_lev, ti, si]

    # Bracketing: sonde must have valid levels above AND below
    above = (sonde_h > h_s[np.newaxis, :, :]) & sonde_valid
    below = (sonde_h < h_s[np.newaxis, :, :]) & sonde_valid
    has_above = np.any(above, axis=0)
    has_below = np.any(below, axis=0)

    # Height threshold: 2 km above 500 hPa, 1 km below
    above_500 = p_sonde_match <= 250
    max_h = np.where(above_500, 2000.0, 500.0)

    keep = (
        qa
        & has_above & has_below
        & (min_h_diff <= max_h)
        & np.isfinite(u_sonde_match)
        & np.isfinite(h_sonde_match)
    )

    # ----- Wind shear at collocation level -----
    # For each (time, station), find bracketing sonde levels and compute
    # vector wind shear magnitude (m/s per km).
    n_plevs = len(pressure)
    lev_above_idx = np.full_like(best_lev, -1)
    lev_below_idx = np.full_like(best_lev, -1)

    for lev in range(n_plevs):
        is_above = (sonde_h[lev] > h_s) & np.isfinite(sonde_u[lev]) & np.isfinite(sonde_h[lev])
        closer = (lev_above_idx == -1)
        # Pick the first (lowest-altitude) valid level above
        h_cur = np.where(is_above, sonde_h[lev], np.inf)
        h_prev = np.where(lev_above_idx >= 0,
                          sonde_h[lev_above_idx, ti, si], np.inf)
        update = is_above & ((lev_above_idx == -1) | (h_cur < h_prev))
        lev_above_idx = np.where(update, lev, lev_above_idx)

        is_below = (sonde_h[lev] < h_s) & np.isfinite(sonde_u[lev]) & np.isfinite(sonde_h[lev])
        h_cur_b = np.where(is_below, sonde_h[lev], -np.inf)
        h_prev_b = np.where(lev_below_idx >= 0,
                            sonde_h[lev_below_idx, ti, si], -np.inf)
        update_b = is_below & ((lev_below_idx == -1) | (h_cur_b > h_prev_b))
        lev_below_idx = np.where(update_b, lev, lev_below_idx)

    # Extract bracketing values — default NaN if no bracket
    has_bracket = (lev_above_idx >= 0) & (lev_below_idx >= 0)
    la = np.clip(lev_above_idx, 0, n_plevs - 1)
    lb = np.clip(lev_below_idx, 0, n_plevs - 1)

    u_up = sonde_u[la, ti, si]
    v_up = sonde_v[la, ti, si]
    h_up = sonde_h[la, ti, si]
    u_lo = sonde_u[lb, ti, si]
    v_lo = sonde_v[lb, ti, si]
    h_lo = sonde_h[lb, ti, si]

    dz_km = (h_up - h_lo) / 1000.0
    dz_km = np.where(dz_km > 0, dz_km, np.nan)
    shear_mag = np.sqrt((u_up - u_lo)**2 + (v_up - v_lo)**2) / dz_km
    shear_mag = np.where(has_bracket, shear_mag, np.nan)

    kt, ks = np.where(keep)

    if len(kt) == 0:
        return empty

    return {
        "u_stereo": u_s[kt, ks],
        "v_stereo": v_s[kt, ks],
        "h_stereo": h_s[kt, ks],
        "u_sonde": u_sonde_match[kt, ks],
        "v_sonde": v_sonde_match[kt, ks],
        "h_sonde": h_sonde_match[kt, ks],
        "lat": lats[grid_idx[ks]],
        "lon": lons[grid_idx[ks]],
        "pressure_matched": p_sonde_match[kt, ks],
        "h_diff": (h_s[kt, ks] - h_sonde_match[kt, ks]),
        "sigma_h_matched": sigh_s[kt, ks],
        "chi2_matched": chi2_s[kt, ks],
        "shear": shear_mag[kt, ks],
    }


def compute_stats(matches):
    """Compute validation statistics from matched data."""
    if len(matches["u_stereo"]) == 0:
        return {
            "n_matches": 0, "rmsvd": np.nan, "speed_bias": np.nan,
            "corr_speed": np.nan, "corr_u": np.nan, "corr_v": np.nan,
            "h_rmse": np.nan, "h_bias": np.nan, "h_corr": np.nan,
        }
    spd_s = np.sqrt(matches["u_stereo"]**2 + matches["v_stereo"]**2)
    spd_r = np.sqrt(matches["u_sonde"]**2 + matches["v_sonde"]**2)
    return {
        "n_matches": len(matches["u_stereo"]),
        "rmsvd": rmsvd(matches["u_stereo"], matches["v_stereo"],
                        matches["u_sonde"], matches["v_sonde"]),
        "speed_bias": speed_bias(matches["u_stereo"], matches["v_stereo"],
                                  matches["u_sonde"], matches["v_sonde"]),
        "corr_speed": correlation(spd_s, spd_r),
        "corr_u": correlation(matches["u_stereo"], matches["u_sonde"]),
        "corr_v": correlation(matches["v_stereo"], matches["v_sonde"]),
        "h_rmse": height_rmse(matches["h_stereo"], matches["h_sonde"]),
        "h_bias": float(np.mean(matches["h_stereo"] - matches["h_sonde"])),
        "h_corr": correlation(matches["h_stereo"], matches["h_sonde"]),
    }


# ---------------------------------------------------------------------------
# Existing plots
# ---------------------------------------------------------------------------


def plot_wind_scatter(all_matches, all_stats):
    """Stereo vs sonde wind speed scatter, per band."""
    active = [(b, m) for b, m in all_matches.items() if len(m["u_stereo"]) > 0]
    if not active:
        return

    n = len(active)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    sc = None
    for i, (band, matches) in enumerate(active):
        ax = axes_flat[i]
        spd_stereo = np.sqrt(matches["u_stereo"]**2 + matches["v_stereo"]**2)
        spd_sonde = np.sqrt(matches["u_sonde"]**2 + matches["v_sonde"]**2)
        h_km = matches["h_stereo"] / 1000.0

        sc = ax.scatter(spd_sonde, spd_stereo, c=h_km, s=15, alpha=0.6,
                        cmap="turbo", vmin=0, vmax=16)
        lim = max(spd_sonde.max(), spd_stereo.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Sonde speed (m/s)")
        ax.set_ylabel("Stereo speed (m/s)")

        stats = all_stats[band]
        ax.set_title(
            f"{band}  n={stats['n_matches']}  "
            f"RMSVD={stats['rmsvd']:.1f}  r={stats['corr_speed']:.2f}"
        )

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cax, label="Height (km)")
    fig.suptitle(f"Stereo vs Sonde Wind Speed — {MONTHS[0]} to {MONTHS[-1]}", fontsize=14)

    path = os.path.join(FIGURES_DIR, "stereo_igra_wind_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_uv_scatter(all_matches):
    """U and V component scatter for C14."""
    m = all_matches.get("C14", {})
    if len(m.get("u_stereo", [])) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    h_km = m["h_stereo"] / 1000.0

    for ax, comp_s, comp_r, label in [
        (axes[0], m["u_stereo"], m["u_sonde"], "U wind (m/s)"),
        (axes[1], m["v_stereo"], m["v_sonde"], "V wind (m/s)"),
    ]:
        sc = ax.scatter(comp_r, comp_s, c=h_km, s=15, alpha=0.6,
                        cmap="turbo", vmin=0, vmax=16)
        lim = max(abs(comp_r).max(), abs(comp_s).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Sonde {label}")
        ax.set_ylabel(f"Stereo {label}")
        r_val = correlation(comp_s, comp_r)
        ax.set_title(f"{label}  r={r_val:.2f}")

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cax, label="Height (km)")
    fig.suptitle(f"Stereo vs Sonde Wind Components (C14) — {MONTHS[0]} to {MONTHS[-1]}", fontsize=13)

    path = os.path.join(FIGURES_DIR, "stereo_igra_uv_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_station_map(all_matches):
    """Station map colored by vector wind difference for C14."""
    m = all_matches.get("C14", {})
    if len(m.get("u_stereo", [])) == 0:
        return

    vd = np.sqrt((m["u_stereo"] - m["u_sonde"])**2 +
                 (m["v_stereo"] - m["v_sonde"])**2)

    fig, ax = plt.subplots(figsize=(12, 7),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([-170, 10, -60, 70], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")

    sc = ax.scatter(m["lon"], m["lat"], c=vd, s=25, cmap="YlOrRd",
                    vmin=0, vmax=20, transform=ccrs.PlateCarree(),
                    edgecolors="k", linewidths=0.3, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Vector wind difference (m/s)")

    ax.set_title(f"Stereo-Sonde Wind Difference (C14) — {MONTHS[0]} to {MONTHS[-1]}")
    ax.gridlines(draw_labels=True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "stereo_igra_station_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_vertical_stats(all_matches):
    """RMSVD, speed bias, and collocation count binned by pressure."""
    pressure_bins = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100]
    bin_centers = [(pressure_bins[i] + pressure_bins[i+1]) / 2
                   for i in range(len(pressure_bins) - 1)]
    n_bins = len(bin_centers)

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    ax_rmsvd, ax_bias, ax_count = axes

    for band in BANDS:
        m = all_matches.get(band, {})
        if len(m.get("u_stereo", [])) == 0:
            continue

        p_matched = m["pressure_matched"]
        rmsvd_binned = []
        bias_binned = []
        count_binned = []

        for j in range(n_bins):
            mask = (p_matched >= pressure_bins[j+1]) & (p_matched < pressure_bins[j])
            n = mask.sum()
            count_binned.append(n)
            if n >= 3:
                rmsvd_binned.append(
                    rmsvd(m["u_stereo"][mask], m["v_stereo"][mask],
                          m["u_sonde"][mask], m["v_sonde"][mask])
                )
                bias_binned.append(
                    speed_bias(m["u_stereo"][mask], m["v_stereo"][mask],
                               m["u_sonde"][mask], m["v_sonde"][mask])
                )
            else:
                rmsvd_binned.append(np.nan)
                bias_binned.append(np.nan)

        ax_rmsvd.plot(rmsvd_binned, bin_centers, "o-", label=band, markersize=5)
        ax_bias.plot(bias_binned, bin_centers, "o-", label=band, markersize=5)
        ax_count.plot(count_binned, bin_centers, "o-", label=band, markersize=5)

    # RMSVD panel
    ax_rmsvd.set_ylabel("Pressure (hPa)")
    ax_rmsvd.set_xlabel("RMSVD (m/s)")
    ax_rmsvd.set_yscale("log")
    ax_rmsvd.invert_yaxis()
    ax_rmsvd.set_ylim(1050, 90)
    ax_rmsvd.grid(alpha=0.3)
    ax_rmsvd.set_title("RMSVD")
    ax_rmsvd.legend(fontsize=8)

    # Bias panel
    ax_bias.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_bias.set_xlabel("Speed Bias (m/s)")
    ax_bias.grid(alpha=0.3)
    ax_bias.set_title("Speed Bias (stereo - sonde)")

    # Count panel
    ax_count.set_xlabel("Number of collocations")
    ax_count.grid(alpha=0.3)
    ax_count.set_title("Collocation Count")

    fig.suptitle(f"Vertical Validation Statistics — {MONTHS[0]} to {MONTHS[-1]}", fontsize=14)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "stereo_igra_vertical_stats.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# New diagnostic plots
# ---------------------------------------------------------------------------


def plot_height_scatter(all_matches, all_stats):
    """Stereo vs sonde height scatter, per band."""
    active = [(b, m) for b, m in all_matches.items() if len(m["u_stereo"]) > 0]
    if not active:
        return

    n = len(active)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    sc = None
    for i, (band, matches) in enumerate(active):
        ax = axes_flat[i]
        h_st = matches["h_stereo"] / 1000.0
        h_so = matches["h_sonde"] / 1000.0
        vd = np.sqrt((matches["u_stereo"] - matches["u_sonde"])**2 +
                      (matches["v_stereo"] - matches["v_sonde"])**2)

        sc = ax.scatter(h_so, h_st, c=vd, s=15, alpha=0.6,
                        cmap="YlOrRd", vmin=0, vmax=20)
        lim = max(h_so.max(), h_st.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Sonde height (km)")
        ax.set_ylabel("Stereo height (km)")

        stats = all_stats[band]
        ax.set_title(
            f"{band}  n={stats['n_matches']}  "
            f"hRMSE={stats['h_rmse']/1000:.1f}km  "
            f"hBias={stats['h_bias']/1000:+.1f}km  "
            f"r={stats['h_corr']:.2f}"
        )

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cax, label="Wind vector diff (m/s)")
    fig.suptitle(f"Stereo vs Sonde Height — {MONTHS[0]} to {MONTHS[-1]}", fontsize=14)

    path = os.path.join(FIGURES_DIR, "stereo_igra_height_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_height_diff_hist(all_matches):
    """Histogram of (h_stereo - h_sonde) per band."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for band in BANDS:
        m = all_matches.get(band, {})
        if len(m.get("h_diff", [])) == 0:
            continue
        hd_km = m["h_diff"] / 1000.0
        ax.hist(hd_km, bins=40, range=(-5, 5), alpha=0.4, label=f"{band} ({len(hd_km)})")
        ax.axvline(np.mean(hd_km), linestyle="--", linewidth=0.8)

    ax.set_xlabel("Height difference: stereo - sonde (km)")
    ax.set_ylabel("Count")
    ax.set_title(f"Height Assignment Error Distribution — {MONTHS[0]} to {MONTHS[-1]}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    path = os.path.join(FIGURES_DIR, "stereo_igra_height_diff_hist.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_chi2_sensitivity(stereo_ds_c14, igra_ds, min_height):
    """RMSVD and N vs chi2_max threshold for C14."""
    chi2_vals = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
    rmsvd_list = []
    n_list = []

    for c2 in chi2_vals:
        m = collocate(stereo_ds_c14, igra_ds, min_height=min_height, chi2_max=c2)
        s = compute_stats(m)
        rmsvd_list.append(s["rmsvd"])
        n_list.append(s["n_matches"])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1, color2 = "tab:blue", "tab:red"

    ax1.plot(chi2_vals, rmsvd_list, "o-", color=color1, markersize=6)
    ax1.set_xlabel("chi2_max threshold")
    ax1.set_ylabel("RMSVD (m/s)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(chi2_vals, n_list, "s--", color=color2, markersize=6)
    ax2.set_ylabel("N matches", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle(f"Chi-squared Sensitivity (C14) — {MONTHS[0]} to {MONTHS[-1]}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "stereo_igra_chi2_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_layer_bars(all_matches):
    """Grouped bar chart of RMSVD by band and tropospheric layer."""
    layers = [
        ("Low (>=700)", lambda p: p >= 700),
        ("Mid (400-700)", lambda p: (p >= 400) & (p < 700)),
        ("High (<400)", lambda p: p < 400),
    ]
    colors = ["#4c72b0", "#dd8452", "#55a868"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(BANDS))
    width = 0.25

    for i, (label, mask_fn) in enumerate(layers):
        vals = []
        for band in BANDS:
            m = all_matches.get(band, {})
            p = m.get("pressure_matched", np.array([]))
            if len(p) == 0:
                vals.append(0)
                continue
            sel = mask_fn(p)
            if sel.sum() >= 3:
                vals.append(rmsvd(m["u_stereo"][sel], m["v_stereo"][sel],
                                  m["u_sonde"][sel], m["v_sonde"][sel]))
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=label, color=colors[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(BANDS)
    ax.set_ylabel("RMSVD (m/s)")
    ax.set_title(f"RMSVD by Band and Tropospheric Layer — {MONTHS[0]} to {MONTHS[-1]}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(FIGURES_DIR, "stereo_igra_layer_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_bias_vs_shear(all_matches):
    """Speed bias vs wind shear at collocation level for C14."""
    m = all_matches.get("C14", {})
    shear = m.get("shear", np.array([]))
    if len(shear) == 0:
        return

    spd_s = np.sqrt(m["u_stereo"]**2 + m["v_stereo"]**2)
    spd_r = np.sqrt(m["u_sonde"]**2 + m["v_sonde"]**2)
    spd_err = spd_s - spd_r
    valid = np.isfinite(shear)

    if valid.sum() < 10:
        return

    shear_v = shear[valid]
    err_v = spd_err[valid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter
    ax = axes[0]
    ax.scatter(shear_v, err_v, s=10, alpha=0.4, c="steelblue")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Wind shear (m/s per km)")
    ax.set_ylabel("Speed error: stereo - sonde (m/s)")
    ax.set_title("Speed Error vs Local Wind Shear (C14)")
    ax.set_xlim(0, min(shear_v.max() * 1.05, 30))
    ax.grid(alpha=0.3)

    # Right: binned RMSVD and bias vs shear
    ax2 = axes[1]
    shear_edges = np.arange(0, 22, 3)
    bin_centers = (shear_edges[:-1] + shear_edges[1:]) / 2
    rmsvd_binned = []
    bias_binned = []
    count_binned = []

    for j in range(len(shear_edges) - 1):
        sel = valid.copy()
        sel &= (shear >= shear_edges[j]) & (shear < shear_edges[j + 1])
        n = sel.sum()
        count_binned.append(n)
        if n >= 5:
            rmsvd_binned.append(
                rmsvd(m["u_stereo"][sel], m["v_stereo"][sel],
                      m["u_sonde"][sel], m["v_sonde"][sel]))
            bias_binned.append(
                speed_bias(m["u_stereo"][sel], m["v_stereo"][sel],
                           m["u_sonde"][sel], m["v_sonde"][sel]))
        else:
            rmsvd_binned.append(np.nan)
            bias_binned.append(np.nan)

    ax2.plot(bin_centers, rmsvd_binned, "o-", color="tab:blue", label="RMSVD")
    ax2.plot(bin_centers, bias_binned, "s--", color="tab:red", label="Speed bias")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Wind shear (m/s per km)")
    ax2.set_ylabel("m/s")
    ax2.set_title("Binned RMSVD & Bias vs Shear (C14)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Annotate counts
    for j, (xc, n) in enumerate(zip(bin_centers, count_binned)):
        ax2.annotate(f"n={n}", (xc, rmsvd_binned[j] if np.isfinite(rmsvd_binned[j]) else 0),
                     fontsize=7, ha="center", va="bottom")

    fig.suptitle(f"Wind Shear Impact on Stereo Accuracy (C14) — {MONTHS[0]} to {MONTHS[-1]}",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "stereo_igra_bias_vs_shear.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


LAYERS = [
    ("Low (>=700 hPa)", lambda p: p >= 700),
    ("Mid (400-700 hPa)", lambda p: (p >= 400) & (p < 700)),
    ("High (<400 hPa)", lambda p: p < 400),
]


def print_layer_summary(all_matches):
    """Print per-band, per-layer statistics table."""
    print("\n" + "=" * 100)
    print(f"Layer-Stratified Statistics — {MONTHS[0]} to {MONTHS[-1]}")
    print("=" * 100)
    print(f"{'Band':>6s} {'Layer':>20s} {'N':>6s} {'RMSVD':>8s} "
          f"{'SpdBias':>8s} {'r(speed)':>8s} {'hRMSE':>8s}")
    print("-" * 100)

    for band in BANDS:
        m = all_matches.get(band, {})
        p = m.get("pressure_matched", np.array([]))
        if len(p) == 0:
            continue
        for layer_name, mask_fn in LAYERS:
            sel = mask_fn(p)
            n = sel.sum()
            if n < 3:
                print(f"{band:>6s} {layer_name:>20s} {n:>6d} {'--':>8s} "
                      f"{'--':>8s} {'--':>8s} {'--':>8s}")
                continue
            rv = rmsvd(m["u_stereo"][sel], m["v_stereo"][sel],
                       m["u_sonde"][sel], m["v_sonde"][sel])
            sb = speed_bias(m["u_stereo"][sel], m["v_stereo"][sel],
                            m["u_sonde"][sel], m["v_sonde"][sel])
            spd_s = np.sqrt(m["u_stereo"][sel]**2 + m["v_stereo"][sel]**2)
            spd_r = np.sqrt(m["u_sonde"][sel]**2 + m["v_sonde"][sel]**2)
            rs = correlation(spd_s, spd_r)
            hr = height_rmse(m["h_stereo"][sel], m["h_sonde"][sel])
            print(f"{band:>6s} {layer_name:>20s} {n:>6d} {rv:>8.2f} "
                  f"{sb:>8.2f} {rs:>8.3f} {hr:>8.0f}")
    print("=" * 100)


def print_summary(all_stats):
    """Print per-band summary table."""
    print("\n" + "=" * 110)
    print(f"Stereo Winds vs IGRA Radiosondes — {MONTHS[0]} to {MONTHS[-1]}")
    print("=" * 110)
    print(f"{'Band':>6s} {'N':>6s} {'RMSVD':>8s} {'SpdBias':>8s} "
          f"{'r(speed)':>8s} {'r(u)':>8s} {'r(v)':>8s} "
          f"{'hRMSE':>8s} {'hBias':>8s} {'r(h)':>8s}")
    print("-" * 110)
    for band in BANDS:
        s = all_stats.get(band, {})
        if s.get("n_matches", 0) == 0:
            print(f"{band:>6s} {'0':>6s}" + " {:>8s}".format("--") * 8)
        else:
            print(f"{band:>6s} {s['n_matches']:>6d} {s['rmsvd']:>8.2f} "
                  f"{s['speed_bias']:>8.2f} {s['corr_speed']:>8.3f} "
                  f"{s['corr_u']:>8.3f} {s['corr_v']:>8.3f} "
                  f"{s['h_rmse']:>8.0f} {s['h_bias']:>8.0f} "
                  f"{s['h_corr']:>8.3f}")
    print("=" * 110)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Load stereo C14 to find which times have data
    print("Finding filled stereo time steps...")
    ref_ds = load_stereo("C14", STEREO_TIMES)
    center = ref_ds["u_wind"].isel(y=2712, x=2712).values
    times = ref_ds.time.values[np.isfinite(center)]
    times = pd.DatetimeIndex(times)
    print(f"  {len(times)} filled time steps")

    # Load IGRA once for all bands
    print("Loading IGRA data...")
    igra_ds = load_igra(times)
    igra_ds.load()
    print(f"  IGRA: {dict(igra_ds.sizes)}")

    all_matches = {}
    all_stats = {}
    stereo_ds_c14 = None

    for band in BANDS:
        print(f"\nProcessing {band}...")
        stereo_ds = load_stereo(band, times)
        stereo_ds.load()
        matches = collocate(stereo_ds, igra_ds, min_height=MIN_HEIGHT.get(band, 0))
        stats = compute_stats(matches)
        all_matches[band] = matches
        all_stats[band] = stats
        if band == "C14":
            stereo_ds_c14 = stereo_ds
        print(f"  {band}: {stats['n_matches']} matches, RMSVD={stats.get('rmsvd', 'N/A')}")

    print_summary(all_stats)
    print_layer_summary(all_matches)

    print("\nGenerating plots...")
    plot_wind_scatter(all_matches, all_stats)
    plot_uv_scatter(all_matches)
    plot_station_map(all_matches)
    plot_vertical_stats(all_matches)
    plot_height_scatter(all_matches, all_stats)
    plot_height_diff_hist(all_matches)
    plot_layer_bars(all_matches)
    plot_bias_vs_shear(all_matches)

    if stereo_ds_c14 is not None:
        print("\nRunning chi2 sensitivity analysis (C14)...")
        plot_chi2_sensitivity(stereo_ds_c14, igra_ds, MIN_HEIGHT.get("C14", 0))

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
