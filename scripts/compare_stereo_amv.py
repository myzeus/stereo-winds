"""Compare stereo wind retrievals with GOES-16 operational AMVs (DMW).

Collocates dense stereo winds from ArrayLake (zeus-ai/geo-stereo-winds) with
GOES-16 ABI L2 Derived Motion Winds (DMW) from the NOAA public S3 bucket
for January–April 2024, across matching ABI bands (C08, C09, C10, C14).

DMW data is sparse (one wind per target box), while stereo winds are dense
(every pixel). Collocation uses a 5×5 neighborhood median at each AMV
target location, with the same QA filtering as the IGRA comparison.
"""

import logging
import os
import warnings

import arraylake
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Bands present in both stereo and DMW products
BANDS = ["C08", "C09", "C10", "C14"]
MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04"]
SAT = GOES16_CONFIG

MIN_HEIGHT = {"C08": 2000, "C09": 2000, "C10": 2000, "C14": 1000}

STEREO_TIMES = pd.date_range("2024-01-01", "2024-04-30", freq="12h")

S3_BUCKET = "noaa-goes16"
DMW_PRODUCT = "ABI-L2-DMWF"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def pressure_to_height(p_hpa):
    """Convert pressure (hPa) to geopotential height (m) via ICAO standard atmosphere.

    Troposphere (p >= 226.32 hPa, h <= 11 km):
        h = 44330.77 * (1 - (p/1013.25)^0.190263)
    Lower stratosphere (p < 226.32 hPa):
        h = 11000 - 6341.62 * ln(p/226.32)
    """
    p = np.asarray(p_hpa, dtype=np.float64)
    tropo = p >= 226.32
    h = np.where(
        tropo,
        44330.77 * (1.0 - (p / 1013.25) ** 0.190263),
        11000.0 - 6341.62 * np.log(np.clip(p, 1e-6, None) / 226.32),
    )
    return h.astype(np.float32)


def dmw_wind_components(speed, direction):
    """Convert meteorological wind speed/direction to u/v components.

    Meteorological convention: direction is where the wind comes FROM,
    measured clockwise from north.
    """
    dir_rad = np.deg2rad(direction)
    u = -speed * np.sin(dir_rad)
    v = -speed * np.cos(dir_rad)
    return u, v


def _height_gradient(h_2d):
    """Sobel height gradient magnitude (m/pixel) on a 2D field."""
    h_filled = np.where(np.isfinite(h_2d), h_2d, 0.0)
    gx = sobel(h_filled, axis=1)
    gy = sobel(h_filled, axis=0)
    return np.sqrt(gx**2 + gy**2) / 8.0


# ---------------------------------------------------------------------------
# DMW loading from S3
# ---------------------------------------------------------------------------


def _init_s3():
    """Create an anonymous S3 filesystem for the NOAA public bucket."""
    return s3fs.S3FileSystem(anon=True)


def find_dmw_file(fs, time, band):
    """Find the DMW file on S3 closest to the given time and band.

    Returns the S3 path or None if not found.
    """
    doy = time.strftime("%j")
    year = time.strftime("%Y")
    hour = time.strftime("%H")
    prefix = f"{S3_BUCKET}/{DMW_PRODUCT}/{year}/{doy}/{hour}/"
    band_num = band[1:]  # "C14" -> "14"

    try:
        files = fs.glob(f"{prefix}OR_{DMW_PRODUCT}-M6C{band_num}_G16_s*")
    except Exception:
        return None

    if not files:
        return None

    # If multiple files in the hour, pick the one closest to target time
    if len(files) == 1:
        return files[0]

    # Parse start times from filenames and find closest
    target_ts = time.timestamp()
    best_file = None
    best_diff = float("inf")
    for f in files:
        fname = os.path.basename(f)
        # Extract start time: s20240010000205 -> 2024-001-00:00:20.5
        s_idx = fname.index("_s") + 2
        s_str = fname[s_idx : s_idx + 11]  # YYYYJJJHHMM
        try:
            ft = pd.Timestamp.strptime(s_str, "%Y%j%H%M").timestamp()
        except Exception:
            continue
        diff = abs(ft - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_file = f
    return best_file


def load_dmw(fs, s3_path):
    """Load a single DMW NetCDF file from S3.

    Filters to DQF==0, converts speed/direction to u/v, pressure to height.
    Returns dict of 1D arrays or None on failure.
    """
    try:
        with fs.open(s3_path, "rb") as f:
            ds = xr.open_dataset(f, engine="h5netcdf")

            dqf = ds["DQF"].values
            good = dqf == 0

            if good.sum() == 0:
                ds.close()
                return None

            lat = ds["lat"].values[good]
            lon = ds["lon"].values[good]
            spd = ds["wind_speed"].values[good]
            wdir = ds["wind_direction"].values[good]
            pres = ds["pressure"].values[good]

            ds.close()

        # Filter out non-finite values
        valid = np.isfinite(spd) & np.isfinite(wdir) & np.isfinite(pres) & np.isfinite(lat)
        if valid.sum() == 0:
            return None

        lat, lon = lat[valid], lon[valid]
        spd, wdir, pres = spd[valid], wdir[valid], pres[valid]

        u, v = dmw_wind_components(spd, wdir)
        h = pressure_to_height(pres)

        return dict(lat=lat, lon=lon, u=u, v=v, h=h, pressure=pres, speed=spd)

    except Exception as e:
        log.warning(f"  Failed to load {s3_path}: {e}")
        return None


def load_dmw_for_times(times, band):
    """Load and concatenate DMW data for all times and a single band.

    Returns dict with concatenated arrays and time_idx (index into `times`).
    """
    fs = _init_s3()
    arrays = {k: [] for k in ["lat", "lon", "u", "v", "h", "pressure", "speed"]}
    time_idx_list = []
    n_loaded = 0

    for i, t in enumerate(times):
        t_pd = pd.Timestamp(t)
        path = find_dmw_file(fs, t_pd, band)
        if path is None:
            continue

        data = load_dmw(fs, path)
        if data is None:
            continue

        for k in arrays:
            arrays[k].append(data[k])
        time_idx_list.append(np.full(len(data["lat"]), i, dtype=np.int32))
        n_loaded += 1

    if n_loaded == 0:
        return None

    result = {k: np.concatenate(v) for k, v in arrays.items()}
    result["time_idx"] = np.concatenate(time_idx_list)
    log.info(f"  Loaded DMW: {n_loaded} files, {len(result['lat']):,} good targets")
    return result


# ---------------------------------------------------------------------------
# Stereo loading (from ArrayLake, same as IGRA script)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Collocation
# ---------------------------------------------------------------------------


def collocate(stereo_ds, amv_data, min_height=0, box_half=2,
              sigma_h_max_low=1000, sigma_h_max_mid=2000,
              sigma_h_max_high=1000,
              h_grad_max=3000, chi2_max=0.2,
              max_h_diff=3000):
    """Collocate stereo winds with AMV targets across all shared times.

    Processes per-timestep: for each stereo time, finds AMV targets at that
    time, maps their lat/lon to pixel coordinates, extracts 5×5 neighborhood
    medians from the stereo grid, and accumulates matched pairs.

    Parameters
    ----------
    stereo_ds : xr.Dataset with dims (time, y, x)
    amv_data : dict from load_dmw_for_times() with lat, lon, u, v, h,
               pressure, time_idx
    min_height : minimum cloud-top height (m) to accept
    box_half : half-width of neighborhood box (default 2 -> 5x5)
    sigma_h_max : max formal height uncertainty (m)
    h_grad_max : max height gradient (m/pixel)
    chi2_max : max chi-squared residual
    max_h_diff : max height difference (m) between stereo and AMV

    Returns
    -------
    dict with matched arrays
    """
    keys = [
        "u_stereo", "v_stereo", "h_stereo",
        "u_amv", "v_amv", "h_amv",
        "lat", "lon", "pressure",
        "h_diff", "sigma_h_matched", "chi2_matched",
    ]
    accum = {k: [] for k in keys}

    n_times = len(stereo_ds.time)
    time_idx = amv_data["time_idx"]

    # Pixel offsets for neighborhood box
    offsets = np.arange(-box_half, box_half + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()

    # Load full grids into memory
    u_grid = stereo_ds["u_wind"].values       # (n_times, 5424, 5424)
    v_grid = stereo_ds["v_wind"].values
    h_grid = stereo_ds["cloud_top_height"].values
    chi2_grid = stereo_ds["chi_squared"].values
    qf_grid = stereo_ds["quality_flag"].values
    sigh_grid = stereo_ds["sigma_h"].values

    for t in range(n_times):
        # AMV targets for this timestep
        mask_t = time_idx == t
        n_amv = mask_t.sum()
        if n_amv == 0:
            continue

        lat_t = amv_data["lat"][mask_t]
        lon_t = amv_data["lon"][mask_t]
        u_amv_t = amv_data["u"][mask_t]
        v_amv_t = amv_data["v"][mask_t]
        h_amv_t = amv_data["h"][mask_t]
        p_amv_t = amv_data["pressure"][mask_t]

        # Convert AMV lat/lon to GOES pixel coordinates
        x_fg, y_fg = geodetic_to_fixed_grid(lat_t, lon_t, SAT)
        cols_f, rows_f = scanning_angle_to_pixel(x_fg, y_fg, SAT)

        on_grid = np.isfinite(cols_f) & np.isfinite(rows_f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cols_i = np.where(on_grid, np.round(cols_f).astype(int), -1)
            rows_i = np.where(on_grid, np.round(rows_f).astype(int), -1)
        on_grid &= (rows_i >= 0) & (rows_i < 5424) & (cols_i >= 0) & (cols_i < 5424)

        idx = np.where(on_grid)[0]
        if len(idx) == 0:
            continue

        r = rows_i[idx]
        c = cols_i[idx]

        # Neighbor row/col: (n_pts, n_neighbors)
        r_nb = np.clip(r[:, None] + dy[None, :], 0, 5423)
        c_nb = np.clip(c[:, None] + dx[None, :], 0, 5423)

        # Extract neighborhoods from this time step
        u_nb = u_grid[t][r_nb, c_nb]
        v_nb = v_grid[t][r_nb, c_nb]
        h_nb = h_grid[t][r_nb, c_nb]
        chi2_nb = chi2_grid[t][r_nb, c_nb]
        qf_nb = qf_grid[t][r_nb, c_nb]
        sigh_nb = sigh_grid[t][r_nb, c_nb]

        # Height gradient for this time step
        hgrad = _height_gradient(h_grid[t])
        hgrad_nb = hgrad[r_nb, c_nb]

        # Per-pixel QA mask — height-dependent sigma_h threshold
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

        # Mask and nanmedian
        u_nb = np.where(qa_nb, u_nb, np.nan)
        v_nb = np.where(qa_nb, v_nb, np.nan)
        h_nb = np.where(qa_nb, h_nb, np.nan)
        chi2_nb = np.where(qa_nb, chi2_nb, np.nan)
        sigh_nb = np.where(qa_nb, sigh_nb, np.nan)

        with np.errstate(all="ignore"):
            u_s = np.nanmedian(u_nb, axis=1)
            v_s = np.nanmedian(v_nb, axis=1)
            h_s = np.nanmedian(h_nb, axis=1)
            chi2_s = np.nanmedian(chi2_nb, axis=1)
            sigh_s = np.nanmedian(sigh_nb, axis=1)

        n_valid = np.sum(qa_nb, axis=1)
        qa = np.isfinite(u_s) & np.isfinite(h_s) & (n_valid >= 3)

        # Height difference filter
        hdiff = h_s - h_amv_t[idx]
        keep = qa & (np.abs(hdiff) <= max_h_diff)

        ki = np.where(keep)[0]
        if len(ki) == 0:
            continue

        accum["u_stereo"].append(u_s[ki])
        accum["v_stereo"].append(v_s[ki])
        accum["h_stereo"].append(h_s[ki])
        accum["u_amv"].append(u_amv_t[idx[ki]])
        accum["v_amv"].append(v_amv_t[idx[ki]])
        accum["h_amv"].append(h_amv_t[idx[ki]])
        accum["lat"].append(lat_t[idx[ki]])
        accum["lon"].append(lon_t[idx[ki]])
        accum["pressure"].append(p_amv_t[idx[ki]])
        accum["h_diff"].append(hdiff[ki])
        accum["sigma_h_matched"].append(sigh_s[ki])
        accum["chi2_matched"].append(chi2_s[ki])

    if all(len(v) == 0 for v in accum.values()):
        return {k: np.array([]) for k in keys}

    return {k: np.concatenate(v) for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(matches):
    """Compute validation statistics from matched data."""
    if len(matches["u_stereo"]) == 0:
        return {
            "n_matches": 0, "rmsvd": np.nan, "speed_bias": np.nan,
            "corr_speed": np.nan, "corr_u": np.nan, "corr_v": np.nan,
            "h_rmse": np.nan, "h_bias": np.nan, "h_corr": np.nan,
        }
    spd_s = np.sqrt(matches["u_stereo"]**2 + matches["v_stereo"]**2)
    spd_a = np.sqrt(matches["u_amv"]**2 + matches["v_amv"]**2)
    return {
        "n_matches": len(matches["u_stereo"]),
        "rmsvd": rmsvd(matches["u_stereo"], matches["v_stereo"],
                        matches["u_amv"], matches["v_amv"]),
        "speed_bias": speed_bias(matches["u_stereo"], matches["v_stereo"],
                                  matches["u_amv"], matches["v_amv"]),
        "corr_speed": correlation(spd_s, spd_a),
        "corr_u": correlation(matches["u_stereo"], matches["u_amv"]),
        "corr_v": correlation(matches["v_stereo"], matches["v_amv"]),
        "h_rmse": height_rmse(matches["h_stereo"], matches["h_amv"]),
        "h_bias": float(np.mean(matches["h_stereo"] - matches["h_amv"])),
        "h_corr": correlation(matches["h_stereo"], matches["h_amv"]),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_wind_scatter(all_matches, all_stats):
    """Stereo vs AMV wind speed scatter, per band."""
    active = [(b, m) for b, m in all_matches.items() if len(m["u_stereo"]) > 0]
    if not active:
        return

    n = len(active)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    sc = None
    for i, (band, matches) in enumerate(active):
        ax = axes_flat[i]
        spd_stereo = np.sqrt(matches["u_stereo"]**2 + matches["v_stereo"]**2)
        spd_amv = np.sqrt(matches["u_amv"]**2 + matches["v_amv"]**2)
        h_km = matches["h_stereo"] / 1000.0

        # Subsample for plotting if too many points
        n_pts = len(spd_stereo)
        if n_pts > 20000:
            idx = np.random.default_rng(42).choice(n_pts, 20000, replace=False)
            spd_stereo_p, spd_amv_p, h_km_p = spd_stereo[idx], spd_amv[idx], h_km[idx]
        else:
            spd_stereo_p, spd_amv_p, h_km_p = spd_stereo, spd_amv, h_km

        sc = ax.scatter(spd_amv_p, spd_stereo_p, c=h_km_p, s=5, alpha=0.3,
                        cmap="turbo", vmin=0, vmax=16, rasterized=True)
        lim = max(spd_amv.max(), spd_stereo.max()) * 1.05
        lim = min(lim, 100)
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("AMV speed (m/s)")
        ax.set_ylabel("Stereo speed (m/s)")

        stats = all_stats[band]
        ax.set_title(
            f"{band}  n={stats['n_matches']:,}  "
            f"RMSVD={stats['rmsvd']:.1f}  r={stats['corr_speed']:.2f}"
        )

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cax, label="Height (km)")
    fig.suptitle(f"Stereo vs AMV Wind Speed — {MONTHS[0]} to {MONTHS[-1]}", fontsize=14)

    path = os.path.join(FIGURES_DIR, "stereo_amv_wind_scatter.png")
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

    # Subsample
    n_pts = len(m["u_stereo"])
    if n_pts > 20000:
        idx = np.random.default_rng(42).choice(n_pts, 20000, replace=False)
    else:
        idx = np.arange(n_pts)

    for ax, comp_s, comp_a, label in [
        (axes[0], m["u_stereo"], m["u_amv"], "U wind (m/s)"),
        (axes[1], m["v_stereo"], m["v_amv"], "V wind (m/s)"),
    ]:
        sc = ax.scatter(comp_a[idx], comp_s[idx], c=h_km[idx], s=5, alpha=0.3,
                        cmap="turbo", vmin=0, vmax=16, rasterized=True)
        lim = max(abs(comp_a).max(), abs(comp_s).max()) * 1.1
        lim = min(lim, 80)
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"AMV {label}")
        ax.set_ylabel(f"Stereo {label}")
        r_val = correlation(comp_s, comp_a)
        ax.set_title(f"{label}  r={r_val:.2f}")

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cax, label="Height (km)")
    fig.suptitle(f"Stereo vs AMV Wind Components (C14) — {MONTHS[0]} to {MONTHS[-1]}", fontsize=13)

    path = os.path.join(FIGURES_DIR, "stereo_amv_uv_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_amv_map(all_matches):
    """Spatial map of vector wind difference for C14."""
    m = all_matches.get("C14", {})
    if len(m.get("u_stereo", [])) == 0:
        return

    vd = np.sqrt((m["u_stereo"] - m["u_amv"])**2 +
                 (m["v_stereo"] - m["v_amv"])**2)

    # Subsample for plotting
    n_pts = len(vd)
    if n_pts > 5000:
        idx = np.random.default_rng(42).choice(n_pts, 5000, replace=False)
    else:
        idx = np.arange(n_pts)

    fig, ax = plt.subplots(figsize=(12, 7),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([-170, 10, -60, 70], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")

    sc = ax.scatter(m["lon"][idx], m["lat"][idx], c=vd[idx], s=10, cmap="YlOrRd",
                    vmin=0, vmax=20, transform=ccrs.PlateCarree(),
                    edgecolors="none", alpha=0.6, rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Vector wind difference (m/s)")

    ax.set_title(f"Stereo–AMV Wind Difference (C14) — {MONTHS[0]} to {MONTHS[-1]}")
    ax.gridlines(draw_labels=True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "stereo_amv_spatial_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_vertical_stats(all_matches):
    """RMSVD, speed bias, and count binned by pressure."""
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

        p_matched = m["pressure"]
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
                          m["u_amv"][mask], m["v_amv"][mask])
                )
                bias_binned.append(
                    speed_bias(m["u_stereo"][mask], m["v_stereo"][mask],
                               m["u_amv"][mask], m["v_amv"][mask])
                )
            else:
                rmsvd_binned.append(np.nan)
                bias_binned.append(np.nan)

        ax_rmsvd.plot(rmsvd_binned, bin_centers, "o-", label=band, markersize=5)
        ax_bias.plot(bias_binned, bin_centers, "o-", label=band, markersize=5)
        ax_count.plot(count_binned, bin_centers, "o-", label=band, markersize=5)

    ax_rmsvd.set_ylabel("Pressure (hPa)")
    ax_rmsvd.set_xlabel("RMSVD (m/s)")
    ax_rmsvd.set_yscale("log")
    ax_rmsvd.invert_yaxis()
    ax_rmsvd.set_ylim(1050, 90)
    ax_rmsvd.grid(alpha=0.3)
    ax_rmsvd.set_title("RMSVD")
    ax_rmsvd.legend(fontsize=8)

    ax_bias.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_bias.set_xlabel("Speed Bias (m/s)")
    ax_bias.grid(alpha=0.3)
    ax_bias.set_title("Speed Bias (stereo - AMV)")

    ax_count.set_xlabel("Number of collocations")
    ax_count.grid(alpha=0.3)
    ax_count.set_title("Collocation Count")

    fig.suptitle(f"Vertical Validation Statistics — {MONTHS[0]} to {MONTHS[-1]}", fontsize=14)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "stereo_amv_vertical_stats.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_height_scatter(all_matches, all_stats):
    """Stereo vs AMV height scatter, per band."""
    active = [(b, m) for b, m in all_matches.items() if len(m["u_stereo"]) > 0]
    if not active:
        return

    n = len(active)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    sc = None
    for i, (band, matches) in enumerate(active):
        ax = axes_flat[i]
        h_st = matches["h_stereo"] / 1000.0
        h_am = matches["h_amv"] / 1000.0
        vd = np.sqrt((matches["u_stereo"] - matches["u_amv"])**2 +
                      (matches["v_stereo"] - matches["v_amv"])**2)

        # Subsample
        n_pts = len(h_st)
        if n_pts > 20000:
            idx = np.random.default_rng(42).choice(n_pts, 20000, replace=False)
        else:
            idx = np.arange(n_pts)

        sc = ax.scatter(h_am[idx], h_st[idx], c=vd[idx], s=5, alpha=0.3,
                        cmap="YlOrRd", vmin=0, vmax=20, rasterized=True)
        lim = max(h_am.max(), h_st.max()) * 1.05
        lim = min(lim, 18)
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("AMV height (km)")
        ax.set_ylabel("Stereo height (km)")

        stats = all_stats[band]
        ax.set_title(
            f"{band}  n={stats['n_matches']:,}  "
            f"hRMSE={stats['h_rmse']/1000:.1f}km  "
            f"hBias={stats['h_bias']/1000:+.1f}km  "
            f"r={stats['h_corr']:.2f}"
        )

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cax, label="Wind vector diff (m/s)")
    fig.suptitle(f"Stereo vs AMV Height — {MONTHS[0]} to {MONTHS[-1]}", fontsize=14)

    path = os.path.join(FIGURES_DIR, "stereo_amv_height_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_height_diff_hist(all_matches):
    """Histogram of (h_stereo - h_amv) per band."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for band in BANDS:
        m = all_matches.get(band, {})
        if len(m.get("h_diff", [])) == 0:
            continue
        hd_km = m["h_diff"] / 1000.0
        ax.hist(hd_km, bins=60, range=(-5, 5), alpha=0.4,
                label=f"{band} ({len(hd_km):,})")
        ax.axvline(np.mean(hd_km), linestyle="--", linewidth=0.8)

    ax.set_xlabel("Height difference: stereo - AMV (km)")
    ax.set_ylabel("Count")
    ax.set_title(f"Height Assignment Difference Distribution — {MONTHS[0]} to {MONTHS[-1]}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    path = os.path.join(FIGURES_DIR, "stereo_amv_height_diff_hist.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


LAYERS = [
    ("Low (>=700 hPa)", lambda p: p >= 700),
    ("Mid (400-700 hPa)", lambda p: (p >= 400) & (p < 700)),
    ("High (<400 hPa)", lambda p: p < 400),
]


def plot_layer_bars(all_matches):
    """Grouped bar chart of RMSVD by band and tropospheric layer."""
    colors = ["#4c72b0", "#dd8452", "#55a868"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(BANDS))
    width = 0.25

    for i, (label, mask_fn) in enumerate(LAYERS):
        vals = []
        for band in BANDS:
            m = all_matches.get(band, {})
            p = m.get("pressure", np.array([]))
            if len(p) == 0:
                vals.append(0)
                continue
            sel = mask_fn(p)
            if sel.sum() >= 3:
                vals.append(rmsvd(m["u_stereo"][sel], m["v_stereo"][sel],
                                  m["u_amv"][sel], m["v_amv"][sel]))
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=label, color=colors[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(BANDS)
    ax.set_ylabel("RMSVD (m/s)")
    ax.set_title(f"RMSVD by Band and Tropospheric Layer — {MONTHS[0]} to {MONTHS[-1]}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(FIGURES_DIR, "stereo_amv_layer_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_speed_dependence(all_matches):
    """RMSVD and speed bias binned by AMV wind speed for C14."""
    m = all_matches.get("C14", {})
    if len(m.get("u_amv", [])) == 0:
        return

    spd_amv = np.sqrt(m["u_amv"]**2 + m["v_amv"]**2)
    spd_stereo = np.sqrt(m["u_stereo"]**2 + m["v_stereo"]**2)
    spd_err = spd_stereo - spd_amv

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter
    ax = axes[0]
    n_pts = len(spd_amv)
    if n_pts > 20000:
        idx = np.random.default_rng(42).choice(n_pts, 20000, replace=False)
    else:
        idx = np.arange(n_pts)
    ax.scatter(spd_amv[idx], spd_err[idx], s=3, alpha=0.2, c="steelblue", rasterized=True)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("AMV wind speed (m/s)")
    ax.set_ylabel("Speed error: stereo - AMV (m/s)")
    ax.set_title("Speed Error vs AMV Wind Speed (C14)")
    ax.set_xlim(0, min(spd_amv.max() * 1.05, 80))
    ax.grid(alpha=0.3)

    # Right: binned RMSVD and bias
    ax2 = axes[1]
    spd_edges = np.arange(0, 70, 5)
    bin_centers = (spd_edges[:-1] + spd_edges[1:]) / 2
    rmsvd_binned = []
    bias_binned = []
    count_binned = []

    for j in range(len(spd_edges) - 1):
        sel = (spd_amv >= spd_edges[j]) & (spd_amv < spd_edges[j + 1])
        n = sel.sum()
        count_binned.append(n)
        if n >= 10:
            rmsvd_binned.append(
                rmsvd(m["u_stereo"][sel], m["v_stereo"][sel],
                      m["u_amv"][sel], m["v_amv"][sel]))
            bias_binned.append(
                speed_bias(m["u_stereo"][sel], m["v_stereo"][sel],
                           m["u_amv"][sel], m["v_amv"][sel]))
        else:
            rmsvd_binned.append(np.nan)
            bias_binned.append(np.nan)

    ax2.plot(bin_centers, rmsvd_binned, "o-", color="tab:blue", label="RMSVD")
    ax2.plot(bin_centers, bias_binned, "s--", color="tab:red", label="Speed bias")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("AMV wind speed (m/s)")
    ax2.set_ylabel("m/s")
    ax2.set_title("Binned RMSVD & Bias vs AMV Speed (C14)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    for j, (xc, n) in enumerate(zip(bin_centers, count_binned)):
        if np.isfinite(rmsvd_binned[j]):
            ax2.annotate(f"n={n:,}", (xc, rmsvd_binned[j]),
                         fontsize=6, ha="center", va="bottom")

    fig.suptitle(f"Speed Dependence of Stereo Accuracy (C14) — {MONTHS[0]} to {MONTHS[-1]}",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "stereo_amv_speed_dependence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_chi2_sensitivity(stereo_ds_c14, amv_data_c14, min_height):
    """RMSVD and N vs chi2_max threshold for C14."""
    chi2_vals = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
    rmsvd_list = []
    n_list = []

    for c2 in chi2_vals:
        m = collocate(stereo_ds_c14, amv_data_c14, min_height=min_height, chi2_max=c2)
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
    path = os.path.join(FIGURES_DIR, "stereo_amv_chi2_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def print_layer_summary(all_matches):
    """Print per-band, per-layer statistics table."""
    print("\n" + "=" * 100)
    print(f"Layer-Stratified Statistics — {MONTHS[0]} to {MONTHS[-1]}")
    print("=" * 100)
    print(f"{'Band':>6s} {'Layer':>20s} {'N':>8s} {'RMSVD':>8s} "
          f"{'SpdBias':>8s} {'r(speed)':>8s} {'hRMSE':>8s}")
    print("-" * 100)

    for band in BANDS:
        m = all_matches.get(band, {})
        p = m.get("pressure", np.array([]))
        if len(p) == 0:
            continue
        for layer_name, mask_fn in LAYERS:
            sel = mask_fn(p)
            n = sel.sum()
            if n < 3:
                print(f"{band:>6s} {layer_name:>20s} {n:>8d} {'--':>8s} "
                      f"{'--':>8s} {'--':>8s} {'--':>8s}")
                continue
            rv = rmsvd(m["u_stereo"][sel], m["v_stereo"][sel],
                       m["u_amv"][sel], m["v_amv"][sel])
            sb = speed_bias(m["u_stereo"][sel], m["v_stereo"][sel],
                            m["u_amv"][sel], m["v_amv"][sel])
            spd_s = np.sqrt(m["u_stereo"][sel]**2 + m["v_stereo"][sel]**2)
            spd_a = np.sqrt(m["u_amv"][sel]**2 + m["v_amv"][sel]**2)
            rs = correlation(spd_s, spd_a)
            hr = height_rmse(m["h_stereo"][sel], m["h_amv"][sel])
            print(f"{band:>6s} {layer_name:>20s} {n:>8d} {rv:>8.2f} "
                  f"{sb:>8.2f} {rs:>8.3f} {hr:>8.0f}")
    print("=" * 100)


def print_summary(all_stats):
    """Print per-band summary table."""
    print("\n" + "=" * 110)
    print(f"Stereo Winds vs Operational AMVs — {MONTHS[0]} to {MONTHS[-1]}")
    print("=" * 110)
    print(f"{'Band':>6s} {'N':>8s} {'RMSVD':>8s} {'SpdBias':>8s} "
          f"{'r(speed)':>8s} {'r(u)':>8s} {'r(v)':>8s} "
          f"{'hRMSE':>8s} {'hBias':>8s} {'r(h)':>8s}")
    print("-" * 110)
    for band in BANDS:
        s = all_stats.get(band, {})
        if s.get("n_matches", 0) == 0:
            print(f"{band:>6s} {'0':>8s}" + " {:>8s}".format("--") * 8)
        else:
            print(f"{band:>6s} {s['n_matches']:>8d} {s['rmsvd']:>8.2f} "
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

    all_matches = {}
    all_stats = {}
    stereo_ds_c14 = None
    amv_data_c14 = None

    for band in BANDS:
        print(f"\nProcessing {band}...")

        # Load DMW from S3
        print(f"  Loading DMW ({band}) from S3...")
        amv_data = load_dmw_for_times(times, band)
        if amv_data is None:
            print(f"  No DMW data found for {band}")
            all_matches[band] = {k: np.array([]) for k in [
                "u_stereo", "v_stereo", "h_stereo",
                "u_amv", "v_amv", "h_amv",
                "lat", "lon", "pressure", "h_diff",
                "sigma_h_matched", "chi2_matched",
            ]}
            all_stats[band] = compute_stats(all_matches[band])
            continue

        # Load stereo from ArrayLake
        print(f"  Loading stereo ({band}) from ArrayLake...")
        stereo_ds = load_stereo(band, times)
        stereo_ds.load()

        # Collocate
        print(f"  Collocating...")
        matches = collocate(stereo_ds, amv_data, min_height=MIN_HEIGHT.get(band, 0))
        stats = compute_stats(matches)
        all_matches[band] = matches
        all_stats[band] = stats

        if band == "C14":
            stereo_ds_c14 = stereo_ds
            amv_data_c14 = amv_data

        print(f"  {band}: {stats['n_matches']:,} matches, RMSVD={stats.get('rmsvd', 'N/A')}")

    print_summary(all_stats)
    print_layer_summary(all_matches)

    print("\nGenerating plots...")
    plot_wind_scatter(all_matches, all_stats)
    plot_uv_scatter(all_matches)
    plot_amv_map(all_matches)
    plot_vertical_stats(all_matches)
    plot_height_scatter(all_matches, all_stats)
    plot_height_diff_hist(all_matches)
    plot_layer_bars(all_matches)
    plot_speed_dependence(all_matches)

    if stereo_ds_c14 is not None and amv_data_c14 is not None:
        print("\nRunning chi2 sensitivity analysis (C14)...")
        plot_chi2_sensitivity(stereo_ds_c14, amv_data_c14, MIN_HEIGHT.get("C14", 0))

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
