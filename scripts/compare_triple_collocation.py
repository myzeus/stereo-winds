"""Triple collocation: radiosondes, operational AMVs, and stereo winds.

Finds points where IGRA radiosonde profiles, GOES-16 operational AMVs (DMW),
and dense stereo wind retrievals all observe the same atmospheric column at
the same time.  Anchored on sonde station locations: extracts stereo 5x5
neighbourhood medians at each station, then finds the nearest DMW target
within 50 km whose height is consistent with the stereo/sonde match.

Produces 3-way intercomparison statistics and diagnostic plots for C14
(11.2 um IR), January-April 2024.
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

BANDS = ["C08", "C09", "C10", "C14"]
MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04"]
SAT = GOES16_CONFIG
MIN_HEIGHT = {"C08": 2000, "C09": 2000, "C10": 2000, "C14": 1000}

STEREO_TIMES = pd.date_range("2024-01-01", "2024-04-30", freq="12h")

S3_BUCKET = "noaa-goes16"
DMW_PRODUCT = "ABI-L2-DMWF"

AMV_RADIUS_PX = 25        # ~50 km at ~2 km/pixel
MAX_H_DIFF = 2000          # max height difference (m) for any pair
PIXEL_KM = 2.0
MIN_SPEED = 2.5            # minimum wind speed (m/s) — AMVs filter below ~3 m/s


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def pressure_to_height(p_hpa):
    """ICAO standard atmosphere: pressure (hPa) -> geopotential height (m)."""
    p = np.asarray(p_hpa, dtype=np.float64)
    tropo = p >= 226.32
    h = np.where(
        tropo,
        44330.77 * (1.0 - (p / 1013.25) ** 0.190263),
        11000.0 - 6341.62 * np.log(np.clip(p, 1e-6, None) / 226.32),
    )
    return h.astype(np.float32)


def dmw_wind_components(speed, direction):
    """Meteorological wind speed/direction -> u/v."""
    dir_rad = np.deg2rad(direction)
    return -speed * np.sin(dir_rad), -speed * np.cos(dir_rad)


def _height_gradient(h_2d):
    """Sobel height-gradient magnitude (m/pixel)."""
    h_filled = np.where(np.isfinite(h_2d), h_2d, 0.0)
    gx = sobel(h_filled, axis=1)
    gy = sobel(h_filled, axis=0)
    return np.sqrt(gx**2 + gy**2) / 8.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".stereo_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def load_stereo(band, times):
    """Load stereo winds from ArrayLake, with local disk cache."""
    cache_path = os.path.join(CACHE_DIR, f"stereo_{band}.zarr")

    if os.path.exists(cache_path):
        ds = xr.open_zarr(cache_path)
        avail = ds.time.values
        use = np.isin(avail, times.values)
        return ds.isel(time=use)

    # First time: load from ArrayLake and cache locally
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
    subset = combined.isel(time=use)

    # Cache to disk
    print(f"    Caching {band} to {cache_path} ...")
    subset.to_zarr(cache_path, mode="w")

    return subset


def load_igra(times):
    """Load IGRA 2024 radiosonde data."""
    client = arraylake.Client()
    repo = client.get_repo("zeus-ai/obs-xvec")
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, group="igra/2024", consolidated=False)
    return ds.sel(time=times)


def _init_s3():
    return s3fs.S3FileSystem(anon=True)


def find_dmw_file(fs, time, band):
    """Find DMW file on S3 closest to the given time and band."""
    doy = time.strftime("%j")
    year = time.strftime("%Y")
    hour = time.strftime("%H")
    prefix = f"{S3_BUCKET}/{DMW_PRODUCT}/{year}/{doy}/{hour}/"
    band_num = band[1:]
    try:
        files = fs.glob(f"{prefix}OR_{DMW_PRODUCT}-M6C{band_num}_G16_s*")
    except Exception:
        return None
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    target_ts = time.timestamp()
    best_file, best_diff = None, float("inf")
    for f in files:
        fname = os.path.basename(f)
        s_idx = fname.index("_s") + 2
        s_str = fname[s_idx: s_idx + 11]
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
    """Load a single DMW file from S3. Returns dict or None."""
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
        valid = np.isfinite(spd) & np.isfinite(wdir) & np.isfinite(pres) & np.isfinite(lat)
        if valid.sum() == 0:
            return None
        lat, lon = lat[valid], lon[valid]
        spd, wdir, pres = spd[valid], wdir[valid], pres[valid]
        u, v = dmw_wind_components(spd, wdir)
        h = pressure_to_height(pres)
        return dict(lat=lat, lon=lon, u=u, v=v, h=h, pressure=pres)
    except Exception as e:
        log.warning(f"  Failed to load {s3_path}: {e}")
        return None


def load_dmw_for_times(times, band):
    """Load and concatenate DMW data for all times."""
    fs = _init_s3()
    arrays = {k: [] for k in ["lat", "lon", "u", "v", "h", "pressure"]}
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
# Triple collocation
# ---------------------------------------------------------------------------

def station_pixel_coords(lats, lons):
    """Convert station lat/lon to GOES pixel row/col."""
    x_fg, y_fg = geodetic_to_fixed_grid(lats, lons, SAT)
    cols_f, rows_f = scanning_angle_to_pixel(x_fg, y_fg, SAT)
    on_grid = np.isfinite(cols_f) & np.isfinite(rows_f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cols_i = np.where(on_grid, np.round(cols_f).astype(int), -1)
        rows_i = np.where(on_grid, np.round(rows_f).astype(int), -1)
    on_grid &= (rows_i >= 0) & (rows_i < 5424) & (cols_i >= 0) & (cols_i < 5424)
    return rows_i, cols_i, on_grid


def triple_collocate(stereo_ds, igra_ds, amv_data,
                     min_height=MIN_HEIGHT, box_half=2,
                     sigma_h_max_low=1000, sigma_h_max_mid=2000,
                     sigma_h_max_high=1000,
                     h_grad_max=3000, chi2_max=0.2):
    """Find triplets where sonde, AMV, and stereo all observe the same point."""
    keys = [
        "u_stereo", "v_stereo", "h_stereo",
        "u_sonde", "v_sonde", "h_sonde", "pressure_sonde",
        "u_amv", "v_amv", "h_amv", "pressure_amv",
        "lat", "lon", "distance_amv_km",
    ]
    accum = {k: [] for k in keys}

    # Station pixel coords (fixed across time)
    sta_lats = igra_ds["lat"].values
    sta_lons = igra_ds["lon"].values
    sta_rows, sta_cols, sta_on_grid = station_pixel_coords(sta_lats, sta_lons)
    grid_idx = np.where(sta_on_grid)[0]
    if len(grid_idx) == 0:
        return {k: np.array([]) for k in keys}

    r_sta = sta_rows[grid_idx]
    c_sta = sta_cols[grid_idx]
    n_sta = len(grid_idx)

    # Neighbourhood offsets
    offsets = np.arange(-box_half, box_half + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()

    # Load full stereo grids
    n_times = len(stereo_ds.time)
    u_grid = stereo_ds["u_wind"].values
    v_grid = stereo_ds["v_wind"].values
    h_grid = stereo_ds["cloud_top_height"].values
    chi2_grid = stereo_ds["chi_squared"].values
    qf_grid = stereo_ds["quality_flag"].values
    sigh_grid = stereo_ds["sigma_h"].values

    # IGRA data
    pressure = igra_ds["pressure"].values
    sonde_u = igra_ds["u"].values[:, :, grid_idx]
    sonde_v = igra_ds["v"].values[:, :, grid_idx]
    sonde_h = igra_ds["geopotential_height"].values[:, :, grid_idx]
    sonde_valid = np.isfinite(sonde_h) & np.isfinite(sonde_u) & np.isfinite(sonde_v)

    amv_time_idx = amv_data["time_idx"]

    for t in range(n_times):
        # --- AMVs for this time step ---
        amv_mask_t = amv_time_idx == t
        if amv_mask_t.sum() == 0:
            continue

        amv_lat_t = amv_data["lat"][amv_mask_t]
        amv_lon_t = amv_data["lon"][amv_mask_t]
        amv_u_t = amv_data["u"][amv_mask_t]
        amv_v_t = amv_data["v"][amv_mask_t]
        amv_h_t = amv_data["h"][amv_mask_t]
        amv_p_t = amv_data["pressure"][amv_mask_t]

        # AMV pixel coords
        x_fg, y_fg = geodetic_to_fixed_grid(amv_lat_t, amv_lon_t, SAT)
        amv_cols_f, amv_rows_f = scanning_angle_to_pixel(x_fg, y_fg, SAT)
        amv_on = np.isfinite(amv_cols_f) & np.isfinite(amv_rows_f)
        amv_r = np.where(amv_on, np.round(amv_rows_f).astype(int), -9999)
        amv_c = np.where(amv_on, np.round(amv_cols_f).astype(int), -9999)

        # --- Stereo extraction at sonde stations ---
        r_nb = np.clip(r_sta[:, None] + dy[None, :], 0, 5423)
        c_nb = np.clip(c_sta[:, None] + dx[None, :], 0, 5423)

        u_nb = u_grid[t][r_nb, c_nb]
        v_nb = v_grid[t][r_nb, c_nb]
        h_nb = h_grid[t][r_nb, c_nb]
        chi2_nb = chi2_grid[t][r_nb, c_nb]
        qf_nb = qf_grid[t][r_nb, c_nb]
        sigh_nb = sigh_grid[t][r_nb, c_nb]
        hgrad = _height_gradient(h_grid[t])
        hgrad_nb = hgrad[r_nb, c_nb]

        spd_nb = np.sqrt(u_nb**2 + v_nb**2)
        # Height-dependent sigma_h threshold: relaxed at mid-levels
        # where parallax sensitivity is weakest
        sigh_thresh = np.where(
            h_nb < 3000, sigma_h_max_low,        # low clouds
            np.where(h_nb < 7000, sigma_h_max_mid,  # mid-level
                     sigma_h_max_high),            # high clouds
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
        u_nb = np.where(qa_nb, u_nb, np.nan)
        v_nb = np.where(qa_nb, v_nb, np.nan)
        h_nb = np.where(qa_nb, h_nb, np.nan)

        with np.errstate(all="ignore"):
            u_s = np.nanmedian(u_nb, axis=1)
            v_s = np.nanmedian(v_nb, axis=1)
            h_s = np.nanmedian(h_nb, axis=1)

        n_valid_nb = np.sum(qa_nb, axis=1)
        stereo_spd = np.sqrt(u_s**2 + v_s**2)
        stereo_ok = (
            np.isfinite(u_s) & np.isfinite(h_s)
            & (n_valid_nb >= 3)
            & (stereo_spd >= MIN_SPEED)
        )

        # --- Triple height-consistent matching ---
        # For each station with valid stereo, find nearest AMV within
        # radius, then require BOTH AMV and stereo heights are close
        # to a sonde level (independent height filtering).
        candidate_idx = np.where(stereo_ok)[0]
        if len(candidate_idx) == 0:
            continue

        for si_local in candidate_idx:
            rs = r_sta[si_local]
            cs = c_sta[si_local]
            hs = h_s[si_local]

            # Find nearest AMV within spatial radius
            dr = amv_r.astype(np.float32) - rs
            dc = amv_c.astype(np.float32) - cs
            dist_px = np.sqrt(dr**2 + dc**2)

            near = amv_on & (dist_px <= AMV_RADIUS_PX)
            if not near.any():
                continue

            dist_near = np.where(near, dist_px, np.inf)
            best = np.argmin(dist_near)
            h_amv_best = amv_h_t[best]

            # Find nearest sonde level to AMV height
            sonde_h_col = sonde_h[:, t, si_local]      # (n_plevs,)
            sonde_v_col = sonde_valid[:, t, si_local]
            h_diff_lev = np.abs(sonde_h_col - h_amv_best)
            h_diff_lev = np.where(sonde_v_col, h_diff_lev, np.inf)
            best_lev = np.argmin(h_diff_lev)

            if not np.isfinite(h_diff_lev[best_lev]):
                continue

            h_so = sonde_h_col[best_lev]

            # Bracketing: sonde must have valid levels above AND below
            above = (sonde_h_col > h_amv_best) & sonde_v_col
            below = (sonde_h_col < h_amv_best) & sonde_v_col
            if not (above.any() and below.any()):
                continue

            # Height threshold for AMV-sonde match
            p_match = pressure[best_lev]
            max_h = 2000.0 if p_match <= 250 else 500.0
            if h_diff_lev[best_lev] > max_h:
                continue

            # Also require stereo height close to sonde level
            if abs(hs - h_so) > MAX_H_DIFF:
                continue

            u_so = sonde_u[best_lev, t, si_local]
            v_so = sonde_v[best_lev, t, si_local]
            if not (np.isfinite(u_so) and np.isfinite(h_so)):
                continue

            accum["u_stereo"].append(u_s[si_local])
            accum["v_stereo"].append(v_s[si_local])
            accum["h_stereo"].append(hs)
            accum["u_sonde"].append(u_so)
            accum["v_sonde"].append(v_so)
            accum["h_sonde"].append(h_so)
            accum["pressure_sonde"].append(p_match)
            accum["u_amv"].append(amv_u_t[best])
            accum["v_amv"].append(amv_v_t[best])
            accum["h_amv"].append(h_amv_best)
            accum["pressure_amv"].append(amv_p_t[best])
            accum["lat"].append(sta_lats[grid_idx[si_local]])
            accum["lon"].append(sta_lons[grid_idx[si_local]])
            accum["distance_amv_km"].append(float(dist_px[best]) * PIXEL_KM)

    if all(len(v) == 0 for v in accum.values()):
        return {k: np.array([]) for k in keys}

    return {k: np.asarray(v, dtype=np.float64) for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _pair_stats(u1, v1, u2, v2, h1, h2):
    """Compute stats for one pair."""
    if len(u1) == 0:
        return {"rmsvd": np.nan, "speed_bias": np.nan, "corr_speed": np.nan,
                "h_rmse": np.nan, "h_bias": np.nan}
    spd1 = np.sqrt(u1**2 + v1**2)
    spd2 = np.sqrt(u2**2 + v2**2)
    return {
        "rmsvd": rmsvd(u1, v1, u2, v2),
        "speed_bias": speed_bias(u1, v1, u2, v2),
        "corr_speed": correlation(spd1, spd2),
        "h_rmse": height_rmse(h1, h2),
        "h_bias": float(np.mean(h1 - h2)),
    }


def compute_triple_stats(m):
    """Compute pairwise stats for all three pairs."""
    return {
        "n": len(m["u_stereo"]),
        "stereo_vs_sonde": _pair_stats(
            m["u_stereo"], m["v_stereo"], m["u_sonde"], m["v_sonde"],
            m["h_stereo"], m["h_sonde"]),
        "stereo_vs_amv": _pair_stats(
            m["u_stereo"], m["v_stereo"], m["u_amv"], m["v_amv"],
            m["h_stereo"], m["h_amv"]),
        "amv_vs_sonde": _pair_stats(
            m["u_amv"], m["v_amv"], m["u_sonde"], m["v_sonde"],
            m["h_amv"], m["h_sonde"]),
    }


def triple_collocation_errors(m):
    """Estimate per-system error variances via triple collocation.

    Stoffelen (1998) formulation.  Systems: S=stereo, A=AMV, R=radiosonde.

    Classic TC (per component, e.g. u):
        σ²_s = mean((S-A)(S-R))
        σ²_a = mean((A-S)(A-R))
        σ²_r = mean((R-S)(R-A))

    When satellite errors are correlated (shared representativeness σ²_rep):
        σ²_s_classic = σ²_s_true - σ²_rep   (lower bound on stereo)
        σ²_a_classic = σ²_a_true - σ²_rep   (lower bound on AMV)
        σ²_r_classic = σ²_r_true + σ²_rep   (upper bound on sonde)

    The representativeness cannot be uniquely solved with 3 datasets
    (underdetermined: 4 unknowns, 3 equations).  But we can bound it:
        σ²_rep <= min(σ²_r_classic, ...)  since σ²_r_true >= 0
        σ²_rep <= σ²_s_classic + σ²_rep  (trivial)

    We report classic TC estimates with the bias interpretation.
    """
    n = len(m["u_stereo"])
    if n < 10:
        return None

    result = {}
    for comp in ["u", "v"]:
        s = m[f"{comp}_stereo"]
        a = m[f"{comp}_amv"]
        r = m[f"{comp}_sonde"]

        # Remove mean biases before TC (standard practice)
        s = s - np.mean(s)
        a = a - np.mean(a)
        r = r - np.mean(r)

        # Classic TC variances
        var_s = float(np.mean((s - a) * (s - r)))
        var_a = float(np.mean((a - s) * (a - r)))
        var_r = float(np.mean((r - s) * (r - a)))

        # Clamp negative variances (can happen with small samples)
        var_s = max(var_s, 0.0)
        var_a = max(var_a, 0.0)
        var_r = max(var_r, 0.0)

        # Upper bound on representativeness: σ²_rep <= σ²_r_classic
        # (since σ²_r_true >= 0 implies σ²_rep <= σ²_r_classic)
        # Also σ²_rep cannot exceed the pairwise covariance of satellite
        # residuals vs sonde: cov_sa = mean((S-R)(A-R)) = σ²_r + σ²_rep
        # We estimate the bound as min of the classic sonde variance
        # (which is σ²_r_true + σ²_rep).
        var_rep_upper = var_r  # upper bound

        result[comp] = {
            "var_s": var_s, "var_a": var_a, "var_r": var_r,
            "var_rep_upper": var_rep_upper,
        }

    def _safe_sqrt(x):
        return np.sqrt(max(x, 0.0))

    u, v = result["u"], result["v"]

    # Upper bound on representativeness
    rep_upper = _safe_sqrt(u["var_rep_upper"] + v["var_rep_upper"])

    # If correlated, true satellite errors are HIGHER than classic TC:
    #   σ²_true = σ²_classic + σ²_rep
    # Upper bound on true satellite errors uses rep_upper
    return {
        "n": n,
        "rmse_stereo": _safe_sqrt(u["var_s"] + v["var_s"]),
        "rmse_amv": _safe_sqrt(u["var_a"] + v["var_a"]),
        "rmse_sonde": _safe_sqrt(u["var_r"] + v["var_r"]),
        "rmse_stereo_upper": _safe_sqrt(
            u["var_s"] + u["var_rep_upper"] + v["var_s"] + v["var_rep_upper"]),
        "rmse_amv_upper": _safe_sqrt(
            u["var_a"] + u["var_rep_upper"] + v["var_a"] + v["var_rep_upper"]),
        "rmse_sonde_lower": 0.0,  # lower bound is 0 (if all sonde "error" is rep)
        "rmse_rep_upper": rep_upper,
        "components": result,
    }


def print_tc_errors(tc, label=""):
    """Print TC error decomposition table."""
    if tc is None:
        return
    print(f"\n{'=' * 85}")
    print(f"Triple Collocation Error Decomposition (Stoffelen 1998){label}")
    print(f"N = {tc['n']:,}")
    print(f"{'=' * 85}")
    print(f"{'System':<10s} {'TC RMSE':>10s} {'Range if correlated':>25s}  Interpretation")
    print("-" * 85)
    s_range = f"[{tc['rmse_stereo']:.2f} – {tc['rmse_stereo_upper']:.2f}]"
    a_range = f"[{tc['rmse_amv']:.2f} – {tc['rmse_amv_upper']:.2f}]"
    r_range = f"[0.00 – {tc['rmse_sonde']:.2f}]"
    print(f"{'Stereo':<10s} {tc['rmse_stereo']:>10.2f} {s_range:>25s}"
          f"  lower bound (if S-A correlated)")
    print(f"{'AMV':<10s} {tc['rmse_amv']:>10.2f} {a_range:>25s}"
          f"  lower bound (if S-A correlated)")
    print(f"{'Sonde':<10s} {tc['rmse_sonde']:>10.2f} {r_range:>25s}"
          f"  upper bound (absorbs rep error)")
    print(f"\n  Representativeness upper bound (Stereo–AMV): "
          f"<= {tc['rmse_rep_upper']:.2f} m/s")
    print(f"{'=' * 85}")

    # Per-component detail
    for comp in ["u", "v"]:
        c = tc["components"][comp]
        print(f"  {comp}-component:  "
              f"σ²_stereo={c['var_s']:.2f}  σ²_amv={c['var_a']:.2f}  "
              f"σ²_sonde={c['var_r']:.2f}  σ²_rep<={c['var_rep_upper']:.2f}")


def print_summary(stats):
    """Print triple-collocation summary table."""
    n = stats["n"]
    print("\n" + "=" * 90)
    print(f"Triple Collocation Summary (C14) — {MONTHS[0]} to {MONTHS[-1]}  N = {n:,}")
    print("=" * 90)
    print(f"{'Pair':<22s} {'RMSVD':>8s} {'SpdBias':>8s} {'r(spd)':>8s} "
          f"{'hRMSE':>8s} {'hBias':>8s}")
    print("-" * 90)
    for label, key in [
        ("Stereo vs Sonde", "stereo_vs_sonde"),
        ("Stereo vs AMV", "stereo_vs_amv"),
        ("AMV vs Sonde", "amv_vs_sonde"),
    ]:
        s = stats[key]
        print(f"{label:<22s} {s['rmsvd']:>8.2f} {s['speed_bias']:>+8.2f} "
              f"{s['corr_speed']:>8.3f} {s['h_rmse']:>8.0f} {s['h_bias']:>+8.0f}")
    print("=" * 90)


LAYERS = [
    ("Low (>=700 hPa)", lambda p: p >= 700),
    ("Mid (400-700 hPa)", lambda p: (p >= 400) & (p < 700)),
    ("High (<400 hPa)", lambda p: p < 400),
]


def print_layer_summary(m):
    """Print layer-stratified triple-collocation stats."""
    if len(m["u_stereo"]) == 0:
        return
    p = m["pressure_sonde"]
    print("\n" + "=" * 100)
    print(f"Layer-Stratified Triple Collocation — {MONTHS[0]} to {MONTHS[-1]}")
    print("=" * 100)
    print(f"{'Layer':<22s} {'N':>6s}  {'RMSVD_SvR':>9s} {'RMSVD_SvA':>9s} "
          f"{'RMSVD_AvR':>9s}  {'Bias_SvR':>8s} {'Bias_SvA':>8s} {'Bias_AvR':>8s}")
    print("-" * 100)
    for label, mask_fn in LAYERS:
        sel = mask_fn(p)
        n = int(sel.sum())
        if n < 3:
            print(f"{label:<22s} {n:>6d}  {'--':>9s} {'--':>9s} {'--':>9s}"
                  f"  {'--':>8s} {'--':>8s} {'--':>8s}")
            continue
        svr = rmsvd(m["u_stereo"][sel], m["v_stereo"][sel],
                    m["u_sonde"][sel], m["v_sonde"][sel])
        sva = rmsvd(m["u_stereo"][sel], m["v_stereo"][sel],
                    m["u_amv"][sel], m["v_amv"][sel])
        avr = rmsvd(m["u_amv"][sel], m["v_amv"][sel],
                    m["u_sonde"][sel], m["v_sonde"][sel])
        b_svr = speed_bias(m["u_stereo"][sel], m["v_stereo"][sel],
                           m["u_sonde"][sel], m["v_sonde"][sel])
        b_sva = speed_bias(m["u_stereo"][sel], m["v_stereo"][sel],
                           m["u_amv"][sel], m["v_amv"][sel])
        b_avr = speed_bias(m["u_amv"][sel], m["v_amv"][sel],
                           m["u_sonde"][sel], m["v_sonde"][sel])
        print(f"{label:<22s} {n:>6d}  {svr:>9.2f} {sva:>9.2f} {avr:>9.2f}"
              f"  {b_svr:>+8.2f} {b_sva:>+8.2f} {b_avr:>+8.2f}")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_tc_error_bars(all_tc):
    """Bar chart of TC error decomposition with uncertainty ranges."""
    bands_with_tc = [b for b in BANDS if all_tc.get(b) is not None]
    if not bands_with_tc:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bands_with_tc))
    width = 0.22

    stereo = [all_tc[b]["rmse_stereo"] for b in bands_with_tc]
    stereo_up = [all_tc[b]["rmse_stereo_upper"] for b in bands_with_tc]
    amv = [all_tc[b]["rmse_amv"] for b in bands_with_tc]
    amv_up = [all_tc[b]["rmse_amv_upper"] for b in bands_with_tc]
    sonde = [all_tc[b]["rmse_sonde"] for b in bands_with_tc]

    # Bars at classic TC values, error bars up to upper bound
    stereo_err = [[0]*len(x), [su - sl for sl, su in zip(stereo, stereo_up)]]
    amv_err = [[0]*len(x), [au - al for al, au in zip(amv, amv_up)]]
    sonde_err = [[0]*len(x), [s for s in sonde]]  # lower bound is 0

    ax.bar(x - width, stereo, width, label="Stereo (lower bound)",
           color="#4c72b0", yerr=stereo_err, capsize=4, error_kw={"lw": 1.5})
    ax.bar(x, amv, width, label="AMV (lower bound)",
           color="#dd8452", yerr=amv_err, capsize=4, error_kw={"lw": 1.5})
    ax.bar(x + width, sonde, width, label="Sonde (upper bound)",
           color="#55a868", yerr=sonde_err, capsize=4, error_kw={"lw": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(bands_with_tc)
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title(f"Triple Collocation Error Estimates — {MONTHS[0]} to {MONTHS[-1]}\n"
                 f"Bars = classic TC; whiskers = range if satellite errors correlated")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(FIGURES_DIR, "triple_tc_errors.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_triple_scatter(m):
    """3-panel speed scatter: stereo-vs-sonde, stereo-vs-AMV, AMV-vs-sonde."""
    if len(m["u_stereo"]) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [
        ("Stereo", "Sonde", m["u_stereo"], m["v_stereo"],
         m["u_sonde"], m["v_sonde"], m["h_stereo"]),
        ("Stereo", "AMV", m["u_stereo"], m["v_stereo"],
         m["u_amv"], m["v_amv"], m["h_stereo"]),
        ("AMV", "Sonde", m["u_amv"], m["v_amv"],
         m["u_sonde"], m["v_sonde"], m["h_amv"]),
    ]
    sc = None
    for ax, (l1, l2, u1, v1, u2, v2, h_color) in zip(axes, pairs):
        s1 = np.sqrt(u1**2 + v1**2)
        s2 = np.sqrt(u2**2 + v2**2)
        h_km = h_color / 1000.0
        sc = ax.scatter(s2, s1, c=h_km, s=15, alpha=0.6,
                        cmap="turbo", vmin=0, vmax=16)
        lim = max(s1.max(), s2.max()) * 1.05
        lim = min(lim, 80)
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{l2} speed (m/s)")
        ax.set_ylabel(f"{l1} speed (m/s)")
        rv = rmsvd(u1, v1, u2, v2)
        r = correlation(s1, s2)
        ax.set_title(f"{l1} vs {l2}  RMSVD={rv:.1f}  r={r:.2f}")
    fig.subplots_adjust(right=0.92)
    cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cax, label="Height (km)")
    fig.suptitle(f"Triple Collocation — Wind Speed (C14, N={len(m['u_stereo']):,})"
                 f" — {MONTHS[0]} to {MONTHS[-1]}", fontsize=13)
    path = os.path.join(FIGURES_DIR, "triple_wind_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_triple_height(m):
    """3-panel height scatter for all pairs."""
    if len(m["h_stereo"]) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [
        ("Stereo", "Sonde", m["h_stereo"], m["h_sonde"]),
        ("Stereo", "AMV", m["h_stereo"], m["h_amv"]),
        ("AMV", "Sonde", m["h_amv"], m["h_sonde"]),
    ]
    vd = np.sqrt((m["u_stereo"] - m["u_sonde"])**2 +
                  (m["v_stereo"] - m["v_sonde"])**2)
    for ax, (l1, l2, h1, h2) in zip(axes, pairs):
        h1k, h2k = h1 / 1000, h2 / 1000
        ax.scatter(h2k, h1k, c=vd, s=15, alpha=0.6, cmap="YlOrRd",
                   vmin=0, vmax=20)
        lim = max(h1k.max(), h2k.max()) * 1.05
        lim = min(lim, 18)
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{l2} height (km)")
        ax.set_ylabel(f"{l1} height (km)")
        hr = height_rmse(h1, h2)
        hb = float(np.mean(h1 - h2))
        ax.set_title(f"{l1} vs {l2}  hRMSE={hr/1000:.1f}km  hBias={hb/1000:+.1f}km")
    fig.suptitle(f"Triple Collocation — Height (C14, N={len(m['h_stereo']):,})"
                 f" — {MONTHS[0]} to {MONTHS[-1]}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "triple_height_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_triple_vertical(m):
    """RMSVD and speed bias binned by pressure for all three pairs."""
    if len(m["u_stereo"]) == 0:
        return
    pressure_bins = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100]
    bin_centers = [(pressure_bins[i] + pressure_bins[i+1]) / 2
                   for i in range(len(pressure_bins) - 1)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    ax_rmsvd, ax_bias = axes
    p = m["pressure_sonde"]
    pair_defs = [
        ("Stereo vs Sonde", m["u_stereo"], m["v_stereo"], m["u_sonde"], m["v_sonde"]),
        ("Stereo vs AMV", m["u_stereo"], m["v_stereo"], m["u_amv"], m["v_amv"]),
        ("AMV vs Sonde", m["u_amv"], m["v_amv"], m["u_sonde"], m["v_sonde"]),
    ]
    for label, u1, v1, u2, v2 in pair_defs:
        rv_list, sb_list = [], []
        for j in range(len(bin_centers)):
            mask = (p >= pressure_bins[j+1]) & (p < pressure_bins[j])
            n = mask.sum()
            if n >= 5:
                rv_list.append(rmsvd(u1[mask], v1[mask], u2[mask], v2[mask]))
                sb_list.append(speed_bias(u1[mask], v1[mask], u2[mask], v2[mask]))
            else:
                rv_list.append(np.nan)
                sb_list.append(np.nan)
        ax_rmsvd.plot(rv_list, bin_centers, "o-", label=label, markersize=5)
        ax_bias.plot(sb_list, bin_centers, "o-", label=label, markersize=5)
    ax_rmsvd.set_ylabel("Pressure (hPa)")
    ax_rmsvd.set_xlabel("RMSVD (m/s)")
    ax_rmsvd.set_yscale("log")
    ax_rmsvd.invert_yaxis()
    ax_rmsvd.set_ylim(1050, 90)
    ax_rmsvd.grid(alpha=0.3)
    ax_rmsvd.set_title("RMSVD")
    ax_rmsvd.legend(fontsize=8)
    ax_bias.axvline(0, color="gray", lw=0.5, ls="--")
    ax_bias.set_xlabel("Speed Bias (m/s)")
    ax_bias.grid(alpha=0.3)
    ax_bias.set_title("Speed Bias")
    ax_bias.legend(fontsize=8)
    fig.suptitle(f"Triple Collocation — Vertical Stats (C14, N={len(m['u_stereo']):,})"
                 f" — {MONTHS[0]} to {MONTHS[-1]}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "triple_vertical_stats.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_triple_map(m):
    """Station map colored by stereo-sonde VD."""
    if len(m["lat"]) == 0:
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
    cbar.set_label("Stereo-Sonde VD (m/s)")
    ax.set_title(f"Triple Collocation Locations (C14) — {MONTHS[0]} to {MONTHS[-1]}"
                 f"  N={len(m['lat']):,}")
    ax.gridlines(draw_labels=True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "triple_station_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_triple_uv(m):
    """U and V component scatter for all three pairs (2x3 grid)."""
    if len(m["u_stereo"]) == 0:
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    pairs = [
        ("Stereo", "Sonde", m["u_stereo"], m["v_stereo"],
         m["u_sonde"], m["v_sonde"]),
        ("Stereo", "AMV", m["u_stereo"], m["v_stereo"],
         m["u_amv"], m["v_amv"]),
        ("AMV", "Sonde", m["u_amv"], m["v_amv"],
         m["u_sonde"], m["v_sonde"]),
    ]
    h_km = m["h_stereo"] / 1000.0
    for col, (l1, l2, u1, v1, u2, v2) in enumerate(pairs):
        for row, (comp1, comp2, comp_name) in enumerate([
            (u1, u2, "U"), (v1, v2, "V"),
        ]):
            ax = axes[row, col]
            ax.scatter(comp2, comp1, c=h_km, s=15, alpha=0.6,
                       cmap="turbo", vmin=0, vmax=16)
            lim = max(abs(comp1).max(), abs(comp2).max()) * 1.1
            lim = min(lim, 80)
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{l2} {comp_name} (m/s)")
            ax.set_ylabel(f"{l1} {comp_name} (m/s)")
            r_val = correlation(comp1, comp2)
            ax.set_title(f"{l1} vs {l2} — {comp_name}  r={r_val:.2f}")
    fig.suptitle(f"Triple Collocation — U/V Components (C14, N={len(m['u_stereo']):,})"
                 f" — {MONTHS[0]} to {MONTHS[-1]}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "triple_uv_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_band_summary(all_stats):
    """Print per-band, per-pair summary table."""
    print("\n" + "=" * 120)
    print(f"Triple Collocation Per-Band Summary — {MONTHS[0]} to {MONTHS[-1]}")
    print("=" * 120)
    print(f"{'Band':>6s} {'N':>6s}  "
          f"{'SvR_RMSVD':>9s} {'SvR_Bias':>8s} {'SvR_r':>7s}  "
          f"{'SvA_RMSVD':>9s} {'SvA_Bias':>8s} {'SvA_r':>7s}  "
          f"{'AvR_RMSVD':>9s} {'AvR_Bias':>8s} {'AvR_r':>7s}")
    print("-" * 120)
    for band in BANDS:
        s = all_stats.get(band)
        if s is None or s["n"] == 0:
            print(f"{band:>6s} {'0':>6s}" + "  " + ("  ".join(["--"] * 9)))
            continue
        svr = s["stereo_vs_sonde"]
        sva = s["stereo_vs_amv"]
        avr = s["amv_vs_sonde"]
        print(f"{band:>6s} {s['n']:>6d}  "
              f"{svr['rmsvd']:>9.2f} {svr['speed_bias']:>+8.2f} {svr['corr_speed']:>7.3f}  "
              f"{sva['rmsvd']:>9.2f} {sva['speed_bias']:>+8.2f} {sva['corr_speed']:>7.3f}  "
              f"{avr['rmsvd']:>9.2f} {avr['speed_bias']:>+8.2f} {avr['corr_speed']:>7.3f}")
    print("=" * 120)

    # Height summary
    print(f"\n{'Band':>6s} {'N':>6s}  "
          f"{'SvR_hRMSE':>9s} {'SvR_hBias':>9s}  "
          f"{'SvA_hRMSE':>9s} {'SvA_hBias':>9s}  "
          f"{'AvR_hRMSE':>9s} {'AvR_hBias':>9s}")
    print("-" * 80)
    for band in BANDS:
        s = all_stats.get(band)
        if s is None or s["n"] == 0:
            continue
        svr = s["stereo_vs_sonde"]
        sva = s["stereo_vs_amv"]
        avr = s["amv_vs_sonde"]
        print(f"{band:>6s} {s['n']:>6d}  "
              f"{svr['h_rmse']:>9.0f} {svr['h_bias']:>+9.0f}  "
              f"{sva['h_rmse']:>9.0f} {sva['h_bias']:>+9.0f}  "
              f"{avr['h_rmse']:>9.0f} {avr['h_bias']:>+9.0f}")
    print("-" * 80)


def print_band_tc_summary(all_tc):
    """Print per-band TC error decomposition."""
    bands_with_tc = [b for b in BANDS if all_tc.get(b) is not None]
    if not bands_with_tc:
        return
    print("\n" + "=" * 110)
    print(f"TC Error Decomposition Per Band (Stoffelen 1998) — {MONTHS[0]} to {MONTHS[-1]}")
    print("  Classic TC: satellite estimates are LOWER bounds if S-A errors correlated")
    print("              sonde estimate is UPPER bound (absorbs representativeness)")
    print("=" * 110)
    print(f"{'Band':>6s} {'N':>6s}  {'Stereo':>8s} {'[upper]':>8s}  "
          f"{'AMV':>8s} {'[upper]':>8s}  {'Sonde':>8s} {'[lower]':>8s}  "
          f"{'Rep<=':>8s}")
    print("-" * 110)
    for band in bands_with_tc:
        tc = all_tc[band]
        print(f"{band:>6s} {tc['n']:>6d}  "
              f"{tc['rmse_stereo']:>8.2f} {tc['rmse_stereo_upper']:>8.2f}  "
              f"{tc['rmse_amv']:>8.2f} {tc['rmse_amv_upper']:>8.2f}  "
              f"{tc['rmse_sonde']:>8.2f} {'0.00':>8s}  "
              f"{tc['rmse_rep_upper']:>8.2f}")
    print("=" * 110)


def main():
    print("Finding filled stereo time steps...")
    ref_ds = load_stereo("C14", STEREO_TIMES)
    center = ref_ds["u_wind"].isel(y=2712, x=2712).values
    times = ref_ds.time.values[np.isfinite(center)]
    times = pd.DatetimeIndex(times)
    print(f"  {len(times)} filled time steps")

    print("Loading IGRA data...")
    igra_ds = load_igra(times)
    igra_ds.load()
    print(f"  IGRA: {dict(igra_ds.sizes)}")

    all_matches = {}
    all_stats = {}

    for band in BANDS:
        print(f"\n--- {band} ---")

        print(f"  Loading DMW ({band}) from S3...")
        amv_data = load_dmw_for_times(times, band)
        if amv_data is None:
            print(f"  No DMW data found for {band}")
            all_stats[band] = {"n": 0}
            continue

        print(f"  Loading stereo ({band}) from ArrayLake...")
        stereo_ds = load_stereo(band, times)
        stereo_ds.load()

        print(f"  Running triple collocation...")
        matches = triple_collocate(stereo_ds, igra_ds, amv_data,
                                   min_height=MIN_HEIGHT.get(band, 0))
        n = len(matches["u_stereo"])
        print(f"  {band}: {n:,} triple-collocated points")

        all_matches[band] = matches
        all_stats[band] = compute_triple_stats(matches)

    # Per-band summary
    print_band_summary(all_stats)

    # TC error decomposition per band
    all_tc = {}
    for band in BANDS:
        m = all_matches.get(band, {})
        if len(m.get("u_stereo", [])) > 0:
            all_tc[band] = triple_collocation_errors(m)

    print_band_tc_summary(all_tc)

    # Detailed stats and TC for C14
    if all_stats.get("C14", {}).get("n", 0) > 0:
        print_summary(all_stats["C14"])
        print_layer_summary(all_matches["C14"])
        if all_tc.get("C14") is not None:
            print_tc_errors(all_tc["C14"], label=" — C14")

    # Plots for C14
    m = all_matches.get("C14", {})
    if len(m.get("u_stereo", [])) > 0:
        print("\nGenerating plots (C14)...")
        plot_triple_scatter(m)
        plot_triple_uv(m)
        plot_triple_height(m)
        plot_triple_vertical(m)
        plot_triple_map(m)
        plot_tc_error_bars(all_tc)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
