"""Quadruple collocation: radiosondes, operational AMVs, ERA5, and stereo winds.

Validates the fine-tuned stereo-wind teacher over the held-out test period
(default 2025-10 / 2025-11 — months the model never trained on) against three
independent references co-located at the same atmospheric columns:

    stereo  — fine-tuned RAFT + WLS retrieval (cached full-disk zarrs)
    sonde   — IGRA radiosondes (collocation parquet)
    AMV     — operational NOAA GOES-19 ABI Derived Motion Winds (S3)
    ERA5    — reanalysis (arco-era5), sampled at the matched column/height

Anchored on IGRA station locations: stereo 5x5 neighbourhood medians at each
station, nearest AMV within ~50 km whose height is consistent, then ERA5
sampled at the station/height as a fourth member.

Reports all pairwise intercomparison statistics and a triple-collocation error
decomposition (Stoffelen 1998) restricted to {stereo, AMV, sonde} — ERA5
assimilates AMVs and sondes, so it is NOT independent and is reported as a
correlated pairwise reference only.

Adapted from compare_triple_collocation.py (GOES-16 / ArrayLake / 2024) for the
GOES-19 grid, local cached retrievals, and the IGRA collocation parquet.
"""

import argparse
import logging
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from scipy.ndimage import sobel

from stereo_winds.config import GOES19_CONFIG
from stereo_winds.validation.era5 import (
    DEFAULT_LEVELS,
    load_era5_single_time,
    open_era5_reader,
    sample_era5,
)
from stereo_winds.validation.metrics import (
    correlation,
    height_rmse,
    rmsvd,
    speed_bias,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (GOES-19, test months)
# ---------------------------------------------------------------------------
SAT = GOES19_CONFIG
SAT_ID = "G19"
S3_BUCKET = "noaa-goes19"
DMW_PRODUCT = "ABI-L2-DMWF"

DEFAULT_BANDS = ["C08", "C09", "C10", "C14"]
DEFAULT_MONTHS = ["2025-10", "2025-11"]
MIN_HEIGHT = {"C08": 2000, "C09": 2000, "C10": 2000, "C14": 1000}

AMV_RADIUS_PX = 25         # ~50 km at ~2 km/pixel
MAX_H_DIFF = 2000          # max height difference (m) for any pair
PIXEL_KM = 2.0
MIN_SPEED = 2.5            # min wind speed (m/s); AMVs filter below ~3 m/s
GRID = 5424               # GOES ABI full-disk side length

# GOES-19/18 overlap footprint, used to bbox-subset the ERA5 reads.
ERA5_LAT_BBOX = (-65.0, 65.0)
ERA5_LON_BBOX = (-150.0, -20.0)


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
# Stereo loading (local cached retrievals)
# ---------------------------------------------------------------------------
def stereo_zarr_path(cache_dir, label, band, month, n_iter):
    """Cache path: {cache_dir}/{label}_{band}_{YYYYMM}_iter{n}.zarr.

    The band token is optional in the label scheme; we try with and without it.
    """
    ym = month.replace("-", "")
    cand = [
        os.path.join(cache_dir, f"{label}_{band}_{ym}_iter{n_iter}.zarr"),
        os.path.join(cache_dir, f"{label}_{ym}_iter{n_iter}.zarr"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    return cand[0]


def load_stereo(cache_dir, label, band, months, n_iter, times=None,
                override_path=None):
    """Load cached stereo retrievals for a band across months; optional time subset.

    ``override_path``: load this single multi-time zarr directly (e.g. a student
    inference output) instead of the per-month label pattern, so the same
    collocation machinery evaluates the student as a "stereo" member.
    """
    if override_path is not None:
        if not os.path.exists(override_path):
            log.warning(f"  missing override zarr: {override_path}")
            return None
        c = xr.open_zarr(override_path, consolidated=False).sortby("time")
        if times is not None:
            c = c.isel(time=np.isin(c.time.values, np.asarray(times)))
        return c
    dsets = []
    for month in months:
        path = stereo_zarr_path(cache_dir, label, band, month, n_iter)
        if not os.path.exists(path):
            log.warning(f"  missing stereo cache: {path}")
            continue
        dsets.append(xr.open_zarr(path, consolidated=False))
    if not dsets:
        return None
    combined = xr.concat(dsets, dim="time") if len(dsets) > 1 else dsets[0]
    combined = combined.sortby("time")
    if times is not None:
        use = np.isin(combined.time.values, np.asarray(times))
        combined = combined.isel(time=use)
    return combined


# ---------------------------------------------------------------------------
# IGRA loading (collocation parquet)
# ---------------------------------------------------------------------------
def load_igra(parquet_path, times, months, max_dt_min=30):
    """Build an IGRA dataset on a (level, time, station) grid from the parquet.

    Filters to the requested months over ALL stations.  The time axis is the
    passed stereo ``times``; each parquet row is mapped to the nearest stereo
    time within ``max_dt_min`` minutes.

    Returns an xr.Dataset with u, v, geopotential_height on (level, time,
    station) plus per-station lat, lon, row_19, col_19.
    """
    df = pd.read_parquet(parquet_path)
    df["goes_time"] = pd.to_datetime(df["goes_time"])
    ym = df["goes_time"].dt.strftime("%Y%m")
    keep_ym = {m.replace("-", "") for m in months}
    df = df[ym.isin(keep_ym)].copy()
    if len(df) == 0:
        raise ValueError(f"No IGRA parquet rows in months {sorted(keep_ym)}")

    times = pd.DatetimeIndex(np.asarray(times).astype("datetime64[ns]"))
    t_ns = times.values.astype("int64")
    # Map each row's goes_time to the nearest stereo time within tolerance.
    row_ns = df["goes_time"].values.astype("datetime64[ns]").astype("int64")
    pos = np.searchsorted(t_ns, row_ns)
    pos = np.clip(pos, 1, len(t_ns) - 1)
    left, right = t_ns[pos - 1], t_ns[pos]
    pick_left = (row_ns - left) <= (right - row_ns)
    ti = np.where(pick_left, pos - 1, pos)
    dt_ok = np.abs(row_ns - t_ns[ti]) <= int(max_dt_min * 60 * 1e9)
    df = df.assign(_ti=ti)[dt_ok]
    if len(df) == 0:
        raise ValueError("No IGRA rows matched the stereo times within tolerance")

    stations = np.sort(df["station_idx"].astype(int).unique())
    levels = np.sort(df["pressure_hpa"].astype(float).unique())[::-1]  # descending hPa
    sta_pos = {int(s): i for i, s in enumerate(stations)}
    lev_pos = {float(p): i for i, p in enumerate(levels)}

    n_lev, n_time, n_sta = len(levels), len(times), len(stations)
    u = np.full((n_lev, n_time, n_sta), np.nan)
    v = np.full((n_lev, n_time, n_sta), np.nan)
    h = np.full((n_lev, n_time, n_sta), np.nan)

    si = df["station_idx"].astype(int).map(sta_pos).values
    li = df["pressure_hpa"].astype(float).map(lev_pos).values
    tt = df["_ti"].values
    u[li, tt, si] = df["u"].values
    v[li, tt, si] = df["v"].values
    h[li, tt, si] = df["height_m"].values

    # Per-station static fields (first occurrence).
    first = df.drop_duplicates("station_idx").set_index(
        df.drop_duplicates("station_idx")["station_idx"].astype(int)
    )
    lat = np.array([first.loc[int(s), "lat"] for s in stations], dtype=np.float64)
    lon = np.array([first.loc[int(s), "lon"] for s in stations], dtype=np.float64)
    row = np.array([first.loc[int(s), "row_19"] for s in stations], dtype=np.float64)
    col = np.array([first.loc[int(s), "col_19"] for s in stations], dtype=np.float64)

    ds = xr.Dataset(
        dict(
            u=(("level", "time", "station"), u),
            v=(("level", "time", "station"), v),
            geopotential_height=(("level", "time", "station"), h),
            lat=("station", lat),
            lon=("station", lon),
            row_19=("station", row),
            col_19=("station", col),
            pressure=("level", levels.astype(float)),
        ),
        coords=dict(level=levels.astype(float), time=times.values,
                    station=stations),
    )
    log.info(f"  IGRA: {n_sta} stations, {n_lev} levels, "
             f"{int(np.isfinite(u).any(axis=0).sum())} station-times with data")
    return ds


# ---------------------------------------------------------------------------
# NOAA DMW (AMV) loading from S3 — GOES-19
# ---------------------------------------------------------------------------
def _init_s3():
    return s3fs.S3FileSystem(anon=True)


def find_dmw_file(fs, time, band):
    """Find the GOES-19 DMW file on S3 closest to the given time and band."""
    doy = time.strftime("%j")
    year = time.strftime("%Y")
    hour = time.strftime("%H")
    prefix = f"{S3_BUCKET}/{DMW_PRODUCT}/{year}/{doy}/{hour}/"
    band_num = band[1:]
    try:
        files = fs.glob(f"{prefix}OR_{DMW_PRODUCT}-M6C{band_num}_{SAT_ID}_s*")
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


def amv_pixel_coords(lats, lons):
    """GOES-19 lat/lon -> pixel row/col (float)."""
    from stereo_winds.navigation import geodetic_to_fixed_grid, scanning_angle_to_pixel
    x_fg, y_fg = geodetic_to_fixed_grid(lats, lons, SAT)
    cols_f, rows_f = scanning_angle_to_pixel(x_fg, y_fg, SAT)
    return rows_f, cols_f


# ---------------------------------------------------------------------------
# Quad collocation
# ---------------------------------------------------------------------------
def quad_collocate(stereo_ds, igra_ds, amv_data, min_height=0, box_half=2,
                   sigma_h_max_low=1000, sigma_h_max_mid=2000,
                   sigma_h_max_high=1000, h_grad_max=3000, chi2_max=0.2):
    """Find points where stereo, sonde and AMV observe the same column.

    Anchored on IGRA stations (using the parquet's row_19/col_19 pixels).
    Records the scene timestamp per match so ERA5 can be attached afterward.
    """
    keys = [
        "u_stereo", "v_stereo", "h_stereo",
        "u_sonde", "v_sonde", "h_sonde", "pressure_sonde",
        "u_amv", "v_amv", "h_amv", "pressure_amv",
        "lat", "lon", "distance_amv_km", "time",
        "row_sta", "col_sta",
    ]
    accum = {k: [] for k in keys}

    # Station pixels straight from the parquet (full-disk, no chunk offset).
    sta_lats = igra_ds["lat"].values
    sta_lons = igra_ds["lon"].values
    sta_rows = igra_ds["row_19"].values
    sta_cols = igra_ds["col_19"].values
    on_grid = (
        np.isfinite(sta_rows) & np.isfinite(sta_cols)
        & (sta_rows >= 0) & (sta_rows < GRID)
        & (sta_cols >= 0) & (sta_cols < GRID)
    )
    grid_idx = np.where(on_grid)[0]
    if len(grid_idx) == 0:
        return {k: np.array([]) for k in keys}

    r_sta = np.round(sta_rows[grid_idx]).astype(int)
    c_sta = np.round(sta_cols[grid_idx]).astype(int)

    offsets = np.arange(-box_half, box_half + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()

    scene_times = stereo_ds.time.values
    n_times = len(scene_times)
    u_grid = stereo_ds["u_wind"].values
    v_grid = stereo_ds["v_wind"].values
    h_grid = stereo_ds["cloud_top_height"].values
    chi2_grid = stereo_ds["chi_squared"].values
    qf_grid = stereo_ds["quality_flag"].values
    sigh_grid = stereo_ds["sigma_h"].values

    pressure = igra_ds["pressure"].values
    sonde_u = igra_ds["u"].values[:, :, grid_idx]
    sonde_v = igra_ds["v"].values[:, :, grid_idx]
    sonde_h = igra_ds["geopotential_height"].values[:, :, grid_idx]
    sonde_valid = np.isfinite(sonde_h) & np.isfinite(sonde_u) & np.isfinite(sonde_v)

    amv_time_idx = amv_data["time_idx"]

    for t in range(n_times):
        amv_mask_t = amv_time_idx == t
        if amv_mask_t.sum() == 0:
            continue

        amv_lat_t = amv_data["lat"][amv_mask_t]
        amv_lon_t = amv_data["lon"][amv_mask_t]
        amv_u_t = amv_data["u"][amv_mask_t]
        amv_v_t = amv_data["v"][amv_mask_t]
        amv_h_t = amv_data["h"][amv_mask_t]
        amv_p_t = amv_data["pressure"][amv_mask_t]

        amv_rows_f, amv_cols_f = amv_pixel_coords(amv_lat_t, amv_lon_t)
        amv_on = np.isfinite(amv_cols_f) & np.isfinite(amv_rows_f)
        amv_r = np.where(amv_on, np.round(amv_rows_f).astype(int), -9999)
        amv_c = np.where(amv_on, np.round(amv_cols_f).astype(int), -9999)

        r_nb = np.clip(r_sta[:, None] + dy[None, :], 0, GRID - 1)
        c_nb = np.clip(c_sta[:, None] + dx[None, :], 0, GRID - 1)

        u_nb = u_grid[t][r_nb, c_nb]
        v_nb = v_grid[t][r_nb, c_nb]
        h_nb = h_grid[t][r_nb, c_nb]
        chi2_nb = chi2_grid[t][r_nb, c_nb]
        qf_nb = qf_grid[t][r_nb, c_nb]
        sigh_nb = sigh_grid[t][r_nb, c_nb]
        hgrad = _height_gradient(h_grid[t])
        hgrad_nb = hgrad[r_nb, c_nb]

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

        candidate_idx = np.where(stereo_ok)[0]
        if len(candidate_idx) == 0:
            continue

        for si_local in candidate_idx:
            rs = r_sta[si_local]
            cs = c_sta[si_local]
            hs = h_s[si_local]

            dr = amv_r.astype(np.float32) - rs
            dc = amv_c.astype(np.float32) - cs
            dist_px = np.sqrt(dr**2 + dc**2)
            near = amv_on & (dist_px <= AMV_RADIUS_PX)
            if not near.any():
                continue
            dist_near = np.where(near, dist_px, np.inf)
            best = int(np.argmin(dist_near))
            h_amv_best = amv_h_t[best]

            sonde_h_col = sonde_h[:, t, si_local]
            sonde_v_col = sonde_valid[:, t, si_local]
            h_diff_lev = np.abs(sonde_h_col - h_amv_best)
            h_diff_lev = np.where(sonde_v_col, h_diff_lev, np.inf)
            best_lev = int(np.argmin(h_diff_lev))
            if not np.isfinite(h_diff_lev[best_lev]):
                continue
            h_so = sonde_h_col[best_lev]

            above = (sonde_h_col > h_amv_best) & sonde_v_col
            below = (sonde_h_col < h_amv_best) & sonde_v_col
            if not (above.any() and below.any()):
                continue

            p_match = pressure[best_lev]
            max_h = 2000.0 if p_match <= 250 else 500.0
            if h_diff_lev[best_lev] > max_h:
                continue
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
            accum["time"].append(scene_times[t])
            accum["row_sta"].append(int(rs))
            accum["col_sta"].append(int(cs))

    if all(len(v) == 0 for v in accum.values()):
        return {k: np.array([]) for k in keys}
    out = {}
    for k, v in accum.items():
        out[k] = (np.asarray(v, dtype="datetime64[ns]") if k == "time"
                  else np.asarray(v, dtype=np.float64))
    return out


def attach_era5(all_matches, reader, levels=DEFAULT_LEVELS):
    """Sample ERA5 (u, v) at each match's column/height; fill u_era5/v_era5.

    Loads each unique scene time once across all bands (ERA5 is band-agnostic),
    samples at the station lat/lon and the matched sonde-level height.
    """
    # Collect all unique scene times across bands.
    all_times = []
    for m in all_matches.values():
        if len(m.get("time", [])):
            all_times.append(np.asarray(m["time"]))
    if not all_times:
        return
    uniq = np.unique(np.concatenate(all_times))

    # Initialize era5 arrays.
    for m in all_matches.values():
        n = len(m.get("u_stereo", []))
        m["u_era5"] = np.full(n, np.nan)
        m["v_era5"] = np.full(n, np.nan)

    for T in uniq:
        try:
            e1 = load_era5_single_time(reader, T, ERA5_LAT_BBOX, ERA5_LON_BBOX)
        except Exception as exc:
            log.warning(f"  ERA5 load failed for {T}: {exc}")
            continue
        for m in all_matches.values():
            if not len(m.get("time", [])):
                continue
            idx = np.where(np.asarray(m["time"]) == T)[0]
            if idx.size == 0:
                continue
            u, v = sample_era5(e1, m["lat"][idx], m["lon"][idx], m["h_sonde"][idx])
            m["u_era5"][idx] = u
            m["v_era5"][idx] = v


def attach_student(all_matches, student_zarr_pattern, box_half=2):
    """Sample the student winds at each teacher-accepted match's station pixel.

    Adds ``u_student``/``v_student``/``h_student`` to every band's match dict,
    evaluated on the SAME points the teacher QA already selected (no student-side
    QA gate) so all members share an identical N.  ``student_zarr_pattern`` must
    contain a ``{band}`` token (one full-disk student inference zarr per band).
    """
    offsets = np.arange(-box_half, box_half + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()
    for band, m in all_matches.items():
        n = len(m.get("u_stereo", []))
        m["u_student"] = np.full(n, np.nan)
        m["v_student"] = np.full(n, np.nan)
        m["h_student"] = np.full(n, np.nan)
        if n == 0:
            continue
        path = student_zarr_pattern.format(band=band)
        if not os.path.exists(path):
            log.warning(f"  student zarr missing for {band}: {path}")
            continue
        sds = xr.open_zarr(path, consolidated=False)
        st_times = sds.time.values
        rr = np.round(m["row_sta"]).astype(int)
        cc = np.round(m["col_sta"]).astype(int)
        mtimes = np.asarray(m["time"])
        H, W = sds["u_wind"].shape[-2:]
        for T in np.unique(mtimes):
            ti = np.where(st_times == T)[0]
            if ti.size == 0:
                continue
            ti = int(ti[0])
            u2d = sds["u_wind"].isel(time=ti).values
            v2d = sds["v_wind"].isel(time=ti).values
            h2d = sds["cloud_top_height"].isel(time=ti).values
            for j in np.where(mtimes == T)[0]:
                r0, c0 = rr[j] + dy, cc[j] + dx
                ok = (r0 >= 0) & (r0 < H) & (c0 >= 0) & (c0 < W)
                r0, c0 = r0[ok], c0[ok]
                ub = u2d[r0, c0]
                if np.isfinite(ub).any():
                    m["u_student"][j] = np.nanmedian(ub)
                    m["v_student"][j] = np.nanmedian(v2d[r0, c0])
                    m["h_student"][j] = np.nanmedian(h2d[r0, c0])
        sds.close()
        n_ok = int(np.isfinite(m["u_student"]).sum())
        log.info(f"  student attached for {band}: {n_ok}/{n} points with valid winds")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _pair_stats(u1, v1, u2, v2, h1=None, h2=None):
    finite = np.isfinite(u1) & np.isfinite(v1) & np.isfinite(u2) & np.isfinite(v2)
    if finite.sum() == 0:
        return {"n": 0, "rmsvd": np.nan, "speed_bias": np.nan,
                "corr_speed": np.nan, "h_rmse": np.nan, "h_bias": np.nan}
    u1, v1, u2, v2 = u1[finite], v1[finite], u2[finite], v2[finite]
    spd1 = np.sqrt(u1**2 + v1**2)
    spd2 = np.sqrt(u2**2 + v2**2)
    out = {
        "n": int(finite.sum()),
        "rmsvd": rmsvd(u1, v1, u2, v2),
        "speed_bias": speed_bias(u1, v1, u2, v2),
        "corr_speed": correlation(spd1, spd2),
        "h_rmse": np.nan, "h_bias": np.nan,
    }
    if h1 is not None and h2 is not None:
        h1f, h2f = h1[finite], h2[finite]
        hh = np.isfinite(h1f) & np.isfinite(h2f)
        if hh.sum() > 0:
            out["h_rmse"] = height_rmse(h1f[hh], h2f[hh])
            out["h_bias"] = float(np.mean(h1f[hh] - h2f[hh]))
    return out


PAIRS = [
    ("stereo", "sonde"), ("stereo", "amv"), ("stereo", "era5"),
    ("amv", "sonde"), ("era5", "sonde"), ("amv", "era5"),
    # Student member (present only with --student-zarr; guarded below).
    ("student", "sonde"), ("student", "amv"), ("student", "stereo"),
]


def compute_quad_stats(m):
    """All pairwise stats among {stereo, sonde, amv, era5}."""
    res = {"n": len(m.get("u_stereo", []))}
    for a, b in PAIRS:
        if f"u_{a}" not in m or f"u_{b}" not in m:
            continue
        res[f"{a}_vs_{b}"] = _pair_stats(
            m[f"u_{a}"], m[f"v_{a}"], m[f"u_{b}"], m[f"v_{b}"],
            m.get(f"h_{a}"), m.get(f"h_{b}"),
        )
    return res


def triple_collocation_errors(m):
    """Stoffelen-1998 TC error variances for {stereo, AMV, sonde} (independent)."""
    n = len(m.get("u_stereo", []))
    if n < 10:
        return None
    res = {}
    for comp in ["u", "v"]:
        s = m[f"{comp}_stereo"]; a = m[f"{comp}_amv"]; r = m[f"{comp}_sonde"]
        ok = np.isfinite(s) & np.isfinite(a) & np.isfinite(r)
        s, a, r = s[ok] - np.mean(s[ok]), a[ok] - np.mean(a[ok]), r[ok] - np.mean(r[ok])
        res[comp] = {
            "var_s": max(float(np.mean((s - a) * (s - r))), 0.0),
            "var_a": max(float(np.mean((a - s) * (a - r))), 0.0),
            "var_r": max(float(np.mean((r - s) * (r - a))), 0.0),
        }
    u, v = res["u"], res["v"]
    sq = lambda x: float(np.sqrt(max(x, 0.0)))
    return {
        "n": int(np.isfinite(m["u_stereo"]).sum()),
        "rmse_stereo": sq(u["var_s"] + v["var_s"]),
        "rmse_amv": sq(u["var_a"] + v["var_a"]),
        "rmse_sonde": sq(u["var_r"] + v["var_r"]),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_quad_summary(all_stats, months):
    print("\n" + "=" * 100)
    print(f"Quadruple Collocation — {months[0]} to {months[-1]}  (held-out test months)")
    print("=" * 100)
    hdr = f"{'band':<6s}{'pair':<18s}{'N':>7s}{'RMSVD':>9s}{'SpBias':>9s}{'corr':>7s}{'hRMSE':>9s}{'hBias':>9s}"
    print(hdr)
    print("-" * 100)
    for band, st in all_stats.items():
        if st.get("n", 0) == 0:
            print(f"{band:<6s}(no matches)")
            continue
        for a, b in PAIRS:
            key = f"{a}_vs_{b}"
            if key not in st:
                continue
            s = st[key]
            print(f"{band:<6s}{key:<18s}{s['n']:>7,}{s['rmsvd']:>9.2f}"
                  f"{s['speed_bias']:>+9.2f}{s['corr_speed']:>7.2f}"
                  f"{s['h_rmse']:>9.0f}{s['h_bias']:>+9.0f}")
        print("-" * 100)


def print_tc(all_tc):
    print("\nTriple-collocation error estimates (Stoffelen 1998) — {stereo, AMV, sonde}")
    print("ERA5 excluded: it assimilates AMVs & sondes (not independent).")
    print(f"{'band':<6s}{'N':>7s}{'rmse_stereo':>13s}{'rmse_amv':>11s}{'rmse_sonde':>12s}")
    print("-" * 50)
    for band, tc in all_tc.items():
        if tc is None:
            continue
        print(f"{band:<6s}{tc['n']:>7,}{tc['rmse_stereo']:>13.2f}"
              f"{tc['rmse_amv']:>11.2f}{tc['rmse_sonde']:>12.2f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
MEMBERS = ["stereo", "sonde", "amv", "era5"]
COLORS = {"stereo": "#d6604d", "sonde": "#1a1a1a", "amv": "#4393c3", "era5": "#5aae61"}


def plot_uv_scatter(m, band, out_dir):
    """Each reference (sonde/amv/era5) vs stereo, u and v."""
    refs = [r for r in ("sonde", "amv", "era5") if f"u_{r}" in m]
    fig, axes = plt.subplots(2, len(refs), figsize=(4.2 * len(refs), 8))
    axes = np.atleast_2d(axes)
    for j, ref in enumerate(refs):
        for i, comp in enumerate(("u", "v")):
            ax = axes[i, j]
            x = m[f"{comp}_stereo"]; y = m[f"{comp}_{ref}"]
            ok = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[ok], y[ok], s=6, alpha=0.3, color=COLORS[ref])
            lim = 60
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            r = correlation(x[ok], y[ok]) if ok.sum() else np.nan
            ax.set_title(f"{comp} stereo vs {ref}  r={r:.2f} N={ok.sum():,}", fontsize=9)
            ax.set_xlabel(f"stereo {comp} (m/s)"); ax.set_ylabel(f"{ref} {comp} (m/s)")
    fig.suptitle(f"Quad collocation u/v — {band}", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, f"quad_uv_{band}.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_speed_scatter(m, band, out_dir):
    refs = [r for r in ("sonde", "amv", "era5") if f"u_{r}" in m]
    fig, axes = plt.subplots(1, len(refs), figsize=(4.2 * len(refs), 4))
    axes = np.atleast_1d(axes)
    sp = lambda a: np.sqrt(m[f"u_{a}"] ** 2 + m[f"v_{a}"] ** 2)
    s_st = sp("stereo")
    for j, ref in enumerate(refs):
        ax = axes[j]; s_r = sp(ref)
        ok = np.isfinite(s_st) & np.isfinite(s_r)
        ax.scatter(s_st[ok], s_r[ok], s=6, alpha=0.3, color=COLORS[ref])
        ax.plot([0, 80], [0, 80], "k--", lw=0.8)
        ax.set_xlim(0, 80); ax.set_ylim(0, 80)
        ax.set_title(f"|V| stereo vs {ref}  N={ok.sum():,}", fontsize=9)
        ax.set_xlabel("stereo |V| (m/s)"); ax.set_ylabel(f"{ref} |V| (m/s)")
    fig.suptitle(f"Quad collocation speed — {band}", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, f"quad_speed_{band}.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True,
                    help="Directory with cached retrieval zarrs")
    ap.add_argument("--label", default="hreg1_step75",
                    help="Cache label prefix (default hreg1_step75)")
    ap.add_argument("--iter", type=int, default=12, dest="n_iter",
                    help="Solver iteration count in the cache filename (default 12)")
    ap.add_argument("--parquet", required=True,
                    help="IGRA collocation parquet (igra_all_collocation.parquet)")
    ap.add_argument("--months", nargs="+", default=DEFAULT_MONTHS,
                    help="Test months YYYY-MM (default 2025-10 2025-11)")
    ap.add_argument("--bands", nargs="+", default=DEFAULT_BANDS)
    ap.add_argument("--no-era5", action="store_true")
    ap.add_argument("--stereo-zarr", default=None,
                    help="Override: use this single multi-time zarr as the stereo "
                         "member (e.g. a student inference output) instead of the "
                         "label cache pattern. Only sensible with a single --bands.")
    ap.add_argument("--chi2-max", type=float, default=0.2,
                    help="Max chi-squared for the stereo QA gate (default 0.2, "
                         "tuned to the teacher's physics chi2). Raise (e.g. 1e9 to "
                         "disable) when the stereo member is a student whose "
                         "distilled chi2 is on a different scale.")
    ap.add_argument("--sigma-h-scale", type=float, default=1.0,
                    help="Multiply the sigma_h QA thresholds by this factor. Use "
                         ">1 for a student member whose distilled height "
                         "uncertainty is larger than the teacher's.")
    ap.add_argument("--student-zarr", default=None,
                    help="Attach the student winds as an extra member "
                         "(u_student/v_student/h_student) sampled at the SAME "
                         "teacher-accepted points, so all members share N. "
                         "Pattern with a '{band}' token (one zarr per band).")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "figures"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Reference times from the C14 cache (center-pixel-filled scenes).
    print("Finding filled stereo time steps (C14 cache)...", flush=True)
    ref = load_stereo(args.cache_dir, args.label, "C14", args.months, args.n_iter,
                      override_path=args.stereo_zarr)
    if ref is None:
        raise SystemExit("No C14 cache found — generate retrievals first.")
    center = ref["u_wind"].isel(y=GRID // 2, x=GRID // 2).values
    times = pd.DatetimeIndex(ref.time.values[np.isfinite(center)])
    print(f"  {len(times)} filled time steps over {args.months}", flush=True)

    print("Loading IGRA from parquet...", flush=True)
    igra_ds = load_igra(args.parquet, times, args.months)
    igra_ds.load()

    all_matches, all_stats = {}, {}
    for band in args.bands:
        print(f"\n--- {band} ---", flush=True)
        amv_data = load_dmw_for_times(times, band)
        if amv_data is None:
            print(f"  No DMW data for {band}")
            all_stats[band] = {"n": 0}
            continue
        stereo_ds = load_stereo(args.cache_dir, args.label, band, args.months,
                                args.n_iter, times=times,
                                override_path=args.stereo_zarr)
        if stereo_ds is None:
            print(f"  No stereo cache for {band}")
            all_stats[band] = {"n": 0}
            continue
        stereo_ds.load()
        matches = quad_collocate(stereo_ds, igra_ds, amv_data,
                                 min_height=MIN_HEIGHT.get(band, 0),
                                 chi2_max=args.chi2_max,
                                 sigma_h_max_low=1000 * args.sigma_h_scale,
                                 sigma_h_max_mid=2000 * args.sigma_h_scale,
                                 sigma_h_max_high=1000 * args.sigma_h_scale)
        print(f"  {band}: {len(matches['u_stereo']):,} stereo+sonde+AMV matches")
        all_matches[band] = matches

    # ERA5 as the fourth member (one load per unique scene time, across bands).
    if not args.no_era5 and all_matches:
        print("\nAttaching ERA5 (arco-era5)...", flush=True)
        reader = open_era5_reader()
        attach_era5(all_matches, reader)

    # Student as an extra member, sampled at the same teacher-accepted points.
    if args.student_zarr and all_matches:
        print("\nAttaching student winds at teacher-accepted points...", flush=True)
        attach_student(all_matches, args.student_zarr)

    for band, m in all_matches.items():
        all_stats[band] = compute_quad_stats(m)

    print_quad_summary(all_stats, args.months)
    all_tc = {b: triple_collocation_errors(m) for b, m in all_matches.items()
              if len(m.get("u_stereo", []))}
    print_tc(all_tc)

    print("\nGenerating plots...", flush=True)
    for band, m in all_matches.items():
        if len(m.get("u_stereo", [])) == 0:
            continue
        print("  " + plot_uv_scatter(m, band, args.out_dir))
        print("  " + plot_speed_scatter(m, band, args.out_dir))

    # Save matched arrays for downstream analysis.
    npz = os.path.join(args.out_dir, "quad_matches.npz")
    flat = {}
    for band, m in all_matches.items():
        for k, v in m.items():
            flat[f"{band}__{k}"] = (v.astype("datetime64[ns]").astype("int64")
                                    if k == "time" else v)
    np.savez(npz, **flat)
    print(f"\nSaved matches -> {npz}")
    print(f"Figures -> {args.out_dir}/")


if __name__ == "__main__":
    main()
