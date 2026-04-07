"""Evaluate RAFT checkpoints against IGRA radiosondes.

Runs RAFT + solver on scenes from the GOES-19/18 Zarr cubes,
then collocates with IGRA sonde profiles from ArrayLake.
"""

import sys
import os
import glob

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "zeus"))

import arraylake
import numpy as np
import torch
import xarray as xr
from scipy.ndimage import sobel

from stereo_winds.config import GOES19_CONFIG, GOES18_CONFIG
from stereo_winds.solver import build_design_matrix, solve_stereo_winds, pixels_to_wind_ms
from stereo_winds.disparity import StereoDisparity
from stereo_winds.time_model import compute_scene_times
from stereo_winds.navigation import geodetic_to_fixed_grid, scanning_angle_to_pixel
from stereo_winds.validation.metrics import rmsvd, speed_bias, height_rmse, correlation

# --- Config ---
DATA_DIR = os.path.join(BASE, "data", "stereo_training")
SAT_A = GOES19_CONFIG
SAT_B = GOES18_CONFIG
DT_MINUTES = 10.0

# Parallax
pdata = np.load(os.path.join(DATA_DIR, "parallax_goes19_goes18.npz"))
w_u, w_v = pdata["w_u"], pdata["w_v"]
scene_times = compute_scene_times(None, DT_MINUTES, SAT_A, SAT_B)
H_matrix = build_design_matrix(
    w_u, w_v,
    dt_a_minus=scene_times["A_minus"], dt_a_plus=scene_times["A_plus"],
    dt_b_minus=scene_times["B_minus"], dt_b_plus=scene_times["B_plus"],
)

# Load Zarr stores for Jan + Feb 2026
ds_a_list = [
    xr.open_zarr(os.path.join(DATA_DIR, "goes19_C14_202601.zarr")),
    xr.open_zarr(os.path.join(DATA_DIR, "goes19_C14_202602.zarr")),
]
ds_b_list = [
    xr.open_zarr(os.path.join(DATA_DIR, "goes18_remap_goes19_C14_202601.zarr")),
    xr.open_zarr(os.path.join(DATA_DIR, "goes18_remap_goes19_C14_202602.zarr")),
]

all_times_a = set()
all_times_b = set()
for d in ds_a_list:
    all_times_a.update(d.time.values)
for d in ds_b_list:
    all_times_b.update(d.time.values)

delta = np.timedelta64(10, "m")

# Pick times at 00Z and 12Z (sonde launch times) for Jan + Feb 2026
eval_times = []
for month, days in [("2026-01", 31), ("2026-02", 28)]:
    for day in range(1, days + 1):
        for hour in [0, 12]:
            t0 = np.datetime64(f"{month}-{day:02d}T{hour:02d}:00")
            if (t0 in all_times_a and t0 in all_times_b
                    and t0 - delta in all_times_a and t0 - delta in all_times_b
                    and t0 + delta in all_times_a and t0 + delta in all_times_b):
                eval_times.append(t0)

# Limit to ~10 scenes for fast eval
eval_times = eval_times[:10]
print(f"Using {len(eval_times)} eval times")

# Load IGRA — preload only the needed time indices into memory
print("Loading IGRA 2026...", flush=True)
client = arraylake.Client()
repo = client.get_repo("zeus-ai/obs-xvec")
session = repo.readonly_session("main")
igra_lazy = xr.open_zarr(session.store, group="igra/2026", consolidated=False)
igra_times_full = igra_lazy.time.values

# For each eval time, find the nearest IGRA time index
needed_indices = sorted(set(
    int(np.argmin(np.abs(igra_times_full - t.astype("datetime64[ns]"))))
    for t in eval_times
))
print(f"  Loading {len(needed_indices)} IGRA time slices into memory...", flush=True)
igra = igra_lazy.isel(time=needed_indices).load()
print(f"  Loaded {igra.sizes['time']} times, {igra.sizes['geometry']} stations", flush=True)
igra_lat = igra["lat"].values
igra_lon = igra["lon"].values

# Station pixel coords
lats = igra_lat
lons = igra_lon
x_fg, y_fg = geodetic_to_fixed_grid(lats, lons, SAT_A)
cols_f, rows_f = scanning_angle_to_pixel(x_fg, y_fg, SAT_A)
on_grid = np.isfinite(cols_f) & np.isfinite(rows_f)
cols_i = np.where(on_grid, np.round(cols_f).astype(int), -1)
rows_i = np.where(on_grid, np.round(rows_f).astype(int), -1)
on_grid &= (rows_i >= 0) & (rows_i < 5424) & (cols_i >= 0) & (cols_i < 5424)
grid_idx = np.where(on_grid)[0]
print(f"  {len(grid_idx)} IGRA stations on GOES-19 grid")

# QA thresholds
QA = dict(chi2_max=0.2, sigma_h_max=5000.0, h_grad_max=3000.0,
          wind_speed_max=100.0, min_height=1000.0)


def histogram_equalize(image, n_bins=500):
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image)
    vals = image[finite]
    hist, bins = np.histogram(vals, n_bins, density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    out = np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape)
    out[~finite] = 0.0
    return out.astype(np.float32)


def run_checkpoint(compat_path, label):
    """Run RAFT + solver on all eval times, collocate with IGRA."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {compat_path}")
    print(f"{'='*60}")

    disp = StereoDisparity(
        model_ckpt_path=compat_path, tile_size=512, overlap=256,
        batch_size=8, device="cuda",
    )

    all_matches = {k: [] for k in [
        "u_stereo", "v_stereo", "h_stereo",
        "u_sonde", "v_sonde", "h_sonde",
    ]}

    for ti, t0 in enumerate(eval_times):
        t_m = t0 - delta
        t_p = t0 + delta

        # Find the right Zarr store for this time
        def _load(ds_list, t):
            for ds in ds_list:
                if t in ds.time.values:
                    return ds.Rad.sel(time=t).values.astype(np.float32)
            raise KeyError(f"Time {t} not found in any store")

        a_minus = _load(ds_a_list, t_m)
        a0 = _load(ds_a_list, t0)
        a_plus = _load(ds_a_list, t_p)
        b_minus = _load(ds_b_list, t_m)
        b_plus = _load(ds_b_list, t_p)

        valid = (np.isfinite(a0) & np.isfinite(a_minus) & np.isfinite(a_plus)
                 & np.isfinite(b_minus) & np.isfinite(b_plus))

        # Histogram equalize for RAFT
        images = {
            "A_minus": histogram_equalize(a_minus),
            "A0": histogram_equalize(a0),
            "A_plus": histogram_equalize(a_plus),
            "B_minus": histogram_equalize(b_minus),
            "B_plus": histogram_equalize(b_plus),
        }

        flows = disp.compute_all(images)
        for k in flows:
            flows[k][:, ~valid] = np.nan

        sol = solve_stereo_winds(flows, H_matrix, sat_a=SAT_A, sat_b=SAT_B, n_iter=3)
        u_ms, v_ms = pixels_to_wind_ms(sol["V_u"], sol["V_v"], SAT_A, dt_seconds=1.0)
        h = sol["h"]
        qf = sol["quality_flag"].copy()
        qf[~valid] = 0.0
        chi2 = sol["chi2"]
        sigma_h = sol["sigma_h"]

        # QA mask
        h_filled = np.where(np.isfinite(h), h, 0.0)
        grad = np.sqrt(sobel(h_filled, axis=1)**2 + sobel(h_filled, axis=0)**2) / 8.0
        spd = np.sqrt(u_ms**2 + v_ms**2)
        qa = ((qf > 0) & np.isfinite(h) & np.isfinite(chi2)
              & (chi2 <= QA["chi2_max"])
              & np.isfinite(sigma_h) & (sigma_h <= QA["sigma_h_max"])
              & (grad <= QA["h_grad_max"])
              & np.isfinite(spd) & (spd <= QA["wind_speed_max"])
              & (h >= QA["min_height"]) & (h <= 20000))

        # Match to IGRA at this time (igra is already in memory)
        igra_times = igra.time.values
        t0_dt = t0.astype("datetime64[ns]")
        time_diff = np.abs(igra_times - t0_dt)
        nearest_idx = int(np.argmin(time_diff))
        if time_diff[nearest_idx] > np.timedelta64(6, "h"):
            continue

        sonde_u = igra["u"].values[:, nearest_idx, grid_idx]       # (n_plevs, n_sta)
        sonde_v = igra["v"].values[:, nearest_idx, grid_idx]
        sonde_h = igra["geopotential_height"].values[:, nearest_idx, grid_idx]

        r = rows_i[grid_idx]
        c = cols_i[grid_idx]

        # 5x5 neighborhood median for stereo
        box = 2
        offsets = np.arange(-box, box + 1)
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        dy, dx = dy.ravel(), dx.ravel()

        r_nb = np.clip(r[:, None] + dy[None, :], 0, 5423)
        c_nb = np.clip(c[:, None] + dx[None, :], 0, 5423)

        u_nb = u_ms[r_nb, c_nb]
        v_nb = v_ms[r_nb, c_nb]
        h_nb = h[r_nb, c_nb]
        qa_nb = qa[r_nb, c_nb]

        u_nb = np.where(qa_nb, u_nb, np.nan)
        v_nb = np.where(qa_nb, v_nb, np.nan)
        h_nb = np.where(qa_nb, h_nb, np.nan)

        with np.errstate(all="ignore"):
            u_s = np.nanmedian(u_nb, axis=1)
            v_s = np.nanmedian(v_nb, axis=1)
            h_s = np.nanmedian(h_nb, axis=1)

        n_valid = np.sum(qa_nb, axis=1)
        sta_ok = np.isfinite(u_s) & np.isfinite(h_s) & (n_valid >= 3)

        # Match to nearest sonde level
        sonde_valid = np.isfinite(sonde_h) & np.isfinite(sonde_u) & np.isfinite(sonde_v)
        h_diff = np.abs(sonde_h - h_s[np.newaxis, :])
        h_diff = np.where(sonde_valid, h_diff, np.inf)
        best_lev = np.argmin(h_diff, axis=0)

        si = np.arange(len(grid_idx))
        u_son = sonde_u[best_lev, si]
        v_son = sonde_v[best_lev, si]
        h_son = sonde_h[best_lev, si]
        min_hdiff = h_diff[best_lev, si]

        # Bracketing check
        above = (sonde_h > h_s[np.newaxis, :]) & sonde_valid
        below = (sonde_h < h_s[np.newaxis, :]) & sonde_valid
        has_bracket = np.any(above, axis=0) & np.any(below, axis=0)

        keep = (sta_ok & has_bracket & (min_hdiff <= 2000)
                & np.isfinite(u_son) & np.isfinite(h_son))

        if keep.sum() > 0:
            all_matches["u_stereo"].append(u_s[keep])
            all_matches["v_stereo"].append(v_s[keep])
            all_matches["h_stereo"].append(h_s[keep])
            all_matches["u_sonde"].append(u_son[keep])
            all_matches["v_sonde"].append(v_son[keep])
            all_matches["h_sonde"].append(h_son[keep])

        n_so_far = sum(len(x) for x in all_matches["u_stereo"])
        print(f"  [{ti+1}/{len(eval_times)}] {n_so_far} matches so far", flush=True)

    # Concatenate
    for k in all_matches:
        all_matches[k] = np.concatenate(all_matches[k]) if all_matches[k] else np.array([])

    n = len(all_matches["u_stereo"])
    if n == 0:
        print(f"  No matches found!")
        return

    rv = rmsvd(all_matches["u_stereo"], all_matches["v_stereo"],
               all_matches["u_sonde"], all_matches["v_sonde"])
    sb = speed_bias(all_matches["u_stereo"], all_matches["v_stereo"],
                    all_matches["u_sonde"], all_matches["v_sonde"])
    hr = height_rmse(all_matches["h_stereo"], all_matches["h_sonde"])
    hb = float(np.mean(all_matches["h_stereo"] - all_matches["h_sonde"]))
    hc = correlation(all_matches["h_stereo"], all_matches["h_sonde"])

    print(f"\n  {label:40s}")
    print(f"  N={n:>6,}  H_RMSE={hr:>7.0f}m  H_bias={hb:>+7.0f}m  "
          f"H_corr={hc:.4f}  RMSVD={rv:>6.2f}m/s  SpBias={sb:>+6.2f}m/s")
    return dict(n=n, h_rmse=hr, h_bias=hb, h_corr=hc, rv=rv, sb=sb)


# --- Checkpoints to evaluate ---
PRETRAINED = os.path.join(BASE, "zeus/zeus/networks/weights/raft-128.202509.epoch1434.ckpt")

checkpoints = [
    (PRETRAINED, "Pretrained"),
    (os.path.join(BASE, "stereo-winds-v2/9uvz4rkm/checkpoints/stepstep=00400.ckpt"),
     "Containment step400"),
]

# Run evaluations — StereoDisparity handles checkpoint conversion
results = {}
for ckpt, label in checkpoints:
    r = run_checkpoint(ckpt, label)
    if r:
        results[label] = r

# Summary
print("\n" + "=" * 100)
print(f"{'Model':40s}  {'N':>6s}  {'H_RMSE':>7s}  {'H_bias':>7s}  "
      f"{'H_corr':>6s}  {'RMSVD':>7s}  {'SpBias':>7s}")
print("-" * 100)
for label, r in results.items():
    print(f"{label:40s}  {r['n']:>6,}  {r['h_rmse']:>7.0f}m  {r['h_bias']:>+7.0f}m  "
          f"{r['h_corr']:>6.4f}  {r['rv']:>6.2f}m/s  {r['sb']:>+6.2f}m/s")
print("-" * 100)
