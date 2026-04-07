"""Collocate cached stereo retrievals with IGRA radiosondes.

Reads stereo Zarr stores produced by cache_stereo_retrievals.py and
compares against IGRA sonde profiles from ArrayLake.
"""

import sys
import os
import argparse
import glob

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "zeus"))

import arraylake
import numpy as np
import xarray as xr
from scipy.ndimage import sobel

from stereo_winds.config import GOES19_CONFIG
from stereo_winds.navigation import geodetic_to_fixed_grid, scanning_angle_to_pixel
from stereo_winds.validation.metrics import rmsvd, speed_bias, height_rmse, correlation


SAT = GOES19_CONFIG
QA = dict(chi2_max=0.2, sigma_h_max=5000.0, h_grad_max=3000.0,
          wind_speed_max=100.0, min_height=1000.0)


def evaluate_zarr(zarr_path, label, igra, igra_lat, igra_lon, grid_idx, rows_i, cols_i):
    """Collocate one cached retrieval store with IGRA."""
    print(f"\n=== {label}: {zarr_path} ===", flush=True)
    ds = xr.open_zarr(zarr_path)
    times = ds.time.values
    n_times = len(times)
    print(f"  {n_times} times", flush=True)

    all_matches = {k: [] for k in [
        "u_stereo", "v_stereo", "h_stereo",
        "u_sonde", "v_sonde", "h_sonde",
    ]}

    igra_times = igra.time.values

    for ti in range(n_times):
        t0 = times[ti]
        u_grid = ds["u_wind"].isel(time=ti).values
        v_grid = ds["v_wind"].isel(time=ti).values
        h = ds["cloud_top_height"].isel(time=ti).values
        chi2 = ds["chi_squared"].isel(time=ti).values
        sigma_h = ds["sigma_h"].isel(time=ti).values
        qf = ds["quality_flag"].isel(time=ti).values

        # QA
        h_filled = np.where(np.isfinite(h), h, 0.0)
        grad = np.sqrt(sobel(h_filled, axis=1)**2 + sobel(h_filled, axis=0)**2) / 8.0
        spd = np.sqrt(u_grid**2 + v_grid**2)
        qa = ((qf > 0) & np.isfinite(h) & np.isfinite(chi2)
              & (chi2 <= QA["chi2_max"])
              & np.isfinite(sigma_h) & (sigma_h <= QA["sigma_h_max"])
              & (grad <= QA["h_grad_max"])
              & np.isfinite(spd) & (spd <= QA["wind_speed_max"])
              & (h >= QA["min_height"]) & (h <= 20000))

        # Match IGRA time
        time_diff = np.abs(igra_times - t0.astype("datetime64[ns]"))
        nearest_idx = int(np.argmin(time_diff))
        if time_diff[nearest_idx] > np.timedelta64(6, "h"):
            continue

        sonde_u = igra["u"].values[:, nearest_idx, grid_idx]
        sonde_v = igra["v"].values[:, nearest_idx, grid_idx]
        sonde_h = igra["geopotential_height"].values[:, nearest_idx, grid_idx]

        r = rows_i[grid_idx]
        c = cols_i[grid_idx]

        # 5x5 neighborhood median
        box = 2
        offsets = np.arange(-box, box + 1)
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        dy, dx = dy.ravel(), dx.ravel()
        r_nb = np.clip(r[:, None] + dy[None, :], 0, 5423)
        c_nb = np.clip(c[:, None] + dx[None, :], 0, 5423)

        u_nb = u_grid[r_nb, c_nb]
        v_nb = v_grid[r_nb, c_nb]
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

        sonde_valid = np.isfinite(sonde_h) & np.isfinite(sonde_u) & np.isfinite(sonde_v)
        h_diff = np.abs(sonde_h - h_s[np.newaxis, :])
        h_diff = np.where(sonde_valid, h_diff, np.inf)
        best_lev = np.argmin(h_diff, axis=0)

        si = np.arange(len(grid_idx))
        u_son = sonde_u[best_lev, si]
        v_son = sonde_v[best_lev, si]
        h_son = sonde_h[best_lev, si]
        min_hdiff = h_diff[best_lev, si]

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

    for k in all_matches:
        all_matches[k] = np.concatenate(all_matches[k]) if all_matches[k] else np.array([])

    n = len(all_matches["u_stereo"])
    if n == 0:
        print("  No matches found!")
        return None

    rv = rmsvd(all_matches["u_stereo"], all_matches["v_stereo"],
               all_matches["u_sonde"], all_matches["v_sonde"])
    sb = speed_bias(all_matches["u_stereo"], all_matches["v_stereo"],
                    all_matches["u_sonde"], all_matches["v_sonde"])
    hr = height_rmse(all_matches["h_stereo"], all_matches["h_sonde"])
    hb = float(np.mean(all_matches["h_stereo"] - all_matches["h_sonde"]))
    hc = correlation(all_matches["h_stereo"], all_matches["h_sonde"])
    return dict(label=label, n=n, h_rmse=hr, h_bias=hb, h_corr=hc, rv=rv, sb=sb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_paths", nargs="+", help="Cached stereo Zarr paths")
    args = parser.parse_args()

    # Load IGRA station coords (lazy) then preload only needed times
    print("Loading IGRA 2026 metadata...", flush=True)
    client = arraylake.Client()
    repo = client.get_repo("zeus-ai/obs-xvec")
    session = repo.readonly_session("main")
    igra_lazy = xr.open_zarr(session.store, group="igra/2026", consolidated=False)
    igra_times_full = igra_lazy.time.values

    # Collect all unique IGRA time indices needed across all zarr stores
    needed_idx_set = set()
    for zp in args.zarr_paths:
        ds = xr.open_zarr(zp)
        for t in ds.time.values:
            idx = int(np.argmin(np.abs(igra_times_full - t.astype("datetime64[ns]"))))
            needed_idx_set.add(idx)
    needed_indices = sorted(needed_idx_set)
    print(f"  Loading {len(needed_indices)} IGRA time slices...", flush=True)
    igra = igra_lazy.isel(time=needed_indices).load()
    print(f"  Loaded {igra.sizes['time']} times, {igra.sizes['geometry']} stations", flush=True)

    igra_lat = igra["lat"].values
    igra_lon = igra["lon"].values

    # Station pixel coords on GOES-19 grid
    x_fg, y_fg = geodetic_to_fixed_grid(igra_lat, igra_lon, SAT)
    cols_f, rows_f = scanning_angle_to_pixel(x_fg, y_fg, SAT)
    on_grid = np.isfinite(cols_f) & np.isfinite(rows_f)
    cols_i = np.where(on_grid, np.round(cols_f).astype(int), -1)
    rows_i = np.where(on_grid, np.round(rows_f).astype(int), -1)
    on_grid &= (rows_i >= 0) & (rows_i < 5424) & (cols_i >= 0) & (cols_i < 5424)
    grid_idx = np.where(on_grid)[0]
    print(f"  {len(grid_idx)} stations on GOES-19 grid", flush=True)

    results = []
    for zp in args.zarr_paths:
        label = os.path.basename(zp).replace(".zarr", "")
        r = evaluate_zarr(zp, label, igra, igra_lat, igra_lon, grid_idx, rows_i, cols_i)
        if r:
            results.append(r)

    print("\n" + "=" * 100)
    print(f"{'Model':40s}  {'N':>6s}  {'H_RMSE':>7s}  {'H_bias':>7s}  {'H_corr':>6s}  {'RMSVD':>7s}  {'SpBias':>7s}")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:40s}  {r['n']:>6,}  {r['h_rmse']:>7.0f}m  {r['h_bias']:>+7.0f}m  "
              f"{r['h_corr']:>6.4f}  {r['rv']:>6.2f}m/s  {r['sb']:>+6.2f}m/s")
    print("-" * 100)


if __name__ == "__main__":
    main()
