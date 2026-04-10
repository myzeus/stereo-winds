"""Evaluate stereo retrievals at radiosonde stations against IGRA + ERA5.

For each stereo NetCDF file, extracts winds at IGRA station locations,
matches to the closest sonde level, and also loads ERA5 at those same
points. Reports RMSVD for stereo-vs-sonde and ERA5-vs-sonde side by side.

Usage:
    python scripts/eval_at_sondes.py \
        output/4way_eval/pretrained_20260108_1200.nc \
        output/4way_eval/finetuned_20260108_1200.nc
"""

import argparse
import logging
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import arraylake
import numpy as np
import xarray as xr
from scipy.ndimage import sobel

from stereo_winds.config import GOES19_CONFIG
from stereo_winds.navigation import geodetic_to_fixed_grid, scanning_angle_to_pixel
from stereo_winds.validation.metrics import rmsvd, speed_bias, height_rmse, correlation
from stereo_winds.validation.era5 import load_era5_for_stereo, resolve_sat_config

logger = logging.getLogger(__name__)

SAT = GOES19_CONFIG
QA = dict(chi2_max=0.2, sigma_h_max=5000.0, h_grad_max=3000.0,
          wind_speed_max=100.0, min_height=1000.0)


def evaluate_at_sondes(stereo_path: str, igra, grid_idx, rows_i, cols_i, era5_ds=None):
    """Evaluate one stereo NetCDF at sonde stations."""
    stereo = xr.open_dataset(stereo_path, engine="h5netcdf").load()
    label = os.path.basename(stereo_path).replace(".nc", "")

    t0 = np.asarray(stereo.time.values).astype("datetime64[ns]")
    igra_times = igra.time.values
    nearest_idx = int(np.argmin(np.abs(igra_times - t0)))
    if np.abs(igra_times[nearest_idx] - t0) > np.timedelta64(6, "h"):
        print(f"  No IGRA time near {t0}", flush=True)
        return None

    # Stereo fields
    u_grid = stereo["u_wind"].values
    v_grid = stereo["v_wind"].values
    h_grid = stereo["cloud_top_height"].values
    chi2_grid = stereo["chi_squared"].values
    sigma_h_grid = stereo["sigma_h"].values
    qf_grid = stereo["quality_flag"].values

    # QA
    h_filled = np.where(np.isfinite(h_grid), h_grid, 0.0)
    grad = np.sqrt(sobel(h_filled, axis=1)**2 + sobel(h_filled, axis=0)**2) / 8.0
    spd = np.sqrt(u_grid**2 + v_grid**2)
    qa = ((qf_grid > 0) & np.isfinite(h_grid) & np.isfinite(chi2_grid)
          & (chi2_grid <= QA["chi2_max"])
          & np.isfinite(sigma_h_grid) & (sigma_h_grid <= QA["sigma_h_max"])
          & (grad <= QA["h_grad_max"])
          & np.isfinite(spd) & (spd <= QA["wind_speed_max"])
          & (h_grid >= QA["min_height"]) & (h_grid <= 20000))

    # Sonde data
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
    h_nb = h_grid[r_nb, c_nb]
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

    # Match to closest sonde level with bracketing
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
    keep = sta_ok & has_bracket & (min_hdiff <= 2000) & np.isfinite(u_son) & np.isfinite(h_son)

    result = {
        "label": label,
        "stereo_u": u_s[keep], "stereo_v": v_s[keep], "stereo_h": h_s[keep],
        "sonde_u": u_son[keep], "sonde_v": v_son[keep], "sonde_h": h_son[keep],
        "n": int(keep.sum()),
        "lat": igra["lat"].values[grid_idx[keep]],
        "lon": igra["lon"].values[grid_idx[keep]],
    }

    # ERA5 at matched sonde locations/heights
    if era5_ds is not None:
        era5_u_all = era5_ds["u_component_of_wind"].values  # (levels, lat, lon)
        era5_v_all = era5_ds["v_component_of_wind"].values
        era5_h_all = era5_ds["geometric_height"].values     # (levels, lat, lon)
        era5_lat = era5_ds["lat"].values if "lat" in era5_ds.coords else era5_ds["latitude"].values
        era5_lon = era5_ds["lon"].values if "lon" in era5_ds.coords else era5_ds["longitude"].values

        era5_u_match = []
        era5_v_match = []
        for i in np.where(keep)[0]:
            lat_i = igra["lat"].values[grid_idx[i]]
            lon_i = igra["lon"].values[grid_idx[i]]
            h_target = h_son[i]
            # Find nearest ERA5 grid point
            lat_idx = int(np.argmin(np.abs(era5_lat - lat_i)))
            lon_idx = int(np.argmin(np.abs(era5_lon - lon_i)))
            # Find closest ERA5 level by height
            era5_h_col = era5_h_all[:, lat_idx, lon_idx]
            lev_idx = int(np.argmin(np.abs(era5_h_col - h_target)))
            era5_u_match.append(era5_u_all[lev_idx, lat_idx, lon_idx])
            era5_v_match.append(era5_v_all[lev_idx, lat_idx, lon_idx])
        result["era5_u"] = np.array(era5_u_match)
        result["era5_v"] = np.array(era5_v_match)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stereo_files", nargs="+", help="Stereo NetCDF files")
    parser.add_argument("--no-era5", action="store_true", help="Skip ERA5 comparison")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load IGRA
    print("Loading IGRA...", flush=True)
    client = arraylake.Client()
    repo = client.get_repo("zeus-ai/obs-xvec")
    session = repo.readonly_session("main")
    igra_lazy = xr.open_zarr(session.store, group="igra/2026", consolidated=False)

    # Find needed times
    needed = set()
    for f in args.stereo_files:
        ds = xr.open_dataset(f, engine="h5netcdf")
        t0 = np.asarray(ds.time.values).astype("datetime64[ns]")
        idx = int(np.argmin(np.abs(igra_lazy.time.values - t0)))
        needed.add(idx)
        ds.close()
    igra = igra_lazy.isel(time=sorted(needed)).load()
    print(f"  Loaded {igra.sizes['time']} times, {igra.sizes['geometry']} stations", flush=True)

    # Station pixel coords
    lats = igra["lat"].values
    lons = igra["lon"].values
    x_fg, y_fg = geodetic_to_fixed_grid(lats, lons, SAT)
    cols_f, rows_f = scanning_angle_to_pixel(x_fg, y_fg, SAT)
    on_grid = np.isfinite(cols_f) & np.isfinite(rows_f)
    cols_i = np.where(on_grid, np.round(cols_f).astype(int), -1)
    rows_i = np.where(on_grid, np.round(rows_f).astype(int), -1)
    on_grid &= (rows_i >= 0) & (rows_i < 5424) & (cols_i >= 0) & (cols_i < 5424)
    grid_idx = np.where(on_grid)[0]
    print(f"  {len(grid_idx)} stations on grid", flush=True)

    # Load ERA5 (once, shared across checkpoints)
    era5_ds = None
    if not args.no_era5:
        print("Loading ERA5...", flush=True)
        first_stereo = xr.open_dataset(args.stereo_files[0], engine="h5netcdf").load()
        try:
            era5_ds = load_era5_for_stereo(first_stereo)
            print(f"  ERA5 loaded: {dict(era5_ds.sizes)}", flush=True)
        except Exception as e:
            print(f"  ERA5 load failed: {e}", flush=True)
        first_stereo.close()

    # Evaluate each checkpoint
    results = []
    for f in args.stereo_files:
        print(f"\nEvaluating: {os.path.basename(f)}", flush=True)
        r = evaluate_at_sondes(f, igra, grid_idx, rows_i, cols_i, era5_ds)
        if r:
            results.append(r)

    # Summary
    print("\n" + "=" * 110)
    has_era5 = results and "era5_u" in results[0]
    if has_era5:
        print(f"{'Model':<40s}  {'N':>5s}  {'Stereo RMSVD':>13s}  {'Stereo SpBias':>13s}  {'ERA5 RMSVD':>11s}  {'ERA5 SpBias':>11s}")
    else:
        print(f"{'Model':<40s}  {'N':>5s}  {'Stereo RMSVD':>13s}  {'Stereo SpBias':>13s}")
    print("-" * 110)
    for r in results:
        rv_s = rmsvd(r["stereo_u"], r["stereo_v"], r["sonde_u"], r["sonde_v"])
        sb_s = speed_bias(r["stereo_u"], r["stereo_v"], r["sonde_u"], r["sonde_v"])
        line = f"{r['label']:<40s}  {r['n']:>5,}  {rv_s:>12.2f}m/s  {sb_s:>+12.2f}m/s"
        if has_era5 and "era5_u" in r:
            rv_e = rmsvd(r["era5_u"], r["era5_v"], r["sonde_u"], r["sonde_v"])
            sb_e = speed_bias(r["era5_u"], r["era5_v"], r["sonde_u"], r["sonde_v"])
            line += f"  {rv_e:>10.2f}m/s  {sb_e:>+10.2f}m/s"
        print(line)
    print("-" * 110)


if __name__ == "__main__":
    main()
