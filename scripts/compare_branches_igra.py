"""Compare stereo wind retrievals from two ArrayLake branches against IGRA.

Runs the IGRA radiosonde collocation for both 'main' and 'solver-v2'
branches side by side, printing comparative statistics.
"""

import os
import sys

import arraylake
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import sobel

from stereo_winds.config import GOES16_CONFIG
from stereo_winds.navigation import geodetic_to_fixed_grid, scanning_angle_to_pixel
from stereo_winds.validation.metrics import correlation, height_rmse, rmsvd, speed_bias

SAT = GOES16_CONFIG
BANDS = ["C14"]
MONTHS = ["2024-01"]
MIN_HEIGHT = {"C14": 1000}
STEREO_TIMES = pd.date_range("2024-01-01", "2024-01-14", freq="12h")
BRANCHES = ["main", "solver-v3"]


def load_stereo(band, times, branch="main"):
    """Load stereo winds from ArrayLake for a specific branch."""
    client = arraylake.Client()
    repo = client.get_repo("zeus-ai/geo-stereo-winds")
    session = repo.readonly_session(branch)
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
    """Load IGRA 2024 radiosonde data."""
    client = arraylake.Client()
    repo = client.get_repo("zeus-ai/obs-xvec")
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, group="igra/2024", consolidated=False)
    return ds.sel(time=times)


def station_pixel_coords(lats, lons):
    x_fg, y_fg = geodetic_to_fixed_grid(lats, lons, SAT)
    cols_f, rows_f = scanning_angle_to_pixel(x_fg, y_fg, SAT)
    on_grid = np.isfinite(cols_f) & np.isfinite(rows_f)
    cols_i = np.where(on_grid, np.round(cols_f).astype(int), -1)
    rows_i = np.where(on_grid, np.round(rows_f).astype(int), -1)
    on_grid &= (rows_i >= 0) & (rows_i < 5424) & (cols_i >= 0) & (cols_i < 5424)
    return rows_i, cols_i, on_grid


def _height_gradient(h_2d):
    h_filled = np.where(np.isfinite(h_2d), h_2d, 0.0)
    gx = sobel(h_filled, axis=1)
    gy = sobel(h_filled, axis=0)
    return np.sqrt(gx**2 + gy**2) / 8.0


def collocate(stereo_ds, igra_ds, min_height=0, box_half=2,
              sigma_h_max=2000, h_grad_max=3000, chi2_max=0.2):
    """Collocate stereo winds with IGRA profiles (same as compare_stereo_igra.py)."""
    empty = {k: np.array([]) for k in [
        "u_stereo", "v_stereo", "h_stereo",
        "u_sonde", "v_sonde", "h_sonde",
        "pressure_matched", "h_diff",
        "sigma_h", "sigma_u", "sigma_v", "chi2",
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

    offsets = np.arange(-box_half, box_half + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()

    r_nb = np.clip(r[:, None] + dy[None, :], 0, 5423)
    c_nb = np.clip(c[:, None] + dx[None, :], 0, 5423)

    u_grid = stereo_ds["u_wind"].values
    v_grid = stereo_ds["v_wind"].values
    h_grid = stereo_ds["cloud_top_height"].values
    chi2_grid = stereo_ds["chi_squared"].values
    qf_grid = stereo_ds["quality_flag"].values
    sigh_grid = stereo_ds["sigma_h"].values

    hgrad_grid = np.stack([_height_gradient(h_grid[t]) for t in range(n_times)])

    u_nb = u_grid[:, r_nb, c_nb]
    v_nb = v_grid[:, r_nb, c_nb]
    h_nb = h_grid[:, r_nb, c_nb]
    chi2_nb = chi2_grid[:, r_nb, c_nb]
    qf_nb = qf_grid[:, r_nb, c_nb]
    sigh_nb = sigh_grid[:, r_nb, c_nb]
    hgrad_nb = hgrad_grid[:, r_nb, c_nb]

    spd_nb = np.sqrt(u_nb**2 + v_nb**2)
    qa_nb = (
        np.isfinite(u_nb) & np.isfinite(h_nb)
        & (qf_nb > 0.5)
        & (chi2_nb <= chi2_max)
        & (sigh_nb <= sigma_h_max)
        & (hgrad_nb <= h_grad_max)
        & (spd_nb <= 100)
        & (h_nb >= min_height) & (h_nb <= 20000)
    )

    u_nb = np.where(qa_nb, u_nb, np.nan)
    v_nb = np.where(qa_nb, v_nb, np.nan)
    h_nb = np.where(qa_nb, h_nb, np.nan)

    sigh_nb_q = np.where(qa_nb, sigh_nb, np.nan)
    chi2_nb_q = np.where(qa_nb, chi2_nb, np.nan)

    # Also extract sigma_u, sigma_v from the stereo grids
    sigu_grid = stereo_ds["sigma_u"].values
    sigv_grid = stereo_ds["sigma_v"].values
    sigu_nb = sigu_grid[:, r_nb, c_nb]
    sigv_nb = sigv_grid[:, r_nb, c_nb]
    sigu_nb_q = np.where(qa_nb, sigu_nb, np.nan)
    sigv_nb_q = np.where(qa_nb, sigv_nb, np.nan)

    with np.errstate(all="ignore"):
        u_s = np.nanmedian(u_nb, axis=2)
        v_s = np.nanmedian(v_nb, axis=2)
        h_s = np.nanmedian(h_nb, axis=2)
        sigh_s = np.nanmedian(sigh_nb_q, axis=2)
        sigu_s = np.nanmedian(sigu_nb_q, axis=2)
        sigv_s = np.nanmedian(sigv_nb_q, axis=2)
        chi2_s = np.nanmedian(chi2_nb_q, axis=2)

    n_valid = np.sum(qa_nb, axis=2)
    qa = np.isfinite(u_s) & np.isfinite(h_s) & (n_valid >= 3)

    sonde_u = igra_ds["u"].values[:, :, grid_idx]
    sonde_v = igra_ds["v"].values[:, :, grid_idx]
    sonde_h = igra_ds["geopotential_height"].values[:, :, grid_idx]
    sonde_valid = np.isfinite(sonde_h) & np.isfinite(sonde_u) & np.isfinite(sonde_v)

    h_diff_all = np.abs(sonde_h - h_s[np.newaxis, :, :])
    h_diff_all = np.where(sonde_valid, h_diff_all, np.inf)
    best_lev = np.argmin(h_diff_all, axis=0)
    ti, si = np.meshgrid(np.arange(n_times), np.arange(n_sta), indexing="ij")

    u_sonde_match = sonde_u[best_lev, ti, si]
    v_sonde_match = sonde_v[best_lev, ti, si]
    h_sonde_match = sonde_h[best_lev, ti, si]
    p_sonde_match = pressure[best_lev]
    min_h_diff = h_diff_all[best_lev, ti, si]

    above = (sonde_h > h_s[np.newaxis, :, :]) & sonde_valid
    below = (sonde_h < h_s[np.newaxis, :, :]) & sonde_valid
    has_above = np.any(above, axis=0)
    has_below = np.any(below, axis=0)

    above_500 = p_sonde_match <= 250
    max_h = np.where(above_500, 2000.0, 500.0)

    keep = (
        qa & has_above & has_below
        & (min_h_diff <= max_h)
        & np.isfinite(u_sonde_match) & np.isfinite(h_sonde_match)
    )

    kt, ks = np.where(keep)
    if len(kt) == 0:
        return empty

    return {
        "u_stereo": u_s[kt, ks], "v_stereo": v_s[kt, ks], "h_stereo": h_s[kt, ks],
        "u_sonde": u_sonde_match[kt, ks], "v_sonde": v_sonde_match[kt, ks],
        "h_sonde": h_sonde_match[kt, ks],
        "pressure_matched": p_sonde_match[kt, ks],
        "h_diff": h_s[kt, ks] - h_sonde_match[kt, ks],
        "sigma_h": sigh_s[kt, ks],
        "sigma_u": sigu_s[kt, ks],
        "sigma_v": sigv_s[kt, ks],
        "chi2": chi2_s[kt, ks],
    }


def compute_stats(matches):
    if len(matches["u_stereo"]) == 0:
        return {"n_matches": 0, "rmsvd": np.nan, "speed_bias": np.nan,
                "corr_speed": np.nan, "h_rmse": np.nan, "h_bias": np.nan}
    spd_s = np.sqrt(matches["u_stereo"]**2 + matches["v_stereo"]**2)
    spd_r = np.sqrt(matches["u_sonde"]**2 + matches["v_sonde"]**2)
    return {
        "n_matches": len(matches["u_stereo"]),
        "rmsvd": rmsvd(matches["u_stereo"], matches["v_stereo"],
                        matches["u_sonde"], matches["v_sonde"]),
        "speed_bias": speed_bias(matches["u_stereo"], matches["v_stereo"],
                                  matches["u_sonde"], matches["v_sonde"]),
        "corr_speed": correlation(spd_s, spd_r),
        "h_rmse": height_rmse(matches["h_stereo"], matches["h_sonde"]),
        "h_bias": float(np.mean(matches["h_stereo"] - matches["h_sonde"])),
    }


def main():
    # Find times that have data on BOTH branches
    print("Finding filled time steps on each branch...")
    branch_times = {}
    for branch in BRANCHES:
        ref_ds = load_stereo("C14", STEREO_TIMES, branch=branch)
        center = ref_ds["u_wind"].isel(y=2712, x=2712).values
        filled = ref_ds.time.values[np.isfinite(center)]
        branch_times[branch] = set(filled)
        print(f"  {branch}: {len(filled)} filled")

    # Use intersection of times
    common = sorted(branch_times[BRANCHES[0]] & branch_times[BRANCHES[1]])
    if not common:
        print("No common filled time steps — aborting.")
        return
    times = pd.DatetimeIndex(common)
    print(f"  Common: {len(times)} time steps")

    # Load IGRA once
    print("\nLoading IGRA data...")
    igra_ds = load_igra(times)
    igra_ds.load()
    print(f"  IGRA: {dict(igra_ds.sizes)}")

    # Run collocation for each branch and band
    results = {}   # {branch: {band: stats}}
    all_matches = {}  # {branch: {band: matches}}
    for branch in BRANCHES:
        results[branch] = {}
        all_matches[branch] = {}
        for band in BANDS:
            print(f"\n[{branch}] Processing {band}...")
            stereo_ds = load_stereo(band, times, branch=branch)
            stereo_ds.load()
            matches = collocate(stereo_ds, igra_ds, min_height=MIN_HEIGHT.get(band, 0))
            stats = compute_stats(matches)
            results[branch][band] = stats
            all_matches[branch][band] = matches
            print(f"  {band}: {stats['n_matches']} matches, RMSVD={stats['rmsvd']:.2f}")

    # Print side-by-side comparison
    print("\n" + "=" * 100)
    print(f"Branch Comparison: Stereo vs IGRA — {STEREO_TIMES[0].date()} to {STEREO_TIMES[-1].date()}")
    print(f"Common time steps: {len(times)}")
    print("=" * 100)
    print(f"{'Band':>6s} {'Branch':>12s} {'N':>6s} {'RMSVD':>8s} {'SpdBias':>8s} "
          f"{'r(speed)':>8s} {'hRMSE':>8s} {'hBias':>8s}")
    print("-" * 100)
    for band in BANDS:
        for branch in BRANCHES:
            s = results[branch][band]
            if s["n_matches"] == 0:
                print(f"{band:>6s} {branch:>12s} {'0':>6s}" + " {:>8s}".format("--") * 5)
            else:
                print(f"{band:>6s} {branch:>12s} {s['n_matches']:>6d} "
                      f"{s['rmsvd']:>8.2f} {s['speed_bias']:>+8.2f} "
                      f"{s['corr_speed']:>8.3f} {s['h_rmse']:>8.0f} "
                      f"{s['h_bias']:>+8.0f}")
        print()
    print("=" * 100)

    # --- RMSVD vs sigma_h analysis ---
    print("\n" + "=" * 110)
    print("Uncertainty Calibration: RMSVD and Height RMSE binned by sigma_h")
    print("=" * 110)

    sigma_h_edges = [0, 50, 100, 200, 500, 1000, 2000]
    for band in BANDS:
        print(f"\n  {band}:")
        print(f"  {'Branch':>12s} {'sigma_h bin':>16s} {'N':>6s} "
              f"{'RMSVD':>8s} {'hRMSE':>8s} {'med_sigma_h':>12s}")
        print("  " + "-" * 90)
        for branch in BRANCHES:
            m = all_matches[branch][band]
            if len(m["u_stereo"]) == 0:
                continue
            sh = m["sigma_h"]
            for i in range(len(sigma_h_edges) - 1):
                lo, hi = sigma_h_edges[i], sigma_h_edges[i + 1]
                sel = (sh >= lo) & (sh < hi)
                n = int(sel.sum())
                if n < 3:
                    print(f"  {branch:>12s} {f'[{lo}-{hi})':>16s} {n:>6d} "
                          f"{'--':>8s} {'--':>8s} {'--':>12s}")
                    continue
                rv = rmsvd(m["u_stereo"][sel], m["v_stereo"][sel],
                           m["u_sonde"][sel], m["v_sonde"][sel])
                hr = height_rmse(m["h_stereo"][sel], m["h_sonde"][sel])
                med_sh = float(np.median(sh[sel]))
                print(f"  {branch:>12s} {f'[{lo}-{hi})':>16s} {n:>6d} "
                      f"{rv:>8.2f} {hr:>8.0f} {med_sh:>12.0f}")
            # Also show >= last edge
            sel = sh >= sigma_h_edges[-1]
            n = int(sel.sum())
            if n >= 3:
                rv = rmsvd(m["u_stereo"][sel], m["v_stereo"][sel],
                           m["u_sonde"][sel], m["v_sonde"][sel])
                hr = height_rmse(m["h_stereo"][sel], m["h_sonde"][sel])
                med_sh = float(np.median(sh[sel]))
                print(f"  {branch:>12s} {f'>={sigma_h_edges[-1]}':>16s} {n:>6d} "
                      f"{rv:>8.2f} {hr:>8.0f} {med_sh:>12.0f}")

    # --- Sigma distribution summary ---
    print("\n" + "=" * 90)
    print("Sigma Distribution Summary")
    print("=" * 90)
    print(f"  {'Branch':>12s} {'N':>6s} {'med_sig_h':>10s} {'med_sig_u':>10s} "
          f"{'med_sig_v':>10s} {'med_chi2':>10s}")
    print("  " + "-" * 70)
    for branch in BRANCHES:
        m = all_matches[branch]["C14"]
        if len(m["u_stereo"]) == 0:
            continue
        print(f"  {branch:>12s} {len(m['sigma_h']):>6d} "
              f"{np.median(m['sigma_h']):>10.1f} "
              f"{np.median(m['sigma_u']):>10.4f} "
              f"{np.median(m['sigma_v']):>10.4f} "
              f"{np.median(m['chi2']):>10.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
