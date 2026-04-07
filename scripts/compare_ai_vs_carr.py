"""1-to-1 comparison of AI (RAFT) vs Carr et al. stereo wind retrievals.

For each Carr retrieval site, finds the nearest AI pixel via geodetic-to-pixel
mapping and compares height, u/v wind, and speed. Produces 5 comparison figures
and prints a summary statistics table.

Scene: GOES-16/18, 2025-01-08 19:00 UTC, C14 (IR) + C04 (Cirrus)
"""

import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "zeus"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import datetime as dt
import xarray as xr
from scipy import stats
from scipy.ndimage import sobel

from stereo_winds.solver import (
    build_design_matrix,
    solve_stereo_winds,
    pixels_to_wind_ms,
    estimate_cross_sat_offset,
    correct_cross_sat_offset,
)
from stereo_winds.time_model import compute_scene_times
from stereo_winds.navigation import (
    geodetic_to_fixed_grid,
    scanning_angle_to_pixel,
)
from stereo_winds.config import GOES16_CONFIG, GOES18_CONFIG
from stereo_winds.validation.metrics import rmsvd, speed_bias, height_rmse, correlation
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =========================================================================
# Configuration
# =========================================================================
DATE = dt.date(2025, 1, 8)
DATE_STR = DATE.strftime("%Y-%m-%d")
DOY = DATE.strftime("%j")
T0 = dt.datetime(2025, 1, 8, 19, 0)
DT_MINUTES = 10.0
TIMES_HHMM = ["1850", "1900", "1910"]

SUPTITLE_DATE = "GOES-16/18, 2025-01-08 19:00 UTC"

cache_root = os.path.join(BASE, "cache")
cache_dir = os.path.join(cache_root, "20250108")
carr_dir = os.path.join(BASE, "carr_data")

MAP_EXTENT = [-155, -60, -70, 72]
PROJ = ccrs.PlateCarree()
DPI = 200

H_MIN, H_MAX = 0, 16000
CMAP_H = plt.cm.turbo
NORM_H = mcolors.Normalize(vmin=H_MIN, vmax=H_MAX)

# QA filter thresholds
QA_THRESHOLDS = dict(
    chi2_max=0.2,          # goodness of fit (3 DOF)
    sigma_h_max=5000.0,    # height uncertainty in meters
    h_grad_max=3000.0,     # height gradient in meters/pixel (cloud edge detector)
    wind_speed_max=100.0,  # sanity check on wind speed (m/s)
    w_mag_min=0.0003,      # min parallax sensitivity (px/m) — rejects weak-parallax pixels
)


# =========================================================================
# Data loading helpers
# =========================================================================
def load_carr_data(filepath):
    """Load a Carr et al. retrieval NetCDF and return a dict of numpy arrays."""
    ds = xr.open_dataset(filepath)
    lat = ds["lat"].values
    lon = ds["lon"].values
    u = ds["V_3D"].values[:, 0]
    v = ds["V_3D"].values[:, 1]
    h = ds["H_3D"].values
    sig_h = ds["sig_H_3D"].values
    dqf = ds["DQF_3D"].values
    ds.close()

    spd = np.sqrt(u**2 + v**2)
    good = (
        (dqf == 0)
        & np.isfinite(h) & np.isfinite(u) & np.isfinite(v)
        & (h >= 0) & (h <= 20000)
    )
    return dict(
        lat=lat, lon=lon, u=u, v=v, h=h, sig_h=sig_h, spd=spd,
        good=good, n_total=len(lat), n_good=int(np.sum(good)),
    )


def solve_band(band, band_label):
    """Re-run the solver for one band using cached RAFT flows. Return result dict.

    Uses standard GOES-16/18 configs and derives validity from the RAFT flows
    themselves, avoiding the need to reload raw satellite images.
    """
    print(f"\n{'='*60}")
    print(f"  BAND: {band_label}")
    print(f"{'='*60}")

    sat_g16 = GOES16_CONFIG
    sat_g18 = GOES18_CONFIG

    # Load cached RAFT flows — validity comes from flow finiteness
    print("  Loading cached RAFT flows...")
    disparities = {}
    for key in ["D1", "D2", "D3", "D4"]:
        cache_path = os.path.join(cache_dir, f"flow_g16g18_19z_{band}_{key}.npy")
        f = np.load(cache_path)
        if f.ndim == 4:
            f = f[0]
        disparities[key] = f
        print(f"    {key}: finite={np.isfinite(f[0]).sum():,}")

    # Solve
    print("  Solving...")
    parallax_cache = os.path.join(cache_root, "parallax_goes16_goes18.npz")
    data = np.load(parallax_cache)
    w_u, w_v = data["w_u"], data["w_v"]

    # Note: rebuild_final.py already applies a linear parallax correction
    # to the cached flows, so we do NOT apply a second correction here.

    scene_times = compute_scene_times(T0, DT_MINUTES, sat_g16, sat_g18)
    H_matrix = build_design_matrix(
        w_u, w_v,
        dt_a_minus=scene_times["A_minus"],
        dt_a_plus=scene_times["A_plus"],
        dt_b_minus=scene_times["B_minus"],
        dt_b_plus=scene_times["B_plus"],
    )

    t_solve = time.time()
    solution = solve_stereo_winds(
        disparities, H_matrix, sat_a=sat_g16, sat_b=sat_g18, n_iter=3,
    )
    elapsed = time.time() - t_solve
    print(f"    Solved in {elapsed:.1f}s (n_iter=3)")
    if solution["delta_h_history"]:
        for i, dh in enumerate(solution["delta_h_history"]):
            print(f"      Iter {i+1}: median |Δh| = {dh:.1f} m")

    u_ms, v_ms = pixels_to_wind_ms(solution["V_u"], solution["V_v"], sat_g16, dt_seconds=1.0)
    h = solution["h"]
    qf = solution["quality_flag"].copy()

    n_good = int((qf > 0).sum())
    print(f"    QC-passed: {n_good:,}")

    return dict(
        h=h, u=u_ms, v=v_ms, qf=qf, sat=sat_g16, n_good=n_good,
        chi2=solution["chi2"],
        sigma_h=solution["sigma_h"],
        sigma_u=solution["sigma_u"],
        sigma_v=solution["sigma_v"],
        w_u=w_u, w_v=w_v,
    )


# =========================================================================
# QA filters
# =========================================================================
def compute_height_gradient(h_field):
    """Compute per-pixel height gradient magnitude using Sobel filters.

    Returns gradient in meters per pixel (Sobel kernel normalized by 8).
    NaN pixels are filled with 0, so boundaries produce large gradients.
    """
    h_filled = np.where(np.isfinite(h_field), h_field, 0.0)
    grad_x = sobel(h_filled, axis=1)
    grad_y = sobel(h_filled, axis=0)
    return np.sqrt(grad_x**2 + grad_y**2) / 8.0


def apply_qa_filters(ai, thresholds):
    """Apply QA filters to the AI solution and return a per-pixel pass/fail mask.

    Returns (qa_mask, filter_stats) where qa_mask is a (H, W) bool array
    and filter_stats is a dict with per-filter rejection counts.
    """
    h = ai["h"]
    qf = ai["qf"]
    base_good = (qf > 0) & np.isfinite(h)
    n_base = int(base_good.sum())

    # Individual filter masks
    chi2_ok = np.isfinite(ai["chi2"]) & (ai["chi2"] <= thresholds["chi2_max"])
    sigma_h_ok = np.isfinite(ai["sigma_h"]) & (ai["sigma_h"] <= thresholds["sigma_h_max"])

    h_grad = compute_height_gradient(h)
    grad_ok = h_grad <= thresholds["h_grad_max"]
    ai["h_grad"] = h_grad  # stash for diagnostics

    speed = np.sqrt(ai["u"]**2 + ai["v"]**2)
    speed_ok = np.isfinite(speed) & (speed <= thresholds["wind_speed_max"])

    # Parallax sensitivity filter — reject weak-parallax pixels (high latitude)
    w_mag = np.sqrt(ai["w_u"]**2 + ai["w_v"]**2)
    parallax_ok = w_mag >= thresholds["w_mag_min"]

    # Combine
    qa_mask = base_good & chi2_ok & sigma_h_ok & grad_ok & speed_ok & parallax_ok

    # Per-filter rejection stats (how many base-good pixels each filter removes)
    filter_stats = {
        "n_base_good": n_base,
        "chi2": int((base_good & ~chi2_ok).sum()),
        "sigma_h": int((base_good & ~sigma_h_ok).sum()),
        "h_grad": int((base_good & ~grad_ok).sum()),
        "wind_speed": int((base_good & ~speed_ok).sum()),
        "w_mag": int((base_good & ~parallax_ok).sum()),
        "n_final": int(qa_mask.sum()),
    }

    return qa_mask, filter_stats


# =========================================================================
# Site matching: Carr lat/lon → nearest AI pixel
# =========================================================================
def match_sites(carr, ai, sat, qa_mask=None):
    """For each good Carr site, find nearest AI pixel and build paired arrays.

    Returns dict with paired arrays for both Carr and AI values, plus the
    lat/lon and height of each matched pair.
    """
    good_carr = carr["good"]
    lat_g = carr["lat"][good_carr]
    lon_g = carr["lon"][good_carr]
    h_carr = carr["h"][good_carr]
    u_carr = carr["u"][good_carr]
    v_carr = carr["v"][good_carr]

    # Carr lat/lon → GOES-16 fixed grid → pixel
    x_ang, y_ang = geodetic_to_fixed_grid(lat_g, lon_g, sat, h_m=0.0)
    col_f, row_f = scanning_angle_to_pixel(x_ang, y_ang, sat)

    col_i = np.round(col_f).astype(int)
    row_i = np.round(row_f).astype(int)

    # Clip to grid bounds
    in_bounds = (
        (col_i >= 0) & (col_i < sat.n_cols)
        & (row_i >= 0) & (row_i < sat.n_rows)
        & np.isfinite(col_f) & np.isfinite(row_f)
    )

    col_i = np.clip(col_i, 0, sat.n_cols - 1)
    row_i = np.clip(row_i, 0, sat.n_rows - 1)

    # Extract AI values at matched pixels
    h_ai = ai["h"][row_i, col_i]
    u_ai = ai["u"][row_i, col_i]
    v_ai = ai["v"][row_i, col_i]
    qf_ai = ai["qf"][row_i, col_i]

    # Keep only pairs where both pass QC
    matched = (
        in_bounds
        & (qf_ai > 0)
        & np.isfinite(h_ai) & np.isfinite(u_ai) & np.isfinite(v_ai)
    )
    if qa_mask is not None:
        matched = matched & qa_mask[row_i, col_i]

    n_matched = int(matched.sum())
    print(f"    Matched: {n_matched:,} / {int(good_carr.sum()):,} Carr good sites")

    return dict(
        lat=lat_g[matched], lon=lon_g[matched],
        carr_h=h_carr[matched], carr_u=u_carr[matched], carr_v=v_carr[matched],
        carr_spd=np.sqrt(u_carr[matched]**2 + v_carr[matched]**2),
        ai_h=h_ai[matched], ai_u=u_ai[matched], ai_v=v_ai[matched],
        ai_spd=np.sqrt(u_ai[matched]**2 + v_ai[matched]**2),
        n_matched=n_matched,
    )


def compute_all_metrics(m):
    """Compute comparison metrics from a matched dict."""
    h_rmse = height_rmse(m["ai_h"], m["carr_h"])
    h_bias = float(np.mean(m["ai_h"] - m["carr_h"]))
    h_corr = correlation(m["ai_h"], m["carr_h"])
    rv = rmsvd(m["ai_u"], m["ai_v"], m["carr_u"], m["carr_v"])
    sb = speed_bias(m["ai_u"], m["ai_v"], m["carr_u"], m["carr_v"])
    spd_corr = correlation(m["ai_spd"], m["carr_spd"])
    u_rmse = float(np.sqrt(np.mean((m["ai_u"] - m["carr_u"])**2)))
    v_rmse = float(np.sqrt(np.mean((m["ai_v"] - m["carr_v"])**2)))
    return dict(
        n_matched=m["n_matched"],
        height_rmse=h_rmse, height_bias=h_bias, height_corr=h_corr,
        rmsvd=rv, speed_bias=sb, speed_corr=spd_corr,
        u_rmse=u_rmse, v_rmse=v_rmse,
    )


# =========================================================================
# Main
# =========================================================================
print("=" * 60)
print(f"AI vs CARR COMPARISON — {DATE_STR} 19:00 UTC")
print(f"GOES-16/18, C14 + C04")
print("=" * 60)

# --- Step 1: Load Carr data ---
print("\nLoading Carr baseline data...")
carr_c14 = load_carr_data(os.path.join(carr_dir, "GOES_GOES_B14_20250081900208.nc"))
carr_c04 = load_carr_data(os.path.join(carr_dir, "GOES_GOES_B04_20250081900208.nc"))
print(f"  C14: {carr_c14['n_good']:,} / {carr_c14['n_total']:,} good sites")
print(f"  C04: {carr_c04['n_good']:,} / {carr_c04['n_total']:,} good sites")

# --- Step 2: Re-run AI solver ---
ai_c14 = solve_band("C14", "C14 (11.2 µm IR)")
ai_c04 = solve_band("C04", "C04 (1.37 µm Cirrus)")

# --- Step 3: Apply QA filters ---
print("\n" + "=" * 60)
print("QA FILTERING")
print("=" * 60)

qa_desc = (f"chi2<={QA_THRESHOLDS['chi2_max']}, "
           f"sigma_h<={QA_THRESHOLDS['sigma_h_max']:.0f}m, "
           f"grad<={QA_THRESHOLDS['h_grad_max']:.0f}m/px, "
           f"spd<={QA_THRESHOLDS['wind_speed_max']:.0f}m/s, "
           f"w_mag>={QA_THRESHOLDS['w_mag_min']}")

for band_label, ai in [("C14", ai_c14), ("C04", ai_c04)]:
    qa_mask, fstats = apply_qa_filters(ai, QA_THRESHOLDS)
    ai["qa_mask"] = qa_mask
    print(f"\n  {band_label} QA Filter Report:")
    print(f"    Base good (qf > 0):       {fstats['n_base_good']:>10,}")
    for name, thr_key, op in [("chi2", "chi2_max", ">"), ("sigma_h", "sigma_h_max", ">"),
                               ("h_grad", "h_grad_max", ">"), ("wind_speed", "wind_speed_max", ">"),
                               ("w_mag", "w_mag_min", "<")]:
        print(f"    Rejected by {name:<12s} ({op}{QA_THRESHOLDS[thr_key]:>8.4f}): {fstats[name]:>10,}")
    print(f"    Final good:               {fstats['n_final']:>10,}")

# --- Step 4: Match sites (before and after QA) ---
print("\n" + "=" * 60)
print("MATCHING SITES")
print("=" * 60)

print("\n  C14 (no QA filter):")
m14_nf = match_sites(carr_c14, ai_c14, ai_c14["sat"])
print("  C14 (with QA filter):")
m14 = match_sites(carr_c14, ai_c14, ai_c14["sat"], qa_mask=ai_c14["qa_mask"])
print("  C04 (no QA filter):")
m04_nf = match_sites(carr_c04, ai_c04, ai_c04["sat"])
print("  C04 (with QA filter):")
m04 = match_sites(carr_c04, ai_c04, ai_c04["sat"], qa_mask=ai_c04["qa_mask"])

# --- Step 5: Compute metrics ---
met14_nf = compute_all_metrics(m14_nf)
met14 = compute_all_metrics(m14)
met04_nf = compute_all_metrics(m04_nf)
met04 = compute_all_metrics(m04)


# =========================================================================
# Figure 1: Scatter Plots — Height and Wind (2x3)
# =========================================================================
print("\n" + "=" * 60)
print("GENERATING FIGURES")
print("=" * 60)

print("\nFigure 1: Scatter plots (height, u, v)...")
fig1, axes1 = plt.subplots(2, 3, figsize=(22, 14))

for row, (label, m, met) in enumerate([
    ("C14 (IR)", m14, met14),
    ("C04 (Cirrus)", m04, met04),
]):
    # Height scatter
    ax = axes1[row, 0]
    sc = ax.scatter(
        m["carr_h"] / 1000, m["ai_h"] / 1000,
        c=m["carr_h"] / 1000, cmap="turbo", vmin=0, vmax=16,
        s=0.5, alpha=0.3, rasterized=True,
    )
    lim = [0, 18]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.7)
    # Linear regression
    slope, intercept, r, _, _ = stats.linregress(m["carr_h"], m["ai_h"])
    x_fit = np.array(lim) * 1000
    ax.plot(x_fit / 1000, (slope * x_fit + intercept) / 1000, "r-", lw=1, alpha=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Carr Height (km)", fontsize=11)
    ax.set_ylabel("AI Height (km)", fontsize=11)
    ax.set_title(f"{label} — Height", fontsize=12, fontweight="bold")
    ax.text(
        0.05, 0.95,
        f"N = {met['n_matched']:,}\n"
        f"RMSE = {met['height_rmse']:.0f} m\n"
        f"Bias = {met['height_bias']:+.0f} m\n"
        f"r = {met['height_corr']:.3f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    # u-wind scatter
    ax = axes1[row, 1]
    ax.scatter(
        m["carr_u"], m["ai_u"],
        c=m["carr_h"] / 1000, cmap="turbo", vmin=0, vmax=16,
        s=0.5, alpha=0.3, rasterized=True,
    )
    lim_w = [-60, 60]
    ax.plot(lim_w, lim_w, "k--", lw=1, alpha=0.7)
    slope_u, int_u, _, _, _ = stats.linregress(m["carr_u"], m["ai_u"])
    x_fit_u = np.array(lim_w)
    ax.plot(x_fit_u, slope_u * x_fit_u + int_u, "r-", lw=1, alpha=0.8)
    ax.set_xlim(lim_w)
    ax.set_ylim(lim_w)
    ax.set_xlabel("Carr u-wind (m/s)", fontsize=11)
    ax.set_ylabel("AI u-wind (m/s)", fontsize=11)
    ax.set_title(f"{label} — u-wind", fontsize=12, fontweight="bold")
    ax.text(
        0.05, 0.95,
        f"RMSE = {met['u_rmse']:.2f} m/s\nr = {correlation(m['ai_u'], m['carr_u']):.3f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    # v-wind scatter
    ax = axes1[row, 2]
    ax.scatter(
        m["carr_v"], m["ai_v"],
        c=m["carr_h"] / 1000, cmap="turbo", vmin=0, vmax=16,
        s=0.5, alpha=0.3, rasterized=True,
    )
    ax.plot(lim_w, lim_w, "k--", lw=1, alpha=0.7)
    slope_v, int_v, _, _, _ = stats.linregress(m["carr_v"], m["ai_v"])
    ax.plot(x_fit_u, slope_v * x_fit_u + int_v, "r-", lw=1, alpha=0.8)
    ax.set_xlim(lim_w)
    ax.set_ylim(lim_w)
    ax.set_xlabel("Carr v-wind (m/s)", fontsize=11)
    ax.set_ylabel("AI v-wind (m/s)", fontsize=11)
    ax.set_title(f"{label} — v-wind", fontsize=12, fontweight="bold")
    ax.text(
        0.05, 0.95,
        f"RMSE = {met['v_rmse']:.2f} m/s\nr = {correlation(m['ai_v'], m['carr_v']):.3f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

fig1.suptitle(
    f"AI (RAFT) vs Carr — Scatter Comparison\n{SUPTITLE_DATE}\nQA: {qa_desc}",
    fontsize=15, fontweight="bold",
)
fig1.tight_layout()
out1 = os.path.join(BASE, "compare_ai_vs_carr_scatter.png")
fig1.savefig(out1, dpi=DPI, bbox_inches="tight")
print(f"  Saved: {out1}")
plt.close(fig1)


# =========================================================================
# Figure 2: Wind Vector Comparison — Speed/Direction (2x2)
# =========================================================================
print("\nFigure 2: Wind vector comparison...")
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))

# (a) Speed scatter C14
ax = axes2[0, 0]
sc = ax.scatter(
    m14["carr_spd"], m14["ai_spd"],
    c=m14["carr_h"] / 1000, cmap="turbo", vmin=0, vmax=16,
    s=0.5, alpha=0.3, rasterized=True,
)
ax.plot([0, 80], [0, 80], "k--", lw=1, alpha=0.7)
ax.set_xlim(0, 80)
ax.set_ylim(0, 80)
ax.set_xlabel("Carr Speed (m/s)", fontsize=11)
ax.set_ylabel("AI Speed (m/s)", fontsize=11)
ax.set_title("(a) Speed: C14 (IR)", fontsize=12, fontweight="bold")
ax.text(
    0.05, 0.95,
    f"N = {met14['n_matched']:,}\n"
    f"RMSVD = {met14['rmsvd']:.2f} m/s\n"
    f"Bias = {met14['speed_bias']:+.2f} m/s\n"
    f"r = {met14['speed_corr']:.3f}",
    transform=ax.transAxes, fontsize=9, va="top",
    bbox=dict(boxstyle="round", fc="white", alpha=0.8),
)
plt.colorbar(sc, ax=ax, shrink=0.8, label="Carr Height (km)")

# (b) Speed scatter C04
ax = axes2[0, 1]
sc = ax.scatter(
    m04["carr_spd"], m04["ai_spd"],
    c=m04["carr_h"] / 1000, cmap="turbo", vmin=0, vmax=16,
    s=0.5, alpha=0.3, rasterized=True,
)
ax.plot([0, 80], [0, 80], "k--", lw=1, alpha=0.7)
ax.set_xlim(0, 80)
ax.set_ylim(0, 80)
ax.set_xlabel("Carr Speed (m/s)", fontsize=11)
ax.set_ylabel("AI Speed (m/s)", fontsize=11)
ax.set_title("(b) Speed: C04 (Cirrus)", fontsize=12, fontweight="bold")
ax.text(
    0.05, 0.95,
    f"N = {met04['n_matched']:,}\n"
    f"RMSVD = {met04['rmsvd']:.2f} m/s\n"
    f"Bias = {met04['speed_bias']:+.2f} m/s\n"
    f"r = {met04['speed_corr']:.3f}",
    transform=ax.transAxes, fontsize=9, va="top",
    bbox=dict(boxstyle="round", fc="white", alpha=0.8),
)
plt.colorbar(sc, ax=ax, shrink=0.8, label="Carr Height (km)")

# (c) Speed bias vs height (binned)
ax = axes2[1, 0]
h_edges = np.arange(0, 17000, 1000)
h_centers = (h_edges[:-1] + h_edges[1:]) / 2

for label, m, color in [("C14", m14, "steelblue"), ("C04", m04, "firebrick")]:
    spd_diff = m["ai_spd"] - m["carr_spd"]
    bin_means = []
    bin_stds = []
    for i in range(len(h_edges) - 1):
        mask = (m["carr_h"] >= h_edges[i]) & (m["carr_h"] < h_edges[i + 1])
        if mask.sum() > 10:
            bin_means.append(np.mean(spd_diff[mask]))
            bin_stds.append(np.std(spd_diff[mask]))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    valid_bins = np.isfinite(bin_means)
    ax.plot(h_centers[valid_bins] / 1000, bin_means[valid_bins], "o-",
            color=color, label=label, markersize=4)
    ax.fill_between(
        h_centers[valid_bins] / 1000,
        (bin_means - bin_stds)[valid_bins],
        (bin_means + bin_stds)[valid_bins],
        alpha=0.2, color=color,
    )

ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("Carr Height (km)", fontsize=11)
ax.set_ylabel("Speed Bias (AI - Carr, m/s)", fontsize=11)
ax.set_title("(c) Speed Bias vs Height", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(0, 16)

# (d) RMSVD vs height (binned)
ax = axes2[1, 1]
for label, m, color in [("C14", m14, "steelblue"), ("C04", m04, "firebrick")]:
    bin_rmsvd = []
    for i in range(len(h_edges) - 1):
        mask = (m["carr_h"] >= h_edges[i]) & (m["carr_h"] < h_edges[i + 1])
        if mask.sum() > 10:
            du = m["ai_u"][mask] - m["carr_u"][mask]
            dv = m["ai_v"][mask] - m["carr_v"][mask]
            bin_rmsvd.append(np.sqrt(np.mean(du**2 + dv**2)))
        else:
            bin_rmsvd.append(np.nan)
    bin_rmsvd = np.array(bin_rmsvd)
    valid_bins = np.isfinite(bin_rmsvd)
    ax.plot(h_centers[valid_bins] / 1000, bin_rmsvd[valid_bins], "o-",
            color=color, label=label, markersize=4)

ax.set_xlabel("Carr Height (km)", fontsize=11)
ax.set_ylabel("RMSVD (m/s)", fontsize=11)
ax.set_title("(d) RMSVD vs Height", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(0, 16)

fig2.suptitle(
    f"AI (RAFT) vs Carr — Wind Vector Comparison\n{SUPTITLE_DATE}\nQA: {qa_desc}",
    fontsize=15, fontweight="bold",
)
fig2.tight_layout()
out2 = os.path.join(BASE, "compare_ai_vs_carr_wind.png")
fig2.savefig(out2, dpi=DPI, bbox_inches="tight")
print(f"  Saved: {out2}")
plt.close(fig2)


# =========================================================================
# Figure 3: Height Difference Maps (1x2)
# =========================================================================
print("\nFigure 3: Height difference maps...")
fig3, axes3 = plt.subplots(1, 2, figsize=(24, 12), subplot_kw={"projection": PROJ})

for panel, (label, m) in enumerate([("C14 (IR)", m14), ("C04 (Cirrus)", m04)]):
    ax = axes3[panel]
    ax.add_feature(cfeature.OCEAN, facecolor="#0a0a2e", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#1a1a1a", zorder=0)
    ax.coastlines(resolution="50m", color="white", linewidth=0.6)
    ax.set_extent(MAP_EXTENT, crs=PROJ)

    dh_km = (m["ai_h"] - m["carr_h"]) / 1000
    sc = ax.scatter(
        m["lon"], m["lat"], c=dh_km,
        cmap="RdBu_r", vmin=-5, vmax=5,
        s=1.5, alpha=0.7, transform=PROJ, rasterized=True,
    )
    ax.set_title(
        f"{label}\nHeight Difference (AI - Carr)\n"
        f"N = {m['n_matched']:,}, mean = {np.mean(dh_km):+.2f} km",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(sc, ax=ax, shrink=0.6, label="Height Difference (km)")

fig3.suptitle(
    f"AI (RAFT) vs Carr — Height Difference Map\n{SUPTITLE_DATE}\nQA: {qa_desc}",
    fontsize=15, fontweight="bold",
)
fig3.tight_layout()
out3 = os.path.join(BASE, "compare_ai_vs_carr_height_map.png")
fig3.savefig(out3, dpi=DPI, bbox_inches="tight")
print(f"  Saved: {out3}")
plt.close(fig3)


# =========================================================================
# Figure 4: Difference Histograms (1x3)
# =========================================================================
print("\nFigure 4: Difference histograms...")
fig4, axes4 = plt.subplots(1, 3, figsize=(22, 6))

# Height difference
ax = axes4[0]
dh14 = (m14["ai_h"] - m14["carr_h"]) / 1000
dh04 = (m04["ai_h"] - m04["carr_h"]) / 1000
bins_dh = np.linspace(-10, 10, 100)
ax.hist(dh14, bins=bins_dh, alpha=0.7, color="steelblue",
        label=f"C14 (n={m14['n_matched']:,})", density=True)
ax.hist(dh04, bins=bins_dh, alpha=0.7, color="firebrick",
        label=f"C04 (n={m04['n_matched']:,})", density=True)
ax.axvline(np.mean(dh14), color="steelblue", ls="--", lw=1.2)
ax.axvline(np.mean(dh04), color="firebrick", ls="--", lw=1.2)
ax.axvline(0, color="black", ls="-", lw=0.8, alpha=0.5)
ax.set_xlabel("Height Difference (AI - Carr, km)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Height Difference", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)

# Speed difference
ax = axes4[1]
dspd14 = m14["ai_spd"] - m14["carr_spd"]
dspd04 = m04["ai_spd"] - m04["carr_spd"]
bins_ds = np.linspace(-30, 30, 100)
ax.hist(dspd14, bins=bins_ds, alpha=0.7, color="steelblue",
        label=f"C14 (n={m14['n_matched']:,})", density=True)
ax.hist(dspd04, bins=bins_ds, alpha=0.7, color="firebrick",
        label=f"C04 (n={m04['n_matched']:,})", density=True)
ax.axvline(np.mean(dspd14), color="steelblue", ls="--", lw=1.2)
ax.axvline(np.mean(dspd04), color="firebrick", ls="--", lw=1.2)
ax.axvline(0, color="black", ls="-", lw=0.8, alpha=0.5)
ax.set_xlabel("Speed Difference (AI - Carr, m/s)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Speed Difference", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)

# Wind direction difference
ax = axes4[2]
dir_carr14 = np.degrees(np.arctan2(m14["carr_u"], m14["carr_v"])) % 360
dir_ai14 = np.degrees(np.arctan2(m14["ai_u"], m14["ai_v"])) % 360
ddir14 = dir_ai14 - dir_carr14
ddir14 = (ddir14 + 180) % 360 - 180  # wrap to [-180, 180]

dir_carr04 = np.degrees(np.arctan2(m04["carr_u"], m04["carr_v"])) % 360
dir_ai04 = np.degrees(np.arctan2(m04["ai_u"], m04["ai_v"])) % 360
ddir04 = dir_ai04 - dir_carr04
ddir04 = (ddir04 + 180) % 360 - 180

bins_dd = np.linspace(-180, 180, 90)
# Filter out near-calm winds where direction is meaningless
spd_thresh = 2.0
mask14 = (m14["carr_spd"] > spd_thresh) & (m14["ai_spd"] > spd_thresh)
mask04 = (m04["carr_spd"] > spd_thresh) & (m04["ai_spd"] > spd_thresh)
ax.hist(ddir14[mask14], bins=bins_dd, alpha=0.7, color="steelblue",
        label=f"C14 (n={mask14.sum():,})", density=True)
ax.hist(ddir04[mask04], bins=bins_dd, alpha=0.7, color="firebrick",
        label=f"C04 (n={mask04.sum():,})", density=True)
ax.axvline(0, color="black", ls="-", lw=0.8, alpha=0.5)
ax.set_xlabel("Direction Difference (AI - Carr, deg)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Wind Direction Difference (spd > 2 m/s)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)

fig4.suptitle(
    f"AI (RAFT) vs Carr — Difference Histograms\n{SUPTITLE_DATE}\nQA: {qa_desc}",
    fontsize=14, fontweight="bold",
)
fig4.tight_layout()
out4 = os.path.join(BASE, "compare_ai_vs_carr_histograms.png")
fig4.savefig(out4, dpi=DPI, bbox_inches="tight")
print(f"  Saved: {out4}")
plt.close(fig4)


# =========================================================================
# Figure 5: Summary Statistics Table
# =========================================================================
print("\nFigure 5: Summary table (before/after QA)...")

def _fmt_met(met):
    """Format a metrics dict into a column of strings."""
    return [
        f"{met['n_matched']:,}",
        f"{met['height_rmse']:.0f}",
        f"{met['height_bias']:+.0f}",
        f"{met['height_corr']:.4f}",
        f"{met['rmsvd']:.2f}",
        f"{met['speed_bias']:+.2f}",
        f"{met['speed_corr']:.4f}",
        f"{met['u_rmse']:.2f}",
        f"{met['v_rmse']:.2f}",
    ]

row_labels = [
    "N matched", "Height RMSE (m)", "Height bias (m)", "Height correlation",
    "RMSVD (m/s)", "Speed bias (m/s)", "Speed correlation",
    "u-wind RMSE (m/s)", "v-wind RMSE (m/s)",
]
cols_nf14 = _fmt_met(met14_nf)
cols_qa14 = _fmt_met(met14)
cols_nf04 = _fmt_met(met04_nf)
cols_qa04 = _fmt_met(met04)

table_data = [[lab, a, b, c, d] for lab, a, b, c, d
              in zip(row_labels, cols_nf14, cols_qa14, cols_nf04, cols_qa04)]

fig5, ax5 = plt.subplots(figsize=(16, 5))
ax5.axis("off")

col_labels = ["Metric", "C14 Raw", "C14 QA", "C04 Raw", "C04 QA"]
table = ax5.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Style header row
for j in range(5):
    cell = table[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

# Alternate row shading
for i in range(len(table_data)):
    for j in range(5):
        cell = table[i + 1, j]
        if i % 2 == 0:
            cell.set_facecolor("#ecf0f1")
        else:
            cell.set_facecolor("white")

fig5.suptitle(
    f"AI (RAFT) vs Carr — Summary Statistics\n{SUPTITLE_DATE}\nQA: {qa_desc}",
    fontsize=14, fontweight="bold", y=0.98,
)
out5 = os.path.join(BASE, "compare_ai_vs_carr_summary.png")
fig5.savefig(out5, dpi=DPI, bbox_inches="tight")
print(f"  Saved: {out5}")
plt.close(fig5)


# =========================================================================
# Console summary
# =========================================================================
print("\n" + "=" * 60)
print("SUMMARY STATISTICS (Raw → QA-filtered)")
print("=" * 60)
print(f"\n{'Metric':<25s}  {'C14 Raw':>10s}  {'C14 QA':>10s}  {'C04 Raw':>10s}  {'C04 QA':>10s}")
print("-" * 75)
for row in table_data:
    print(f"{row[0]:<25s}  {row[1]:>10s}  {row[2]:>10s}  {row[3]:>10s}  {row[4]:>10s}")
print("-" * 75)

print(f"\nPlots saved:")
for p in [out1, out2, out3, out4, out5]:
    print(f"  {p}")
print("\nDone!")
