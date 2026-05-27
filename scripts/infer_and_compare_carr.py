#!/usr/bin/env python
"""Run RAFT + WLS solver on a cached stereo scene and compare to Carr et al.

Reads the 5 .npy scenes produced by ``cache_scene_from_s3.py`` (or the
2026-03-13 sample tiles already on adapt), runs the pretrained RAFT
checkpoint, solves the stereo wind problem, and writes:

- ``stereo_retrieval.nc``: full-disk u/v/h/uncertainties/QF
- 4 stage diagnostic PNGs (inputs, RAFT flows, solver outputs, QA)
- 5 Carr comparison PNGs (scatter, wind, height map, histograms, summary)

Example
-------
    python scripts/infer_and_compare_carr.py \\
        --ckpt $DATA/weights/windflow.raft.202508.epoch254.ckpt \\
        --carr-nc $DATA/carr_data/GOES_GOES_B14_20250081900208.nc \\
        --scene-dir $DATA/zarrs/C14/20250108_1900 \\
        --time 2025-01-08T19:00 \\
        --sat-a goes16 --sat-b goes18 --band C14 \\
        --out-dir $NOBACKUP/stereo-winds/runs/2025-01-08T1900_C14
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats
from scipy.ndimage import sobel

from stereo_winds.config import SATELLITE_CONFIGS
from stereo_winds.disparity import StereoDisparity
from stereo_winds.navigation import geodetic_to_fixed_grid, scanning_angle_to_pixel
from stereo_winds.output import create_output_dataset, write_netcdf
from stereo_winds.solver import (
    build_design_matrix,
    compute_parallax_vectors,
    pixels_to_wind_ms,
    solve_stereo_winds,
)
from stereo_winds.time_model import compute_scene_times
from stereo_winds.validation.metrics import (
    correlation,
    height_rmse,
    rmsvd,
    speed_bias,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def histogram_equalize(image, n_bins: int = 500):
    """CDF-based histogram equalization to [0, 1]."""
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)
    vals = image[finite]
    hist, bins = np.histogram(vals, n_bins, density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    out = np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape)
    out[~finite] = 0.0
    return out.astype(np.float32)


def colormap_array(arr, cmap, vmin, vmax):
    """Apply a colormap to a numeric array, returning RGB uint8."""
    cm = plt.colormaps[cmap]
    normed = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    normed = np.nan_to_num(normed, nan=0.0)
    return (cm(normed)[..., :3] * 255).astype(np.uint8)


def compute_height_gradient(h_field):
    """Sobel gradient magnitude (m per pixel)."""
    h_filled = np.where(np.isfinite(h_field), h_field, 0.0)
    gx = sobel(h_filled, axis=1)
    gy = sobel(h_filled, axis=0)
    return np.sqrt(gx**2 + gy**2) / 8.0


# -----------------------------------------------------------------------------
# Carr loading / matching
# -----------------------------------------------------------------------------
def load_carr_data(filepath):
    """Load a Carr et al. retrieval NetCDF into numpy arrays."""
    try:
        ds = xr.open_dataset(filepath, engine="h5netcdf")
    except Exception:
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
        good=good, n_total=len(lat), n_good=int(good.sum()),
    )


QA_THRESHOLDS = dict(
    chi2_max=0.2,
    sigma_h_max=5000.0,
    h_grad_max=3000.0,
    wind_speed_max=100.0,
    w_mag_min=0.0003,
)


def apply_qa_filters(ai, thresholds=QA_THRESHOLDS):
    """Apply the standard 5-filter QA mask. Returns (mask, per-filter stats)."""
    h = ai["h"]
    qf = ai["qf"]
    base = (qf > 0) & np.isfinite(h)
    n_base = int(base.sum())

    chi2_ok = np.isfinite(ai["chi2"]) & (ai["chi2"] <= thresholds["chi2_max"])
    sigma_h_ok = np.isfinite(ai["sigma_h"]) & (ai["sigma_h"] <= thresholds["sigma_h_max"])

    h_grad = compute_height_gradient(h)
    grad_ok = h_grad <= thresholds["h_grad_max"]
    ai["h_grad"] = h_grad

    speed = np.sqrt(ai["u"]**2 + ai["v"]**2)
    speed_ok = np.isfinite(speed) & (speed <= thresholds["wind_speed_max"])

    w_mag = np.sqrt(ai["w_u"]**2 + ai["w_v"]**2)
    par_ok = w_mag >= thresholds["w_mag_min"]

    qa = base & chi2_ok & sigma_h_ok & grad_ok & speed_ok & par_ok

    stats_ = {
        "n_base_good": n_base,
        "chi2": int((base & ~chi2_ok).sum()),
        "sigma_h": int((base & ~sigma_h_ok).sum()),
        "h_grad": int((base & ~grad_ok).sum()),
        "wind_speed": int((base & ~speed_ok).sum()),
        "w_mag": int((base & ~par_ok).sum()),
        "n_final": int(qa.sum()),
    }
    return qa, stats_


def match_sites(carr, ai, sat, qa_mask=None):
    """Match each Carr site to the nearest AI pixel; keep pairs where both pass QC."""
    g = carr["good"]
    lat_g = carr["lat"][g]
    lon_g = carr["lon"][g]
    h_c = carr["h"][g]
    u_c = carr["u"][g]
    v_c = carr["v"][g]

    x_ang, y_ang = geodetic_to_fixed_grid(lat_g, lon_g, sat, h_m=0.0)
    col_f, row_f = scanning_angle_to_pixel(x_ang, y_ang, sat)

    col_i = np.round(col_f).astype(int)
    row_i = np.round(row_f).astype(int)
    in_bounds = (
        (col_i >= 0) & (col_i < sat.n_cols)
        & (row_i >= 0) & (row_i < sat.n_rows)
        & np.isfinite(col_f) & np.isfinite(row_f)
    )
    col_i = np.clip(col_i, 0, sat.n_cols - 1)
    row_i = np.clip(row_i, 0, sat.n_rows - 1)

    h_ai = ai["h"][row_i, col_i]
    u_ai = ai["u"][row_i, col_i]
    v_ai = ai["v"][row_i, col_i]
    qf_ai = ai["qf"][row_i, col_i]

    keep = (
        in_bounds & (qf_ai > 0)
        & np.isfinite(h_ai) & np.isfinite(u_ai) & np.isfinite(v_ai)
    )
    if qa_mask is not None:
        keep = keep & qa_mask[row_i, col_i]

    n = int(keep.sum())
    return dict(
        lat=lat_g[keep], lon=lon_g[keep],
        carr_h=h_c[keep], carr_u=u_c[keep], carr_v=v_c[keep],
        carr_spd=np.sqrt(u_c[keep]**2 + v_c[keep]**2),
        ai_h=h_ai[keep], ai_u=u_ai[keep], ai_v=v_ai[keep],
        ai_spd=np.sqrt(u_ai[keep]**2 + v_ai[keep]**2),
        n=n,
    )


def compute_metrics(m):
    if m["n"] == 0:
        return dict(n=0, h_rmse=np.nan, h_bias=np.nan, h_corr=np.nan,
                    rv=np.nan, sb=np.nan, s_corr=np.nan,
                    u_rmse=np.nan, v_rmse=np.nan)
    return dict(
        n=m["n"],
        h_rmse=height_rmse(m["ai_h"], m["carr_h"]),
        h_bias=float(np.mean(m["ai_h"] - m["carr_h"])),
        h_corr=correlation(m["ai_h"], m["carr_h"]),
        rv=rmsvd(m["ai_u"], m["ai_v"], m["carr_u"], m["carr_v"]),
        sb=speed_bias(m["ai_u"], m["ai_v"], m["carr_u"], m["carr_v"]),
        s_corr=correlation(m["ai_spd"], m["carr_spd"]),
        u_rmse=float(np.sqrt(np.mean((m["ai_u"] - m["carr_u"])**2))),
        v_rmse=float(np.sqrt(np.mean((m["ai_v"] - m["carr_v"])**2))),
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
DPI = 150


def downsample(arr, stride=4):
    """Subsample a 2D array for fast plotting on the 5424^2 grid."""
    return arr[::stride, ::stride]


def plot_stage1_inputs(scenes, out_path, title):
    """1x5 row of the 5 input scenes (histogram-equalized)."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax, name in zip(axes, ["A_minus", "A0", "A_plus", "B_minus", "B_plus"]):
        eq = histogram_equalize(scenes[name])
        ax.imshow(downsample(eq), cmap="gray", vmin=0, vmax=1)
        finite_pct = 100 * np.isfinite(scenes[name]).mean()
        ax.set_title(f"{name}\nfinite={finite_pct:.1f}%", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Stage 1 — Input Scenes\n{title}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_path)


def plot_stage2_raft_flows(disparities, out_path, title):
    """2x4 grid: u and v components of D1-D4."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    keys = ["D1", "D2", "D3", "D4"]
    labels = ["D1: A0->A_minus (temporal)", "D2: A0->A_plus (temporal)",
              "D3: A0->B_minus (cross-sat)", "D4: A0->B_plus (cross-sat)"]
    FLOW_LIM = 10
    for j, (k, lab) in enumerate(zip(keys, labels)):
        d = disparities[k]
        for i, comp in enumerate(["u", "v"]):
            ax = axes[i, j]
            im = ax.imshow(
                downsample(d[i]), cmap="RdBu_r", vmin=-FLOW_LIM, vmax=FLOW_LIM,
            )
            ax.set_title(f"{lab} ({comp})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Stage 2 — RAFT Flow Fields (px)\n{title}",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(right=0.93)
    cax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cax, label="Displacement (px)")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_path)


def plot_stage3_solver_outputs(solution, u_ms, v_ms, out_path, title):
    """1x4 row: height (km), V_u px/s, V_v px/s, chi2."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    h_km = solution["h"] / 1000
    panels = [
        (h_km, "Cloud-Top Height (km)", "turbo", 0, 18),
        (solution["V_u"], "V_u (px/s)", "RdBu_r", -10, 10),
        (solution["V_v"], "V_v (px/s)", "RdBu_r", -10, 10),
        (solution["chi2"], "chi^2", "hot", 0, 5),
    ]
    for ax, (arr, lab, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(
            downsample(arr), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        ax.set_title(lab, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"Stage 3 — Solver Outputs\n{title}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_path)


def plot_stage4_qa(h, qa_mask, qa_stats, out_path, title):
    """1x2: solved height before vs after QA, with rejection counts overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    h_km = h / 1000
    h_pre = np.where(np.isfinite(h_km), h_km, np.nan)
    h_post = np.where(qa_mask, h_km, np.nan)

    im0 = axes[0].imshow(downsample(h_pre), cmap="turbo", vmin=0, vmax=18)
    axes[0].set_title(f"Pre-QA Height (km)\nn_finite={int(np.isfinite(h_pre).sum()):,}",
                      fontsize=11, fontweight="bold")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im0, ax=axes[0], shrink=0.7)

    im1 = axes[1].imshow(downsample(h_post), cmap="turbo", vmin=0, vmax=18)
    axes[1].set_title(f"Post-QA Height (km)\nn_passed={qa_stats['n_final']:,}",
                      fontsize=11, fontweight="bold")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], shrink=0.7)

    rejections = (
        f"QA rejections (of n_base={qa_stats['n_base_good']:,}):\n"
        f"  chi2 > {QA_THRESHOLDS['chi2_max']}: {qa_stats['chi2']:,}\n"
        f"  sigma_h > {QA_THRESHOLDS['sigma_h_max']:.0f} m: {qa_stats['sigma_h']:,}\n"
        f"  h_grad > {QA_THRESHOLDS['h_grad_max']:.0f} m/px: {qa_stats['h_grad']:,}\n"
        f"  speed > {QA_THRESHOLDS['wind_speed_max']:.0f} m/s: {qa_stats['wind_speed']:,}\n"
        f"  w_mag < {QA_THRESHOLDS['w_mag_min']}: {qa_stats['w_mag']:,}"
    )
    fig.text(0.02, 0.02, rejections, fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    fig.suptitle(f"Stage 4 — QA Filtering\n{title}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_path)


def plot_compare_scatter(m, met, out_path, title):
    """1x3 scatter: height, u, v vs Carr."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Height
    ax = axes[0]
    ax.scatter(m["carr_h"]/1000, m["ai_h"]/1000, c=m["carr_h"]/1000,
               cmap="turbo", vmin=0, vmax=16, s=0.5, alpha=0.3, rasterized=True)
    ax.plot([0, 18], [0, 18], "k--", lw=1)
    if m["n"] > 1:
        slope, intercept, _, _, _ = stats.linregress(m["carr_h"], m["ai_h"])
        ax.plot([0, 18], [intercept/1000, (slope*18000+intercept)/1000],
                "r-", lw=1, alpha=0.8)
    ax.set_xlim(0, 18); ax.set_ylim(0, 18)
    ax.set_xlabel("Carr Height (km)"); ax.set_ylabel("AI Height (km)")
    ax.set_title("Height", fontweight="bold")
    ax.text(0.05, 0.95,
            f"N = {met['n']:,}\nRMSE = {met['h_rmse']:.0f} m\n"
            f"Bias = {met['h_bias']:+.0f} m\nr = {met['h_corr']:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # u-wind
    ax = axes[1]
    ax.scatter(m["carr_u"], m["ai_u"], c=m["carr_h"]/1000,
               cmap="turbo", vmin=0, vmax=16, s=0.5, alpha=0.3, rasterized=True)
    ax.plot([-60, 60], [-60, 60], "k--", lw=1)
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
    ax.set_xlabel("Carr u-wind (m/s)"); ax.set_ylabel("AI u-wind (m/s)")
    ax.set_title("u-wind", fontweight="bold")
    ax.text(0.05, 0.95,
            f"RMSE = {met['u_rmse']:.2f} m/s\n"
            f"r = {correlation(m['ai_u'], m['carr_u']):.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # v-wind
    ax = axes[2]
    ax.scatter(m["carr_v"], m["ai_v"], c=m["carr_h"]/1000,
               cmap="turbo", vmin=0, vmax=16, s=0.5, alpha=0.3, rasterized=True)
    ax.plot([-60, 60], [-60, 60], "k--", lw=1)
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
    ax.set_xlabel("Carr v-wind (m/s)"); ax.set_ylabel("AI v-wind (m/s)")
    ax.set_title("v-wind", fontweight="bold")
    ax.text(0.05, 0.95,
            f"RMSE = {met['v_rmse']:.2f} m/s\n"
            f"r = {correlation(m['ai_v'], m['carr_v']):.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.suptitle(f"RAFT Stereo Winds vs Carr — Scatter\n{title}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_path)


def plot_compare_wind(m, met, out_path, title):
    """2x2: speed scatter, RMSVD vs height, speed bias vs height, height bias vs height."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Speed scatter
    ax = axes[0, 0]
    sc = ax.scatter(m["carr_spd"], m["ai_spd"], c=m["carr_h"]/1000,
                    cmap="turbo", vmin=0, vmax=16, s=0.5, alpha=0.3, rasterized=True)
    ax.plot([0, 80], [0, 80], "k--", lw=1)
    ax.set_xlim(0, 80); ax.set_ylim(0, 80)
    ax.set_xlabel("Carr Speed (m/s)"); ax.set_ylabel("AI Speed (m/s)")
    ax.set_title("(a) Speed", fontweight="bold")
    ax.text(0.05, 0.95,
            f"N = {met['n']:,}\nRMSVD = {met['rv']:.2f} m/s\n"
            f"Bias = {met['sb']:+.2f} m/s\nr = {met['s_corr']:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Carr Height (km)")

    # (b) RMSVD vs height (binned)
    ax = axes[0, 1]
    h_edges = np.arange(0, 17000, 1000)
    h_centers = (h_edges[:-1] + h_edges[1:]) / 2
    bin_rmsvd, bin_n = [], []
    for i in range(len(h_edges) - 1):
        mask = (m["carr_h"] >= h_edges[i]) & (m["carr_h"] < h_edges[i+1])
        n = int(mask.sum())
        bin_n.append(n)
        if n > 10:
            du = m["ai_u"][mask] - m["carr_u"][mask]
            dv = m["ai_v"][mask] - m["carr_v"][mask]
            bin_rmsvd.append(np.sqrt(np.mean(du**2 + dv**2)))
        else:
            bin_rmsvd.append(np.nan)
    valid = np.isfinite(bin_rmsvd)
    ax.plot(h_centers[valid]/1000, np.array(bin_rmsvd)[valid], "o-", color="steelblue", markersize=5)
    ax.set_xlabel("Carr Height (km)"); ax.set_ylabel("RMSVD (m/s)")
    ax.set_title("(b) RMSVD vs Height", fontweight="bold")
    ax.set_xlim(0, 16); ax.grid(alpha=0.3)

    # (c) Speed bias vs height (binned)
    ax = axes[1, 0]
    spd_diff = m["ai_spd"] - m["carr_spd"]
    bin_means, bin_stds = [], []
    for i in range(len(h_edges) - 1):
        mask = (m["carr_h"] >= h_edges[i]) & (m["carr_h"] < h_edges[i+1])
        if mask.sum() > 10:
            bin_means.append(np.mean(spd_diff[mask]))
            bin_stds.append(np.std(spd_diff[mask]))
        else:
            bin_means.append(np.nan); bin_stds.append(np.nan)
    bin_means = np.array(bin_means); bin_stds = np.array(bin_stds)
    valid = np.isfinite(bin_means)
    ax.plot(h_centers[valid]/1000, bin_means[valid], "o-", color="firebrick", markersize=5)
    ax.fill_between(h_centers[valid]/1000,
                    (bin_means - bin_stds)[valid], (bin_means + bin_stds)[valid],
                    alpha=0.2, color="firebrick")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Carr Height (km)"); ax.set_ylabel("AI - Carr speed (m/s)")
    ax.set_title("(c) Speed Bias vs Height", fontweight="bold")
    ax.set_xlim(0, 16); ax.grid(alpha=0.3)

    # (d) Height bias vs height (binned)
    ax = axes[1, 1]
    h_diff = (m["ai_h"] - m["carr_h"]) / 1000
    bin_means, bin_stds = [], []
    for i in range(len(h_edges) - 1):
        mask = (m["carr_h"] >= h_edges[i]) & (m["carr_h"] < h_edges[i+1])
        if mask.sum() > 10:
            bin_means.append(np.mean(h_diff[mask]))
            bin_stds.append(np.std(h_diff[mask]))
        else:
            bin_means.append(np.nan); bin_stds.append(np.nan)
    bin_means = np.array(bin_means); bin_stds = np.array(bin_stds)
    valid = np.isfinite(bin_means)
    ax.plot(h_centers[valid]/1000, bin_means[valid], "o-", color="darkgreen", markersize=5)
    ax.fill_between(h_centers[valid]/1000,
                    (bin_means - bin_stds)[valid], (bin_means + bin_stds)[valid],
                    alpha=0.2, color="darkgreen")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Carr Height (km)"); ax.set_ylabel("AI - Carr height (km)")
    ax.set_title("(d) Height Bias vs Height", fontweight="bold")
    ax.set_xlim(0, 16); ax.grid(alpha=0.3)

    fig.suptitle(f"RAFT Stereo Winds vs Carr — Wind & Height Analysis\n{title}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_path)


def plot_compare_height_map(m, out_path, title):
    """Geographic map of Carr-AI height difference."""
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.OCEAN, facecolor="#0a0a2e", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#1a1a1a", zorder=0)
    ax.coastlines(resolution="50m", color="white", linewidth=0.6)
    ax.set_extent([-155, -60, -70, 72], crs=ccrs.PlateCarree())

    dh_km = (m["ai_h"] - m["carr_h"]) / 1000
    sc = ax.scatter(m["lon"], m["lat"], c=dh_km,
                    cmap="RdBu_r", vmin=-5, vmax=5,
                    s=2, alpha=0.7, transform=ccrs.PlateCarree(), rasterized=True)
    ax.set_title(f"AI - Carr Height (km)\nN = {m['n']:,}, mean = {np.mean(dh_km):+.2f} km",
                 fontsize=12, fontweight="bold")
    plt.colorbar(sc, ax=ax, shrink=0.6, label="AI - Carr height (km)")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_path)


def plot_compare_histograms(m, out_path, title):
    """1x3: dh, dspeed, ddirection."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    dh = (m["ai_h"] - m["carr_h"]) / 1000
    axes[0].hist(dh, bins=np.linspace(-10, 10, 100), color="steelblue", alpha=0.7, density=True)
    axes[0].axvline(np.mean(dh), color="black", ls="--", lw=1.2)
    axes[0].axvline(0, color="gray", lw=0.8)
    axes[0].set_xlabel("AI - Carr height (km)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Height diff (mean = {np.mean(dh):+.2f} km)", fontweight="bold")

    dspd = m["ai_spd"] - m["carr_spd"]
    axes[1].hist(dspd, bins=np.linspace(-30, 30, 100), color="firebrick", alpha=0.7, density=True)
    axes[1].axvline(np.mean(dspd), color="black", ls="--", lw=1.2)
    axes[1].axvline(0, color="gray", lw=0.8)
    axes[1].set_xlabel("AI - Carr speed (m/s)")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Speed diff (mean = {np.mean(dspd):+.2f} m/s)", fontweight="bold")

    dir_carr = np.degrees(np.arctan2(m["carr_u"], m["carr_v"])) % 360
    dir_ai = np.degrees(np.arctan2(m["ai_u"], m["ai_v"])) % 360
    ddir = ((dir_ai - dir_carr) + 180) % 360 - 180
    spd_thresh = 2.0
    msk = (m["carr_spd"] > spd_thresh) & (m["ai_spd"] > spd_thresh)
    axes[2].hist(ddir[msk], bins=np.linspace(-180, 180, 90), color="darkgreen", alpha=0.7, density=True)
    axes[2].axvline(0, color="gray", lw=0.8)
    axes[2].set_xlabel("AI - Carr direction (deg)")
    axes[2].set_ylabel("Density")
    axes[2].set_title(f"Direction diff (spd > {spd_thresh} m/s, n = {int(msk.sum()):,})", fontweight="bold")

    fig.suptitle(f"RAFT vs Carr — Difference Histograms\n{title}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_path)


def plot_compare_summary(met_raw, met_qa, qa_stats, out_path, title):
    """Summary statistics table image."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    rows = [
        ("N matched", f"{met_raw['n']:,}", f"{met_qa['n']:,}"),
        ("Height RMSE (m)", f"{met_raw['h_rmse']:.0f}", f"{met_qa['h_rmse']:.0f}"),
        ("Height bias (m)", f"{met_raw['h_bias']:+.0f}", f"{met_qa['h_bias']:+.0f}"),
        ("Height correlation", f"{met_raw['h_corr']:.4f}", f"{met_qa['h_corr']:.4f}"),
        ("RMSVD (m/s)", f"{met_raw['rv']:.2f}", f"{met_qa['rv']:.2f}"),
        ("Speed bias (m/s)", f"{met_raw['sb']:+.2f}", f"{met_qa['sb']:+.2f}"),
        ("Speed correlation", f"{met_raw['s_corr']:.4f}", f"{met_qa['s_corr']:.4f}"),
        ("u-wind RMSE (m/s)", f"{met_raw['u_rmse']:.2f}", f"{met_qa['u_rmse']:.2f}"),
        ("v-wind RMSE (m/s)", f"{met_raw['v_rmse']:.2f}", f"{met_qa['v_rmse']:.2f}"),
    ]
    table = ax.table(
        cellText=[list(r) for r in rows],
        colLabels=["Metric", "Raw (no QA)", "After QA"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(len(rows)):
        for j in range(3):
            cell = table[i + 1, j]
            cell.set_facecolor("#ecf0f1" if i % 2 == 0 else "white")

    fig.suptitle(f"Summary Statistics\n{title}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="RAFT checkpoint path")
    parser.add_argument("--carr-nc", required=True, help="Carr baseline NetCDF")
    parser.add_argument("--scene-dir", required=True,
                        help="Directory with the 5 cached .npy scenes")
    parser.add_argument("--time", required=True,
                        help="Center time (ISO format, e.g. 2025-01-08T19:00)")
    parser.add_argument("--sat-a", default="goes16", choices=list(SATELLITE_CONFIGS.keys()))
    parser.add_argument("--sat-b", default="goes18", choices=list(SATELLITE_CONFIGS.keys()))
    parser.add_argument("--band", default="C14")
    parser.add_argument("--dt-minutes", type=float, default=10.0)
    parser.add_argument("--n-iter", type=int, default=3,
                        help="Solver outer iterations (default: 3)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lowmem", action="store_true",
                        help="Use serial RAFT for low GPU memory")
    parser.add_argument("--parallax-cache", default=None,
                        help="Optional path to cache the parallax LUT npz")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = Path(args.scene_dir)
    t0 = dt.datetime.fromisoformat(args.time)

    sat_a = SATELLITE_CONFIGS[args.sat_a]
    sat_b = SATELLITE_CONFIGS[args.sat_b]
    title = (f"{args.sat_a.upper()}/{args.sat_b.upper()} "
             f"{t0:%Y-%m-%d %H:%M} UTC, {args.band}")

    # --- Load 5 cached scenes ---
    logger.info("Loading cached scenes from %s", scene_dir)
    scenes = {}
    for name in ["A_minus", "A0", "A_plus", "B_minus", "B_plus"]:
        scenes[name] = np.load(scene_dir / f"{name}.npy").astype(np.float32)
        logger.info("  %s: shape=%s, finite=%.1f%%",
                    name, scenes[name].shape,
                    100 * np.isfinite(scenes[name]).mean())

    valid = np.all(
        [np.isfinite(scenes[k]) for k in scenes],
        axis=0,
    )
    logger.info("Joint-valid pixels: %d (%.1f%%)", int(valid.sum()),
                100 * valid.mean())

    plot_stage1_inputs(scenes, out_dir / "stage1_inputs.png", title)

    # --- Parallax / design matrix ---
    if args.parallax_cache and os.path.exists(args.parallax_cache):
        logger.info("Loading parallax cache %s", args.parallax_cache)
        data = np.load(args.parallax_cache)
        w_u, w_v = data["w_u"], data["w_v"]
    else:
        logger.info("Computing parallax vectors %s -> %s...", args.sat_a, args.sat_b)
        w_u, w_v = compute_parallax_vectors(sat_a, sat_b)
        if args.parallax_cache:
            np.savez_compressed(args.parallax_cache, w_u=w_u, w_v=w_v)
            logger.info("Cached parallax to %s", args.parallax_cache)

    scene_times = compute_scene_times(t0, args.dt_minutes, sat_a, sat_b)
    H_matrix = build_design_matrix(
        w_u, w_v,
        dt_a_minus=scene_times["A_minus"], dt_a_plus=scene_times["A_plus"],
        dt_b_minus=scene_times["B_minus"], dt_b_plus=scene_times["B_plus"],
    )

    # --- RAFT inference ---
    logger.info("Running RAFT (ckpt=%s)...", Path(args.ckpt).name)
    t_raft = time.time()
    disp = StereoDisparity(
        model_ckpt_path=args.ckpt,
        tile_size=args.tile_size, overlap=args.overlap,
        batch_size=args.batch_size, device=args.device,
        lowmem=args.lowmem,
    )
    # RAFT needs the scenes histogram-equalized to [0, 1]
    images_eq = {k: histogram_equalize(scenes[k]) for k in scenes}
    flows = disp.compute_all(images_eq)
    for k in flows:
        flows[k][:, ~valid] = np.nan
    logger.info("  RAFT: %.1f s", time.time() - t_raft)

    plot_stage2_raft_flows(flows, out_dir / "stage2_raft_flows.png", title)

    # --- Solve ---
    logger.info("Solving stereo system (n_iter=%d)...", args.n_iter)
    t_solve = time.time()
    solution = solve_stereo_winds(
        flows, H_matrix, sat_a=sat_a, sat_b=sat_b, n_iter=args.n_iter,
    )
    logger.info("  Solver: %.1f s", time.time() - t_solve)
    if solution.get("delta_h_history"):
        for i, dh in enumerate(solution["delta_h_history"]):
            logger.info("    iter %d median |dh|: %.1f m", i + 1, dh)

    u_ms, v_ms = pixels_to_wind_ms(
        solution["V_u"], solution["V_v"], sat_a, dt_seconds=1.0,
    )
    qf = solution["quality_flag"].copy()
    qf[~valid] = 0.0

    plot_stage3_solver_outputs(solution, u_ms, v_ms,
                                out_dir / "stage3_solver_outputs.png", title)

    # --- Build AI result dict ---
    ai = dict(
        h=solution["h"], u=u_ms, v=v_ms, qf=qf,
        chi2=solution["chi2"], sigma_h=solution["sigma_h"],
        w_u=w_u, w_v=w_v,
    )

    # --- QA ---
    qa_mask, qa_stats = apply_qa_filters(ai)
    logger.info("QA filtering:")
    logger.info("  base good:       %d", qa_stats["n_base_good"])
    for k in ["chi2", "sigma_h", "h_grad", "wind_speed", "w_mag"]:
        logger.info("  rejected by %-12s: %d", k, qa_stats[k])
    logger.info("  final good:      %d", qa_stats["n_final"])

    plot_stage4_qa(ai["h"], qa_mask, qa_stats,
                    out_dir / "stage4_qa_mask.png", title)

    # --- Write NetCDF ---
    nc_solution = {
        "u_wind": u_ms, "v_wind": v_ms, "h": solution["h"],
        "sigma_u": solution["sigma_u"], "sigma_v": solution["sigma_v"],
        "sigma_h": solution["sigma_h"],
        "chi2": solution["chi2"],
        "p_u": solution.get("p_u", np.zeros_like(u_ms)),
        "p_v": solution.get("p_v", np.zeros_like(u_ms)),
        "quality_flag": qf.astype(np.float32),
    }
    nc_path = out_dir / "stereo_retrieval.nc"
    ds_out = create_output_dataset(nc_solution, sat_a, t0)
    write_netcdf(ds_out, nc_path)
    logger.info("Wrote NetCDF: %s", nc_path)

    # --- Carr comparison ---
    logger.info("Loading Carr data from %s", args.carr_nc)
    carr = load_carr_data(args.carr_nc)
    logger.info("  Carr: %d / %d good sites", carr["n_good"], carr["n_total"])

    logger.info("Matching sites (no QA)...")
    m_raw = match_sites(carr, ai, sat_a)
    logger.info("  matched: %d", m_raw["n"])
    logger.info("Matching sites (with QA)...")
    m_qa = match_sites(carr, ai, sat_a, qa_mask=qa_mask)
    logger.info("  matched: %d", m_qa["n"])

    met_raw = compute_metrics(m_raw)
    met_qa = compute_metrics(m_qa)

    plot_compare_scatter(m_qa, met_qa, out_dir / "compare_scatter.png", title)
    plot_compare_wind(m_qa, met_qa, out_dir / "compare_wind.png", title)
    plot_compare_height_map(m_qa, out_dir / "compare_height_map.png", title)
    plot_compare_histograms(m_qa, out_dir / "compare_histograms.png", title)
    plot_compare_summary(met_raw, met_qa, qa_stats,
                          out_dir / "compare_summary.png", title)

    # --- Console summary ---
    print()
    print("=" * 70)
    print(f"RAFT Stereo Winds vs Carr — {title}")
    print("=" * 70)
    print(f"{'Metric':<25s}  {'Raw':>12s}  {'After QA':>12s}")
    print("-" * 70)
    fmt = [
        ("N matched", "n", "{:>12,}"),
        ("Height RMSE (m)", "h_rmse", "{:>12.0f}"),
        ("Height bias (m)", "h_bias", "{:>+12.0f}"),
        ("Height corr", "h_corr", "{:>12.4f}"),
        ("RMSVD (m/s)", "rv", "{:>12.2f}"),
        ("Speed bias (m/s)", "sb", "{:>+12.2f}"),
        ("Speed corr", "s_corr", "{:>12.4f}"),
        ("u-wind RMSE (m/s)", "u_rmse", "{:>12.2f}"),
        ("v-wind RMSE (m/s)", "v_rmse", "{:>12.2f}"),
    ]
    for label, key, f in fmt:
        print(f"{label:<25s}  {f.format(met_raw[key])}  {f.format(met_qa[key])}")
    print("=" * 70)
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
