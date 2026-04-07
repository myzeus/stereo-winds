"""Compare pretrained vs fine-tuned RAFT against Carr et al. baseline.

Re-runs RAFT with both checkpoints on the 2025-01-08 GOES-16/18 C14 scene,
solves stereo winds, and compares against Carr retrieval sites.
"""

import sys
import os
import glob
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "zeus"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import datetime as dt
import xarray as xr
import torch
from scipy import stats
from scipy.ndimage import sobel

from stereo_winds.config import GOES16_CONFIG, GOES18_CONFIG
from stereo_winds.remap import load_remap_lut, remap_image
from stereo_winds.solver import (
    build_design_matrix, solve_stereo_winds, pixels_to_wind_ms,
)
from stereo_winds.data_loading import load_native_abi
from stereo_winds.disparity import StereoDisparity
from stereo_winds.time_model import compute_scene_times
from stereo_winds.navigation import geodetic_to_fixed_grid, scanning_angle_to_pixel
from stereo_winds.validation.metrics import rmsvd, speed_bias, height_rmse, correlation

# =========================================================================
# Configuration
# =========================================================================
T0 = dt.datetime(2025, 1, 8, 19, 0)
DT_MINUTES = 10.0
BAND = "C14"

sat_a = GOES16_CONFIG
sat_b = GOES18_CONFIG

cache_root = "/home/ubuntu/earthnet-us-east-3/cache"
carr_dir = os.path.join(BASE, "carr_data")

PRETRAINED_CKPT = os.path.join(
    BASE, "zeus/zeus/networks/weights/raft-128.202509.epoch1434.ckpt"
)
tuned_ckpts = sorted(
    [p for p in glob.glob(os.path.join(BASE, "stereo-winds*/*/checkpoints/*.ckpt"))
     if os.path.exists(p)],
    key=os.path.getmtime,
)
TUNED_CKPT = tuned_ckpts[-1]

# Convert fine-tuned PL checkpoint for FlowRunner
_ft_data = torch.load(TUNED_CKPT, map_location="cpu", weights_only=False)
_ft_sd = {k.removeprefix("raft."): v
           for k, v in _ft_data["state_dict"].items() if k.startswith("raft.")}
TUNED_COMPAT = os.path.join(os.path.dirname(TUNED_CKPT), "tuned_compat_carr.pt")
if not os.path.exists(TUNED_COMPAT):
    torch.save({"model": _ft_sd, "global_step": _ft_data.get("global_step", 0)},
               TUNED_COMPAT)
del _ft_data, _ft_sd

print(f"Pretrained: {os.path.basename(PRETRAINED_CKPT)}")
print(f"Fine-tuned: {TUNED_CKPT}")

# =========================================================================
# Load shared data
# =========================================================================
print("\n--- Loading data ---")

# Remap LUT + parallax
col_b, row_b = load_remap_lut(os.path.join(cache_root, "remap_lut_goes16_goes18.npz"))
pdata = np.load(os.path.join(cache_root, "parallax_goes16_goes18.npz"))
w_u, w_v = pdata["w_u"], pdata["w_v"]

scene_times = compute_scene_times(T0, DT_MINUTES, sat_a, sat_b)
H_matrix = build_design_matrix(
    w_u, w_v,
    dt_a_minus=scene_times["A_minus"], dt_a_plus=scene_times["A_plus"],
    dt_b_minus=scene_times["B_minus"], dt_b_plus=scene_times["B_plus"],
)

# Load raw satellite images
t_minus = T0 - dt.timedelta(minutes=DT_MINUTES)
t_plus = T0 + dt.timedelta(minutes=DT_MINUTES)


def find_abi(sat_id, band, target_time):
    """Find cached ABI file."""
    year = target_time.year
    doy = int(target_time.strftime("%j"))
    hour = target_time.hour
    hhmm = target_time.strftime("%H%M")
    search = os.path.join(
        cache_root, sat_id, "ABI", str(year), f"{doy:03d}", f"{hour:02d}",
        f"*{band}*s{year}{doy:03d}{hhmm}*.nc"
    )
    matches = glob.glob(search)
    if not matches:
        raise FileNotFoundError(f"No file matching {search}")
    return matches[0]


a_minus, _ = load_native_abi(find_abi("goes16", BAND, t_minus), satellite_id="goes16")
a0, _      = load_native_abi(find_abi("goes16", BAND, T0), satellite_id="goes16")
a_plus, _  = load_native_abi(find_abi("goes16", BAND, t_plus), satellite_id="goes16")
b_minus    = remap_image(load_native_abi(find_abi("goes18", BAND, t_minus), satellite_id="goes18")[0], col_b, row_b)
b_plus     = remap_image(load_native_abi(find_abi("goes18", BAND, t_plus), satellite_id="goes18")[0], col_b, row_b)

valid = (np.isfinite(a0) & np.isfinite(a_minus) & np.isfinite(a_plus)
         & np.isfinite(b_minus) & np.isfinite(b_plus))
print(f"  Valid: {valid.sum():,}")

images = {"A_minus": a_minus, "A0": a0, "A_plus": a_plus,
          "B_minus": b_minus, "B_plus": b_plus}

# =========================================================================
# Load Carr baseline
# =========================================================================
print("\n--- Loading Carr data ---")


def load_carr(filepath):
    ds = xr.open_dataset(filepath, engine="h5netcdf")
    lat, lon = ds["lat"].values, ds["lon"].values
    u, v = ds["V_3D"].values[:, 0], ds["V_3D"].values[:, 1]
    h = ds["H_3D"].values
    dqf = ds["DQF_3D"].values
    ds.close()
    good = (dqf == 0) & np.isfinite(h) & np.isfinite(u) & np.isfinite(v) & (h >= 0) & (h <= 20000)
    return dict(lat=lat, lon=lon, u=u, v=v, h=h, good=good, n_good=int(good.sum()))


carr = load_carr(os.path.join(carr_dir, "GOES_GOES_B14_20250081900208.nc"))
print(f"  Carr C14: {carr['n_good']:,} good sites")


# =========================================================================
# Run RAFT + solve for each checkpoint
# =========================================================================
def run_raft_and_solve(ckpt_path, label):
    print(f"\n--- {label} ---")
    t0 = time.time()
    disp = StereoDisparity(model_ckpt_path=ckpt_path, tile_size=512, overlap=256,
                           batch_size=8, device="cuda")
    flows = disp.compute_all(images)
    for key in flows:
        flows[key][:, ~valid] = np.nan
    print(f"  RAFT: {time.time()-t0:.1f}s")

    solution = solve_stereo_winds(flows, H_matrix, sat_a=sat_a, sat_b=sat_b, n_iter=3)
    u_ms, v_ms = pixels_to_wind_ms(solution["V_u"], solution["V_v"], sat_a, dt_seconds=1.0)
    h = solution["h"]
    qf = solution["quality_flag"].copy()
    qf[~valid] = 0.0

    return dict(h=h, u=u_ms, v=v_ms, qf=qf, chi2=solution["chi2"],
                sigma_h=solution["sigma_h"], w_u=w_u, w_v=w_v)


result_pre = run_raft_and_solve(PRETRAINED_CKPT, "Pretrained")
result_ft  = run_raft_and_solve(TUNED_COMPAT, "Fine-tuned")


# =========================================================================
# QA filtering
# =========================================================================
QA_THRESHOLDS = dict(
    chi2_max=0.2,
    sigma_h_max=5000.0,
    h_grad_max=3000.0,
    wind_speed_max=100.0,
    w_mag_min=0.0003,
)
qa_desc = (f"chi2<={QA_THRESHOLDS['chi2_max']}, "
           f"sigma_h<={QA_THRESHOLDS['sigma_h_max']:.0f}m, "
           f"grad<={QA_THRESHOLDS['h_grad_max']:.0f}m/px, "
           f"spd<={QA_THRESHOLDS['wind_speed_max']:.0f}m/s, "
           f"w_mag>={QA_THRESHOLDS['w_mag_min']}")
print(f"\nQA thresholds: {qa_desc}")


def compute_qa_mask(ai, thresholds):
    """Apply QA filters and return per-pixel bool mask."""
    h = ai["h"]
    base = (ai["qf"] > 0) & np.isfinite(h)
    chi2_ok = np.isfinite(ai["chi2"]) & (ai["chi2"] <= thresholds["chi2_max"])
    sigma_ok = np.isfinite(ai["sigma_h"]) & (ai["sigma_h"] <= thresholds["sigma_h_max"])
    # Height gradient (Sobel)
    h_filled = np.where(np.isfinite(h), h, 0.0)
    grad = np.sqrt(sobel(h_filled, axis=1)**2 + sobel(h_filled, axis=0)**2) / 8.0
    grad_ok = grad <= thresholds["h_grad_max"]
    spd = np.sqrt(ai["u"]**2 + ai["v"]**2)
    spd_ok = np.isfinite(spd) & (spd <= thresholds["wind_speed_max"])
    w_mag = np.sqrt(ai["w_u"]**2 + ai["w_v"]**2)
    par_ok = w_mag >= thresholds["w_mag_min"]
    qa = base & chi2_ok & sigma_ok & grad_ok & spd_ok & par_ok
    print(f"  QA: {int(base.sum()):,} base → {int(qa.sum()):,} passed")
    return qa


print("\nPretrained QA:")
qa_pre = compute_qa_mask(result_pre, QA_THRESHOLDS)
print("Fine-tuned QA:")
qa_ft = compute_qa_mask(result_ft, QA_THRESHOLDS)


# =========================================================================
# Match Carr sites to AI pixels
# =========================================================================
def match_carr(carr_data, ai_result, sat, qa_mask=None):
    g = carr_data["good"]
    lat_g, lon_g = carr_data["lat"][g], carr_data["lon"][g]
    h_c, u_c, v_c = carr_data["h"][g], carr_data["u"][g], carr_data["v"][g]

    x_ang, y_ang = geodetic_to_fixed_grid(lat_g, lon_g, sat, h_m=0.0)
    col_f, row_f = scanning_angle_to_pixel(x_ang, y_ang, sat)
    ci, ri = np.round(col_f).astype(int), np.round(row_f).astype(int)

    in_bounds = (ci >= 0) & (ci < sat.n_cols) & (ri >= 0) & (ri < sat.n_rows) & np.isfinite(col_f)
    ci, ri = np.clip(ci, 0, sat.n_cols-1), np.clip(ri, 0, sat.n_rows-1)

    h_ai = ai_result["h"][ri, ci]
    u_ai = ai_result["u"][ri, ci]
    v_ai = ai_result["v"][ri, ci]
    qf_ai = ai_result["qf"][ri, ci]

    ok = in_bounds & (qf_ai > 0) & np.isfinite(h_ai) & np.isfinite(u_ai) & np.isfinite(v_ai)
    if qa_mask is not None:
        ok = ok & qa_mask[ri, ci]

    return dict(
        carr_h=h_c[ok], carr_u=u_c[ok], carr_v=v_c[ok],
        carr_spd=np.sqrt(u_c[ok]**2 + v_c[ok]**2),
        ai_h=h_ai[ok], ai_u=u_ai[ok], ai_v=v_ai[ok],
        ai_spd=np.sqrt(u_ai[ok]**2 + v_ai[ok]**2),
        lat=lat_g[ok], lon=lon_g[ok],
        n=int(ok.sum()),
    )


print("\n  Raw (no QA):")
m_pre_raw = match_carr(carr, result_pre, sat_a)
m_ft_raw  = match_carr(carr, result_ft, sat_a)
print(f"    pretrained={m_pre_raw['n']:,}, fine-tuned={m_ft_raw['n']:,}")

print("  With QA:")
m_pre = match_carr(carr, result_pre, sat_a, qa_mask=qa_pre)
m_ft  = match_carr(carr, result_ft, sat_a, qa_mask=qa_ft)
print(f"    pretrained={m_pre['n']:,}, fine-tuned={m_ft['n']:,}")

# Matched-count QA: tighten FT chi2 threshold so the same number of pixels
# pass QA as the pretrained, giving a fair apples-to-apples comparison.
n_pre_qa = int(qa_pre.sum())
ft_chi2_valid = result_ft["chi2"][result_ft["qf"] > 0]
ft_chi2_valid = ft_chi2_valid[np.isfinite(ft_chi2_valid)]
ft_chi2_thr = float(np.percentile(ft_chi2_valid, 100 * n_pre_qa / len(ft_chi2_valid)))
print(f"\n  Matched-count QA: tightening FT chi2 to {ft_chi2_thr:.4f} to match {n_pre_qa:,} pre QA pixels")
qa_ft_matched = compute_qa_mask(
    result_ft,
    {**QA_THRESHOLDS, "chi2_max": ft_chi2_thr},
)
print("  With matched-count QA:")
m_ft_matched = match_carr(carr, result_ft, sat_a, qa_mask=qa_ft_matched)
print(f"    fine-tuned (matched)={m_ft_matched['n']:,}")


def metrics(m):
    return dict(
        n=m["n"],
        h_rmse=height_rmse(m["ai_h"], m["carr_h"]),
        h_bias=float(np.mean(m["ai_h"] - m["carr_h"])),
        h_corr=correlation(m["ai_h"], m["carr_h"]),
        rv=rmsvd(m["ai_u"], m["ai_v"], m["carr_u"], m["carr_v"]),
        sb=speed_bias(m["ai_u"], m["ai_v"], m["carr_u"], m["carr_v"]),
        s_corr=correlation(m["ai_spd"], m["carr_spd"]),
    )


met_pre_raw = metrics(m_pre_raw)
met_ft_raw  = metrics(m_ft_raw)
met_pre = metrics(m_pre)
met_ft  = metrics(m_ft)
met_ft_matched = metrics(m_ft_matched)


# =========================================================================
# Figure 1: Side-by-side scatter plots (2x3)
# =========================================================================
print("\n--- Generating plots ---")

fig, axes = plt.subplots(2, 3, figsize=(22, 14))

for row, (label, m, met) in enumerate([
    ("Pretrained", m_pre, met_pre),
    ("Fine-tuned", m_ft, met_ft),
]):
    # Height
    ax = axes[row, 0]
    ax.scatter(m["carr_h"]/1000, m["ai_h"]/1000, c=m["carr_h"]/1000,
               cmap="turbo", vmin=0, vmax=16, s=0.5, alpha=0.3, rasterized=True)
    ax.plot([0, 18], [0, 18], "k--", lw=1)
    slope, intercept, r, _, _ = stats.linregress(m["carr_h"], m["ai_h"])
    ax.plot([0, 18], [intercept/1000, (slope*18000+intercept)/1000], "r-", lw=1)
    ax.set_xlim(0, 18); ax.set_ylim(0, 18)
    ax.set_xlabel("Carr Height (km)"); ax.set_ylabel("AI Height (km)")
    ax.set_title(f"{label} — Height", fontweight="bold")
    ax.text(0.05, 0.95, f"N={met['n']:,}\nRMSE={met['h_rmse']:.0f}m\n"
            f"Bias={met['h_bias']:+.0f}m\nr={met['h_corr']:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # u-wind
    ax = axes[row, 1]
    ax.scatter(m["carr_u"], m["ai_u"], c=m["carr_h"]/1000,
               cmap="turbo", vmin=0, vmax=16, s=0.5, alpha=0.3, rasterized=True)
    ax.plot([-60, 60], [-60, 60], "k--", lw=1)
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
    ax.set_xlabel("Carr u-wind (m/s)"); ax.set_ylabel("AI u-wind (m/s)")
    ax.set_title(f"{label} — u-wind", fontweight="bold")
    u_rmse = float(np.sqrt(np.mean((m["ai_u"]-m["carr_u"])**2)))
    ax.text(0.05, 0.95, f"RMSE={u_rmse:.2f} m/s\nr={correlation(m['ai_u'],m['carr_u']):.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # v-wind
    ax = axes[row, 2]
    ax.scatter(m["carr_v"], m["ai_v"], c=m["carr_h"]/1000,
               cmap="turbo", vmin=0, vmax=16, s=0.5, alpha=0.3, rasterized=True)
    ax.plot([-60, 60], [-60, 60], "k--", lw=1)
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
    ax.set_xlabel("Carr v-wind (m/s)"); ax.set_ylabel("AI v-wind (m/s)")
    ax.set_title(f"{label} — v-wind", fontweight="bold")
    v_rmse = float(np.sqrt(np.mean((m["ai_v"]-m["carr_v"])**2)))
    ax.text(0.05, 0.95, f"RMSE={v_rmse:.2f} m/s\nr={correlation(m['ai_v'],m['carr_v']):.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

fig.suptitle(f"AI vs Carr — Pretrained vs Fine-tuned\nGOES-16/18 {T0.strftime('%Y-%m-%d %H:%M')} UTC, {BAND}\nQA: {qa_desc}",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out1 = os.path.join(BASE, "compare_ft_vs_carr_scatter.png")
fig.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: {out1}")


# =========================================================================
# Figure 2: Metrics comparison (bar chart + RMSVD/bias vs height)
# =========================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))

# Bar chart
metric_names = ["Height\nRMSE (m)", "Height\nBias (m)", "RMSVD\n(m/s)", "Speed\nBias (m/s)"]
vals_pre = [met_pre["h_rmse"], abs(met_pre["h_bias"]), met_pre["rv"], abs(met_pre["sb"])]
vals_ft  = [met_ft["h_rmse"],  abs(met_ft["h_bias"]),  met_ft["rv"],  abs(met_ft["sb"])]

x = np.arange(len(metric_names))
w = 0.35
axes2[0].bar(x - w/2, vals_pre, w, label="Pretrained", color="C0")
axes2[0].bar(x + w/2, vals_ft,  w, label="Fine-tuned", color="C1")
axes2[0].set_xticks(x)
axes2[0].set_xticklabels(metric_names, fontsize=10)
axes2[0].set_ylabel("Value")
axes2[0].set_title("Metrics vs Carr (lower = better)", fontweight="bold")
axes2[0].legend()

# RMSVD vs height
h_edges = np.arange(0, 17000, 1000)
h_centers = (h_edges[:-1] + h_edges[1:]) / 2

for label, m, color in [("Pretrained", m_pre, "C0"), ("Fine-tuned", m_ft, "C1")]:
    bin_rv = []
    for i in range(len(h_edges)-1):
        mask = (m["carr_h"] >= h_edges[i]) & (m["carr_h"] < h_edges[i+1])
        if mask.sum() > 10:
            du = m["ai_u"][mask] - m["carr_u"][mask]
            dv = m["ai_v"][mask] - m["carr_v"][mask]
            bin_rv.append(np.sqrt(np.mean(du**2 + dv**2)))
        else:
            bin_rv.append(np.nan)
    bin_rv = np.array(bin_rv)
    ok = np.isfinite(bin_rv)
    axes2[1].plot(h_centers[ok]/1000, bin_rv[ok], "o-", color=color, label=label, markersize=4)

axes2[1].set_xlabel("Carr Height (km)")
axes2[1].set_ylabel("RMSVD (m/s)")
axes2[1].set_title("RMSVD vs Height", fontweight="bold")
axes2[1].legend()
axes2[1].set_xlim(0, 16)

# Height bias vs height
for label, m, color in [("Pretrained", m_pre, "C0"), ("Fine-tuned", m_ft, "C1")]:
    bin_bias = []
    for i in range(len(h_edges)-1):
        mask = (m["carr_h"] >= h_edges[i]) & (m["carr_h"] < h_edges[i+1])
        if mask.sum() > 10:
            bin_bias.append(np.mean(m["ai_h"][mask] - m["carr_h"][mask]) / 1000)
        else:
            bin_bias.append(np.nan)
    bin_bias = np.array(bin_bias)
    ok = np.isfinite(bin_bias)
    axes2[2].plot(h_centers[ok]/1000, bin_bias[ok], "o-", color=color, label=label, markersize=4)

axes2[2].axhline(0, color="gray", ls="--", lw=0.8)
axes2[2].set_xlabel("Carr Height (km)")
axes2[2].set_ylabel("Height Bias (AI - Carr, km)")
axes2[2].set_title("Height Bias vs Height", fontweight="bold")
axes2[2].legend()
axes2[2].set_xlim(0, 16)

fig2.suptitle(f"Pretrained vs Fine-tuned — Validation against Carr\n{T0.strftime('%Y-%m-%d %H:%M')} UTC, {BAND}\nQA: {qa_desc}",
              fontsize=13, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.92])
out2 = os.path.join(BASE, "compare_ft_vs_carr_metrics.png")
fig2.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"  Saved: {out2}")


# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 95)
print(f"{'Metric':<25s}  {'Pre Raw':>10s}  {'Pre QA':>10s}  {'FT Raw':>10s}  {'FT QA':>10s}  {'FT Match':>10s}")
print("-" * 95)
for label, key, fmt in [
    ("N matched",          "n",      ">10,"),
    ("Height RMSE (m)",    "h_rmse", ">10.0f"),
    ("Height bias (m)",    "h_bias", ">+10.0f"),
    ("Height correlation", "h_corr", ">10.4f"),
    ("RMSVD (m/s)",        "rv",     ">10.2f"),
    ("Speed bias (m/s)",   "sb",     ">+10.2f"),
    ("Speed correlation",  "s_corr", ">10.4f"),
]:
    vals = [met_pre_raw[key], met_pre[key], met_ft_raw[key], met_ft[key], met_ft_matched[key]]
    cols = "  ".join(f"{v:{fmt}}" for v in vals)
    print(f"{label:<25s}  {cols}")
print("-" * 95)
print(f"\nQA: {qa_desc}")
print(f"FT Match: chi2 tightened to {ft_chi2_thr:.4f} to match Pre QA pixel count")
print("\nDone!")
