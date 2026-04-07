"""Compare stereo winds from pretrained vs fine-tuned RAFT checkpoints.

Runs both checkpoints on the same scene and generates side-by-side plots
of height, chi2, and wind fields.
"""

import sys
import os
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "zeus"))

import numpy as np
import datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from stereo_winds.config import GOES18_CONFIG, SatelliteConfig
from stereo_winds.remap import load_remap_lut, remap_image
from stereo_winds.solver import (
    build_design_matrix, solve_stereo_winds, pixels_to_wind_ms,
)
from stereo_winds.data_loading import load_native_abi
from stereo_winds.disparity import StereoDisparity
from stereo_winds.time_model import compute_scene_times

# =========================================================================
# Configuration
# =========================================================================
T0 = dt.datetime(2026, 3, 13, 19, 0)
DT_MINUTES = 10.0
DATE_STR = T0.strftime("%Y%m%d")
BAND = "C14"

GOES19_CONFIG = SatelliteConfig(
    satellite_id="goes19", sub_lon_deg=-75.0, sweep="x",
)
sat_a = GOES19_CONFIG
sat_b = GOES18_CONFIG

cache_root = os.path.join(BASE, "cache")
cache_dir = os.path.join(cache_root, DATE_STR)

PRETRAINED_CKPT = os.path.join(
    BASE, "zeus/zeus/networks/weights/raft-128.202509.epoch1434.ckpt"
)
# Find the latest fine-tuned checkpoint
import glob
tuned_ckpts = sorted(
    glob.glob(os.path.join(BASE, "stereo-winds/*/checkpoints/*.ckpt")),
    key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
)
tuned_ckpts = [p for p in tuned_ckpts if os.path.exists(p)]
if not tuned_ckpts:
    print("ERROR: No fine-tuned checkpoints found")
    sys.exit(1)
TUNED_CKPT = tuned_ckpts[-1]

print(f"Pretrained: {os.path.basename(PRETRAINED_CKPT)}")
print(f"Fine-tuned: {TUNED_CKPT}")

# Convert fine-tuned PL checkpoint to FlowRunner format.
# PL state_dict has "raft." prefix on all keys; FlowRunner expects bare RAFT keys.
import torch
_ft_data = torch.load(TUNED_CKPT, map_location="cpu", weights_only=False)
_ft_sd = {k.removeprefix("raft."): v
           for k, v in _ft_data["state_dict"].items()
           if k.startswith("raft.")}
TUNED_COMPAT = os.path.join(os.path.dirname(TUNED_CKPT), "tuned_compat.pt")
if not os.path.exists(TUNED_COMPAT):
    torch.save({"model": _ft_sd, "global_step": _ft_data.get("global_step", 0)},
               TUNED_COMPAT)
    print(f"  Converted to: {TUNED_COMPAT}")
del _ft_data, _ft_sd

# =========================================================================
# Load data (shared between both runs)
# =========================================================================
print("\n--- Loading data ---")

# Remap LUT + parallax
remap_path = os.path.join(cache_root, "remap_lut_g19_g18.npz")
col_b, row_b = load_remap_lut(remap_path)

par_path = os.path.join(cache_root, "parallax_goes19_goes18.npz")
pdata = np.load(par_path)
w_u, w_v = pdata["w_u"], pdata["w_v"]

# Design matrix
scene_times = compute_scene_times(T0, DT_MINUTES, sat_a, sat_b)
H_matrix = build_design_matrix(
    w_u, w_v,
    dt_a_minus=scene_times["A_minus"],
    dt_a_plus=scene_times["A_plus"],
    dt_b_minus=scene_times["B_minus"],
    dt_b_plus=scene_times["B_plus"],
)

# Load and remap scenes
t_minus = T0 - dt.timedelta(minutes=DT_MINUTES)
t_plus = T0 + dt.timedelta(minutes=DT_MINUTES)

import s3fs
fs = s3fs.S3FileSystem(anon=True)

def find_and_download(bucket, sat_id, band, target_time):
    """Find the file closest to target_time on S3 and download."""
    hhmm = target_time.strftime("%H%M")
    year = target_time.year
    doy = target_time.strftime("%j")
    hour = target_time.hour

    s3_dir = f"{bucket}/ABI-L1b-RadF/{year}/{doy}/{hour:02d}/"
    all_files = fs.ls(s3_dir)
    matches = [f for f in all_files if band in os.path.basename(f)
               and f"s{year}{doy}{hhmm}" in os.path.basename(f)]
    if not matches:
        matches = [f for f in all_files if band in os.path.basename(f)]
        matches.sort(key=lambda f: abs(
            int(os.path.basename(f).split("_s")[1][7:11]) - int(hhmm)))
    if not matches:
        raise FileNotFoundError(f"No {band} for {sat_id} near {hhmm} in {s3_dir}")

    s3path = matches[0]
    fname = os.path.basename(s3path)
    local_dir = os.path.join(cache_dir, sat_id)
    os.makedirs(local_dir, exist_ok=True)
    local = os.path.join(local_dir, fname)
    if not os.path.exists(local):
        print(f"  Downloading {fname}...")
        fs.get(s3path, local)
    else:
        print(f"  Cached: {fname}")
    return local

files = {
    "a_minus": find_and_download("noaa-goes19", "goes19", BAND, t_minus),
    "a0":      find_and_download("noaa-goes19", "goes19", BAND, T0),
    "a_plus":  find_and_download("noaa-goes19", "goes19", BAND, t_plus),
    "b_minus": find_and_download("noaa-goes18", "goes18", BAND, t_minus),
    "b_plus":  find_and_download("noaa-goes18", "goes18", BAND, t_plus),
}

a_minus, _ = load_native_abi(files["a_minus"], satellite_id="goes19")
a0, _      = load_native_abi(files["a0"], satellite_id="goes19")
a_plus, _  = load_native_abi(files["a_plus"], satellite_id="goes19")
b_minus    = remap_image(load_native_abi(files["b_minus"], satellite_id="goes18")[0], col_b, row_b)
b_plus     = remap_image(load_native_abi(files["b_plus"], satellite_id="goes18")[0], col_b, row_b)

valid = (np.isfinite(a0) & np.isfinite(a_minus) & np.isfinite(a_plus)
         & np.isfinite(b_minus) & np.isfinite(b_plus))
print(f"  Valid pixels: {valid.sum():,}")

images = {
    "A_minus": a_minus, "A0": a0, "A_plus": a_plus,
    "B_minus": b_minus, "B_plus": b_plus,
}


# =========================================================================
# Run both checkpoints
# =========================================================================
def run_pipeline(ckpt_path, label):
    """Run RAFT + solver with a given checkpoint."""
    print(f"\n--- Running {label} ---")
    t0 = time.time()

    disp = StereoDisparity(
        model_ckpt_path=ckpt_path, tile_size=512, overlap=256,
        batch_size=8, device="cuda",
    )
    flows = disp.compute_all(images)
    for key in flows:
        flows[key][:, ~valid] = np.nan
    print(f"  RAFT: {time.time() - t0:.1f}s")

    t1 = time.time()
    solution = solve_stereo_winds(flows, H_matrix)
    u_ms, v_ms = pixels_to_wind_ms(solution["V_u"], solution["V_v"], sat_a, dt_seconds=1.0)
    print(f"  Solve: {time.time() - t1:.1f}s")

    h = solution["h"]
    chi2 = solution["chi2"]
    qf = solution["quality_flag"].copy()
    qf[~valid] = 0.0
    good = (qf > 0) & (h >= 0) & (h <= 20000)

    print(f"  QC-passed: {int(good.sum()):,}")
    if good.any():
        print(f"  Median h: {np.nanmedian(h[good])/1000:.1f} km")
        spd = np.sqrt(u_ms**2 + v_ms**2)
        print(f"  Median wind: {np.nanmedian(spd[good]):.1f} m/s")

    return {
        "h": h, "chi2": chi2, "u_ms": u_ms, "v_ms": v_ms,
        "good": good, "flows": flows,
    }


result_pre = run_pipeline(PRETRAINED_CKPT, "Pretrained")
result_ft  = run_pipeline(TUNED_COMPAT, "Fine-tuned")

# =========================================================================
# Plot comparison
# =========================================================================
print("\n--- Plotting comparison ---")

# IR background
vmin_r, vmax_r = np.nanpercentile(a0[np.isfinite(a0)], [1, 99])
ir_img = 1.0 - np.clip((a0 - vmin_r) / (vmax_r - vmin_r), 0, 1)

fig, axes = plt.subplots(3, 3, figsize=(20, 18))

titles_row = ["Pretrained", "Fine-tuned", "Difference"]
labels = {"pre": result_pre, "ft": result_ft}

# --- Row 0: Height ---
for j, (key, res) in enumerate(labels.items()):
    h_km = np.where(res["good"], res["h"] / 1000, np.nan)
    im = axes[0, j].imshow(h_km, cmap="turbo", vmin=0, vmax=18, interpolation="nearest")
    axes[0, j].set_title(f"Height (km) — {titles_row[j]}", fontsize=11)
    axes[0, j].axis("off")
    fig.colorbar(im, ax=axes[0, j], fraction=0.046, pad=0.04)

# Height difference
both_good = result_pre["good"] & result_ft["good"]
h_diff = np.where(both_good, (result_ft["h"] - result_pre["h"]) / 1000, np.nan)
im = axes[0, 2].imshow(h_diff, cmap="RdBu_r", vmin=-5, vmax=5, interpolation="nearest")
axes[0, 2].set_title(f"Height diff (km) — FT minus Pre", fontsize=11)
axes[0, 2].axis("off")
fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

# --- Row 1: Chi2 ---
for j, (key, res) in enumerate(labels.items()):
    chi2_vis = np.where(res["good"], np.clip(res["chi2"], 0, 5), np.nan)
    im = axes[1, j].imshow(chi2_vis, cmap="hot", vmin=0, vmax=5, interpolation="nearest")
    axes[1, j].set_title(f"Chi2 — {titles_row[j]}", fontsize=11)
    axes[1, j].axis("off")
    fig.colorbar(im, ax=axes[1, j], fraction=0.046, pad=0.04)

# Chi2 difference
chi2_diff = np.where(both_good, result_ft["chi2"] - result_pre["chi2"], np.nan)
im = axes[1, 2].imshow(chi2_diff, cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
axes[1, 2].set_title(f"Chi2 diff — FT minus Pre", fontsize=11)
axes[1, 2].axis("off")
fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

# --- Row 2: Wind speed ---
for j, (key, res) in enumerate(labels.items()):
    spd = np.where(res["good"], np.sqrt(res["u_ms"]**2 + res["v_ms"]**2), np.nan)
    im = axes[2, j].imshow(spd, cmap="viridis", vmin=0, vmax=40, interpolation="nearest")
    axes[2, j].set_title(f"Wind speed (m/s) — {titles_row[j]}", fontsize=11)
    axes[2, j].axis("off")
    fig.colorbar(im, ax=axes[2, j], fraction=0.046, pad=0.04)

# Wind speed difference
spd_pre = np.sqrt(result_pre["u_ms"]**2 + result_pre["v_ms"]**2)
spd_ft = np.sqrt(result_ft["u_ms"]**2 + result_ft["v_ms"]**2)
spd_diff = np.where(both_good, spd_ft - spd_pre, np.nan)
im = axes[2, 2].imshow(spd_diff, cmap="RdBu_r", vmin=-10, vmax=10, interpolation="nearest")
axes[2, 2].set_title(f"Wind speed diff (m/s) — FT minus Pre", fontsize=11)
axes[2, 2].axis("off")
fig.colorbar(im, ax=axes[2, 2], fraction=0.046, pad=0.04)

fig.suptitle(
    f"Checkpoint Comparison — GOES-19/18, {T0.strftime('%Y-%m-%d %H:%M')} UTC, {BAND}\n"
    f"Pre: {os.path.basename(PRETRAINED_CKPT)}  |  FT: {os.path.basename(TUNED_CKPT)}",
    fontsize=13, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out_path = os.path.join(BASE, "compare_checkpoints.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: {out_path}")

# =========================================================================
# Summary statistics
# =========================================================================
print("\n" + "=" * 60)
print("Summary statistics (QC-passed pixels)")
print("=" * 60)
for label, res in [("Pretrained", result_pre), ("Fine-tuned", result_ft)]:
    g = res["good"]
    n = int(g.sum())
    h = res["h"][g]
    chi2 = res["chi2"][g]
    spd = np.sqrt(res["u_ms"][g]**2 + res["v_ms"][g]**2)
    print(f"\n  {label}:")
    print(f"    QC-passed pixels: {n:,}")
    print(f"    Height: median={np.median(h)/1000:.1f} km, mean={np.mean(h)/1000:.1f} km, std={np.std(h)/1000:.1f} km")
    print(f"    Chi2:   median={np.median(chi2):.3f}, mean={np.mean(chi2):.3f}, p95={np.percentile(chi2, 95):.3f}")
    print(f"    Wind:   median={np.median(spd):.1f} m/s, mean={np.mean(spd):.1f} m/s")

if both_good.any():
    print(f"\n  Pixels in common: {int(both_good.sum()):,}")
    h_d = (result_ft["h"][both_good] - result_pre["h"][both_good]) / 1000
    print(f"  Height diff:  mean={np.mean(h_d):.2f} km, RMSD={np.sqrt(np.mean(h_d**2)):.2f} km")
    chi2_d = result_ft["chi2"][both_good] - result_pre["chi2"][both_good]
    print(f"  Chi2 diff:    mean={np.mean(chi2_d):.3f} (negative = FT better)")

# =========================================================================
# Histograms and scatter plots
# =========================================================================
print("\n--- Generating histogram and scatter plots ---")

h_pre = result_pre["h"][both_good] / 1000
h_ft  = result_ft["h"][both_good] / 1000
chi2_pre = result_pre["chi2"][both_good]
chi2_ft  = result_ft["chi2"][both_good]
spd_pre_bg = np.sqrt(result_pre["u_ms"][both_good]**2 + result_pre["v_ms"][both_good]**2)
spd_ft_bg  = np.sqrt(result_ft["u_ms"][both_good]**2 + result_ft["v_ms"][both_good]**2)

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

# --- Row 0: Histograms ---
# Height histogram
axes2[0, 0].hist(h_pre, bins=100, range=(0, 18), alpha=0.6, label="Pretrained", color="C0", density=True)
axes2[0, 0].hist(h_ft, bins=100, range=(0, 18), alpha=0.6, label="Fine-tuned", color="C1", density=True)
axes2[0, 0].set_xlabel("Height (km)")
axes2[0, 0].set_ylabel("Density")
axes2[0, 0].set_title("Height distribution")
axes2[0, 0].legend()

# Chi2 histogram
axes2[0, 1].hist(chi2_pre, bins=100, range=(0, 10), alpha=0.6, label="Pretrained", color="C0", density=True)
axes2[0, 1].hist(chi2_ft, bins=100, range=(0, 10), alpha=0.6, label="Fine-tuned", color="C1", density=True)
axes2[0, 1].set_xlabel("Chi2")
axes2[0, 1].set_ylabel("Density")
axes2[0, 1].set_title("Chi2 distribution")
axes2[0, 1].legend()

# Wind speed histogram
axes2[0, 2].hist(spd_pre_bg, bins=100, range=(0, 50), alpha=0.6, label="Pretrained", color="C0", density=True)
axes2[0, 2].hist(spd_ft_bg, bins=100, range=(0, 50), alpha=0.6, label="Fine-tuned", color="C1", density=True)
axes2[0, 2].set_xlabel("Wind speed (m/s)")
axes2[0, 2].set_ylabel("Density")
axes2[0, 2].set_title("Wind speed distribution")
axes2[0, 2].legend()

# --- Row 1: 2D density scatter plots ---
from matplotlib.colors import LogNorm

# Subsample for scatter density (15M pixels is too many)
rng = np.random.default_rng(0)
n_pts = min(500_000, int(both_good.sum()))
idx = rng.choice(int(both_good.sum()), n_pts, replace=False)

# Height scatter
axes2[1, 0].hist2d(h_pre[idx], h_ft[idx], bins=200, range=[[0, 18], [0, 18]],
                    cmap="magma", norm=LogNorm())
axes2[1, 0].plot([0, 18], [0, 18], "w--", lw=1, alpha=0.7)
axes2[1, 0].set_xlabel("Pretrained height (km)")
axes2[1, 0].set_ylabel("Fine-tuned height (km)")
axes2[1, 0].set_title(f"Height scatter (RMSD={np.sqrt(np.mean((h_ft-h_pre)**2)):.2f} km)")
axes2[1, 0].set_aspect("equal")

# Chi2 scatter
axes2[1, 1].hist2d(np.clip(chi2_pre[idx], 0, 10), np.clip(chi2_ft[idx], 0, 10),
                    bins=200, range=[[0, 10], [0, 10]], cmap="magma", norm=LogNorm())
axes2[1, 1].plot([0, 10], [0, 10], "w--", lw=1, alpha=0.7)
axes2[1, 1].set_xlabel("Pretrained chi2")
axes2[1, 1].set_ylabel("Fine-tuned chi2")
frac_improved = (chi2_ft < chi2_pre).mean()
axes2[1, 1].set_title(f"Chi2 scatter ({frac_improved:.0%} improved)")
axes2[1, 1].set_aspect("equal")

# Wind speed scatter
axes2[1, 2].hist2d(np.clip(spd_pre_bg[idx], 0, 50), np.clip(spd_ft_bg[idx], 0, 50),
                    bins=200, range=[[0, 50], [0, 50]], cmap="magma", norm=LogNorm())
axes2[1, 2].plot([0, 50], [0, 50], "w--", lw=1, alpha=0.7)
axes2[1, 2].set_xlabel("Pretrained wind speed (m/s)")
axes2[1, 2].set_ylabel("Fine-tuned wind speed (m/s)")
axes2[1, 2].set_title("Wind speed scatter")
axes2[1, 2].set_aspect("equal")

fig2.suptitle(
    f"Pretrained vs Fine-tuned — GOES-19/18, {T0.strftime('%Y-%m-%d %H:%M')} UTC, {BAND}\n"
    f"N={int(both_good.sum()):,} common QC-passed pixels",
    fontsize=13, fontweight="bold",
)
fig2.tight_layout(rect=[0, 0, 1, 0.93])
out2 = os.path.join(BASE, "compare_checkpoints_stats.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"  Saved: {out2}")

print("\nDone!")
