"""Run all 4 RAFT disparity pairs for stereo wind retrieval and plot.

D1: A0 → A_minus  (GOES-16 temporal backward)
D2: A0 → A_plus   (GOES-16 temporal forward)
D3: A0 → B_minus  (GOES-16 → GOES-18 remapped, backward)
D4: A0 → B_plus   (GOES-16 → GOES-18 remapped, forward)
"""

import sys
sys.path.insert(0, "zeus")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import glob

from stereo_winds.data_loading import load_native_abi
from stereo_winds.remap import build_remap_lut, remap_image, save_remap_lut, load_remap_lut

# --- Load all scenes ---
print("Loading GOES-16 scenes...")
nc_a_tm = glob.glob("/Users/tj/repos/stereo-winds/cache/goes16/ABI/2024/015/18/*1850*.nc")[0]
nc_a_t0 = glob.glob("/Users/tj/repos/stereo-winds/cache/goes16/ABI/2024/015/19/*1900*.nc")[0]
nc_a_tp = glob.glob("/Users/tj/repos/stereo-winds/cache/goes16/ABI/2024/015/19/*1910*.nc")[0]

a_minus, sat_a = load_native_abi(nc_a_tm, "goes16")
a0, _ = load_native_abi(nc_a_t0, "goes16")
a_plus, _ = load_native_abi(nc_a_tp, "goes16")

print("Loading GOES-18 scenes...")
nc_b_tm = glob.glob("/Users/tj/repos/stereo-winds/cache/goes18/ABI/2024/015/18/*1850*.nc")[0]
nc_b_tp = glob.glob("/Users/tj/repos/stereo-winds/cache/goes18/ABI/2024/015/19/*1910*.nc")[0]

b_minus_native, sat_b = load_native_abi(nc_b_tm, "goes18")
b_plus_native, _ = load_native_abi(nc_b_tp, "goes18")

# --- Remap GOES-18 to GOES-16 grid ---
lut_path = "/Users/tj/repos/stereo-winds/cache/remap_lut_g16_g18.npz"
if os.path.exists(lut_path):
    print("Loading cached remap LUT...")
    col_b, row_b = load_remap_lut(lut_path)
else:
    print("Building remap LUT...")
    t0 = time.time()
    col_b, row_b = build_remap_lut(sat_a, sat_b)
    save_remap_lut(col_b, row_b, lut_path)
    print(f"  Built in {time.time() - t0:.1f}s")

valid_overlap = np.isfinite(col_b) & np.isfinite(row_b)
print(f"  Overlap: {valid_overlap.sum()}/{valid_overlap.size} ({100*valid_overlap.sum()/valid_overlap.size:.1f}%)")

print("Remapping GOES-18 scenes...")
b_minus = remap_image(b_minus_native, col_b, row_b)
b_plus = remap_image(b_plus_native, col_b, row_b)

# --- Valid data mask (finite in all 5 scenes) ---
valid = (
    np.isfinite(a0) & np.isfinite(a_minus) & np.isfinite(a_plus)
    & np.isfinite(b_minus) & np.isfinite(b_plus)
)
print(f"  Valid in all 5 scenes: {valid.sum()}/{valid.size} ({100*valid.sum()/valid.size:.1f}%)")

# --- Run RAFT on all 4 pairs ---
from zeus.inference.inference_flows import FlowRunner

ckpt = "/Users/tj/repos/stereo-winds/zeus/zeus/networks/weights/raft.g5nr.tar"
cache_dir = "/Users/tj/repos/stereo-winds/cache"

pairs = {
    "D1": ("A0 -> A_minus (temporal bwd)", a0, a_minus),
    "D2": ("A0 -> A_plus (temporal fwd)", a0, a_plus),
    "D3": ("A0 -> B_minus (cross-sat bwd)", a0, b_minus),
    "D4": ("A0 -> B_plus (cross-sat fwd)", a0, b_plus),
}

flows = {}
runner = None

for key, (label, img1, img2) in pairs.items():
    cache_path = os.path.join(cache_dir, f"flow_stereo_19z_{key}.npy")
    if os.path.exists(cache_path):
        print(f"Loading cached {key}...")
        flows[key] = np.load(cache_path)
    else:
        if runner is None:
            print("Initializing FlowRunner...")
            runner = FlowRunner(
                model_ckpt_path=ckpt, model_name="raft",
                tile_size=512, overlap=256, batch_size=8, device="mps",
            )
        print(f"Running RAFT {key}: {label}...")
        t_start = time.time()
        flow = runner.forward(
            img1[np.newaxis, np.newaxis, :, :],
            img2[np.newaxis, np.newaxis, :, :],
        )
        print(f"  Done in {time.time() - t_start:.1f}s")
        np.save(cache_path, flow)
        flows[key] = flow

# Squeeze batch dim and apply NaN mask
for key in flows:
    f = flows[key]
    if f.ndim == 4:
        f = f[0]  # (2, H, W)
    f[:, ~valid] = np.nan
    flows[key] = f

# --- Convert to m/s ---
px_scale = abs(sat_a.satellite_height_m * sat_a.scale_x)
dt = 600.0
print(f"Pixel scale: {px_scale:.0f} m, dt: {dt:.0f}s")

# --- Plot all 4 flow fields ---
import cartopy.crs as ccrs
import cartopy.feature as cfeature

H = sat_a.satellite_height_m
ext = [
    sat_a.x_offset * H,
    (sat_a.x_offset + sat_a.scale_x * (sat_a.n_cols - 1)) * H,
    (sat_a.y_offset + sat_a.scale_y * (sat_a.n_rows - 1)) * H,
    sat_a.y_offset * H,
]
geo = ccrs.Geostationary(
    central_longitude=sat_a.sub_lon_deg,
    satellite_height=H, sweep_axis=sat_a.sweep,
)

# Figure 1: All 4 pairs — du and speed
fig, axes = plt.subplots(4, 3, figsize=(22, 28),
                         subplot_kw={"projection": geo})

labels = [
    ("D1", "A0 → A- (temporal bwd)"),
    ("D2", "A0 → A+ (temporal fwd)"),
    ("D3", "A0 → B- (cross-sat bwd)"),
    ("D4", "A0 → B+ (cross-sat fwd)"),
]

for i, (key, label) in enumerate(labels):
    f = flows[key]
    du = f[0] * px_scale / dt
    dv = f[1] * px_scale / dt
    spd = np.sqrt(du**2 + dv**2)

    ax = axes[i, 0]
    im = ax.imshow(du, origin="upper", extent=ext, cmap="RdBu_r", vmin=-40, vmax=40)
    ax.coastlines(resolution="50m", color="black", linewidth=0.5)
    ax.set_title(f"{label}\nu (m/s)")
    plt.colorbar(im, ax=ax, shrink=0.6)

    ax = axes[i, 1]
    im = ax.imshow(dv, origin="upper", extent=ext, cmap="RdBu_r", vmin=-40, vmax=40)
    ax.coastlines(resolution="50m", color="black", linewidth=0.5)
    ax.set_title(f"v (m/s)")
    plt.colorbar(im, ax=ax, shrink=0.6)

    ax = axes[i, 2]
    im = ax.imshow(spd, origin="upper", extent=ext, cmap="magma", vmin=0, vmax=50)
    ax.coastlines(resolution="50m", color="white", linewidth=0.5)
    ax.set_title(f"speed (m/s)")
    plt.colorbar(im, ax=ax, shrink=0.6)

fig.suptitle("All 4 Disparity Pairs — GOES-16/18, 2024-01-15 19:00 UTC, C14", fontsize=15)
fig.tight_layout()
fig.savefig("/Users/tj/repos/stereo-winds/stereo_all_pairs.png", dpi=150, bbox_inches="tight")
print("Saved: stereo_all_pairs.png")

# Figure 2: Parallax isolation — cross-sat minus temporal
fig2, axes2 = plt.subplots(2, 3, figsize=(22, 14),
                           subplot_kw={"projection": geo})

for i, (key_cross, key_temp, label) in enumerate([
    ("D3", "D1", "Backward: D3-D1 (B- minus A-)"),
    ("D4", "D2", "Forward: D4-D2 (B+ minus A+)"),
]):
    diff = flows[key_cross] - flows[key_temp]
    du_diff = diff[0] * px_scale / dt
    dv_diff = diff[1] * px_scale / dt
    mag_diff = np.sqrt(du_diff**2 + dv_diff**2)

    vmax_d = float(np.nanpercentile(mag_diff, 99))

    ax = axes2[i, 0]
    im = ax.imshow(du_diff, origin="upper", extent=ext, cmap="RdBu_r",
                   vmin=-vmax_d, vmax=vmax_d)
    ax.coastlines(resolution="50m", color="black", linewidth=0.5)
    ax.set_title(f"{label}\nΔu (m/s)")
    plt.colorbar(im, ax=ax, shrink=0.6)

    ax = axes2[i, 1]
    im = ax.imshow(dv_diff, origin="upper", extent=ext, cmap="RdBu_r",
                   vmin=-vmax_d, vmax=vmax_d)
    ax.coastlines(resolution="50m", color="black", linewidth=0.5)
    ax.set_title(f"Δv (m/s)")
    plt.colorbar(im, ax=ax, shrink=0.6)

    ax = axes2[i, 2]
    im = ax.imshow(mag_diff, origin="upper", extent=ext, cmap="hot", vmin=0, vmax=vmax_d)
    ax.coastlines(resolution="50m", color="cyan", linewidth=0.5)
    ax.set_title(f"|Δ| parallax magnitude (m/s)")
    plt.colorbar(im, ax=ax, shrink=0.6)

fig2.suptitle("Parallax Isolation: Cross-Satellite minus Temporal Flow\nSignal should correlate with cloud height", fontsize=14)
fig2.tight_layout()
fig2.savefig("/Users/tj/repos/stereo-winds/stereo_parallax.png", dpi=150, bbox_inches="tight")
print("Saved: stereo_parallax.png")
