"""Run RAFT optical flow on GOES-16 19:00 UTC triplet with proper NaN masking."""

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

nc_tm = glob.glob("/Users/tj/repos/stereo-winds/cache/goes16/ABI/2024/015/18/*1850*.nc")[0]
nc_t0 = glob.glob("/Users/tj/repos/stereo-winds/cache/goes16/ABI/2024/015/19/*1900*.nc")[0]
nc_tp = glob.glob("/Users/tj/repos/stereo-winds/cache/goes16/ABI/2024/015/19/*1910*.nc")[0]

print("Loading images...")
data_t0, sat = load_native_abi(nc_t0, "goes16")
data_tm, _ = load_native_abi(nc_tm, "goes16")
data_tp, _ = load_native_abi(nc_tp, "goes16")
print(f"  Shapes: t0={data_t0.shape}, t-={data_tm.shape}, t+={data_tp.shape}")

# Build a valid-data mask: pixel must have finite data in ALL 3 scenes
valid_mask = np.isfinite(data_t0) & np.isfinite(data_tm) & np.isfinite(data_tp)
print(f"  Valid in all 3 scenes: {valid_mask.sum()}/{valid_mask.size} ({100*valid_mask.sum()/valid_mask.size:.1f}%)")

# Initialize FlowRunner
from zeus.inference.inference_flows import FlowRunner

ckpt = "/Users/tj/repos/stereo-winds/zeus/zeus/networks/weights/raft.g5nr.tar"

cache_bwd = "/Users/tj/repos/stereo-winds/cache/flow_g16_19z_bwd.npy"
cache_fwd = "/Users/tj/repos/stereo-winds/cache/flow_g16_19z_fwd.npy"

if os.path.exists(cache_bwd) and os.path.exists(cache_fwd):
    print("Loading cached flows...")
    flow_bwd = np.load(cache_bwd)
    flow_fwd = np.load(cache_fwd)
else:
    print("Initializing FlowRunner...")
    runner = FlowRunner(
        model_ckpt_path=ckpt, model_name="raft",
        tile_size=512, overlap=256, batch_size=8, device="mps",
    )

    img1 = data_t0[np.newaxis, np.newaxis, :, :]

    print("Running RAFT: t0 → t-10min...")
    t_start = time.time()
    flow_bwd = runner.forward(img1, data_tm[np.newaxis, np.newaxis, :, :])
    print(f"  Done in {time.time() - t_start:.1f}s")
    np.save(cache_bwd, flow_bwd)

    print("Running RAFT: t0 → t+10min...")
    t_start = time.time()
    flow_fwd = runner.forward(img1, data_tp[np.newaxis, np.newaxis, :, :])
    print(f"  Done in {time.time() - t_start:.1f}s")
    np.save(cache_fwd, flow_fwd)

# Squeeze batch dim: (1, 2, H, W) → (2, H, W)
flow_bwd = flow_bwd[0]
flow_fwd = flow_fwd[0]

# Apply NaN mask: set flow to NaN where any input scene had missing data
flow_bwd[:, ~valid_mask] = np.nan
flow_fwd[:, ~valid_mask] = np.nan

# Convert pixel displacement to m/s
px_scale = abs(sat.satellite_height_m * sat.scale_x)  # ~2004 m
dt = 600.0
print(f"  Pixel scale: {px_scale:.0f} m, dt: {dt:.0f} s")

u_bwd = flow_bwd[0] * px_scale / dt
v_bwd = flow_bwd[1] * px_scale / dt
spd_bwd = np.sqrt(u_bwd**2 + v_bwd**2)

u_fwd = flow_fwd[0] * px_scale / dt
v_fwd = flow_fwd[1] * px_scale / dt
spd_fwd = np.sqrt(u_fwd**2 + v_fwd**2)

print(f"  Backward: median={np.nanmedian(spd_bwd):.1f}, p99={np.nanpercentile(spd_bwd, 99):.1f} m/s")
print(f"  Forward:  median={np.nanmedian(spd_fwd):.1f}, p99={np.nanpercentile(spd_fwd, 99):.1f} m/s")

# --- Plot ---
import cartopy.crs as ccrs
import cartopy.feature as cfeature

H = sat.satellite_height_m
ext = [
    sat.x_offset * H,
    (sat.x_offset + sat.scale_x * (sat.n_cols - 1)) * H,
    (sat.y_offset + sat.scale_y * (sat.n_rows - 1)) * H,
    sat.y_offset * H,
]
geo = ccrs.Geostationary(central_longitude=sat.sub_lon_deg, satellite_height=H, sweep_axis=sat.sweep)

fig = plt.figure(figsize=(22, 16))

# Row 1: Backward flow
ax1 = fig.add_subplot(2, 3, 1, projection=geo)
im1 = ax1.imshow(u_bwd, origin="upper", extent=ext, cmap="RdBu_r", vmin=-40, vmax=40)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5)
ax1.set_title("Backward: u-wind (m/s)")
plt.colorbar(im1, ax=ax1, shrink=0.6)

ax2 = fig.add_subplot(2, 3, 2, projection=geo)
im2 = ax2.imshow(v_bwd, origin="upper", extent=ext, cmap="RdBu_r", vmin=-40, vmax=40)
ax2.coastlines(resolution="50m", color="black", linewidth=0.5)
ax2.set_title("Backward: v-wind (m/s)")
plt.colorbar(im2, ax=ax2, shrink=0.6)

ax3 = fig.add_subplot(2, 3, 3, projection=geo)
im3 = ax3.imshow(spd_bwd, origin="upper", extent=ext, cmap="magma", vmin=0, vmax=50)
ax3.coastlines(resolution="50m", color="white", linewidth=0.5)
ax3.set_title("Backward: speed (m/s)")
plt.colorbar(im3, ax=ax3, shrink=0.6)

# Row 2: Forward flow
ax4 = fig.add_subplot(2, 3, 4, projection=geo)
im4 = ax4.imshow(u_fwd, origin="upper", extent=ext, cmap="RdBu_r", vmin=-40, vmax=40)
ax4.coastlines(resolution="50m", color="black", linewidth=0.5)
ax4.set_title("Forward: u-wind (m/s)")
plt.colorbar(im4, ax=ax4, shrink=0.6)

ax5 = fig.add_subplot(2, 3, 5, projection=geo)
im5 = ax5.imshow(v_fwd, origin="upper", extent=ext, cmap="RdBu_r", vmin=-40, vmax=40)
ax5.coastlines(resolution="50m", color="black", linewidth=0.5)
ax5.set_title("Forward: v-wind (m/s)")
plt.colorbar(im5, ax=ax5, shrink=0.6)

ax6 = fig.add_subplot(2, 3, 6, projection=geo)
im6 = ax6.imshow(spd_fwd, origin="upper", extent=ext, cmap="magma", vmin=0, vmax=50)
ax6.coastlines(resolution="50m", color="white", linewidth=0.5)
ax6.set_title("Forward: speed (m/s)")
plt.colorbar(im6, ax=ax6, shrink=0.6)

fig.suptitle("GOES-16 C14 Optical Flow Winds — 2024-01-15 19:00 UTC\nTop: t0→t-10min (backward)  |  Bottom: t0→t+10min (forward)", fontsize=14)
fig.tight_layout()
fig.savefig("/Users/tj/repos/stereo-winds/windflow_goes16_19z.png", dpi=150, bbox_inches="tight")
print("Saved: windflow_goes16_19z.png")
