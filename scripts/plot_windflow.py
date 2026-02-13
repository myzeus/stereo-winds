"""Run RAFT optical flow on GOES-16 temporal pair and plot results."""

import sys
sys.path.insert(0, "zeus")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time

from stereo_winds.data_loading import load_native_abi

# Load GOES-16 t0 and t-10min
nc_t0 = "/Users/tj/repos/stereo-winds/cache/goes16/ABI/2024/015/18/OR_ABI-L1b-RadF-M6C14_G16_s20240151800206_e20240151809514_c20240151809568.nc"
nc_tm = "/Users/tj/repos/stereo-winds/cache/goes16/ABI/2024/015/17/OR_ABI-L1b-RadF-M6C14_G16_s20240151750206_e20240151759514_c20240151759570.nc"
nc_tp = "/Users/tj/repos/stereo-winds/cache/goes16/ABI/2024/015/18/OR_ABI-L1b-RadF-M6C14_G16_s20240151810206_e20240151819514_c20240151819581.nc"

print("Loading images...")
data_t0, sat = load_native_abi(nc_t0, "goes16")
data_tm, _ = load_native_abi(nc_tm, "goes16")
data_tp, _ = load_native_abi(nc_tp, "goes16")
print(f"  Shapes: t0={data_t0.shape}, t-={data_tm.shape}, t+={data_tp.shape}")

# Initialize FlowRunner
from zeus.inference.inference_flows import FlowRunner

ckpt = "/Users/tj/repos/stereo-winds/zeus/zeus/networks/weights/raft.g5nr.tar"
print("Initializing FlowRunner...")
runner = FlowRunner(
    model_ckpt_path=ckpt,
    model_name="raft",
    tile_size=512,
    overlap=256,
    batch_size=8,
    device="mps",
)

# Run optical flow with caching
import os
cache_bwd = "/Users/tj/repos/stereo-winds/cache/flow_g16_bwd.npy"
cache_fwd = "/Users/tj/repos/stereo-winds/cache/flow_g16_fwd.npy"

if os.path.exists(cache_bwd):
    print("Loading cached backward flow...")
    flow_bwd = np.load(cache_bwd)
else:
    print("Running RAFT: t0 → t-10min...")
    img1 = data_t0[np.newaxis, np.newaxis, :, :]
    img2 = data_tm[np.newaxis, np.newaxis, :, :]
    t_start = time.time()
    flow_bwd = runner.forward(img1, img2)
    print(f"  Done in {time.time() - t_start:.1f}s, shape={flow_bwd.shape}")
    np.save(cache_bwd, flow_bwd)

if os.path.exists(cache_fwd):
    print("Loading cached forward flow...")
    flow_fwd = np.load(cache_fwd)
else:
    print("Running RAFT: t0 → t+10min...")
    img1 = data_t0[np.newaxis, np.newaxis, :, :]
    img2 = data_tp[np.newaxis, np.newaxis, :, :]
    t_start = time.time()
    flow_fwd = runner.forward(img1, img2)
    print(f"  Done in {time.time() - t_start:.1f}s, shape={flow_fwd.shape}")
    np.save(cache_fwd, flow_fwd)

# Convert pixel displacement to m/s
# Pixel scale at nadir: H * scale = 35786023 * 5.6e-5 ≈ 2004 m
# dt = 600s
px_scale = abs(sat.satellite_height_m * sat.scale_x)  # ~2004 m
dt = 600.0  # 10 minutes
print(f"  Pixel scale: {px_scale:.0f} m, dt: {dt:.0f} s")

# flow shape is (1, 2, H, W) — squeeze batch dim
flow_bwd = flow_bwd[0]  # (2, H, W)
flow_fwd = flow_fwd[0]  # (2, H, W)

u_bwd = flow_bwd[0] * px_scale / dt
v_bwd = flow_bwd[1] * px_scale / dt
spd_bwd = np.sqrt(u_bwd**2 + v_bwd**2)

u_fwd = flow_fwd[0] * px_scale / dt
v_fwd = flow_fwd[1] * px_scale / dt
spd_fwd = np.sqrt(u_fwd**2 + v_fwd**2)

print(f"  Backward wind speed: median={np.nanmedian(spd_bwd):.1f}, max={np.nanmax(spd_bwd):.1f} m/s")
print(f"  Forward wind speed: median={np.nanmedian(spd_fwd):.1f}, max={np.nanmax(spd_fwd):.1f} m/s")

# --- Plot ---
import cartopy.crs as ccrs
import cartopy.feature as cfeature

H = sat.satellite_height_m
x_min = sat.x_offset * H
x_max = (sat.x_offset + sat.scale_x * (sat.n_cols - 1)) * H
y_max = sat.y_offset * H
y_min = (sat.y_offset + sat.scale_y * (sat.n_rows - 1)) * H
ext = [x_min, x_max, y_min, y_max]
geo = ccrs.Geostationary(central_longitude=sat.sub_lon_deg, satellite_height=H, sweep_axis=sat.sweep)

fig = plt.figure(figsize=(22, 16))

# Row 1: Backward flow (t0 → t-10min)
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

# Row 2: Forward flow (t0 → t+10min)
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

fig.suptitle("GOES-16 C14 Optical Flow Winds — 2024-01-15 18:00 UTC\nTop: t0→t-10min (backward)  |  Bottom: t0→t+10min (forward)", fontsize=14)
fig.tight_layout()
fig.savefig("/Users/tj/repos/stereo-winds/windflow_goes16.png", dpi=150, bbox_inches="tight")
print("Saved: windflow_goes16.png")
