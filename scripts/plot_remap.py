"""Plot GOES-18 remapped to GOES-16 viewpoint, with comparison panels."""

import sys
sys.path.insert(0, "zeus")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import time

from stereo_winds.data_loading import load_native_abi
from stereo_winds.remap import build_remap_lut, remap_image

# Load both satellites
nc16 = "/Users/tj/repos/stereo-winds/cache/goes16/ABI/2024/015/18/OR_ABI-L1b-RadF-M6C14_G16_s20240151800206_e20240151809514_c20240151809568.nc"
nc18 = "/Users/tj/repos/stereo-winds/cache/goes18/ABI/2024/015/18/OR_ABI-L1b-RadF-M6C14_G18_s20240151800216_e20240151809524_c20240151809568.nc"

print("Loading GOES-16...")
data16, sat16 = load_native_abi(nc16, "goes16")
print(f"  Shape: {data16.shape}, range [{np.nanmin(data16):.1f}, {np.nanmax(data16):.1f}]")

print("Loading GOES-18...")
data18, sat18 = load_native_abi(nc18, "goes18")
print(f"  Shape: {data18.shape}, range [{np.nanmin(data18):.1f}, {np.nanmax(data18):.1f}]")

# Build remap LUT (GOES-16 grid → GOES-18 pixel coords)
print("Building remap LUT...")
t0 = time.time()
col_b, row_b = build_remap_lut(sat16, sat18)
print(f"  LUT built in {time.time() - t0:.1f}s")

valid = np.isfinite(col_b)
print(f"  Valid overlap: {valid.sum()} / {valid.size} pixels ({100*valid.sum()/valid.size:.1f}%)")

# Remap GOES-18 onto GOES-16 grid
print("Remapping GOES-18 to GOES-16 grid...")
data18_remapped = remap_image(data18, col_b, row_b)
print(f"  Remapped range: [{np.nanmin(data18_remapped):.1f}, {np.nanmax(data18_remapped):.1f}]")

# --- Plot ---
H = sat16.satellite_height_m
x_min = sat16.x_offset * H
x_max = (sat16.x_offset + sat16.scale_x * (sat16.n_cols - 1)) * H
y_max = sat16.y_offset * H
y_min = (sat16.y_offset + sat16.scale_y * (sat16.n_rows - 1)) * H
ext16 = [x_min, x_max, y_min, y_max]

H18 = sat18.satellite_height_m
x_min18 = sat18.x_offset * H18
x_max18 = (sat18.x_offset + sat18.scale_x * (sat18.n_cols - 1)) * H18
y_max18 = sat18.y_offset * H18
y_min18 = (sat18.y_offset + sat18.scale_y * (sat18.n_rows - 1)) * H18
ext18 = [x_min18, x_max18, y_min18, y_max18]

geo16 = ccrs.Geostationary(central_longitude=sat16.sub_lon_deg, satellite_height=H, sweep_axis=sat16.sweep)
geo18 = ccrs.Geostationary(central_longitude=sat18.sub_lon_deg, satellite_height=H18, sweep_axis=sat18.sweep)

vmin, vmax = 20, 180

fig = plt.figure(figsize=(20, 14))

# Panel 1: GOES-16 native
ax1 = fig.add_subplot(2, 2, 1, projection=geo16)
ax1.imshow(data16, origin="upper", extent=ext16, cmap="gray_r", vmin=vmin, vmax=vmax)
ax1.coastlines(resolution="50m", color="cyan", linewidth=0.7)
ax1.set_title("GOES-16 (native)", fontsize=12)

# Panel 2: GOES-18 native
ax2 = fig.add_subplot(2, 2, 2, projection=geo18)
ax2.imshow(data18, origin="upper", extent=ext18, cmap="gray_r", vmin=vmin, vmax=vmax)
ax2.coastlines(resolution="50m", color="cyan", linewidth=0.7)
ax2.set_title("GOES-18 (native)", fontsize=12)

# Panel 3: GOES-18 remapped onto GOES-16 grid
ax3 = fig.add_subplot(2, 2, 3, projection=geo16)
ax3.imshow(data18_remapped, origin="upper", extent=ext16, cmap="gray_r", vmin=vmin, vmax=vmax)
ax3.coastlines(resolution="50m", color="cyan", linewidth=0.7)
ax3.set_title("GOES-18 remapped to GOES-16 grid", fontsize=12)

# Panel 4: Difference (GOES-16 minus GOES-18 remapped) = parallax signal
ax4 = fig.add_subplot(2, 2, 4, projection=geo16)
diff = data16.astype(np.float64) - data18_remapped.astype(np.float64)
d99 = np.nanpercentile(np.abs(diff), 99)
im4 = ax4.imshow(diff, origin="upper", extent=ext16, cmap="RdBu_r", vmin=-d99, vmax=d99)
ax4.coastlines(resolution="50m", color="black", linewidth=0.7)
ax4.set_title("G16 − G18_remapped (parallax signal)", fontsize=12)
plt.colorbar(im4, ax=ax4, shrink=0.7, label="Radiance diff")

fig.suptitle("GOES-16 / GOES-18 Remap Verification — 2024-01-15 18:00 UTC, Band C14", fontsize=14)
fig.tight_layout()
fig.savefig("/Users/tj/repos/stereo-winds/remap_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: remap_comparison.png")
