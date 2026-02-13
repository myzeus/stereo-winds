"""Plot GOES-16 C14 full disk in native fixed-grid with coastlines."""

import sys
sys.path.insert(0, "zeus")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

from stereo_winds.data_loading import load_native_abi

nc = "/Users/tj/repos/stereo-winds/cache/goes16/ABI/2024/015/18/OR_ABI-L1b-RadF-M6C14_G16_s20240151800206_e20240151809514_c20240151809568.nc"
data, sat = load_native_abi(nc, "goes16")

print(f"Loaded: {data.shape}, range [{np.nanmin(data):.1f}, {np.nanmax(data):.1f}]")

# Geostationary projection for cartopy
geo_crs = ccrs.Geostationary(
    central_longitude=sat.sub_lon_deg,
    satellite_height=sat.satellite_height_m,
    sweep_axis=sat.sweep,
)

# Image extent in projection coordinates (meters from projection center)
H = sat.satellite_height_m
x_min = sat.x_offset * H
x_max = (sat.x_offset + sat.scale_x * (sat.n_cols - 1)) * H
y_max = sat.y_offset * H  # north edge (row 0)
y_min = (sat.y_offset + sat.scale_y * (sat.n_rows - 1)) * H
extent = [x_min, x_max, y_min, y_max]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": geo_crs})
ax.imshow(data, origin="upper", extent=extent, cmap="gray_r", vmin=20, vmax=180)
ax.coastlines(resolution="50m", color="cyan", linewidth=0.7)
ax.add_feature(cfeature.BORDERS, edgecolor="cyan", linewidth=0.3)
ax.set_title("GOES-16 Band C14 (11.2 µm) — 2024-01-15 18:00 UTC\nNative Fixed Grid", fontsize=13)

fig.tight_layout()
fig.savefig("/Users/tj/repos/stereo-winds/goes16_c14.png", dpi=150, bbox_inches="tight")
print("Saved: goes16_c14.png")
