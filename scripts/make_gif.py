#!/usr/bin/env python
"""Generate a 4:3 looping GIF thumbnail from Arraylake stereo wind data.

Renders clean frames (no colorbars or legends) with a logo overlay,
then assembles into a compact looping GIF.

Usage:
    python scripts/make_gif.py                         # First 24 hours
    python scripts/make_gif.py --start 0 --end 48      # Custom range
    python scripts/make_gif.py --width 800              # Larger output
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import argparse
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from stereo_winds.config import SatelliteConfig

# --- Config ---
REPO = "zeus-ai/geo-stereo-winds"
BRANCH = "solver-v4"
GROUP = "goes19_goes18/1h/C14/2026-01"
LOGO_PATH = Path("Zeus_Logo_BlueBackground.png")

sat = SatelliteConfig(satellite_id="goes19", sub_lon_deg=-75.0, sweep="x")
H_sat = sat.satellite_height_m
MS_TO_KT = 1.94384

geo = ccrs.Geostationary(central_longitude=sat.sub_lon_deg,
                          satellite_height=H_sat, sweep_axis=sat.sweep)
pc = ccrs.PlateCarree()

ext = [sat.x_offset * H_sat,
       (sat.x_offset + sat.scale_x * (sat.n_cols - 1)) * H_sat,
       (sat.y_offset + sat.scale_y * (sat.n_rows - 1)) * H_sat,
       sat.y_offset * H_sat]

# Height colormap (barbs)
H_MIN, H_MAX = 0, 16000
CMAP_H = plt.cm.turbo
NORM_H = mcolors.Normalize(vmin=H_MIN, vmax=H_MAX)
N_BINS = 12
H_EDGES = np.linspace(H_MIN, H_MAX, N_BINS + 1)

# Wind speed colormap (background)
SPD_MIN, SPD_MAX = 0, 60
CMAP_SPD = plt.cm.twilight
NORM_SPD = mcolors.Normalize(vmin=SPD_MIN, vmax=SPD_MAX)

STRIDE = 60
FRAME_DIR = Path("output/gif_frames")
OUTPUT_GIF = Path("output/stereo_winds_thumbnail.gif")
PALETTE = Path("output/_gif_palette.png")


def render_frame(ds_t, time_val, frame_idx, frame_dir, logo_img):
    """Render a clean wind barb frame — no colorbars or legends."""
    u = ds_t["u_wind"].values
    v = ds_t["v_wind"].values
    h = ds_t["cloud_top_height"].values
    qf = ds_t["quality_flag"].values

    spd = np.sqrt(np.where(np.isfinite(u), u, 0)**2 +
                  np.where(np.isfinite(v), v, 0)**2)
    good = (qf > 0) & np.isfinite(h) & (h >= 0) & (h <= 20000) & (spd < 100)
    u_m = np.where(good, u, np.nan)
    v_m = np.where(good, v, np.nan)
    h_m = np.where(good, h, np.nan)
    spd_bg = np.where(good, spd, np.nan)

    # 4:3 figure — extent in geostationary coords fills frame exactly
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_axes([0, 0, 1, 1], projection=geo)
    y_half = 5.6e6     # slightly past poles for padding
    x_half = y_half * 4 / 3  # enforce 4:3 ratio matching figure
    ax.set_global()
    ax.set_xlim(-x_half, x_half)
    ax.set_ylim(-y_half, y_half)
    ax.set_facecolor('white')

    # Background: wind speed
    ax.imshow(spd_bg, origin="upper", extent=ext, cmap=CMAP_SPD,
              norm=NORM_SPD, alpha=0.7, interpolation="nearest")
    ax.coastlines(resolution="110m", color="cyan", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="cyan", alpha=0.4)

    # Subsampled barbs
    rows_s = np.arange(0, sat.n_rows, STRIDE)
    cols_s = np.arange(0, sat.n_cols, STRIDE)
    cg, rg = np.meshgrid(cols_s, rows_s)
    u_s = u_m[rg, cg]
    v_s = v_m[rg, cg]
    h_s = h_m[rg, cg]
    spd_s = np.sqrt(np.where(np.isfinite(u_s), u_s, 0)**2 +
                    np.where(np.isfinite(v_s), v_s, 0)**2)
    x_s = (cg * sat.scale_x + sat.x_offset) * H_sat
    y_s = (rg * sat.scale_y + sat.y_offset) * H_sat
    good_b = np.isfinite(u_s) & np.isfinite(v_s) & np.isfinite(h_s) & (spd_s > 0.5)

    for i in range(N_BINS):
        lo, hi = H_EDGES[i], H_EDGES[i + 1]
        mask = good_b & (h_s >= lo) & (h_s < hi)
        if not mask.any():
            continue
        color = CMAP_H(NORM_H((lo + hi) / 2))
        ax.barbs(x_s[mask], y_s[mask],
                 u_s[mask] * MS_TO_KT, v_s[mask] * MS_TO_KT,
                 length=4.5, linewidth=0.35,
                 barb_increments=dict(half=5, full=10, flag=50),
                 color=color, zorder=3 + i)

    # Timestamp (small, bottom-left)
    t_str = str(time_val)[:16].replace("T", " ") + " UTC"
    ax.text(0.02, 0.03, t_str, transform=ax.transAxes,
            fontsize=10, color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=2))

    # Logo (top-right corner)
    if logo_img is not None:
        im_box = OffsetImage(logo_img, zoom=0.04, alpha=0.85)
        ab = AnnotationBbox(im_box, (0.98, 0.97), xycoords="axes fraction",
                            frameon=False, box_alignment=(1, 1))
        ax.add_artist(ab)

    out_path = frame_dir / f"frame_{frame_idx:04d}.png"
    fig.savefig(str(out_path), dpi=150, facecolor="white",
                pad_inches=0)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate GIF thumbnail")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=24, help="End frame index (exclusive)")
    parser.add_argument("--fps", type=int, default=6, help="GIF framerate")
    parser.add_argument("--width", type=int, default=640, help="Output width (height = width*3/4)")
    parser.add_argument("-o", "--output", type=str, default=str(OUTPUT_GIF))
    args = parser.parse_args()

    output = Path(args.output)
    height = int(args.width * 3 / 4)

    # Load logo
    logo_img = None
    if LOGO_PATH.exists():
        logo_img = mpimg.imread(str(LOGO_PATH))
        print(f"Loaded logo: {LOGO_PATH}")

    # Open Arraylake data
    from arraylake import Client
    print(f"Opening {REPO} branch={BRANCH} group={GROUP}...")
    repo = Client().get_repo(REPO)
    store = repo.readonly_session(BRANCH).store
    ds = xr.open_zarr(store, group=GROUP, consolidated=False)

    n_times = len(ds.time)
    args.end = min(args.end, n_times)
    n_frames = args.end - args.start
    print(f"Rendering {n_frames} clean frames...")

    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    # Render frames in batches
    BATCH = 6
    for batch_start in range(args.start, args.end, BATCH):
        batch_end = min(batch_start + BATCH, args.end)
        print(f"  Loading frames {batch_start}..{batch_end - 1}")
        batch_ds = ds.isel(time=slice(batch_start, batch_end)).load()

        for fi in range(batch_start, batch_end):
            out_path = FRAME_DIR / f"frame_{fi:04d}.png"
            if out_path.exists():
                print(f"  Frame {fi} exists, skipping")
                continue
            local_idx = fi - batch_start
            time_val = batch_ds.time.values[local_idx]
            print(f"  Frame {fi}: {time_val}")
            ds_t = batch_ds.isel(time=local_idx)
            render_frame(ds_t, time_val, fi, FRAME_DIR, logo_img)

        del batch_ds

    # Assemble GIF via ffmpeg (two-pass palette)
    print(f"\nAssembling {args.width}x{height} GIF at {args.fps} fps...")
    vf = f"scale={args.width}:{height}:flags=lanczos"

    # Pass 1: palette
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-start_number", str(args.start),
        "-i", str(FRAME_DIR / "frame_%04d.png"),
        "-frames:v", str(n_frames),
        "-vf", f"{vf},palettegen=stats_mode=diff",
        str(PALETTE),
    ], check=True, capture_output=True)

    # Pass 2: GIF
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-start_number", str(args.start),
        "-i", str(FRAME_DIR / "frame_%04d.png"),
        "-i", str(PALETTE),
        "-frames:v", str(n_frames),
        "-lavfi", f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3",
        "-loop", "0",
        str(output),
    ], check=True, capture_output=True)

    PALETTE.unlink(missing_ok=True)

    size_mb = output.stat().st_size / 1024 / 1024
    print(f"Saved: {output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
