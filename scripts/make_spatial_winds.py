"""Side-by-side spatial stereo-wind maps over the GOES-16/18 overlap.

Pretrained (init) vs tuned, height-binned wind barbs over an IR background
on a geostationary projection. Style follows
stereo-winds-lambda/scripts/plot_ai_vs_carr_barbs.py.

Reads cached full-disk retrieval zarrs (u_wind, v_wind, cloud_top_height,
chi_squared, sigma_h, quality_flag) and the source IR monthly zarr for the
background. Both models share the GOES-16 fixed grid.

Example
-------
    python scripts/make_spatial_winds.py \\
        --init-zarr  $CACHE/init_ep254_202501_iter3.zarr \\
        --tuned-zarr $CACHE/hreg1s75_202501_iter3.zarr \\
        --source-zarr $DATA/zarrs/goes16_C14_202501.zarr \\
        --time-index 0 --stride 40 \\
        --out $RUNS/.../spatial_winds_202501_t0.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import sobel

from stereo_winds.config import GOES16_CONFIG

SAT = GOES16_CONFIG
H_MIN, H_MAX = 0, 16000
CMAP_H = plt.cm.turbo
NORM_H = mcolors.Normalize(vmin=H_MIN, vmax=H_MAX)
MS_TO_KT = 1.94384
N_BINS = 12
H_EDGES = np.linspace(H_MIN, H_MAX, N_BINS + 1)
DEFAULT_EXTENT = [-135, -25, -55, 60]   # GOES-16/18 overlap

QA = dict(chi2_max=0.2, sigma_h_max=5000.0, h_grad_max=3000.0,
          wind_speed_max=100.0, min_height=1000.0)


def qa_mask(u, v, h, chi2, sigma_h, qf):
    h_filled = np.where(np.isfinite(h), h, 0.0)
    grad = np.hypot(sobel(h_filled, axis=1), sobel(h_filled, axis=0)) / 8.0
    spd = np.hypot(u, v)
    return ((qf > 0) & np.isfinite(h) & np.isfinite(chi2)
            & (chi2 <= QA["chi2_max"])
            & np.isfinite(sigma_h) & (sigma_h <= QA["sigma_h_max"])
            & (grad <= QA["h_grad_max"])
            & np.isfinite(spd) & (spd <= QA["wind_speed_max"])
            & (h >= QA["min_height"]) & (h <= 20000))


def load_scene(zarr_path, ti):
    ds = xr.open_zarr(zarr_path)
    if "time" in ds.dims:
        ds = ds.isel(time=ti)
    g = lambda k: ds[k].values
    return dict(u=g("u_wind"), v=g("v_wind"), h=g("cloud_top_height"),
                chi2=g("chi_squared"), sigma_h=g("sigma_h"), qf=g("quality_flag"),
                time=np.datetime64(ds.time.values))


def plot_barbs_binned(ax, x, y, u_kt, v_kt, h, good, transform, length=5, lw=0.4):
    """Wind barbs in height bins, colored by altitude (layered)."""
    for i in range(N_BINS):
        lo, hi = H_EDGES[i], H_EDGES[i + 1]
        m = good & (h >= lo) & (h < hi)
        if not m.any():
            continue
        ax.barbs(x[m], y[m], u_kt[m], v_kt[m], length=length, linewidth=lw,
                 barb_increments=dict(half=5, full=10, flag=50),
                 color=CMAP_H(NORM_H((lo + hi) / 2)), zorder=3 + i, transform=transform)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-zarr", required=True)
    ap.add_argument("--tuned-zarr", required=True)
    ap.add_argument("--source-zarr", required=True, help="IR monthly zarr for background")
    ap.add_argument("--time-index", type=int, default=0)
    ap.add_argument("--stride", type=int, default=40, help="barb subsample step (px)")
    ap.add_argument("--extent", type=float, nargs=4, default=DEFAULT_EXTENT,
                    metavar=("W", "E", "S", "N"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--chi2-max", type=float, default=QA["chi2_max"],
                    help="QA chi-squared cut (raise to loosen QA / pass more "
                         "barbs; default 0.2)")
    ap.add_argument("--no-qa", action="store_true",
                    help="Bypass the QA mask entirely — show every finite "
                         "retrieval (no chi2/sigma_h/gradient/speed cuts)")
    ap.add_argument("--left-label", default="init (--init-zarr)")
    ap.add_argument("--right-label", default="tuned (--tuned-zarr)")
    args = ap.parse_args()
    QA["chi2_max"] = args.chi2_max
    print(f"QA chi2_max = {QA['chi2_max']}  no_qa={args.no_qa}", flush=True)

    def scene_mask(s):
        if args.no_qa:
            return np.isfinite(s["u"]) & np.isfinite(s["v"]) & np.isfinite(s["h"])
        return qa_mask(**{k: s[k] for k in ["u", "v", "h", "chi2", "sigma_h", "qf"]})

    # Band label from the source zarr filename (e.g. goes16_C08_202501.zarr -> C08)
    import re
    m = re.search(r"_(C\d{2})_", Path(args.source_zarr).name)
    band = m.group(1) if m else "C??"

    si = load_scene(args.init_zarr, args.time_index)
    st = load_scene(args.tuned_zarr, args.time_index)
    print(f"band {band}  init time {si['time']}  tuned time {st['time']}", flush=True)

    qi = scene_mask(si)
    qt = scene_mask(st)
    print(f"QA-pass: init {int(qi.sum()):,}  tuned {int(qt.sum()):,}", flush=True)

    # IR background from source zarr at the scene time
    src = xr.open_zarr(args.source_zarr)
    rad = src["Rad"].sel(time=si["time"]).values.astype(np.float32)
    vmin_r, vmax_r = np.nanpercentile(rad[np.isfinite(rad)], [1, 99])
    ir = 1.0 - np.clip((rad - vmin_r) / (vmax_r - vmin_r), 0, 1)  # inverted: cold=bright

    # Geostationary projection + image extent (meters)
    H = SAT.satellite_height_m
    ext = [SAT.x_offset * H,
           (SAT.x_offset + SAT.scale_x * (SAT.n_cols - 1)) * H,
           (SAT.y_offset + SAT.scale_y * (SAT.n_rows - 1)) * H,
           SAT.y_offset * H]
    geo = ccrs.Geostationary(central_longitude=SAT.sub_lon_deg,
                             satellite_height=H, sweep_axis=SAT.sweep)
    pc = ccrs.PlateCarree()

    # Subsample barb grid
    s = args.stride
    rs = np.arange(0, SAT.n_rows, s)
    cs = np.arange(0, SAT.n_cols, s)
    cg, rg = np.meshgrid(cs, rs)
    x_sub = (cg * SAT.scale_x + SAT.x_offset) * H
    y_sub = (rg * SAT.scale_y + SAT.y_offset) * H

    fig, axes = plt.subplots(1, 2, figsize=(26, 13), subplot_kw=dict(projection=geo))
    for ax, scene, q, label in [(axes[0], si, qi, args.left_label),
                                 (axes[1], st, qt, args.right_label)]:
        ax.imshow(ir, origin="upper", extent=ext, cmap="gray", vmin=0, vmax=1, alpha=0.9, zorder=0)
        ax.coastlines(resolution="50m", color="cyan", linewidth=0.7)
        ax.set_extent(args.extent, crs=pc)

        u_sub = np.where(q, scene["u"], np.nan)[rg, cg]
        v_sub = np.where(q, scene["v"], np.nan)[rg, cg]
        h_sub = np.where(q, scene["h"], np.nan)[rg, cg]
        spd = np.hypot(np.nan_to_num(u_sub), np.nan_to_num(v_sub))
        good = np.isfinite(u_sub) & np.isfinite(h_sub) & (spd > 0.5)
        plot_barbs_binned(ax, x_sub, y_sub, u_sub * MS_TO_KT, v_sub * MS_TO_KT,
                          h_sub, good, transform=geo)

        hmed = np.nanmedian(np.where(q, scene["h"], np.nan)) / 1000
        ax.set_title(f"{label}\n{str(scene['time'])[:16]} UTC, {band}   "
                     f"n_QA={int(q.sum()):,}, median h={hmed:.1f} km",
                     fontsize=13, fontweight="bold")
        ax.text(0.5, -0.04, f"{int(good.sum())} barbs (stride {s})",
                transform=ax.transAxes, ha="center", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=CMAP_H, norm=NORM_H); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, label="feature-tracked height")
    cbar.set_ticks(np.arange(0, 17000, 2000))
    cbar.set_ticklabels([f"{int(t/1000)} km" for t in np.arange(0, 17000, 2000)])
    fig.suptitle("Stereo winds over GOES-16/18 overlap — pretrained vs tuned "
                 "(barbs colored by feature-tracked height)", fontsize=15, fontweight="bold", y=0.97)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
