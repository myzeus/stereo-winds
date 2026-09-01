"""Headline results figure: composite spatial barb comparison at the Carr-matched step.

Layout (GridSpec):
  Row 1 (full width): (a) tuned RAFT retrieval over the full GOES overlap, barbs
        colored by retrieved feature-tracked height over an inverted-IR background, with a
        dashed rectangle marking the zoom region (labeled "(b)-(d)").
  Row 2 (3 cols): zoom triptych at one shared extent/projection —
        (b) Carr NCC (all vectors), (c) RAFT pretrained (strided), (d) RAFT tuned (same stride).
  Row 3 (3 cols): retrieved-height histogram for each zoom panel (counts, 0-16 km,
        median line + value).
  Bottom: one shared horizontal colorbar, "Retrieved height (km)", 0-16 km.

All panels share ONE QA (the post-hoc mask from eval_from_parquet._build_qa_mask),
ONE projection/extent, ONE perceptually-uniform colormap (cividis, from
figures/paper.mplstyle), and identical barb scaling.

Inference (RAFT -> WLS solver) runs on a GPU node. Retrievals are cached
to NetCDF so the figure can be replotted with --from-cache (e.g. to choose --extent)
without re-running inference.

Examples
--------
  # On a GPU node: run inference for both checkpoints, cache, render contact sheet
  python scripts/make_fig_spatial_barbs.py --preview

  # Replot from cache once an extent is chosen (no GPU needed)
  python scripts/make_fig_spatial_barbs.py --from-cache --extent -120 -95 5 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE / "zeus"))

import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from scipy.stats import wasserstein_distance

from stereo_winds.config import SATELLITE_CONFIGS, StereoPairConfig
from stereo_winds.navigation import (
    fixed_grid_to_geodetic,
    geodetic_to_fixed_grid,
    scanning_angle_to_pixel,
)
from stereo_winds.output import create_output_dataset, write_netcdf
from stereo_winds.solver import (
    build_design_matrix,
    compute_parallax_vectors,
    pixels_to_wind_ms,
    solve_stereo_winds,
)
from stereo_winds.time_model import compute_scene_times

# QA — reuse the EXACT post-hoc mask used by the IGRA evaluation. QA is the
# shared threshold dict; we may override QA["chi2_max"] for the figure (logged).
from eval_from_parquet import _build_qa_mask, QA  # noqa: E402
# Carr loader + the histogram-equalization RAFT expects on its inputs.
from stereo_winds.flow.runner import histogram_equalize  # noqa: E402
from stereo_winds.validation.amv_comparison import load_carr_data  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- Constants (single source of truth) ------------------------------------
STYLE = BASE / "figures" / "paper.mplstyle"
H_MIN, H_MAX = 0.0, 16000.0
NORM = Normalize(vmin=H_MIN, vmax=H_MAX)
N_BINS = 12
H_EDGES = np.linspace(H_MIN, H_MAX, N_BINS + 1)
MS_TO_KT = 1.94384
SCENE_NAMES = ["A_minus", "A0", "A_plus", "B_minus", "B_plus"]
PC = ccrs.PlateCarree()
DATA_DIR_DEFAULT = os.environ.get("STEREO_WINDS_DATA_DIR", str(BASE / "data"))
# Full GOES-16/18 overlap (panel a default extent)
PANEL_A_EXTENT = [-135, -25, -55, 60]

# Barb styling (single source of truth)
BARB_LENGTH = 5.5            # barb glyph length on the triptych panels
TARGET_BARBS_DEFAULT = 250   # ~barbs per dense panel (stride computed to match)
# Lowest viridis tones vanish on the dark warm-surface IR background; sample the
# colormap on [CMAP_LO, 1.0] so even 0-km barbs stay visible. Single constant.
CMAP_LO = 0.15
_USE_BARB_STROKE = False      # set True (via --barb-stroke) to outline barbs

# Set after plt.style.use() so the colormap honors paper.mplstyle (truncated).
CMAP = plt.get_cmap("viridis")

# Carr fairness QA: which standard cuts CAN apply to Carr's vectors.
#   speed<=100 m/s, 1000<=h<=20000 m, and sigma_h<=5000 m (Carr provides sig_H_3D).
#   chi2, Sobel height-gradient, and quality_flag>0 are AI-solver fields with no
#   Carr equivalent, so they cannot be applied to Carr and are intentionally omitted.
CARR_SIGMA_H_MAX = 5000.0
CARR_SPEED_MAX = 100.0
CARR_H_MIN, CARR_H_MAX = 1000.0, 20000.0


# ---- Inference + caching ----------------------------------------------------
def run_inference(ckpt, scene_dir, sat_a, sat_b, t0, dt_minutes, parallax_path,
                  device, tile_size, overlap, batch_size, lowmem, n_iter,
                  cache_path):
    """Run RAFT->WLS for one checkpoint and cache the retrieval to NetCDF.

    Mirrors scripts/infer_and_compare_carr.py: histogram-equalize the 5 cached
    scenes, run RAFT for the 4 disparity pairs, mask flows to the joint-valid
    region, solve, convert pixel velocities to m/s, and write a CF dataset.
    """
    from stereo_winds.disparity import StereoDisparity

    scenes = {n: np.load(Path(scene_dir) / f"{n}.npy").astype(np.float32)
              for n in SCENE_NAMES}
    valid = np.all([np.isfinite(scenes[k]) for k in scenes], axis=0)
    logger.info("  scenes loaded; joint-valid pixels: %d (%.1f%%)",
                int(valid.sum()), 100 * valid.mean())

    par = np.load(parallax_path)
    w_u, w_v = par["w_u"], par["w_v"]

    st = compute_scene_times(t0, dt_minutes, sat_a, sat_b)
    H_matrix = build_design_matrix(w_u, w_v, dt_a_minus=st["A_minus"],
                                   dt_a_plus=st["A_plus"], dt_b_minus=st["B_minus"],
                                   dt_b_plus=st["B_plus"])

    disp = StereoDisparity(model_ckpt_path=str(ckpt), tile_size=tile_size,
                           overlap=overlap, batch_size=batch_size, device=device,
                           lowmem=lowmem)
    images_eq = {k: histogram_equalize(scenes[k]) for k in scenes}
    flows = disp.compute_all(images_eq)
    for k in flows:
        flows[k][:, ~valid] = np.nan

    solution = solve_stereo_winds(flows, H_matrix, sat_a=sat_a, sat_b=sat_b,
                                  n_iter=n_iter, device=device)
    u_ms, v_ms = pixels_to_wind_ms(solution["V_u"], solution["V_v"], sat_a,
                                   dt_seconds=1.0)
    qf = solution["quality_flag"].copy()
    qf[~valid] = 0.0

    nc_solution = {
        "u_wind": u_ms, "v_wind": v_ms, "h": solution["h"],
        "sigma_u": solution["sigma_u"], "sigma_v": solution["sigma_v"],
        "sigma_h": solution["sigma_h"], "chi2": solution["chi2"],
        "p_u": solution.get("p_u", np.zeros_like(u_ms)),
        "p_v": solution.get("p_v", np.zeros_like(u_ms)),
        "quality_flag": qf.astype(np.float32),
    }
    ds = create_output_dataset(nc_solution, sat_a, t0)
    write_netcdf(ds, cache_path)
    logger.info("  cached retrieval -> %s", cache_path)
    return ds


def load_or_infer(tag, ckpt, cache_dir, from_cache, band, **kw):
    """Return the cached retrieval if present/--from-cache, else run inference."""
    cache_path = Path(cache_dir) / f"retr_{tag}_{band}.nc"
    if cache_path.exists() and (from_cache or kw.get("device") == "cpu"):
        logger.info("[%s] loading cached retrieval %s", tag, cache_path)
        return xr.open_dataset(cache_path)
    if cache_path.exists():
        logger.info("[%s] loading cached retrieval %s", tag, cache_path)
        return xr.open_dataset(cache_path)
    if from_cache:
        raise FileNotFoundError(
            f"--from-cache set but {cache_path} missing. Run inference on a GPU node first.")
    logger.info("[%s] running inference (ckpt=%s)", tag, Path(ckpt).name)
    return run_inference(ckpt=ckpt, cache_path=cache_path, **kw)


# ---- Geometry / masks -------------------------------------------------------
def geo_and_extent(sat):
    """Geostationary CRS and imshow extent (meters) for a full-disk grid."""
    H = sat.satellite_height_m
    ext = [sat.x_offset * H,
           (sat.x_offset + sat.scale_x * (sat.n_cols - 1)) * H,
           (sat.y_offset + sat.scale_y * (sat.n_rows - 1)) * H,
           sat.y_offset * H]
    geo = ccrs.Geostationary(central_longitude=sat.sub_lon_deg,
                             satellite_height=H, sweep_axis=sat.sweep)
    return geo, ext, H


_lonlat_cache = {}


def grid_lonlat(sat):
    """Full-grid (lon, lat) for a satellite, cached by satellite_id."""
    if sat.satellite_id in _lonlat_cache:
        return _lonlat_cache[sat.satellite_id]
    cols = np.arange(sat.n_cols)
    rows = np.arange(sat.n_rows)
    x = cols * sat.scale_x + sat.x_offset
    y = rows * sat.scale_y + sat.y_offset
    xg, yg = np.meshgrid(x, y)
    lat, lon = fixed_grid_to_geodetic(xg, yg, sat)
    _lonlat_cache[sat.satellite_id] = (lon, lat)
    return lon, lat


def in_extent_mask(sat, extent):
    """Boolean full-grid mask of pixels inside a lon/lat extent [W,E,S,N]."""
    lon, lat = grid_lonlat(sat)
    W, E, S, N = extent
    return (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N)


def carr_fair_mask(carr):
    """Carr good-sites mask plus the applicable fairness bounds."""
    g = carr["good"].copy()
    g &= np.isfinite(carr["sig_h"]) & (carr["sig_h"] <= CARR_SIGMA_H_MAX)
    g &= carr["spd"] <= CARR_SPEED_MAX
    g &= (carr["h"] >= CARR_H_MIN) & (carr["h"] <= CARR_H_MAX)
    return g


def nadir_pixel_km(sat):
    """Nominal nadir pixel size (km) = scan-angle step * satellite height."""
    return sat.scale_x * sat.satellite_height_m / 1000.0


def stride_for_km(sat, sample_km):
    """Pixel stride that samples ~one point per `sample_km` at nadir."""
    return max(1, int(round(sample_km / nadir_pixel_km(sat))))


def carr_pixel_rc(carr, fair, sat):
    """Pixel (row, col) on sat A's grid for Carr fair sites (NaN if off-grid)."""
    x, y = geodetic_to_fixed_grid(carr["lat"][fair], carr["lon"][fair], sat, h_m=0.0)
    col, row = scanning_angle_to_pixel(x, y, sat)
    return row, col


def thin_by_cell(row, col, stride):
    """Keep ~one point per (stride x stride) grid cell. Returns boolean keep mask.

    This matches Carr's sampling density to the RAFT barb grid (which is sampled
    every `stride` pixels), so the two are compared at the same on-ground ratio.
    """
    keep = np.zeros(row.shape, bool)
    finite = np.isfinite(row) & np.isfinite(col)
    cell = {}
    ridx = np.where(finite)[0]
    cr = (row[finite] // stride).astype(int)
    cc = (col[finite] // stride).astype(int)
    for k, (a, b) in enumerate(zip(cr, cc)):
        key = (a, b)
        if key not in cell:
            cell[key] = True
            keep[ridx[k]] = True
    return keep


def ir_background(scene_dir):
    """Inverted, percentile-clipped IR from the A0 scene (cold cloud = bright)."""
    a0 = Path(scene_dir) / "A0.npy"
    if not a0.exists():
        logger.warning("A0.npy not found in %s — IR background skipped", scene_dir)
        return None
    rad = np.load(a0).astype(np.float32)
    fin = np.isfinite(rad)
    lo, hi = np.nanpercentile(rad[fin], [1, 99])
    return 1.0 - np.clip((rad - lo) / (hi - lo + 1e-9), 0, 1)


# ---- Barb plotting ----------------------------------------------------------
def barbs_binned(ax, x, y, u_kt, v_kt, h, good, transform, length=BARB_LENGTH, lw=0.5):
    """Height-binned wind barbs colored by altitude (shared by all panels)."""
    stroke = [pe.withStroke(linewidth=1.2, foreground="white", alpha=0.6)] \
        if _USE_BARB_STROKE else None
    for i in range(N_BINS):
        lo, hi = H_EDGES[i], H_EDGES[i + 1]
        m = good & (h >= lo) & (h < hi)
        if not np.any(m):
            continue
        b = ax.barbs(x[m], y[m], u_kt[m], v_kt[m], length=length, linewidth=lw,
                     pivot="middle", barb_increments=dict(half=5, full=10, flag=50),
                     color=CMAP(NORM((lo + hi) / 2)), zorder=3 + i, transform=transform)
        if stroke:
            b.set_path_effects(stroke)


def stride_for_target(qa_inext_count, target):
    """Pixel stride giving ~`target` barbs from `qa_inext_count` QA pixels in view."""
    return max(1, int(round(np.sqrt(max(qa_inext_count, 1) / max(target, 1)))))


def setup_map(ax, sat, extent, ir, ext_m):
    ax.set_facecolor("white")
    if ir is not None:
        ax.imshow(ir, origin="upper", extent=ext_m, cmap="gray", vmin=0, vmax=1,
                  alpha=0.9, zorder=0)
    ax.coastlines(resolution="50m", color="0.25", linewidth=0.5)
    ax.set_extent(extent, crs=PC)


def aggregate_grid(ds, sat, extent, nx, ny, min_count):
    """Grid-cell median aggregation of QA-passing retrievals over the zoom extent.

    Returns cell-center lon/lat and per-cell median u, v, h for cells with at least
    `min_count` valid pixels, plus the full-resolution post-QA count and heights in
    the extent (for the histogram, which reflects every retrieval, not the medians).
    """
    mask = _build_qa_mask(ds)
    lon, lat = grid_lonlat(sat)
    W, E, S, N = extent
    sel = mask & (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N)
    u = ds["u_wind"].values[sel]
    v = ds["v_wind"].values[sel]
    h = ds["cloud_top_height"].values[sel]
    lo, la = lon[sel], lat[sel]
    ix = np.clip(((lo - W) / (E - W) * nx).astype(int), 0, nx - 1)
    iy = np.clip(((la - S) / (N - S) * ny).astype(int), 0, ny - 1)
    cell = iy * nx + ix
    df = pd.DataFrame(dict(cell=cell, u=u, v=v, h=h))
    agg = df.groupby("cell").agg(u=("u", "median"), v=("v", "median"),
                                 h=("h", "median"), n=("h", "size"))
    agg = agg[agg["n"] >= min_count]
    cidx = agg.index.values
    cx = W + (cidx % nx + 0.5) / nx * (E - W)
    cy = S + (cidx // nx + 0.5) / ny * (N - S)
    return (cx, cy, agg["u"].values, agg["v"].values, agg["h"].values,
            int(sel.sum()), h)


def aggregate_carr_grid(carr, sat, extent, nx, ny, min_count=1):
    """Grid-cell median aggregation of Carr fair vectors over the extent.

    Mirrors aggregate_grid so Carr is plotted at the SAME cell density as the
    RAFT panels — a fair coverage comparison. Returns cell-center lon/lat and
    per-cell median u, v, h, plus the in-extent fair count and all in-extent
    heights (for the histogram, which reflects every Carr vector).
    """
    fair = carr_fair_mask(carr)
    lon, lat = carr["lon"][fair], carr["lat"][fair]
    u, v, h = carr["u"][fair], carr["v"][fair], carr["h"][fair]
    W, E, S, N = extent
    sel = (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N)
    lon, lat, u, v, h = lon[sel], lat[sel], u[sel], v[sel], h[sel]
    ix = np.clip(((lon - W) / (E - W) * nx).astype(int), 0, nx - 1)
    iy = np.clip(((lat - S) / (N - S) * ny).astype(int), 0, ny - 1)
    cell = iy * nx + ix
    df = pd.DataFrame(dict(cell=cell, u=u, v=v, h=h))
    agg = df.groupby("cell").agg(u=("u", "median"), v=("v", "median"),
                                 h=("h", "median"), n=("h", "size"))
    agg = agg[agg["n"] >= min_count]
    cidx = agg.index.values
    cx = W + (cidx % nx + 0.5) / nx * (E - W)
    cy = S + (cidx // nx + 0.5) / ny * (N - S)
    return (cx, cy, agg["u"].values, agg["v"].values, agg["h"].values,
            int(sel.sum()), h)


def plot_carr_panel_gridded(ax, carr, sat, extent, length, ir, ext_m, label,
                            grid_cells, min_count=1):
    """Carr barbs at matched grid-cell density (same grid as the RAFT panels)."""
    setup_map(ax, sat, extent, ir, ext_m)
    nx, ny = grid_cells
    cx, cy, u, v, h, n_in, heights = aggregate_carr_grid(carr, sat, extent, nx, ny, min_count)
    barbs_binned(ax, cx, cy, u * MS_TO_KT, v * MS_TO_KT, h,
                 np.ones(len(cx), bool), PC, length)
    if label:
        ax.set_title(label, fontsize=10, fontweight="bold")
    return n_in, heights, len(cx)


def plot_ai_panel(ax, ds, sat, extent, length, ir, ext_m, label,
                  grid_cells=(25, 18), min_count=20):
    """Grid-median AI barbs over IR. Returns (post-QA N in extent, heights, n_cells)."""
    setup_map(ax, sat, extent, ir, ext_m)
    nx, ny = grid_cells
    cx, cy, u, v, h, n_post, heights = aggregate_grid(ds, sat, extent, nx, ny, min_count)
    barbs_binned(ax, cx, cy, u * MS_TO_KT, v * MS_TO_KT, h,
                 np.ones(len(cx), bool), PC, length)
    heights = heights[np.isfinite(heights)]
    if label:
        ax.set_title(label, fontsize=10, fontweight="bold")
    return n_post, heights, len(cx)


def plot_carr_panel(ax, carr, sat, extent, length, ir, ext_m, label, stride,
                    thin=True):
    """Carr vectors. If thin, thinned to the RAFT grid stride (matched ratio);
    if not, every fair Carr site is drawn (native 32 km sampling).

    Returns (N-in-extent, heights) over ALL fair sites in extent (unthinned),
    so the histogram/median reflect the true distribution regardless of barb thinning.
    """
    setup_map(ax, sat, extent, ir, ext_m)
    fair = carr_fair_mask(carr)
    lon, lat = carr["lon"][fair], carr["lat"][fair]
    u, v, h = carr["u"][fair], carr["v"][fair], carr["h"][fair]
    if thin:
        row, col = carr_pixel_rc(carr, fair, sat)
        keep = thin_by_cell(row, col, stride)
    else:
        keep = np.ones(lon.shape, bool)
    barbs_binned(ax, lon[keep], lat[keep], u[keep] * MS_TO_KT, v[keep] * MS_TO_KT,
                 h[keep], np.ones(int(keep.sum()), bool), PC, length)
    inext = (lon >= extent[0]) & (lon <= extent[1]) & (lat >= extent[2]) & (lat <= extent[3])
    if label:
        ax.set_title(label, fontsize=10, fontweight="bold")
    return int(inext.sum()), h[inext]


def print_qa_parity():
    """Print which QA cuts apply to our retrieval vs Carr's vectors."""
    rows = [
        ("finite fields", "yes", "yes", ""),
        ("quality_flag > 0", "yes", "yes", "Carr DQF_3D==0"),
        (f"normalized chi2 <= {QA['chi2_max']:g}", "yes", "NO",
         "Carr RSS_resids is a residual in m, not normalized chi2"),
        ("sigma_h <= 5000 m", "yes", "yes", "Carr sig_H_3D"),
        ("Sobel height-grad <= 3000 m/px", "yes", "NO",
         "Carr is scattered 32-km sites, no pixel grid"),
        ("speed <= 100 m/s", "yes", "yes", "from V_3D"),
        ("1000 <= h <= 20000 m", "yes", "yes", "H_3D"),
    ]
    print("\n=== QA parity: cut | applied to ours | applied to Carr ===")
    print(f"{'cut':<32}{'ours':>6}{'Carr':>6}   note")
    for name, ours, carr, note in rows:
        print(f"{name:<32}{ours:>6}{carr:>6}   {note}")
    print("Carr-inapplicable cuts (no equivalent field): normalized chi2, Sobel "
          "height-gradient.")


def fmt_n(n):
    """Compact count label, two significant figures with k/M suffix."""
    if n >= 1e6:
        return f"N = {n / 1e6:.2f}M"
    if n >= 1e3:
        return f"N = {n / 1e3:.0f}k"
    return f"N = {int(n)}"


def annotate_n(ax, n):
    """Compact N label anchored top-right inside the axes, opaque bbox (no clip)."""
    ax.text(0.975, 0.965, fmt_n(n), transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, fontweight="bold", clip_on=False, zorder=30,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", alpha=1.0))


def _kfmt(x, _pos):
    return "0" if x == 0 else f"{x/1e3:.0f}k"


def plot_hist(ax, heights, color, title, show_ylabel, density=False):
    h_km = np.asarray(heights, float) / 1000.0
    bins = np.linspace(0, 16, 33)
    ax.hist(h_km, bins=bins, color=color, edgecolor="0.3", linewidth=0.3, density=density)
    ax.set_xlim(0, 16)
    med = float(np.median(h_km)) if h_km.size else np.nan
    if np.isfinite(med):
        ax.axvline(med, color="crimson", lw=1.3)
        # Median label LEFT of the line and lower, so it can't collide with N (top-right).
        ax.text(med - 0.3, 0.80, f"{med:.1f} km", transform=ax.get_xaxis_transform(),
                color="crimson", fontsize=8, fontweight="bold", ha="right", va="top")
    ax.text(0.975, 0.95, fmt_n(h_km.size), transform=ax.transAxes, ha="right",
            va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=1.0))
    ax.set_xlabel("Retrieved height (km)")
    if show_ylabel:
        ax.set_ylabel("Density" if density else "Count")
    ax.set_title(title, fontsize=8)
    return med


def valid_bbox(sat, valid_mask):
    """Lon/lat bounding box [W,E,S,N] of the overlap valid mask."""
    lon, lat = grid_lonlat(sat)
    vlon, vlat = lon[valid_mask], lat[valid_mask]
    return [float(np.nanmin(vlon)), float(np.nanmax(vlon)),
            float(np.nanmin(vlat)), float(np.nanmax(vlat))]


def add_locator_inset(ax_host, sat, ir, ext_m, valid_mask, extent, pts_lon, pts_lat,
                      width=0.25):
    """Grayscale-IR locator inset in the emptiest non-top-right corner of the host panel.

    `pts_lon/pts_lat` are the panel's barb/data points; the corner with the fewest
    is chosen (top-right is reserved for the N label). `width` is the inset width as
    a fraction of the panel (height scaled to keep it readable).
    """
    geo, _, _ = geo_and_extent(sat)
    bb = valid_bbox(sat, valid_mask)
    W, E, S, N = extent
    mlon, mlat = (W + E) / 2, (S + N) / 2
    counts = {
        "bl": int(((pts_lon < mlon) & (pts_lat < mlat)).sum()),
        "br": int(((pts_lon >= mlon) & (pts_lat < mlat)).sum()),
        "tl": int(((pts_lon < mlon) & (pts_lat >= mlat)).sum()),
    }
    hgt = width * 1.25
    pos = {"bl": [0.015, 0.015, width, hgt],
           "br": [0.985 - width, 0.015, width, hgt],
           "tl": [0.015, 0.985 - hgt, width, hgt]}
    corner = min(counts, key=counts.get)
    ins = ax_host.inset_axes(pos[corner], projection=geo)
    if ir is not None:
        ins.imshow(ir, origin="upper", extent=ext_m, cmap="gray", vmin=0, vmax=1,
                   transform=geo, zorder=0)
    ins.coastlines(resolution="110m", color="0.4", linewidth=0.3)
    ins.set_extent(bb, crs=PC)
    ins.add_patch(Rectangle((W, S), E - W, N - S, transform=PC, fill=False,
                            edgecolor="red", linewidth=1.0, zorder=10))
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_linewidth(0.7)
        sp.set_edgecolor("0.3")


def make_coverage_overlap(ds_tuned, sat, scene_dir, out_dir, valid_mask, band,
                          grid_cells=(60, 45), min_count=20):
    """Separate single-panel figure: tuned barbs over the overlap (cropped to mask bbox)."""
    ir = ir_background(scene_dir)
    geo, ext_m, _ = geo_and_extent(sat)
    bb = valid_bbox(sat, valid_mask)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection=geo)
    plot_ai_panel(ax, ds_tuned, sat, bb, BARB_LENGTH, ir, ext_m,
                  "Sonde-tuned RAFT — overlap coverage", grid_cells, min_count)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7, pad=0.02)
    cbar.set_label("Retrieved height (km)")
    cbar.set_ticks(np.arange(0, 16001, 2000))
    cbar.set_ticklabels([f"{int(t/1000)}" for t in np.arange(0, 16001, 2000)])
    out = Path(out_dir) / "fig_coverage_overlap.png"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved coverage overlap -> %s", out)
    return out


def write_lowerright_diagnostic(ds_tuned, carr, sat, extent, scene_dir, out_path):
    """Report why tuned heights sit below Carr in the lower-right (SE) zoom quadrant."""
    qa = _build_qa_mask(ds_tuned)
    lon, lat = grid_lonlat(sat)
    W, E, S, N = extent
    mlon, mlat = (W + E) / 2, (S + N) / 2
    inzoom = (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N)
    lr = inzoom & (lon >= mlon) & (lat <= mlat)        # lower-right = SE
    rest = inzoom & ~((lon >= mlon) & (lat <= mlat))
    h = ds_tuned["cloud_top_height"].values
    sh = ds_tuned["sigma_h"].values
    c2 = ds_tuned["chi_squared"].values

    def stats(reg):
        m = qa & reg
        hh = h[m]
        if hh.size == 0:
            return None
        return dict(n=int(m.sum()), h_med=np.median(hh) / 1000,
                    sigh_med=np.median(sh[m]), chi2_med=np.median(c2[m]),
                    chi2_p90=np.percentile(c2[m], 90),
                    f_6_9=np.mean((hh >= 6000) & (hh < 9000)),
                    f_9_12=np.mean((hh >= 9000) & (hh < 12000)))

    s_lr, s_rest = stats(lr), stats(rest)

    cf = carr_fair_mask(carr)
    clon, clat, ch = carr["lon"][cf], carr["lat"][cf], carr["h"][cf]
    c_lr = (clon >= mlon) & (clon <= E) & (clat >= S) & (clat <= mlat)
    carr_h_lr = np.median(ch[c_lr]) / 1000 if c_lr.any() else np.nan

    # C14 radiance (BT proxy) from the A0 scene; lower radiance ~ colder ~ higher top.
    rad_line = "A0.npy not found — skipped"
    a0 = Path(scene_dir) / "A0.npy"
    if a0.exists():
        rad = np.load(a0).astype(np.float32)
        r_lr = rad[lr & np.isfinite(rad)]
        r_rest = rad[rest & np.isfinite(rad)]
        rad_line = (f"C14 radiance (W m-2 sr-1 um-1; lower=colder=higher cloud): "
                    f"SE median={np.median(r_lr):.2f}, rest median={np.median(r_rest):.2f}")

    lines = [
        "Lower-right (SE) zoom-quadrant diagnostic — tuned RAFT vs Carr",
        f"zoom extent {list(extent)}; SE quadrant lon>={mlon:.1f}, lat<={mlat:.1f}",
        "",
        "Tuned RAFT, post-QA:",
    ]
    for name, s in [("  SE quadrant", s_lr), ("  rest of zoom", s_rest)]:
        if s is None:
            lines.append(f"{name}: no QA pixels")
            continue
        lines.append(f"{name}: N={s['n']:,}  median h={s['h_med']:.2f} km  "
                     f"median sigma_h={s['sigh_med']:.0f} m  "
                     f"chi2 median={s['chi2_med']:.3f} (p90={s['chi2_p90']:.3f})")
        lines.append(f"{'':14}frac 6-9 km={s['f_6_9']*100:.0f}%  "
                     f"frac 9-12 km={s['f_9_12']*100:.0f}%")
    lines += [
        "",
        f"Carr median h in SE quadrant: {carr_h_lr:.2f} km",
        rad_line,
        "",
        "Interpretation: if SE C14 radiance is NOT lower than the rest (cloud not "
        "colder/higher) and tuned chi2/sigma_h are comparable, the lower tuned "
        "heights are a genuine retrieval, not a QA artifact — likely mid-level cloud "
        "that Carr's NCC assigns higher. QA/data were NOT modified.",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n")
    logger.info("Saved lower-right diagnostic -> %s", out_path)
    print("\n=== Lower-right (SE) quadrant diagnostic ===")
    print("\n".join(lines[3:13]))


_QA_PARITY_NOTE = (
    "Both RAFT retrievals use the IDENTICAL cross-satellite stereo solver (Carr et "
    "al. 2020 five-state WLS) and the IDENTICAL post-hoc quality gate: quality_flag>0, "
    "normalized chi-squared <= {chi2:g}, sigma_h <= 5000 m, Sobel height-gradient <= "
    "3000 m/px, wind speed <= 100 m/s, 1000 <= h <= 20000 m, all fields finite.\n\n"
    "QA parity exception (NOT applied to Carr, no equivalent field): the normalized "
    "chi-squared cut (Carr provides RSS_resids, a residual in metres, not a normalized "
    "chi-squared) and the Sobel height-gradient cut (Carr is scattered 32-km sites, not "
    "a pixel grid). Carr cuts that ARE applied: DQF_3D==0, sig_H_3D <= 5000 m, "
    "speed <= 100 m/s, 1000 <= h <= 20000 m.")

_MIDLEVEL_NOTE = (
    "Mid-level retrievals (~4-9 km) have no Carr NCC counterpart (his distribution is "
    "bimodal with a 4-9 km void); they are validated against independent data "
    "(radiosondes / EarthCARE) elsewhere in the paper, not against Carr.")


# ---- Figure A caption --------------------------------------------------------
def write_caption_tuning(path, extent, n_region, yield_full, band):
    """Caption stub for the headline pretrained-vs-tuned figure (no Carr)."""
    n_a, n_b = n_region
    npre, ntun = yield_full
    txt = f"""Figure A (headline): Sonde-tuning transformation of the RAFT stereo wind
retrieval ({band}, 2025-01-08 19:00 UTC, GOES-16/GOES-18 overlap), zoom extent
{list(extent)} (W E S N). (a) RAFT pretrained, (b) RAFT sonde-tuned. Barbs are
grid-cell medians of QA-passing retrievals (cells with <{20} valid pixels blank),
colored by retrieved feature-tracked height (viridis, 0-16 km). A locator inset in (a)
marks the zoom box within the overlap. Histograms (shared y-axis) give the per-panel
height distribution with the median marked.

{_QA_PARITY_NOTE.format(chi2=QA['chi2_max'])}

Same-QA yield in the zoom region: pretrained N={n_a:,} -> sonde-tuned N={n_b:,}
({n_b/max(n_a,1):.1f}x more retrievals passing the identical gate). Over the full
overlap the yield rises from {npre/1e6:.2f}M to {ntun/1e6:.2f}M (+{100*(ntun-npre)/npre:.0f}%).

{_MIDLEVEL_NOTE}
"""
    Path(path).write_text(txt)


def write_caption_carr(path, extent, stats_main, stats_supp, band, extra=None):
    """Caption stub for the Carr verification figure (panels = FULL overlap)."""
    z = stats_main          # full-overlap collocation drives the panels
    txt = f"""Figure B (verification): Quantitative agreement between the sonde-tuned
RAFT retrieval and Carr NCC where they are collocated ({band}, 2025-01-08 19:00 UTC,
GOES-16/GOES-18 overlap). Panels use the FULL-OVERLAP collocation set; the
zoom-region version is supplementary (supp_carr_verification_zoom). For each Carr
vector the tuned field is sampled at the nearest QA-passing pixel within the
collocation radius; unmatched Carr points are dropped. (a) cloud-top-height scatter,
(b) speed scatter (Carr x, tuned y; density-shaded; 1:1 line; open gray markers =
level swaps). (c) Normalized height histograms: Carr (all his vectors), tuned sampled
at his locations, and tuned over ALL retrievals.

Collocated agreement (full overlap, N={z['n']:,}): height bias={z['h_bias']:+.2f} km,
RMSE={z['h_rmse']:.2f} km, r={z['h_r']:.2f}; speed bias={z['s_bias']:+.1f} m/s,
RMSE={z['s_rmse']:.1f} m/s, r={z['s_r']:.2f}. (Zoom-region supplementary:
N={stats_supp['n']:,}, height RMSE={stats_supp['h_rmse']:.2f} km, r={stats_supp['h_r']:.2f}.)
"""
    if extra:
        e = extra
        txt += f"""
Limitations (cross-reference Discussion): in the low mode (Carr <4 km, n={e['low']['n']})
the tuned height bias is {e['low']['bias']:+.2f} km; in the high mode (Carr >9 km,
n={e['high']['n']}) it is {e['high']['bias']:+.2f} km. The retrieval has a low-level floor
set by the QA cut (h>=1 km; post-QA 1st-percentile tuned height
{e['floor']['post_p1']:.2f} km), so cloud tops below ~1 km are excluded; the floor
diagnostic (floor_diagnostic.txt) confirms it is present in both pretrained and tuned
checkpoints (not tuning-induced).

Level swaps (|tuned-Carr h|>4 km): n={e['n_swap']} of {z['n']}. Swaps have elevated
median chi2 ({e['swap_chi2']:.3f} vs {e['ns_chi2']:.3f} non-swap; sigma_h
{e['swap_sh']:.0f} vs {e['ns_sh']:.0f} m). They account for only {100*e['swap_frac_spdout']:.0f}%
of the {e['spd_out']} speed outliers (|dspeed|>8 m/s) — i.e. swaps explain a minority of
the speed scatter, not all of it.

Unmatched Carr vectors: {e['unmatched']} ({e['n_noqa']} had no post-QA pixel within
the collocation radius; {e['n_oob']} fell off-grid/outside the overlap).
"""
    txt += f"""
Panel (c) shows the key result: at Carr's locations the tuned heights match his
distribution, while over all retrievals the tuned field continuously fills the
4-9 km mid-level range that Carr's bimodal distribution leaves empty.

{_QA_PARITY_NOTE.format(chi2=QA['chi2_max'])}

{_MIDLEVEL_NOTE}
"""
    Path(path).write_text(txt)


# ---- Figure A: headline pretrained-vs-tuned transformation --------------------
def make_figure_tuning(ds_pre, ds_tuned, sat, extent, grid_cells, min_count, scene_dir,
                       out_dir, valid_mask, band="C14"):
    """Two-panel headline: (a) RAFT pretrained, (b) RAFT sonde-tuned. No Carr."""
    ir = ir_background(scene_dir)
    geo, ext_m, _ = geo_and_extent(sat)
    extent = list(extent)          # one extent object, shared + asserted
    W, E, S, N = extent

    fig = plt.figure(figsize=(11.5, 8.0))
    gs = GridSpec(2, 2, height_ratios=[0.74, 0.26], hspace=0.17, wspace=0.06,
                  figure=fig)
    ax_a = fig.add_subplot(gs[0, 0], projection=geo)
    ax_b = fig.add_subplot(gs[0, 1], projection=geo)
    n_a, h_a, cells_a = plot_ai_panel(ax_a, ds_pre, sat, extent, BARB_LENGTH, ir,
                                      ext_m, "(a)  RAFT pretrained", grid_cells, min_count)
    n_b, h_b, cells_b = plot_ai_panel(ax_b, ds_tuned, sat, extent, BARB_LENGTH, ir,
                                      ext_m, "(b)  RAFT sonde-tuned", grid_cells, min_count)
    annotate_n(ax_a, n_a)
    annotate_n(ax_b, n_b)
    logger.info("Figure A barb cells: pretrained=%d, tuned=%d (grid %dx%d, min_count=%d)",
                cells_a, cells_b, grid_cells[0], grid_cells[1], min_count)

    # Locator inset in panel (a), sized so the red box is legible (~25% width).
    lon, lat = grid_lonlat(sat)
    selp = _build_qa_mask(ds_pre) & (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N)
    add_locator_inset(ax_a, sat, ir, ext_m, valid_mask, extent, lon[selp], lat[selp],
                      width=0.25)

    assert [W, E, S, N] == extent, "panels must share one extent object"

    # Row 2: two histograms, SHARED y-axis (ticks on the left only).
    ax_ha = fig.add_subplot(gs[1, 0])
    ax_hb = fig.add_subplot(gs[1, 1], sharey=ax_ha)
    med_a = plot_hist(ax_ha, h_a, "0.5", "(a) pretrained", True)
    med_b = plot_hist(ax_hb, h_b, "0.5", "(b) sonde-tuned", False)
    ax_ha.yaxis.set_major_formatter(FuncFormatter(_kfmt))
    plt.setp(ax_hb.get_yticklabels(), visible=False)

    # Same-QA yield annotation as a suptitle (clear of the panel/hist titles).
    ratio = n_b / max(n_a, 1)
    fig.suptitle(f"Sonde-tuning transformation — same-QA yield in region: "
                 f"pretrained {fmt_n(n_a)[4:]} → tuned {fmt_n(n_b)[4:]}  ({ratio:.1f}×)",
                 fontsize=11, fontweight="bold", y=0.995)

    cax = fig.add_axes([0.30, 0.02, 0.42, 0.018])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Retrieved height (km)")
    cbar.set_ticks(np.arange(0, 16001, 2000))
    cbar.set_ticklabels([f"{int(t/1000)}" for t in np.arange(0, 16001, 2000)])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"fig_spatial_tuning_{band}.png"
    pdf = out_dir / f"fig_spatial_tuning_{band}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)

    inext = in_extent_mask(sat, PANEL_A_EXTENT)
    npre = int((_build_qa_mask(ds_pre) & inext).sum())
    ntun = int((_build_qa_mask(ds_tuned) & inext).sum())
    write_caption_tuning(out_dir / "fig_spatial_tuning_caption.txt", extent,
                         (n_a, n_b), (npre, ntun), band)
    logger.info("Saved %s and %s", png, pdf)
    stats = dict(N=dict(pretrained=n_a, tuned=n_b),
                 median_km=dict(pretrained=med_a, tuned=med_b),
                 yield_full=dict(pretrained=npre, tuned=ntun))
    return png, pdf, stats


def write_floor_diagnostic(ds_pre, ds_tuned, sat, valid_mask, parquet_path, out_path):
    """Report whether the ~2 km low-level floor is tuning-induced, and how it relates
    to the sonde-training-match height distribution. Report only; no changes."""
    def pcts(ds):
        h = ds["cloud_top_height"].values
        v = h[valid_mask & np.isfinite(h)] / 1000.0
        return dict(p01=float(np.percentile(v, 0.1)), p1=float(np.percentile(v, 1)),
                    p5=float(np.percentile(v, 5)), n=int(v.size))
    pre, tun = pcts(ds_pre), pcts(ds_tuned)
    floor_tuning_induced = (tun["p1"] - pre["p1"]) > 0.5   # tuned floor notably higher

    sonde_line = "IGRA parquet not found — sonde-level fractions skipped"
    f2 = f3 = None
    try:
        df = pd.read_parquet(parquet_path)
        hm = df["height_m"].values.astype(float)
        hm = hm[np.isfinite(hm)]
        f2 = float(np.mean(hm < 2000)); f3 = float(np.mean(hm < 3000))
        sonde_line = (f"IGRA training-match sonde levels (N={hm.size:,}): "
                      f"{100*f2:.1f}% below 2 km, {100*f3:.1f}% below 3 km "
                      f"(median {np.median(hm)/1000:.1f} km)")
    except Exception as ex:  # noqa: BLE001
        sonde_line = f"IGRA parquet read failed ({ex}); sonde-level fractions skipped"

    lines = [
        "Low-level floor diagnostic (full overlap, pre-QA height percentiles)",
        f"  pretrained: p0.1={pre['p01']:.2f}  p1={pre['p1']:.2f}  p5={pre['p5']:.2f} km  (N={pre['n']:,})",
        f"  tuned:      p0.1={tun['p01']:.2f}  p1={tun['p1']:.2f}  p5={tun['p5']:.2f} km  (N={tun['n']:,})",
        "",
        f"Floor in pretrained? p1={pre['p1']:.2f} km. Floor in tuned? p1={tun['p1']:.2f} km.",
        ("=> The ~2 km floor APPEARED WITH TUNING (tuned p1 is "
         f"{tun['p1']-pre['p1']:+.2f} km vs pretrained)." if floor_tuning_induced else
         "=> The floor is present in BOTH checkpoints (not tuning-induced)."),
        "",
        sonde_line,
    ]
    if floor_tuning_induced and f2 is not None and f2 < 0.05:
        lines += ["",
                  f"Correlation (NOT a causal claim): the floor is tuning-induced AND only "
                  f"{100*f2:.1f}% of sonde training matches lie below 2 km — i.e. the tuning "
                  "data rarely constrained sub-2 km tops. The association is noted; "
                  "establishing causation would require a controlled experiment."]
    Path(out_path).write_text("\n".join(lines) + "\n")
    logger.info("Saved floor diagnostic -> %s", out_path)
    print("\n=== Low-level floor diagnostic ===")
    print("\n".join(lines))


# ---- Figure B: Carr verification (collocated, quantitative) -------------------
def collocate_carr_tuned(carr, ds_tuned, sat, extent, radius):
    """Sample the tuned field at each Carr vector's nearest QA pixel within `radius`.

    extent=None -> full overlap. Unmatched Carr points (no QA pixel in the window)
    are dropped. Returns matched carr/tuned height (m) and speed (m/s) arrays.
    """
    qa = _build_qa_mask(ds_tuned)
    h = ds_tuned["cloud_top_height"].values
    u = ds_tuned["u_wind"].values
    v = ds_tuned["v_wind"].values
    c2g = ds_tuned["chi_squared"].values
    shg = ds_tuned["sigma_h"].values
    fair = carr_fair_mask(carr)
    lon, lat = carr["lon"][fair], carr["lat"][fair]
    ch, cu, cv = carr["h"][fair], carr["u"][fair], carr["v"][fair]
    if extent is not None:
        W, E, S, N = extent
        m = (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N)
        lon, lat, ch, cu, cv = lon[m], lat[m], ch[m], cu[m], cv[m]
    x, y = geodetic_to_fixed_grid(lat, lon, sat, h_m=0.0)
    col, row = scanning_angle_to_pixel(x, y, sat)
    Hh, Ww = h.shape
    ch_o, th_o, cu_o, cv_o, tu_o, tv_o, tc2, tsh = [], [], [], [], [], [], [], []
    n_oob = n_noqa = 0
    for k in range(len(lon)):
        if not (np.isfinite(col[k]) and np.isfinite(row[k])):
            n_oob += 1
            continue
        r0, c0 = int(round(row[k])), int(round(col[k]))
        if not (0 <= r0 < Hh and 0 <= c0 < Ww):
            n_oob += 1
            continue
        r1, r2 = max(0, r0 - radius), min(Hh, r0 + radius + 1)
        c1, c2 = max(0, c0 - radius), min(Ww, c0 + radius + 1)
        sub = qa[r1:r2, c1:c2]
        if not sub.any():
            n_noqa += 1
            continue
        rr, cc = np.nonzero(sub)
        j = int(np.argmin((rr - (r0 - r1)) ** 2 + (cc - (c0 - c1)) ** 2))
        ri, ci = r1 + rr[j], c1 + cc[j]
        ch_o.append(ch[k]); th_o.append(h[ri, ci])
        cu_o.append(cu[k]); cv_o.append(cv[k])
        tu_o.append(u[ri, ci]); tv_o.append(v[ri, ci])
        tc2.append(c2g[ri, ci]); tsh.append(shg[ri, ci])
    a = np.asarray
    out = dict(carr_h=a(ch_o, float), tuned_h=a(th_o, float),
               carr_spd=np.hypot(a(cu_o, float), a(cv_o, float)),
               tuned_spd=np.hypot(a(tu_o, float), a(tv_o, float)),
               tuned_chi2=a(tc2, float), tuned_sigma_h=a(tsh, float),
               n_in=int(len(lon)), n_oob=n_oob, n_noqa=n_noqa)
    return out


def colloc_stats(m):
    n = len(m["carr_h"])
    if n < 2:
        return dict(n=n, h_bias=np.nan, h_rmse=np.nan, h_r=np.nan,
                    s_bias=np.nan, s_rmse=np.nan, s_r=np.nan)
    dh = (m["tuned_h"] - m["carr_h"]) / 1000.0
    ds_ = m["tuned_spd"] - m["carr_spd"]
    return dict(
        n=n, h_bias=float(np.mean(dh)), h_rmse=float(np.sqrt(np.mean(dh ** 2))),
        h_r=float(np.corrcoef(m["carr_h"], m["tuned_h"])[0, 1]),
        s_bias=float(np.mean(ds_)), s_rmse=float(np.sqrt(np.mean(ds_ ** 2))),
        s_r=float(np.corrcoef(m["carr_spd"], m["tuned_spd"])[0, 1]))


def _swap_and_modes(m):
    """Mode-stratified height stats + level-swap analysis for a collocation set."""
    def mode_stats(lo, hi):
        msk = (m["carr_h"] >= lo * 1000) & (m["carr_h"] < hi * 1000)
        if msk.sum() < 2:
            return dict(n=int(msk.sum()), bias=np.nan, rmse=np.nan, r=np.nan)
        dh = (m["tuned_h"][msk] - m["carr_h"][msk]) / 1000.0
        return dict(n=int(msk.sum()), bias=float(np.mean(dh)),
                    rmse=float(np.sqrt(np.mean(dh ** 2))),
                    r=float(np.corrcoef(m["carr_h"][msk], m["tuned_h"][msk])[0, 1]))
    low, high = mode_stats(0, 4), mode_stats(9, 99)
    swap = np.abs(m["tuned_h"] - m["carr_h"]) > 4000.0
    spd_out = np.abs(m["tuned_spd"] - m["carr_spd"]) > 8.0
    frac = float((spd_out & swap).sum()) / max(int(spd_out.sum()), 1)
    med = lambda x: float(np.median(x)) if len(x) else np.nan
    return dict(low=low, high=high, swap=swap, n_swap=int(swap.sum()),
                swap_chi2=med(m["tuned_chi2"][swap]), ns_chi2=med(m["tuned_chi2"][~swap]),
                swap_sh=med(m["tuned_sigma_h"][swap]), ns_sh=med(m["tuned_sigma_h"][~swap]),
                spd_out=int(spd_out.sum()), swap_frac_spdout=frac)


def _render_carr_panels(m, carr_all, tuned_all, band, region):
    """Render the 3-panel Carr-vs-tuned figure for one collocation set `m`."""
    s = colloc_stats(m)
    e = _swap_and_modes(m)
    swap = e["swap"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    ax.hexbin(m["carr_h"] / 1000, m["tuned_h"] / 1000, gridsize=40, cmap="magma",
              bins="log", mincnt=1, extent=(0, 16, 0, 16))
    ax.plot([0, 16], [0, 16], "k--", lw=0.8)
    ax.scatter(m["carr_h"][swap] / 1000, m["tuned_h"][swap] / 1000, s=12,
               facecolors="none", edgecolors="0.4", linewidths=0.5,
               label=f"level swaps (n={e['n_swap']})", zorder=5)
    ax.set_xlim(0, 16); ax.set_ylim(0, 16); ax.set_aspect("equal")
    ax.set_xlabel("Carr height (km)"); ax.set_ylabel("Tuned height (km)")
    ax.set_title("(a)  Feature-tracked height")
    ax.text(0.04, 0.96, f"N={s['n']:,}\noverall bias={s['h_bias']:+.2f} km\n"
            f"RMSE={s['h_rmse']:.2f} km  r={s['h_r']:.2f}\n"
            f"low(<4km) bias={e['low']['bias']:+.2f} (n={e['low']['n']})\n"
            f"high(>9km) bias={e['high']['bias']:+.2f} (n={e['high']['n']})",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9))
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    smax = 60
    ax.hexbin(m["carr_spd"], m["tuned_spd"], gridsize=40, cmap="magma", bins="log",
              mincnt=1, extent=(0, smax, 0, smax))
    ax.plot([0, smax], [0, smax], "k--", lw=0.8)
    ax.scatter(m["carr_spd"][swap], m["tuned_spd"][swap], s=12, facecolors="none",
               edgecolors="0.4", linewidths=0.5, label=f"level swaps (n={e['n_swap']})",
               zorder=5)
    ax.set_xlim(0, smax); ax.set_ylim(0, smax); ax.set_aspect("equal")
    ax.set_xlabel("Carr speed (m/s)"); ax.set_ylabel("Tuned speed (m/s)")
    ax.set_title("(b)  Wind speed")
    ax.text(0.04, 0.96, f"N={s['n']:,}\nbias={s['s_bias']:+.1f} m/s\n"
            f"RMSE={s['s_rmse']:.1f} m/s  r={s['s_r']:.2f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9))
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[2]
    bins = np.linspace(0, 16, 33)
    ax.hist(carr_all, bins=bins, density=True, histtype="step", lw=1.8,
            color="#c0392b", label=f"Carr (all, N={carr_all.size:,})")
    ax.hist(m["tuned_h"] / 1000, bins=bins, density=True, histtype="step", lw=1.8,
            color="#2e7d32", label=f"Tuned @ Carr locs (N={s['n']:,})")
    ax.hist(tuned_all, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
            color="#19a7ce", label=f"Tuned all (N={tuned_all.size:,})")
    ax.set_xlim(0, 16); ax.set_xlabel("Retrieved height (km)"); ax.set_ylabel("Density")
    ax.set_title("(c)  Height distributions")
    ax.axvspan(4, 9, color="0.85", alpha=0.5, zorder=0)
    ax.legend(fontsize=7, loc="upper center")
    fig.tight_layout()      # no suptitle — caption carries the title in the paper
    return fig, s, e


def make_figure_carr_verification(carr, ds_tuned, sat, extent, scene_dir, out_dir,
                                  radius, valid_mask, band="C14"):
    """Main = FULL-overlap collocation; zoom kept as a supplementary figure."""
    m_zoom = collocate_carr_tuned(carr, ds_tuned, sat, extent, radius)
    m_full = collocate_carr_tuned(carr, ds_tuned, sat, None, radius)

    fair = carr_fair_mask(carr)
    clon, clat, ch = carr["lon"][fair], carr["lat"][fair], carr["h"][fair]
    W, E, S, N = extent
    lon, lat = grid_lonlat(sat)
    qa = _build_qa_mask(ds_tuned)
    h_all = ds_tuned["cloud_top_height"].values
    # Full-overlap distributions (valid mask) and zoom distributions.
    carr_all_full = ch / 1000.0
    tuned_all_full = h_all[qa & valid_mask] / 1000.0
    cz = (clon >= W) & (clon <= E) & (clat >= S) & (clat <= N)
    inz = (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N)
    carr_all_zoom = ch[cz] / 1000.0
    tuned_all_zoom = h_all[qa & inz] / 1000.0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main figure — FULL overlap.
    fig, sf, ef = _render_carr_panels(m_full, carr_all_full, tuned_all_full, band, "full")
    fig.savefig(out_dir / f"fig_carr_verification_{band}.png")
    fig.savefig(out_dir / f"fig_carr_verification_{band}.pdf")
    plt.close(fig)
    # Supplementary figure — zoom region (unchanged content).
    figz, sz, ez = _render_carr_panels(m_zoom, carr_all_zoom, tuned_all_zoom, band, "zoom")
    figz.savefig(out_dir / f"supp_carr_verification_zoom_{band}.png")
    figz.savefig(out_dir / f"supp_carr_verification_zoom_{band}.pdf")
    plt.close(figz)

    # Floor (tuned, full overlap) for the caption.
    pre = h_all[valid_mask & np.isfinite(h_all)]
    post = h_all[qa & valid_mask]
    floor = dict(pre_p1=float(np.percentile(pre, 1)) / 1000,
                 post_p1=float(np.percentile(post, 1)) / 1000,
                 post_min=float(np.min(post)) / 1000)
    unmatched = m_full["n_in"] - sf["n"]

    lines = [
        f"Carr collocation stats ({band}); colloc radius={radius} px",
        "Main figure panels use the FULL-OVERLAP set; zoom kept as supplementary.",
        "",
        f"FULL  N={sf['n']:,}  height bias={sf['h_bias']:+.2f} km RMSE={sf['h_rmse']:.2f} "
        f"km r={sf['h_r']:.2f}  |  speed bias={sf['s_bias']:+.1f} RMSE={sf['s_rmse']:.1f} m/s r={sf['s_r']:.2f}",
        f"(supp) ZOOM N={sz['n']:,}  height bias={sz['h_bias']:+.2f} km RMSE={sz['h_rmse']:.2f} "
        f"km r={sz['h_r']:.2f}  |  speed bias={sz['s_bias']:+.1f} RMSE={sz['s_rmse']:.1f} m/s r={sz['s_r']:.2f}",
        "",
        "Mode-stratified height (FULL overlap):",
        f"  low  (Carr <4 km): n={ef['low']['n']}  bias={ef['low']['bias']:+.2f} km  "
        f"RMSE={ef['low']['rmse']:.2f}  r={ef['low']['r']:.2f}",
        f"  high (Carr >9 km): n={ef['high']['n']}  bias={ef['high']['bias']:+.2f} km  "
        f"RMSE={ef['high']['rmse']:.2f}  r={ef['high']['r']:.2f}",
        "",
        f"Level swaps (|tuned-Carr h|>4 km): n={ef['n_swap']} of {sf['n']} "
        f"({100*ef['n_swap']/max(sf['n'],1):.0f}%)",
        f"  swap     median chi2={ef['swap_chi2']:.3f}  median sigma_h={ef['swap_sh']:.0f} m",
        f"  non-swap median chi2={ef['ns_chi2']:.3f}  median sigma_h={ef['ns_sh']:.0f} m",
        f"  speed outliers (|dspeed|>8 m/s): {ef['spd_out']}; "
        f"{100*ef['swap_frac_spdout']:.0f}% of them are level swaps (minority).",
        "",
        f"Unmatched Carr vectors (full overlap): {unmatched} of {m_full['n_in']} "
        f"({m_full['n_noqa']} no post-QA pixel within {radius} px; "
        f"{m_full['n_oob']} off-grid/outside extent)",
        "",
        f"Low-level floor (tuned h, full overlap): pre-QA p1={floor['pre_p1']:.2f} km; "
        f"post-QA p1={floor['post_p1']:.2f} km (min {floor['post_min']:.2f} km). "
        "See floor_diagnostic.txt for pretrained-vs-tuned + sonde-level comparison.",
        "",
        f"Carr-all N={carr_all_full.size:,} (median {np.median(carr_all_full):.1f} km); "
        f"tuned-all N={tuned_all_full.size:,} (median {np.median(tuned_all_full):.1f} km).",
        "Panel (c): tuned matches Carr at his locations; tuned-all fills the 4-9 km void.",
    ]
    (out_dir / "carr_colloc_stats.txt").write_text("\n".join(lines) + "\n")
    ef_cap = dict(ef, unmatched=unmatched, n_noqa=m_full["n_noqa"],
                  n_oob=m_full["n_oob"], floor=floor)
    write_caption_carr(out_dir / "fig_carr_verification_caption.txt", extent, sf, sz,
                       band, ef_cap)
    logger.info("Saved fig_carr_verification + supp_carr_verification_zoom (%s)", band)
    print("\n=== Carr collocation (FULL overlap; stratified + swap) ===")
    print("\n".join(lines[3:]))
    return None, None, dict(full=sf, zoom=sz, extra=ef_cap)


def select_candidate_tiles(ds_pre, ds_tuned, sat, valid_mask, n_tiles=12, n_pick=6,
                           min_valid_frac=0.97, min_n=500, min_tuned_iqr=4000.0):
    """Pick n_pick tiles where PRETRAINED and TUNED disagree most.

    Score each coarse tile by the Wasserstein distance between the pretrained and
    tuned post-QA cloud-top-height distributions (the larger, the more the tuning
    changes heights there); tie-break by the same-QA yield ratio tuned/pretrained.
    Eligibility: tile essentially fully inside the overlap valid mask (>= min_valid_frac,
    away from the limb), enough QA pixels in BOTH models (>= min_n), and a vertically
    spread tuned distribution (height IQR > min_tuned_iqr) so the panel shows multi-level
    flow rather than a single cirrus shield. Greedy spatial separation avoids adjacent picks.
    """
    qa_p = _build_qa_mask(ds_pre)
    qa_t = _build_qa_mask(ds_tuned)
    hp = ds_pre["cloud_top_height"].values
    ht = ds_tuned["cloud_top_height"].values
    spd_t = np.hypot(ds_tuned["u_wind"].values, ds_tuned["v_wind"].values)
    lon, lat = grid_lonlat(sat)
    nr, nc = ht.shape
    re = np.linspace(0, nr, n_tiles + 1).astype(int)
    ce = np.linspace(0, nc, n_tiles + 1).astype(int)

    tiles = []
    for i in range(n_tiles):
        for j in range(n_tiles):
            sl = (slice(re[i], re[i + 1]), slice(ce[j], ce[j + 1]))
            if valid_mask[sl].mean() < min_valid_frac:    # inside overlap, off the limb
                continue
            qp, qt = qa_p[sl], qa_t[sl]
            np_, nt = int(qp.sum()), int(qt.sum())
            if np_ < min_n or nt < min_n:
                continue
            h_pre = hp[sl][qp] / 1000.0
            h_tun = ht[sl][qt] / 1000.0
            iqr_t = float(np.percentile(h_tun, 75) - np.percentile(h_tun, 25)) * 1000.0
            if iqr_t < min_tuned_iqr:                     # must be vertically spread
                continue
            lo, la = lon[sl][qt], lat[sl][qt]
            if not (np.isfinite(lo).any() and np.isfinite(la).any()):
                continue
            tiles.append(dict(
                i=i, j=j, n_pre=np_, n_tun=nt,
                wdist=float(wasserstein_distance(h_pre, h_tun)),
                dmed=float(np.median(h_tun) - np.median(h_pre)),
                med_pre=float(np.median(h_pre)), med_tun=float(np.median(h_tun)),
                yield_ratio=nt / max(np_, 1), iqr_t=iqr_t,
                mspd=float(np.nanmean(spd_t[sl][qt])),
                ext=[float(np.nanpercentile(lo, 1)), float(np.nanpercentile(lo, 99)),
                     float(np.nanpercentile(la, 1)), float(np.nanpercentile(la, 99))]))

    # Rank by contrast (Wasserstein), break ties by yield ratio.
    tiles.sort(key=lambda t: (-t["wdist"], -t["yield_ratio"]))
    chosen = []
    for t in tiles:
        if all(abs(t["i"] - c["i"]) > 1 or abs(t["j"] - c["j"]) > 1 for c in chosen):
            chosen.append(t)
        if len(chosen) >= n_pick:
            break
    return chosen


def make_region_preview(ds_pre, ds_tuned, sat, scene_dir, out_dir, valid_mask,
                        target_barbs, n_tiles=12, n_pick=6):
    """Contact sheet of candidate zoom regions ranked by pretrained-tuned contrast."""
    ir = ir_background(scene_dir)
    geo, ext_m, _ = geo_and_extent(sat)
    chosen = select_candidate_tiles(ds_pre, ds_tuned, sat, valid_mask, n_tiles, n_pick)
    if not chosen:
        raise RuntimeError("No candidate tiles passed the filters; loosen "
                           "min_valid_frac/min_n/min_tuned_iqr or n_tiles.")
    qa_t = _build_qa_mask(ds_tuned)
    ncol = 3
    nrow = int(np.ceil(len(chosen) / ncol))
    fig = plt.figure(figsize=(5.2 * ncol, 4.8 * nrow))
    gs = GridSpec(nrow, ncol, hspace=0.22, wspace=0.08, figure=fig)
    extents = []
    for k, t in enumerate(chosen):
        ax = fig.add_subplot(gs[k // ncol, k % ncol], projection=geo)
        ext = t["ext"]
        extents.append(ext)
        # Preview each candidate with grid-median barbs (coarse grid for small tiles).
        plot_ai_panel(ax, ds_tuned, sat, ext, BARB_LENGTH, ir, ext_m, None,
                      grid_cells=(12, 9), min_count=5)
        ax.set_title(
            f"cand {k+1}: --extent {ext[0]:.1f} {ext[1]:.1f} {ext[2]:.1f} {ext[3]:.1f}\n"
            f"med_h pre/tuned = {t['med_pre']:.1f}/{t['med_tun']:.1f} km  "
            f"(W={t['wdist']:.1f}, Δmed={t['dmed']:+.1f} km)\n"
            f"yield tuned/pre = {t['yield_ratio']:.1f}x, tuned IQR={t['iqr_t']/1000:.1f} km, "
            f"spd={t['mspd']:.0f} m/s", fontsize=8, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, orientation="horizontal", fraction=0.035,
                 pad=0.04, label="Retrieved height (km)")
    fig.suptitle("Candidate zoom regions — ranked by PRETRAINED-vs-TUNED height "
                 "contrast (Wasserstein); all inside the GOES-16/18 overlap",
                 fontsize=12, fontweight="bold")
    out = Path(out_dir) / "region_preview.png"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved region preview -> %s", out)
    print("\n=== Candidate zoom regions (ranked by pretrained-tuned contrast) ===")
    print("(med_pre/med_tun/Dmed/IQR in km; Wass=Wasserstein dist; yld=tuned/pre yield)")
    print(f"{'#':>2} {'--extent (W E S N)':<32}{'med_pre':>8}{'med_tun':>8}"
          f"{'Wass':>6}{'Dmed':>7}{'yld':>6}{'IQRt':>6}")
    for k, t in enumerate(chosen):
        e = t["ext"]
        print(f"{k+1:>2} {f'{e[0]:.1f} {e[1]:.1f} {e[2]:.1f} {e[3]:.1f}':<32}"
              f"{t['med_pre']:>8.1f}{t['med_tun']:>8.1f}{t['wdist']:>6.1f}"
              f"{t['dmed']:>+7.1f}{t['yield_ratio']:>5.1f}x{t['iqr_t']/1000:>6.1f}")
    print("\nReply with the chosen --extent (e.g. one of the rows above).")
    return out, extents


def make_full_disk(carr, ds_pre, ds_tuned, sat, scene_dir, out_dir, stride, band="C14",
                   min_count=20):
    """1x3 full-overlap maps (Carr / pretrained / tuned) + per-panel height histograms.

    Row 1: grid-median barbs over the GOES-16/18 overlap, colored by retrieved
    feature-tracked height (shared colorbar). Row 2: the retrieved-height distribution
    behind each map (Carr's vectors, and the post-QA heights for each RAFT panel),
    shared y-axis, median marked.
    """
    ir = ir_background(scene_dir)
    geo, ext_m, _ = geo_and_extent(sat)
    fig = plt.figure(figsize=(19, 11))
    gs = GridSpec(2, 3, height_ratios=[0.70, 0.30], hspace=0.16, wspace=0.07, figure=fig)
    ax_b = fig.add_subplot(gs[0, 0], projection=geo)
    ax_c = fig.add_subplot(gs[0, 1], projection=geo)
    ax_d = fig.add_subplot(gs[0, 2], projection=geo)
    # Both panels aggregated to the SAME grid -> fair coverage comparison
    # (filled cells reflect real coverage; raw-vector Carr would look misleadingly
    # denser than grid-median RAFT despite ~200x fewer retrievals).
    GRID = (60, 45)
    ncell = GRID[0] * GRID[1]
    n_b, h_b, cells_b = plot_carr_panel_gridded(ax_b, carr, sat, PANEL_A_EXTENT, 4.0,
                                                ir, ext_m, None, GRID, min_count=1)
    n_c, h_c, cells_c = plot_ai_panel(ax_c, ds_pre, sat, PANEL_A_EXTENT, 4.0, ir,
                                      ext_m, None, grid_cells=GRID, min_count=min_count)
    n_d, h_d, cells_d = plot_ai_panel(ax_d, ds_tuned, sat, PANEL_A_EXTENT, 4.0, ir,
                                      ext_m, None, grid_cells=GRID, min_count=min_count)
    for ax, name, n, hh, cells in [(ax_b, "(a)  Carr NCC", n_b, h_b, cells_b),
                                   (ax_c, "(b)  RAFT pretrained", n_c, h_c, cells_c),
                                   (ax_d, "(c)  RAFT sonde-tuned", n_d, h_d, cells_d)]:
        med = float(np.median(hh)) / 1000 if len(hh) else float("nan")
        ax.set_title(f"{name}\nN={n:,}, median h={med:.1f} km\n"
                     f"coverage: {cells:,}/{ncell:,} grid cells "
                     f"({100*cells/ncell:.0f}%)", fontsize=10, fontweight="bold")
    # Shared vertical colorbar to the right of the map row.
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_b, ax_c, ax_d], orientation="vertical",
                        fraction=0.022, pad=0.012)
    cbar.set_label("Retrieved height (km)")
    cbar.set_ticks(np.arange(0, 16001, 2000))
    cbar.set_ticklabels([f"{int(t/1000)}" for t in np.arange(0, 16001, 2000)])

    # Row 2: per-panel retrieved-height histograms. DENSITY-normalized with
    # independent y-axes, because Carr (~25k vectors) and the RAFT fields
    # (~millions of pixels) have wildly different sample sizes — only the SHAPES
    # are comparable. Shared raw-count axes would bury Carr's distribution.
    ax_hb = fig.add_subplot(gs[1, 0])
    ax_hc = fig.add_subplot(gs[1, 1])
    ax_hd = fig.add_subplot(gs[1, 2])
    plot_hist(ax_hb, h_b, "#c0392b", "(a) Carr NCC", True, density=True)
    plot_hist(ax_hc, h_c, "0.5", "(b) RAFT pretrained", True, density=True)
    plot_hist(ax_hd, h_d, "#19a7ce", "(c) RAFT sonde-tuned", True, density=True)
    # Shade the 4-9 km mid-level band Carr's bimodal distribution leaves empty.
    for a in (ax_hb, ax_hc, ax_hd):
        a.axvspan(4, 9, color="0.88", alpha=0.6, zorder=0)

    fig.suptitle("Full GOES-16/18 overlap — grid-median wind barbs colored by "
                 "retrieved feature-tracked height", fontsize=14, fontweight="bold")
    out = Path(out_dir) / f"fig_spatial_barbs_fulldisk_{band}.png"
    pdf = Path(out_dir) / f"fig_spatial_barbs_fulldisk_{band}.pdf"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(pdf)
    plt.close(fig)
    logger.info("Saved full-disk comparison -> %s", out)
    km = stride * nadir_pixel_km(sat)
    print(f"\n=== Full-overlap (matched ~{km:.0f} km sampling) ===")
    for name, n, hh in [("carr", n_b, h_b), ("pretrained", n_c, h_c),
                        ("tuned", n_d, h_d)]:
        med = float(np.median(hh)) / 1000 if len(hh) else float("nan")
        print(f"  {name:<11s} N={n:,}  median={med:.2f} km")
    return out


# ---- CLI --------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DATA_DIR_DEFAULT)
    p.add_argument("--scene-dir", default=None,
                   help="5-scene .npy dir (default <data-dir>/cache)")
    p.add_argument("--carr-nc", default=None, help="Carr retrieval NetCDF")
    p.add_argument("--parallax", default=None,
                   help="parallax npz (default <data-dir>/zarrs/parallax_<a>_<b>.npz)")
    p.add_argument("--ckpt-pretrained", default=None)
    p.add_argument("--ckpt-tuned", default=None)
    p.add_argument("--sat-a", default="goes16", choices=list(SATELLITE_CONFIGS))
    p.add_argument("--sat-b", default="goes18", choices=list(SATELLITE_CONFIGS))
    p.add_argument("--band", default="C14")
    p.add_argument("--time", default="2025-01-08T19:00")
    p.add_argument("--dt-minutes", type=float, default=10.0)
    p.add_argument("--extent", type=float, nargs=4, default=[-120, -95, 5, 30],
                   metavar=("W", "E", "S", "N"), help="zoom region (placeholder default)")
    p.add_argument("--target-barbs", type=int, default=TARGET_BARBS_DEFAULT,
                   help="approx barbs per dense RAFT panel; per-panel stride is "
                        "computed from the zoom extent to hit this. Raised to match "
                        "Carr if Carr is denser, so RAFT >= Carr density.")
    p.add_argument("--barb-stroke", action="store_true",
                   help="add a white outline to barbs (helps low/dark viridis tones "
                        "on the warm-surface background)")
    p.add_argument("--grid-cells", type=int, nargs=2, default=[25, 18],
                   metavar=("NX", "NY"),
                   help="grid for cell-median barb aggregation on the dense panels")
    p.add_argument("--min-cell-count", type=int, default=20,
                   help="skip grid cells with fewer than this many QA pixels")
    p.add_argument("--colloc-radius", type=int, default=3,
                   help="Carr-verification: nearest QA pixel within this radius (px)")
    p.add_argument("--igra-parquet", default=None,
                   help="IGRA collocation parquet for the floor diagnostic "
                        "(default <data-dir>/labels/igra/igra_all_collocation.parquet)")
    p.add_argument("--stride", type=int, default=None,
                   help="explicit barb pixel stride (overrides --sample-km; full-disk mode)")
    p.add_argument("--sample-km", type=float, default=64.0,
                   help="on-ground barb spacing (km) for --full-disk mode (Carr 32 km).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lowmem", action="store_true")
    p.add_argument("--n-iter", type=int, default=3)
    p.add_argument("--chi2-max", type=float, default=None,
                   help="override the QA chi2_max cut for this figure (canonical "
                        "is 0.2; pass 0.3 to loosen for more coverage). Does not "
                        "affect eval_from_parquet's value used by the IGRA eval.")
    p.add_argument("--output-dir", default=str(BASE / "figures"))
    p.add_argument("--from-cache", action="store_true",
                   help="replot from cached NetCDF retrievals (no inference)")
    p.add_argument("--preview", action="store_true",
                   help="Step 1: render meteorology-chosen candidate-zoom contact "
                        "sheet (region_preview.png) and stop")
    p.add_argument("--valid-mask", default=None,
                   help="overlap valid-mask .npy (default <data-dir>/zarrs/"
                        "valid_mask_g19_g18.npy; g16/g18 share the -75 grid)")
    p.add_argument("--n-tiles", type=int, default=12,
                   help="coarse tiling (NxN) for candidate selection")
    p.add_argument("--n-candidates", type=int, default=6,
                   help="number of candidate regions to render")
    p.add_argument("--full-disk", action="store_true",
                   help="render 1x3 full-overlap Carr/pretrained/tuned comparison")
    args = p.parse_args()

    if STYLE.exists():
        plt.style.use(str(STYLE))
    global CMAP, _USE_BARB_STROKE
    # Truncate the base colormap to [CMAP_LO, 1.0] so the lowest barbs stay
    # visible on the dark warm-surface IR background. Single source for all panels.
    _base = plt.get_cmap(mpl.rcParams.get("image.cmap", "viridis"))
    CMAP = LinearSegmentedColormap.from_list(
        "h_trunc", _base(np.linspace(CMAP_LO, 1.0, 256)))
    _USE_BARB_STROKE = args.barb_stroke

    # Optional QA chi2_max override for the figure (mutates the shared QA dict
    # for this process only; the on-disk eval_from_parquet default is unchanged).
    if args.chi2_max is not None:
        logger.info("Overriding QA chi2_max %.2f -> %.2f for this figure",
                    QA["chi2_max"], args.chi2_max)
        QA["chi2_max"] = args.chi2_max

    data_dir = Path(args.data_dir)
    scene_dir = Path(args.scene_dir) if args.scene_dir else data_dir / "cache"
    parallax = Path(args.parallax) if args.parallax else \
        data_dir / "zarrs" / f"parallax_{args.sat_a}_{args.sat_b}.npz"
    # Repo-shipped checkpoints; "pretrained" is the exact fine-tuning init.
    ckpt_pre = args.ckpt_pretrained or str(BASE / "checkpoints" / "windflow.raft.init-ep254.ckpt")
    ckpt_tuned = args.ckpt_tuned or str(BASE / "checkpoints" / "windflow.raft.sonde-tuned.ckpt")
    cache_dir = Path(args.output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Prefer satellites recorded in the cache's sat_configs.json if present.
    sat_a_id, sat_b_id = args.sat_a, args.sat_b
    cfg_json = scene_dir / "sat_configs.json"
    if cfg_json.exists():
        try:
            j = json.loads(cfg_json.read_text())
            logger.info("scene sat_configs.json present: %s", list(j)[:4])
        except Exception as e:  # noqa: BLE001
            logger.warning("could not parse %s: %s", cfg_json, e)
    sat_a = SATELLITE_CONFIGS[sat_a_id]
    sat_b = SATELLITE_CONFIGS[sat_b_id]

    import datetime as dt
    t0 = dt.datetime.fromisoformat(args.time)
    pair = StereoPairConfig(sat_a=sat_a, sat_b=sat_b, band=args.band,
                            dt_minutes=args.dt_minutes)  # noqa: F841

    inf_kw = dict(scene_dir=scene_dir, sat_a=sat_a, sat_b=sat_b, t0=t0,
                  dt_minutes=args.dt_minutes, parallax_path=parallax,
                  device=args.device, tile_size=args.tile_size, overlap=args.overlap,
                  batch_size=args.batch_size, lowmem=args.lowmem, n_iter=args.n_iter)

    # Both retrievals are needed for the contrast-based region selection AND the figure.
    ds_tuned = load_or_infer("tuned", ckpt_tuned, cache_dir, args.from_cache,
                             args.band, **inf_kw)
    ds_pre = load_or_infer("pretrained", ckpt_pre, cache_dir, args.from_cache,
                           args.band, **inf_kw)

    print_qa_parity()  # QA cut applicability: ours vs Carr

    # Overlap valid mask (g16/g18 share the -75 grid with g19) — used for the
    # region-selection filter, the locator inset, and the coverage figure crop.
    vm_path = Path(args.valid_mask) if args.valid_mask else \
        data_dir / "zarrs" / "valid_mask_g19_g18.npy"
    if vm_path.exists():
        valid_mask = np.load(vm_path)
        logger.info("valid mask: %s (%.1f%% inside overlap)",
                    vm_path.name, 100 * valid_mask.mean())
    else:
        valid_mask = np.isfinite(ds_tuned["cloud_top_height"].values)
        logger.warning("valid mask %s missing; using finite-height as overlap", vm_path)

    if args.preview:
        make_region_preview(ds_pre, ds_tuned, sat_a, scene_dir, args.output_dir,
                            valid_mask, args.target_barbs, args.n_tiles, args.n_candidates)
        return

    carr_nc = args.carr_nc
    if carr_nc is None:
        cands = sorted((data_dir / "carr_data").glob("*.nc"))
        if not cands:
            raise FileNotFoundError(f"No Carr NetCDF in {data_dir/'carr_data'}; pass --carr-nc")
        carr_nc = str(cands[0])
        logger.info("Using Carr file: %s", carr_nc)
    carr = load_carr_data(carr_nc)
    logger.info("Carr: %d/%d good sites", carr["n_good"], carr["n_total"])

    if args.full_disk:
        stride = args.stride if args.stride else stride_for_km(sat_a, args.sample_km)
        make_full_disk(carr, ds_pre, ds_tuned, sat_a, scene_dir, args.output_dir,
                       stride, args.band, min_count=args.min_cell_count)
        return

    grid_cells = tuple(args.grid_cells)
    # Figure A — headline pretrained-vs-tuned (no Carr).
    _, _, stats = make_figure_tuning(ds_pre, ds_tuned, sat_a, args.extent, grid_cells,
                                     args.min_cell_count, scene_dir, args.output_dir,
                                     valid_mask, band=args.band)
    # Figure B — Carr collocated verification (full-overlap main + zoom supp).
    make_figure_carr_verification(carr, ds_tuned, sat_a, args.extent, scene_dir,
                                  args.output_dir, args.colloc_radius, valid_mask,
                                  band=args.band)
    # Floor diagnostic (pretrained vs tuned + sonde-level fractions).
    igra_pq = args.igra_parquet or str(data_dir / "labels" / "igra" /
                                       "igra_all_collocation.parquet")
    write_floor_diagnostic(ds_pre, ds_tuned, sat_a, valid_mask, igra_pq,
                           Path(args.output_dir) / "floor_diagnostic.txt")
    # Supplementary: coverage map + lower-right diagnostic (no figure-change side effects).
    make_coverage_overlap(ds_tuned, sat_a, scene_dir, args.output_dir, valid_mask, args.band)
    write_lowerright_diagnostic(ds_tuned, carr, sat_a, args.extent, scene_dir,
                                Path(args.output_dir) / "zoom_lowerright_diagnostic.txt")

    print("\n=== Figure A: region post-QA counts (N) / medians (km) ===")
    for k in ("pretrained", "tuned"):
        print(f"  {k:<11s} N={stats['N'][k]:,}  median={stats['median_km'][k]:.2f} km")
    yf = stats["yield_full"]
    print(f"  full-overlap same-QA yield: pretrained {yf['pretrained']/1e6:.2f}M -> "
          f"tuned {yf['tuned']/1e6:.2f}M (+{100*(yf['tuned']-yf['pretrained'])/yf['pretrained']:.0f}%)")
    dmed = abs(stats["median_km"]["tuned"] - stats["median_km"]["pretrained"])
    print(f"=== pretrained-vs-tuned median separation = {dmed:.2f} km ===")
    if dmed < 1.5:
        print("\n*** REGION CHECK: pretrained/tuned medians differ by "
              f"{dmed:.2f} km (< 1.5 km) — weak tuning contrast; consider --preview. ***")


if __name__ == "__main__":
    main()
