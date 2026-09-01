#!/usr/bin/env python
"""Methodology figure — system-architecture panel for the stereo-winds paper.

Renders the full data flow of the cross-satellite optical-flow stereo wind
retrieval as a labeled box-and-arrow diagram with *real data thumbnails*
embedded at each stage:

    5 IR scenes (A-, A0, A+, B-, B+)        [inputs]
        |
    RAFT optical flow  ->  D1..D4           [4 disparity fields]
        | + parallax sensitivity w, scan-time offsets dt
    per-pixel 5-state WLS solver            [Carr et al. 2020]
        |
    winds (barbs colored by height) + feature-tracked height   [outputs]

Default scene: GOES-16 (A) / GOES-18 (B), 2024-01-15 19:00 UTC, band C14, all
cached locally under ``cache/`` (5 raw IR ``.nc`` scenes + 4 precomputed RAFT
disparity fields ``flow_stereo_19z_D{1..4}.npy`` + remap LUT), so the figure
builds end-to-end on CPU with no GPU or cluster access.

Examples
--------
    # build from local cache (solves a crop on CPU, writes png/pdf + cache)
    python scripts/fig_methodology_architecture.py

    # re-render from the cached solution without re-solving
    python scripts/fig_methodology_architecture.py --from-cache
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Arc, FancyArrowPatch, FancyBboxPatch, Rectangle)

from stereo_winds.config import GOES16_CONFIG, GOES18_CONFIG  # noqa: E402
from stereo_winds.qa import height_gradient  # noqa: E402
from stereo_winds.remap import load_remap_lut, remap_image  # noqa: E402
from stereo_winds.solver import (  # noqa: E402
    build_design_matrix,
    compute_parallax_vectors,
    solve_stereo_winds,
)
from stereo_winds.time_model import compute_scene_times  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fig_methodology")

REPO = Path(__file__).resolve().parent.parent

# ---- established paper palette (matches fig_parallax_concept.py) -------------
COLOR_A = "#e8743b"        # satellite A (GOES-16)
COLOR_B = "#19a7ce"        # satellite B (GOES-18)
COLOR_PARALLAX = "#c0392b"  # parallax / cross-sat geometry
COLOR_WIND = "#2e7d32"      # wind
COLOR_BOX = "#f4f4f4"       # stage-box fill
COLOR_BOX_EDGE = "#333333"
COLOR_ARROW = "#444444"

# ---- height color scale (single source of truth: viridis 0-16 km) -----------
H_VMIN, H_VMAX = 0.0, 16000.0       # meters
HNORM = Normalize(vmin=H_VMIN, vmax=H_VMAX)
HCMAP = plt.get_cmap("viridis")
N_HBINS = 12
H_EDGES = np.linspace(H_VMIN, H_VMAX, N_HBINS + 1)
MS_TO_KT = 1.94384

# QA cuts (mirror scripts/eval_from_parquet._build_qa_mask)
QA = dict(chi2_max=0.2, sigma_h_max=5000.0, h_grad_max=3000.0,
          wind_speed_max=100.0, min_height=1000.0)

SCENE_FILES = {
    # scene -> (satellite dir, HHMM of scan start)
    "A_minus": ("goes16", "1850"),
    "A0": ("goes16", "1900"),
    "A_plus": ("goes16", "1910"),
    "B_minus": ("goes18", "1850"),
    "B_plus": ("goes18", "1910"),
}
FLOW_FILES = {  # solver disparity key -> cached .npy basename
    "D1": "flow_stereo_19z_D1.npy",
    "D2": "flow_stereo_19z_D2.npy",
    "D3": "flow_stereo_19z_D3.npy",
    "D4": "flow_stereo_19z_D4.npy",
}

# Default crop: NE-Pacific subtropics (lat ~22N, lon ~119W), in the heart of the
# GOES-16/18 overlap. Windy high-cirrus / jet region (median ~26 m/s, ~9 km) — the
# same well-validated region as the headline results figure. The window is biased
# north/east of the scene's low-cloud trades (to the south/west) so the expansion
# keeps the high-cloud character. Override with --crop.
DEFAULT_CROP = (950, 1950, 396, 1396)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _find_scene_nc(cache_dir: Path, sat: str, day: str, hhmm: str) -> Path:
    """Locate a cached ABI L1b RadF .nc by satellite/day/scan-start HHMM."""
    pattern = f"{cache_dir}/{sat}/ABI/2024/{day}/**/*_s2024{day}{hhmm}*.nc"
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(f"no scene matching {pattern}")
    return Path(hits[0])


def load_radiance(nc_path: Path) -> np.ndarray:
    """Read ABI Rad, return (n_rows, n_cols) float32 with row 0 = north.

    The raw ABI L1b RadF file already stores Rad north->south (y decreasing,
    row 0 = north), matching the cached flows / parallax / remap-LUT grid, so
    no flip is applied. (`load_goes_scene` flips only because satpy returns
    south->north; we read the netCDF directly here.)
    """
    with xr.open_dataset(nc_path) as ds:
        return ds["Rad"].values.astype(np.float32)


def fill_disk_holes(rad: np.ndarray, on_disk: np.ndarray,
                    fallback: np.ndarray | None = None) -> np.ndarray:
    """Fill NaN dropouts *inside* the disk so they don't render as holes.

    Some ABI scenes (e.g. the off-center A± times) have missing scan-line
    pixels and even large contiguous dropout blocks. We fill holes from
    ``fallback`` (the complete A0 scene — clouds barely move in ±10 min, so
    this is seamless), falling back to nearest-finite for anything still NaN.
    Off-disk pixels (``~on_disk``) are left NaN so space stays transparent.
    Display only — the retrieval comes from the (gap-free) cached flows.
    """
    holes = on_disk & ~np.isfinite(rad)
    if not holes.any():
        return rad
    out = rad.copy()
    if fallback is not None:
        from_fb = holes & np.isfinite(fallback)
        out[from_fb] = fallback[from_fb]
    still = on_disk & ~np.isfinite(out)
    if still.any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(~np.isfinite(out), return_distances=False,
                                     return_indices=True)
        out[still] = out[tuple(idx)][still]
    return out


def load_flow(cache_dir: Path, key: str) -> np.ndarray:
    """Load a cached RAFT disparity field, squeezed to (2, H, W)."""
    arr = np.load(cache_dir / FLOW_FILES[key])
    return np.squeeze(arr).astype(np.float64)  # (1,2,H,W) -> (2,H,W)


def get_parallax(sat_a, sat_b, cache_dir: Path):
    """Per-pixel parallax sensitivity (w_u, w_v); cached to npz."""
    path = cache_dir / f"parallax_{sat_a.satellite_id}_{sat_b.satellite_id}.npz"
    if path.exists():
        logger.info("Loading cached parallax %s", path.name)
        d = np.load(path)
        return d["w_u"], d["w_v"]
    logger.info("Computing parallax sensitivity vectors (full disk)...")
    w_u, w_v = compute_parallax_vectors(sat_a, sat_b)
    np.savez_compressed(path, w_u=w_u, w_v=w_v)
    return w_u, w_v


# ---------------------------------------------------------------------------
# Crop selection
# ---------------------------------------------------------------------------
def auto_crop(overlap: np.ndarray, size: int) -> tuple[int, int, int, int]:
    """Center a ``size`` x ``size`` window on the overlap-mask centroid."""
    rr, cc = np.where(overlap)
    rc, ccc = int(rr.mean()), int(cc.mean())
    n = overlap.shape[0]
    r0 = int(np.clip(rc - size // 2, 0, n - size))
    c0 = int(np.clip(ccc - size // 2, 0, n - size))
    return r0, r0 + size, c0, c0 + size


# ---------------------------------------------------------------------------
# Solve (single-pass WLS on the crop)
# ---------------------------------------------------------------------------
def solve_crop(sat_a, sat_b, t0, dt_minutes, crop, cache_dir, flows_full=None,
               w_full=None):
    """Single-pass WLS solve over the crop. Returns a solution dict (cropped)."""
    r0, r1, c0, c1 = crop
    if w_full is None:
        w_u, w_v = get_parallax(sat_a, sat_b, cache_dir)
    else:
        w_u, w_v = w_full
    w_u_c, w_v_c = w_u[r0:r1, c0:c1], w_v[r0:r1, c0:c1]

    st = compute_scene_times(t0, dt_minutes, sat_a, sat_b)
    H_mat = build_design_matrix(
        w_u_c, w_v_c,
        dt_a_minus=st["A_minus"], dt_a_plus=st["A_plus"],
        dt_b_minus=st["B_minus"], dt_b_plus=st["B_plus"],
    )

    flows = {}
    for key in ("D1", "D2", "D3", "D4"):
        f = flows_full[key] if flows_full else load_flow(cache_dir, key)
        flows[key] = f[:, r0:r1, c0:c1]

    # Single-pass: sat_a/sat_b omitted so the solver does not try to recompute
    # full-disk parallax mid-iteration (which would shape-mismatch the crop).
    sol = solve_stereo_winds(flows, H_mat, n_iter=1, device="cpu")
    # compute_pixel_scale returns full-grid arrays; crop to the window.
    from stereo_winds.navigation import compute_pixel_scale
    dx_m, dy_m = compute_pixel_scale(sat_a)
    dx_c, dy_c = dx_m[r0:r1, c0:c1], dy_m[r0:r1, c0:c1]
    sol["u_wind"] = sol["V_u"] * dx_c
    sol["v_wind"] = sol["V_v"] * dy_c
    # σ_u, σ_v come out of the solver in pixel/s (like V); convert to m/s.
    sol["sigma_u"] = sol["sigma_u"] * np.abs(dx_c)
    sol["sigma_v"] = sol["sigma_v"] * np.abs(dy_c)
    return sol


def solve_full_disk(sat_a, sat_b, t0, dt_minutes, cache_dir, w_full=None,
                    block_rows=600, a_valid=None):
    """Full-disk single-pass WLS solve, done in row-blocks to bound memory.

    The (H, W, 8, 5) design matrix is ~9 GB at full disk, so we build/solve it
    one horizontal strip at a time and assemble the per-pixel outputs.

    ``a_valid`` : optional (H, W) bool mask of where satellite A has data; the
    cached flows are filled everywhere, so where A is masked the flows are set
    to NaN and the solver yields NaN (no spurious off-disk retrievals).
    """
    from stereo_winds.navigation import compute_pixel_scale
    if w_full is None:
        w_u, w_v = get_parallax(sat_a, sat_b, cache_dir)
    else:
        w_u, w_v = w_full
    n = sat_a.n_rows
    st = compute_scene_times(t0, dt_minutes, sat_a, sat_b)
    dx_m, dy_m = compute_pixel_scale(sat_a)
    flows_mm = {k: np.squeeze(np.load(cache_dir / FLOW_FILES[k], mmap_mode="r"))
                for k in FLOW_FILES}

    keys = ["h", "u_wind", "v_wind", "sigma_h", "sigma_u", "sigma_v",
            "chi2", "quality_flag"]
    out = {k: np.full((n, n), np.nan, dtype=np.float32) for k in keys}
    for r0 in range(0, n, block_rows):
        r1 = min(r0 + block_rows, n)
        H_mat = build_design_matrix(
            w_u[r0:r1], w_v[r0:r1],
            dt_a_minus=st["A_minus"], dt_a_plus=st["A_plus"],
            dt_b_minus=st["B_minus"], dt_b_plus=st["B_plus"],
        )
        flows_b = {k: np.array(flows_mm[k][:, r0:r1], dtype=np.float64)
                   for k in FLOW_FILES}
        if a_valid is not None:
            bad = ~a_valid[r0:r1]
            for k in flows_b:
                flows_b[k][:, bad] = np.nan
        sb = solve_stereo_winds(flows_b, H_mat, n_iter=1, device="cpu")
        out["h"][r0:r1] = sb["h"]
        out["u_wind"][r0:r1] = sb["V_u"] * dx_m[r0:r1]
        out["v_wind"][r0:r1] = sb["V_v"] * dy_m[r0:r1]
        out["sigma_h"][r0:r1] = sb["sigma_h"]
        out["sigma_u"][r0:r1] = sb["sigma_u"] * np.abs(dx_m[r0:r1])
        out["sigma_v"][r0:r1] = sb["sigma_v"] * np.abs(dy_m[r0:r1])
        out["chi2"][r0:r1] = sb["chi2"]
        out["quality_flag"][r0:r1] = sb["quality_flag"]
        logger.info("  solved rows %d:%d / %d", r0, r1, n)
    return out


def qa_mask(sol) -> np.ndarray:
    """Standard QA mask on a (cropped) solution dict (mirrors eval_from_parquet)."""
    h = sol["h"]
    spd = np.hypot(sol["u_wind"], sol["v_wind"])
    grad = height_gradient(h)
    return (
        (sol["quality_flag"] > 0)
        & np.isfinite(h) & np.isfinite(sol["chi2"])
        & (sol["chi2"] <= QA["chi2_max"])
        & np.isfinite(sol["sigma_h"]) & (sol["sigma_h"] <= QA["sigma_h_max"])
        & (grad <= QA["h_grad_max"])
        & np.isfinite(spd) & (spd <= QA["wind_speed_max"])
        & (h >= QA["min_height"]) & (h <= 20000.0)
    )


# ---------------------------------------------------------------------------
# Caching the cropped solution
# ---------------------------------------------------------------------------
def save_solution_cache(path: Path, sol, crop, scenes, flows):
    r0, r1, c0, c1 = crop
    ds = xr.Dataset(
        {
            "u_wind": (("y", "x"), sol["u_wind"].astype("float32")),
            "v_wind": (("y", "x"), sol["v_wind"].astype("float32")),
            "cloud_top_height": (("y", "x"), sol["h"].astype("float32")),
            "sigma_h": (("y", "x"), sol["sigma_h"].astype("float32")),
            "sigma_u": (("y", "x"), sol["sigma_u"].astype("float32")),
            "sigma_v": (("y", "x"), sol["sigma_v"].astype("float32")),
            "chi_squared": (("y", "x"), sol["chi2"].astype("float32")),
            "quality_flag": (("y", "x"), sol["quality_flag"].astype("float32")),
        },
        attrs=dict(crop_r0=r0, crop_r1=r1, crop_c0=c0, crop_c1=c1),
    )
    for name, arr in scenes.items():
        ds[f"scene_{name}"] = (("y", "x"), arr.astype("float32"))
    for key, f in flows.items():
        ds[f"flow_{key}_u"] = (("y", "x"), f[0].astype("float32"))
        ds[f"flow_{key}_v"] = (("y", "x"), f[1].astype("float32"))
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    logger.info("Cached cropped solution -> %s", path)


def load_solution_cache(path: Path):
    ds = xr.open_dataset(path)
    crop = (ds.attrs["crop_r0"], ds.attrs["crop_r1"],
            ds.attrs["crop_c0"], ds.attrs["crop_c1"])
    nan = np.full(ds["u_wind"].shape, np.nan, "float32")
    sol = dict(
        u_wind=ds["u_wind"].values, v_wind=ds["v_wind"].values,
        h=ds["cloud_top_height"].values, sigma_h=ds["sigma_h"].values,
        sigma_u=ds["sigma_u"].values if "sigma_u" in ds else nan,
        sigma_v=ds["sigma_v"].values if "sigma_v" in ds else nan,
        chi2=ds["chi_squared"].values, quality_flag=ds["quality_flag"].values,
    )
    scenes = {n: ds[f"scene_{n}"].values for n in SCENE_FILES}
    flows = {k: np.stack([ds[f"flow_{k}_u"].values, ds[f"flow_{k}_v"].values])
             for k in FLOW_FILES}
    return sol, crop, scenes, flows


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _stage_box(fig, rect, label, edge=COLOR_BOX_EDGE, lw=1.1, label_color="#111"):
    """Draw a rounded stage box (figure coords) with a top-left label."""
    x, y, w, h = rect
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
        transform=fig.transFigure, facecolor=COLOR_BOX, edgecolor=edge,
        linewidth=lw, zorder=1, clip_on=False,
    )
    fig.patches.append(box)
    if label:
        fig.text(x + 0.012, y + h - 0.012, label, ha="left", va="top",
                 fontsize=9, fontweight="bold", color=label_color, zorder=5)


def _down_arrow(fig, x, y0, y1, color=COLOR_ARROW, lw=1.6, label=None,
                label_side="right"):
    a = FancyArrowPatch(
        (x, y0), (x, y1), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=14, lw=lw, color=color, zorder=4,
        clip_on=False,
    )
    fig.patches.append(a)
    if label:
        dx = 0.012 if label_side == "right" else -0.012
        ha = "left" if label_side == "right" else "right"
        fig.text(x + dx, (y0 + y1) / 2, label, ha=ha, va="center",
                 fontsize=7.5, style="italic", color=color, zorder=5)


def _right_arrow(fig, x0, x1, y, color=COLOR_ARROW, lw=1.6, label=None):
    a = FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=14, lw=lw, color=color, zorder=4,
        clip_on=False,
    )
    fig.patches.append(a)
    if label:
        fig.text((x0 + x1) / 2, y + 0.012, label, ha="center", va="bottom",
                 fontsize=7.5, style="italic", color=color, zorder=5)


def draw_satellite(ax, x, y, label, color, body_w=0.11, body_h=0.07):
    """Flat schematic satellite glyph (axes data coords): a rectangular body
    with two flanking solar panels, and a colored label above. Used twice in
    the stereo-geometry schematic (one per satellite)."""
    ax.add_patch(Rectangle((x - body_w / 2, y - body_h / 2), body_w, body_h,
                           facecolor="white", edgecolor=color, linewidth=0.9,
                           zorder=7))
    pw, ph = body_w * 0.85, body_h * 0.5  # solar panels
    gap = body_w * 0.18
    for sx in (x - body_w / 2 - gap - pw, x + body_w / 2 + gap):
        ax.add_patch(Rectangle((sx, y - ph / 2), pw, ph, facecolor=color,
                               edgecolor=color, linewidth=0.5, alpha=0.5,
                               zorder=7))
    ax.text(x, y + body_h / 2 + 0.04, label, ha="center", va="bottom",
            fontsize=7, color=color, fontweight="bold", zorder=8)


def _thumb(fig, rect, title, title_color="#222"):
    """Add an inset axis (figure coords) for a thumbnail; return the axis."""
    ax = fig.add_axes(rect, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_edgecolor("#888")
    if title:
        ax.set_title(title, fontsize=7.5, pad=2, color=title_color)
    return ax


def _show_ir(ax, rad, p=(2, 98)):
    """Grayscale brightness-inverted IR (cold clouds bright)."""
    finite = np.isfinite(rad)
    if finite.any():
        lo, hi = np.percentile(rad[finite], p)
    else:
        lo, hi = 0, 1
    ax.imshow(rad, cmap="gray_r", vmin=lo, vmax=hi, origin="upper",
              interpolation="nearest")


def _show_flow_mag(ax, flow, vmax=None):
    """Flow-magnitude image (pixels of displacement)."""
    mag = np.hypot(flow[0], flow[1])
    if vmax is None:
        finite = np.isfinite(mag)
        vmax = np.percentile(mag[finite], 98) if finite.any() else 1.0
    ax.imshow(mag, cmap="magma", vmin=0, vmax=vmax, origin="upper",
              interpolation="nearest")
    return vmax


def _barbs_by_height(ax, sol, good, stride, length=4.6, lw=0.5):
    """Wind barbs colored by feature-tracked height over the crop (knots)."""
    h = sol["h"]
    u_kt = sol["u_wind"] * MS_TO_KT
    v_kt = sol["v_wind"] * MS_TO_KT
    ny, nx = h.shape
    yy, xx = np.mgrid[0:ny:stride, 0:nx:stride]
    for i in range(N_HBINS):
        lo, hi = H_EDGES[i], H_EDGES[i + 1]
        sel = good & (h >= lo) & (h < hi)
        m = sel[::stride, ::stride]
        if not np.any(m):
            continue
        ax.barbs(
            xx[m], yy[m], u_kt[::stride, ::stride][m], v_kt[::stride, ::stride][m],
            length=length, linewidth=lw, pivot="middle",
            barb_increments=dict(half=5, full=10, flag=50),
            color=HCMAP(HNORM((lo + hi) / 2)), zorder=3 + i,
        )
    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)


# ---------------------------------------------------------------------------
# Standalone thumbnail export (for assembling the figure externally, e.g. BioRender)
# ---------------------------------------------------------------------------
def _clean_axes(nx, ny, scale=900):
    """A borderless axes-only figure matched to an (ny, nx) array's aspect."""
    asp = ny / nx
    fig = plt.figure(figsize=(scale / 100.0, scale / 100.0 * asp), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    return fig, ax


def _save_field_over_ir(out_dir, fname, field, good, ir, cmap, norm):
    """Save a scalar field over a faint IR layer; `good` picks shown pixels."""
    ny, nx = field.shape
    fig, ax = _clean_axes(nx, ny)
    finite = np.isfinite(ir)
    lo, hi = (np.percentile(ir[finite], (2, 98)) if finite.any() else (0, 1))
    ax.imshow(ir, cmap="gray_r", vmin=lo, vmax=hi, origin="upper",
              interpolation="nearest", alpha=0.35)
    ax.imshow(np.where(good, field, np.nan), cmap=cmap, norm=norm,
              origin="upper", interpolation="nearest")
    ax.set_xlim(0, nx); ax.set_ylim(ny, 0)
    fig.savefig(out_dir / fname, dpi=150, transparent=False)
    plt.close(fig)


def _save_colorbar(out_dir, prefix, cmap, norm, label, ticks=None,
                   ticklabels=None):
    """Save standalone horizontal + vertical colorbars (transparent)."""
    from matplotlib.colorbar import ColorbarBase
    for orient, figsize, rect in (
        ("horizontal", (4.2, 0.7), [0.04, 0.45, 0.92, 0.32]),
        ("vertical", (0.95, 4.2), [0.18, 0.05, 0.30, 0.90]),
    ):
        fig = plt.figure(figsize=figsize)
        cax = fig.add_axes(rect)
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation=orient)
        cb.set_label(label, fontsize=9)
        if ticks is not None:
            cb.set_ticks(ticks)
            if ticklabels is not None:
                cb.set_ticklabels(ticklabels)
        cb.ax.tick_params(labelsize=8)
        fig.savefig(out_dir / f"{prefix}_{orient}.png", dpi=300,
                    transparent=True, bbox_inches="tight")
        plt.close(fig)


def export_thumbs(sol, crop, scenes, flows, out_dir: Path):
    """Write each panel as its own clean PNG (no boxes/labels/axes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Dense outputs: show the full retrieval (no QA mask) — only drop genuinely
    # non-finite pixels. Display norms clip out-of-range values.
    valid_h = np.isfinite(sol["h"])
    valid_w = valid_h & np.isfinite(sol["u_wind"]) & np.isfinite(sol["v_wind"])
    ny, nx = sol["h"].shape

    # --- inputs: grayscale brightness-inverted IR (cold clouds bright) -------
    for name, rad in scenes.items():
        finite = np.isfinite(rad)
        lo, hi = (np.percentile(rad[finite], (2, 98)) if finite.any() else (0, 1))
        plt.imsave(out_dir / f"input_{name}.png", rad, cmap="gray_r",
                   vmin=lo, vmax=hi, origin="upper")

    # --- optical flow: magnitude, one scale per pair ------------------------
    def _vmax(keys):
        return max(np.nanpercentile(np.hypot(flows[k][0], flows[k][1]), 98)
                   for k in keys)
    vmk = {"D1": _vmax(("D1", "D2")), "D2": _vmax(("D1", "D2")),
           "D3": _vmax(("D3", "D4")), "D4": _vmax(("D3", "D4"))}
    for key, f in flows.items():
        mag = np.hypot(f[0], f[1])
        plt.imsave(out_dir / f"flow_{key}.png", mag, cmap="magma",
                   vmin=0, vmax=vmk[key], origin="upper")

    # --- output: dense feature-tracked height over faint IR -----------------------
    fig, ax = _clean_axes(nx, ny)
    finite = np.isfinite(scenes["A0"])
    lo, hi = (np.percentile(scenes["A0"][finite], (2, 98)) if finite.any()
              else (0, 1))
    ax.imshow(scenes["A0"], cmap="gray_r", vmin=lo, vmax=hi, origin="upper",
              interpolation="nearest", alpha=0.35)
    ax.imshow(np.where(valid_h, sol["h"], np.nan), cmap=HCMAP, norm=HNORM,
              origin="upper", interpolation="nearest")
    ax.set_xlim(0, nx); ax.set_ylim(ny, 0)
    fig.savefig(out_dir / "output_height.png", dpi=150, transparent=False)
    plt.close(fig)

    # --- output: dense wind barbs colored by height, over IR ----------------
    stride = max(1, int(round(np.sqrt(max(valid_w.sum(), 1) / 280.0))))
    fig, ax = _clean_axes(nx, ny)
    ax.imshow(scenes["A0"], cmap="gray_r", vmin=lo, vmax=hi, origin="upper",
              interpolation="nearest")
    _barbs_by_height(ax, sol, valid_w, stride)
    ax.set_xlim(0, nx); ax.set_ylim(ny, 0)
    fig.savefig(out_dir / "output_barbs_over_ir.png", dpi=150)
    plt.close(fig)

    # --- output: dense wind barbs only, transparent background --------------
    fig, ax = _clean_axes(nx, ny)
    _barbs_by_height(ax, sol, valid_w, stride)
    ax.set_xlim(0, nx); ax.set_ylim(ny, 0)
    fig.savefig(out_dir / "output_barbs_transparent.png", dpi=150,
                transparent=True)
    plt.close(fig)

    # --- standalone height colorbars (horizontal + vertical) ----------------
    hticks = np.arange(0, 16001, 4000)
    _save_colorbar(out_dir, "colorbar_height", HCMAP, HNORM,
                   "feature-tracked height (km)", ticks=hticks,
                   ticklabels=[f"{t/1000:.0f}" for t in hticks])

    # --- dense formal-uncertainty fields (sigma_h, sigma_u, sigma_v) --------
    # Sequential 'cividis' (distinct from viridis height & magma flow). Scale
    # each to the 2nd-98th percentile of its finite values (they are tight, so
    # scaling from 0 would wash them out).
    UCMAP = plt.get_cmap("cividis")
    unc = [
        ("sigma_h", sol["sigma_h"] / 1000.0, "height uncertainty σ_h (km)"),
        ("sigma_u", sol["sigma_u"], "zonal-wind uncertainty σ_u (m/s)"),
        ("sigma_v", sol["sigma_v"], "meridional-wind uncertainty σ_v (m/s)"),
    ]
    for key, field, label in unc:
        fmask = np.isfinite(field)
        vals = field[fmask]
        if vals.size:
            vmin, vmax = (float(np.percentile(vals, 2)),
                          float(np.percentile(vals, 98)))
        else:
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = Normalize(vmin=vmin, vmax=vmax)
        _save_field_over_ir(out_dir, f"output_{key}.png", field, fmask,
                            scenes["A0"], UCMAP, norm)
        _save_colorbar(out_dir, f"colorbar_{key}", UCMAP, norm, label)

    # --- manifest -----------------------------------------------------------
    names = (["input_%s.png" % n for n in scenes]
             + ["flow_%s.png" % k for k in flows]
             + ["output_height.png", "output_barbs_over_ir.png",
                "output_barbs_transparent.png",
                "output_sigma_h.png", "output_sigma_u.png",
                "output_sigma_v.png",
                "colorbar_height_*.png", "colorbar_sigma_{h,u,v}_*.png"])
    (out_dir / "MANIFEST.txt").write_text(
        "Standalone thumbnails for external figure assembly (e.g. BioRender).\n"
        "All imagery is the same crop of the GOES-16 fixed grid.\n\n"
        "Inputs (grayscale, brightness-inverted IR; cold clouds = bright):\n"
        "  input_A_minus/A0/A_plus = GOES-16 triplet at t-dt, t, t+dt\n"
        "  input_B_minus/B_plus    = GOES-18 pair (remapped to A's grid) at t-+dt\n"
        "Optical flow (magnitude, magma; temporal pairs share one scale, "
        "cross-sat pairs another):\n"
        "  flow_D1=A0->A-, D2=A0->A+ (temporal); D3=A0->B-, D4=A0->B+ (cross-sat)\n"
        "Outputs:\n"
        "  output_height.png            = QA feature-tracked height (viridis) over faint IR\n"
        "  output_barbs_over_ir.png     = wind barbs colored by height over IR\n"
        "  output_barbs_transparent.png = wind barbs only, transparent background\n"
        "Formal uncertainties (cividis, QA-masked over faint IR; "
        "scaled to 98th pct):\n"
        "  output_sigma_h.png = feature-tracked height uncertainty (km)\n"
        "  output_sigma_u.png = zonal-wind uncertainty (m/s)\n"
        "  output_sigma_v.png = meridional-wind uncertainty (m/s)\n"
        "Colorbars (transparent, _horizontal & _vertical each):\n"
        "  colorbar_height_*  = feature-tracked height scale (0-16 km)\n"
        "  colorbar_sigma_h_* / sigma_u_* / sigma_v_* = uncertainty scales\n"
    )
    logger.info("Wrote %d thumbnails -> %s", len(names), out_dir)


def export_thumbs_fulldisk(sol, scenes, flows, a_valid, overlap, out_dir: Path,
                           ds=4):
    """Full-disk standalone thumbnails. Rasters are downsampled by `ds` and
    transparent off-disk (NaN); flows are masked where A has no data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    s = np.s_[::ds, ::ds]
    a0d = scenes["A0"][s]
    on = np.isfinite(a0d)
    lo, hi = (np.percentile(a0d[on], (2, 98)) if on.any() else (0, 1))
    avd = a_valid[s]
    h = sol["h"][s]
    cov = (overlap[s] & avd & np.isfinite(h) & (h > 0) & (h < 20000)
           & np.isfinite(sol["u_wind"][s]) & np.isfinite(sol["v_wind"][s]))

    # --- inputs: grayscale brightness-inverted IR (transparent off-disk) ----
    for name, rad in scenes.items():
        rd = rad[s]
        fin = np.isfinite(rd)
        rlo, rhi = (np.percentile(rd[fin], (2, 98)) if fin.any() else (0, 1))
        plt.imsave(out_dir / f"input_{name}.png", rd, cmap="gray_r",
                   vmin=rlo, vmax=rhi, origin="upper")

    # --- optical flow: magnitude, masked to each pair's input domain --------
    # Temporal pairs (D1/D2): GOES-16 (A) domain. Cross-sat pairs (D3/D4):
    # GOES-18 (B) coverage = the geometric remap-overlap mask (clean limb).
    ovd = overlap[s]
    fdom = {"D1": avd & np.isfinite(scenes["A_minus"][s]),
            "D2": avd & np.isfinite(scenes["A_plus"][s]),
            "D3": ovd, "D4": ovd}

    def _vmax(keys):
        return max(np.nanpercentile(
            np.where(fdom[k], np.hypot(flows[k][0][s], flows[k][1][s]), np.nan),
            98) for k in keys)
    vmk = {"D1": _vmax(("D1", "D2")), "D2": _vmax(("D1", "D2")),
           "D3": _vmax(("D3", "D4")), "D4": _vmax(("D3", "D4"))}
    for key, f in flows.items():
        mag = np.where(fdom[key], np.hypot(f[0][s], f[1][s]), np.nan)
        plt.imsave(out_dir / f"flow_{key}.png", mag, cmap="magma",
                   vmin=0, vmax=vmk[key], origin="upper")

    # --- dense feature-tracked height over the coverage lens ----------------------
    _save_field_over_ir(out_dir, "output_height.png", h, cov, a0d, HCMAP, HNORM)

    # --- dense u, v wind components (diverging, shared symmetric scale) ------
    ud, vd = sol["u_wind"][s], sol["v_wind"][s]
    uvvals = np.concatenate([ud[cov].ravel(), vd[cov].ravel()])
    vmag = float(np.percentile(np.abs(uvvals), 98)) if uvvals.size else 1.0
    vmag = vmag if vmag > 0 else 1.0
    vnorm = Normalize(vmin=-vmag, vmax=vmag)
    VCMAP = plt.get_cmap("RdBu_r")
    _save_field_over_ir(out_dir, "output_u.png", ud, cov, a0d, VCMAP, vnorm)
    _save_field_over_ir(out_dir, "output_v.png", vd, cov, a0d, VCMAP, vnorm)
    _save_colorbar(out_dir, "colorbar_uv", VCMAP, vnorm, "wind component (m/s)")

    # --- χ² residual (primary post-hoc QA filter, kept where ≤ 0.2) ---------
    chi2d = sol["chi2"][s]
    cfin = cov & np.isfinite(chi2d)
    cvmax = max(0.4, float(np.percentile(chi2d[cfin], 95)) if cfin.any() else 0.4)
    cnorm = Normalize(vmin=0.0, vmax=cvmax)
    CCMAP = plt.get_cmap("magma")
    _save_field_over_ir(out_dir, "output_chi2.png", chi2d, cfin, a0d, CCMAP, cnorm)
    _save_colorbar(out_dir, "colorbar_chi2", CCMAP, cnorm, "χ²  (QA keeps ≤ 0.2)")

    # --- dense wind barbs colored by height ---------------------------------
    sol_d = {"h": h, "u_wind": sol["u_wind"][s], "v_wind": sol["v_wind"][s]}
    ny, nx = h.shape
    stride = max(1, int(round(np.sqrt(max(cov.sum(), 1) / 1500.0))))
    for fname, transp in (("output_barbs_over_ir.png", False),
                          ("output_barbs_transparent.png", True)):
        fig, ax = _clean_axes(nx, ny)
        if not transp:
            ax.imshow(a0d, cmap="gray_r", vmin=lo, vmax=hi, origin="upper",
                      interpolation="nearest")
        _barbs_by_height(ax, sol_d, cov, stride, length=3.0, lw=0.3)
        ax.set_xlim(0, nx); ax.set_ylim(ny, 0)
        fig.savefig(out_dir / fname, dpi=200, transparent=transp)
        plt.close(fig)

    # --- height colorbar ----------------------------------------------------
    hticks = np.arange(0, 16001, 4000)
    _save_colorbar(out_dir, "colorbar_height", HCMAP, HNORM,
                   "feature-tracked height (km)", ticks=hticks,
                   ticklabels=[f"{t/1000:.0f}" for t in hticks])

    # --- dense uncertainty fields ------------------------------------------
    UCMAP = plt.get_cmap("cividis")
    unc = [
        ("sigma_h", sol["sigma_h"][s] / 1000.0, "height uncertainty σ_h (km)"),
        ("sigma_u", sol["sigma_u"][s], "zonal-wind uncertainty σ_u (m/s)"),
        ("sigma_v", sol["sigma_v"][s], "meridional-wind uncertainty σ_v (m/s)"),
    ] if "sigma_u" in sol else [
        ("sigma_h", sol["sigma_h"][s] / 1000.0, "height uncertainty σ_h (km)")]
    for key, field, label in unc:
        fm = cov & np.isfinite(field)
        vals = field[fm]
        vmn, vmx = ((float(np.percentile(vals, 2)),
                     float(np.percentile(vals, 98))) if vals.size else (0., 1.))
        if vmx <= vmn:
            vmx = vmn + 1e-6
        norm = Normalize(vmin=vmn, vmax=vmx)
        _save_field_over_ir(out_dir, f"output_{key}.png", field, fm, a0d,
                            UCMAP, norm)
        _save_colorbar(out_dir, f"colorbar_{key}", UCMAP, norm, label)

    (out_dir / "MANIFEST.txt").write_text(
        "Full-disk standalone thumbnails for external assembly (e.g. BioRender).\n"
        f"GOES-16 fixed grid, downsampled x{ds}; transparent off-disk.\n"
        "Flows masked where GOES-16 (A) has no data; outputs over the "
        "GOES-16×GOES-18 stereo-overlap lens.\n\n"
        "Inputs (gray_r IR): input_A_minus/A0/A_plus (GOES-16 triplet), "
        "input_B_minus/B_plus (GOES-18, remapped).\n"
        "Optical flow (magma magnitude, per-pair scale, masked to each pair's "
        "input domain): flow_D1..D4.\n"
        "Outputs: output_u.png, output_v.png (RdBu_r winds), output_height.png, "
        "output_chi2.png (QA filter), output_sigma_h/u/v.png; "
        "output_barbs_over_ir.png, output_barbs_transparent.png.\n"
        "Colorbars (_horizontal & _vertical): colorbar_uv_*, colorbar_height_*, "
        "colorbar_chi2_*, colorbar_sigma_h/u/v_*.\n"
    )
    logger.info("Wrote full-disk thumbnails -> %s", out_dir)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def build_figure(sol, crop, scenes, flows, t0, out_base: Path, overlap=None):
    plt.style.use(str(REPO / "figures" / "paper.mplstyle"))
    # Dense outputs (no QA mask) — only drop non-finite pixels.
    valid_h = np.isfinite(sol["h"])
    valid_w = valid_h & np.isfinite(sol["u_wind"]) & np.isfinite(sol["v_wind"])
    ny, nx = sol["h"].shape
    finite = np.isfinite(scenes["A0"])
    lo, hi = (np.percentile(scenes["A0"][finite], (2, 98)) if finite.any()
              else (0, 1))

    fig = plt.figure(figsize=(14.0, 7.2))

    fig.text(0.5, 0.985,
             "Cross-satellite stereo wind retrieval", ha="center", va="top",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.957,
             f"GOES-16 (A) × GOES-18 (B)  ·  "
             f"{t0:%Y-%m-%d %H:%M} UTC  ·  band C14  ·  Δt = ±10 min",
             ha="center", va="top", fontsize=9, color="#444")

    # ===== TOP TIER: horizontal pipeline (inputs → flow → solver) ========
    BAND_Y, BAND_H = 0.515, 0.405  # stage-box vertical extent

    # ---- Stage 1: inputs (5 IR scenes) ----------------------------------
    _stage_box(fig, (0.012, BAND_Y, 0.300, BAND_H),
               "Inputs — 5 infrared scenes")
    tw, th = 0.082, 0.105
    a_titles = [("A_minus", "A−  (t−Δt)"), ("A0", "A₀  (t)"),
                ("A_plus", "A₊  (t+Δt)")]
    b_titles = [("B_minus", "B−  (t−Δt)"), ("B_plus", "B₊  (t+Δt)")]
    xs_a = [0.030, 0.123, 0.216]
    xs_b = [0.077, 0.170]
    for (name, ttl), x in zip(a_titles, xs_a):
        ax = _thumb(fig, (x, 0.715, tw, th), ttl, title_color=COLOR_A)
        _show_ir(ax, scenes[name])
    for (name, ttl), x in zip(b_titles, xs_b):
        ax = _thumb(fig, (x, 0.545, tw, th), ttl, title_color=COLOR_B)
        _show_ir(ax, scenes[name])
    fig.text(0.300, 0.700, "GOES-16\ntriplet (A)", ha="right", va="center",
             fontsize=6.4, color=COLOR_A, fontweight="bold")
    fig.text(0.300, 0.530, "GOES-18\npair (B)", ha="right", va="center",
             fontsize=6.4, color=COLOR_B, fontweight="bold")

    _right_arrow(fig, 0.315, 0.362, 0.72, label="RAFT optical flow")

    # ---- Stage 2: optical flow (4 disparity fields) ---------------------
    _stage_box(fig, (0.365, BAND_Y, 0.270, BAND_H),
               "Optical flow — RAFT disparity")
    flow_meta = [
        ("D1", "D₁: A₀→A−", "temporal", COLOR_A),
        ("D2", "D₂: A₀→A₊", "temporal", COLOR_A),
        ("D3", "D₃: A₀→B−", "cross-sat", COLOR_PARALLAX),
        ("D4", "D₄: A₀→B₊", "cross-sat", COLOR_PARALLAX),
    ]
    fxy = {"D1": (0.405, 0.720), "D2": (0.515, 0.720),
           "D3": (0.405, 0.550), "D4": (0.515, 0.550)}

    # Mask each disparity field to its valid input domain — off-disk/limb RAFT
    # output is garbage. Temporal pairs (D1/D2) use the GOES-16 (A) domain;
    # cross-sat pairs (D3/D4) use the GOES-18 (B) coverage mask: the geometric
    # remap-overlap mask when available (clean limb), else finite remapped B.
    a0_fin = np.isfinite(scenes["A0"])
    b_mask = overlap if overlap is not None else None
    flow_dom = {
        "D1": a0_fin & np.isfinite(scenes["A_minus"]),
        "D2": a0_fin & np.isfinite(scenes["A_plus"]),
        "D3": b_mask if b_mask is not None
        else a0_fin & np.isfinite(scenes["B_minus"]),
        "D4": b_mask if b_mask is not None
        else a0_fin & np.isfinite(scenes["B_plus"]),
    }
    flows_m = {}
    for k, f in flows.items():
        dom = flow_dom[k]
        flows_m[k] = np.stack([np.where(dom, f[0], np.nan),
                               np.where(dom, f[1], np.nan)])

    def _grp_vmax(keys):
        return max(np.nanpercentile(np.hypot(flows_m[k][0], flows_m[k][1]), 98)
                   for k in keys)
    vmax_tmp = _grp_vmax(("D1", "D2"))
    vmax_par = _grp_vmax(("D3", "D4"))
    vmax_by_key = {"D1": vmax_tmp, "D2": vmax_tmp,
                   "D3": vmax_par, "D4": vmax_par}
    for key, ttl, kind, col in flow_meta:
        x, y = fxy[key]
        ax = _thumb(fig, (x, y, 0.100, 0.100), ttl, title_color=col)
        _show_flow_mag(ax, flows_m[key], vmax=vmax_by_key[key])
        med = np.nanmedian(np.hypot(flows_m[key][0], flows_m[key][1]))
        ax.text(0.5, -0.14, f"{kind} · ~{med:.1f} px",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.0,
                color=col, style="italic")

    _right_arrow(fig, 0.638, 0.685, 0.72,
                 label="w(h),  Δt")

    # ---- Stage 3: solver ------------------------------------------------
    _stage_box(fig, (0.688, BAND_Y, 0.300, BAND_H),
               "Per-pixel 5-state WLS (Carr et al., 2020)")
    # Stereo-geometry schematic (replaces the equation block). Two
    # geostationary satellites (A=GOES-16, B=GOES-18) view a cloud point P at
    # height h. Each line of sight, continued to the surface (h=0), lands at a
    # different point on the Earth arc; that ground separation is the measured
    # disparity, which scales with h — the parallax that the 5-state WLS
    # inverts. β is the parallax angle subtended at P by the two satellites.
    # Heights/altitudes are compressed; the diagram is schematic, not to scale.
    sax = fig.add_axes([0.700, 0.582, 0.276, 0.300], zorder=3)
    sax.set_xlim(-1, 1); sax.set_ylim(0, 1); sax.set_axis_off()

    ec, ea, eb = (0.0, -0.62), 1.62, 0.82          # Earth ellipse (flattened)

    def _surf(px, py, sx, sy):
        """Where the line of sight (sx,sy)->(px,py), continued downward,
        first meets the Earth arc (the h=0 surface-projection point)."""
        dx, dy = px - sx, py - sy
        qa = (dx / ea) ** 2 + (dy / eb) ** 2
        qb = 2 * ((px - ec[0]) * dx / ea ** 2 + (py - ec[1]) * dy / eb ** 2)
        qc = ((px - ec[0]) / ea) ** 2 + ((py - ec[1]) / eb) ** 2 - 1
        disc = max(qb * qb - 4 * qa * qc, 0.0)
        ts = [(-qb + s * np.sqrt(disc)) / (2 * qa) for s in (1, -1)]
        t = min(t for t in ts if t > 1e-6)
        return px + t * dx, py + t * dy

    # --- Earth arc (thick stroke + light-gray fill below) -------------------
    xv = np.linspace(-0.98, 0.98, 200)
    yv = ec[1] + eb * np.sqrt(np.clip(1 - (xv / ea) ** 2, 0, 1))
    sax.fill_between(xv, 0, yv, color="#e8e8ea", zorder=1)
    th = np.degrees(np.arccos(0.98 / ea))
    sax.add_patch(Arc(ec, 2 * ea, 2 * eb, angle=0, theta1=th, theta2=180 - th,
                      color="#555", lw=1.3, zorder=4))

    # --- satellites at sub-longitude offsets from the arc's local vertical ---
    # Scene centered on the GOES-16/18 midpoint (-106.2°); each sat sits ±31°
    # of longitude away → symmetric angular offsets from vertical.
    def _sat_xy(off_deg, r=1.47):
        ang = np.radians(90 - off_deg)
        return ec[0] + r * np.cos(ang), ec[1] + r * np.sin(ang)
    ax_xy = _sat_xy(+26.0)   # GOES-16 (A), east  -> right
    bx_xy = _sat_xy(-26.0)   # GOES-18 (B), west  -> left
    draw_satellite(sax, ax_xy[0], ax_xy[1], "GOES-16 (A)", COLOR_A)
    draw_satellite(sax, bx_xy[0], bx_xy[1], "GOES-18 (B)", COLOR_B)

    # --- cloud point P (in the overlap, near the midpoint) ------------------
    P = (0.03, 0.33)
    for (sx, sy), col in ((ax_xy, COLOR_A), (bx_xy, COLOR_B)):
        sgx, sgy = _surf(P[0], P[1], sx, sy)
        sax.plot([sx, P[0]], [sy, P[1]], color=col, lw=0.8, alpha=0.85,
                 zorder=5)                                    # line of sight
        sax.plot([P[0], sgx], [P[1], sgy], color="#777", lw=0.7, ls="--",
                 zorder=5)                                    # P -> surface
        sax.plot(sgx, sgy, marker="o", mfc="none", mec=col, mew=0.9, ms=5,
                 zorder=6)                                    # surface point
    gax = _surf(P[0], P[1], *ax_xy)
    gbx = _surf(P[0], P[1], *bx_xy)

    # --- disparity bracket between the two surface-projection points --------
    ybr = min(gax[1], gbx[1]) - 0.06
    sax.plot([gbx[0], gax[0]], [ybr, ybr], color="#333", lw=0.8, zorder=6)
    for gx_ in (gbx[0], gax[0]):
        sax.plot([gx_, gx_], [ybr, ybr + 0.03], color="#333", lw=0.8, zorder=6)
    sax.text((gax[0] + gbx[0]) / 2, ybr - 0.03, r"disparity $\propto h$",
             ha="center", va="top", fontsize=6.6, color="#333", zorder=6)

    # --- parallax angle β at P ----------------------------------------------
    aA = np.degrees(np.arctan2(ax_xy[1] - P[1], ax_xy[0] - P[0]))
    aB = np.degrees(np.arctan2(bx_xy[1] - P[1], bx_xy[0] - P[0]))
    sax.add_patch(Arc(P, 0.26, 0.26, angle=0, theta1=aA, theta2=aB,
                      color="#444", lw=0.7, zorder=6))
    sax.text(P[0], P[1] + 0.17, r"$\beta$", ha="center", va="bottom",
             fontsize=8, color="#444", zorder=7)
    sax.plot(P[0], P[1], marker="o", color="#222", ms=5, zorder=7)
    sax.text(P[0] - 0.06, P[1] - 0.02, r"$P\,(h)$", ha="right", va="center",
             fontsize=7.5, color="#111", zorder=7)

    # --- single muted caption replacing the equation block ------------------
    fig.text(0.838, 0.560,
             r"8 disparities $\rightarrow$ 5-state per-pixel WLS:  "
             r"$\mathbf{x}=[\,h,\,p_u,\,p_v,\,V_u,\,V_v\,]$   (eq. X, §3.2)",
             ha="center", va="center", fontsize=6.6, color="#555")

    _down_arrow(fig, 0.838, 0.510, 0.452, label="retrieval")

    # ===== BOTTOM TIER: outputs — single wide row of 6 maps ==============
    _stage_box(fig, (0.012, 0.045, 0.976, 0.400),
               "Outputs — dense winds, feature-tracked height & formal uncertainties")

    VCMAP = plt.get_cmap("RdBu_r")  # diverging: red +, blue −
    UCMAP = plt.get_cmap("cividis")
    uv = np.concatenate([sol["u_wind"][valid_w].ravel(),
                         sol["v_wind"][valid_w].ravel()])
    vmag = float(np.percentile(np.abs(uv), 98)) if uv.size else 1.0
    if vmag <= 0:
        vmag = 1.0
    vnorm = Normalize(vmin=-vmag, vmax=vmag)

    def _sig_norm(field):
        v = field[np.isfinite(field)]
        a, b = ((float(np.percentile(v, 2)), float(np.percentile(v, 98)))
                if v.size else (0.0, 1.0))
        return Normalize(vmin=a, vmax=(b if b > a else a + 1e-6)), a, b

    sh, su, sv = sol["sigma_h"] / 1000.0, sol["sigma_u"], sol["sigma_v"]
    nh, ah, bh = _sig_norm(sh)
    nu, au, bu = _sig_norm(su)
    nvv, av, bv = _sig_norm(sv)

    # χ² — the primary post-hoc QA filter (retrievals kept where χ² ≤ 0.2).
    chi2 = sol["chi2"]
    cfin = np.isfinite(chi2)
    cvmax = max(0.4, float(np.percentile(chi2[cfin], 95)) if cfin.any() else 0.4)
    cnorm = Normalize(vmin=0.0, vmax=cvmax)

    # (title, unit, field, cmap, norm, tick-spec)
    maps = [
        ("Eastward wind u", "m/s", np.where(valid_w, sol["u_wind"], np.nan),
         VCMAP, vnorm, [-vmag, 0, vmag]),
        ("Northward wind v", "m/s", np.where(valid_w, sol["v_wind"], np.nan),
         VCMAP, vnorm, [-vmag, 0, vmag]),
        ("Feature-tracked height", "km", np.where(valid_h, sol["h"], np.nan),
         HCMAP, HNORM, list(np.arange(0, 16001, 4000))),
        ("χ² residual", "QA keeps ≤ 0.2", np.where(cfin, chi2, np.nan),
         plt.get_cmap("magma"), cnorm, [0.0, QA["chi2_max"], cvmax]),
        ("σ height", "km", np.where(np.isfinite(sh), sh, np.nan),
         UCMAP, nh, [ah, (ah + bh) / 2, bh]),
        ("σ u", "m/s", np.where(np.isfinite(su), su, np.nan),
         UCMAP, nu, [au, (au + bu) / 2, bu]),
        ("σ v", "m/s", np.where(np.isfinite(sv), sv, np.nan),
         UCMAP, nvv, [av, (av + bv) / 2, bv]),
    ]
    xs7 = [0.030, 0.168, 0.307, 0.445, 0.583, 0.722, 0.860]
    mw, my, mh = 0.115, 0.150, 0.224
    cb_y = 0.128
    for (title, unit, field, cmap, norm, ticks), mx in zip(maps, xs7):
        ax = _thumb(fig, (mx, my, mw, mh), title)
        ax.imshow(scenes["A0"], cmap="gray_r", vmin=lo, vmax=hi, origin="upper",
                  interpolation="nearest", alpha=0.30)
        im = ax.imshow(field, cmap=cmap, norm=norm, origin="upper",
                       interpolation="nearest")
        ax.set_xlim(0, nx); ax.set_ylim(ny, 0)
        cax = fig.add_axes([mx, cb_y, mw, 0.011])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(unit, fontsize=7.0, labelpad=2)
        cb.set_ticks(ticks)
        if cmap is HCMAP:
            cb.set_ticklabels([f"{t/1000:.0f}" for t in ticks])
        else:
            cb.set_ticklabels([f"{t:.0f}" if abs(t) >= 10 else f"{t:.1f}"
                               for t in ticks])
        cb.ax.tick_params(labelsize=6.2)

    fig.text(0.50, 0.062, "plus per-pixel quality flag", ha="center",
             va="center", fontsize=7, color="#555")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300)
        logger.info("Wrote %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full-disk coverage figure
# ---------------------------------------------------------------------------
def save_fulldisk_cache(path: Path, sol):
    ds = xr.Dataset({
        "cloud_top_height": (("y", "x"), sol["h"].astype("float32")),
        "u_wind": (("y", "x"), sol["u_wind"].astype("float32")),
        "v_wind": (("y", "x"), sol["v_wind"].astype("float32")),
        "sigma_h": (("y", "x"), sol["sigma_h"].astype("float32")),
        "sigma_u": (("y", "x"), sol["sigma_u"].astype("float32")),
        "sigma_v": (("y", "x"), sol["sigma_v"].astype("float32")),
        "chi_squared": (("y", "x"), sol["chi2"].astype("float32")),
        "quality_flag": (("y", "x"), sol["quality_flag"].astype("float32")),
    })
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=enc)
    logger.info("Cached full-disk solution -> %s", path)


def load_fulldisk_cache(path: Path):
    ds = xr.open_dataset(path)
    out = dict(
        h=ds["cloud_top_height"].values, u_wind=ds["u_wind"].values,
        v_wind=ds["v_wind"].values, sigma_h=ds["sigma_h"].values,
        chi2=ds["chi_squared"].values, quality_flag=ds["quality_flag"].values,
    )
    if "sigma_u" in ds:
        out["sigma_u"] = ds["sigma_u"].values
        out["sigma_v"] = ds["sigma_v"].values
    return out


def build_fulldisk_figure(sol, a0_rad, overlap, sat_a, t0, out_base: Path,
                          show_barbs=True):
    """Full-disk coverage map: stereo retrieval over the GOES-16 IR disk."""
    plt.style.use(str(REPO / "figures" / "paper.mplstyle"))
    h = sol["h"]
    on_disk = np.isfinite(a0_rad)
    # Coverage = stereo overlap ∩ on-disk ∩ physically valid solve.
    coverage = (overlap & on_disk & np.isfinite(h) & (h > 0) & (h < 20000)
                & np.isfinite(sol["u_wind"]) & np.isfinite(sol["v_wind"]))
    ny, nx = h.shape

    fig = plt.figure(figsize=(8.2, 9.0))
    ax = fig.add_axes([0.02, 0.075, 0.96, 0.86])
    ax.set_axis_off()

    lo, hi = (np.percentile(a0_rad[on_disk], (2, 98)) if on_disk.any()
              else (0, 1))
    ax.imshow(a0_rad, cmap="gray_r", vmin=lo, vmax=hi, origin="upper",
              interpolation="nearest")
    im = ax.imshow(np.where(coverage, h, np.nan), cmap=HCMAP, norm=HNORM,
                   origin="upper", interpolation="nearest")
    if show_barbs:
        stride = max(1, int(round(np.sqrt(max(coverage.sum(), 1) / 700.0))))
        _barbs_by_height(ax, sol, coverage, stride, length=3.4, lw=0.35)
    ax.set_xlim(0, nx); ax.set_ylim(ny, 0)

    # disk-fraction coverage stat (of the visible Earth disk)
    disk_frac = 100.0 * coverage.sum() / max(on_disk.sum(), 1)

    fig.text(0.5, 0.975, "Cross-satellite stereo wind coverage",
             ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.955,
             f"GOES-16 × GOES-18 overlap  ·  {t0:%Y-%m-%d %H:%M} UTC  ·  "
             f"band C14  ·  retrievals over {disk_frac:.0f}% of the GOES-16 disk",
             ha="center", va="top", fontsize=9, color="#444")

    cax = fig.add_axes([0.22, 0.055, 0.56, 0.013])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("feature-tracked height (km)", fontsize=8.5)
    cb.set_ticks(np.arange(0, 16001, 4000))
    cb.set_ticklabels([f"{t/1000:.0f}" for t in np.arange(0, 16001, 4000)])
    cb.ax.tick_params(labelsize=7.5)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=250)
        logger.info("Wrote %s", path)
    plt.close(fig)
    return coverage


def build_fulldisk_uv_figure(sol, a0_rad, coverage, t0, out_base: Path):
    """Full-disk eastward (u) and northward (v) wind-component maps.

    Two side-by-side disk maps over the faint GOES-16 IR background, sharing
    a symmetric diverging scale (RdBu_r) so u and v are directly comparable.
    """
    plt.style.use(str(REPO / "figures" / "paper.mplstyle"))
    on_disk = np.isfinite(a0_rad)
    ny, nx = a0_rad.shape
    lo, hi = (np.percentile(a0_rad[on_disk], (2, 98)) if on_disk.any()
              else (0, 1))

    uv = np.concatenate([sol["u_wind"][coverage].ravel(),
                         sol["v_wind"][coverage].ravel()])
    vmag = float(np.percentile(np.abs(uv), 98)) if uv.size else 1.0
    if vmag <= 0:
        vmag = 1.0
    vnorm = Normalize(vmin=-vmag, vmax=vmag)
    VCMAP = plt.get_cmap("RdBu_r")  # red = eastward/northward (+), blue = (−)

    fig = plt.figure(figsize=(12.0, 6.8))
    panels = [("Eastward wind  u", sol["u_wind"]),
              ("Northward wind  v", sol["v_wind"])]
    rects = [(0.015, 0.105, 0.475, 0.78), (0.510, 0.105, 0.475, 0.78)]
    im = None
    for (title, field), rect in zip(panels, rects):
        ax = fig.add_axes(rect)
        ax.set_axis_off()
        ax.imshow(a0_rad, cmap="gray_r", vmin=lo, vmax=hi, origin="upper",
                  interpolation="nearest")
        im = ax.imshow(np.where(coverage, field, np.nan), cmap=VCMAP,
                       norm=vnorm, origin="upper", interpolation="nearest")
        ax.set_xlim(0, nx); ax.set_ylim(ny, 0)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=4)

    disk_frac = 100.0 * coverage.sum() / max(on_disk.sum(), 1)
    fig.text(0.5, 0.975, "Cross-satellite stereo winds — full-disk components",
             ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             f"GOES-16 × GOES-18 overlap  ·  {t0:%Y-%m-%d %H:%M} UTC  ·  "
             f"band C14  ·  retrievals over {disk_frac:.0f}% of the GOES-16 disk",
             ha="center", va="top", fontsize=9, color="#444")

    cax = fig.add_axes([0.30, 0.062, 0.40, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("wind component (m/s)", fontsize=9)
    ticks = [-vmag, -vmag / 2, 0, vmag / 2, vmag]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{x:.0f}" for x in ticks])
    cb.ax.tick_params(labelsize=8)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=250)
        logger.info("Wrote %s", path)
    plt.close(fig)


def write_caption(path: Path, t0, crop, sol, good):
    r0, r1, c0, c1 = crop
    h = sol["h"][good]
    spd = np.hypot(sol["u_wind"], sol["v_wind"])[good]
    txt = (
        "Figure X. System architecture of the cross-satellite stereo wind "
        "retrieval. Inputs are five infrared (band C14) scenes: a temporal "
        "triplet from satellite A (GOES-16) at t−Δt, t, t+Δt and "
        "a pair from satellite B (GOES-18) at t±Δt (Δt = 10 min), "
        f"shown here for {t0:%Y-%m-%d %H:%M} UTC. A RAFT optical-flow network "
        "computes four disparity fields D₁–D₄: two same-satellite "
        "(temporal) pairs that constrain cloud motion and two cross-satellite "
        "pairs whose displacement encodes height-dependent parallax. The eight "
        "displacement components (u, v of each pair) feed a per-pixel 5-state "
        "weighted least-squares solver (Carr et al., 2020) that recovers the "
        "state vector x = [h, p_u, p_v, V_u, V_v] — feature-tracked height, "
        "co-registration offset, and pixel velocity — using the parallax "
        "sensitivity w(h) and scan-time offsets Δt as the design matrix. "
        "Outputs are the dense eastward (u) and northward (v) wind components "
        "and feature-tracked height (shown as maps), plus formal uncertainties and "
        "χ². The output maps show only quality-controlled retrievals. "
        f"Displayed crop: rows {r0}:{r1}, cols {c0}:{c1} of the GOES-16 fixed "
        "grid (cross-satellite overlap). Retrieved here: median height "
        f"{np.nanmedian(h)/1000:.1f} km, median speed {np.nanmedian(spd):.1f} "
        f"m/s over {int(good.sum())} QA-passing pixels."
    )
    path.write_text(txt + "\n")
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
def _run_full_disk(args, t0, cache_dir):
    sat_a, sat_b = GOES16_CONFIG, GOES18_CONFIG
    day = f"{t0.timetuple().tm_yday:03d}"
    fd_cache = REPO / "figures" / "cache" / "methodology_fulldisk_C14.nc"
    out_base = REPO / "figures" / "fig_methodology_fulldisk_coverage"

    # overlap mask + GOES-16 IR background (A0). a_valid = where A has data.
    col_b, row_b = load_remap_lut(cache_dir / "remap_lut_g16_g18.npz")
    overlap = (np.isfinite(col_b) & np.isfinite(row_b)
               & (col_b >= 0) & (col_b <= sat_b.n_cols - 1)
               & (row_b >= 0) & (row_b <= sat_b.n_rows - 1))
    a0_rad = load_radiance(_find_scene_nc(cache_dir, "goes16", day, "1900"))
    a_valid = np.isfinite(a0_rad)

    if args.from_cache and fd_cache.exists():
        logger.info("Loading full-disk solution from %s", fd_cache)
        sol = load_fulldisk_cache(fd_cache)
    else:
        w_u, w_v = get_parallax(sat_a, sat_b, cache_dir)
        logger.info("Solving full disk (row-blocked) — this takes a minute...")
        sol = solve_full_disk(sat_a, sat_b, t0, args.dt_minutes, cache_dir,
                              w_full=(w_u, w_v), a_valid=a_valid)
        save_fulldisk_cache(fd_cache, sol)

    coverage = build_fulldisk_figure(sol, a0_rad, overlap, sat_a, t0, out_base,
                                     show_barbs=args.barbs)
    build_fulldisk_uv_figure(
        sol, a0_rad, coverage, t0,
        REPO / "figures" / "fig_methodology_fulldisk_uv")
    h = sol["h"][coverage]
    spd = np.hypot(sol["u_wind"], sol["v_wind"])[coverage]
    logger.info("Coverage: %d pixels (%.1f%% of disk); height median %.1f km, "
                "speed median %.1f m/s", int(coverage.sum()),
                100 * coverage.sum() / max(a_valid.sum(), 1),
                np.nanmedian(h) / 1000, np.nanmedian(spd))

    if args.thumbs:
        logger.info("Loading full-disk scenes for thumbnails...")
        scenes = {}
        for name, (sat, hhmm) in SCENE_FILES.items():
            rad = load_radiance(_find_scene_nc(cache_dir, sat, day, hhmm))
            if sat == sat_b.satellite_id:
                rad = remap_image(rad, col_b, row_b)
            scenes[name] = rad
        # Fill NaN dropouts in the GOES-16 (A) scenes from the complete A0.
        a0_ref = scenes["A0"].copy()
        for name, (sat, _) in SCENE_FILES.items():
            if sat == sat_a.satellite_id:
                scenes[name] = fill_disk_holes(scenes[name], a_valid, a0_ref)
        flows = {k: load_flow(cache_dir, k) for k in FLOW_FILES}
        export_thumbs_fulldisk(sol, scenes, flows, a_valid, overlap,
                               Path(args.thumbs_dir))


def load_fulldisk_figure_data(t0, cache_dir: Path):
    """Load full-disk solution + scenes + flows for the assembled figure.

    Reuses the cached full-disk solution (``methodology_fulldisk_C14.nc``) and
    loads the five full-disk IR scenes (B remapped onto A's grid, A dropouts
    filled from A0) and the four full-disk RAFT disparity fields.
    """
    sat_a, sat_b = GOES16_CONFIG, GOES18_CONFIG
    day = f"{t0.timetuple().tm_yday:03d}"
    fd_cache = REPO / "figures" / "cache" / "methodology_fulldisk_C14.nc"
    if not fd_cache.exists():
        raise FileNotFoundError(
            f"{fd_cache} not found — run --full-disk first to build it")
    logger.info("Loading full-disk solution from %s", fd_cache)
    sol = load_fulldisk_cache(fd_cache)

    col_b, row_b = load_remap_lut(cache_dir / "remap_lut_g16_g18.npz")
    overlap = (np.isfinite(col_b) & np.isfinite(row_b)
               & (col_b >= 0) & (col_b <= sat_b.n_cols - 1)
               & (row_b >= 0) & (row_b <= sat_b.n_rows - 1))
    scenes = {}
    for name, (sat, hhmm) in SCENE_FILES.items():
        rad = load_radiance(_find_scene_nc(cache_dir, sat, day, hhmm))
        if sat == sat_b.satellite_id:
            rad = remap_image(rad, col_b, row_b)
        scenes[name] = rad
    a_valid = np.isfinite(scenes["A0"])
    a0_ref = scenes["A0"].copy()
    for name, (sat, _) in SCENE_FILES.items():
        if sat == sat_a.satellite_id:
            scenes[name] = fill_disk_holes(scenes[name], a_valid, a0_ref)

    logger.info("Loading full-disk flow fields...")
    flows = {k: load_flow(cache_dir, k) for k in FLOW_FILES}
    ny, nx = sol["h"].shape
    return sol, (0, ny, 0, nx), scenes, flows, overlap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time", default="2024-01-15T19:00",
                    help="center time t0 (ISO UTC)")
    ap.add_argument("--cache-dir", default=str(REPO / "cache"))
    ap.add_argument("--dt-minutes", type=float, default=10.0)
    ap.add_argument("--crop", type=int, nargs=4, default=None,
                    metavar=("R0", "R1", "C0", "C1"),
                    help="crop window; default = NE-Pacific windy region")
    ap.add_argument("--auto-crop", action="store_true",
                    help="center the crop on the overlap centroid instead of "
                         "the default windy region")
    ap.add_argument("--crop-size", type=int, default=800,
                    help="auto-crop window size (pixels)")
    ap.add_argument("--from-cache", action="store_true",
                    help="re-render from cached cropped solution")
    ap.add_argument("--thumbs", action="store_true",
                    help="export standalone panel thumbnails (for BioRender "
                         "etc.) instead of the assembled figure")
    ap.add_argument("--thumbs-dir",
                    default=str(REPO / "figures" / "methodology_thumbs"))
    ap.add_argument("--full-disk", action="store_true",
                    help="solve the entire GOES-16 disk (row-blocked) and "
                         "render a coverage map of the stereo overlap")
    ap.add_argument("--fulldisk-thumbs", action="store_true",
                    help="assemble the architecture figure from full-disk "
                         "scenes/flows/solution instead of the crop")
    ap.add_argument("--barbs", action="store_true",
                    help="full-disk: overlay a coarse wind-barb field (off by "
                         "default — illegible at full-disk scale)")
    ap.add_argument("--out", default=str(REPO / "figures"
                                         / "fig_methodology_architecture"))
    args = ap.parse_args()

    t0 = dt.datetime.fromisoformat(args.time)
    cache_dir = Path(args.cache_dir)
    out_base = Path(args.out)
    sol_cache = REPO / "figures" / "cache" / "methodology_arch_C14.nc"

    if args.full_disk:
        _run_full_disk(args, t0, cache_dir)
        return

    overlap_fig = None  # B-satellite coverage mask (for D3/D4 flow masking)
    if args.fulldisk_thumbs:
        sol, crop, scenes, flows, overlap_fig = load_fulldisk_figure_data(
            t0, cache_dir)
    elif args.from_cache and sol_cache.exists():
        logger.info("Loading cropped solution from %s", sol_cache)
        sol, crop, scenes, flows = load_solution_cache(sol_cache)
    else:
        sat_a, sat_b = GOES16_CONFIG, GOES18_CONFIG
        day = f"{t0.timetuple().tm_yday:03d}"

        # parallax (full disk) + overlap mask from the remap LUT
        w_u, w_v = get_parallax(sat_a, sat_b, cache_dir)
        lut_path = cache_dir / "remap_lut_g16_g18.npz"
        col_b, row_b = load_remap_lut(lut_path)
        overlap = (np.isfinite(col_b) & np.isfinite(row_b)
                   & (col_b >= 0) & (col_b <= sat_b.n_cols - 1)
                   & (row_b >= 0) & (row_b <= sat_b.n_rows - 1))

        if args.crop:
            crop = tuple(args.crop)
        elif args.auto_crop:
            crop = auto_crop(overlap, args.crop_size)
        else:
            crop = DEFAULT_CROP
        r0, r1, c0, c1 = crop
        logger.info("Crop: rows %d:%d cols %d:%d  (overlap frac %.2f)",
                    r0, r1, c0, c1, overlap[r0:r1, c0:c1].mean())

        # load + crop the 5 IR scenes (B remapped onto A's grid)
        scenes = {}
        for name, (sat, hhmm) in SCENE_FILES.items():
            nc = _find_scene_nc(cache_dir, sat, day, hhmm)
            rad = load_radiance(nc)
            if sat == sat_b.satellite_id:
                rad = remap_image(rad, col_b, row_b)
            scenes[name] = rad[r0:r1, c0:c1]
        # Fill NaN dropouts in the GOES-16 (A) scenes from the complete A0.
        on_disk_c = np.isfinite(scenes["A0"])
        a0_ref = scenes["A0"].copy()
        for name, (sat, _) in SCENE_FILES.items():
            if sat == GOES16_CONFIG.satellite_id:
                scenes[name] = fill_disk_holes(scenes[name], on_disk_c, a0_ref)

        overlap_fig = overlap[r0:r1, c0:c1]
        # load + crop the 4 flow fields
        flows = {k: load_flow(cache_dir, k)[:, r0:r1, c0:c1] for k in FLOW_FILES}

        sol = solve_crop(sat_a, sat_b, t0, args.dt_minutes, crop, cache_dir,
                         flows_full=None, w_full=(w_u, w_v))
        save_solution_cache(sol_cache, sol, crop, scenes, flows)

    good = qa_mask(sol)
    h = sol["h"][good]
    spd = np.hypot(sol["u_wind"], sol["v_wind"])[good]
    logger.info("QA-pass pixels: %d (%.1f%%)", int(good.sum()),
                100 * good.mean())
    if good.any():
        logger.info("  height  median %.2f km  range %.2f-%.2f km",
                    np.nanmedian(h) / 1000, np.nanmin(h) / 1000,
                    np.nanmax(h) / 1000)
        logger.info("  speed   median %.1f m/s  max %.1f m/s",
                    np.nanmedian(spd), np.nanmax(spd))

    if args.thumbs:
        export_thumbs(sol, crop, scenes, flows, Path(args.thumbs_dir))
    else:
        build_figure(sol, crop, scenes, flows, t0, out_base,
                     overlap=overlap_fig)
    write_caption(out_base.parent / (out_base.name + "_caption.txt"),
                  t0, crop, sol, good)


if __name__ == "__main__":
    main()
