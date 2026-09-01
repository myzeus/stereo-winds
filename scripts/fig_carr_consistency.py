#!/usr/bin/env python
"""Results figures: consistency of the RAFT stereo retrieval with Carr NCC winds.

Two figures at the Carr-matched timestep (default 2025-01-08 19:00 UTC, C14,
GOES-16 x GOES-18 overlap):

  Figure A  (fig_carr_consistency)      -- 3x3 plate:
      row 1  RAFT dense u / v / h maps
      row 2  Carr sparse-point u / v / h maps (native ~32-km sampling)
      row 3  collocated RAFT-vs-Carr density scatter for u, v, h
             (full-overlap collocation set; bias + RMSD + r + N annotated)

  Figure B  (fig_carr_density_overlay)  -- the density contrast made visual:
      RAFT dense height field with Carr's sparse vectors overlaid, same scene.

Both pipelines use the IDENTICAL cross-satellite five-state WLS solver
(Carr et al. 2020); this figure shows the two independent disparity front-ends
(RAFT optical flow vs Carr NCC) agree where they are collocated, while RAFT
fills the field continuously.

Data + GPU live on the compute cluster. Run once there to build the retrieval cache
(figures/cache/retr_tuned_<band>.nc) and a compact display bundle
(figures/cache/carr_consistency_<band>.npz); the bundle rsyncs back small so
the figures can be re-laid-out locally with --from-npz (no GPU, no big NetCDF).

  # on a GPU node, builds bundle + renders:
  python scripts/fig_carr_consistency.py --from-cache
  # locally, fast layout iteration from the bundle:
  python scripts/fig_carr_consistency.py --from-npz
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

# Reuse the single-source-of-truth infrastructure (do NOT reimplement).
from make_fig_spatial_barbs import (  # noqa: E402
    CMAP, NORM, MS_TO_KT, PANEL_A_EXTENT, DATA_DIR_DEFAULT,
    load_or_infer, geo_and_extent, carr_fair_mask, barbs_binned,
)
from eval_from_parquet import _build_qa_mask  # noqa: E402
from stereo_winds.validation.amv_comparison import load_carr_data  # noqa: E402
from stereo_winds.config import SATELLITE_CONFIGS  # noqa: E402
from stereo_winds.navigation import (  # noqa: E402
    compute_grid_latlon, geodetic_to_fixed_grid, scanning_angle_to_pixel,
)
from stereo_winds.config import GOES16_CONFIG  # noqa: E402
from dataclasses import replace as _dc_replace  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STYLE = BASE / "figures" / "paper.mplstyle"
PC = ccrs.PlateCarree()

# --- Colormaps / norms -------------------------------------------------------
# Height reuses the results-figure height ramp (viridis, 0-16 km) so this figure
# is visually consistent with the spatial-barb results figure. u, v use a
# diverging ramp centered on zero (eastward/northward sign is the signal).
H_CMAP, H_NORM = CMAP, NORM               # viridis, Normalize(0, 16000) m
UV_CMAP = "RdBu_r"
COLLOC_RADIUS_DEFAULT = 3                 # px; matches make_fig_spatial_barbs
DS_FACTOR_DEFAULT = 4                     # display downsample for dense maps


# ---- Collocation (u, v, h) --------------------------------------------------
def collocate_uvh(carr, ds, sat, extent, radius):
    """Sample the RAFT field at each Carr vector's nearest QA pixel within `radius`.

    Mirrors make_fig_spatial_barbs.collocate_carr_tuned but keeps the u, v, h
    components (that helper collapses to speed). extent=None -> full overlap.
    Unmatched Carr points (no QA pixel in the window) are dropped.
    """
    qa = _build_qa_mask(ds)
    h = ds["cloud_top_height"].values
    u = ds["u_wind"].values
    v = ds["v_wind"].values
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
    out = {k: [] for k in ("carr_u", "carr_v", "carr_h",
                           "raft_u", "raft_v", "raft_h")}
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
        out["carr_u"].append(cu[k]); out["carr_v"].append(cv[k]); out["carr_h"].append(ch[k])
        out["raft_u"].append(u[ri, ci]); out["raft_v"].append(v[ri, ci]); out["raft_h"].append(h[ri, ci])
    d = {k: np.asarray(val, float) for k, val in out.items()}
    d.update(n=len(d["carr_h"]), n_oob=n_oob, n_noqa=n_noqa)
    return d


# ---- Display bundle ---------------------------------------------------------
def _downsample_masked(field, mask, f):
    """Stride-downsample a field to ~1/f resolution, NaN where QA fails."""
    a = np.where(mask, field, np.nan).astype(np.float32)
    return a[::f, ::f]


def build_bundle(ds, carr, sat, extent, radius, ds_factor, bundle_path,
                 model_label="RAFT stereo"):
    """Assemble everything the two figures need into a compact npz for reuse."""
    qa = _build_qa_mask(ds)
    u = ds["u_wind"].values
    v = ds["v_wind"].values
    h = ds["cloud_top_height"].values

    # Dense display maps (downsampled; transparent where QA fails).
    u_map = _downsample_masked(u, qa, ds_factor)
    v_map = _downsample_masked(v, qa, ds_factor)
    h_map = _downsample_masked(h, qa, ds_factor)

    # Symmetric u/v color range from the QA-passing field (robust 98th pct).
    uv_vmax = float(np.nanpercentile(np.abs(np.concatenate([u[qa], v[qa]])), 98))

    # Carr sparse points (fair vectors inside the overlap) for the maps + overlay.
    fair = carr_fair_mask(carr)
    W, E, S, N = extent
    clon, clat = carr["lon"][fair], carr["lat"][fair]
    inx = (clon >= W) & (clon <= E) & (clat >= S) & (clat <= N)
    carr_pts = dict(lon=clon[inx], lat=clat[inx],
                    u=carr["u"][fair][inx], v=carr["v"][fair][inx],
                    h=carr["h"][fair][inx])

    # Collocated pairs for the scatter (full-overlap set).
    m = collocate_uvh(carr, ds, sat, extent=None, radius=radius)

    geo, ext_m, H = geo_and_extent(sat)
    n_raft_qa = int(qa.sum())

    logger.info("bundle: RAFT QA pixels=%d  Carr fair-in-overlap=%d  collocated=%d",
                n_raft_qa, carr_pts["lon"].size, m["n"])

    np.savez_compressed(
        bundle_path,
        u_map=u_map, v_map=v_map, h_map=h_map,
        ds_factor=ds_factor, uv_vmax=uv_vmax,
        ext_m=np.asarray(ext_m, float), extent=np.asarray(extent, float),
        sub_lon=float(sat.sub_lon_deg), sat_h=float(H), sweep=np.array(sat.sweep),
        carr_lon=carr_pts["lon"], carr_lat=carr_pts["lat"],
        carr_u=carr_pts["u"], carr_v=carr_pts["v"], carr_h=carr_pts["h"],
        c_u=m["carr_u"], c_v=m["carr_v"], c_h=m["carr_h"],
        r_u=m["raft_u"], r_v=m["raft_v"], r_h=m["raft_h"],
        n_raft_qa=n_raft_qa, n_carr=carr_pts["lon"].size, n_colloc=m["n"],
        model_label=np.array(model_label),
    )
    logger.info("wrote display bundle -> %s", bundle_path)
    return dict(np.load(bundle_path, allow_pickle=True))


def _geo_from_bundle(b):
    return ccrs.Geostationary(central_longitude=float(b["sub_lon"]),
                              satellite_height=float(b["sat_h"]),
                              sweep_axis=str(b["sweep"]))


# ---- Figure A: maps + scatter ----------------------------------------------
def _map_ax(fig, gs, geo, extent):
    ax = fig.add_subplot(gs, projection=geo)
    ax.set_facecolor("0.92")
    ax.coastlines(resolution="50m", color="0.25", linewidth=0.4)
    ax.set_extent(extent, crs=PC)
    return ax


def _scatter(ax, xc, yc, lo, hi, cmap="magma", unit=""):
    """Density hexbin of RAFT (y) vs Carr (x) with 1:1 line and stats box."""
    good = np.isfinite(xc) & np.isfinite(yc)
    xc, yc = xc[good], yc[good]
    ax.hexbin(xc, yc, gridsize=45, cmap=cmap, bins="log", mincnt=1,
              extent=(lo, hi, lo, hi))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    bias = float(np.mean(yc - xc))
    rmsd = float(np.sqrt(np.mean((yc - xc) ** 2)))
    r = float(np.corrcoef(xc, yc)[0, 1]) if xc.size > 1 else np.nan
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.text(0.04, 0.96,
            f"N={xc.size:,}\nbias={bias:+.2f} {unit}\nRMSD={rmsd:.2f} {unit}\nr={r:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9))


def make_fig_maps_scatter(b, out_base, model_label="RAFT stereo", cover_deg=0.3):
    geo = _geo_from_bundle(b)
    ext_m = list(b["ext_m"]); extent = list(b["extent"])
    uv_vmax = float(b["uv_vmax"])           # map color range (98th pct of field)
    uv_norm = Normalize(-uv_vmax, uv_vmax)
    # Common-grid rendering (as in make_fig_maps_two_models): bin the dense
    # WindFlow field AND the sparse Carr vectors onto the same ~cover_deg
    # lat/lon cells so both fill identical cell geometry.
    W, E, S, N = [float(x) for x in extent]
    lat_b, lon_b = _raft_latlon(b["h_map"].shape, int(b["ds_factor"]))
    # Scatter axis range is set from the collocated pairs (NOT the field color
    # range) so the high-speed tails are shown, not cropped. Stats are always on
    # the full arrays; this only controls what the hexbin displays.
    uv_pairs = np.abs(np.concatenate([b["c_u"], b["r_u"], b["c_v"], b["r_v"]]))
    uv_pairs = uv_pairs[np.isfinite(uv_pairs)]
    scatter_lim = max(20.0, float(np.ceil(np.percentile(uv_pairs, 99.5) / 5) * 5))
    cols = [
        ("u", "$u$  (m s$^{-1}$)", UV_CMAP, uv_norm, b["u_map"], b["carr_u"]),
        ("v", "$v$  (m s$^{-1}$)", UV_CMAP, uv_norm, b["v_map"], b["carr_v"]),
        ("h", "$h$  (km)", H_CMAP, H_NORM, b["h_map"], b["carr_h"]),
    ]
    clon, clat = b["carr_lon"], b["carr_lat"]
    letters = "abcdefghi"  # a-c row1, d-f row2, g-i row3 (left-to-right)

    def _tag(ax, k):
        ax.text(0.0, 1.02, f"({letters[k]})", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=9, fontweight="bold")

    # Native \textwidth sizing (~7 in) so fonts print at their nominal size.
    fig = plt.figure(figsize=(7.0, 6.9))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 1.05],
                  hspace=0.26, wspace=0.14, left=0.075, right=0.99,
                  top=0.90, bottom=0.065)

    for j, (key, label, cmap, norm, field, cval) in enumerate(cols):
        # Row 1: WindFlow field binned onto the common grid.
        axr = _map_ax(fig, gs[0, j], geo, extent)
        arr, xe, ye = _grid_to_cells(lon_b, lat_b, field, W, E, S, N, cover_deg)
        im = axr.pcolormesh(xe, ye, arr, cmap=cmap, norm=norm,
                            transform=PC, shading="flat", zorder=1)
        if j == 0:
            axr.text(-0.06, 0.5, model_label, transform=axr.transAxes,
                     rotation=90, va="center", ha="center", fontsize=9,
                     fontweight="bold")
        if j == 2:                                    # h panel: coverage stats
            c33, c55 = coverage_at(b, 0.3), coverage_at(b, 0.5)
            axr.text(0.02, 0.02,
                     f"boxes w/ wind\n33 km: {c33['raft']:.0%}\n55 km: {c55['raft']:.0%}\n"
                     f"px fill: {c33['pix_fill']:.0%}",
                     transform=axr.transAxes, va="bottom", ha="left", fontsize=6.0,
                     color="0.15", linespacing=1.15,
                     bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.2))
        axr.set_title(label, fontsize=10)
        _tag(axr, j)

        # Row 2: Carr vectors binned onto the SAME grid.
        axc = _map_ax(fig, gs[1, j], geo, extent)
        arr, xe, ye = _grid_to_cells(clon, clat, cval, W, E, S, N, cover_deg)
        axc.pcolormesh(xe, ye, arr, cmap=cmap, norm=norm,
                       transform=PC, shading="flat", zorder=1)
        if j == 2:                                    # h panel: coverage stats
            c33, c55 = coverage_at(b, 0.3), coverage_at(b, 0.5)
            axc.text(0.02, 0.02,
                     f"boxes w/ wind\n33 km: {c33['carr']:.0%}\n55 km: {c55['carr']:.0%}",
                     transform=axc.transAxes, va="bottom", ha="left", fontsize=6.0,
                     color="0.15", linespacing=1.15,
                     bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.2))
        _tag(axc, 3 + j)
        if j == 0:
            axc.text(-0.06, 0.5, "Carr NCC", transform=axc.transAxes,
                     rotation=90, va="center", ha="center", fontsize=9,
                     fontweight="bold")

        # Shared colorbar spanning the two map rows for this column.
        p_top = axr.get_position(); p_bot = axc.get_position()
        cax = fig.add_axes([p_top.x0, p_bot.y0 - 0.018, p_top.width, 0.010])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                          extend="max" if key == "h" else "both")
        if key == "h":  # relabel to km to match the title and the scatter
            ticks = np.linspace(0, 16000, 5)
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{int(t/1000)}" for t in ticks])

        # Row 3: collocated RAFT-vs-Carr density scatter.
        axs = fig.add_subplot(gs[2, j])
        if key == "h":
            _scatter(axs, b["c_h"] / 1000, b["r_h"] / 1000, 0, 16, unit="km")
            axs.set_xticks(np.arange(0, 17, 4))
            axs.set_yticks(np.arange(0, 17, 4))
            axs.set_xlabel("Carr $h$ (km)")
            axs.set_ylabel(f"{model_label} $h$ (km)")
        else:
            _scatter(axs, b[f"c_{key}"], b[f"r_{key}"], -scatter_lim, scatter_lim,
                     unit="m s$^{-1}$")
            axs.set_xlabel(f"Carr ${key}$ (m s$^{{-1}}$)")
            axs.set_ylabel(f"{model_label} ${key}$ (m s$^{{-1}}$)")
        _tag(axs, 6 + j)

    fig.suptitle(f"Consistency of {model_label} winds with Carr NCC "
                 "(GOES-16 × GOES-18 overlap)", fontsize=11, y=0.985)
    n_raft = int(b["n_raft_qa"]); n_carr = int(b["n_carr"]); n_col = int(b["n_colloc"])
    fig.text(0.5, 0.952,
             f"maps binned to a common {cover_deg:g}° (~{cover_deg*111:.0f} km) grid   ·   "
             f"{model_label}: {n_raft:,} QA pixels   ·   Carr NCC: {n_carr:,} "
             f"vectors   ·   {n_col:,} collocated", ha="center", va="top",
             fontsize=8, color="0.3")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_base}.{ext}", dpi=300)
    plt.close(fig)
    logger.info("wrote %s.{png,pdf}", out_base)


# ---- Figure B: two side-by-side coverage maps (dense field vs sparse points) --
def make_fig_two_maps(b, out_base, model_label="RAFT stereo"):
    """RAFT dense height field beside Carr's sparse height retrieval, same scene.

    Two maps of the identical C14 timestep and extent: the continuous RAFT field
    vs Carr's scattered points. The field-vs-points contrast is the ~220x density
    difference, made visual and honest (no fabricated Carr coverage).
    """
    geo = _geo_from_bundle(b)
    ext_m = list(b["ext_m"]); extent = list(b["extent"])
    n_raft = int(b["n_raft_qa"]); n_carr = int(b["n_carr"])

    fig = plt.figure(figsize=(13.0, 6.6))
    axl = fig.add_axes([0.015, 0.10, 0.475, 0.78], projection=geo)
    axr = fig.add_axes([0.500, 0.10, 0.475, 0.78], projection=geo)
    for ax in (axl, axr):
        ax.set_facecolor("0.92")
        ax.coastlines(resolution="50m", color="0.25", linewidth=0.4, zorder=3)

    im = axl.imshow(b["h_map"], origin="upper", extent=ext_m, transform=geo,
                    cmap=H_CMAP, norm=H_NORM, zorder=1)
    axl.set_extent(extent, crs=PC)
    axl.set_title(f"(a)  {model_label}", fontsize=12, fontweight="bold")

    axr.scatter(b["carr_lon"], b["carr_lat"], c=b["carr_h"], cmap=H_CMAP,
                norm=H_NORM, s=2.0, linewidths=0, transform=PC, zorder=2)
    axr.set_extent(extent, crs=PC)
    axr.set_title("(b)  Carr NCC", fontsize=12, fontweight="bold")

    for ax, n, lab in ((axl, n_raft, "QA pixels"), (axr, n_carr, "vectors")):
        ax.text(0.015, 0.985, f"N = {n:,} {lab}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9),
                zorder=20)

    cax = fig.add_axes([0.30, 0.055, 0.40, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Feature-tracked height (km)")
    ticks = np.linspace(0, 16000, 9)
    cb.set_ticks(ticks); cb.set_ticklabels([f"{int(t/1000)}" for t in ticks])

    fig.suptitle("Coverage contrast at a shared C14 scene "
                 "(GOES-16 × GOES-18 overlap, 2025-01-08 19:00 UTC)",
                 fontsize=13, fontweight="bold", y=0.99)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_base}.{ext}", dpi=300)
    plt.close(fig)
    logger.info("wrote %s.{png,pdf}", out_base)


# ---- Cell-based spatial coverage (WindFlow dense-QA vs Carr sparse) ----------
def _raft_latlon(h_map_shape, ds_factor):
    """lat/lon of each downsampled RAFT map pixel on the GOES-16 base grid."""
    f = int(ds_factor)
    cfg = _dc_replace(GOES16_CONFIG, scale_x=GOES16_CONFIG.scale_x * f,
                      scale_y=GOES16_CONFIG.scale_y * f,
                      n_rows=h_map_shape[0], n_cols=h_map_shape[1])
    return compute_grid_latlon(cfg)               # (lat, lon), NaN off-disk


def _cell_keys(lon, lat, W, E, S, N, deg):
    lon = np.asarray(lon); lat = np.asarray(lat)
    m = (np.isfinite(lon) & np.isfinite(lat)
         & (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N))
    ci = np.floor((lon[m] - W) / deg).astype(np.int64)
    cj = np.floor((lat[m] - S) / deg).astype(np.int64)
    return ci * 1000000 + cj


def coverage_at(b, deg, _cache={}):
    """Cell-coverage of WindFlow QA pixels vs Carr points at cell size `deg`.

    Returns dict with raft/carr covered-cell fractions of the union (the
    stereo-overlap region something retrieves), plus WindFlow's mean per-cell
    fill fraction (QA subpixels / on-disk subpixels) so "covered" is honest.
    """
    W, E, S, N = [float(x) for x in b["extent"]]
    key = id(b)
    if key not in _cache:                          # lat/lon reconstruction is the cost
        _cache[key] = _raft_latlon(b["h_map"].shape, int(b["ds_factor"]))
    lat, lon = _cache[key]
    qa = np.isfinite(b["h_map"])
    ondisk = np.isfinite(lat) & np.isfinite(lon)
    rk = _cell_keys(lon[qa], lat[qa], W, E, S, N, deg)
    ok = _cell_keys(lon[ondisk], lat[ondisk], W, E, S, N, deg)
    ck = _cell_keys(b["carr_lon"], b["carr_lat"], W, E, S, N, deg)
    from collections import Counter
    qa_cnt, od_cnt = Counter(rk.tolist()), Counter(ok.tolist())
    rset, cset = set(qa_cnt), set(ck.tolist())
    union = rset | cset
    n = len(union) or 1
    # pixel fill WITHIN the stereo footprint (union cells): QA px / on-disk px.
    on_in_union = sum(od_cnt[k] for k in union if k in od_cnt)
    qa_in_union = sum(qa_cnt[k] for k in union)
    return dict(raft=len(rset) / n, carr=len(cset) / n,
                raft_only=len(rset - cset) / n, carr_only=len(cset - rset) / n,
                pix_fill=qa_in_union / on_in_union if on_in_union else 0.0,
                n_raft=len(rset), n_carr=len(cset), n_union=len(union))


def _grid_to_cells(lon, lat, val, W, E, S, N, deg):
    """Mean of `val` per lat/lon cell -> (masked arr[nlat,nlon], lon_edges, lat_edges).

    Renders dense rasters and sparse points identically: both are binned onto the
    same regular lat/lon grid so a WindFlow field and a Carr point cloud fill the
    same cell geometry (fair coverage comparison).
    """
    lon = np.asarray(lon).ravel(); lat = np.asarray(lat).ravel()
    val = np.asarray(val).ravel()
    m = (np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
         & (lon >= W) & (lon < E) & (lat >= S) & (lat < N))
    nlon = max(1, int(round((E - W) / deg)))
    nlat = max(1, int(round((N - S) / deg)))
    ci = np.clip(np.floor((lon[m] - W) / deg).astype(int), 0, nlon - 1)
    cj = np.clip(np.floor((lat[m] - S) / deg).astype(int), 0, nlat - 1)
    flat = cj * nlon + ci
    ssum = np.bincount(flat, weights=val[m], minlength=nlat * nlon)
    scnt = np.bincount(flat, minlength=nlat * nlon)
    with np.errstate(invalid="ignore", divide="ignore"):
        arr = np.where(scnt > 0, ssum / scnt, np.nan).reshape(nlat, nlon)
    lon_edges = W + np.arange(nlon + 1) * deg
    lat_edges = S + np.arange(nlat + 1) * deg
    return np.ma.masked_invalid(arr), lon_edges, lat_edges


def make_fig_coverage_resolution(b_pre, b_tuned, out_base,
                                 pre_label="Pre-trained WindFlow",
                                 tuned_label="Sonde-tuned WindFlow"):
    """Coverage vs cell size: the WindFlow-vs-Carr comparison is resolution
    dependent. WindFlow (dense) fills more small cells; Carr's uniform ~32 km
    sampling catches up / overtakes as cells coarsen. One panel, one crossover.
    """
    degs = np.array([0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0])
    km = degs * 111.0
    pre = [coverage_at(b_pre, d) for d in degs]
    tun = [coverage_at(b_tuned, d) for d in degs]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(km, [c["raft"] * 100 for c in tun], "o-", color="#1b6ca8",
            lw=2.0, ms=5, label=f"{tuned_label} (QA)")
    ax.plot(km, [c["raft"] * 100 for c in pre], "s--", color="#6ba3c9",
            lw=1.6, ms=4, label=f"{pre_label} (QA)")
    ax.plot(km, [c["carr"] * 100 for c in pre], "^-", color="#b5482f",
            lw=2.0, ms=5, label="Carr NCC (ungated)")
    ax.set_xlabel("grid cell size  [km]")
    ax.set_ylabel("grid boxes with a wind  [% of overlap union]")
    ax.set_title("Product coverage vs grid resolution")
    ax.set_ylim(0, 100); ax.set_xlim(km.min(), km.max())
    ax.axvline(32.0, color="0.6", lw=1.0, ls=":")
    ax.text(32.5, 4, "Carr\nsampling", fontsize=8, color="0.4", va="bottom")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_base}.{ext}", dpi=300)
    plt.close(fig)
    logger.info("wrote %s.{png,pdf,svg}", out_base)


# ---- Figure A2: pre-trained + fine-tuned + Carr map rows + scatter ----------
def make_fig_maps_two_models(b_pre, b_tuned, out_base,
                             pre_label="Pre-trained WindFlow",
                             tuned_label="Sonde-tuned WindFlow", cover_deg=0.3):
    """4-row plate contrasting two models against Carr on the same scene:

        row 1  pre-trained WindFlow dense u / v / h maps
        row 2  fine-tuned WindFlow dense u / v / h maps  (denser QA coverage)
        row 3  Carr sparse-point u / v / h maps
        row 4  pre-trained-vs-Carr collocation density scatter

    Both model rows share one u/v color range so the coverage/consistency gain
    from fine-tuning is directly comparable.
    """
    geo = _geo_from_bundle(b_pre)             # same scene/sat for both bundles
    ext_m = list(b_pre["ext_m"]); extent = list(b_pre["extent"])
    uv_vmax = max(float(b_pre["uv_vmax"]), float(b_tuned["uv_vmax"]))
    uv_norm = Normalize(-uv_vmax, uv_vmax)
    # Row-4 consistency scatter uses the PRE-TRAINED model vs Carr.
    b_sc, sc_label = b_pre, pre_label
    uv_pairs = np.abs(np.concatenate([b_sc["c_u"], b_sc["r_u"],
                                      b_sc["c_v"], b_sc["r_v"]]))
    uv_pairs = uv_pairs[np.isfinite(uv_pairs)]
    scatter_lim = max(20.0, float(np.ceil(np.percentile(uv_pairs, 99.5) / 5) * 5))
    cols = [("u", "$u$  (m s$^{-1}$)", UV_CMAP, uv_norm),
            ("v", "$v$  (m s$^{-1}$)", UV_CMAP, uv_norm),
            ("h", "$h$  (km)", H_CMAP, H_NORM)]
    model_rows = [(b_pre, pre_label), (b_tuned, tuned_label)]
    clon, clat = b_pre["carr_lon"], b_pre["carr_lat"]
    letters = "abcdefghijkl"

    # Common-grid rendering: bin dense WindFlow AND sparse Carr onto the same
    # ~cover_deg lat/lon cells so both fill identical cell geometry (fair
    # coverage comparison; each Carr vector paints its true ~32 km footprint).
    W, E, S, N = [float(x) for x in extent]
    latlon = {id(b): _raft_latlon(b["h_map"].shape, int(b["ds_factor"]))
              for b, _ in model_rows}

    def _tag(ax, k):
        ax.text(0.0, 1.02, f"({letters[k]})", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=9, fontweight="bold")

    fig = plt.figure(figsize=(6.0, 7.7))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1.0, 1.0, 1.0, 1.05],
                  hspace=0.28, wspace=0.14, left=0.085, right=0.99,
                  top=0.915, bottom=0.052)

    for j, (key, label, cmap, norm) in enumerate(cols):
        top_ax = None
        for i, (b, rlab) in enumerate(model_rows):          # rows 1-2: model maps
            ax = _map_ax(fig, gs[i, j], geo, extent)
            lat_b, lon_b = latlon[id(b)]
            arr, xe, ye = _grid_to_cells(lon_b, lat_b, b[f"{key}_map"],
                                         W, E, S, N, cover_deg)
            im = ax.pcolormesh(xe, ye, arr, cmap=cmap, norm=norm,
                               transform=PC, shading="flat", zorder=1)
            if i == 0:
                ax.set_title(label, fontsize=10)
                top_ax = ax
            if j == 0:
                ax.text(-0.07, 0.5, rlab, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=8.5, fontweight="bold")
            if j == 2:                                    # h panel: coverage stats
                c33, c55 = coverage_at(b, 0.3), coverage_at(b, 0.5)
                ax.text(0.02, 0.02,
                        f"boxes w/ wind\n33 km: {c33['raft']:.0%}\n55 km: {c55['raft']:.0%}\n"
                        f"px fill: {c33['pix_fill']:.0%}",
                        transform=ax.transAxes, va="bottom", ha="left", fontsize=6.0,
                        color="0.15", linespacing=1.15,
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.2))
            _tag(ax, i * 3 + j)

        axc = _map_ax(fig, gs[2, j], geo, extent)           # row 3: Carr, same grid
        arr, xe, ye = _grid_to_cells(clon, clat, b_pre[f"carr_{key}"],
                                     W, E, S, N, cover_deg)
        axc.pcolormesh(xe, ye, arr, cmap=cmap, norm=norm,
                       transform=PC, shading="flat", zorder=1)
        if j == 0:
            axc.text(-0.07, 0.5, "Carr NCC", transform=axc.transAxes, rotation=90,
                     va="center", ha="center", fontsize=8.5, fontweight="bold")
        if j == 2:                                        # h panel: coverage stats
            c33, c55 = coverage_at(b_pre, 0.3), coverage_at(b_pre, 0.5)
            axc.text(0.02, 0.02,
                     f"boxes w/ wind\n33 km: {c33['carr']:.0%}\n55 km: {c55['carr']:.0%}",
                     transform=axc.transAxes, va="bottom", ha="left", fontsize=6.0,
                     color="0.15", linespacing=1.15,
                     bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.2))
        _tag(axc, 6 + j)

        # one colorbar per column, under the three map rows
        p_top = top_ax.get_position(); p_bot = axc.get_position()
        cax = fig.add_axes([p_top.x0, p_bot.y0 - 0.014, p_top.width, 0.008])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                          extend="max" if key == "h" else "both")
        if key == "h":
            ticks = np.linspace(0, 16000, 5)
            cb.set_ticks(ticks); cb.set_ticklabels([f"{int(t/1000)}" for t in ticks])

        axs = fig.add_subplot(gs[3, j])                     # row 4: pre-trained vs Carr
        if key == "h":
            _scatter(axs, b_sc["c_h"] / 1000, b_sc["r_h"] / 1000, 0, 16, unit="km")
            axs.set_xticks(np.arange(0, 17, 4)); axs.set_yticks(np.arange(0, 17, 4))
            axs.set_xlabel("Carr $h$ (km)")
            axs.set_ylabel(f"{sc_label} $h$ (km)")
        else:
            _scatter(axs, b_sc[f"c_{key}"], b_sc[f"r_{key}"],
                     -scatter_lim, scatter_lim, unit="m s$^{-1}$")
            axs.set_xlabel(f"Carr ${key}$ (m s$^{{-1}}$)")
            axs.set_ylabel(f"{sc_label} ${key}$ (m s$^{{-1}}$)")
        _tag(axs, 9 + j)

    fig.suptitle("Pre-trained vs sonde-tuned WindFlow consistency with Carr NCC "
                 "(GOES-16 × GOES-18 overlap)", fontsize=10.5, y=0.99)
    fig.text(0.5, 0.958,
             f"maps binned to a common {cover_deg:g}° (~{cover_deg*111:.0f} km) grid   ·   "
             f"{pre_label}: {int(b_pre['n_raft_qa']):,} QA px   ·   "
             f"{tuned_label}: {int(b_tuned['n_raft_qa']):,} QA px   ·   "
             f"Carr NCC: {int(b_pre['n_carr']):,} vectors",
             ha="center", va="top", fontsize=7.0, color="0.3")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_base}.{ext}", dpi=300)
    plt.close(fig)
    logger.info("wrote %s.{png,pdf,svg}", out_base)


# ---- Driver -----------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DATA_DIR_DEFAULT)
    p.add_argument("--scene-dir", default=None,
                   help="5-scene .npy dir (default <data-dir>/cache)")
    p.add_argument("--carr-nc", default=None, help="Carr retrieval NetCDF")
    p.add_argument("--parallax", default=None,
                   help="parallax_<a>_<b>.npz (default <data-dir>/zarrs/...)")
    p.add_argument("--ckpt", default=None,
                   help="model checkpoint (default keyed off --model-tag)")
    p.add_argument("--model-tag", default="tuned",
                   help="which retrieval to use / cache as retr_<tag>_<band>.nc")
    p.add_argument("--model-label", default=None,
                   help="display label for the model (default keyed off --model-tag)")
    p.add_argument("--sat-a", default="goes16", choices=list(SATELLITE_CONFIGS))
    p.add_argument("--sat-b", default="goes18", choices=list(SATELLITE_CONFIGS))
    p.add_argument("--band", default="C14")
    p.add_argument("--time", default="2025-01-08T19:00")
    p.add_argument("--dt-minutes", type=float, default=10.0)
    p.add_argument("--colloc-radius", type=int, default=COLLOC_RADIUS_DEFAULT)
    p.add_argument("--ds-factor", type=int, default=DS_FACTOR_DEFAULT,
                   help="display downsample factor for the dense maps")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lowmem", action="store_true")
    p.add_argument("--n-iter", type=int, default=3)
    p.add_argument("--output-dir", default=str(BASE / "figures"))
    p.add_argument("--from-cache", action="store_true",
                   help="use cached retrieval NetCDF (skip GPU inference)")
    p.add_argument("--from-npz", action="store_true",
                   help="render from an existing display bundle (no data/GPU)")
    p.add_argument("--tuned-bundle", default=None,
                   help="add a fine-tuned-model map row: path to its "
                        "carr_consistency_<band>.npz bundle. The consistency "
                        "figure becomes a 4-row pre-trained/fine-tuned/Carr plate.")
    p.add_argument("--cover-grid-deg", type=float, default=0.3,
                   help="common lat/lon cell size (deg) for the two-model plate; "
                        "both WindFlow and Carr are binned onto this grid "
                        "(default 0.3 deg ~= Carr's ~32 km sampling).")
    args = p.parse_args()

    # Checkpoints shipped with the repo. "pretrained" is the exact init the
    # sonde-tuned model was fine-tuned from (its weights are also embedded in
    # the tuned checkpoint as raft_ref.*), so the two rows are comparable.
    # Override either with --ckpt.
    DEFAULT_CKPT = {"tuned": BASE / "checkpoints" / "windflow.raft.sonde-tuned.ckpt",
                    "pretrained": BASE / "checkpoints" / "windflow.raft.init-ep254.ckpt"}
    DEFAULT_LABEL = {"tuned": "Sonde-tuned WindFlow", "pretrained": "Pre-trained WindFlow"}
    tag = args.model_tag

    plt.style.use(str(STYLE))
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = cache_dir / f"carr_consistency_{tag}_{args.band}.npz"
    out_maps = out_dir / f"fig_carr_consistency_{tag}_{args.band}"
    out_cov = out_dir / f"fig_carr_coverage_{tag}_{args.band}"

    if args.from_npz:
        if not bundle_path.exists():
            raise FileNotFoundError(f"--from-npz set but {bundle_path} missing.")
        logger.info("loading display bundle %s", bundle_path)
        b = dict(np.load(bundle_path, allow_pickle=True))
        model_label = args.model_label or (
            str(b["model_label"]) if "model_label" in b
            else DEFAULT_LABEL.get(tag, tag))
    else:
        model_label = args.model_label or DEFAULT_LABEL.get(tag, tag)
        data_dir = Path(args.data_dir)
        scene_dir = Path(args.scene_dir) if args.scene_dir else data_dir / "cache"
        sat = SATELLITE_CONFIGS[args.sat_a]
        ckpt = args.ckpt or str(DEFAULT_CKPT.get(tag, DEFAULT_CKPT["tuned"]))
        parallax = args.parallax or str(
            data_dir / "zarrs" / f"parallax_{args.sat_a}_{args.sat_b}.npz")
        t0 = np.datetime64(args.time)

        carr_nc = args.carr_nc
        if carr_nc is None:
            cands = sorted((data_dir / "carr_data").glob("*.nc"))
            if not cands:
                raise FileNotFoundError(
                    f"No Carr NetCDF in {data_dir/'carr_data'}; pass --carr-nc")
            carr_nc = str(cands[0])
        logger.info("Carr NetCDF: %s", carr_nc)
        carr = load_carr_data(carr_nc)

        ds = load_or_infer(
            tag, ckpt, cache_dir, args.from_cache, args.band,
            scene_dir=scene_dir, sat_a=sat, sat_b=SATELLITE_CONFIGS[args.sat_b],
            t0=t0, dt_minutes=args.dt_minutes, parallax_path=parallax,
            device=args.device, tile_size=args.tile_size, overlap=args.overlap,
            batch_size=args.batch_size, lowmem=args.lowmem, n_iter=args.n_iter)

        b = build_bundle(ds, carr, sat, PANEL_A_EXTENT, args.colloc_radius,
                         args.ds_factor, bundle_path, model_label=model_label)

    if args.tuned_bundle:
        tb = Path(args.tuned_bundle)
        if not tb.exists():
            raise FileNotFoundError(f"--tuned-bundle {tb} missing.")
        b_tuned = dict(np.load(tb, allow_pickle=True))
        tuned_label = (str(b_tuned["model_label"])
                       if b_tuned.get("model_label") not in (None, np.array(None))
                       and str(b_tuned.get("model_label")) != "None"
                       else "Sonde-tuned WindFlow")
        make_fig_maps_two_models(b, b_tuned, str(out_maps),
                                 pre_label=model_label, tuned_label=tuned_label,
                                 cover_deg=args.cover_grid_deg)
        make_fig_coverage_resolution(
            b, b_tuned, str(out_dir / f"fig_coverage_resolution_{args.band}"),
            pre_label=model_label, tuned_label=tuned_label)
    else:
        make_fig_maps_scatter(b, str(out_maps), model_label=model_label,
                              cover_deg=args.cover_grid_deg)
    make_fig_two_maps(b, str(out_cov), model_label=model_label)
    logger.info("done.")


if __name__ == "__main__":
    main()
