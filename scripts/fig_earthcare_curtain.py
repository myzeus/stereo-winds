#!/usr/bin/env python
"""Figure: EarthCARE CPR reflectivity curtain overlaid with stereo (teacher)
and single-satellite (student) retrieved cloud-top winds, during a deep-
convection / wind-shear event.

The CPR radar curtain (along-track distance x height, dBZ) is the independent
vertical view of the cloud field.  Along the EarthCARE nadir track we sample
each retrieval's cloud-top wind (u, v) and plot it as a wind barb *placed at
its own retrieved cloud-top height* -- so the barbs ride the cloud tops and
their change along-track / with height is the wind shear.

    python scripts/fig_earthcare_curtain.py \
        --granule ECA_..._CPR_NOM_1B_...h5 \
        --teacher-nc teacher.nc --student-nc student.nc \
        --lat-range -5 20 --out figures/fig_earthcare_curtain.png
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO))
from stereo_winds.config import GOES19_CONFIG
from stereo_winds import navigation as nav
from stereo_winds.validation.earthcare_curtain import load_cpr_curtain

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

COLOR_TEACHER = "#e8743b"   # orange (stereo, cross-satellite)
COLOR_STUDENT = "#2a9d8f"   # teal (single-satellite)
COLOR_AMV = "#4393c3"       # steel blue (operational NOAA AMV)
COLOR_ERA5 = "#6a51a3"      # purple (ERA5 reanalysis)


def qa_mask(u, v, h, chi2, qf, chi2_max):
    spd = np.sqrt(u ** 2 + v ** 2)
    return (
        (qf > 0) & np.isfinite(h) & np.isfinite(u) & np.isfinite(v)
        & (chi2 <= chi2_max) & (h >= 800.0) & (h <= 20000.0) & (spd <= 120.0)
    )


def sample_along_track(ds, lat, lon, chi2_max=1.0, box=2):
    """Neighborhood-median sample of retrieval u/v/h along an EC track.

    Returns u, v (m/s), h (m) arrays (NaN where no valid pixels)."""
    u2 = ds["u_wind"].values
    v2 = ds["v_wind"].values
    h2 = ds["cloud_top_height"].values
    chi = ds["chi_squared"].values if "chi_squared" in ds else np.zeros_like(h2)
    qf = ds["quality_flag"].values if "quality_flag" in ds else np.ones_like(h2)
    H, W = h2.shape

    x, y = nav.geodetic_to_fixed_grid(lat, lon, GOES19_CONFIG, 8000.0)
    col, row = nav.scanning_angle_to_pixel(x, y, GOES19_CONFIG)
    n = lat.size
    uu = np.full(n, np.nan); vv = np.full(n, np.nan); hh = np.full(n, np.nan)
    for i in range(n):
        c, r = col[i], row[i]
        if not (np.isfinite(c) and np.isfinite(r)):
            continue
        c, r = int(round(c)), int(round(r))
        if not (box <= r < H - box and box <= c < W - box):
            continue
        sl = (slice(r - box, r + box + 1), slice(c - box, c + box + 1))
        m = qa_mask(u2[sl], v2[sl], h2[sl], chi[sl], qf[sl], chi2_max)
        if m.sum() >= 1:
            uu[i] = np.median(u2[sl][m])
            vv[i] = np.median(v2[sl][m])
            hh[i] = np.median(h2[sl][m])
    return uu, vv, hh


def collocate_amv(track_lat, track_lon, track_dist, amv, corridor_km=75.0,
                  chi2=None):
    """Snap AMV points to the nadir track: keep those within corridor_km of the
    track, returning their along-track distance + height + wind (per band)."""
    from scipy.spatial import cKDTree
    lat0 = float(np.nanmean(track_lat))
    kx = 111.32 * np.cos(np.radians(lat0)); ky = 110.57
    tree = cKDTree(np.column_stack([track_lon * kx, track_lat * ky]))
    out = []
    bands = amv.get("band")
    nb = int(bands.max()) + 1 if bands is not None else 1
    for bi in range(nb):
        m = (bands == bi) if bands is not None else np.ones(amv["lat"].size, bool)
        if m.sum() == 0:
            out.append((np.array([]),) * 4); continue
        pts = np.column_stack([amv["lon"][m] * kx, amv["lat"][m] * ky])
        d, idx = tree.query(pts, k=1)
        keep = d <= corridor_km
        out.append((track_dist[idx[keep]], amv["h"][m][keep],
                    amv["u"][m][keep], amv["v"][m][keep]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granule", default=None,
                    help="CPR_NOM_1B granule (required unless --from-bundle)")
    ap.add_argument("--teacher-ncs", nargs="+", default=None,
                    help="one or more single-band teacher NetCDFs (pooled → multi-level)")
    ap.add_argument("--student-nc", default=None,
                    help="all-band student NetCDF with a 'band' dim (pooled → multi-level)")
    ap.add_argument("--lat-range", nargs=2, type=float, default=None,
                    help="crop track to lat lo hi")
    ap.add_argument("--dist-range", nargs=2, type=float, default=None,
                    help="crop to along-track distance window (km) after loading")
    ap.add_argument("--amv-npz", default=None,
                    help="NOAA AMV npz (lat/lon/u/v/h/band); enables multi-panel mode")
    ap.add_argument("--era5-from", default=None,
                    help="ERA5 curtain npz (e_dp/e_hp/e_up/e_vp along track); adds a 4th panel")
    ap.add_argument("--time", default=None,
                    help="override the title timestamp (e.g. 2025-11-07T21:00) if the "
                         "bundle/granule frame time is missing")
    ap.add_argument("--corridor-km", type=float, default=75.0,
                    help="keep AMVs within this cross-track distance of the nadir track")
    ap.add_argument("--chi2-max", type=float, default=1.0)
    ap.add_argument("--barb-stride", type=int, default=60,
                    help="place a barb every N along-track samples")
    ap.add_argument("--dbz-min", type=float, default=-28.0)
    ap.add_argument("--dbz-max", type=float, default=20.0)
    ap.add_argument("--orient", choices=["horizontal", "vertical", "grid"], default="horizontal",
                    help="multi-panel layout: horizontal (1xN), vertical (Nx1), or grid (2xceil(N/2))")
    ap.add_argument("--dump-bundle", default=None,
                    help="save curtain + sampled winds to an npz for offline re-plotting")
    ap.add_argument("--from-bundle", default=None,
                    help="re-plot from a bundle npz (no NetCDF/granule/gh061 needed)")
    ap.add_argument("--out", default=str(REPO / "figures/fig_earthcare_curtain.png"))
    args = ap.parse_args()

    style = REPO / "figures" / "paper.mplstyle"
    if style.exists():
        plt.style.use(str(style))

    if not args.from_bundle and not (args.granule and args.teacher_ncs and args.student_nc):
        ap.error("need --granule, --teacher-ncs and --student-nc (or use --from-bundle)")

    frame_start = ""
    if args.from_bundle:
        # ---- offline: everything comes from the portable bundle ----
        b = np.load(args.from_bundle, allow_pickle=True)
        dist, hgt, refl = b["dist"], b["hgt"], b["refl"]
        lat, lon = b["lat"], b["lon"]
        frame_start = str(b["frame_start"])
        teacher_bands = [(b["t_u"][i], b["t_v"][i], b["t_h"][i])
                         for i in range(b["t_u"].shape[0])]
        student_bands = [(b["s_u"][i], b["s_v"][i], b["s_h"][i])
                         for i in range(b["s_u"].shape[0])]
        amv_pts = None
        if "a_dp" in b.files and b["a_dp"].size:
            ab = b["a_band"]
            amv_pts = [(b["a_dp"][ab == k], b["a_hp"][ab == k],
                        b["a_up"][ab == k], b["a_vp"][ab == k])
                       for k in range(int(ab.max()) + 1)]
        log.info(f"loaded bundle {args.from_bundle}: curtain {refl.shape}")
    else:
        lr = tuple(args.lat_range) if args.lat_range else None
        cur = load_cpr_curtain(args.granule, lat_range=lr)
        if args.dist_range:
            d = cur["distance_km"].values
            keep = (d >= args.dist_range[0]) & (d <= args.dist_range[1])
            cur = cur.isel(along_track=keep)
        lat = cur["lat"].values; lon = cur["lon"].values
        dist = cur["distance_km"].values
        hgt = cur["height"].values
        refl = cur["reflectivity"].values  # (along_track, height) dBZ
        frame_start = str(cur.attrs.get("frame_start", ""))
        log.info(f"curtain: {refl.shape} rays, lat [{lat.min():.1f},{lat.max():.1f}] "
                 f"dist {dist.max():.0f} km")

        teacher_bands = []
        for nc in args.teacher_ncs:
            d = xr.open_dataset(nc)
            if "time" in d.dims:
                d = d.isel(time=0)
            teacher_bands.append(sample_along_track(d.squeeze(), lat, lon, args.chi2_max))
        student_bands = []
        dss = xr.open_dataset(args.student_nc)
        if "band" in dss.dims:
            for bi in range(dss.sizes["band"]):
                student_bands.append(
                    sample_along_track(dss.isel(band=bi), lat, lon, args.chi2_max))
        else:
            student_bands.append(sample_along_track(dss.squeeze(), lat, lon, args.chi2_max))

        amv_pts = None
        if args.amv_npz:
            az = np.load(args.amv_npz)
            amv = {k: az[k] for k in az.files}
            amv_pts = collocate_amv(lat, lon, dist, amv, corridor_km=args.corridor_km)

        if args.dump_bundle:
            d2 = dict(
                dist=dist, hgt=hgt, refl=refl, lat=lat, lon=lon, frame_start=frame_start,
                t_u=np.array([t[0] for t in teacher_bands]),
                t_v=np.array([t[1] for t in teacher_bands]),
                t_h=np.array([t[2] for t in teacher_bands]),
                s_u=np.array([s[0] for s in student_bands]),
                s_v=np.array([s[1] for s in student_bands]),
                s_h=np.array([s[2] for s in student_bands]),
            )
            if amv_pts is not None:
                d2["a_dp"] = np.concatenate([p[0] for p in amv_pts]) if amv_pts else np.array([])
                d2["a_hp"] = np.concatenate([p[1] for p in amv_pts])
                d2["a_up"] = np.concatenate([p[2] for p in amv_pts])
                d2["a_vp"] = np.concatenate([p[3] for p in amv_pts])
                d2["a_band"] = np.concatenate([np.full(len(p[0]), k, int)
                                               for k, p in enumerate(amv_pts)])
            np.savez_compressed(args.dump_bundle, **d2)
            log.info(f"=== dumped bundle {args.dump_bundle}")

    nt = sum(int(np.isfinite(h).sum()) for _, _, h in teacher_bands)
    ns = sum(int(np.isfinite(h).sum()) for _, _, h in student_bands)
    log.info(f"teacher barbs (all bands): {nt}  student barbs (all bands): {ns}")
    if amv_pts is not None:
        na = sum(len(p[0]) for p in amv_pts)
        log.info(f"AMV points within {args.corridor_km:.0f} km corridor: {na}")

    # ERA5 reanalysis curtain (its own npz; independent of bundle/live mode)
    era5_pts = None
    if args.era5_from:
        ez = np.load(args.era5_from)
        era5_pts = [(ez["e_dp"], ez["e_hp"], ez["e_up"], ez["e_vp"])]
        log.info(f"ERA5 barbs: {ez['e_dp'].size}")

    st = args.barb_stride
    idx = np.arange(0, lat.size, st)
    X, Y = np.meshgrid(dist, hgt, indexing="ij")
    reflm = np.ma.masked_invalid(refl)

    def draw_curtain(ax):
        return ax.pcolormesh(X, Y, reflm, cmap="turbo", vmin=args.dbz_min,
                             vmax=args.dbz_max, shading="auto", zorder=1)

    def draw_bands(ax, band_list, color):
        drew = False
        for (u, v, h) in band_list:
            sel = idx[np.isfinite(h[idx]) & np.isfinite(u[idx])]
            if sel.size == 0:
                continue
            ax.barbs(dist[sel], h[sel] / 1000.0, u[sel], v[sel],
                     length=5.6, linewidth=0.7, color=color, zorder=5,
                     alpha=0.9, barb_increments=dict(half=2.5, full=5, flag=25))
            ax.scatter(dist[sel], h[sel] / 1000.0, s=4, color=color,
                       edgecolor="white", linewidth=0.25, zorder=6)
            drew = True
        return drew

    def draw_points(ax, pts_list, color):
        """AMV points (already sparse) — barb every point, no strided subsample."""
        n = 0
        for (dp, hp, up, vp) in pts_list:
            ok = np.isfinite(hp) & np.isfinite(up)
            if ok.sum() == 0:
                continue
            ax.barbs(dp[ok], hp[ok] / 1000.0, up[ok], vp[ok],
                     length=5.6, linewidth=0.7, color=color, zorder=5,
                     alpha=0.9, barb_increments=dict(half=2.5, full=5, flag=25))
            ax.scatter(dp[ok], hp[ok] / 1000.0, s=4, color=color,
                       edgecolor="white", linewidth=0.25, zorder=6)
            n += int(ok.sum())
        return n

    def style_ax(ax, label, show_x=True, show_y=True):
        ax.set_ylim(0, hgt.max())
        ax.set_xlim(dist.min(), dist.max())
        ax.text(0.015, 0.94, label, transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", va="top", zorder=20,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.95))
        ax.set_ylabel("Height (km)") if show_y else ax.tick_params(labelleft=False)
        ax.set_xlabel("Along-track distance (km)") if show_x else ax.tick_params(labelbottom=False)

    t0 = (args.time or frame_start)[:16]

    if amv_pts is None:
        # legacy single panel: teacher + student overlaid
        fig, ax = plt.subplots(figsize=(11.5, 5.2))
        pm = draw_curtain(ax)
        draw_bands(ax, teacher_bands, COLOR_TEACHER)
        draw_bands(ax, student_bands, COLOR_STUDENT)
        style_ax(ax, "", True, True)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0], [0], color=COLOR_TEACHER, lw=2, label="Stereo teacher"),
                           Line2D([0], [0], color=COLOR_STUDENT, lw=2, label="Single-sat student")],
                  loc="upper right", frameon=True, framealpha=0.9, fontsize=8)
        cb = fig.colorbar(pm, ax=ax, pad=0.01, fraction=0.045)
        cb.set_label("CPR reflectivity (dBZ)")
        ax.set_title(f"EarthCARE CPR curtain + multi-band cloud-top winds  ·  "
                     f"{t0}Z  ·  lat [{lat.min():.0f}, {lat.max():.0f}]", fontsize=10)
        fig.tight_layout()
    else:
        sources = [("(a) Stereo teacher", COLOR_TEACHER, teacher_bands, "bands"),
                   ("(b) Single-sat student", COLOR_STUDENT, student_bands, "bands"),
                   ("(c) NOAA GOES-19 AMV", COLOR_AMV, amv_pts, "points")]
        if era5_pts is not None:
            sources.append(("(d) ERA5 reanalysis", COLOR_ERA5, era5_pts, "points"))
        n = len(sources)
        suptitle = (f"Cloud-top winds vs. EarthCARE CPR reflectivity  ·  {t0}Z  ·  "
                    f"ITCZ deep convection  ·  lat [{lat.min():.0f}, {lat.max():.0f}]")
        if args.orient == "vertical":
            nrow, ncol, figsize = n, 1, (11.5, 3.6 * n)
        elif args.orient == "grid":
            ncol = 2; nrow = int(np.ceil(n / 2)); figsize = (13.0, 4.2 * nrow)
        else:  # horizontal 1xN across the page top
            nrow, ncol, figsize = 1, n, (3.7 * n + 1.4, 4.3)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize,
                                 sharex=(args.orient == "vertical"),
                                 sharey=(args.orient != "vertical"), squeeze=False)
        axflat = axes.ravel()
        pm = None
        for i, (lab, col, data, kind) in enumerate(sources):
            ax = axflat[i]
            rr, cc = divmod(i, ncol)
            pm = draw_curtain(ax)
            (draw_bands if kind == "bands" else draw_points)(ax, data, col)
            style_ax(ax, lab, show_x=(rr == nrow - 1), show_y=(cc == 0))
        for k in range(n, nrow * ncol):
            axflat[k].axis("off")
        if args.orient == "vertical":
            axflat[0].set_title(suptitle, fontsize=11)
            fig.subplots_adjust(left=0.07, right=0.90, top=0.955, bottom=0.06, hspace=0.09)
            cax = fig.add_axes([0.915, 0.06, 0.014, 0.895])
        else:
            fig.suptitle(suptitle, fontsize=10, y=0.995)
            fig.subplots_adjust(left=0.05, right=0.93, top=0.90, bottom=0.10,
                                wspace=0.06, hspace=0.14)
            cax = fig.add_axes([0.94, 0.10, 0.010, 0.80])
        fig.colorbar(pm, cax=cax).set_label("CPR reflectivity (dBZ)")
    for ext in ("png", "pdf"):
        p = Path(args.out).with_suffix(f".{ext}")
        fig.savefig(p, dpi=300)
        log.info(f"=== wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
