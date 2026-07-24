#!/usr/bin/env python
"""Teacher (cross-satellite stereo) vs Student (single-satellite) wind retrieval,
both validated against HELD-OUT IGRA radiosondes (test months 2025-10/11).

Both retrievals are collocated to the SAME sondes and inner-joined on
(band, station_idx, scene_time) so they are scored on the EXACT same points.
The reference is IGRA (u,v) interpolated in z to EACH retrieval's OWN cloud-top
height (Eq. 8 bracketing scheme).  QA gate = the eval_from_parquet standard
(chi2<=0.2 + sigma_h<=5km + height-gradient + speed), applied identically.

Layout mirrors the fine-tuning figure: 2x3 hexbin (rows teacher/student;
cols u, v, speed; retrieved vs sonde) + a per-band RMSVD bar row.

    # on ADAPT (reads full-disk zarrs + parquet), dump the small matched table:
    python scripts/fig_student_teacher_igra.py \
        --teacher-cache-dir .../quad_test/cache --student-dir .../student_eval_mb \
        --parquet .../igra_all_collocation.parquet \
        --dump-matched st_igra_matched.parquet --out fig_student_teacher_igra.png
    # then locally, iterate styling with no data deps:
    python scripts/fig_student_teacher_igra.py --from-matched st_igra_matched.parquet \
        --out figures/fig_student_teacher_igra.png
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

import eval_from_parquet as efp  # noqa: E402
from eval_from_parquet import _build_qa_mask, neighborhood_median  # noqa: E402
from fig_finetuning_improvement import (  # noqa: E402
    _interp_to_height, apply_holdout, _lim, TEST_MONTHS,
)
from stereo_winds.validation.metrics import rmsvd, speed_bias, correlation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STYLE = BASE / "figures" / "paper.mplstyle"
COLOR_TEACHER = "#e8743b"    # cross-satellite stereo
COLOR_STUDENT = "#2a9d8f"    # single-satellite


def collocate_zarr(path, band, df, min_scenes=2):
    """Collocate one retrieval zarr to IGRA; Eq.8 interp to its OWN height.

    Works on any full-disk zarr (teacher per-month, or student single file for
    both months); month is tagged from each scene time.  Mirrors the
    fig_finetuning_improvement.collocate loop but takes an explicit path.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("  [%s] missing %s", band, path)
        return []
    ds = xr.open_zarr(path, consolidated=False)
    if "time" not in ds.dims:
        ds = ds.expand_dims("time")
    if ds.sizes["time"] < min_scenes:
        logger.warning("  [%s] only %d scene(s) -> skip", band, ds.sizes["time"])
        return []
    times = ds.time.values
    dloc = df.copy()
    dloc["goes_time"] = pd.to_datetime(dloc["goes_time"]).values.astype("datetime64[ns]")
    pq_times = dloc["goes_time"].values

    out = []
    for ti, t in enumerate(times):
        t_ns = np.datetime64(t).astype("datetime64[ns]")
        dt = np.abs(pq_times - t_ns).astype("timedelta64[s]").astype(np.int64)
        nearby = dloc[dt <= 6 * 3600]
        if len(nearby) == 0:
            continue
        nearest = nearby.iloc[np.abs(nearby["goes_time"].values - t_ns).argmin()]["goes_time"]
        df_time = nearby[nearby["goes_time"] == nearest]
        sl = ds.isel(time=ti).load()
        qa = _build_qa_mask(sl)
        u_grid = sl["u_wind"].values; v_grid = sl["v_wind"].values
        h_grid = sl["cloud_top_height"].values
        for (sidx,), grp in df_time.groupby(["station_idx"]):
            if len(grp) < 2:
                continue
            prof = grp.sort_values("pressure_hpa", ascending=False)
            sh = prof["height_m"].values.astype(np.float64)
            su = prof["u"].values.astype(np.float64)
            sv = prof["v"].values.astype(np.float64)
            valid = np.isfinite(sh) & np.isfinite(su) & np.isfinite(sv)
            if valid.sum() < 2:
                continue
            r = int(prof["row_19"].iloc[0]); c = int(prof["col_19"].iloc[0])
            if not (0 <= r < h_grid.shape[0] and 0 <= c < h_grid.shape[1]):
                continue
            ra = np.array([r]); ca = np.array([c])
            h_s = neighborhood_median(h_grid, ra, ca, qa)[0]
            u_s = neighborhood_median(u_grid, ra, ca, qa)[0]
            v_s = neighborhood_median(v_grid, ra, ca, qa)[0]
            if not (np.isfinite(h_s) and np.isfinite(u_s) and np.isfinite(v_s)):
                continue
            ref = _interp_to_height(sh, su, sv, valid, h_s)
            if ref is None:
                continue
            out.append(dict(band=band, station_idx=int(sidx),
                            t=int(t_ns.astype("datetime64[ns]").astype(np.int64)),
                            month=str(t)[:7], u_hat=float(u_s), v_hat=float(v_s),
                            h_hat=float(h_s), u_ref=float(ref[0]), v_ref=float(ref[1])))
    logger.info("  [%s] %s: %d scenes -> %d collocations", band, path.name,
                len(times), len(out))
    return out


def build_matched(args):
    df = pd.read_parquet(args.parquet)
    print(f"=== IGRA parquet: {len(df)} rows, {df['station_idx'].nunique()} stations")
    months_ym = [m.replace("-", "") for m in args.months]

    print("=== Collocating TEACHER (cross-sat stereo) ===")
    trows = []
    for b in args.bands:
        for ym in months_ym:
            p = Path(args.teacher_cache_dir) / f"{args.teacher_label}_{b}_{ym}_iter{args.n_iter}.zarr"
            trows.extend(collocate_zarr(p, b, df))
    dteach = pd.DataFrame(trows)

    print("=== Collocating STUDENT (single-sat) ===")
    srows = []
    for b in args.bands:
        p = Path(args.student_dir) / args.student_tmpl.format(band=b)
        srows.extend(collocate_zarr(p, b, df))
    dstud = pd.DataFrame(srows)

    print("=== Holdout accounting (test months) ===")
    dteach = apply_holdout(dteach, "teacher")
    dstud = apply_holdout(dstud, "student")
    if len(dteach) == 0 or len(dstud) == 0:
        print("!! one side has no collocations"); return None

    key = ["band", "station_idx", "t"]
    merged = dteach.merge(dstud, on=key, suffixes=("_t", "_s"))
    print(f"=== inner join on {key}: N_teacher={len(dteach)} N_student={len(dstud)} "
          f"N_common={len(merged)}")
    return merged


def block(m, tag):
    return (m[f"u_hat_{tag}"].values, m[f"v_hat_{tag}"].values,
            m[f"u_ref_{tag}"].values, m[f"v_ref_{tag}"].values)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher-cache-dir")
    p.add_argument("--teacher-label", default="hreg1s75")
    p.add_argument("--student-dir")
    p.add_argument("--student-tmpl", default="student_quad_ab_{band}.zarr")
    p.add_argument("--parquet")
    p.add_argument("--bands", nargs="+", default=["C08", "C09", "C10", "C12", "C14"])
    p.add_argument("--months", nargs="+", default=TEST_MONTHS)
    p.add_argument("--n-iter", type=int, default=3)
    p.add_argument("--chi2-max", type=float, default=0.2,
                   help="QA chi2 gate applied identically to both retrievals")
    p.add_argument("--min-common", type=int, default=50)
    p.add_argument("--dump-matched", default=None)
    p.add_argument("--from-matched", default=None)
    p.add_argument("--out", default=str(BASE / "figures" / "fig_student_teacher_igra.png"))
    args = p.parse_args()

    def _read(path):
        return pd.read_csv(path) if str(path).endswith(".csv") else pd.read_parquet(path)

    def _write(df, path):
        (df.to_csv(path, index=False) if str(path).endswith(".csv")
         else df.to_parquet(path))

    if args.from_matched:
        merged = _read(args.from_matched)
        print(f"=== loaded matched table: N={len(merged)}")
    else:
        efp.QA["chi2_max"] = args.chi2_max     # gate applied inside _build_qa_mask
        print(f"=== QA chi2_max = {efp.QA['chi2_max']}")
        merged = build_matched(args)
        if merged is None or len(merged) == 0:
            print("!! no common collocations"); return
        if args.dump_matched:
            _write(merged, args.dump_matched)
            # also write a portable CSV sibling for dependency-free local re-plot
            if not str(args.dump_matched).endswith(".csv"):
                _write(merged, str(Path(args.dump_matched).with_suffix(".csv")))
            print(f"=== dumped {args.dump_matched}")

    uht, vht, urt, vrt = block(merged, "t")
    uhs, vhs, urs, vrs = block(merged, "s")

    def agg_rmsvd(m, tag):
        u_h, v_h, u_r, v_r = block(m, tag)
        return rmsvd(u_h, v_h, u_r, v_r)

    rv_t_all, rv_s_all = agg_rmsvd(merged, "t"), agg_rmsvd(merged, "s")

    print("=== Per-band (common set) ===")
    band_rows = []
    for b in args.bands:
        mb = merged[merged["band"] == b]
        if len(mb) == 0:
            continue
        rt, rs = agg_rmsvd(mb, "t"), agg_rmsvd(mb, "s")
        band_rows.append(dict(band=b, n=len(mb), rmsvd_t=rt, rmsvd_s=rs,
                              gap=100.0 * (rs - rt) / rt))
        print(f"  {b}: N={len(mb):5d}  RMSVD teacher={rt:5.2f}  student={rs:5.2f}"
              + ("   [<min-common, excluded from bars]" if len(mb) < args.min_common else ""))
    bar_df = pd.DataFrame([r for r in band_rows if r["n"] >= args.min_common])
    dropped = [r["band"] for r in band_rows if r["n"] < args.min_common]

    print(f"=== AGGREGATE (common N={len(merged)}) ===")
    print(f"  RMSVD teacher={rv_t_all:.3f}  student={rv_s_all:.3f}")
    print(f"  speed_bias teacher={speed_bias(uht,vht,urt,vrt):+.2f}  "
          f"student={speed_bias(uhs,vhs,urs,vrs):+.2f}")

    # ---- Figure -----------------------------------------------------------
    if STYLE.exists():
        plt.style.use(str(STYLE))
    lim_u, _ = _lim(np.concatenate([uht, urt, uhs, urs]))
    lim_v, _ = _lim(np.concatenate([vht, vrt, vhs, vrs]))
    sp_all = np.concatenate([np.hypot(uht, vht), np.hypot(urt, vrt),
                             np.hypot(uhs, vhs), np.hypot(urs, vrs)])
    lim_s = np.ceil((float(np.nanmax(sp_all)) + 1e-6) / 5) * 5

    fig = plt.figure(figsize=(12.5, 12.0))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.85],
                  hspace=0.30, wspace=0.28, left=0.08, right=0.90, top=0.92, bottom=0.07)
    rows = [("Teacher — cross-satellite stereo", uht, vht, urt, vrt),
            ("Student — single-satellite", uhs, vhs, urs, vrs)]
    col_specs = [("u", "$u$", -lim_u, lim_u, False),
                 ("v", "$v$", -lim_v, lim_v, False),
                 ("speed", "wind speed", 0, lim_s, True)]
    hbs = []
    axgrid = [[None] * 3, [None] * 3]
    for i, (rlabel, u_h, v_h, u_r, v_r) in enumerate(rows):
        sp_h, sp_r = np.hypot(u_h, v_h), np.hypot(u_r, v_r)
        for j, (ckey, clabel, lo, hi, is_sp) in enumerate(col_specs):
            ax = fig.add_subplot(gs[i, j]); axgrid[i][j] = ax
            x, y = ({"u": (u_r, u_h), "v": (v_r, v_h)}.get(ckey, (sp_r, sp_h)))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=1)
            hb = ax.hexbin(x, y, gridsize=40, cmap="magma", bins="log", mincnt=1,
                           extent=(lo, hi, lo, hi))
            hbs.append(hb)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
            if is_sp:
                stat = (f"N={len(x):,}\nbias={speed_bias(u_h, v_h, u_r, v_r):+.2f}\n"
                        f"RMSVD={rmsvd(u_h, v_h, u_r, v_r):.2f}\nr={correlation(sp_h, sp_r):.2f}")
            else:
                bias = float(np.mean(y - x)); rmse = float(np.sqrt(np.mean((y - x) ** 2)))
                stat = f"N={len(x):,}\nbias={bias:+.2f}\nRMSD={rmse:.2f}\nr={correlation(y, x):.2f}"
            ax.text(0.04, 0.96, stat, transform=ax.transAxes, va="top", ha="left",
                    fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9))
            ax.set_xlabel(f"Radiosonde {clabel} (m s$^{{-1}}$)")
            ax.set_ylabel(f"Retrieved {clabel} (m s$^{{-1}}$)")
            if i == 0:
                ax.set_title(clabel, fontsize=11)
            if j == 0:
                ax.text(-0.30, 0.5, rlabel, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10.5, fontweight="bold")
    vmax = max(hb.get_array().max() for hb in hbs)
    for hb in hbs:
        hb.set_norm(LogNorm(vmin=1, vmax=vmax))
    fig.canvas.draw()
    p_top = axgrid[0][2].get_position(); p_bot = axgrid[1][2].get_position()
    cax = fig.add_axes([p_top.x1 + 0.018, p_bot.y0, 0.013, p_top.y1 - p_bot.y0])
    fig.colorbar(hbs[0], cax=cax, orientation="vertical").set_label("count")

    # ---- Bar panel --------------------------------------------------------
    axb = fig.add_subplot(gs[2, :])
    if len(bar_df):
        x = np.arange(len(bar_df)); w = 0.38
        axb.bar(x - w / 2, bar_df["rmsvd_t"], w, label="Teacher (stereo)", color=COLOR_TEACHER)
        axb.bar(x + w / 2, bar_df["rmsvd_s"], w, label="Student (single-sat)", color=COLOR_STUDENT)
        for xi, row in zip(x, bar_df.itertuples()):
            top = max(row.rmsvd_t, row.rmsvd_s)
            axb.text(xi, top + 0.15, f"{row.gap:+.0f}%\n(N={row.n})", ha="center",
                     va="bottom", fontsize=8, color="0.3")
        axb.set_xticks(x); axb.set_xticklabels(bar_df["band"])
        axb.set_ylabel("RMSVD (m s$^{-1}$)")
        axb.set_ylim(0, float(bar_df[["rmsvd_t", "rmsvd_s"]].values.max()) * 1.25)
        axb.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0.0, 1.0))
        axb.set_title("Per-band RMSVD vs held-out radiosondes "
                      "(label = student excess over teacher)", fontsize=11)
    if dropped:
        axb.text(0.01, 0.97, f"excluded (N<{args.min_common}): {', '.join(dropped)}",
                 transform=axb.transAxes, va="top", fontsize=8, color="0.4")

    fig.suptitle(
        f"Single-satellite student vs cross-satellite teacher, vs held-out IGRA radiosondes  ·  "
        f"aggregate RMSVD  teacher {rv_t_all:.2f}  /  student {rv_s_all:.2f} m s$^{{-1}}$  "
        f"($\\chi^2\\!\\leq\\!{args.chi2_max:g}$,  N={len(merged):,})",
        fontsize=12.5, fontweight="bold", y=0.975)
    for ext in ("png", "pdf"):
        fig.savefig(str(Path(args.out).with_suffix(f".{ext}")), dpi=300)
        print(f"=== wrote {Path(args.out).with_suffix('.' + ext)}")


if __name__ == "__main__":
    main()
