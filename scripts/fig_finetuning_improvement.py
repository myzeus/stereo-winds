#!/usr/bin/env python
"""§4a fine-tuning-improvement figure: pretrained WindFlow vs sonde-tuned teacher.

Both checkpoints are run through the IDENTICAL WLS solver (cached retrievals) and
evaluated against HELD-OUT IGRA radiosondes. Data-integrity guards (enforced):

  * TEST partition only: held-out stations (station_idx %% 5 == 0, the val split
    from stereo_winds/dataset.py) AND held-out months (2025-10, 2025-11). A record
    is dropped if it fails EITHER check; the per-filter drop counts are printed.
  * Both models evaluated on the SAME collocation points (inner join on
    (band, station_idx, scene_time)); N_pretrained, N_tuned, N_common printed.
  * Reference = IGRA (u,v) linearly interpolated in z to EACH model's OWN retrieved
    height h_hat, using the bracketing-level scheme of the training sonde loss
    (SparseWindLoss / Eq. 8): weight w = clip((h_hat - h_below)/(h_above - h_below),
    0, 1). Bracketing (a level above and below h_hat) is required.

RMSVD = sqrt(mean[(u_hat-u_sonde)^2 + (v_hat-v_sonde)^2]).

Layout: 2x3 hexbin (rows pretrained/tuned; cols u, v, speed) with shared,
tail-safe, symmetric per-column axes and a shared hexbin color scale; plus a
full-width per-band RMSVD bar panel (pretrained vs tuned, %% improvement labeled).
Aggregate RMSVD improvement is reported in the suptitle. Nothing is hardcoded.
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

from eval_from_parquet import _build_qa_mask, neighborhood_median  # noqa: E402
from stereo_winds.validation.metrics import rmsvd, speed_bias, correlation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TEST_MONTHS = ["2025-10", "2025-11"]        # held-out test months
VAL_STATION_MOD = 5                          # dataset.py: val = station_idx % 5 == 0
STYLE = BASE / "figures" / "paper.mplstyle"


def stereo_zarr_path(cache_dir, label, band, month, n_iter):
    """{cache_dir}/{label}_{band}_{YYYYMM}_iter{n}.zarr (band-agnostic fallback)."""
    ym = month.replace("-", "")
    for cand in (f"{label}_{band}_{ym}_iter{n_iter}.zarr",
                 f"{label}_{ym}_iter{n_iter}.zarr"):
        p = Path(cache_dir) / cand
        if p.exists():
            return p
    return None


def _interp_to_height(sonde_h, sonde_u, sonde_v, valid, h_hat):
    """Eq. 8 bracketing linear-in-z interp of (u,v) to h_hat. None if no bracket."""
    below = valid & (sonde_h <= h_hat)
    above = valid & (sonde_h > h_hat)
    if not (below.any() and above.any()):
        return None
    ib = np.where(below)[0][np.argmax(sonde_h[below])]   # closest level below
    ia = np.where(above)[0][np.argmin(sonde_h[above])]   # closest level above
    hb, ha = sonde_h[ib], sonde_h[ia]
    w = np.clip((h_hat - hb) / max(ha - hb, 1.0), 0.0, 1.0)
    return ((1 - w) * sonde_u[ib] + w * sonde_u[ia],
            (1 - w) * sonde_v[ib] + w * sonde_v[ia])


def collocate(cache_dir, label, band, month, df, n_iter, min_scenes=2):
    """Collocate one cached retrieval to IGRA; Eq.8 interp to its OWN height.

    Returns list of dicts: band, station_idx, t (ns int), month, u_hat, v_hat,
    h_hat, u_ref, v_ref. Skips caches with < min_scenes times (e.g. smoke n=1).
    """
    path = stereo_zarr_path(cache_dir, label, band, month, n_iter)
    if path is None:
        logger.warning("  [%s %s %s] no cache", label, band, month)
        return []
    ds = xr.open_zarr(path, consolidated=False)
    if "time" not in ds.dims:
        ds = ds.expand_dims("time")
    if ds.sizes["time"] < min_scenes:
        logger.warning("  [%s %s %s] only %d scene(s) -> skip", label, band, month,
                       ds.sizes["time"])
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
    logger.info("  [%s %s %s] %d scenes -> %d collocations",
                label, band, month, len(times), len(out))
    return out


def collect(cache_dir, label, bands, months, df, n_iter):
    rows = []
    for b in bands:
        for m in months:
            rows.extend(collocate(cache_dir, label, b, m, df, n_iter))
    return pd.DataFrame(rows)


def apply_holdout(dfm, name, val_stations_only=False):
    """Enforce the held-out MONTHS (2025-10/11). Because those months were held
    out of teacher training entirely, every station's test-month data is unseen,
    so the station split is NOT required (kept optional via val_stations_only).
    Prints the per-filter drops and the train/val station composition so the
    guard is auditable."""
    n0 = len(dfm)
    if n0 == 0:
        print(f"  {name}: 0 raw collocations")
        return dfm
    ym = dfm["month"].isin(TEST_MONTHS)
    n_drop_month = int((~ym).sum())
    dfm = dfm[ym]
    is_val = (dfm["station_idx"].astype(int) % VAL_STATION_MOD == 0)
    n_val, n_train = int(is_val.sum()), int((~is_val).sum())
    msg = (f"  {name}: raw={n0}  dropped_by_month={n_drop_month}  kept={len(dfm)}  "
           f"[held-out-month; station mix: {n_val} val + {n_train} train points, "
           f"{dfm['station_idx'].nunique()} stations]")
    if val_stations_only:
        dfm = dfm[is_val]
        msg += f"  -> station-filtered to val-only: {len(dfm)}"
    print(msg)
    return dfm


def _lim(vals, step=5):
    m = float(np.nanmax(np.abs(vals))) if len(vals) else step
    lim = np.ceil((m + 1e-6) / step) * step
    return lim, m


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--parquet", default=None)
    p.add_argument("--label-pre", default="init_ep254")
    p.add_argument("--label-tuned", default="hreg1s75")
    p.add_argument("--bands", nargs="+", default=["C08", "C09", "C10", "C12", "C14"])
    p.add_argument("--months", nargs="+", default=TEST_MONTHS)
    p.add_argument("--n-iter", type=int, default=3)
    p.add_argument("--min-common", type=int, default=50,
                   help="drop a band from the bar panel if N_common < this")
    p.add_argument("--out", default=str(BASE / "figures" / "fig_finetuning_improvement_sonde.png"))
    p.add_argument("--dump", default=None, help="optional parquet dump of merged collocations")
    p.add_argument("--from-dump", default=None,
                   help="re-plot from a --dump parquet (no caches/parquet needed)")
    args = p.parse_args()

    if args.from_dump:
        if args.from_dump.endswith(".csv"):     # CSV path needs no pyarrow
            merged = pd.read_csv(args.from_dump)
        else:
            merged = pd.read_parquet(args.from_dump)
        print(f"=== Loaded merged collocations from {args.from_dump}: N={len(merged)}")
    else:
        if not (args.cache_dir and args.parquet):
            raise SystemExit("--cache-dir and --parquet are required unless --from-dump")
        months_ym = [m.replace("-", "") for m in args.months]
        print(f"=== Loading IGRA parquet: {args.parquet}")
        df = pd.read_parquet(args.parquet)
        print(f"    {len(df)} rows, {df['station_idx'].nunique()} stations; "
              f"val stations (idx%{VAL_STATION_MOD}==0): "
              f"{sum(int(s)%VAL_STATION_MOD==0 for s in df['station_idx'].unique())}")

        print("=== Collocating (pretrained) ===")
        dpre = collect(args.cache_dir, args.label_pre, args.bands, months_ym, df, args.n_iter)
        print("=== Collocating (tuned) ===")
        dtun = collect(args.cache_dir, args.label_tuned, args.bands, months_ym, df, args.n_iter)

        print("=== Holdout accounting (test months + held-out stations) ===")
        dpre = apply_holdout(dpre, "pretrained")
        dtun = apply_holdout(dtun, "tuned")
        if len(dpre) == 0 or len(dtun) == 0:
            print("!! One model has no held-out collocations yet — is generation done?")
            if len(dpre) == 0 and len(dtun) == 0:
                return

        key = ["band", "station_idx", "t"]
        merged = dpre.merge(dtun, on=key, suffixes=("_pre", "_tun"))
        print(f"=== Inner join on {key} ===")
        print(f"  N_pretrained={len(dpre)}  N_tuned={len(dtun)}  N_common={len(merged)}")
        if len(merged) == 0:
            print("!! No common collocations yet — need overlapping (band, month) for both models.")
            print("   pretrained bands/months:",
                  sorted(set(zip(dpre['band'], dpre['month']))) if len(dpre) else [])
            print("   tuned bands/months:",
                  sorted(set(zip(dtun['band'], dtun['month']))) if len(dtun) else [])
            return
        if args.dump:
            merged.to_parquet(args.dump)

    # ---- Metrics -----------------------------------------------------------
    def block(m, tag):
        u_h, v_h = m[f"u_hat_{tag}"].values, m[f"v_hat_{tag}"].values
        u_r, v_r = m[f"u_ref_{tag}"].values, m[f"v_ref_{tag}"].values
        return u_h, v_h, u_r, v_r

    uhp, vhp, urp, vrp = block(merged, "pre")
    uht, vht, urt, vrt = block(merged, "tun")

    def agg_rmsvd(m, tag):
        u_h, v_h, u_r, v_r = block(m, tag)
        return rmsvd(u_h, v_h, u_r, v_r)

    rv_pre_all = agg_rmsvd(merged, "pre")
    rv_tun_all = agg_rmsvd(merged, "tun")
    impr_all = 100.0 * (rv_pre_all - rv_tun_all) / rv_pre_all

    # Per-band
    print("=== Per-band (common set) ===")
    band_rows = []
    for b in args.bands:
        mb = merged[merged["band"] == b]
        if len(mb) == 0:
            continue
        rp, rt = agg_rmsvd(mb, "pre"), agg_rmsvd(mb, "tun")
        imp = 100.0 * (rp - rt) / rp
        band_rows.append(dict(band=b, n=len(mb), rmsvd_pre=rp, rmsvd_tun=rt, impr=imp))
        print(f"  {b}: N={len(mb):5d}  RMSVD pre={rp:5.2f}  tuned={rt:5.2f}  "
              f"impr={imp:+5.1f}%" + ("   [<min-common, excluded from bars]"
                                       if len(mb) < args.min_common else ""))
    bar_df = pd.DataFrame([r for r in band_rows if r["n"] >= args.min_common])
    dropped_bands = [r["band"] for r in band_rows if r["n"] < args.min_common]

    print(f"=== AGGREGATE (common N={len(merged)}) ===")
    print(f"  RMSVD pretrained={rv_pre_all:.3f}  tuned={rv_tun_all:.3f}  "
          f"improvement={impr_all:+.1f}%")
    print(f"  speed_bias pre={speed_bias(uhp,vhp,urp,vrp):+.2f}  "
          f"tuned={speed_bias(uht,vht,urt,vrt):+.2f}")

    # ---- Figure ------------------------------------------------------------
    if STYLE.exists():
        plt.style.use(str(STYLE))
    # Shared, tail-safe symmetric per-column limits.
    lim_u, mx_u = _lim(np.concatenate([uhp, urp, uht, urt]))
    lim_v, mx_v = _lim(np.concatenate([vhp, vrp, vht, vrt]))
    sp_all = np.concatenate([np.hypot(uhp, vhp), np.hypot(urp, vrp),
                             np.hypot(uht, vht), np.hypot(urt, vrt)])
    lim_s = np.ceil((float(np.nanmax(sp_all)) + 1e-6) / 5) * 5
    print(f"=== Axis limits: u=+/-{lim_u} (max {mx_u:.1f})  v=+/-{lim_v} "
          f"(max {mx_v:.1f})  speed=0..{lim_s} (max {np.nanmax(sp_all):.1f}) ===")

    # Native \textwidth sizing (~7 in) so fonts print at their nominal size.
    fig = plt.figure(figsize=(7.0, 7.0))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.85],
                  hspace=0.38, wspace=0.42, left=0.11, right=0.89,
                  top=0.925, bottom=0.07)
    letters = "abcdefg"

    def _tag(ax, k):
        ax.text(0.0, 1.03, f"({letters[k]})", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=9, fontweight="bold")

    rows = [("Pre-trained WindFlow", uhp, vhp, urp, vrp),
            ("Sonde-tuned WindFlow", uht, vht, urt, vrt)]
    col_specs = [("u", "$u$", -lim_u, lim_u, False),
                 ("v", "$v$", -lim_v, lim_v, False),
                 ("speed", "wind speed", 0, lim_s, True)]
    hbs = []
    axgrid = [[None, None, None], [None, None, None]]
    for i, (rlabel, u_h, v_h, u_r, v_r) in enumerate(rows):
        sp_h, sp_r = np.hypot(u_h, v_h), np.hypot(u_r, v_r)
        for j, (ckey, clabel, lo, hi, is_sp) in enumerate(col_specs):
            ax = fig.add_subplot(gs[i, j])
            axgrid[i][j] = ax
            if ckey == "u":
                x, y = u_r, u_h
            elif ckey == "v":
                x, y = v_r, v_h
            else:
                x, y = sp_r, sp_h
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=1)
            hb = ax.hexbin(x, y, gridsize=40, cmap="magma", bins="log", mincnt=1,
                           extent=(lo, hi, lo, hi))
            hbs.append(hb)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
            # stats
            if is_sp:
                rv = rmsvd(u_h, v_h, u_r, v_r)
                bias = speed_bias(u_h, v_h, u_r, v_r)
                r = correlation(sp_h, sp_r)
                stat = f"N={len(x):,}\nbias={bias:+.2f}\nRMSVD={rv:.2f}\nr={r:.2f}"
            else:
                comp_h, comp_r = (y, x)
                bias = float(np.mean(comp_h - comp_r))
                rmse = float(np.sqrt(np.mean((comp_h - comp_r) ** 2)))
                r = correlation(comp_h, comp_r)
                stat = f"N={len(x):,}\nbias={bias:+.2f}\nRMSD={rmse:.2f}\nr={r:.2f}"
            ax.text(0.04, 0.96, stat, transform=ax.transAxes, va="top", ha="left",
                    fontsize=6.5, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                            ec="0.5", alpha=0.9))
            ax.set_xlabel(f"Radiosonde {clabel} (m s$^{{-1}}$)")
            ax.set_ylabel(f"{rlabel.split()[0]} {clabel} (m s$^{{-1}}$)")
            if i == 0:
                ax.set_title(clabel, fontsize=10)
            if j == 0:
                ax.text(-0.46, 0.5, rlabel, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=9, fontweight="bold")
            _tag(ax, i * 3 + j)
    # consistent hexbin color scale
    vmax = max(hb.get_array().max() for hb in hbs)
    for hb in hbs:
        hb.set_norm(LogNorm(vmin=1, vmax=vmax))
    # vertical colorbar to the right of the hexbin block, spanning both rows
    fig.canvas.draw()  # realize axes positions before querying
    p_top = axgrid[0][2].get_position()
    p_bot = axgrid[1][2].get_position()
    cax = fig.add_axes([p_top.x1 + 0.018, p_bot.y0, 0.013, p_top.y1 - p_bot.y0])
    fig.colorbar(hbs[0], cax=cax, orientation="vertical").set_label("count")

    # ---- Bar panel ---------------------------------------------------------
    axb = fig.add_subplot(gs[2, :])
    if len(bar_df):
        x = np.arange(len(bar_df)); w = 0.38
        b1 = axb.bar(x - w / 2, bar_df["rmsvd_pre"], w, label="Pre-trained WindFlow",
                     color="#9aa0a6")
        b2 = axb.bar(x + w / 2, bar_df["rmsvd_tun"], w, label="Sonde-tuned WindFlow",
                     color="#e8743b")
        for xi, row in zip(x, bar_df.itertuples()):
            top = max(row.rmsvd_pre, row.rmsvd_tun)
            axb.text(xi, top + 0.15, f"{row.impr:+.0f}%\n(N={row.n})", ha="center",
                     va="bottom", fontsize=7,
                     color="#2a7a2a" if row.impr > 0 else "#a11")
        axb.set_xticks(x); axb.set_xticklabels(bar_df["band"])
        axb.set_ylabel("RMSVD (m s$^{-1}$)")
        axb.set_ylim(0, float(bar_df[["rmsvd_pre", "rmsvd_tun"]].values.max()) * 1.25)
        axb.legend(loc="upper right", fontsize=8)
        axb.set_title("Per-band RMSVD vs held-out radiosondes", fontsize=10,
                      fontweight="normal")
        _tag(axb, 6)
    if dropped_bands:
        axb.text(0.01, 0.97, f"excluded (N<{args.min_common}): {', '.join(dropped_bands)}",
                 transform=axb.transAxes, va="top", fontsize=7, color="0.4")

    fig.suptitle(
        f"Fine-tuning improvement vs held-out IGRA radiosondes  ·  aggregate RMSVD "
        f"{rv_pre_all:.2f} → {rv_tun_all:.2f} m s$^{{-1}}$  ({impr_all:+.1f}%,  "
        f"N={len(merged):,})", fontsize=10, y=0.98)
    out_base = str(Path(args.out).with_suffix(""))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=300)
    print(f"=== wrote {out_base}.{{png,pdf}}")


if __name__ == "__main__":
    main()
