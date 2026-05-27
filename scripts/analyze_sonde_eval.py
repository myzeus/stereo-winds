#!/usr/bin/env python
"""Rich init-vs-tuned held-out radiosonde comparison.

Reuses eval_from_parquet.evaluate_store (same neighborhood-median QA and
bracketing match as the headline numbers) to collect full matched
DataFrames for two models, then produces a multi-panel diagnostic figure
and expanded statistics (overall, per-layer, per-speed-bin, paired).

Example
-------
    python scripts/analyze_sonde_eval.py \\
        --parquet $DATA/labels/igra/igra_all_collocation.parquet \\
        --init-glob  "$CACHE/init_ep254_*_iter3.zarr" \\
        --tuned-glob "$CACHE/tuned_b82e_*_iter3.zarr" \\
        --out-dir    $RUNS/sonde_eval_largeN/analysis
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))
sys.path.insert(0, str(BASE / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_from_parquet import evaluate_store
from stereo_winds.validation.metrics import correlation, height_rmse, rmsvd, speed_bias


def collect(parquet_df, zarr_glob, label):
    """Run evaluate_store across all zarrs matching the glob; return a DataFrame."""
    paths = sorted(glob.glob(zarr_glob))
    rows = []
    for p in paths:
        rows.extend(evaluate_store(p, parquet_df, label))
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df["spd_stereo"] = np.hypot(df["u_stereo"], df["v_stereo"])
    df["spd_sonde"] = np.hypot(df["u_sonde"], df["v_sonde"])
    df["spd_err"] = df["spd_stereo"] - df["spd_sonde"]
    df["vec_diff"] = np.hypot(df["u_stereo"] - df["u_sonde"],
                              df["v_stereo"] - df["v_sonde"])
    # meteorological direction (from-which-blowing); error wrapped to [-180,180]
    dir_s = np.degrees(np.arctan2(-df["u_stereo"], -df["v_stereo"])) % 360
    dir_o = np.degrees(np.arctan2(-df["u_sonde"], -df["v_sonde"])) % 360
    ddir = (dir_s - dir_o + 180) % 360 - 180
    df["dir_err"] = ddir
    return df


LAYERS = [("Low\n(>=700)", lambda p: p >= 700),
          ("Mid\n(400-700)", lambda p: (p >= 400) & (p < 700)),
          ("High\n(<400)", lambda p: p < 400)]
PLEVELS = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100]


def layer_stats(df):
    out = {}
    for name, fn in LAYERS:
        sub = df[fn(df["pressure_hpa"])]
        if len(sub) >= 3:
            out[name] = dict(
                n=len(sub),
                rmsvd=rmsvd(sub["u_stereo"], sub["v_stereo"], sub["u_sonde"], sub["v_sonde"]),
                bias=speed_bias(sub["u_stereo"], sub["v_stereo"], sub["u_sonde"], sub["v_sonde"]),
                hrmse=height_rmse(sub["h_stereo"], sub["h_sonde"]),
            )
        else:
            out[name] = dict(n=len(sub), rmsvd=np.nan, bias=np.nan, hrmse=np.nan)
    return out


def overall(df):
    return dict(
        n=len(df),
        rmsvd=rmsvd(df["u_stereo"], df["v_stereo"], df["u_sonde"], df["v_sonde"]),
        bias=speed_bias(df["u_stereo"], df["v_stereo"], df["u_sonde"], df["v_sonde"]),
        cs=correlation(df["spd_stereo"], df["spd_sonde"]),
        cu=correlation(df["u_stereo"], df["u_sonde"]),
        cv=correlation(df["v_stereo"], df["v_sonde"]),
        hrmse=height_rmse(df["h_stereo"], df["h_sonde"]),
        hbias=float((df["h_stereo"] - df["h_sonde"]).mean()),
        dir_mae=float(df["dir_err"].abs().median()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--init-glob", required=True)
    ap.add_argument("--tuned-glob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", choices=["all", "train", "val"], default="val")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    if args.split != "all":
        is_val = (df["station_idx"].astype(int) % 5 == 0)
        df = df[is_val if args.split == "val" else ~is_val]
    print(f"parquet split={args.split}: {len(df)} rows, {df['station_idx'].nunique()} stations")

    di = collect(df, args.init_glob, "init")
    dt = collect(df, args.tuned_glob, "tuned")
    print(f"init matches:  {len(di)}")
    print(f"tuned matches: {len(dt)}")

    # ---- expanded stats table (stdout + file) ----
    lines = []
    def P(s=""):
        lines.append(s); print(s)
    oi, ot = overall(di), overall(dt)
    P("=" * 78)
    P(f"{'Metric':<22}{'init_ep254':>16}{'tuned_b82e':>16}{'delta':>16}")
    P("-" * 78)
    P(f"{'N matched':<22}{oi['n']:>16}{ot['n']:>16}{ot['n']-oi['n']:>+16}")
    for lbl, k in [("RMSVD (m/s)","rmsvd"),("Speed bias (m/s)","bias"),
                   ("Speed corr","cs"),("u corr","cu"),("v corr","cv"),
                   ("Height RMSE (m)","hrmse"),("Height bias (m)","hbias"),
                   ("|Dir err| median","dir_mae")]:
        a, b = oi[k], ot[k]
        P(f"{lbl:<22}{a:>16.3f}{b:>16.3f}{b-a:>+16.3f}")
    P("=" * 78)
    P("Per-layer (N / RMSVD / SpdBias / hRMSE):")
    li, lt = layer_stats(di), layer_stats(dt)
    for name, _ in LAYERS:
        nm = name.replace(chr(10), " ")
        a, b = li[name], lt[name]
        P(f"  {nm:<16} init  N={a['n']:>3}  RMSVD={a['rmsvd']:>6.2f}  "
          f"bias={a['bias']:>+6.2f}  hRMSE={a['hrmse']:>6.0f}")
        P(f"  {nm:<16} tuned N={b['n']:>3}  RMSVD={b['rmsvd']:>6.2f}  "
          f"bias={b['bias']:>+6.2f}  hRMSE={b['hrmse']:>6.0f}")
    (out / "stats.txt").write_text("\n".join(lines))

    # ---- multi-panel figure ----
    fig, ax = plt.subplots(3, 4, figsize=(24, 17))

    def speed_scatter(a, d, title):
        sc = a.scatter(d["spd_sonde"], d["spd_stereo"], c=d["h_stereo"]/1000,
                       cmap="turbo", vmin=0, vmax=16, s=18, alpha=0.6)
        lim = 70
        a.plot([0, lim], [0, lim], "k--", lw=0.8); a.set_xlim(0, lim); a.set_ylim(0, lim)
        a.set_aspect("equal"); a.set_xlabel("Sonde speed (m/s)"); a.set_ylabel("Stereo speed (m/s)")
        o = overall(d)
        a.set_title(f"{title}\nn={o['n']} RMSVD={o['rmsvd']:.2f} bias={o['bias']:+.2f} r={o['cs']:.2f}")
        return sc

    def comp_scatter(a, d, comp, title):
        a.scatter(d[f"{comp}_sonde"], d[f"{comp}_stereo"], c=d["h_stereo"]/1000,
                  cmap="turbo", vmin=0, vmax=16, s=18, alpha=0.6)
        a.plot([-60, 60], [-60, 60], "k--", lw=0.8); a.set_xlim(-60, 60); a.set_ylim(-60, 60)
        a.set_aspect("equal"); a.set_xlabel(f"Sonde {comp} (m/s)"); a.set_ylabel(f"Stereo {comp} (m/s)")
        a.set_title(f"{title}  r={correlation(d[f'{comp}_stereo'], d[f'{comp}_sonde']):.2f}")

    def height_scatter(a, d, title):
        a.scatter(d["h_sonde"]/1000, d["h_stereo"]/1000, c=d["vec_diff"],
                  cmap="YlOrRd", vmin=0, vmax=20, s=18, alpha=0.6)
        a.plot([0, 16], [0, 16], "k--", lw=0.8); a.set_xlim(0, 16); a.set_ylim(0, 16)
        a.set_aspect("equal"); a.set_xlabel("Sonde height (km)"); a.set_ylabel("Stereo height (km)")
        a.set_title(f"{title}  hRMSE={height_rmse(d['h_stereo'], d['h_sonde']):.0f} m")

    sc = speed_scatter(ax[0,0], di, "Speed — init")
    speed_scatter(ax[0,1], dt, "Speed — tuned")
    height_scatter(ax[0,2], di, "Height — init")
    height_scatter(ax[0,3], dt, "Height — tuned")
    comp_scatter(ax[1,0], dt, "u", "u-wind — tuned")
    comp_scatter(ax[1,1], dt, "v", "v-wind — tuned")

    # speed-error hist
    a = ax[1,2]
    bins = np.linspace(-30, 30, 61)
    a.hist(di["spd_err"], bins=bins, alpha=0.5, density=True, label=f"init (n={len(di)})", color="steelblue")
    a.hist(dt["spd_err"], bins=bins, alpha=0.5, density=True, label=f"tuned (n={len(dt)})", color="firebrick")
    a.axvline(0, color="k", lw=0.6); a.axvline(di["spd_err"].mean(), color="steelblue", ls="--")
    a.axvline(dt["spd_err"].mean(), color="firebrick", ls="--")
    a.set_xlabel("Speed error stereo-sonde (m/s)"); a.set_ylabel("density")
    a.set_title("Speed error"); a.legend(fontsize=9)

    # direction-error hist (speed>3 only, where dir is meaningful)
    a = ax[1,3]
    bins = np.linspace(-180, 180, 73)
    mi = di["spd_sonde"] > 3; mt = dt["spd_sonde"] > 3
    a.hist(di.loc[mi, "dir_err"], bins=bins, alpha=0.5, density=True, label=f"init (n={int(mi.sum())})", color="steelblue")
    a.hist(dt.loc[mt, "dir_err"], bins=bins, alpha=0.5, density=True, label=f"tuned (n={int(mt.sum())})", color="firebrick")
    a.axvline(0, color="k", lw=0.6)
    a.set_xlabel("Direction error (deg, spd>3)"); a.set_ylabel("density")
    a.set_title("Direction error"); a.legend(fontsize=9)

    # RMSVD vs pressure profile
    centers = [(PLEVELS[i]+PLEVELS[i+1])/2 for i in range(len(PLEVELS)-1)]
    def profile(d, fn):
        vals = []
        for i in range(len(PLEVELS)-1):
            m = (d["pressure_hpa"] >= PLEVELS[i+1]) & (d["pressure_hpa"] < PLEVELS[i])
            sub = d[m]
            vals.append(fn(sub) if len(sub) >= 3 else np.nan)
        return np.array(vals)
    a = ax[2,0]
    a.plot(profile(di, lambda s: rmsvd(s["u_stereo"],s["v_stereo"],s["u_sonde"],s["v_sonde"])), centers, "o-", color="steelblue", label="init")
    a.plot(profile(dt, lambda s: rmsvd(s["u_stereo"],s["v_stereo"],s["u_sonde"],s["v_sonde"])), centers, "o-", color="firebrick", label="tuned")
    a.set_yscale("log"); a.invert_yaxis(); a.set_ylim(1050, 90)
    a.set_xlabel("RMSVD (m/s)"); a.set_ylabel("Pressure (hPa)"); a.set_title("RMSVD vs pressure"); a.legend(); a.grid(alpha=0.3)

    a = ax[2,1]
    a.plot(profile(di, lambda s: speed_bias(s["u_stereo"],s["v_stereo"],s["u_sonde"],s["v_sonde"])), centers, "o-", color="steelblue", label="init")
    a.plot(profile(dt, lambda s: speed_bias(s["u_stereo"],s["v_stereo"],s["u_sonde"],s["v_sonde"])), centers, "o-", color="firebrick", label="tuned")
    a.axvline(0, color="k", lw=0.6); a.set_yscale("log"); a.invert_yaxis(); a.set_ylim(1050, 90)
    a.set_xlabel("Speed bias (m/s)"); a.set_ylabel("Pressure (hPa)"); a.set_title("Speed bias vs pressure"); a.legend(); a.grid(alpha=0.3)

    # RMSVD vs sonde-speed bin
    a = ax[2,2]
    sedges = np.arange(0, 70, 10); scent = (sedges[:-1]+sedges[1:])/2
    def spd_profile(d):
        vals = []
        for i in range(len(sedges)-1):
            m = (d["spd_sonde"] >= sedges[i]) & (d["spd_sonde"] < sedges[i+1])
            sub = d[m]
            vals.append(rmsvd(sub["u_stereo"],sub["v_stereo"],sub["u_sonde"],sub["v_sonde"]) if len(sub)>=3 else np.nan)
        return np.array(vals)
    a.plot(scent, spd_profile(di), "o-", color="steelblue", label="init")
    a.plot(scent, spd_profile(dt), "o-", color="firebrick", label="tuned")
    a.set_xlabel("Sonde speed bin (m/s)"); a.set_ylabel("RMSVD (m/s)"); a.set_title("RMSVD vs wind speed"); a.legend(); a.grid(alpha=0.3)

    # geographic map of tuned vector diff
    a = ax[2,3]
    sc2 = a.scatter(dt["lon"], dt["lat"], c=dt["vec_diff"], cmap="YlOrRd", vmin=0, vmax=20, s=30, edgecolors="k", linewidths=0.3)
    a.set_xlabel("lon"); a.set_ylabel("lat"); a.set_title("Tuned: vector diff by station")
    plt.colorbar(sc2, ax=a, shrink=0.8, label="vec diff (m/s)")

    cax = fig.add_axes([0.92, 0.67, 0.008, 0.2])
    plt.colorbar(sc, cax=cax, label="stereo height (km)")
    fig.suptitle("Held-out IGRA radiosonde performance — init_ep254 vs tuned_b82e (val split)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.91, 0.97])
    fig.savefig(out / "sonde_analysis.png", dpi=130, bbox_inches="tight")
    print(f"\nSaved {out/'sonde_analysis.png'} and {out/'stats.txt'}")


if __name__ == "__main__":
    main()
