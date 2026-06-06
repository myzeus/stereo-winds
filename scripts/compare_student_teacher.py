"""Aggregate teacher-overlap metrics across all common times.

Reads a multi-time student inference Zarr (from
``cache_student_inference.py``, matching the stereo-cache schema) and one or
more teacher chunk Zarrs (from ``cache_student_dataset.py``).  For every
common time, crops the student to the teacher's overlap bbox
(``row_offset``/``col_offset`` attrs), masks by ``quality_flag >= 1``, and
concatenates all valid pixels across time into long vectors.

Reports (one summary, all in physical units):
- RMSVD, per-component RMSE, mean wind-speed bias (m/s)
- height RMSE (m), height bias (m)
- correlations: u, v, h
- sigma-calibration: fraction of |residual| <= predicted sigma, per target

All reductions go through ``stereo_winds.validation.metrics`` so numbers are
directly comparable with prior stereo-retrieval evaluations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_winds.validation.metrics import (
    correlation, height_rmse, rmsvd, speed_bias,
)


def _open(p):
    return xr.open_zarr(p, chunks=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student-zarr", required=True,
                    help="Student cache Zarr (cache_student_inference.py output).")
    ap.add_argument("--teacher-zarr", nargs="+", required=True,
                    help="One or more teacher chunk Zarrs.")
    ap.add_argument("--qa-min", type=int, default=1,
                    help="Minimum teacher quality_flag to count a pixel.")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-png", default=None,
                    help="If set, writes a 6-panel scatter/residual PNG.")
    args = ap.parse_args()

    S = _open(args.student_zarr)
    teachers = [_open(p) for p in args.teacher_zarr]
    # Use the first teacher's crop bbox (all chunks share the same bbox).
    ro = int(teachers[0].attrs.get("row_offset", 0))
    co = int(teachers[0].attrs.get("col_offset", 0))
    bh = int(teachers[0].sizes["y"])
    bw = int(teachers[0].sizes["x"])

    # Build maps: time -> integer index for both sides.  ``sel(time=t)`` can
    # silently return an empty time-dim when ``t`` doesn't match the index
    # dtype exactly, so use isel against an explicit map; canonicalize times
    # to seconds-precision to avoid ns-vs-us-vs-ns float-noise mismatches.
    def _canon(t):
        return np.datetime64(t, "s")
    where = {}
    for ti, tch in enumerate(teachers):
        for j, t in enumerate(np.asarray(tch.time.values)):
            where.setdefault(_canon(t), (ti, j))
    s_idx = {_canon(t): i for i, t in enumerate(np.asarray(S.time.values))}
    common = sorted(t for t in where if t in s_idx)
    print(f"student: {len(s_idx)} times | teacher: {len(where)} times "
          f"| common: {len(common)}")

    accum = {k: [] for k in ("u_s", "u_t", "v_s", "v_t", "h_s", "h_t",
                             "su", "sv", "sh")}
    n_pix = 0
    for t in common:
        ti, j = where[t]
        ls = teachers[ti].isel(time=j)
        ss = S.isel(time=s_idx[t])
        u_t = ls["u_wind"].values; v_t = ls["v_wind"].values
        h_t = ls["cloud_top_height"].values  # meters
        qf = ls["quality_flag"].values
        u_s = ss["u_wind"].values[ro:ro + bh, co:co + bw]
        v_s = ss["v_wind"].values[ro:ro + bh, co:co + bw]
        h_s = ss["cloud_top_height"].values[ro:ro + bh, co:co + bw]
        su = ss["sigma_u"].values[ro:ro + bh, co:co + bw]
        sv = ss["sigma_v"].values[ro:ro + bh, co:co + bw]
        sh = ss["sigma_h"].values[ro:ro + bh, co:co + bw]
        m = ((qf >= args.qa_min)
             & np.isfinite(u_t) & np.isfinite(u_s)
             & np.isfinite(v_t) & np.isfinite(v_s)
             & np.isfinite(h_t) & np.isfinite(h_s))
        if not m.any():
            continue
        accum["u_s"].append(u_s[m]); accum["u_t"].append(u_t[m])
        accum["v_s"].append(v_s[m]); accum["v_t"].append(v_t[m])
        accum["h_s"].append(h_s[m]); accum["h_t"].append(h_t[m])
        accum["su"].append(su[m]); accum["sv"].append(sv[m]); accum["sh"].append(sh[m])
        n_pix += int(m.sum())

    if n_pix == 0:
        print("no valid overlap pixels — nothing to report")
        return

    cat = {k: np.concatenate(v) for k, v in accum.items()}
    print(f"valid overlap pixels across {len(common)} times: {n_pix:,}")

    # Reductions via the shared validation/metrics module.
    rep = {
        "n_times": len(common),
        "n_pixels": int(n_pix),
        "rmsvd_m_s": float(rmsvd(cat["u_s"], cat["v_s"], cat["u_t"], cat["v_t"])),
        "rmse_u_m_s": float(np.sqrt(np.mean((cat["u_s"] - cat["u_t"]) ** 2))),
        "rmse_v_m_s": float(np.sqrt(np.mean((cat["v_s"] - cat["v_t"]) ** 2))),
        "speed_bias_m_s": float(speed_bias(cat["u_s"], cat["v_s"], cat["u_t"], cat["v_t"])),
        "height_rmse_m": float(height_rmse(cat["h_s"], cat["h_t"])),
        "height_bias_m": float(np.mean(cat["h_s"] - cat["h_t"])),
        "corr_u": float(correlation(cat["u_s"], cat["u_t"])),
        "corr_v": float(correlation(cat["v_s"], cat["v_t"])),
        "corr_h": float(correlation(cat["h_s"], cat["h_t"])),
        "calib_u": float(np.mean(np.abs(cat["u_s"] - cat["u_t"]) <= cat["su"])),
        "calib_v": float(np.mean(np.abs(cat["v_s"] - cat["v_t"]) <= cat["sv"])),
        "calib_h": float(np.mean(np.abs(cat["h_s"] - cat["h_t"]) <= cat["sh"])),
    }
    for k in ("rmsvd_m_s", "rmse_u_m_s", "rmse_v_m_s", "speed_bias_m_s",
              "height_rmse_m", "height_bias_m",
              "corr_u", "corr_v", "corr_h",
              "calib_u", "calib_v", "calib_h"):
        print(f"  {k:18s} {rep[k]:10.4f}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"saved {args.out_json}")

    if args.out_png:
        # Subsample for plotting (millions of pixels would crush matplotlib)
        rng = np.random.default_rng(0)
        idx = rng.choice(n_pix, size=min(n_pix, 100_000), replace=False)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ax, (s_arr, t_arr, title, unit) in zip(
            axes[0],
            [(cat["u_s"], cat["u_t"], "u", "m/s"),
             (cat["v_s"], cat["v_t"], "v", "m/s"),
             (cat["h_s"], cat["h_t"], "h", "m")],
        ):
            ax.hexbin(t_arr[idx], s_arr[idx], gridsize=80, mincnt=1, cmap="viridis")
            lim = max(np.abs(t_arr[idx]).max(), np.abs(s_arr[idx]).max())
            if title == "h":
                ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            else:
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.plot([-lim, lim], [-lim, lim], 'r--', lw=0.5)
            ax.set_xlabel(f"teacher {title} ({unit})")
            ax.set_ylabel(f"student {title} ({unit})")
            r = correlation(s_arr, t_arr)
            ax.set_title(f"{title}  corr={r:.3f}")
        for ax, (s_arr, t_arr, title, unit) in zip(
            axes[1],
            [(cat["u_s"], cat["u_t"], "u", "m/s"),
             (cat["v_s"], cat["v_t"], "v", "m/s"),
             (cat["h_s"], cat["h_t"], "h", "m")],
        ):
            d = s_arr - t_arr
            ax.hist(d, bins=80, range=(np.percentile(d, 1), np.percentile(d, 99)),
                    color="steelblue", alpha=0.7)
            ax.axvline(0, c="r", lw=0.5)
            ax.set_xlabel(f"student - teacher  {title} ({unit})")
            ax.set_title(f"residual  mean={float(d.mean()):+.2f}  rms={float(np.sqrt((d**2).mean())):.2f}")
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=120, bbox_inches="tight")
        print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
