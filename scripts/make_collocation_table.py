#!/usr/bin/env python
"""Emit a LaTeX collocation table comparing the sonde-tuned stereo winds, the
single-satellite wind student, operational NOAA AMVs, and ERA5 against IGRA
radiosondes, per ABI band, over the held-out test months.

Reads the matched arrays from ``compare_quad_collocation.py`` plus the student
attached at the stereo-accepted points (``quad_matches_student.npz`` — the quad
run augmented with ``{band}__{u,v,h}_student``) and writes a booktabs table to
``figures/tab_collocation.tex``.

Every system is evaluated on the SAME per-band point set: the common mask where
stereo, student, AMV, ERA5 and the sonde are all finite. Dropping the handful of
points a system is missing (e.g. scenes without student input imagery) keeps $N$
matched within each band, so the rows are directly comparable.

    python scripts/make_collocation_table.py
"""
import argparse
from pathlib import Path

import numpy as np

from stereo_winds.validation.metrics import correlation, rmsvd, speed_bias

REPO = Path(__file__).resolve().parent.parent
WAVELENGTH = {"C08": "6.2", "C09": "6.9", "C10": "7.3", "C14": "11.2"}


def common_mask(d, band, systems):
    """Points where the sonde and every listed system are finite (matched N)."""
    ok = np.isfinite(d[f"{band}__u_sonde"]) & np.isfinite(d[f"{band}__v_sonde"])
    for s in systems:
        ok &= np.isfinite(d[f"{band}__u_{s}"]) & np.isfinite(d[f"{band}__v_{s}"])
    return ok


def pair_vs_sonde(d, band, sys, mask):
    u = d[f"{band}__u_{sys}"][mask]; v = d[f"{band}__v_{sys}"][mask]
    ur = d[f"{band}__u_sonde"][mask]; vr = d[f"{band}__v_sonde"][mask]
    rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
    return dict(
        n=int(mask.sum()),
        rmsvd=rmsvd(u, v, ur, vr),
        spbias=speed_bias(u, v, ur, vr),
        bias_u=float(np.mean(u - ur)), rmse_u=rms(u - ur), r_u=correlation(u, ur),
        bias_v=float(np.mean(v - vr)), rmse_v=rms(v - vr), r_v=correlation(v, vr),
    )


def tc_errors(d, band, mask):
    """Stoffelen-1998 TC RMS error for {stereo, amv, sonde} on the matched set."""
    var = {"stereo": 0.0, "amv": 0.0, "sonde": 0.0}
    for comp in ("u", "v"):
        s = d[f"{band}__{comp}_stereo"][mask]
        a = d[f"{band}__{comp}_amv"][mask]
        r = d[f"{band}__{comp}_sonde"][mask]
        s, a, r = s - s.mean(), a - a.mean(), r - r.mean()
        var["stereo"] += max(float(np.mean((s - a) * (s - r))), 0.0)
        var["amv"] += max(float(np.mean((a - s) * (a - r))), 0.0)
        var["sonde"] += max(float(np.mean((r - s) * (r - a))), 0.0)
    return {k: float(np.sqrt(v)) for k, v in var.items()}


# Row order per band. Student is inserted after stereo when present in the npz.
SYS_ROWS = [("stereo", "Stereo (sonde-tuned)"),
            ("student", "Student (single-sat)$^{\\ddagger}$"),
            ("amv", "NOAA AMV"),
            ("era5", "ERA5$^{\\dagger}$")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(REPO / "quad_matches_student.npz"))
    ap.add_argument("--bands", nargs="+", default=["C08", "C09", "C10", "C14"])
    ap.add_argument("--out", default=str(REPO / "figures/tab_collocation.tex"))
    args = ap.parse_args()
    d = np.load(args.npz)

    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Per-band, per-observation-type validation of the sonde-tuned stereo "
             r"winds and the single-satellite wind student against operational NOAA GOES-19 "
             r"derived-motion AMVs and ERA5 reanalysis over the held-out test months "
             r"(October--November 2025; never seen in training), collocated with IGRA "
             r"radiosondes. Component statistics (bias, RMSE, correlation $r$ for the zonal "
             r"$u$ and meridional $v$ components) and the vector RMS difference (RMSVD) are "
             r"computed against the radiosondes. All systems are evaluated at the identical "
             r"per-band point set (points any system is missing are dropped from every system, "
             r"so $N$ is matched within each band); the student is sampled at the "
             r"stereo-accepted points. $\varepsilon_{\mathrm{TC}}$ is the Stoffelen (1998) "
             r"triple-collocation intrinsic error from the mutually independent "
             r"\{stereo, AMV, sonde\} triplet. Best (lowest) intrinsic error per band in "
             r"\textbf{bold}. $^{\dagger}$ERA5 assimilates both AMVs and radiosondes and is "
             r"therefore not independent---its close agreement with the sondes is inflated "
             r"and it is excluded from the triple-collocation decomposition. $^{\ddagger}$The "
             r"student is distilled from the stereo retrieval and is likewise not independent "
             r"of it, so it too is excluded from the triple-collocation decomposition.}")
    L.append(r"\label{tab:collocation}")
    L.append(r"\begin{tabular}{llrrrrrrrrc}")
    L.append(r"\toprule")
    L.append(r"& & & \multicolumn{3}{c}{$u$ component} & \multicolumn{3}{c}{$v$ component} & & \\")
    L.append(r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}")
    L.append(r"Band & System & $N$ & Bias & RMSE & $r$ & Bias & RMSE & $r$ "
             r"& RMSVD & $\varepsilon_{\mathrm{TC}}$ \\")
    L.append(r" & & & \multicolumn{2}{c}{(m\,s$^{-1}$)} & & \multicolumn{2}{c}{(m\,s$^{-1}$)} & "
             r"& (m\,s$^{-1}$) & (m\,s$^{-1}$) \\")
    L.append(r"\midrule")

    for bi, band in enumerate(args.bands):
        # Only rows whose system is actually in this npz (student is optional).
        rows = [(s, name) for s, name in SYS_ROWS if f"{band}__u_{s}" in d.files]
        systems = [s for s, _ in rows]
        mask = common_mask(d, band, systems)   # matched N across every row
        tc = tc_errors(d, band, mask)
        best = min(tc["stereo"], tc["amv"])     # TC decomposition only for independent legs
        head = (rf"\multirow{{{len(rows)}}}{{*}}{{\shortstack[l]{{{band}\\"
                rf"{WAVELENGTH[band]}\,$\mu$m}}}}")
        for si, (sys, name) in enumerate(rows):
            p = pair_vs_sonde(d, band, sys, mask)
            if sys in ("era5", "student"):      # not independent of the triplet
                tccell = "---"
            else:
                val = tc[sys]
                tccell = rf"\textbf{{{val:.2f}}}" if abs(val - best) < 1e-9 else f"{val:.2f}"
            lead = head if si == 0 else ""
            L.append(rf"{lead} & {name} & {p['n']} "
                     rf"& ${p['bias_u']:+.2f}$ & {p['rmse_u']:.2f} & {p['r_u']:.2f} "
                     rf"& ${p['bias_v']:+.2f}$ & {p['rmse_v']:.2f} & {p['r_v']:.2f} "
                     rf"& {p['rmsvd']:.2f} & {tccell} \\")
        L.append(r"\midrule" if bi < len(args.bands) - 1 else r"\bottomrule")

    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")

    tex = "\n".join(L) + "\n"
    Path(args.out).write_text(tex)
    print(tex)
    print(f"=== wrote {args.out}")


if __name__ == "__main__":
    main()
