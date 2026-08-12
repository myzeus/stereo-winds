#!/usr/bin/env python
"""Paper figure: quad-collocation of stereo (teacher) vs AMV, ERA5, radiosonde.

(a) Stereo wind speed vs radiosonde speed (C14) with regression + 1:1 line —
    the jet-underestimation signature (slope < 1).
(b) Speed bias vs sonde speed, binned, for stereo / AMV / ERA5 — stereo
    under-reads jets worst, AMV less, ERA5 ~unbiased (reanalysis reference).
Data: quad_matches.npz (held-out 2025-10/11, all IGRA).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("figures/paper.mplstyle")
D = np.load("quad_matches.npz", allow_pickle=True)
B = "C14"  # most matches (N=469)


def g(k):
    return D[f"{B}__{k}"]


so = np.hypot(g("u_sonde"), g("v_sonde"))
srcs = {"Stereo": ("u_stereo", "v_stereo", "#3b4cc0"),
        "AMV":    ("u_amv", "v_amv", "#e08214"),
        "ERA5":   ("u_era5", "v_era5", "#1a9850")}

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.4))

# --- (a) stereo speed vs sonde speed ---
sp = np.hypot(g("u_stereo"), g("v_stereo"))
fin = np.isfinite(sp) & np.isfinite(so)
spf, sof = sp[fin], so[fin]
axa.scatter(sof, spf, s=6, alpha=0.35, color="#3b4cc0", edgecolors="none")
lim = [0, max(sof.max(), spf.max()) * 1.05]
axa.plot(lim, lim, "k--", lw=0.8, label="1:1")
m, b = np.polyfit(sof, spf, 1)
xs = np.array(lim)
axa.plot(xs, m * xs + b, color="#b2182b", lw=1.5,
         label=f"fit: {m:.2f}·x + {b:.1f}")
axa.set_xlim(lim); axa.set_ylim(lim)
axa.set_xlabel("Radiosonde speed (m s$^{-1}$)")
axa.set_ylabel("Stereo speed (m s$^{-1}$)")
axa.set_title(f"(a) Stereo vs sonde speed  (N={fin.sum()})")
axa.legend(loc="upper left", frameon=False, fontsize=8)
axa.text(0.97, 0.06, f"slope {m:.2f}\n→ jets under-read", transform=axa.transAxes,
         ha="right", va="bottom", fontsize=8, color="#b2182b")

# --- (b) speed bias vs sonde-speed bin, all sources ---
bins = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 70)]
centers = [np.mean(bb) for bb in bins]
for name, (uk, vk, col) in srcs.items():
    spx = np.hypot(g(uk), g(vk))
    fin = np.isfinite(spx) & np.isfinite(so)
    spx, soo = spx[fin], so[fin]
    biases = []
    for lo, hi in bins:
        mm = (soo >= lo) & (soo < hi)
        biases.append(np.mean(spx[mm] - soo[mm]) if mm.sum() >= 3 else np.nan)
    axb.plot(centers, biases, "o-", color=col, lw=1.5, ms=4, label=name)
axb.axhline(0, color="k", lw=0.6)
axb.set_xlabel("Radiosonde speed (m s$^{-1}$)")
axb.set_ylabel("Speed bias (m s$^{-1}$)")
axb.set_title("(b) Speed bias vs wind speed")
axb.legend(loc="lower left", frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig("figures/fig_quad_collocation.png", dpi=200, bbox_inches="tight")
fig.savefig("figures/fig_quad_collocation.pdf", bbox_inches="tight")
print("wrote figures/fig_quad_collocation.{png,pdf}")
print(f"  stereo speed regression slope = {m:.3f}")
