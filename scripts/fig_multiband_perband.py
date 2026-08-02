#!/usr/bin/env python
"""Paper figure: multi-band student per-band skill vs radiosondes.

Per flow band (ordered by wavelength / sampled level: WV 6.2-7.3um → high
cloud, IR 9.6-11um → low cloud): u & v correlation and RMSVD vs IGRA.
Shows the multi-level structure — WV bands give strong high-cloud winds,
IR bands the weaker low-cloud regime.  Numbers: v3 (base-32 + per-band loss +
dihedral aug), held-out 2025-10 test scenes.  C08 is under-sampled (N=3) and
flagged.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("figures/paper.mplstyle")

# band: (wavelength label, level, N, u_corr, v_corr, RMSVD)
BANDS = [
    ("C08", "6.2 µm", "high", 3,   0.738, 0.998, 3.56),
    ("C09", "6.9 µm", "high", 59,  0.903, 0.915, 3.53),
    ("C10", "7.3 µm", "high", 162, 0.911, 0.921, 3.88),
    ("C12", "9.6 µm", "low",  27,  0.604, 0.669, 7.89),
    ("C14", "11 µm",  "low",  130, 0.436, 0.598, 7.40),
]
names = [b[0] for b in BANDS]
wl = [b[1] for b in BANDS]
Ns = [b[3] for b in BANDS]
uc = np.array([b[4] for b in BANDS])
vc = np.array([b[5] for b in BANDS])
rms = np.array([b[6] for b in BANDS])
sparse = np.array([n < 10 for b, *_ in [BANDS] for n in Ns])  # C08 low-N flag
x = np.arange(len(BANDS))

fig, (axc, axr) = plt.subplots(1, 2, figsize=(7.4, 3.4))

# --- (a) u/v correlation ---
w = 0.38
for i in range(len(BANDS)):
    a = 0.35 if Ns[i] < 10 else 1.0   # fade the under-sampled band
    axc.bar(x[i] - w/2, uc[i], w, color="#3b4cc0", alpha=a,
            label="u corr" if i == 1 else None)
    axc.bar(x[i] + w/2, vc[i], w, color="#e08214", alpha=a,
            label="v corr" if i == 1 else None)
axc.axhline(0.9, color="0.5", ls=":", lw=0.8)
axc.set_ylim(0, 1.05)
axc.set_xticks(x)
axc.set_xticklabels([f"{n}\n{w}" for n, w in zip(names, wl)])
axc.set_ylabel("Correlation vs radiosonde")
axc.set_title("(a) Per-band wind correlation")
axc.legend(loc="lower left", frameon=False, fontsize=8)
for i in range(len(BANDS)):
    axc.text(x[i], 0.02, f"N={Ns[i]}", ha="center", va="bottom", fontsize=7,
             color="0.3")
axc.text(len(BANDS)-1, 0.93, "0.9", fontsize=7, color="0.5", va="bottom", ha="right")

# --- (b) RMSVD ---
for i, b in enumerate(BANDS):
    col = "#4a6fe3" if b[2] == "high" else "#c94a4a"
    axr.bar(x[i], rms[i], 0.6, color=col, alpha=0.35 if Ns[i] < 10 else 1.0)
axr.set_xticks(x)
axr.set_xticklabels(names)
axr.set_ylabel("RMSVD vs radiosonde (m s$^{-1}$)")
axr.set_title("(b) Per-band RMSVD")
# level legend
from matplotlib.patches import Patch
axr.legend(handles=[Patch(color="#4a6fe3", label="WV → high cloud"),
                    Patch(color="#c94a4a", label="IR → low cloud")],
           loc="upper left", frameon=False, fontsize=8)
axr.text(0.0, -0.02, "C08 under-sampled (N=3)", transform=axr.transAxes,
         fontsize=7, color="0.4", va="top")

fig.tight_layout()
fig.savefig("figures/fig_multiband_perband.png", dpi=200, bbox_inches="tight")
fig.savefig("figures/fig_multiband_perband.pdf", bbox_inches="tight")
print("wrote figures/fig_multiband_perband.{png,pdf}")
