#!/usr/bin/env python
"""Quick alignment check: cross-correlate A0 and B_minus on small clear-sky patches.

If the remap is correct, the peak of the cross-correlation should be at
(0, 0) over a ground-level region. A consistent non-zero peak across
multiple ground patches indicates a remap LUT translation error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import correlate2d


def histogram_equalize(image, n_bins=500):
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)
    vals = image[finite]
    hist, bins = np.histogram(vals, n_bins, density=True)
    cdf = hist.cumsum() / hist.sum()
    out = np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape)
    out[~finite] = 0.0
    return out.astype(np.float32)


def peak_offset(a, b, search_radius=20):
    """FFT-based phase correlation peak (dy, dx). Both must be the same shape."""
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    a = np.nan_to_num(a, 0.0)
    b = np.nan_to_num(b, 0.0)
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    cps = cross / (np.abs(cross) + 1e-12)
    corr = np.real(np.fft.ifft2(cps))
    corr = np.fft.fftshift(corr)
    h, w = corr.shape
    # restrict to a search window around the center
    cy, cx = h // 2, w // 2
    win = corr[cy - search_radius:cy + search_radius + 1,
               cx - search_radius:cx + search_radius + 1]
    peak = np.unravel_index(np.argmax(win), win.shape)
    dy = peak[0] - search_radius
    dx = peak[1] - search_radius
    return dy, dx, float(win.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True,
                        help="Directory with the 5 .npy scenes")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--out", required=True, help="Output png path")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    a0 = np.load(scene_dir / "A0.npy").astype(np.float32)
    bm = np.load(scene_dir / "B_minus.npy").astype(np.float32)
    bp = np.load(scene_dir / "B_plus.npy").astype(np.float32)

    print(f"Loaded scenes from {scene_dir}")
    print(f"  A0: shape={a0.shape}, finite={100*np.isfinite(a0).mean():.1f}%")
    print(f"  B_minus: shape={bm.shape}, finite={100*np.isfinite(bm).mean():.1f}%")

    # Sample 9 patches in a 3x3 grid across the overlap region.
    # Pick the western half of the disk where goes18 dominates and there
    # are typically large clear-sky areas for unambiguous matching.
    ps = args.patch_size
    H, W = a0.shape
    # Find pixels where both A0 and B_minus are finite — that's the overlap
    overlap = np.isfinite(a0) & np.isfinite(bm) & np.isfinite(bp)
    print(f"  Overlap pixels: {int(overlap.sum()):,} ({100*overlap.mean():.1f}%)")
    rows, cols = np.where(overlap)
    if len(rows) == 0:
        raise RuntimeError("No overlap pixels")
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    print(f"  Overlap bbox: rows [{r_min}, {r_max}], cols [{c_min}, {c_max}]")

    r_grid = np.linspace(r_min + ps // 2, r_max - ps // 2, 3).astype(int)
    c_grid = np.linspace(c_min + ps // 2, c_max - ps // 2, 3).astype(int)
    print(f"  Sampling at rows {r_grid}, cols {c_grid}")

    results_bm = []  # (r, c, dy, dx, peak)
    results_bp = []
    for r in r_grid:
        for c in c_grid:
            half = ps // 2
            a_patch = a0[r-half:r+half, c-half:c+half]
            bm_patch = bm[r-half:r+half, c-half:c+half]
            bp_patch = bp[r-half:r+half, c-half:c+half]
            if not (np.isfinite(a_patch).any()
                    and np.isfinite(bm_patch).any()
                    and np.isfinite(bp_patch).any()):
                results_bm.append((r, c, None, None, 0))
                results_bp.append((r, c, None, None, 0))
                continue
            a_eq = histogram_equalize(a_patch)
            bm_eq = histogram_equalize(bm_patch)
            bp_eq = histogram_equalize(bp_patch)
            dy_bm, dx_bm, p_bm = peak_offset(a_eq, bm_eq, search_radius=30)
            dy_bp, dx_bp, p_bp = peak_offset(a_eq, bp_eq, search_radius=30)
            results_bm.append((r, c, dy_bm, dx_bm, p_bm))
            results_bp.append((r, c, dy_bp, dx_bp, p_bp))

    print(f"\n{'Patch (r,c)':<15s} {'A0 vs B_minus dy,dx':>22s} {'peak':>6s}   "
          f"{'A0 vs B_plus dy,dx':>22s} {'peak':>6s}")
    print("-" * 90)
    for (r, c, dy_m, dx_m, p_m), (_, _, dy_p, dx_p, p_p) in zip(results_bm, results_bp):
        s_m = f"({dy_m:+d}, {dx_m:+d})" if dy_m is not None else "  N/A"
        s_p = f"({dy_p:+d}, {dx_p:+d})" if dy_p is not None else "  N/A"
        print(f"({r:5d}, {c:5d}) {s_m:>22s} {p_m:>6.3f}   {s_p:>22s} {p_p:>6.3f}")

    # Quick visual: show 1 patch with A0, B_minus, |A0-B_minus|
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plot_idx = 0
    for i, r in enumerate(r_grid):
        for j, c in enumerate(c_grid):
            ax = axes[i, j]
            half = ps // 2
            a_patch = a0[r-half:r+half, c-half:c+half]
            bm_patch = bm[r-half:r+half, c-half:c+half]
            a_eq = histogram_equalize(a_patch)
            bm_eq = histogram_equalize(bm_patch)
            diff = a_eq - bm_eq
            im = ax.imshow(diff, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
            rec_dy = results_bm[plot_idx][2]
            rec_dx = results_bm[plot_idx][3]
            label = f"r={r}, c={c}\n"
            if rec_dy is not None:
                label += f"peak offset (dy, dx) = ({rec_dy:+d}, {rec_dx:+d})"
            else:
                label += "no overlap"
            ax.set_title(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            plot_idx += 1

    fig.suptitle("A0 - B_minus (histogram-equalized), per patch with FFT-correlation peak",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved diagnostic figure to {args.out}")


if __name__ == "__main__":
    main()
