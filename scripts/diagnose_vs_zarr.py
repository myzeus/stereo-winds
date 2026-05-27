#!/usr/bin/env python
"""Compare a freshly-cached scene against the equivalent slice of an existing
monthly zarr store. The historical good-result workflow used those zarrs;
if my cache differs, that's the bug.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--time", required=True,
                   help="ISO time (must be present in both --scene-dir cache and zarr)")
    p.add_argument("--scene-dir", required=True,
                   help="Directory with cached A0.npy etc.")
    p.add_argument("--zarr-a", required=True,
                   help="sat-A monthly zarr (e.g. goes16_C14_202501.zarr)")
    p.add_argument("--zarr-b", required=True,
                   help="sat-B-remapped monthly zarr (e.g. goes18_remap_goes16_C14_202501.zarr)")
    p.add_argument("--out", required=True, help="Output png")
    args = p.parse_args()

    t0 = np.datetime64(datetime.fromisoformat(args.time))
    scene_dir = Path(args.scene_dir)

    a_cache = np.load(scene_dir / "A0.npy").astype(np.float32)
    b_cache = np.load(scene_dir / "B_minus.npy").astype(np.float32)

    ds_a = xr.open_zarr(args.zarr_a)
    ds_b = xr.open_zarr(args.zarr_b)
    print(f"Zarr A times: first {ds_a.time.values[0]}, last {ds_a.time.values[-1]}")

    # Find t0 in zarr A
    matches_a = np.where(ds_a.time.values == t0)[0]
    matches_b_minus = np.where(ds_b.time.values == t0 - np.timedelta64(10, "m"))[0]
    if len(matches_a) == 0:
        print(f"ERROR: t0={t0} not in zarr A. Closest:",
              ds_a.time.values[np.argmin(np.abs(ds_a.time.values - t0))])
        sys.exit(1)
    if len(matches_b_minus) == 0:
        print(f"ERROR: t0-10min not in zarr B. Closest:",
              ds_b.time.values[np.argmin(np.abs(ds_b.time.values - (t0 - np.timedelta64(10, 'm'))))])
        sys.exit(1)

    a_zarr = ds_a.Rad.isel(time=matches_a[0]).values.astype(np.float32)
    b_zarr = ds_b.Rad.isel(time=matches_b_minus[0]).values.astype(np.float32)

    print(f"\nA0 comparison (sat-A at t0):")
    print(f"  cache: shape={a_cache.shape}, finite={100*np.isfinite(a_cache).mean():.1f}%, "
          f"range=[{np.nanmin(a_cache):.2g}, {np.nanmax(a_cache):.2g}]")
    print(f"  zarr:  shape={a_zarr.shape}, finite={100*np.isfinite(a_zarr).mean():.1f}%, "
          f"range=[{np.nanmin(a_zarr):.2g}, {np.nanmax(a_zarr):.2g}]")
    if a_cache.shape != a_zarr.shape:
        print("  SHAPE MISMATCH")
    else:
        diff = a_cache - a_zarr
        joint_finite = np.isfinite(a_cache) & np.isfinite(a_zarr)
        if joint_finite.any():
            print(f"  Δ on joint-finite ({int(joint_finite.sum()):,} px):")
            print(f"    mean={float(np.mean(diff[joint_finite])):+.3g}, "
                  f"std={float(np.std(diff[joint_finite])):.3g}, "
                  f"max|Δ|={float(np.max(np.abs(diff[joint_finite]))):.3g}")
            print(f"    finite agree: cache={int(np.isfinite(a_cache).sum()):,}, "
                  f"zarr={int(np.isfinite(a_zarr).sum()):,}, "
                  f"diff={int(np.isfinite(a_cache).sum() - np.isfinite(a_zarr).sum()):+,}")

    print(f"\nB_minus comparison (sat-B-remapped at t0-10min):")
    print(f"  cache: shape={b_cache.shape}, finite={100*np.isfinite(b_cache).mean():.1f}%, "
          f"range=[{np.nanmin(b_cache):.2g}, {np.nanmax(b_cache):.2g}]")
    print(f"  zarr:  shape={b_zarr.shape}, finite={100*np.isfinite(b_zarr).mean():.1f}%, "
          f"range=[{np.nanmin(b_zarr):.2g}, {np.nanmax(b_zarr):.2g}]")
    if b_cache.shape != b_zarr.shape:
        print("  SHAPE MISMATCH")
    else:
        diff_b = b_cache - b_zarr
        joint_finite_b = np.isfinite(b_cache) & np.isfinite(b_zarr)
        if joint_finite_b.any():
            print(f"  Δ on joint-finite ({int(joint_finite_b.sum()):,} px):")
            print(f"    mean={float(np.mean(diff_b[joint_finite_b])):+.3g}, "
                  f"std={float(np.std(diff_b[joint_finite_b])):.3g}, "
                  f"max|Δ|={float(np.max(np.abs(diff_b[joint_finite_b]))):.3g}")
            print(f"    finite agree: cache={int(np.isfinite(b_cache).sum()):,}, "
                  f"zarr={int(np.isfinite(b_zarr).sum()):,}, "
                  f"diff={int(np.isfinite(b_cache).sum() - np.isfinite(b_zarr).sum()):+,}")

    # Visual: 2x3 — A_cache, A_zarr, A_diff | B_cache, B_zarr, B_diff
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    stride = 8
    ds = lambda a: a[::stride, ::stride]

    def vmin_vmax(*arrs):
        finite = [a[np.isfinite(a)] for a in arrs if np.isfinite(a).any()]
        if not finite:
            return 0, 1
        return float(np.percentile(np.concatenate(finite), 1)), \
               float(np.percentile(np.concatenate(finite), 99))

    v0, v1 = vmin_vmax(a_cache, a_zarr)
    axes[0, 0].imshow(ds(a_cache), cmap="gray", vmin=v0, vmax=v1)
    axes[0, 0].set_title(f"A0 cached (from S3)\nfinite={100*np.isfinite(a_cache).mean():.1f}%",
                          fontsize=11)
    axes[0, 1].imshow(ds(a_zarr), cmap="gray", vmin=v0, vmax=v1)
    axes[0, 1].set_title(f"A0 zarr\nfinite={100*np.isfinite(a_zarr).mean():.1f}%",
                          fontsize=11)
    if a_cache.shape == a_zarr.shape:
        d = a_cache - a_zarr
        d = np.where(np.isfinite(d), d, 0)
        axes[0, 2].imshow(ds(d), cmap="RdBu_r", vmin=-5, vmax=5)
        axes[0, 2].set_title(f"A0 cache - A0 zarr (clipped ±5)", fontsize=11)

    v0b, v1b = vmin_vmax(b_cache, b_zarr)
    axes[1, 0].imshow(ds(b_cache), cmap="gray", vmin=v0b, vmax=v1b)
    axes[1, 0].set_title(f"B_minus cached (S3+local remap)\n"
                          f"finite={100*np.isfinite(b_cache).mean():.1f}%", fontsize=11)
    axes[1, 1].imshow(ds(b_zarr), cmap="gray", vmin=v0b, vmax=v1b)
    axes[1, 1].set_title(f"B_minus zarr\n"
                          f"finite={100*np.isfinite(b_zarr).mean():.1f}%", fontsize=11)
    if b_cache.shape == b_zarr.shape:
        d = b_cache - b_zarr
        d = np.where(np.isfinite(d), d, 0)
        axes[1, 2].imshow(ds(d), cmap="RdBu_r", vmin=-5, vmax=5)
        axes[1, 2].set_title(f"B_minus cache - B_minus zarr (clipped ±5)", fontsize=11)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Cached-from-S3 vs monthly-zarr at {args.time}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
