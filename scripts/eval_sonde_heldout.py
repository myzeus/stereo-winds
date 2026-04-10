"""Evaluate a sonde-trained checkpoint on held-out IGRA stations.

For each held-out validation patch, runs RAFT + solver, extracts the wind
at the labeled station pixel, and compares to the closest sonde level.
Reports RMSVD, speed bias, height RMSE.
"""

import argparse
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import numpy as np
import torch
import xarray as xr
from torch.utils.data import ConcatDataset, DataLoader

from stereo_winds.dataset import IGRADataset
from stereo_winds.lightning_module import StereoWindsModule
from stereo_winds.validation.metrics import (
    correlation,
    height_rmse,
    rmsvd,
    speed_bias,
)


DATA_DIR = BASE / "data" / "stereo_training"
PARALLAX = str(DATA_DIR / "parallax_goes19_goes18.npz")
VALID_MASK = str(DATA_DIR / "valid_mask_g19_g18.npy")
IGRA_COLLOCATION = str(BASE / "data" / "igra" / "igra_all_collocation.parquet")
RAFT_CKPT = str(BASE / "zeus/zeus/networks/weights/raft-128.202509.epoch1434.ckpt")


def evaluate_checkpoint(ckpt_path: str, label: str, max_samples: int = 100):
    print(f"\n=== {label} ===", flush=True)
    print(f"  ckpt: {ckpt_path}", flush=True)

    # Build module and load checkpoint (handle PL or pretrained format)
    module = StereoWindsModule(raft_ckpt=ckpt_path, sonde_weight=1.0, iters=12)
    module.cuda().eval()

    # Build held-out dataset across all available months
    sat_a = sorted(DATA_DIR.glob("goes19_C14_*.zarr"))
    sat_b = sorted(DATA_DIR.glob("goes18_remap_goes19_C14_*.zarr"))
    per_month = []
    for a, b in zip(sat_a, sat_b):
        try:
            ds = IGRADataset(
                sat_a_zarr=str(a), sat_b_zarr=str(b),
                parallax_path=PARALLAX, valid_mask_path=VALID_MASK,
                igra_collocation_path=IGRA_COLLOCATION,
                train=False, seed=42,
            )
            if len(ds) > 0:
                per_month.append(ds)
        except Exception as e:
            print(f"  skipping {a.name}: {e}", flush=True)
    dataset = ConcatDataset(per_month)
    n_total = len(dataset)
    print(f"  Held-out samples: {n_total} (across {len(per_month)} months)", flush=True)

    if max_samples and n_total > max_samples:
        # Deterministic stride
        stride = max(1, n_total // max_samples)
        indices = list(range(0, n_total, stride))[:max_samples]
    else:
        indices = list(range(n_total))
    print(f"  Evaluating {len(indices)} samples", flush=True)

    u_pred_all, v_pred_all, h_pred_all = [], [], []
    u_son_all, v_son_all, h_son_all = [], [], []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            batch = {k: v.unsqueeze(0).cuda() for k, v in sample.items()}
            try:
                out = module.forward(batch)
            except Exception as e:
                print(f"  [{i}] forward failed: {e}", flush=True)
                continue
            h = out["h"][0]
            x_sol = out["x_sol"][0]
            v_u = x_sol[..., 3]
            v_v = x_sol[..., 4]
            u_ms = (v_u * batch["dx_m"][0])
            v_ms = (v_v * batch["dy_m"][0])

            mask = batch["sonde_pixel_mask"][0]
            sonde_h = batch["sonde_h"][0]
            sonde_u = batch["sonde_u"][0]
            sonde_v = batch["sonde_v"][0]
            sonde_mask = batch["sonde_mask"][0]

            # Process each labeled pixel
            ys, xs = torch.where(mask)
            for yp, xp in zip(ys.tolist(), xs.tolist()):
                hp = h[yp, xp].item()
                up = u_ms[yp, xp].item()
                vp = v_ms[yp, xp].item()
                if not (np.isfinite(hp) and np.isfinite(up) and np.isfinite(vp)):
                    continue
                # Closest sonde level (with bracketing)
                sh = sonde_h[yp, xp].cpu().numpy()
                su = sonde_u[yp, xp].cpu().numpy()
                sv = sonde_v[yp, xp].cpu().numpy()
                sm = sonde_mask[yp, xp].cpu().numpy()
                valid = sm & np.isfinite(sh) & np.isfinite(su) & np.isfinite(sv)
                if not valid.any():
                    continue
                sh_v = sh[valid]
                su_v = su[valid]
                sv_v = sv[valid]
                # Bracket check
                if not ((sh_v > hp).any() and (sh_v < hp).any()):
                    continue
                lev = int(np.argmin(np.abs(sh_v - hp)))
                u_pred_all.append(up)
                v_pred_all.append(vp)
                h_pred_all.append(hp)
                u_son_all.append(float(su_v[lev]))
                v_son_all.append(float(sv_v[lev]))
                h_son_all.append(float(sh_v[lev]))

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(indices)}] {len(u_pred_all)} matches", flush=True)

    n = len(u_pred_all)
    if n == 0:
        print("  No matches!", flush=True)
        return None

    u_pred_all = np.array(u_pred_all)
    v_pred_all = np.array(v_pred_all)
    h_pred_all = np.array(h_pred_all)
    u_son_all = np.array(u_son_all)
    v_son_all = np.array(v_son_all)
    h_son_all = np.array(h_son_all)

    rv = rmsvd(u_pred_all, v_pred_all, u_son_all, v_son_all)
    sb = speed_bias(u_pred_all, v_pred_all, u_son_all, v_son_all)
    hr = height_rmse(h_pred_all, h_son_all)
    hb = float(np.mean(h_pred_all - h_son_all))
    hc = correlation(h_pred_all, h_son_all)

    print(f"\n  N={n}  H_RMSE={hr:.0f}m  H_bias={hb:+.0f}m  H_corr={hc:.4f}  "
          f"RMSVD={rv:.2f}m/s  SpBias={sb:+.2f}m/s", flush=True)
    return dict(label=label, n=n, h_rmse=hr, h_bias=hb, h_corr=hc, rv=rv, sb=sb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint paths to evaluate")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Max held-out samples to evaluate per checkpoint")
    parser.add_argument("--baseline", action="store_true",
                        help="Also evaluate the pretrained baseline")
    args = parser.parse_args()

    results = []
    if args.baseline:
        r = evaluate_checkpoint(RAFT_CKPT, "Pretrained", max_samples=args.max_samples)
        if r:
            results.append(r)

    for ckpt in args.checkpoints:
        label = os.path.basename(ckpt).replace(".ckpt", "")
        r = evaluate_checkpoint(ckpt, label, max_samples=args.max_samples)
        if r:
            results.append(r)

    print("\n" + "=" * 100)
    print(f"{'Model':45s}  {'N':>5s}  {'H_RMSE':>7s}  {'H_bias':>7s}  "
          f"{'H_corr':>6s}  {'RMSVD':>7s}  {'SpBias':>7s}")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:45s}  {r['n']:>5,}  {r['h_rmse']:>7.0f}m  {r['h_bias']:>+7.0f}m  "
              f"{r['h_corr']:>6.4f}  {r['rv']:>6.2f}m/s  {r['sb']:>+6.2f}m/s")
    print("-" * 100)


if __name__ == "__main__":
    main()
