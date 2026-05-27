"""Run RAFT + solver on a set of scenes and cache results to a Zarr store.

Caches u_wind, v_wind, cloud_top_height, chi_squared, sigma_h, sigma_u,
sigma_v, quality_flag for each (time) so downstream evaluations (and the
single-satellite student distillation) can reuse them without re-running RAFT.

Note: sigma_u/sigma_v come straight from the solver's velocity covariance and
are therefore in PIXELS/SECOND, not m/s — multiply by the per-pixel ground
scale (navigation.compute_pixel_scale) to convert. sigma_h is already meters.
"""

import sys
import os
import argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "zeus"))

import numpy as np
import xarray as xr

from stereo_winds.config import SATELLITE_CONFIGS, GOES19_CONFIG, GOES18_CONFIG
from stereo_winds.solver import (
    build_design_matrix, compute_parallax_vectors, solve_stereo_winds,
    pixels_to_wind_ms,
)
from stereo_winds.disparity import StereoDisparity
from stereo_winds.time_model import compute_scene_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--label", required=True, help="Label for output (e.g. 'pretrained', 'step400')")
    parser.add_argument("--month", default="202601", help="YYYYMM (default: 202601)")
    parser.add_argument("--n-scenes", type=int, default=10, help="Number of scenes to process")
    parser.add_argument("--hours", default="00,12", help="Comma-separated hours (default: 00,12)")
    parser.add_argument("--out-dir", default=None, help="Output Zarr directory")
    parser.add_argument("--n-iter", type=int, default=1, help="Solver iterations")
    parser.add_argument("--lowmem", action="store_true", help="Use lowmem (serial) RAFT")
    parser.add_argument("--data-dir", default=None,
                        help="Override DATA_DIR (default: BASE/data/stereo_training)")
    parser.add_argument("--sat-a-tag", default="goes19",
                        choices=list(SATELLITE_CONFIGS.keys()),
                        help="Sat-A id (also used in zarr filename pattern)")
    parser.add_argument("--sat-b-tag", default="goes18",
                        choices=list(SATELLITE_CONFIGS.keys()),
                        help="Sat-B id (also used in zarr filename pattern)")
    parser.add_argument("--band", default="C14",
                        help="ABI band tag in zarr filename (default: C14)")
    parser.add_argument("--parallax-cache", default=None,
                        help="Optional override for the parallax LUT npz path "
                             "(default: <data-dir>/parallax_<sat_a>_<sat_b>.npz). "
                             "Computed on the fly if missing.")
    args = parser.parse_args()

    DATA_DIR = args.data_dir or os.path.join(BASE, "data", "stereo_training")
    out_dir = args.out_dir or os.path.join(BASE, "data", "stereo_cache")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"iter{args.n_iter}"
    if args.lowmem:
        suffix += "_lowmem"
    out_path = os.path.join(out_dir, f"{args.label}_{args.month}_{suffix}.zarr")

    SAT_A = SATELLITE_CONFIGS[args.sat_a_tag]
    SAT_B = SATELLITE_CONFIGS[args.sat_b_tag]
    DT_MINUTES = 10.0

    # Parallax — load cache, or compute on the fly
    parallax_path = args.parallax_cache or os.path.join(
        DATA_DIR, f"parallax_{args.sat_a_tag}_{args.sat_b_tag}.npz",
    )
    if os.path.exists(parallax_path):
        print(f"Loading parallax cache: {parallax_path}", flush=True)
        pdata = np.load(parallax_path)
        w_u, w_v = pdata["w_u"], pdata["w_v"]
    else:
        print(f"Computing parallax {args.sat_a_tag} -> {args.sat_b_tag}...", flush=True)
        w_u, w_v = compute_parallax_vectors(SAT_A, SAT_B)
        if args.parallax_cache:
            np.savez_compressed(parallax_path, w_u=w_u, w_v=w_v)
            print(f"  Saved to {parallax_path}", flush=True)

    scene_times = compute_scene_times(None, DT_MINUTES, SAT_A, SAT_B)
    H_matrix = build_design_matrix(
        w_u, w_v,
        dt_a_minus=scene_times["A_minus"], dt_a_plus=scene_times["A_plus"],
        dt_b_minus=scene_times["B_minus"], dt_b_plus=scene_times["B_plus"],
    )

    # Load Zarr stores
    sat_a_zarr = os.path.join(
        DATA_DIR, f"{args.sat_a_tag}_{args.band}_{args.month}.zarr",
    )
    sat_b_zarr = os.path.join(
        DATA_DIR, f"{args.sat_b_tag}_remap_{args.sat_a_tag}_{args.band}_{args.month}.zarr",
    )
    print(f"Sat A: {sat_a_zarr}", flush=True)
    print(f"Sat B: {sat_b_zarr}", flush=True)
    ds_a = xr.open_zarr(sat_a_zarr)
    ds_b = xr.open_zarr(sat_b_zarr)
    all_times_a = set(ds_a.time.values)
    all_times_b = set(ds_b.time.values)
    delta = np.timedelta64(10, "m")

    # Find sonde-cadence scenes
    year = int(args.month[:4])
    month = int(args.month[4:])
    days_in_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                     7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}[month]
    hours = [int(h) for h in args.hours.split(",")]
    eval_times = []
    for day in range(1, days_in_month + 1):
        for hour in hours:
            t0 = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00")
            if (t0 in all_times_a and t0 in all_times_b
                    and t0 - delta in all_times_a and t0 - delta in all_times_b
                    and t0 + delta in all_times_a and t0 + delta in all_times_b):
                eval_times.append(t0)
    eval_times = eval_times[:args.n_scenes]
    print(f"Processing {len(eval_times)} scenes for {args.label}", flush=True)

    # Init RAFT
    disp = StereoDisparity(model_ckpt_path=args.ckpt, tile_size=512, overlap=256,
                           batch_size=8, device="cuda", lowmem=args.lowmem)

    # Storage
    H, W = 5424, 5424
    n = len(eval_times)
    u_arr = np.full((n, H, W), np.nan, dtype=np.float32)
    v_arr = np.full((n, H, W), np.nan, dtype=np.float32)
    h_arr = np.full((n, H, W), np.nan, dtype=np.float32)
    chi2_arr = np.full((n, H, W), np.nan, dtype=np.float32)
    sigh_arr = np.full((n, H, W), np.nan, dtype=np.float32)
    sigu_arr = np.full((n, H, W), np.nan, dtype=np.float32)
    sigv_arr = np.full((n, H, W), np.nan, dtype=np.float32)
    qf_arr = np.zeros((n, H, W), dtype=np.float32)

    for ti, t0 in enumerate(eval_times):
        print(f"  [{ti+1}/{n}] {t0}", flush=True)
        t_m = t0 - delta
        t_p = t0 + delta
        a_minus = ds_a.Rad.sel(time=t_m).values.astype(np.float32)
        a0 = ds_a.Rad.sel(time=t0).values.astype(np.float32)
        a_plus = ds_a.Rad.sel(time=t_p).values.astype(np.float32)
        b_minus = ds_b.Rad.sel(time=t_m).values.astype(np.float32)
        b_plus = ds_b.Rad.sel(time=t_p).values.astype(np.float32)
        valid = (np.isfinite(a0) & np.isfinite(a_minus) & np.isfinite(a_plus)
                 & np.isfinite(b_minus) & np.isfinite(b_plus))
        images = {"A_minus": a_minus, "A0": a0, "A_plus": a_plus,
                  "B_minus": b_minus, "B_plus": b_plus}

        flows = disp.compute_all(images)
        for k in flows:
            flows[k][:, ~valid] = np.nan

        sol = solve_stereo_winds(flows, H_matrix, sat_a=SAT_A, sat_b=SAT_B, n_iter=args.n_iter)
        u_ms, v_ms = pixels_to_wind_ms(sol["V_u"], sol["V_v"], SAT_A, dt_seconds=1.0)

        u_arr[ti] = u_ms
        v_arr[ti] = v_ms
        h_arr[ti] = sol["h"]
        chi2_arr[ti] = sol["chi2"]
        sigh_arr[ti] = sol["sigma_h"]
        sigu_arr[ti] = sol["sigma_u"]
        sigv_arr[ti] = sol["sigma_v"]
        qf = sol["quality_flag"].copy()
        qf[~valid] = 0.0
        qf_arr[ti] = qf

    # Write Zarr
    out_ds = xr.Dataset(
        {
            "u_wind": (("time", "y", "x"), u_arr),
            "v_wind": (("time", "y", "x"), v_arr),
            "cloud_top_height": (("time", "y", "x"), h_arr),
            "chi_squared": (("time", "y", "x"), chi2_arr),
            "sigma_h": (("time", "y", "x"), sigh_arr),
            "sigma_u": (("time", "y", "x"), sigu_arr),
            "sigma_v": (("time", "y", "x"), sigv_arr),
            "quality_flag": (("time", "y", "x"), qf_arr),
        },
        coords={"time": np.array(eval_times)},
    )
    out_ds["sigma_u"].attrs["units"] = "pixel s-1"
    out_ds["sigma_v"].attrs["units"] = "pixel s-1"
    out_ds["sigma_h"].attrs["units"] = "m"
    out_ds.to_zarr(out_path, mode="w")
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
