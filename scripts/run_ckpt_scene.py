"""Run a stereo retrieval for a single scene with a given checkpoint.

Produces a CF-1.8 compliant NetCDF file that can be fed to
compare_stereo_era5.py, compare_stereo_amv.py, etc.

Usage:
    python scripts/run_ckpt_scene.py --ckpt path/to.ckpt --time 2026-01-08T12:00 --out output.nc
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import numpy as np
import xarray as xr

from stereo_winds.config import GOES19_CONFIG, GOES18_CONFIG
from stereo_winds.disparity import StereoDisparity
from stereo_winds.output import create_output_dataset
from stereo_winds.solver import build_design_matrix, solve_stereo_winds, pixels_to_wind_ms
from stereo_winds.time_model import compute_scene_times

DATA_DIR = Path("/home/ubuntu/earthnet-us-east-3/data/stereo_training")
DT_MINUTES = 10.0
SAT_A = GOES19_CONFIG
SAT_B = GOES18_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="RAFT checkpoint path")
    parser.add_argument("--time", required=True, help="Center time (ISO, e.g. 2026-01-08T12:00)")
    parser.add_argument("--band", default="C14")
    parser.add_argument("--out", required=True, help="Output NetCDF path")
    parser.add_argument("--month", default=None, help="YYYYMM key for Zarr lookup (auto-detected if omitted)")
    args = parser.parse_args()

    t0 = dt.datetime.fromisoformat(args.time)
    delta = np.timedelta64(10, "m")
    t0_np = np.datetime64(args.time)

    # Find the right Zarr stores
    if args.month:
        month = args.month
    else:
        month = t0.strftime("%Y%m")

    # Try goes19 first, then goes16
    for sat_prefix in ["goes19", "goes16"]:
        a_path = DATA_DIR / f"{sat_prefix}_{args.band}_{month}.zarr"
        b_path = DATA_DIR / f"goes18_remap_{sat_prefix}_{args.band}_{month}.zarr"
        if a_path.exists() and b_path.exists():
            break
    else:
        raise FileNotFoundError(f"No Zarr stores found for {args.band} {month}")

    print(f"Loading {a_path.name} + {b_path.name}", flush=True)
    ds_a = xr.open_zarr(str(a_path))
    ds_b = xr.open_zarr(str(b_path))

    a_minus = ds_a.Rad.sel(time=t0_np - delta).values.astype(np.float32)
    a0 = ds_a.Rad.sel(time=t0_np).values.astype(np.float32)
    a_plus = ds_a.Rad.sel(time=t0_np + delta).values.astype(np.float32)
    b_minus = ds_b.Rad.sel(time=t0_np - delta).values.astype(np.float32)
    b_plus = ds_b.Rad.sel(time=t0_np + delta).values.astype(np.float32)

    valid = (np.isfinite(a0) & np.isfinite(a_minus) & np.isfinite(a_plus)
             & np.isfinite(b_minus) & np.isfinite(b_plus))
    images = {"A_minus": a_minus, "A0": a0, "A_plus": a_plus,
              "B_minus": b_minus, "B_plus": b_plus}

    # Parallax + design matrix
    pdata = np.load(str(DATA_DIR / "parallax_goes19_goes18.npz"))
    w_u, w_v = pdata["w_u"], pdata["w_v"]
    scene_times = compute_scene_times(t0, DT_MINUTES, SAT_A, SAT_B)
    H_matrix = build_design_matrix(
        w_u, w_v,
        dt_a_minus=scene_times["A_minus"], dt_a_plus=scene_times["A_plus"],
        dt_b_minus=scene_times["B_minus"], dt_b_plus=scene_times["B_plus"],
    )

    # Run RAFT
    print(f"Running RAFT: {args.ckpt}", flush=True)
    disp = StereoDisparity(model_ckpt_path=args.ckpt, tile_size=512, overlap=256,
                           batch_size=8, device="cuda")
    flows = disp.compute_all(images)
    for k in flows:
        flows[k][:, ~valid] = np.nan

    # Solve
    print("Solving stereo winds...", flush=True)
    solution = solve_stereo_winds(flows, H_matrix, sat_a=SAT_A, sat_b=SAT_B, n_iter=3)
    solution["quality_flag"][~valid] = 0.0

    # Create CF-1.8 dataset
    out_ds = create_output_dataset(solution, SAT_A, t0)
    out_ds.attrs["band"] = args.band
    out_ds.attrs["satellite_a"] = SAT_A.satellite_id
    out_ds.attrs["satellite_b"] = SAT_B.satellite_id
    out_ds.attrs["checkpoint"] = str(args.ckpt)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(str(out_path))
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
