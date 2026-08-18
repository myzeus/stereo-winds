"""Single-satellite student inference on ABI sector products (e.g. RadC).

Downloads the radiance triplets (t-10, t0, t+10 min) for the student's flow
and rad bands straight from the public GOES S3 buckets, assembles in-memory
per-band cubes, and reuses ``infer_student_global.infer_one_time`` — the
model is convolutional and the geometry channels come from the sector
config, so no full-disk Zarr cubes are needed.

Output NetCDF carries the sector x/y scanning-angle coordinates (radians),
matching the stereo pipeline's output grid so the two can be merged.
"""

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))

import numpy as np
import xarray as xr

from stereo_winds.config import SATELLITE_CONFIGS, sector_config
from stereo_winds.data_loading import load_goes_scene
from stereo_winds.disparity import StereoDisparity
from stereo_winds.student_dataset import DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS
from infer_student_global import DT_DELTA, infer_one_time, load_student


def build_cubes(t0, bands, satellite, cache_dir, product):
    """Load (t-10, t0, t+10) per band as in-memory satpy-like cubes.

    Returns (cubes, sector_sat_config): ``cubes[band]`` is an xr.Dataset with
    ``Rad (time, y, x)`` row 0 = north, matching the training cube layout.
    """
    import datetime as dt

    t0_py = t0.astype("datetime64[s]").item()
    delta = dt.timedelta(minutes=int(DT_DELTA / np.timedelta64(1, "m")))
    times = [t0_py - delta, t0_py, t0_py + delta]

    cubes = {}
    runtime_cfg = None
    for b in sorted(bands):
        frames = []
        for t in times:
            # quantity="bt": the student's training cubes carry brightness
            # temperature (the ckpt's per-band rad stats are ~237-281 K);
            # feeding raw radiance breaks the height head. Flows are
            # unaffected (FlowRunner histogram-equalizes internally and
            # rad->BT is monotone).
            data, cfg = load_goes_scene(t, b, satellite, cache_dir,
                                        product=product, quantity="bt")
            frames.append(data)
            runtime_cfg = cfg
        cubes[b] = xr.Dataset(
            {"Rad": (("time", "y", "x"), np.stack(frames))},
            coords={"time": np.array(times, dtype="datetime64[ns]")},
        )
    canonical = SATELLITE_CONFIGS[satellite]
    return cubes, sector_config(canonical, runtime_cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="Student Lightning checkpoint")
    ap.add_argument("--raft-ckpt", required=True)
    ap.add_argument("--sat", default="goes19", choices=list(SATELLITE_CONFIGS))
    ap.add_argument("--time", required=True, help="ISO timestamp (e.g. 2026-08-11T20:00)")
    ap.add_argument("--product", default="ABI-L1b-RadC",
                    help="ABI L1b product (default: ABI-L1b-RadC)")
    ap.add_argument("--flow-bands", default=",".join(DEFAULT_FLOW_BANDS))
    ap.add_argument("--rad-bands", default=",".join(DEFAULT_RAD_BANDS))
    ap.add_argument("--eval-band", default="C14",
                    help="For a multi-band model, which band's winds to write")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--row-strip", type=int, default=512)
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--out-nc", required=True)
    args = ap.parse_args()

    flow_bands = [b for b in args.flow_bands.split(",") if b]
    rad_bands = [b for b in args.rad_bands.split(",") if b]
    t0 = np.datetime64(args.time)

    print(f"loading student: {args.ckpt}", flush=True)
    model = load_student(args.ckpt, device=args.device)
    band_idx = None
    if int(getattr(model, "n_bands", 1)) > 1:
        band_idx = flow_bands.index(args.eval_band)
        print(f"  multi-band model (n_bands={model.n_bands}); writing "
              f"{args.eval_band} (idx {band_idx})", flush=True)

    print("loading radiance triplets from S3/cache...", flush=True)
    cubes, sat = build_cubes(t0, set(flow_bands) | set(rad_bands),
                             args.sat, args.cache_dir, args.product)
    print(f"  grid {sat.n_rows}x{sat.n_cols}, x0={sat.x_offset:.6f} rad", flush=True)

    disp = StereoDisparity(model_ckpt_path=args.raft_ckpt, tile_size=512,
                           overlap=128, batch_size=8, device=args.device)
    print("student inference...", flush=True)
    out = infer_one_time(model, disp, cubes, sat, t0, flow_bands, rad_bands,
                         row_strip=args.row_strip, device=args.device,
                         band_idx=band_idx)

    x_rad = np.arange(sat.n_cols) * sat.scale_x + sat.x_offset
    y_rad = np.arange(sat.n_rows) * sat.scale_y + sat.y_offset
    ds = xr.Dataset(
        {k: (("y", "x"), v) for k, v in out.items()},
        coords={"x": ("x", x_rad), "y": ("y", y_rad)},
        attrs={"time": str(t0), "satellite": sat.satellite_id,
               "product": args.product, "model": "student",
               "eval_band": args.eval_band,
               "flow_bands": ",".join(flow_bands),
               "rad_bands": ",".join(rad_bands)},
    )
    out_path = Path(args.out_nc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    qf = out["quality_flag"]
    print(f"saved {out_path}  (valid: {(qf > 0).mean():.1%})", flush=True)


if __name__ == "__main__":
    main()
