"""Generate single-satellite student input features from one satellite's cubes.

For each valid center time, computes per-channel TEMPORAL RAFT flows
(A0->A_minus and A0->A_plus) for each flow band plus the RAW A0 radiance for
each radiance band, and stores them with per-pixel viewing geometry to a Zarr.
These are the inputs the student model consumes; the teacher stereo retrieval
(``cache_stereo_retrievals.py``) supplies the matching targets.

Radiance is stored RAW (not histogram-equalized): the absolute brightness
carries cloud-top-height information, so the student standardizes it per band
at train time (z-score, the zeus/earthnetv2 ``StandardScalar`` convention)
rather than equalizing it away.

Flows are computed exactly as the teacher computes its temporal pairs: raw
radiance is passed to ``StereoDisparity._run_pair`` and FlowRunner equalizes
internally, so student inputs and teacher labels share flow statistics and the
north-positive v convention.

The store is cropped to the satellite-overlap bounding box (where teacher
labels are valid) with ``row_offset``/``col_offset`` attrs.  Variable layout
matches ``stereo_winds.student_dataset`` (flow_{back,fwd}_{u,v}_{band},
rad_{band}, dx_m, dy_m, sat_zenith).
"""

import sys
import os
import argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "zeus"))

import numpy as np
import xarray as xr

from stereo_winds.config import SATELLITE_CONFIGS
from stereo_winds.disparity import StereoDisparity
from stereo_winds.navigation import compute_pixel_scale, compute_grid_zenith
from stereo_winds.student_dataset import (
    flow_var, rad_var, FLOW_STUBS, DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="RAFT checkpoint path")
    parser.add_argument("--month", default="202601", help="YYYYMM")
    parser.add_argument("--n-scenes", type=int, default=10)
    parser.add_argument("--hours", default="00,12", help="Comma-separated hours")
    parser.add_argument("--flow-bands", default=",".join(DEFAULT_FLOW_BANDS),
                        help="Bands to compute temporal optical flow on")
    parser.add_argument("--rad-bands", default=",".join(DEFAULT_RAD_BANDS),
                        help="Bands whose A0 radiance is stored (default: all IR)")
    parser.add_argument("--sat-a-tag", default="goes19",
                        choices=list(SATELLITE_CONFIGS.keys()))
    parser.add_argument("--data-dir", default=None,
                        help="Dir with {sat}_{band}_{month}.zarr cubes "
                             "(default BASE/data/stereo_training)")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir (default BASE/data/student_features)")
    parser.add_argument("--out-name", default=None,
                        help="Output zarr name (default student_feat_{month}.zarr)")
    parser.add_argument("--valid-mask", required=True,
                        help="Full-disk overlap mask (.npy) defining the crop")
    parser.add_argument("--label-zarr", default=None,
                        help="Teacher cache zarr; restrict to its times so "
                             "features and labels match 1:1")
    parser.add_argument("--margin", type=int, default=256,
                        help="Extra context px around the overlap bbox for RAFT "
                             "tiling (trimmed away before storing)")
    parser.add_argument("--lowmem", action="store_true")
    args = parser.parse_args()

    DATA_DIR = args.data_dir or os.path.join(BASE, "data", "stereo_training")
    out_dir = args.out_dir or os.path.join(BASE, "data", "student_features")
    os.makedirs(out_dir, exist_ok=True)
    out_name = args.out_name or f"student_feat_{args.month}.zarr"
    out_path = os.path.join(out_dir, out_name)

    flow_bands = [b.strip() for b in args.flow_bands.split(",") if b.strip()]
    rad_bands = [b.strip() for b in args.rad_bands.split(",") if b.strip()]
    # Union, flow bands first (preserve a stable, documented order)
    all_bands = list(dict.fromkeys(flow_bands + rad_bands))
    sat = SATELLITE_CONFIGS[args.sat_a_tag]
    delta = np.timedelta64(10, "m")
    print(f"flow bands: {flow_bands}", flush=True)
    print(f"rad  bands: {rad_bands}", flush=True)

    # Open per-band cubes (union of flow + radiance bands)
    cubes = {}
    for b in all_bands:
        p = os.path.join(DATA_DIR, f"{args.sat_a_tag}_{b}_{args.month}.zarr")
        print(f"Band {b}: {p}", flush=True)
        cubes[b] = xr.open_zarr(p)

    # Overlap bounding box from the valid mask
    valid_mask = np.load(args.valid_mask)
    rows, cols = np.where(valid_mask)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    print(f"Overlap bbox: rows[{r0}:{r1}] cols[{c0}:{c1}] "
          f"({r1 - r0}x{c1 - c0})", flush=True)

    # RAFT run window = bbox + margin (clamped); trimmed back to bbox on store
    m = args.margin
    rr0, rr1 = max(0, r0 - m), min(sat.n_rows, r1 + m)
    cc0, cc1 = max(0, c0 - m), min(sat.n_cols, c1 + m)
    tr0, tr1 = r0 - rr0, r0 - rr0 + (r1 - r0)  # trim indices within the run window
    tc0, tc1 = c0 - cc0, c0 - cc0 + (c1 - c0)

    # Time selection (sat-A presence at t0±δ, t0 for the primary flow band)
    primary = cubes[flow_bands[0]]
    times_a = set(primary.time.values)
    year, month = int(args.month[:4]), int(args.month[4:])
    days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}[month]
    hours = [int(h) for h in args.hours.split(",")]
    eval_times = []
    for day in range(1, days + 1):
        for hour in hours:
            t0 = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00")
            if t0 in times_a and (t0 - delta) in times_a and (t0 + delta) in times_a:
                eval_times.append(t0)
    if args.label_zarr:
        label_times = set(xr.open_zarr(args.label_zarr).time.values)
        eval_times = [t for t in eval_times if t in label_times]
    eval_times = eval_times[:args.n_scenes]
    print(f"Processing {len(eval_times)} scenes", flush=True)
    if not eval_times:
        raise SystemExit("No eligible times — check cubes/label-zarr/hours.")

    # Geometry (time-independent), cropped to bbox
    dx_m, dy_m = compute_pixel_scale(sat)
    zen = compute_grid_zenith(sat)
    geom = {
        "dx_m": dx_m[r0:r1, c0:c1].astype(np.float32),
        "dy_m": dy_m[r0:r1, c0:c1].astype(np.float32),
        "sat_zenith": zen[r0:r1, c0:c1].astype(np.float32),
    }

    disp = StereoDisparity(model_ckpt_path=args.ckpt, tile_size=512, overlap=256,
                           batch_size=8, device="cuda", lowmem=args.lowmem)

    bh, bw = r1 - r0, c1 - c0
    n = len(eval_times)
    store = {flow_var(s, b): np.full((n, bh, bw), np.nan, np.float32)
             for b in flow_bands for s in FLOW_STUBS}
    for b in rad_bands:
        store[rad_var(b)] = np.full((n, bh, bw), np.nan, np.float32)

    def _load(b, t):
        return cubes[b].Rad.sel(time=t).isel(
            y=slice(rr0, rr1), x=slice(cc0, cc1)).values.astype(np.float32)

    for ti, t0 in enumerate(eval_times):
        print(f"  [{ti + 1}/{n}] {t0}", flush=True)
        a0_cache = {}  # band -> (a_0, valid) reused for radiance storage

        # Optical flow on the flow bands (raw radiance in; FlowRunner equalizes
        # internally, matching the teacher's temporal pairs)
        for b in flow_bands:
            a_m, a_0, a_p = _load(b, t0 - delta), _load(b, t0), _load(b, t0 + delta)
            valid = np.isfinite(a_m) & np.isfinite(a_0) & np.isfinite(a_p)
            f_back = disp._run_pair(a_0, a_m)  # (2, h, w), v north-positive
            f_fwd = disp._run_pair(a_0, a_p)
            for fl in (f_back, f_fwd):
                fl[:, ~valid] = np.nan
            store[flow_var("flow_back_u", b)][ti] = f_back[0, tr0:tr1, tc0:tc1]
            store[flow_var("flow_back_v", b)][ti] = f_back[1, tr0:tr1, tc0:tc1]
            store[flow_var("flow_fwd_u", b)][ti] = f_fwd[0, tr0:tr1, tc0:tc1]
            store[flow_var("flow_fwd_v", b)][ti] = f_fwd[1, tr0:tr1, tc0:tc1]
            a0_cache[b] = (a_0, np.isfinite(a_0))

        # RAW A0 radiance for every radiance band (reuse A0 where already
        # loaded for flow; otherwise load A0 only).  Standardized per band by
        # the dataset at train time; invalid pixels -> NaN.
        for b in rad_bands:
            a_0, valid = a0_cache.get(b) or (_load(b, t0), None)
            if valid is None:
                valid = np.isfinite(a_0)
            rad = a_0.copy()
            rad[~valid] = np.nan
            store[rad_var(b)][ti] = rad[tr0:tr1, tc0:tc1]

    data_vars = {k: (("time", "y", "x"), v) for k, v in store.items()}
    for k, v in geom.items():
        data_vars[k] = (("y", "x"), v)
    out_ds = xr.Dataset(data_vars, coords={"time": np.array(eval_times)})
    out_ds.attrs.update(
        row_offset=r0, col_offset=c0, satellite_id=sat.satellite_id,
        flow_bands=",".join(flow_bands), rad_bands=",".join(rad_bands),
    )
    out_ds.to_zarr(out_path, mode="w")
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
