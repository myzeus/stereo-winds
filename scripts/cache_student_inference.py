"""Multi-time student inference -> one Zarr matching the stereo-cache schema.

Writes a ``(time, y, x)`` Zarr with the **same variables as
``cache_stereo_retrievals.py``** so the existing IGRA evaluator
(``eval_from_parquet.py``) consumes it as a drop-in for a stereo cache.

The output store is **pre-allocated** with the full chronological time
coordinate up-front (a dask-lazy zero-template, no large allocations), and
each scene is then written via xarray's ``to_zarr(region=...)``.  This
preserves the time-coord ordering — the earlier ``append_dim="time"`` pattern
silently scrambled order, which broke time-based joins downstream.

Resume is tracked via a small ``_written`` sentinel array stored alongside the
data variables; on restart we skip slots whose flag is set.
"""

import argparse
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import dask.array as da
import numpy as np
import xarray as xr

from stereo_winds.config import SATELLITE_CONFIGS
from stereo_winds.disparity import StereoDisparity
from stereo_winds.student_dataset import DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS
from infer_student_global import (
    load_student, open_cubes, infer_one_time, infer_one_time_allbands,
    time_to_yyyymm,
)

VAR_KEYS = ("u_wind", "v_wind", "cloud_top_height", "quality_flag",
            "chi_squared", "sigma_u", "sigma_v", "sigma_h")


# ---------------------------------------------------------------------------
# Reusable helpers (importable from tests + downstream callers)
# ---------------------------------------------------------------------------

def collect_eval_times(teacher_zarr_paths, max_scenes: int = -1) -> np.ndarray:
    """Sorted union of teacher chunk time coords (dtype ``datetime64[ns]``)."""
    times: set = set()
    for p in teacher_zarr_paths:
        times |= set(np.asarray(xr.open_zarr(p, chunks=None).time.values))
    out = sorted(times)
    if max_scenes > 0:
        out = out[:max_scenes]
    return np.asarray(out, dtype="datetime64[ns]")


def prealloc_template(
    out_zarr: str, eval_times_arr: np.ndarray, H: int, W: int, attrs: dict,
) -> None:
    """Create the output Zarr with dask-lazy zero data + canonical time coord."""
    n = len(eval_times_arr)
    data_vars = {
        k: (("time", "y", "x"),
            da.zeros((n, H, W), dtype=np.float32, chunks=(1, H, W)))
        for k in VAR_KEYS
    }
    # Tiny sentinel: per-time "did we write this scene yet" (uint8).
    data_vars["_written"] = (
        ("time",), da.zeros((n,), dtype=np.uint8, chunks=(min(max(n, 1), 1024),)),
    )
    template = xr.Dataset(data_vars, coords={"time": eval_times_arr}, attrs=attrs)
    # compute=False writes the zarr structure only; data chunks default to 0.
    template.to_zarr(out_zarr, mode="w", compute=False)


def write_scene(out_zarr: str, i: int, arrs: dict[str, np.ndarray]) -> None:
    """Region-write one scene's full-disk arrays into time-slot ``i``.

    ``arrs`` must contain every key in :data:`VAR_KEYS` as a ``(H, W)`` array;
    the time coord at slot ``i`` is taken from the pre-allocated template, so
    we don't pass any time coord here.
    """
    scene = xr.Dataset(
        {k: (("time", "y", "x"), arrs[k][np.newaxis]) for k in VAR_KEYS},
    )
    scene.to_zarr(out_zarr, mode="r+", region={"time": slice(i, i + 1)})
    flag = xr.Dataset({"_written": (("time",), np.array([1], dtype=np.uint8))})
    flag.to_zarr(out_zarr, mode="r+", region={"time": slice(i, i + 1)})


def _resume_todo(out_zarr: str) -> np.ndarray:
    """Indices of time slots that still need to be filled."""
    written = np.asarray(
        xr.open_zarr(out_zarr, chunks=None)["_written"].values
    ).astype(bool)
    return np.where(~written)[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--raft-ckpt", required=True)
    ap.add_argument("--sat", default="goes19", choices=list(SATELLITE_CONFIGS))
    ap.add_argument("--data-dir", required=True,
                    help="dir with {sat}_{band}_{yyyymm}.zarr cubes")
    ap.add_argument("--teacher-zarr", nargs="+", required=True,
                    help="Teacher chunk zarr(s); the union of their time coords "
                         "defines the times to infer over.")
    ap.add_argument("--flow-bands", default=",".join(DEFAULT_FLOW_BANDS))
    ap.add_argument("--rad-bands", default=",".join(DEFAULT_RAD_BANDS))
    ap.add_argument("--row-strip", type=int, default=1024)
    ap.add_argument("--eval-band", default="C14",
                    help="For a multi-band model, which flow band's winds to "
                         "write (single-band models ignore this).")
    ap.add_argument("--all-bands", action="store_true",
                    help="Multi-band: write every band from ONE forward pass. "
                         "--out-zarr must contain a '{band}' token; one store "
                         "per band is created (e.g. student_quad_{band}.zarr).")
    ap.add_argument("--out-zarr", required=True)
    ap.add_argument("--max-scenes", type=int, default=-1,
                    help="If >0, cap the number of scenes (debug).")
    args = ap.parse_args()

    flow_bands = [b for b in args.flow_bands.split(",") if b]
    rad_bands = [b for b in args.rad_bands.split(",") if b]
    sat = SATELLITE_CONFIGS[args.sat]
    H, W = sat.n_rows, sat.n_cols

    print(f"loading student: {args.ckpt}", flush=True)
    model = load_student(args.ckpt)
    print(f"  target mu  : {model.target_mu.detach().cpu().numpy()}  (u m/s, v m/s, h km)")
    print(f"  target sd  : {model.target_sd.detach().cpu().numpy()}")
    # Multi-band model: select which band's winds to write (band order == flow_bands).
    band_idx = None
    if int(getattr(model, "n_bands", 1)) > 1:
        band_idx = flow_bands.index(args.eval_band)
        print(f"  multi-band model (n_bands={model.n_bands}); writing band "
              f"{args.eval_band} (idx {band_idx})", flush=True)

    eval_times_arr = collect_eval_times(args.teacher_zarr, args.max_scenes)
    n_scenes = len(eval_times_arr)
    print(f"{n_scenes} candidate times across {len(args.teacher_zarr)} teacher chunks",
          flush=True)

    attrs = {"satellite_id": sat.satellite_id,
             "flow_bands": ",".join(flow_bands), "rad_bands": ",".join(rad_bands)}

    # Which bands (and stores) we write. all-bands => one store per flow band,
    # produced from a single forward pass; else the single --eval-band store.
    if args.all_bands:
        if band_idx is None:
            raise ValueError("--all-bands requires a multi-band model")
        if "{band}" not in args.out_zarr:
            raise ValueError("--all-bands: --out-zarr must contain a '{band}' token")
        stores = {b: args.out_zarr.format(band=b) for b in flow_bands}
    else:
        stores = {args.eval_band if band_idx is not None else "_single": args.out_zarr}

    for path in stores.values():
        if not os.path.exists(path):
            print(f"creating template store: {path}", flush=True)
            prealloc_template(path, eval_times_arr, H, W, attrs)
        else:
            existing_t = np.asarray(xr.open_zarr(path, chunks=None).time.values)
            if not np.array_equal(existing_t, eval_times_arr):
                raise ValueError(
                    f"existing store at {path} has a different time coord; "
                    "delete it or use a fresh --out-zarr")

    # Resume off the first store (all bands are written together per scene).
    todo = _resume_todo(next(iter(stores.values())))
    print(f"{n_scenes - len(todo)} scenes already filled; processing {len(todo)} new",
          flush=True)
    if len(todo) == 0:
        return

    disp = StereoDisparity(model_ckpt_path=args.raft_ckpt, tile_size=512, overlap=128,
                           batch_size=8, device="cuda")

    cubes_by_month: dict[str, dict] = {}

    def get_cubes(yyyymm: str):
        if yyyymm not in cubes_by_month:
            cubes_by_month[yyyymm] = open_cubes(
                args.data_dir, args.sat, yyyymm, set(flow_bands) | set(rad_bands),
            )
        return cubes_by_month[yyyymm]

    for k, i in enumerate(todo):
        t0 = eval_times_arr[int(i)]
        yyyymm = time_to_yyyymm(t0)
        try:
            cubes = get_cubes(yyyymm)
            print(f"[{k+1}/{len(todo)}] (slot {int(i)}) {t0}", flush=True)
            if args.all_bands:
                per_band = infer_one_time_allbands(
                    model, disp, cubes, sat, t0, flow_bands, rad_bands,
                    row_strip=args.row_strip,
                )
                arrs_by_store = {stores[b]: per_band[b] for b in flow_bands}
            else:
                arrs = infer_one_time(
                    model, disp, cubes, sat, t0, flow_bands, rad_bands,
                    row_strip=args.row_strip, band_idx=band_idx,
                )
                arrs_by_store = {args.out_zarr: arrs}
        except Exception as e:
            print(f"  skip slot {int(i)} ({t0}): {type(e).__name__}: {e}", flush=True)
            continue
        for path, arrs in arrs_by_store.items():
            write_scene(path, int(i), arrs)

    print("done", flush=True)


if __name__ == "__main__":
    main()
