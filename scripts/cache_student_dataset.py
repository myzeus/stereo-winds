"""Combined, cropped, chunked, resumable generator for the wind student.

Produces BOTH the teacher stereo labels and the single-satellite student
features in one RAFT pass per scene, evaluated on the satellite-overlap crop
(with a RAFT context margin), and writes ~``chunk_size``-scene Zarr files.

Why this exists (vs the separate cache_stereo_retrievals.py / cache_student_
features.py): those pre-allocate full-disk ``(n, 5424, 5424)`` arrays and OOM
well before 500 scenes, and the teacher runs full-disk (~90 s/scene).  Here the
retrieval runs only in the overlap (~3-5x cheaper, identical in the interior),
writes are chunked (memory-safe) and resumable (skip times already on disk).
The teacher's C14 temporal flows are reused as the student's C14 flow inputs.

Reuses the validated functions: ``solve_stereo_winds``, ``build_design_matrix``,
``pixels_to_wind_ms``, ``StereoDisparity``, ``compute_scene_times``,
``compute_pixel_scale``, ``compute_grid_zenith``.

Output store (one per chunk) is xbatcher-ready: per-scene targets + inputs on
the crop grid, read directly by ``stereo_winds.student_xbatcher``.

    targets : u_wind, v_wind, cloud_top_height, quality_flag,
              chi_squared, sigma_h, sigma_u, sigma_v          (time, y, x)
    inputs  : flow_{back,fwd}_{u,v}_<band>, rad_<band>        (time, y, x)
              dx_m, dy_m, sat_zenith                          (y, x)
    attrs   : row_offset, col_offset, flow_bands, rad_bands, teacher_band, satellite_id
"""

import sys
import os
import argparse
import glob

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "zeus"))

import numpy as np
import xarray as xr

from stereo_winds.config import SATELLITE_CONFIGS
from stereo_winds.solver import (
    build_design_matrix, compute_parallax_vectors, solve_stereo_winds,
    pixels_to_wind_ms,
)
from stereo_winds.disparity import StereoDisparity
from stereo_winds.time_model import compute_scene_times
from stereo_winds.navigation import compute_pixel_scale, compute_grid_zenith
from stereo_winds.qa import compute_qa_flag
from stereo_winds.student_dataset import (
    flow_var, rad_var, rad_tminus_var, rad_tplus_var,
    DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS,
)

DT_MINUTES = 10.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sat-a", default="goes19", choices=list(SATELLITE_CONFIGS))
    ap.add_argument("--sat-b", default="goes18", choices=list(SATELLITE_CONFIGS))
    ap.add_argument("--teacher-band", default="C14")
    ap.add_argument("--flow-bands", default=",".join(DEFAULT_FLOW_BANDS))
    ap.add_argument("--rad-bands", default=",".join(DEFAULT_RAD_BANDS))
    ap.add_argument("--months", required=True, help="Comma-separated YYYYMM list")
    ap.add_argument("--sonde-hours", default="0,12",
                    help="Comma-separated UTC hours to keep (default: 0,12 for IGRA); "
                         "empty string keeps all hours")
    ap.add_argument("--n-scenes", type=int, default=500, help="Total target scenes")
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--data-dir", required=True, help="Dir with the band zarr cubes")
    ap.add_argument("--valid-mask", required=True)
    ap.add_argument("--parallax", default=None,
                    help="parallax_<a>_<b>.npz (computed on the fly if missing)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--margin", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=128, help="RAFT tile overlap (px)")
    ap.add_argument("--n-iter", type=int, default=1)
    ap.add_argument("--solver-device", default="cuda")
    args = ap.parse_args()

    flow_bands = [b for b in args.flow_bands.split(",") if b]
    rad_bands = [b for b in args.rad_bands.split(",") if b]
    tb = args.teacher_band
    all_bands = list(dict.fromkeys(flow_bands + rad_bands + [tb]))
    SAT_A, SAT_B = SATELLITE_CONFIGS[args.sat_a], SATELLITE_CONFIGS[args.sat_b]
    months = [m.strip() for m in args.months.split(",") if m.strip()]
    delta = np.timedelta64(int(DT_MINUTES), "m")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"flow={flow_bands} rad={rad_bands} teacher={tb}", flush=True)

    # Parallax (full-disk), then crop to the overlap bbox.
    if args.parallax and os.path.exists(args.parallax):
        pz = np.load(args.parallax); w_u, w_v = pz["w_u"], pz["w_v"]
        print(f"parallax: {args.parallax}", flush=True)
    else:
        print("computing parallax...", flush=True)
        w_u, w_v = compute_parallax_vectors(SAT_A, SAT_B)

    valid_mask = np.load(args.valid_mask)
    rows, cols = np.where(valid_mask)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    bh, bw = r1 - r0, c1 - c0
    print(f"overlap bbox rows[{r0}:{r1}] cols[{c0}:{c1}] = {bh}x{bw}", flush=True)
    m = args.margin
    rr0, rr1 = max(0, r0 - m), min(SAT_A.n_rows, r1 + m)
    cc0, cc1 = max(0, c0 - m), min(SAT_A.n_cols, c1 + m)
    tr0, tc0 = r0 - rr0, c0 - cc0  # bbox start within the run window

    st = compute_scene_times(None, DT_MINUTES, SAT_A, SAT_B)
    H_matrix = build_design_matrix(
        w_u[r0:r1, c0:c1], w_v[r0:r1, c0:c1],
        dt_a_minus=st["A_minus"], dt_a_plus=st["A_plus"],
        dt_b_minus=st["B_minus"], dt_b_plus=st["B_plus"],
    )

    dx_m, dy_m = compute_pixel_scale(SAT_A)
    zen = compute_grid_zenith(SAT_A)
    geom = {"dx_m": dx_m[r0:r1, c0:c1].astype(np.float32),
            "dy_m": dy_m[r0:r1, c0:c1].astype(np.float32),
            "sat_zenith": zen[r0:r1, c0:c1].astype(np.float32)}
    x_rad = (np.arange(SAT_A.n_cols) * SAT_A.scale_x + SAT_A.x_offset)[c0:c1]
    y_rad = (np.arange(SAT_A.n_rows) * SAT_A.scale_y + SAT_A.y_offset)[r0:r1]

    # Cubes (lazy) keyed by (band, month) for A; teacher band for B.
    def a_cube(b, mm):
        return xr.open_zarr(os.path.join(args.data_dir, f"{args.sat_a}_{b}_{mm}.zarr"))

    def b_cube(mm):
        return xr.open_zarr(os.path.join(
            args.data_dir, f"{args.sat_b}_remap_{args.sat_a}_{tb}_{mm}.zarr"))

    # Eligible center times = ACTUAL cube times with t0+/-10 present in A and B
    # (the cubes are not aligned to HH:00).
    # Strict band-intersection eligibility: every flow band needs t0 +/- dt,
    # every rad band needs t0, B needs t0 +/- dt.  Per-band scan times are
    # offset by tens of seconds within an ABI frame but the cubes we have are
    # built at the same coarse cadence (sparse ~6/day on most bands).  Filtered
    # to sonde hours (00, 12 UTC by default) for IGRA collocation.
    sonde_hours = set(int(h) for h in args.sonde_hours.split(",") if h.strip()) \
        if args.sonde_hours else None
    eval_times = []
    for mm in months:
        try:
            asets = {b: set(a_cube(b, mm).time.values) for b in all_bands}
            tbset = set(b_cube(mm).time.values)
        except Exception as e:
            print(f"skip month {mm}: {e}", flush=True); continue
        for t0 in sorted(asets[tb]):
            if sonde_hours is not None:
                hh = int(np.asarray(t0).astype("datetime64[h]").astype("int64") % 24)
                if hh not in sonde_hours:
                    continue
            ok = all(t0 in asets[b] and t0 - delta in asets[b] and t0 + delta in asets[b]
                     for b in flow_bands)
            ok = ok and all(t0 in asets[b] for b in rad_bands)
            ok = ok and (t0 - delta in tbset) and (t0 + delta in tbset)
            if ok:
                eval_times.append((mm, t0))
    print(f"{len(eval_times)} eligible center-times "
          f"(sonde_hours={sorted(sonde_hours) if sonde_hours else 'all'})", flush=True)
    # Subsample evenly across time for diversity, down to the target count.
    if len(eval_times) > args.n_scenes:
        keep = np.unique(np.linspace(0, len(eval_times) - 1, args.n_scenes).round().astype(int))
        eval_times = [eval_times[i] for i in keep]

    # Resume: skip times already in existing chunk stores.
    done = set()
    existing = sorted(glob.glob(os.path.join(args.out_dir, "chunk_*.zarr")))
    for p in existing:
        try:
            done.update(np.asarray(xr.open_zarr(p).time.values))
        except Exception:
            pass
    eval_times = [(mm, t) for mm, t in eval_times if t not in done]
    chunk_idx = len(existing)
    print(f"{len(done)} done; processing {len(eval_times)} more "
          f"(from chunk {chunk_idx})", flush=True)
    if not eval_times:
        return

    disp = StereoDisparity(model_ckpt_path=args.ckpt, tile_size=512,
                           overlap=args.overlap, batch_size=8, device="cuda")

    FLOW_KEYS = ["flow_back_u", "flow_back_v", "flow_fwd_u", "flow_fwd_v"]
    buf = {k: [] for k in
           ["u_wind", "v_wind", "cloud_top_height", "quality_flag",
            "chi_squared", "sigma_h", "sigma_u", "sigma_v"]}
    for b in flow_bands:
        for k in FLOW_KEYS:
            buf[flow_var(k, b)] = []
    for b in rad_bands:
        buf[rad_var(b)] = []
        buf[rad_tminus_var(b)] = []
        buf[rad_tplus_var(b)] = []
    buf_times = []

    def flush():
        nonlocal chunk_idx, buf, buf_times
        if not buf_times:
            return
        dv = {k: (("time", "y", "x"), np.stack(v).astype(np.float32))
              for k, v in buf.items()}
        for k, v in geom.items():
            dv[k] = (("y", "x"), v)
        out = xr.Dataset(dv, coords={"time": np.array(buf_times),
                                     "y": y_rad, "x": x_rad})
        out.attrs.update(row_offset=r0, col_offset=c0,
                         flow_bands=",".join(flow_bands), rad_bands=",".join(rad_bands),
                         rad_time_frames=3,
                         teacher_band=tb, satellite_id=SAT_A.satellite_id)
        path = os.path.join(args.out_dir, f"chunk_{chunk_idx:03d}.zarr")
        out.to_zarr(path, mode="w")
        print(f"  wrote {path} ({len(buf_times)} scenes)", flush=True)
        chunk_idx += 1
        buf = {k: [] for k in buf}
        buf_times = []

    cur_mm, ca, cb = None, {}, None

    TOL = np.timedelta64(5, "m")  # per-band ABI scan offset is ~tens of seconds

    def load_run(ds, t):
        return ds.Rad.sel(time=t, method="nearest", tolerance=TOL).isel(
            y=slice(rr0, rr1), x=slice(cc0, cc1)).values.astype(np.float32)

    for i, (mm, t0) in enumerate(eval_times):
        if mm != cur_mm:
            ca = {b: a_cube(b, mm) for b in all_bands}
            cb = b_cube(mm)
            cur_mm = mm
        print(f"[{i+1}/{len(eval_times)}] {mm} {t0}", flush=True)
        crop = (slice(tr0, tr0 + bh), slice(tc0, tc0 + bw))
        try:
            # Per-band temporal flows on the run window, trimmed to bbox.
            # a0c[b] = (a_m, a_0, a_p, vb) — the three time frames plus the
            # 3-frame-finite mask.  We stash all three so the rad loop below
            # can reuse them as input frames (no second S3 load needed for
            # bands that also appear in flow_bands).
            a0c, flows_band = {}, {}
            for b in flow_bands:
                a_m, a_0, a_p = (load_run(ca[b], t0 - delta), load_run(ca[b], t0),
                                 load_run(ca[b], t0 + delta))
                vb = np.isfinite(a_m) & np.isfinite(a_0) & np.isfinite(a_p)
                fb = disp._run_pair(a_0, a_m); ff = disp._run_pair(a_0, a_p)
                for fl in (fb, ff):
                    fl[:, ~vb] = np.nan
                flows_band[b] = (fb[:, crop[0], crop[1]], ff[:, crop[0], crop[1]])
                a0c[b] = (a_m, a_0, a_p, vb)

            # Teacher cross-sat pairs (teacher band), trim to bbox.
            # a0c entries are (a_m, a_0, a_p, vb) since the temporal-frame change.
            tb_tuple = a0c.get(tb)
            if tb_tuple is None:
                a0_tb = load_run(ca[tb], t0); vtb = np.isfinite(a0_tb)
            else:
                _a_m, a0_tb, _a_p, vtb = tb_tuple
            b_m, b_p = load_run(cb, t0 - delta), load_run(cb, t0 + delta)
            valid = vtb & np.isfinite(b_m) & np.isfinite(b_p)
            d3 = disp._run_pair(a0_tb, b_m)[:, crop[0], crop[1]]
            d4 = disp._run_pair(a0_tb, b_p)[:, crop[0], crop[1]]
            vbbox = valid[crop[0], crop[1]]
            d1, d2 = flows_band[tb]  # reuse teacher-band temporal flows
            flows = {"D1": d1.copy(), "D2": d2.copy(), "D3": d3, "D4": d4}
            for k in flows:
                flows[k][:, ~vbbox] = np.nan

            sol = solve_stereo_winds(flows, H_matrix, sat_a=SAT_A, sat_b=SAT_B,
                                     n_iter=args.n_iter, device=args.solver_device)
            # u_wind / v_wind in m/s (pixels_to_wind for compute_qa_flag's speed gate).
            u_ms = sol["V_u"] * geom["dx_m"]
            v_ms = sol["V_v"] * geom["dy_m"]
            # Rich Level 0/1/2 QA from chi² / sigma_h / |∇h| / speed gates — same
            # logic backfill_qa_local.py applies; doing it inline keeps future
            # chunks self-consistent without needing a post-write backfill.
            qa_sol = {
                "h": sol["h"], "chi2": sol["chi2"], "sigma_h": sol["sigma_h"],
                "u_wind": u_ms, "v_wind": v_ms,
            }
            qf = compute_qa_flag(qa_sol, vbbox).astype(np.float32)
            qf[~vbbox] = 0.0
            # px/s -> m/s with the bbox-cropped ground scale (reuse u_ms/v_ms).
            rec = {
                "u_wind": u_ms, "v_wind": v_ms,
                "cloud_top_height": sol["h"], "quality_flag": qf,
                "chi_squared": sol["chi2"], "sigma_h": sol["sigma_h"],
                "sigma_u": sol["sigma_u"], "sigma_v": sol["sigma_v"],
            }
            for b in flow_bands:
                fb, ff = flows_band[b]
                rec[flow_var("flow_back_u", b)] = fb[0]; rec[flow_var("flow_back_v", b)] = fb[1]
                rec[flow_var("flow_fwd_u", b)] = ff[0]; rec[flow_var("flow_fwd_v", b)] = ff[1]
            # Radiance frames: t₀, t-Δ, t+Δ per band.  Reuse the three frames
            # already loaded for flow bands; load fresh for rad-only bands.
            for b in rad_bands:
                cached = a0c.get(b)
                if cached is None:
                    a_m_b = load_run(ca[b], t0 - delta)
                    a_0_b = load_run(ca[b], t0)
                    a_p_b = load_run(ca[b], t0 + delta)
                    vb = (np.isfinite(a_m_b) & np.isfinite(a_0_b)
                          & np.isfinite(a_p_b))
                else:
                    a_m_b, a_0_b, a_p_b, vb = cached
                # NB: must NOT shadow `r0` (row_offset) / `c0` (col_offset) which
                # are bound earlier in main() and used in the chunk attrs.
                _rad_t0 = a_0_b.copy(); _rad_t0[~vb] = np.nan
                _rad_tm = a_m_b.copy(); _rad_tm[~vb] = np.nan
                _rad_tp = a_p_b.copy(); _rad_tp[~vb] = np.nan
                rec[rad_var(b)] = _rad_t0[crop[0], crop[1]]
                rec[rad_tminus_var(b)] = _rad_tm[crop[0], crop[1]]
                rec[rad_tplus_var(b)] = _rad_tp[crop[0], crop[1]]
        except Exception as e:
            print(f"  skip {t0}: {type(e).__name__}: {e}", flush=True)
            continue

        for k in buf:
            buf[k].append(rec[k])
        buf_times.append(t0)
        if len(buf_times) >= args.chunk_size:
            flush()
    flush()
    print("done", flush=True)


if __name__ == "__main__":
    main()
