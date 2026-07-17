"""Single-satellite student inference: full-disk u/v/h, teacher-overlap
comparison, and a quick PNG.

Loads a trained ``StudentWindsModel`` checkpoint (the radiance ``StandardScalar``
and the per-target z-score buffers are restored from the checkpoint state, so
inference normalization matches training), pulls one satellite's IR-band
radiance cubes at the requested time, runs RAFT for the per-channel temporal
flows, and forwards the 28-channel input stack through the model in row-strips
so the full disk fits on GPU.

Module-level helpers (``load_student``, ``open_cubes``, ``infer_one_time``)
are intended to be reused by multi-time runners; ``main()`` is the
single-time CLI that produces a NetCDF + PNG.

Output schema (per call to ``infer_one_time``) matches
``scripts/cache_stereo_retrievals.py`` so downstream evaluation tooling
treats the student cache as a drop-in for the stereo cache:

    u_wind            (m/s, east+)
    v_wind            (m/s, north+)
    cloud_top_height  (m)              # converted km -> m at the boundary
    quality_flag      (0 or 2)         # 2 where outputs are finite
    sigma_u           (m/s)
    sigma_v           (m/s)
    sigma_h           (m)
"""

import argparse
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from stereo_winds.config import SATELLITE_CONFIGS, SatelliteConfig
from stereo_winds.disparity import StereoDisparity
from stereo_winds.navigation import compute_pixel_scale, compute_grid_zenith
from stereo_winds.student_zeus_model import StudentWindsModel
from stereo_winds.student_dataset import (
    DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS,
    FLOW_SCALE, PIXEL_SCALE_NORM, ZENITH_NORM,
)

DT_DELTA = np.timedelta64(10, "m")     # temporal flow offset
TIME_TOL = np.timedelta64(5, "m")      # band-time matching tolerance


# ---------------------------------------------------------------------------
# Reusable helpers
# ---------------------------------------------------------------------------

def load_student(ckpt_path: str, device: str = "cuda") -> StudentWindsModel:
    """Load a ``StudentWindsModel`` from a Lightning checkpoint, eval-mode."""
    model = StudentWindsModel.load_from_checkpoint(ckpt_path, map_location=device)
    return model.eval()


def open_cubes(
    data_dir: str, sat_tag: str, yyyymm: str, bands,
) -> dict[str, xr.Dataset]:
    """Open per-band radiance Zarr cubes (chunks=None, fork-safe)."""
    cubes = {}
    for b in set(bands):
        p = os.path.join(data_dir, f"{sat_tag}_{b}_{yyyymm}.zarr")
        cubes[b] = xr.open_zarr(p, chunks=None)
    return cubes


def _stack_inputs(
    cubes: dict[str, xr.Dataset], sat: SatelliteConfig, t0,
    flow_bands, rad_bands, disp: StereoDisparity,
    rad_time_frames: int = 1,
):
    """Build the (C, H, W) input stack: flow/rad/geom with a finite-mask.

    ``rad_time_frames``:
      1 → single-frame rad at t₀ only (legacy).  Total rad channels = n_rad.
      3 → stack t-Δ, t₀, t+Δ per band, matching the 3-frame dataset's
          interleave order ``[t-, t₀, t+]`` per band.  Total rad
          channels = 3 * n_rad.
    """
    delta = DT_DELTA
    tol = TIME_TOL
    if rad_time_frames not in (1, 3):
        raise ValueError(f"rad_time_frames must be 1 or 3, got {rad_time_frames}")

    def load(b, t):
        return cubes[b].Rad.sel(time=t, method="nearest", tolerance=tol).values.astype(np.float32)

    flow_chans = []
    # a0c[b] = (a_m, a_0, a_p, valid) so the rad loop can reuse the three
    # frames already loaded for flow bands.
    a0c: dict[str, tuple] = {}
    for b in flow_bands:
        a_m, a_0, a_p = load(b, t0 - delta), load(b, t0), load(b, t0 + delta)
        valid = np.isfinite(a_m) & np.isfinite(a_0) & np.isfinite(a_p)
        fb = disp._run_pair(a_0, a_m); ff = disp._run_pair(a_0, a_p)
        for fl in (fb, ff):
            fl[:, ~valid] = np.nan
        flow_chans += [fb[0], fb[1], ff[0], ff[1]]
        a0c[b] = (a_m, a_0, a_p, valid)

    rad_chans = []
    for b in rad_bands:
        cached = a0c.get(b)
        if cached is None:
            if rad_time_frames == 3:
                a_m_b = load(b, t0 - delta)
                a_0_b = load(b, t0)
                a_p_b = load(b, t0 + delta)
                vb = (np.isfinite(a_m_b) & np.isfinite(a_0_b)
                      & np.isfinite(a_p_b))
            else:
                a_0_b = load(b, t0)
                vb = np.isfinite(a_0_b)
                a_m_b = a_p_b = None
        else:
            a_m_b, a_0_b, a_p_b, vb = cached
        if rad_time_frames == 3:
            for frame in (a_m_b, a_0_b, a_p_b):
                r = frame.copy(); r[~vb] = np.nan
                rad_chans.append(r)
        else:
            r = a_0_b.copy(); r[~vb] = np.nan
            rad_chans.append(r)

    dx_m, dy_m = compute_pixel_scale(sat)
    zen = compute_grid_zenith(sat)

    flow_arr = np.stack(flow_chans, 0).astype(np.float32) / FLOW_SCALE
    rad_arr = np.stack(rad_chans, 0).astype(np.float32)        # RAW; model standardizes
    geom_arr = np.stack(
        [dx_m / PIXEL_SCALE_NORM, dy_m / PIXEL_SCALE_NORM, zen / ZENITH_NORM], 0,
    ).astype(np.float32)
    finite_mask = (np.isfinite(flow_arr).all(0)
                   & np.isfinite(rad_arr).all(0))
    return (np.nan_to_num(flow_arr), np.nan_to_num(rad_arr),
            np.nan_to_num(geom_arr), finite_mask)


def _forward_full_disk(
    model: StudentWindsModel,
    flow_arr: np.ndarray, rad_arr: np.ndarray, geom_arr: np.ndarray,
    row_strip: int = 1024, halo: int = 8, device: str = "cuda",
    band_idx: int | None = None,
) -> dict[str, np.ndarray]:
    """Forward in row-strips; returns (H, W) arrays in physical units.

    For a multi-band model ``predict`` returns per-band ``(B, nb, H, W)``;
    ``band_idx`` selects one band so the rest of the pipeline stays single-band.
    """
    H, W = flow_arr.shape[1], flow_arr.shape[2]
    keys = ["u_mean", "v_mean", "h_mean", "u_logvar", "v_logvar", "h_logvar"]
    # If the trained model has a chi² head, surface it too.
    if getattr(model, "predict_chi2", False):
        keys.append("chi2")
    out = {k: np.full((H, W), np.nan, np.float32) for k in keys}
    with torch.no_grad():
        for r in range(0, H, row_strip):
            r1_keep = min(H, r + row_strip)
            r0_in = max(0, r - halo)
            r1_in = min(H, r1_keep + halo)
            ft = torch.from_numpy(flow_arr[:, r0_in:r1_in]).unsqueeze(0).to(device)
            rt = torch.from_numpy(rad_arr[:, r0_in:r1_in]).unsqueeze(0).to(device)
            gt = torch.from_numpy(geom_arr[:, r0_in:r1_in]).unsqueeze(0).to(device)
            o = model.predict(ft, rt, gt)
            # Multi-band: predict returns (B, nb, H, W) per key — pick one band.
            if band_idx is not None:
                o = {k: (v[:, band_idx] if v.ndim == 4 else v) for k, v in o.items()}
            keep0 = r - r0_in
            keep1 = keep0 + (r1_keep - r)
            for k in keys:
                out[k][r:r1_keep] = o[k][0, keep0:keep1].cpu().numpy()
    return out


def infer_one_time(
    model: StudentWindsModel,
    disp: StereoDisparity,
    cubes: dict[str, xr.Dataset],
    sat: SatelliteConfig,
    t0,
    flow_bands, rad_bands,
    row_strip: int = 1024,
    halo: int = 8,
    device: str = "cuda",
    band_idx: int | None = None,
) -> dict[str, np.ndarray]:
    """Full-disk student inference at one time.

    ``band_idx`` selects one output band for a multi-band model (else None).

    Returns a dict with the stereo-cache variable schema:
    ``u_wind``, ``v_wind`` (m/s), ``cloud_top_height`` (m),
    ``quality_flag`` (0/2), ``sigma_u``, ``sigma_v`` (m/s), ``sigma_h`` (m).
    """
    # Read rad_time_frames from the trained model so 3-frame ckpts get
    # their inputs assembled correctly without a CLI flag.
    rad_time_frames = int(getattr(model, "rad_time_frames", 1))
    flow_arr, rad_arr, geom_arr, finite_mask = _stack_inputs(
        cubes, sat, t0, flow_bands, rad_bands, disp,
        rad_time_frames=rad_time_frames,
    )
    raw = _forward_full_disk(model, flow_arr, rad_arr, geom_arr,
                             row_strip=row_strip, halo=halo, device=device,
                             band_idx=band_idx)
    u = np.where(finite_mask, raw["u_mean"], np.nan)
    v = np.where(finite_mask, raw["v_mean"], np.nan)
    h_km = np.where(finite_mask, raw["h_mean"], np.nan)
    sigma_u = np.exp(0.5 * raw["u_logvar"])
    sigma_v = np.exp(0.5 * raw["v_logvar"])
    sigma_h_km = np.exp(0.5 * raw["h_logvar"])

    valid = finite_mask & np.isfinite(u) & np.isfinite(v) & np.isfinite(h_km)
    qf = np.where(valid, 2.0, 0.0).astype(np.float32)
    # chi²: distilled head if available, else zeros (legacy student).
    if "chi2" in raw:
        chi2 = np.where(finite_mask, raw["chi2"], np.nan).astype(np.float32)
    else:
        chi2 = np.zeros_like(u, dtype=np.float32)
    return {
        "u_wind": u.astype(np.float32),
        "v_wind": v.astype(np.float32),
        "cloud_top_height": (h_km * 1000.0).astype(np.float32),  # m
        "quality_flag": qf,
        # Distilled teacher chi² (or zeros if the trained head wasn't enabled)
        # so the stereo-cache eval pipeline can apply the strict QA gate
        # without --qa-from teacher.
        "chi_squared": chi2,
        "sigma_u": sigma_u.astype(np.float32),
        "sigma_v": sigma_v.astype(np.float32),
        "sigma_h": (sigma_h_km * 1000.0).astype(np.float32),     # m
    }


def time_to_yyyymm(t0) -> str:
    s = str(np.asarray(t0).astype("datetime64[s]"))
    return s[:4] + s[5:7]


# ---------------------------------------------------------------------------
# Single-time CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="Student Lightning checkpoint")
    ap.add_argument("--raft-ckpt", required=True)
    ap.add_argument("--sat", default="goes19", choices=list(SATELLITE_CONFIGS))
    ap.add_argument("--time", required=True, help="ISO timestamp (e.g. 2025-07-01T12:00)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--flow-bands", default=",".join(DEFAULT_FLOW_BANDS))
    ap.add_argument("--rad-bands", default=",".join(DEFAULT_RAD_BANDS))
    ap.add_argument("--label-zarr", default=None)
    ap.add_argument("--row-strip", type=int, default=1024)
    ap.add_argument("--out-nc", required=True)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    flow_bands = [b for b in args.flow_bands.split(",") if b]
    rad_bands = [b for b in args.rad_bands.split(",") if b]
    sat = SATELLITE_CONFIGS[args.sat]
    t0 = np.datetime64(args.time)

    print(f"loading student: {args.ckpt}", flush=True)
    model = load_student(args.ckpt)
    if model.transform is not None:
        print(f"  rad mu     : {model.transform.mu['rad'].detach().cpu().numpy()}")
        print(f"  rad std    : {model.transform.sd['rad'].detach().cpu().numpy()}")
    print(f"  target mu  : {model.target_mu.detach().cpu().numpy()}  (u m/s, v m/s, h km)")
    print(f"  target sd  : {model.target_sd.detach().cpu().numpy()}")

    cubes = open_cubes(args.data_dir, args.sat, time_to_yyyymm(t0),
                       set(flow_bands) | set(rad_bands))
    disp = StereoDisparity(model_ckpt_path=args.raft_ckpt, tile_size=512, overlap=128,
                           batch_size=8, device="cuda")
    print("inference (full disk)...", flush=True)
    out = infer_one_time(model, disp, cubes, sat, t0, flow_bands, rad_bands,
                         row_strip=args.row_strip)

    u, v = out["u_wind"], out["v_wind"]
    h_m = out["cloud_top_height"]
    speed = np.sqrt(u ** 2 + v ** 2)

    xr.Dataset(
        {
            "u_wind": (("y", "x"), u), "v_wind": (("y", "x"), v),
            "wind_speed": (("y", "x"), speed),
            "cloud_top_height": (("y", "x"), h_m),
            "quality_flag": (("y", "x"), out["quality_flag"]),
            "u_sigma": (("y", "x"), out["sigma_u"]),
            "v_sigma": (("y", "x"), out["sigma_v"]),
            "h_sigma": (("y", "x"), out["sigma_h"]),
        },
        attrs={"time": str(t0), "satellite": sat.satellite_id,
               "flow_bands": ",".join(flow_bands), "rad_bands": ",".join(rad_bands)},
    ).to_netcdf(args.out_nc)
    print(f"saved {args.out_nc}", flush=True)

    if args.label_zarr:
        print("comparing to teacher on overlap...", flush=True)
        L = xr.open_zarr(args.label_zarr, chunks=None)
        ro = int(L.attrs.get("row_offset", 0)); co = int(L.attrs.get("col_offset", 0))
        bh, bw = L.sizes["y"], L.sizes["x"]
        tdiff = np.abs(np.asarray(L.time.values) - t0)
        i = int(tdiff.argmin())
        if tdiff[i] > TIME_TOL:
            print(f"  no teacher time within {TIME_TOL} of {t0}")
        else:
            ls = L.isel(time=i)
            u_t = ls["u_wind"].values; v_t = ls["v_wind"].values
            h_t = ls["cloud_top_height"].values  # meters
            qf = ls["quality_flag"].values
            u_s = u[ro:ro + bh, co:co + bw]
            v_s = v[ro:ro + bh, co:co + bw]
            h_s = h_m[ro:ro + bh, co:co + bw]
            m = ((qf >= 1) & np.isfinite(u_t) & np.isfinite(u_s)
                 & np.isfinite(v_t) & np.isfinite(v_s)
                 & np.isfinite(h_t) & np.isfinite(h_s))
            n = int(m.sum())
            print(f"  match time: {ls.time.values}   valid overlap pixels: {n}")
            if n > 1000:
                du, dv = u_s[m] - u_t[m], v_s[m] - v_t[m]
                dh = h_s[m] - h_t[m]
                print(f"  RMSVD       : {float(np.sqrt((du**2 + dv**2).mean())):6.2f} m/s")
                print(f"  RMSE u/v    : {float(np.sqrt((du**2).mean())):6.2f} / "
                      f"{float(np.sqrt((dv**2).mean())):6.2f} m/s")
                print(f"  RMSE h      : {float(np.sqrt((dh**2).mean())):6.0f} m")
                print(f"  bias u/v/h  : {float(du.mean()):+5.2f} / "
                      f"{float(dv.mean()):+5.2f} m/s   {float(dh.mean()):+5.0f} m")
                print(f"  corr u/v/h  : {float(np.corrcoef(u_s[m], u_t[m])[0, 1]):.3f} / "
                      f"{float(np.corrcoef(v_s[m], v_t[m])[0, 1]):.3f} / "
                      f"{float(np.corrcoef(h_s[m], h_t[m])[0, 1]):.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    ims = [
        axes[0, 0].imshow(u, cmap="RdBu_r", vmin=-40, vmax=40),
        axes[0, 1].imshow(v, cmap="RdBu_r", vmin=-40, vmax=40),
        axes[1, 0].imshow(speed, cmap="turbo", vmin=0, vmax=60),
        axes[1, 1].imshow(h_m / 1000.0, cmap="turbo", vmin=0, vmax=18),
    ]
    titles = ["u (m/s)", "v (m/s)", "wind speed (m/s)", "cloud-top height (km)"]
    for ax, im, t in zip(axes.flat, ims, titles):
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{sat.satellite_id} {t0}  (student inference, full disk)", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=120, bbox_inches="tight")
    print(f"saved {args.out_png}", flush=True)


if __name__ == "__main__":
    main()
