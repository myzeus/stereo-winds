#!/usr/bin/env python
"""Evaluate cached stereo retrievals against IGRA sondes using a local parquet.

Mirrors the bracketing-match logic of ``eval_cached_igra.py`` but reads
the IGRA collocation parquet directly from disk (no arraylake needed,
since adapt is offline). The parquet must have columns:

    station_idx, lat, lon, row_19, col_19, sonde_time, goes_time,
    pressure_hpa, u, v, height_m

(as produced by ``scripts/collocate_igra.py``).

Each row is one (station, sonde_time, goes_time, pressure_level) sample.
Rows with the same (station, goes_time) form a vertical profile.

Example
-------
    python scripts/eval_from_parquet.py \\
        --parquet $DATA/labels/igra/igra_all_collocation.parquet \\
        --stereo  $NOBACKUP/stereo-winds/runs/cached/pretrained_202501_iter3.zarr \\
        --label   pretrained-2025-01 \\
        --plot-dir $NOBACKUP/stereo-winds/runs/cached/eval_plots
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import sobel

from stereo_winds.validation.metrics import (
    correlation, height_rmse, rmsvd, speed_bias,
)

logger = logging.getLogger(__name__)


QA = dict(chi2_max=0.2, sigma_h_max=5000.0, h_grad_max=3000.0,
          wind_speed_max=100.0, min_height=1000.0)


def height_gradient(h_2d):
    h_filled = np.where(np.isfinite(h_2d), h_2d, 0.0)
    gx = sobel(h_filled, axis=1)
    gy = sobel(h_filled, axis=0)
    return np.sqrt(gx**2 + gy**2) / 8.0


def neighborhood_median(grid, r_centers, c_centers, qa_mask, box=2):
    """5x5 neighborhood-median extraction at given station pixels.

    grid : (H, W) float32
    r_centers, c_centers : (n_sta,) int
    qa_mask : (H, W) bool
    returns: (n_sta,) median values (NaN where < 3 valid neighbors)
    """
    H, W = grid.shape
    offsets = np.arange(-box, box + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()
    r_nb = np.clip(r_centers[:, None] + dy[None, :], 0, H - 1)
    c_nb = np.clip(c_centers[:, None] + dx[None, :], 0, W - 1)
    vals = grid[r_nb, c_nb]
    qa_nb = qa_mask[r_nb, c_nb]
    vals = np.where(qa_nb, vals, np.nan)
    n_valid = np.sum(qa_nb, axis=1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(vals, axis=1)
    med = np.where(n_valid >= 3, med, np.nan)
    return med


def _build_qa_mask(ds):
    """Build the standard QA mask from a stereo dataset's solver outputs."""
    h_g = ds["cloud_top_height"].values
    u = ds["u_wind"].values
    v = ds["v_wind"].values
    chi2 = ds["chi_squared"].values
    sigh = ds["sigma_h"].values
    qf = ds["quality_flag"].values
    grad = height_gradient(h_g)
    spd = np.sqrt(u**2 + v**2)
    return (
        (qf > 0) & np.isfinite(h_g) & np.isfinite(chi2)
        & (chi2 <= QA["chi2_max"])
        & np.isfinite(sigh) & (sigh <= QA["sigma_h_max"])
        & (grad <= QA["h_grad_max"])
        & np.isfinite(spd) & (spd <= QA["wind_speed_max"])
        & (h_g >= QA["min_height"]) & (h_g <= 20000)
    )


def _pad_mask_to_shape(local_mask, target_shape, row_offset, col_offset):
    """Embed a cropped mask into a full-disk shape; False outside the crop."""
    out = np.zeros(target_shape, bool)
    h, w = local_mask.shape
    out[row_offset:row_offset + h, col_offset:col_offset + w] = local_mask
    return out


def evaluate_one_time(stereo, t_index, df_time, row_offset=0, col_offset=0,
                      qa_mask=None):
    """Evaluate one time step's stereo grid against sondes at `df_time`.

    stereo : xr.Dataset, single time slice (u_wind, v_wind, ...) loaded
    t_index : original time index (for logging only)
    df_time : parquet rows for goes_times within +/- 6h of stereo.time
    row_offset, col_offset : full-disk → chunk-local index shift. Parquet
        row_19/col_19 are full-disk goes-19 grid coordinates; cropped
        chunks (e.g. data_10ir_3t_*) have a non-zero offset and need it
        subtracted before indexing the local grid.
    qa_mask : optional pre-computed (H, W) bool mask aligned to the stereo
        grid. When supplied (e.g. derived from a teacher chunk's chi²),
        overrides the in-grid QA construction; useful when evaluating a
        student that doesn't emit its own chi².

    Returns a list of dicts (one per matched station) with keys:
        u_stereo, v_stereo, h_stereo,
        u_sonde, v_sonde, h_sonde, pressure_hpa, lat, lon, station_idx
    """
    matches = []
    if len(df_time) == 0:
        return matches

    u_grid = stereo["u_wind"].values
    v_grid = stereo["v_wind"].values
    h_grid = stereo["cloud_top_height"].values

    if qa_mask is None:
        qa_mask = _build_qa_mask(stereo)

    # Group by station to build profiles
    for (sidx,), grp in df_time.groupby(["station_idx"]):
        if len(grp) < 2:
            continue
        prof = grp.sort_values("pressure_hpa", ascending=False)
        sonde_h = prof["height_m"].values.astype(np.float32)
        sonde_u = prof["u"].values.astype(np.float32)
        sonde_v = prof["v"].values.astype(np.float32)
        sonde_p = prof["pressure_hpa"].values.astype(np.float32)

        sonde_valid = np.isfinite(sonde_h) & np.isfinite(sonde_u) & np.isfinite(sonde_v)
        if sonde_valid.sum() < 2:
            continue

        r = int(prof["row_19"].iloc[0]) - row_offset
        c = int(prof["col_19"].iloc[0]) - col_offset
        if not (0 <= r < h_grid.shape[0] and 0 <= c < h_grid.shape[1]):
            continue

        # Neighborhood-median stereo values at this station
        r_arr = np.array([r], dtype=int)
        c_arr = np.array([c], dtype=int)
        h_s = neighborhood_median(h_grid, r_arr, c_arr, qa_mask)[0]
        u_s = neighborhood_median(u_grid, r_arr, c_arr, qa_mask)[0]
        v_s = neighborhood_median(v_grid, r_arr, c_arr, qa_mask)[0]

        if not (np.isfinite(h_s) and np.isfinite(u_s) and np.isfinite(v_s)):
            continue

        # Bracketing-match the stereo height to the sonde profile
        h_diff = np.where(sonde_valid, np.abs(sonde_h - h_s), np.inf)
        best = int(np.argmin(h_diff))
        min_h_diff = float(h_diff[best])

        # Require bracketing
        has_above = np.any((sonde_h > h_s) & sonde_valid)
        has_below = np.any((sonde_h < h_s) & sonde_valid)
        if not (has_above and has_below):
            continue

        # Carr et al. height tolerance: 2 km above 500 hPa, 500 m below
        max_h = 2000.0 if sonde_p[best] <= 250 else 500.0
        if min_h_diff > max_h:
            continue

        matches.append(dict(
            station_idx=int(sidx),
            lat=float(prof["lat"].iloc[0]),
            lon=float(prof["lon"].iloc[0]),
            u_stereo=float(u_s), v_stereo=float(v_s), h_stereo=float(h_s),
            u_sonde=float(sonde_u[best]),
            v_sonde=float(sonde_v[best]),
            h_sonde=float(sonde_h[best]),
            pressure_hpa=float(sonde_p[best]),
        ))

    logger.info("  time idx %d: %d matched stations", t_index, len(matches))
    return matches


def _scalar_int(v, default=0):
    """Unwrap zarr v3 / xarray-serialized scalar ints (lists, ndarrays, etc.)."""
    if v is None:
        return default
    for _ in range(6):
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, (list, tuple)):
            if not v:
                return default
            v = v[0]
        else:
            break
    return int(v)


def _open_qa_source(paths):
    """Open one or more QA-source chunks; concat along time if multiple.

    Assumes all chunks share the same row_offset/col_offset crop (true for
    a single chunk-gen run).
    """
    dsets = []
    for p in paths:
        d = xr.open_zarr(p) if str(p).endswith(".zarr") else \
            xr.open_dataset(p, engine="h5netcdf")
        dsets.append(d)
    qa = xr.concat(dsets, dim="time") if len(dsets) > 1 else dsets[0]
    ro = _scalar_int(dsets[0].attrs.get("row_offset"), 0)
    co = _scalar_int(dsets[0].attrs.get("col_offset"), 0)
    return qa, ro, co


def evaluate_store(stereo_path, df, label, qa_from=None):
    """Evaluate a stereo store (zarr or netcdf) against the parquet.

    qa_from : optional list of paths to a teacher chunk store whose
        chi²/sigma_h/qf will define the QA mask, instead of the stereo
        store's own fields. Use this to evaluate a student cache that
        doesn't emit its own chi² — the mask is built from the teacher's
        solver at the matched time and padded to the student's grid.
    """
    if str(stereo_path).endswith(".zarr"):
        ds = xr.open_zarr(stereo_path)
    else:
        ds = xr.open_dataset(stereo_path, engine="h5netcdf")
    if "time" not in ds.dims:
        ds = ds.expand_dims("time")

    # Cropped chunks store row_offset/col_offset attrs giving the upper-left
    # full-disk index of the crop. The parquet's row_19/col_19 are full-disk
    # coordinates, so we need to subtract these to index the local grid.
    # Full-disk inference outputs default to 0.
    row_offset = _scalar_int(ds.attrs.get("row_offset"), 0)
    col_offset = _scalar_int(ds.attrs.get("col_offset"), 0)

    qa_ds, qa_ro, qa_co = (None, 0, 0)
    if qa_from:
        qa_ds, qa_ro, qa_co = _open_qa_source(qa_from)
        logger.info("QA source: %d times, crop=(row_offset=%d, col_offset=%d)",
                    qa_ds.sizes["time"], qa_ro, qa_co)

    times = ds.time.values
    logger.info("Evaluating %s: %d times (row_offset=%d, col_offset=%d)%s",
                label, len(times), row_offset, col_offset,
                " [QA from external]" if qa_ds is not None else "")

    df["goes_time"] = pd.to_datetime(df["goes_time"]).values.astype("datetime64[ns]")
    parquet_times = df["goes_time"].values

    all_matches = []
    for ti, t in enumerate(times):
        t_ns = np.datetime64(t).astype("datetime64[ns]")
        # Sondes within 6h of stereo time
        time_diff = np.abs(parquet_times - t_ns).astype("timedelta64[s]").astype(int)
        nearby = df[time_diff <= 6 * 3600]
        if len(nearby) == 0:
            logger.info("  time %s: 0 parquet rows within 6h", str(t)[:16])
            continue
        # Of those, restrict to the single closest goes_time
        nearest_t = nearby.iloc[
            (np.abs(nearby["goes_time"].values - t_ns)).argmin()
        ]["goes_time"]
        df_time = nearby[nearby["goes_time"] == nearest_t]
        slice_ = ds.isel(time=ti)
        slice_.load()

        external_mask = None
        if qa_ds is not None:
            # Match by nearest time in qa_ds (chunks usually share the same
            # 10-min cadence as the inference output).
            qa_times = qa_ds.time.values
            j = int(np.argmin(np.abs(qa_times.astype("datetime64[ns]") - t_ns)))
            # Tolerate small jitter; warn if the gap is > 30 min.
            dt = abs(int((qa_times[j] - t_ns).astype("timedelta64[s]")
                         .astype(np.int64)))
            if dt > 1800:
                logger.warning("  qa source nearest time off by %ds at %s",
                               dt, str(t)[:16])
            qa_slice = qa_ds.isel(time=j).load()
            local_mask = _build_qa_mask(qa_slice)
            stu_shape = slice_["cloud_top_height"].shape
            if local_mask.shape == stu_shape:
                external_mask = local_mask
            else:
                # qa source is cropped relative to stereo (full-disk student).
                # Pad with False outside the qa crop. Account for the stereo
                # store's own crop offset (rare; usually 0 for inference).
                external_mask = _pad_mask_to_shape(
                    local_mask, stu_shape,
                    qa_ro - row_offset, qa_co - col_offset)

        matches = evaluate_one_time(slice_, ti, df_time,
                                    row_offset=row_offset,
                                    col_offset=col_offset,
                                    qa_mask=external_mask)
        all_matches.extend(matches)
    return all_matches


def summarize(matches, label):
    """Print overall + per-layer stats."""
    if not matches:
        print(f"\n{label}: NO MATCHES")
        return None
    df = pd.DataFrame(matches)
    n = len(df)
    rv = rmsvd(df["u_stereo"], df["v_stereo"], df["u_sonde"], df["v_sonde"])
    sb = speed_bias(df["u_stereo"], df["v_stereo"], df["u_sonde"], df["v_sonde"])
    hr = height_rmse(df["h_stereo"], df["h_sonde"])
    hb = float(np.mean(df["h_stereo"] - df["h_sonde"]))
    spd_s = np.sqrt(df["u_stereo"]**2 + df["v_stereo"]**2)
    spd_r = np.sqrt(df["u_sonde"]**2 + df["v_sonde"]**2)
    cs = correlation(spd_s, spd_r)
    cu = correlation(df["u_stereo"], df["u_sonde"])
    cv = correlation(df["v_stereo"], df["v_sonde"])

    print("\n" + "=" * 80)
    print(f"{label}: {n} matched stations")
    print("=" * 80)
    print(f"  RMSVD          = {rv:.2f} m/s")
    print(f"  Speed bias     = {sb:+.2f} m/s")
    print(f"  Speed corr     = {cs:.3f}")
    print(f"  u corr         = {cu:.3f}")
    print(f"  v corr         = {cv:.3f}")
    print(f"  Height RMSE    = {hr:.0f} m")
    print(f"  Height bias    = {hb:+.0f} m")

    print("\nLayer-stratified stats:")
    print(f"  {'Layer':<22s} {'N':>6s} {'RMSVD':>8s} {'SpdBias':>8s} {'hRMSE':>8s}")
    layers = [
        ("Low (>=700 hPa)", df["pressure_hpa"] >= 700),
        ("Mid (400-700 hPa)", (df["pressure_hpa"] >= 400) & (df["pressure_hpa"] < 700)),
        ("High (<400 hPa)", df["pressure_hpa"] < 400),
    ]
    for name, mask in layers:
        sub = df[mask]
        n_l = len(sub)
        if n_l < 3:
            print(f"  {name:<22s} {n_l:>6d} {'--':>8s} {'--':>8s} {'--':>8s}")
            continue
        rv_l = rmsvd(sub["u_stereo"], sub["v_stereo"],
                     sub["u_sonde"], sub["v_sonde"])
        sb_l = speed_bias(sub["u_stereo"], sub["v_stereo"],
                          sub["u_sonde"], sub["v_sonde"])
        hr_l = height_rmse(sub["h_stereo"], sub["h_sonde"])
        print(f"  {name:<22s} {n_l:>6d} {rv_l:>8.2f} {sb_l:>+8.2f} {hr_l:>8.0f}")
    print("=" * 80)
    return df


def plot_eval(df_matches, label, plot_dir):
    """Quick speed + height scatter."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    spd_s = np.sqrt(df_matches["u_stereo"]**2 + df_matches["v_stereo"]**2)
    spd_r = np.sqrt(df_matches["u_sonde"]**2 + df_matches["v_sonde"]**2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    h_km = df_matches["h_stereo"] / 1000

    sc = axes[0].scatter(spd_r, spd_s, c=h_km, s=15, alpha=0.6,
                          cmap="turbo", vmin=0, vmax=16)
    lim = max(spd_r.max(), spd_s.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "k--", lw=0.8)
    axes[0].set_xlim(0, lim); axes[0].set_ylim(0, lim)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("Sonde speed (m/s)")
    axes[0].set_ylabel("Stereo speed (m/s)")
    rv = rmsvd(df_matches["u_stereo"], df_matches["v_stereo"],
                df_matches["u_sonde"], df_matches["v_sonde"])
    axes[0].set_title(f"Speed  n={len(df_matches)}  RMSVD={rv:.2f} m/s")

    h_st = df_matches["h_stereo"] / 1000
    h_so = df_matches["h_sonde"] / 1000
    axes[1].scatter(h_so, h_st, c=spd_s, s=15, alpha=0.6,
                    cmap="viridis", vmin=0, vmax=60)
    lim_h = max(h_so.max(), h_st.max()) * 1.05
    axes[1].plot([0, lim_h], [0, lim_h], "k--", lw=0.8)
    axes[1].set_xlim(0, lim_h); axes[1].set_ylim(0, lim_h)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("Sonde height (km)")
    axes[1].set_ylabel("Stereo height (km)")
    hr = height_rmse(df_matches["h_stereo"], df_matches["h_sonde"])
    axes[1].set_title(f"Height  RMSE={hr:.0f} m")

    plt.colorbar(sc, ax=axes, shrink=0.7, label="Stereo height (km)")
    fig.suptitle(f"Stereo vs IGRA — {label}", fontsize=13, fontweight="bold")
    path = plot_dir / f"eval_scatter_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True,
                        help="IGRA collocation parquet path")
    parser.add_argument("--stereo", nargs="+", required=True,
                        help="One or more stereo zarr / NetCDF paths")
    parser.add_argument("--label", default=None,
                        help="Optional label tag (defaults to basename of each --stereo)")
    parser.add_argument("--plot-dir", default=None,
                        help="If set, write scatter plots here")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all",
                        help="Station split to evaluate. 'val' = held-out stations "
                             "(station_idx %% 5 == 0), matching IGRADataset's split.")
    parser.add_argument("--qa-from", nargs="+", default=None,
                        help="Teacher chunk path(s) whose chi²/sigma_h/qf "
                             "define the QA mask, padded to the stereo grid. "
                             "Use this when --stereo is a student cache that "
                             "doesn't emit its own chi² (writes zeros). The "
                             "student inherits the teacher's QA gate so its "
                             "RMSVD is comparable to the teacher's IGRA score.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Loading parquet: %s", args.parquet)
    df = pd.read_parquet(args.parquet)
    logger.info("  %d rows, %d stations, %d goes_times",
                len(df), df["station_idx"].nunique(), df["goes_time"].nunique())

    # Match IGRADataset's deterministic station split (val = idx %% 5 == 0).
    if args.split != "all":
        is_val = (df["station_idx"].astype(int) % 5 == 0)
        keep = is_val if args.split == "val" else ~is_val
        df = df[keep]
        logger.info("  split=%s → %d rows, %d stations",
                    args.split, len(df), df["station_idx"].nunique())

    if args.label:
        # Aggregate matches across all --stereo stores into one summary.
        all_matches = []
        for stereo_path in args.stereo:
            all_matches.extend(evaluate_store(stereo_path, df, args.label,
                                              qa_from=args.qa_from))
        logger.info("Aggregated %d stores → %d total matches",
                    len(args.stereo), len(all_matches))
        df_m = summarize(all_matches, args.label)
        if args.plot_dir and df_m is not None and len(df_m) > 0:
            plot_eval(df_m, args.label, args.plot_dir)
    else:
        for stereo_path in args.stereo:
            label = Path(stereo_path).stem
            matches = evaluate_store(stereo_path, df, label,
                                     qa_from=args.qa_from)
            df_m = summarize(matches, label)
            if args.plot_dir and df_m is not None and len(df_m) > 0:
                plot_eval(df_m, label, args.plot_dir)


if __name__ == "__main__":
    main()
