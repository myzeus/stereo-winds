#!/usr/bin/env python
"""Diagnostic figure: acquisition-time models of GOES ABI vs MTG FCI.

(a) Per-row scan time within a full-disk scan for each sensor on its own
    grid: ABI scans north→south (row 0 earliest), FCI scans south→north
    (bottom rows earliest). Same nominal 10-min cadence, opposite phase.
(b) The per-pixel time difference  t_FCI − t_ABI  (seconds) for the SAME
    nominal slot, on the GOES-19 grid through the real remap LUT — the
    field that now multiplies the velocity columns of the stereo solver
    (via compute_scene_dt_fields). Same-ground-pixel observations differ
    by up to ±10 min despite identical nominal timestamps.

Runs offline (linear scan models; with cluster data the FCI side can use the native
per-pixel index_map→time-LUT field instead).

    python scripts/plot_sensor_timesteps.py [--out figures/sensor_timesteps.png]
"""
import argparse
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stereo_winds.config import GOES19_CONFIG, MTG_I1_CONFIG
from stereo_winds.navigation import compute_grid_latlon
from stereo_winds.remap import build_remap_lut, compute_valid_mask
from stereo_winds.time_model import compute_scene_dt_fields

FCI_N_SWATHS = 70  # FDHSI body chunks per repeat cycle (one S→N swath each)


def fci_row_times_south_north(n_rows: int, duration: float = 600.0) -> np.ndarray:
    """Illustrative FCI per-row time: S→N in discrete swaths.

    Row 0 = north (north-up orientation), scanned LAST. Quantized into
    FCI_N_SWATHS swaths to show the swath structure of the repeat cycle.
    """
    frac_from_south = (n_rows - 1 - np.arange(n_rows)) / n_rows
    swath = np.floor(frac_from_south * FCI_N_SWATHS) / FCI_N_SWATHS
    return swath * duration


def plot_fulldisk_maps(out_path: str):
    """Full-disk per-pixel acquisition-time map for each sensor, own grid."""
    from stereo_winds.navigation import compute_grid_latlon

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))

    for ax, sat, title, times in [
        (axes[0], GOES19_CONFIG, "GOES-19 ABI (N→S linear)",
         (np.arange(GOES19_CONFIG.n_rows) / GOES19_CONFIG.n_rows * 600.0)),
        (axes[1], MTG_I1_CONFIG, f"MTG-I1 FCI (S→N, {FCI_N_SWATHS} swaths)",
         fci_row_times_south_north(MTG_I1_CONFIG.n_rows)),
    ]:
        lat, _ = compute_grid_latlon(sat)
        on_disk = np.isfinite(lat)
        field = np.where(on_disk, times[:, None], np.nan)
        im = ax.imshow(field, cmap="viridis", vmin=0, vmax=600)
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("time within scan  [s]")

    fig.suptitle("Per-pixel acquisition time within one full-disk scan")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/sensor_timesteps.png")
    ap.add_argument("--out-fulldisk", default="figures/sensor_timesteps_fulldisk.png")
    args = ap.parse_args()

    plot_fulldisk_maps(args.out_fulldisk)

    abi, fci = GOES19_CONFIG, MTG_I1_CONFIG

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # ------------------------------------------------------------------ (a)
    ax = axes[0]
    rows_abi = np.arange(abi.n_rows)
    t_abi = rows_abi / abi.n_rows * 600.0
    rows_fci = np.arange(fci.n_rows)
    t_fci = fci_row_times_south_north(fci.n_rows)

    ax.plot(t_abi, rows_abi / abi.n_rows, label="GOES-19 ABI (N→S)", lw=2)
    ax.plot(t_fci, rows_fci / fci.n_rows, label="MTG-I1 FCI (S→N, 70 swaths)", lw=2)
    ax.set_xlabel("time within scan  [s]")
    ax.set_ylabel("normalized row (0 = north edge)")
    ax.invert_yaxis()  # row 0 (north) at top
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    ax.set_title("(a) Per-row acquisition time, one full-disk scan")

    # ------------------------------------------------------------------ (b)
    print("building GOES-19 → MTG-I1 remap LUT ...")
    col_b, row_b = build_remap_lut(abi, fci)
    valid = compute_valid_mask(col_b, row_b, abi, fci, max_zenith=80.0)

    t0 = datetime(2026, 1, 31, 13, 0)
    mk = lambda t: {"t_nominal": t, "t_start": t,
                    "t_end": t + timedelta(seconds=600), "pixel_time": None}
    # Trick: put "B" at the SAME nominal time as A0 so the dt field is purely
    # the cross-sensor scan-phase difference t_FCI − t_ABI.
    time_info = {
        "A_minus": mk(t0 - timedelta(minutes=10)), "A0": mk(t0),
        "A_plus": mk(t0 + timedelta(minutes=10)),
        "B_minus": mk(t0), "B_plus": mk(t0 + timedelta(minutes=10)),
    }
    # Give the B scenes the illustrative S→N swath field as native pixel_time
    base = (np.datetime64(t0) - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    pt_fci = base + fci_row_times_south_north(fci.n_rows)[:, None] * np.ones(
        (1, fci.n_cols))
    time_info["B_minus"]["pixel_time"] = pt_fci

    dts = compute_scene_dt_fields(time_info, abi, fci, col_b, row_b)
    dt_field = np.where(valid, dts["B_minus"], np.nan)

    ax = axes[1]
    im = ax.imshow(dt_field, cmap="RdBu_r", vmin=-600, vmax=600)
    ax.set_title("(b) t$_{FCI}$ − t$_{ABI}$, same nominal slot\n(GOES-19 grid, overlap lune)")
    ax.set_xticks([]), ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Δt  [s]")

    lat, _ = compute_grid_latlon(abi)
    fin = dt_field[np.isfinite(dt_field)]
    print(f"overlap pixels: {fin.size:,}  Δt range [{fin.min():+.0f}, {fin.max():+.0f}] s, "
          f"median {np.median(fin):+.0f} s")

    fig.suptitle("GOES-19 ABI vs MTG-I1 FCI scan timing (nominal 10-min cadence)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
