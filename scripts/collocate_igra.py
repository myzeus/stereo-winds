"""Collocate IGRA radiosonde profiles with GOES-19 pixel grid.

Produces a parquet with one row per (sonde launch, pressure level) for each
IGRA station that lies within the GOES-19 field of view. The parquet is used
by ``stereo_winds.dataset.IGRADataset`` to supervise stereo wind training
against radiosonde wind observations.

Usage:
    python scripts/collocate_igra.py --year 2026 --output data/igra/igra_2026_collocation.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import arraylake
import numpy as np
import pandas as pd
import xarray as xr

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from stereo_winds.config import GOES19_CONFIG
from stereo_winds.navigation import (
    geodetic_to_fixed_grid,
    scanning_angle_to_pixel,
)

logger = logging.getLogger(__name__)

SAT = GOES19_CONFIG
ARRAYLAKE_REPO = "zeus-ai/obs-xvec"


def collocate_year(
    year: int,
    max_time_diff_min: float = 5.0,
    max_height_m: float = 5000.0,
) -> pd.DataFrame:
    """Collocate one year of IGRA data with GOES-19 pixels.

    Args:
        year: Year to process (e.g. 2026).
        max_time_diff_min: Maximum allowed difference between sonde time and
            the snapped GOES 10-minute scan time, in minutes.
        max_height_m: Drop sonde levels above this altitude. Radiosondes drift
            laterally during ascent (30-50 km by 10 km altitude), so upper-air
            measurements are no longer over the launch station and produce
            misaligned labels.

    Returns:
        DataFrame with one row per (sonde, pressure level) for stations on
        the GOES-19 grid.
    """
    logger.info("Loading IGRA %d from ArrayLake (%s)...", year, ARRAYLAKE_REPO)
    client = arraylake.Client()
    repo = client.get_repo(ARRAYLAKE_REPO)
    session = repo.readonly_session("main")
    igra = xr.open_zarr(session.store, group=f"igra/{year}", consolidated=False)

    n_pressure = igra.sizes["pressure"]
    n_time = igra.sizes["time"]
    n_stations = igra.sizes["geometry"]
    logger.info(
        "  IGRA dims: pressure=%d, time=%d, stations=%d",
        n_pressure, n_time, n_stations,
    )

    # Station coordinates (small, eager load)
    lats = igra["lat"].values
    lons = igra["lon"].values

    # Project stations onto GOES-19 grid
    x_fg, y_fg = geodetic_to_fixed_grid(lats, lons, SAT)
    cols_f, rows_f = scanning_angle_to_pixel(x_fg, y_fg, SAT)
    on_grid = np.isfinite(cols_f) & np.isfinite(rows_f)
    cols_i = np.where(on_grid, np.round(cols_f).astype(int), -1)
    rows_i = np.where(on_grid, np.round(rows_f).astype(int), -1)
    on_grid &= (
        (rows_i >= 0) & (rows_i < SAT.n_rows)
        & (cols_i >= 0) & (cols_i < SAT.n_cols)
    )
    station_idx = np.where(on_grid)[0]
    logger.info("  Stations on GOES-19 grid: %d / %d", len(station_idx), n_stations)

    if len(station_idx) == 0:
        return pd.DataFrame()

    # Load only the on-grid stations
    logger.info("  Loading sonde profiles for on-grid stations...")
    igra_sub = igra.isel(geometry=station_idx).load()
    logger.info("    Loaded %d × %d × %d records", n_pressure, n_time, len(station_idx))

    # Reshape into rows
    times = igra_sub.time.values  # (n_time,)
    pressures = igra_sub.pressure.values  # (n_pressure,)
    u = igra_sub["u"].values  # (n_pressure, n_time, n_stations)
    v = igra_sub["v"].values
    h = igra_sub["geopotential_height"].values

    # Filter to standard IGRA launch times: 00 UTC and 12 UTC
    sonde_hours = pd.DatetimeIndex(times).hour.values
    is_launch = (sonde_hours == 0) | (sonde_hours == 12)
    logger.info("  Sonde times at 00/12 UTC: %d / %d", is_launch.sum(), n_time)

    # Snap to nearest GOES 10-min scan
    goes_time = pd.DatetimeIndex(times).round("10min")
    time_diff = np.abs(times - goes_time.values).astype("timedelta64[s]").astype(np.int64)
    time_ok = is_launch & (time_diff <= int(max_time_diff_min * 60))
    logger.info(
        "  Sonde times at 00/12Z within ±%.0f min of GOES scan: %d / %d",
        max_time_diff_min, time_ok.sum(), n_time,
    )

    # Filter to valid (level, time, station) combinations
    valid = (
        np.isfinite(u) & np.isfinite(v) & np.isfinite(h)
        & time_ok[np.newaxis, :, np.newaxis]
        & (h <= max_height_m)  # drop upper-air levels (sonde drift)
    )
    n_valid = int(valid.sum())
    logger.info(
        "  Valid (level, time, station) records (h <= %.0f m): %d",
        max_height_m, n_valid,
    )

    if n_valid == 0:
        return pd.DataFrame()

    p_idx, t_idx, s_idx = np.where(valid)
    df = pd.DataFrame({
        "station_idx": station_idx[s_idx].astype(np.int32),
        "lat": lats[station_idx[s_idx]].astype(np.float32),
        "lon": lons[station_idx[s_idx]].astype(np.float32),
        "row_19": rows_i[station_idx[s_idx]].astype(np.int16),
        "col_19": cols_i[station_idx[s_idx]].astype(np.int16),
        "sonde_time": times[t_idx],
        "goes_time": goes_time[t_idx],
        "pressure_hpa": pressures[p_idx].astype(np.float32),
        "u": u[p_idx, t_idx, s_idx].astype(np.float32),
        "v": v[p_idx, t_idx, s_idx].astype(np.float32),
        "height_m": h[p_idx, t_idx, s_idx].astype(np.float32),
    })

    # Order by goes_time, station, pressure for fast groupby downstream
    df = df.sort_values(["goes_time", "station_idx", "pressure_hpa"]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026, help="Year to process")
    parser.add_argument(
        "--output", required=True, help="Output parquet path",
    )
    parser.add_argument(
        "--max-time-diff-min", type=float, default=5.0,
        help="Maximum sonde-to-GOES time difference in minutes",
    )
    parser.add_argument(
        "--max-height-m", type=float, default=5000.0,
        help="Drop sonde levels above this altitude (sonde drift)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    df = collocate_year(
        args.year,
        max_time_diff_min=args.max_time_diff_min,
        max_height_m=args.max_height_m,
    )
    if len(df) == 0:
        logger.warning("No collocations produced; not writing output.")
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    logger.info("=" * 60)
    logger.info("Wrote %d records to %s", len(df), output)
    logger.info("  Stations: %d", df["station_idx"].nunique())
    logger.info("  GOES scan times: %d", df["goes_time"].nunique())
    logger.info("  Pressure levels: %d", df["pressure_hpa"].nunique())
    logger.info(
        "  Wind speed range: u=[%.1f, %.1f], v=[%.1f, %.1f] m/s",
        df["u"].min(), df["u"].max(), df["v"].min(), df["v"].max(),
    )
    logger.info(
        "  Height range: [%.0f, %.0f] m",
        df["height_m"].min(), df["height_m"].max(),
    )


if __name__ == "__main__":
    main()
