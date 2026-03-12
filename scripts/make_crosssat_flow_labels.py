"""Generate cross-satellite flow training labels from G5NR.

For each center timestep t0 (with neighbors t-1, t+1), produces:
  - A0:  BT on sat_a grid at t0
  - B0:  A0 warped by parallax(CTH at t0) — cross-sat view at t0
  - D1–D4: all 4 GT flow labels from the 8-vector observation model
  - height_m, wind_u, wind_v: ground truth at center time

Auto-discovers valid timestep triplets from G5NR files in the data directory.
Output is a single Zarr with dims (time, row, col).

Usage:
    python scripts/make_crosssat_flow_labels.py --data-dir /path/to/g5nr/
    python scripts/make_crosssat_flow_labels.py --data-dir data/  # local test
"""

import argparse
import glob
import os
import re

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

from stereo_winds.config import GOES16_CONFIG, GOES18_CONFIG, SatelliteConfig
from stereo_winds.navigation import fixed_grid_to_geodetic, pixel_to_scanning_angle
from stereo_winds.solver import compute_parallax_vectors

DT = 1800.0  # 30 minutes between G5NR snapshots

# Coarsen to G5NR native resolution (~0.0625° ≈ 7 km)
SCALE_RATIO = 0.0625 * 111000 / abs(GOES16_CONFIG.satellite_height_m * GOES16_CONFIG.scale_x)

sat_a = SatelliteConfig(
    satellite_id=GOES16_CONFIG.satellite_id,
    sub_lon_deg=GOES16_CONFIG.sub_lon_deg,
    satellite_height_m=GOES16_CONFIG.satellite_height_m,
    scale_x=GOES16_CONFIG.scale_x * SCALE_RATIO,
    scale_y=GOES16_CONFIG.scale_y * SCALE_RATIO,
    x_offset=GOES16_CONFIG.x_offset,
    y_offset=GOES16_CONFIG.y_offset,
    n_rows=int(GOES16_CONFIG.n_rows / SCALE_RATIO),
    n_cols=int(GOES16_CONFIG.n_cols / SCALE_RATIO),
    sweep=GOES16_CONFIG.sweep,
)
sat_b = SatelliteConfig(
    satellite_id=GOES18_CONFIG.satellite_id,
    sub_lon_deg=GOES18_CONFIG.sub_lon_deg,
    satellite_height_m=GOES18_CONFIG.satellite_height_m,
    scale_x=GOES18_CONFIG.scale_x * SCALE_RATIO,
    scale_y=GOES18_CONFIG.scale_y * SCALE_RATIO,
    x_offset=GOES18_CONFIG.x_offset,
    y_offset=GOES18_CONFIG.y_offset,
    n_rows=int(GOES18_CONFIG.n_rows / SCALE_RATIO),
    n_cols=int(GOES18_CONFIG.n_cols / SCALE_RATIO),
    sweep=GOES18_CONFIG.sweep,
)

PX_SCALE = abs(sat_a.satellite_height_m * sat_a.scale_x)
PX_KM = PX_SCALE / 1000


# ── File naming helpers ──────────────────────────────────────────────

def _met_path(data_dir, ts):
    return os.path.join(data_dir, f"c1440_NR.inst30mn_2d_met1_Nx.{ts}.nc4")


def _cloud_path(data_dir, ts):
    return os.path.join(data_dir, f"c1440_NR.inst30mn_3d_CLOUD_Nv.{ts}.nc4")


def _u_path(data_dir, ts):
    return os.path.join(data_dir, f"c1440_NR.inst30mn_3d_U_Nv.{ts}.nc4")


def _v_path(data_dir, ts):
    return os.path.join(data_dir, f"c1440_NR.inst30mn_3d_V_Nv.{ts}.nc4")


def discover_timesteps(data_dir):
    """Find all center times where t-1 and t+1 neighbors exist.

    Returns sorted list of (t_minus, t_center, t_plus) tuples.
    Requires 2d_met1 and 3d CLOUD/U/V at all 3 times.
    """
    pattern = os.path.join(data_dir, "c1440_NR.inst30mn_2d_met1_Nx.*.nc4")
    met_files = sorted(glob.glob(pattern))

    # Parse timestamps
    ts_re = re.compile(r"2d_met1_Nx\.(\d{8}_\d{4}z)\.nc4$")
    timestamps = []
    for f in met_files:
        m = ts_re.search(f)
        if m:
            timestamps.append(m.group(1))
    timestamps.sort()

    if len(timestamps) < 3:
        print(f"Found {len(timestamps)} met files, need at least 3 consecutive.")
        return []

    # Build set for O(1) lookup
    ts_set = set(timestamps)

    def _has_all_files(ts):
        """Check that 2d_met1 + 3d CLOUD/U/V all exist for a timestamp."""
        return (ts in ts_set
                and os.path.exists(_cloud_path(data_dir, ts))
                and os.path.exists(_u_path(data_dir, ts))
                and os.path.exists(_v_path(data_dir, ts)))

    # Find valid triplets: all 3 times need full file sets
    triplets = []
    for t_center in timestamps:
        t_minus = _prev_timestamp(t_center)
        t_plus = _next_timestamp(t_center)
        if not (_has_all_files(t_minus) and _has_all_files(t_center)
                and _has_all_files(t_plus)):
            continue
        triplets.append((t_minus, t_center, t_plus))

    return triplets


def _parse_timestamp(ts):
    """Parse '20060101_0030z' → (date_str, hour, minute)."""
    date_str = ts[:8]
    hour = int(ts[9:11])
    minute = int(ts[11:13])
    return date_str, hour, minute


def _format_timestamp(date_str, hour, minute):
    """Format back to '20060101_0030z'."""
    return f"{date_str}_{hour:02d}{minute:02d}z"


def _prev_timestamp(ts):
    """Get the timestamp 30 minutes before."""
    date_str, hour, minute = _parse_timestamp(ts)
    minute -= 30
    if minute < 0:
        minute += 60
        hour -= 1
    if hour < 0:
        hour += 24
        # Roll back date — simplified: decrement last 2 digits
        day = int(date_str[6:8]) - 1
        if day < 1:
            day = 28  # simplified, good enough for G5NR
            month = int(date_str[4:6]) - 1
            if month < 1:
                month = 12
                year = int(date_str[:4]) - 1
                date_str = f"{year:04d}{month:02d}{day:02d}"
            else:
                date_str = f"{date_str[:4]}{month:02d}{day:02d}"
        else:
            date_str = f"{date_str[:6]}{day:02d}"
    return _format_timestamp(date_str, hour, minute)


def _next_timestamp(ts):
    """Get the timestamp 30 minutes after."""
    date_str, hour, minute = _parse_timestamp(ts)
    minute += 30
    if minute >= 60:
        minute -= 60
        hour += 1
    if hour >= 24:
        hour -= 24
        day = int(date_str[6:8]) + 1
        # Simplified day rollover
        date_str = f"{date_str[:6]}{day:02d}"
    return _format_timestamp(date_str, hour, minute)


# ── Data loading ─────────────────────────────────────────────────────

def load_bt(data_dir, time_label):
    met = xr.open_dataset(_met_path(data_dir, time_label))
    cldtot = met["CLDTOT"].isel(time=0).values
    cldtmp = met["CLDTMP"].isel(time=0).values
    ts = met["TS"].isel(time=0).values
    cldtmp_filled = np.where(np.isfinite(cldtmp), cldtmp, ts)
    bt = cldtot * cldtmp_filled + (1 - cldtot) * ts
    return met["lat"].values, met["lon"].values, bt


def load_cth(data_dir, time_label):
    met = xr.open_dataset(_met_path(data_dir, time_label))
    cldtot = met["CLDTOT"].isel(time=0).values
    cldtmp = met["CLDTMP"].isel(time=0).values
    ts = met["TS"].isel(time=0).values
    cldprs = met["CLDPRS"].isel(time=0).values
    ps = met["PS"].isel(time=0).values
    cldtmp_filled = np.where(np.isfinite(cldtmp), cldtmp, ts)
    R, g = 287.05, 9.80665
    t_avg = 0.5 * (ts + cldtmp_filled)
    ratio = np.where(np.isfinite(cldprs) & (cldprs > 0), ps / cldprs, 1.0)
    cth = (R * t_avg / g) * np.log(np.maximum(ratio, 1.0))
    cth = np.where(cldtot > 0.01, cth * cldtot, 0.0)
    return met["lat"].values, met["lon"].values, cth


def load_cloud_top_wind(data_dir, time_label):
    met = xr.open_dataset(_met_path(data_dir, time_label))
    cldtot = met["CLDTOT"].isel(time=0).values
    cloud_3d = xr.open_dataset(_cloud_path(data_dir, time_label))["CLOUD"].isel(time=0).values
    u_3d = xr.open_dataset(_u_path(data_dir, time_label))["U"].isel(time=0).values
    v_3d = xr.open_dataset(_v_path(data_dir, time_label))["V"].isel(time=0).values
    cloudy = cloud_3d > 0.1
    has_cloud = cloudy.any(axis=0)
    top_level = cloudy.argmax(axis=0)
    lat_idx, lon_idx = np.meshgrid(
        np.arange(cloud_3d.shape[1]), np.arange(cloud_3d.shape[2]), indexing="ij"
    )
    u_cth = u_3d[top_level, lat_idx, lon_idx]
    v_cth = v_3d[top_level, lat_idx, lon_idx]
    u_cth = np.where(has_cloud & (cldtot > 0.01), u_cth, np.nan)
    v_cth = np.where(has_cloud & (cldtot > 0.01), v_cth, np.nan)
    lat = met["lat"].values
    lon = met["lon"].values
    return lat, lon, u_cth, v_cth


# ── Interpolation / warping ──────────────────────────────────────────

def make_interp(lat, lon, data, fill=np.nan):
    data_clean = np.where(np.isfinite(data), data, 0.0)
    return RegularGridInterpolator(
        (lat, lon), data_clean, method="linear", bounds_error=False, fill_value=fill,
    )


def compute_sat_latlon():
    rows, cols = np.arange(sat_a.n_rows), np.arange(sat_a.n_cols)
    cg, rg = np.meshgrid(cols, rows)
    x, y = pixel_to_scanning_angle(cg, rg, sat_a)
    return fixed_grid_to_geodetic(x, y, sat_a)


def sample(interp, lat_sat, lon_sat):
    pts = np.column_stack([lat_sat.ravel(), lon_sat.ravel()])
    return interp(pts).reshape(lat_sat.shape)


def warp(image, du, dv):
    H, W = image.shape
    rr, cc = np.meshgrid(np.arange(H, dtype=np.float64),
                         np.arange(W, dtype=np.float64), indexing="ij")
    return map_coordinates(image, [(rr - dv).ravel(), (cc - du).ravel()],
                           order=1, mode="constant", cval=np.nan).reshape(H, W)


# ── Per-timestep processing ──────────────────────────────────────────

def _sample_wind(data_dir, ts, lat_sat, lon_sat):
    """Load cloud-top wind at a timestep and sample onto sat grid.

    Returns (wu, wv) with NaN where no cloud.
    """
    _, _, u_raw, v_raw = load_cloud_top_wind(data_dir, ts)
    lat, lon = load_bt(data_dir, ts)[:2]  # just need lat/lon grid
    wu = sample(make_interp(lat, lon, u_raw), lat_sat, lon_sat)
    wv = sample(make_interp(lat, lon, v_raw), lat_sat, lon_sat)
    wmask = sample(make_interp(lat, lon, np.where(np.isfinite(u_raw), 1.0, 0.0), fill=0.0),
                   lat_sat, lon_sat)
    wu = np.where(wmask > 0.5, wu, np.nan)
    wv = np.where(wmask > 0.5, wv, np.nan)
    return wu, wv


def process_timestep(data_dir, t_minus, t_center, t_plus,
                     lat_sat, lon_sat, w_u, w_v, sl):
    """Process one timestep triplet using all 3 times.

    Uses time-appropriate CTH and wind for each flow label:
      D1 = -V_px(t-1) * dt                 (backward wind at t-1)
      D2 = +V_px(t+1) * dt                 (forward wind at t+1)
      D3 = w*h(t-1) - V_px(t-1) * dt       (parallax@t-1 + backward wind@t-1)
      D4 = w*h(t+1) + V_px(t+1) * dt       (parallax@t+1 + forward wind@t+1)

    Returns dict with: A0, B0, D1-D4 u/v, height_m, wind_u/v, valid
    """
    # Load BT at center time
    lat, lon, bt_center = load_bt(data_dir, t_center)
    A0 = sample(make_interp(lat, lon, bt_center), lat_sat, lon_sat)

    # Load CTH at all 3 times
    _, _, cth_minus = load_cth(data_dir, t_minus)
    _, _, cth_center = load_cth(data_dir, t_center)
    _, _, cth_plus = load_cth(data_dir, t_plus)
    cth_gm = np.nan_to_num(sample(make_interp(lat, lon, cth_minus), lat_sat, lon_sat))
    cth_g0 = np.nan_to_num(sample(make_interp(lat, lon, cth_center), lat_sat, lon_sat))
    cth_gp = np.nan_to_num(sample(make_interp(lat, lon, cth_plus), lat_sat, lon_sat))

    # Load wind at all 3 times
    wu_m, wv_m = _sample_wind(data_dir, t_minus, lat_sat, lon_sat)
    wu_0, wv_0 = _sample_wind(data_dir, t_center, lat_sat, lon_sat)
    wu_p, wv_p = _sample_wind(data_dir, t_plus, lat_sat, lon_sat)

    # Parallax at each time's CTH
    par_u_m, par_v_m = w_u * cth_gm, w_v * cth_gm
    par_u_0, par_v_0 = w_u * cth_g0, w_v * cth_g0
    par_u_p, par_v_p = w_u * cth_gp, w_v * cth_gp

    # B0: A0 warped by parallax at center time (cross-sat view)
    B0 = warp(A0, par_u_0, par_v_0)

    # Wind → pixel displacement at t-1 and t+1
    def wind_to_px(wu, wv):
        du = wu * DT / PX_SCALE
        dv = wv * DT / PX_SCALE
        return (np.where(np.isfinite(du), du, 0.0),
                np.where(np.isfinite(dv), dv, 0.0))

    wdu_m, wdv_m = wind_to_px(wu_m, wv_m)  # wind displacement at t-1
    wdu_p, wdv_p = wind_to_px(wu_p, wv_p)  # wind displacement at t+1

    # GT flow labels using time-appropriate state
    D1_u = -wdu_m              # temporal backward (wind@t-1)
    D1_v = -wdv_m
    D2_u = wdu_p               # temporal forward (wind@t+1)
    D2_v = wdv_p
    D3_u = par_u_m - wdu_m     # cross-sat backward (parallax@t-1 + wind@t-1)
    D3_v = par_v_m - wdv_m
    D4_u = par_u_p + wdu_p     # cross-sat forward (parallax@t+1 + wind@t+1)
    D4_v = par_v_p + wdv_p

    # Valid mask
    valid = (np.isfinite(A0) & np.isfinite(B0)
             & np.isfinite(par_u_0) & (cth_g0 > 100))

    # Crop and return
    return {
        "A0": A0[sl].astype(np.float32),
        "B0": B0[sl].astype(np.float32),
        "D1_u": D1_u[sl].astype(np.float32),
        "D1_v": D1_v[sl].astype(np.float32),
        "D2_u": D2_u[sl].astype(np.float32),
        "D2_v": D2_v[sl].astype(np.float32),
        "D3_u": D3_u[sl].astype(np.float32),
        "D3_v": D3_v[sl].astype(np.float32),
        "D4_u": D4_u[sl].astype(np.float32),
        "D4_v": D4_v[sl].astype(np.float32),
        "height_m": cth_g0[sl].astype(np.float32),
        "wind_u": wu_0[sl].astype(np.float32),
        "wind_v": wv_0[sl].astype(np.float32),
        "valid": valid[sl],
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate cross-sat flow labels from G5NR")
    parser.add_argument("--data-dir", required=True, help="Directory containing G5NR .nc4 files")
    parser.add_argument("--output", default="data/g5nr_crosssat_flow.zarr",
                        help="Output Zarr path (default: data/g5nr_crosssat_flow.zarr)")
    args = parser.parse_args()

    print(f"Grid: {sat_a.n_rows}x{sat_a.n_cols}, pixel scale: {PX_KM:.1f} km")

    # ── Discover available timesteps ──────────────────────────────────
    triplets = discover_timesteps(args.data_dir)
    if not triplets:
        print("No valid timestep triplets found. Need 2d_met1 files for 3 consecutive "
              "30-min steps, plus 3d CLOUD/U/V at center time.")
        return
    print(f"Found {len(triplets)} valid center time(s):")
    for t_m, t_c, t_p in triplets:
        print(f"  {t_m} → [{t_c}] → {t_p}")

    # ── Compute time-independent geometry (once) ──────────────────────
    print("\nComputing satellite grid...")
    lat_sat, lon_sat = compute_sat_latlon()

    print("Computing parallax vectors...")
    w_u, w_v = compute_parallax_vectors(sat_a, sat_b)

    # Bounding box from satellite overlap geometry (where parallax is defined)
    geom_valid = np.isfinite(w_u) & np.isfinite(lat_sat)
    rows_v, cols_v = np.where(geom_valid)
    r0, r1 = int(rows_v.min()), int(rows_v.max()) + 1
    c0, c1 = int(cols_v.min()), int(cols_v.max()) + 1
    sl = (slice(r0, r1), slice(c0, c1))
    print(f"Overlap bounding box: {r1-r0}x{c1-c0} at ({r0}:{r1}, {c0}:{c1})")

    # ── Process each timestep ─────────────────────────────────────────
    time_vars = ["A0", "B0",
                 "D1_u", "D1_v", "D2_u", "D2_v",
                 "D3_u", "D3_v", "D4_u", "D4_v",
                 "height_m", "wind_u", "wind_v", "valid"]

    for i, (t_m, t_c, t_p) in enumerate(triplets):
        print(f"\n[{i+1}/{len(triplets)}] {t_c}")
        result = process_timestep(args.data_dir, t_m, t_c, t_p,
                                  lat_sat, lon_sat, w_u, w_v, sl)

        n_valid = result["valid"].sum()
        print(f"  {n_valid:,} valid px")

        # Build single-timestep dataset with time dim
        data_vars = {}
        for key in time_vars:
            data_vars[key] = (["time", "row", "col"],
                              result[key][np.newaxis, ...])

        # Add time-independent parallax vectors on first write only
        if i == 0:
            data_vars["w_u"] = (["row", "col"], w_u[sl].astype(np.float32))
            data_vars["w_v"] = (["row", "col"], w_v[sl].astype(np.float32))

        ds = xr.Dataset(
            data_vars,
            coords={"time": [t_c]},
            attrs={
                "title": "Cross-satellite flow training labels from G5NR",
                "sat_a": sat_a.satellite_id,
                "sat_b": sat_b.satellite_id,
                "sub_lon_a": sat_a.sub_lon_deg,
                "sub_lon_b": sat_b.sub_lon_deg,
                "dt_seconds": DT,
                "pixel_scale_m": PX_SCALE,
                "pixel_scale_km": PX_KM,
                "row_offset": r0,
                "col_offset": c0,
                "description": (
                    "Cross-satellite flow labels for RAFT training. "
                    "A0 = BT on sat_a grid, B0 = A0 warped by parallax. "
                    "D1/D2 = temporal flow (pure wind), D3/D4 = cross-sat flow "
                    "(parallax + wind). All labels use analytical observation "
                    "model at center time."
                ),
            },
        )

        if i == 0:
            ds.to_zarr(args.output, mode="w")
        else:
            ds.to_zarr(args.output, append_dim="time")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\nSaved {args.output}")
    ds_out = xr.open_zarr(args.output)
    print(f"  Shape: time={ds_out.sizes['time']}, "
          f"row={ds_out.sizes['row']}, col={ds_out.sizes['col']}")
    print(f"  Times: {list(ds_out.time.values)}")
    print(f"  Variables: {list(ds_out.data_vars)}")


if __name__ == "__main__":
    main()
