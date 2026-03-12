"""Verify flow training labels by running solver on the Zarr D1-D4.

Loads the pre-computed flow labels from g5nr_crosssat_flow.zarr,
feeds them into solve_stereo_winds(), and compares recovered height
and wind against ground truth stored in the same Zarr.
"""

import numpy as np
import xarray as xr

from stereo_winds.config import GOES16_CONFIG, SatelliteConfig
from stereo_winds.solver import (
    build_design_matrix,
    solve_stereo_winds,
    pixels_to_wind_ms,
)

ZARR_PATH = "data/g5nr_crosssat_flow.zarr"
CHI2_THRESH = 0.002


def main():
    ds = xr.open_zarr(ZARR_PATH)
    print(f"Loaded {ZARR_PATH}")
    print(f"  Shape: time={ds.sizes['time']}, row={ds.sizes['row']}, col={ds.sizes['col']}")
    print(f"  Times: {list(ds.time.values)}")

    dt = float(ds.attrs["dt_seconds"])
    px_scale = float(ds.attrs["pixel_scale_m"])

    # Reconstruct coarsened sat_a config for pixels_to_wind_ms
    SCALE_RATIO = px_scale / abs(GOES16_CONFIG.satellite_height_m * GOES16_CONFIG.scale_x)
    sat_a = SatelliteConfig(
        satellite_id=GOES16_CONFIG.satellite_id,
        sub_lon_deg=GOES16_CONFIG.sub_lon_deg,
        satellite_height_m=GOES16_CONFIG.satellite_height_m,
        scale_x=GOES16_CONFIG.scale_x * SCALE_RATIO,
        scale_y=GOES16_CONFIG.scale_y * SCALE_RATIO,
        x_offset=GOES16_CONFIG.x_offset,
        y_offset=GOES16_CONFIG.y_offset,
        n_rows=ds.sizes["row"],
        n_cols=ds.sizes["col"],
        sweep=GOES16_CONFIG.sweep,
    )

    # Parallax vectors (time-independent)
    w_u = ds["w_u"].values
    w_v = ds["w_v"].values

    for ti in range(ds.sizes["time"]):
        t_label = str(ds.time.values[ti])
        print(f"\n{'='*60}")
        print(f"Time {ti}: {t_label}")
        print(f"{'='*60}")

        # Load flow labels
        D1 = np.stack([ds["D1_u"].isel(time=ti).values, ds["D1_v"].isel(time=ti).values])
        D2 = np.stack([ds["D2_u"].isel(time=ti).values, ds["D2_v"].isel(time=ti).values])
        D3 = np.stack([ds["D3_u"].isel(time=ti).values, ds["D3_v"].isel(time=ti).values])
        D4 = np.stack([ds["D4_u"].isel(time=ti).values, ds["D4_v"].isel(time=ti).values])

        # Ground truth
        h_true = ds["height_m"].isel(time=ti).values
        wu_true = ds["wind_u"].isel(time=ti).values
        wv_true = ds["wind_v"].isel(time=ti).values
        valid = ds["valid"].isel(time=ti).values

        # Build design matrix and solve
        H_mat = build_design_matrix(w_u, w_v, -dt, dt, -dt, dt)
        sol = solve_stereo_winds({"D1": D1, "D2": D2, "D3": D3, "D4": D4}, H_mat)

        # Height comparison
        mask = valid & np.isfinite(sol["h"]) & (h_true > 100)
        h_err = sol["h"][mask] - h_true[mask]
        print(f"\nAll pixels ({mask.sum():,}):")
        print(f"  Height RMSE: {np.sqrt(np.mean(h_err**2)):.1f} m")
        print(f"  Height bias: {np.mean(h_err):+.1f} m")

        # Wind comparison
        u_rec, v_rec = pixels_to_wind_ms(sol["V_u"], sol["V_v"], sat_a, dt_seconds=1.0)
        wind_mask = mask & np.isfinite(wu_true) & np.isfinite(u_rec)
        if wind_mask.any():
            u_err = u_rec[wind_mask] - wu_true[wind_mask]
            v_err = v_rec[wind_mask] - wv_true[wind_mask]
            rmsvd = np.sqrt(np.mean(u_err**2 + v_err**2))
            print(f"  Wind RMSVD:  {rmsvd:.2f} m/s")

        print(f"  Chi2 mean:   {np.nanmean(sol['chi2'][mask]):.2e}")

        # Chi2 sweep
        print(f"\nChi2 filtering:")
        for thresh in [1.0, 0.1, 0.01, CHI2_THRESH, 0.001]:
            chi2_pass = mask & (sol["chi2"] < thresh)
            n = chi2_pass.sum()
            pct = 100 * n / mask.sum() if mask.sum() > 0 else 0
            he = sol["h"][chi2_pass] - h_true[chi2_pass]
            rmse_h = np.sqrt(np.mean(he**2)) if n > 0 else np.nan

            wm = chi2_pass & np.isfinite(wu_true) & np.isfinite(u_rec)
            if wm.any():
                ue = u_rec[wm] - wu_true[wm]
                ve = v_rec[wm] - wv_true[wm]
                wrmsvd = np.sqrt(np.mean(ue**2 + ve**2))
            else:
                wrmsvd = np.nan

            marker = " <<<" if thresh == CHI2_THRESH else ""
            print(f"  chi2<{thresh}: {n:,} px ({pct:.1f}%), "
                  f"h RMSE={rmse_h:.0f} m, wind RMSVD={wrmsvd:.2f} m/s{marker}")

        # Position correction
        print(f"\n  p_u mean: {np.nanmean(sol['p_u'][mask]):+.4f} px")
        print(f"  p_v mean: {np.nanmean(sol['p_v'][mask]):+.4f} px")


if __name__ == "__main__":
    main()
