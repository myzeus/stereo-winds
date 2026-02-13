"""Integration tests for the stereo wind pipeline.

Tests the full pipeline with synthetic data (no network/GPU needed).
"""

import numpy as np

from stereo_winds.config import GOES16_CONFIG, GOES18_CONFIG, SatelliteConfig
from stereo_winds.navigation import pixel_to_scanning_angle, fixed_grid_to_geodetic
from stereo_winds.remap import build_remap_lut, remap_image
from stereo_winds.solver import (
    build_design_matrix,
    compute_parallax_vectors,
    pixels_to_wind_ms,
    solve_stereo_winds,
)
from stereo_winds.time_model import compute_scene_times
from stereo_winds.output import create_output_dataset
from datetime import datetime


def _small_config(base: SatelliteConfig, n: int = 32) -> SatelliteConfig:
    return SatelliteConfig(
        satellite_id=base.satellite_id,
        sub_lon_deg=base.sub_lon_deg,
        satellite_height_m=base.satellite_height_m,
        semi_major_m=base.semi_major_m,
        semi_minor_m=base.semi_minor_m,
        sweep=base.sweep,
        scale_x=base.scale_x * (base.n_cols / n),
        scale_y=base.scale_y * (base.n_rows / n),
        x_offset=base.x_offset,
        y_offset=base.y_offset,
        n_rows=n,
        n_cols=n,
    )


class TestEndToEndSynthetic:
    """Full pipeline test with synthetic data."""

    def test_synthetic_pipeline(self):
        """Run the full solver pipeline with synthetic observations."""
        n = 32
        sat_a = _small_config(GOES16_CONFIG, n)
        sat_b = _small_config(GOES18_CONFIG, n)

        # 1. Compute parallax vectors
        w_u, w_v = compute_parallax_vectors(sat_a, sat_b)
        assert w_u.shape == (n, n)

        # Mask where parallax vectors are valid (on-Earth for both sats)
        valid = np.isfinite(w_u) & np.isfinite(w_v) & (np.abs(w_u) + np.abs(w_v) > 0)

        # 2. Build design matrix
        H_mat = build_design_matrix(w_u, w_v, -600.0, 600.0, -600.0, 600.0)
        assert H_mat.shape == (n, n, 8, 5)

        # 3. Generate synthetic observations from known truth
        x_true = np.array([10000.0, 0.3, -0.1, 4.0, -2.0])
        y = np.einsum("...ij,...j->...i", H_mat, x_true)

        # Zero out invalid pixels to avoid NaN contamination
        y[~valid] = 0.0

        disparities = {
            "D1": np.stack([y[:, :, 0], y[:, :, 1]]),
            "D2": np.stack([y[:, :, 2], y[:, :, 3]]),
            "D3": np.stack([y[:, :, 4], y[:, :, 5]]),
            "D4": np.stack([y[:, :, 6], y[:, :, 7]]),
        }

        # 4. Solve
        solution = solve_stereo_winds(disparities, H_mat)

        # 5. Check recovery at valid pixels only
        assert solution["h"].shape == (n, n)
        h_valid = solution["h"][valid]
        assert h_valid.size > 0, "Should have valid overlap pixels"
        np.testing.assert_allclose(np.nanmean(h_valid), 10000.0, atol=5.0)
        np.testing.assert_allclose(np.nanmean(solution["V_u"][valid]), 4.0, atol=0.1)
        np.testing.assert_allclose(np.nanmean(solution["V_v"][valid]), -2.0, atol=0.1)

        # 6. Convert to m/s
        u_ms, v_ms = pixels_to_wind_ms(solution["V_u"], solution["V_v"], sat_a, 600.0)
        # Off-Earth pixels will be NaN; check only valid pixels
        assert np.all(np.isfinite(u_ms[valid]))

        # 7. Create output dataset
        solution["u_wind"] = u_ms
        solution["v_wind"] = v_ms
        t0 = datetime(2024, 1, 15, 12, 0)
        ds = create_output_dataset(solution, sat_a, t0)

        assert "u_wind" in ds.data_vars
        assert "cloud_top_height" in ds.data_vars
        assert "quality_flag" in ds.data_vars
        assert ds.attrs["Conventions"] == "CF-1.8"

    def test_remap_then_solve(self):
        """Test remap LUT + solver integration."""
        n = 16
        sat_a = _small_config(GOES16_CONFIG, n)
        sat_b = _small_config(GOES18_CONFIG, n)

        # Build remap LUT
        col_b, row_b = build_remap_lut(sat_a, sat_b)
        assert col_b.shape == (n, n)

        # Verify some overlap exists
        valid = np.isfinite(col_b) & np.isfinite(row_b)
        assert valid.sum() > 0

        # Remap a synthetic image
        image_b = np.random.rand(n, n).astype(np.float32)
        remapped = remap_image(image_b, col_b, row_b)
        assert remapped.shape == (n, n)
