"""Tests for the five-state stereo wind solver."""

import numpy as np
import pytest

from stereo_winds.config import GOES16_CONFIG, GOES18_CONFIG
from stereo_winds.solver import (
    build_design_matrix,
    compute_parallax_vectors,
    pixels_to_wind_ms,
    solve_stereo_winds,
)


class TestComputeParallaxVectors:
    """Test parallax sensitivity computation."""

    def test_shape(self):
        """Output should match A's grid dimensions."""
        # Use very small grids for speed
        from stereo_winds.config import SatelliteConfig

        sat_a = SatelliteConfig(
            satellite_id="test_a", sub_lon_deg=-75.0, n_rows=16, n_cols=16,
            scale_x=GOES16_CONFIG.scale_x * (5424 / 16),
            scale_y=GOES16_CONFIG.scale_y * (5424 / 16),
            x_offset=GOES16_CONFIG.x_offset,
            y_offset=GOES16_CONFIG.y_offset,
        )
        sat_b = SatelliteConfig(
            satellite_id="test_b", sub_lon_deg=-137.0, n_rows=16, n_cols=16,
            scale_x=GOES18_CONFIG.scale_x * (5424 / 16),
            scale_y=GOES18_CONFIG.scale_y * (5424 / 16),
            x_offset=GOES18_CONFIG.x_offset,
            y_offset=GOES18_CONFIG.y_offset,
        )
        w_u, w_v = compute_parallax_vectors(sat_a, sat_b)
        assert w_u.shape == (16, 16)
        assert w_v.shape == (16, 16)

    def test_nonzero_parallax(self):
        """GOES-16 vs GOES-18 should produce nonzero parallax."""
        from stereo_winds.config import SatelliteConfig

        sat_a = SatelliteConfig(
            satellite_id="test_a", sub_lon_deg=-75.0, n_rows=8, n_cols=8,
            scale_x=GOES16_CONFIG.scale_x * (5424 / 8),
            scale_y=GOES16_CONFIG.scale_y * (5424 / 8),
            x_offset=GOES16_CONFIG.x_offset,
            y_offset=GOES16_CONFIG.y_offset,
        )
        sat_b = SatelliteConfig(
            satellite_id="test_b", sub_lon_deg=-137.0, n_rows=8, n_cols=8,
            scale_x=GOES18_CONFIG.scale_x * (5424 / 8),
            scale_y=GOES18_CONFIG.scale_y * (5424 / 8),
            x_offset=GOES18_CONFIG.x_offset,
            y_offset=GOES18_CONFIG.y_offset,
        )
        w_u, w_v = compute_parallax_vectors(sat_a, sat_b)

        # At least some valid pixels should have nonzero parallax
        valid = np.isfinite(w_u) & np.isfinite(w_v)
        if valid.sum() > 0:
            mag = np.sqrt(w_u[valid] ** 2 + w_v[valid] ** 2)
            assert np.max(mag) > 0, "Parallax should be nonzero"


class TestBuildDesignMatrix:
    """Test design matrix construction."""

    def test_shape(self):
        n = 4
        w_u = np.ones((n, n)) * 0.001
        w_v = np.ones((n, n)) * 0.0005
        H = build_design_matrix(w_u, w_v, -600.0, 600.0, -600.0, 600.0)
        assert H.shape == (n, n, 8, 5)

    def test_temporal_rows_no_parallax(self):
        """Rows 0-3 (same-satellite) should have zero in the height column."""
        n = 2
        w_u = np.ones((n, n)) * 0.001
        w_v = np.ones((n, n)) * 0.0005
        H = build_design_matrix(w_u, w_v, -600.0, 600.0, -600.0, 600.0)

        # Height column (0) should be zero for rows 0-3 (temporal pairs)
        np.testing.assert_array_equal(H[:, :, 0, 0], 0.0)
        np.testing.assert_array_equal(H[:, :, 1, 0], 0.0)
        np.testing.assert_array_equal(H[:, :, 2, 0], 0.0)
        np.testing.assert_array_equal(H[:, :, 3, 0], 0.0)

    def test_cross_sat_rows_have_parallax(self):
        """Rows 4-7 (cross-satellite) should have parallax vectors in height column."""
        n = 2
        w_u = np.ones((n, n)) * 0.001
        w_v = np.ones((n, n)) * 0.0005
        H = build_design_matrix(w_u, w_v, -600.0, 600.0, -600.0, 600.0)

        np.testing.assert_array_equal(H[:, :, 4, 0], w_u)
        np.testing.assert_array_equal(H[:, :, 5, 0], w_v)
        np.testing.assert_array_equal(H[:, :, 6, 0], w_u)
        np.testing.assert_array_equal(H[:, :, 7, 0], w_v)


class TestSolveStereoWinds:
    """Test the WLS solver with synthetic data."""

    def _synthetic_problem(
        self, n=10, h_true=8000.0, V_u=5.0, V_v=-3.0, p_u=0.5, p_v=-0.2,
        noise_sigma=0.0,
    ):
        """Generate a synthetic problem with known solution."""
        # Parallax vectors (constant for simplicity)
        w_u = np.full((n, n), 0.001)
        w_v = np.full((n, n), 0.0003)

        dt_am, dt_ap, dt_bm, dt_bp = -600.0, 600.0, -600.0, 600.0

        H_mat = build_design_matrix(w_u, w_v, dt_am, dt_ap, dt_bm, dt_bp)

        # True state vector
        x_true = np.array([h_true, p_u, p_v, V_u, V_v])

        # Generate synthetic observations: y = H @ x
        y = np.einsum("...ij,...j->...i", H_mat, x_true)

        # Add noise
        if noise_sigma > 0:
            rng = np.random.default_rng(42)
            y += rng.normal(0, noise_sigma, y.shape)

        # Unpack into disparity dict
        disparities = {
            "D1": np.stack([y[:, :, 0], y[:, :, 1]]),  # (2, n, n)
            "D2": np.stack([y[:, :, 2], y[:, :, 3]]),
            "D3": np.stack([y[:, :, 4], y[:, :, 5]]),
            "D4": np.stack([y[:, :, 6], y[:, :, 7]]),
        }

        return disparities, H_mat, x_true

    def test_exact_recovery(self):
        """Solver should recover the true state with no noise (< 0.1% error)."""
        disparities, H_mat, x_true = self._synthetic_problem()
        sol = solve_stereo_winds(disparities, H_mat)

        # Height: ~1m tolerance for 8000m (conditioning of mixed-scale normal matrix)
        np.testing.assert_allclose(sol["h"], x_true[0], atol=1.0)
        np.testing.assert_allclose(sol["p_u"], x_true[1], atol=0.01)
        np.testing.assert_allclose(sol["p_v"], x_true[2], atol=0.01)
        np.testing.assert_allclose(sol["V_u"], x_true[3], atol=0.01)
        np.testing.assert_allclose(sol["V_v"], x_true[4], atol=0.01)

    def test_chi2_near_zero_no_noise(self):
        """Chi-squared should be near zero with no noise."""
        disparities, H_mat, _ = self._synthetic_problem()
        sol = solve_stereo_winds(disparities, H_mat)
        np.testing.assert_allclose(sol["chi2"], 0.0, atol=1e-4)

    def test_recovery_with_noise(self):
        """With moderate noise, solution should be within formal uncertainty."""
        disparities, H_mat, x_true = self._synthetic_problem(
            n=100, noise_sigma=0.5,
        )
        sol = solve_stereo_winds(disparities, H_mat)

        # Mean recovered height should be close to true
        h_mean = np.nanmean(sol["h"])
        np.testing.assert_allclose(h_mean, x_true[0], atol=500.0)

        # Mean wind should be close
        Vu_mean = np.nanmean(sol["V_u"])
        Vv_mean = np.nanmean(sol["V_v"])
        np.testing.assert_allclose(Vu_mean, x_true[3], atol=0.5)
        np.testing.assert_allclose(Vv_mean, x_true[4], atol=0.5)

    def test_quality_flags(self):
        """Negative heights should be flagged."""
        disparities, H_mat, _ = self._synthetic_problem(h_true=-1000.0)
        sol = solve_stereo_winds(disparities, H_mat)
        assert np.all(sol["quality_flag"] == 0.0)

    def test_single_pixel(self):
        """Solver should work for a single pixel (1x1 grid)."""
        disparities, H_mat, x_true = self._synthetic_problem(n=1)
        sol = solve_stereo_winds(disparities, H_mat)
        np.testing.assert_allclose(sol["h"].item(), x_true[0], atol=1.0)


class TestPixelsToWindMs:
    """Test pixel velocity to m/s conversion."""

    def test_zero_velocity(self):
        u, v = pixels_to_wind_ms(0.0, 0.0, GOES16_CONFIG, dt_seconds=600.0)
        assert u == 0.0
        assert v == 0.0

    def test_nonzero(self):
        u, v = pixels_to_wind_ms(
            np.array([1.0]), np.array([1.0]),
            GOES16_CONFIG, dt_seconds=600.0,
        )
        # Pixel scale ~ 35786023 * 5.6e-5 ≈ 2004 m
        # So 1 pixel / 600s ≈ 3.3 m/s
        assert 2.0 < abs(u[0]) < 5.0
        assert 2.0 < abs(v[0]) < 5.0
