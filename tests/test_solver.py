"""Tests for the five-state stereo wind solver."""

import numpy as np
import pytest

from stereo_winds.config import GOES16_CONFIG, GOES18_CONFIG
from stereo_winds.solver import (
    build_design_matrix,
    compute_parallax_vectors,
    compute_parallax_vectors_at_h,
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
    """Test pixel velocity to m/s conversion with per-pixel ground scale."""

    def test_zero_velocity(self):
        V_u = np.zeros((GOES16_CONFIG.n_rows, GOES16_CONFIG.n_cols))
        V_v = np.zeros((GOES16_CONFIG.n_rows, GOES16_CONFIG.n_cols))
        u, v = pixels_to_wind_ms(V_u, V_v, GOES16_CONFIG, dt_seconds=600.0)
        # On-Earth pixels should be 0, off-Earth are NaN
        assert np.nanmax(np.abs(u)) == 0.0
        assert np.nanmax(np.abs(v)) == 0.0

    def test_nadir_pixel_scale(self):
        """At nadir (~row 2712, col 2712), scale should be close to nominal 2 km."""
        V_u = np.zeros((GOES16_CONFIG.n_rows, GOES16_CONFIG.n_cols))
        V_v = np.zeros((GOES16_CONFIG.n_rows, GOES16_CONFIG.n_cols))
        V_u[2712, 2712] = 1.0
        V_v[2712, 2712] = 1.0
        u, v = pixels_to_wind_ms(V_u, V_v, GOES16_CONFIG, dt_seconds=1.0)
        # Nadir pixel scale ~ 2004 m
        assert 1900 < abs(u[2712, 2712]) < 2100
        assert 1900 < abs(v[2712, 2712]) < 2100

    def test_limb_larger_than_nadir(self):
        """Pixels near the limb should have larger ground scale than nadir."""
        from stereo_winds.navigation import compute_pixel_scale
        dx_m, dy_m = compute_pixel_scale(GOES16_CONFIG)
        nadir_dx = dx_m[2712, 2712]
        # Pick a pixel near the limb but still on-Earth (row 200, near north edge)
        limb_dx = dx_m[200, 2712]
        assert np.isfinite(limb_dx), "Near-limb pixel should be on-Earth"
        assert limb_dx > nadir_dx * 1.05, (
            f"Near-limb dx ({limb_dx:.0f}) should be larger than nadir ({nadir_dx:.0f})"
        )

    def test_v_sign_convention(self):
        """Positive V_v should give positive v_ms."""
        V_u = np.zeros((GOES16_CONFIG.n_rows, GOES16_CONFIG.n_cols))
        V_v = np.zeros((GOES16_CONFIG.n_rows, GOES16_CONFIG.n_cols))
        V_v[2712, 2712] = 1.0
        u, v = pixels_to_wind_ms(V_u, V_v, GOES16_CONFIG, dt_seconds=600.0)
        assert v[2712, 2712] > 0, f"v_ms should be positive, got {v[2712, 2712]}"


def _make_small_sats(n=8):
    """Create small test satellite configs for fast parallax tests."""
    from stereo_winds.config import SatelliteConfig
    scale_factor = 5424 / n
    sat_a = SatelliteConfig(
        satellite_id="test_a", sub_lon_deg=-75.0, n_rows=n, n_cols=n,
        scale_x=GOES16_CONFIG.scale_x * scale_factor,
        scale_y=GOES16_CONFIG.scale_y * scale_factor,
        x_offset=GOES16_CONFIG.x_offset,
        y_offset=GOES16_CONFIG.y_offset,
    )
    sat_b = SatelliteConfig(
        satellite_id="test_b", sub_lon_deg=-137.0, n_rows=n, n_cols=n,
        scale_x=GOES18_CONFIG.scale_x * scale_factor,
        scale_y=GOES18_CONFIG.scale_y * scale_factor,
        x_offset=GOES18_CONFIG.x_offset,
        y_offset=GOES18_CONFIG.y_offset,
    )
    return sat_a, sat_b


class TestComputeParallaxVectorsAtH:
    """Test height-dependent parallax computation."""

    def test_shape(self):
        sat_a, sat_b = _make_small_sats(8)
        w_u, w_v = compute_parallax_vectors_at_h(sat_a, sat_b, h_m=5000.0)
        assert w_u.shape == (8, 8)
        assert w_v.shape == (8, 8)

    def test_h0_matches_original(self):
        """At h=0, should match compute_parallax_vectors."""
        sat_a, sat_b = _make_small_sats(8)
        w_u_orig, w_v_orig = compute_parallax_vectors(sat_a, sat_b)
        w_u_at0, w_v_at0 = compute_parallax_vectors_at_h(sat_a, sat_b, h_m=0.0)
        np.testing.assert_allclose(w_u_at0, w_u_orig, atol=1e-10)
        np.testing.assert_allclose(w_v_at0, w_v_orig, atol=1e-10)

    def test_differs_from_h0(self):
        """Parallax at h=10000 should differ from h=0."""
        sat_a, sat_b = _make_small_sats(8)
        w_u_0, w_v_0 = compute_parallax_vectors(sat_a, sat_b)
        w_u_h, w_v_h = compute_parallax_vectors_at_h(sat_a, sat_b, h_m=10000.0)
        valid = np.isfinite(w_u_0) & np.isfinite(w_u_h)
        if valid.sum() > 0:
            diff = np.abs(w_u_h[valid] - w_u_0[valid])
            assert np.max(diff) > 0, "Parallax should change with height"

    def test_per_pixel_h(self):
        """Should accept a per-pixel height array."""
        sat_a, sat_b = _make_small_sats(8)
        h_map = np.random.default_rng(0).uniform(0, 15000, (8, 8))
        w_u, w_v = compute_parallax_vectors_at_h(sat_a, sat_b, h_m=h_map)
        assert w_u.shape == (8, 8)


class TestParallaxRemapConsistency:
    """Verify parallax vectors match the remap geometry displacement."""

    def test_parallax_matches_remap_geometry(self):
        """For clouds at various locations, the parallax vector should
        correctly predict the displacement observed through the B-to-A remap.

        The remap maps B scanning angles → surface lat/lon → A scanning angles.
        The parallax vector must account for this full transformation, not just
        convert B scanning angle changes directly to A pixel units.
        """
        from stereo_winds.navigation import (
            fixed_grid_to_geodetic,
            geodetic_to_fixed_grid,
        )

        sat_a = GOES16_CONFIG
        sat_b = GOES18_CONFIG
        h_test = 10000.0  # 10 km cloud
        dh = 1000.0

        test_points = [
            (0.0, -106.0, "equator_midpoint"),
            (30.0, -120.0, "pacific_south_CA"),
            (40.0, -100.0, "central_US"),
            (20.0, -80.0, "caribbean"),
        ]

        for lat, lon, name in test_points:
            lat_a = np.array([lat])
            lon_a = np.array([lon])

            # Compute parallax using the same logic as compute_parallax_vectors
            xa_0, ya_0 = geodetic_to_fixed_grid(lat_a, lon_a, sat_a, h_m=0.0)
            xa_h, ya_h = geodetic_to_fixed_grid(lat_a, lon_a, sat_a, h_m=dh)
            xb_h, yb_h = geodetic_to_fixed_grid(lat_a, lon_a, sat_b, h_m=dh)
            lat_remap, lon_remap = fixed_grid_to_geodetic(xb_h, yb_h, sat_b)
            xa_remap, ya_remap = geodetic_to_fixed_grid(
                lat_remap, lon_remap, sat_a, h_m=0.0
            )
            w_u = float((xa_remap - xa_0) - (xa_h - xa_0)) / sat_a.scale_x / dh

            # Compute the "true" displacement at h_test through the full geometry
            xb_full, yb_full = geodetic_to_fixed_grid(
                lat_a, lon_a, sat_b, h_m=h_test
            )
            lat_r_full, lon_r_full = fixed_grid_to_geodetic(xb_full, yb_full, sat_b)
            xa_r_full, ya_r_full = geodetic_to_fixed_grid(
                lat_r_full, lon_r_full, sat_a, h_m=0.0
            )
            xa_h_full, ya_h_full = geodetic_to_fixed_grid(
                lat_a, lon_a, sat_a, h_m=h_test
            )
            true_par_u = float(
                (xa_r_full - xa_0) - (xa_h_full - xa_0)
            ) / sat_a.scale_x
            predicted_par_u = h_test * w_u

            # Height error should be < 1% (linearization error only)
            if abs(true_par_u) > 0.01:
                solver_h = true_par_u / w_u
                h_error_pct = abs(solver_h / h_test - 1) * 100
                assert h_error_pct < 1.0, (
                    f"{name}: height error {h_error_pct:.1f}% > 1%"
                )


class TestSolveStereoWindsIterative:
    """Test iterative nonlinear solve."""

    def test_delta_h_history_empty_for_single_iter(self):
        """n_iter=1 should return empty delta_h_history."""
        n = 4
        w_u = np.full((n, n), 0.001)
        w_v = np.full((n, n), 0.0003)
        H_mat = build_design_matrix(w_u, w_v, -600.0, 600.0, -600.0, 600.0)
        x_true = np.array([8000.0, 0.5, -0.2, 5.0, -3.0])
        y = np.einsum("...ij,...j->...i", H_mat, x_true)
        disparities = {
            f"D{i+1}": np.stack([y[:, :, 2*i], y[:, :, 2*i+1]])
            for i in range(4)
        }
        sol = solve_stereo_winds(disparities, H_mat)
        assert sol["delta_h_history"] == []

    def test_iterative_noop_when_no_sats(self):
        """n_iter>1 without sat configs falls back to single solve."""
        n = 4
        w_u = np.full((n, n), 0.001)
        w_v = np.full((n, n), 0.0003)
        H_mat = build_design_matrix(w_u, w_v, -600.0, 600.0, -600.0, 600.0)
        x_true = np.array([8000.0, 0.5, -0.2, 5.0, -3.0])
        y = np.einsum("...ij,...j->...i", H_mat, x_true)
        disparities = {
            f"D{i+1}": np.stack([y[:, :, 2*i], y[:, :, 2*i+1]])
            for i in range(4)
        }
        sol = solve_stereo_winds(disparities, H_mat, n_iter=3)
        assert sol["delta_h_history"] == []
        np.testing.assert_allclose(sol["h"], 8000.0, atol=1.0)

    def test_iterative_converges(self):
        """With real sat configs, iteration should produce delta_h_history."""
        sat_a, sat_b = _make_small_sats(8)
        w_u, w_v = compute_parallax_vectors(sat_a, sat_b)
        H_mat = build_design_matrix(w_u, w_v, -600.0, 600.0, -600.0, 600.0)

        x_true = np.array([10000.0, 0.3, -0.1, 2.0, -1.0])
        y = np.einsum("...ij,...j->...i", H_mat, x_true)
        disparities = {
            f"D{i+1}": np.stack([y[:, :, 2*i], y[:, :, 2*i+1]])
            for i in range(4)
        }
        sol = solve_stereo_winds(
            disparities, H_mat, sat_a=sat_a, sat_b=sat_b, n_iter=3,
        )
        # Should have at least 1 entry in delta_h_history (from iter 1 vs 0)
        assert len(sol["delta_h_history"]) >= 1
