"""Tests for stereo_winds.qa module."""

import numpy as np
import pytest

from stereo_winds.qa import compute_qa_flag, height_gradient


class TestHeightGradient:
    """Tests for the Sobel height-gradient function."""

    def test_flat_field_zero_gradient(self):
        h = np.full((50, 50), 5000.0)
        grad = height_gradient(h)
        assert grad.shape == (50, 50)
        np.testing.assert_allclose(grad, 0.0, atol=1e-10)

    def test_step_edge_large_gradient(self):
        h = np.zeros((50, 50))
        h[:, 25:] = 10000.0
        grad = height_gradient(h)
        # Gradient should be large along the edge at column 25
        assert np.max(grad[:, 24:27]) > 1000.0
        # Far from the edge, gradient should be near zero
        assert np.max(grad[:, :20]) < 1.0
        assert np.max(grad[:, 30:]) < 1.0

    def test_nan_handling(self):
        h = np.full((20, 20), 5000.0)
        h[10, 10] = np.nan
        grad = height_gradient(h)
        assert np.all(np.isfinite(grad))


class TestComputeQaFlag:
    """Tests for the multi-level QA flag computation."""

    def _make_solution(self, n=10, h=5000.0, chi2=0.1, sigma_h=500.0,
                       u=10.0, v=5.0):
        """Build a synthetic solution dict with uniform values."""
        shape = (n, n)
        return {
            "h": np.full(shape, h),
            "chi2": np.full(shape, chi2),
            "sigma_h": np.full(shape, sigma_h),
            "u_wind": np.full(shape, u),
            "v_wind": np.full(shape, v),
        }

    def test_all_nan_gives_level0(self):
        sol = self._make_solution(h=np.nan)
        mask = np.ones((10, 10), dtype=bool)
        qa = compute_qa_flag(sol, mask)
        assert np.all(qa == 0.0)

    def test_negative_height_gives_level0(self):
        sol = self._make_solution(h=-100.0)
        mask = np.ones((10, 10), dtype=bool)
        qa = compute_qa_flag(sol, mask)
        assert np.all(qa == 0.0)

    def test_zenith_mask_false_gives_level0(self):
        sol = self._make_solution()
        mask = np.zeros((10, 10), dtype=bool)
        qa = compute_qa_flag(sol, mask)
        assert np.all(qa == 0.0)

    def test_high_chi2_gives_level0(self):
        sol = self._make_solution(chi2=15.0)
        mask = np.ones((10, 10), dtype=bool)
        qa = compute_qa_flag(sol, mask, chi2_threshold=10.0)
        assert np.all(qa == 0.0)

    def test_basic_valid_gives_level1(self):
        sol = self._make_solution(chi2=5.0, sigma_h=5000.0)
        mask = np.ones((10, 10), dtype=bool)
        qa = compute_qa_flag(sol, mask)
        # chi2=5.0 passes level 1 but sigma_h=5000 fails level 2
        assert np.all(qa == 1.0)

    def test_strict_gives_level2(self):
        sol = self._make_solution(h=10000.0, chi2=0.1, sigma_h=500.0, u=10.0, v=5.0)
        mask = np.ones((10, 10), dtype=bool)
        qa = compute_qa_flag(sol, mask)
        assert np.all(qa == 2.0)

    def test_chi2_boundary_level2(self):
        """chi2=0.2 should still pass level 2."""
        sol = self._make_solution(chi2=0.2, sigma_h=500.0)
        mask = np.ones((10, 10), dtype=bool)
        qa = compute_qa_flag(sol, mask)
        assert np.all(qa == 2.0)

    def test_chi2_above_02_stays_level1(self):
        """chi2=0.3 passes level 1 but not level 2."""
        sol = self._make_solution(chi2=0.3, sigma_h=500.0)
        mask = np.ones((10, 10), dtype=bool)
        qa = compute_qa_flag(sol, mask)
        assert np.all(qa == 1.0)

    def test_sigma_h_height_dependent_low_clouds(self):
        """Low clouds (h<3000) need sigma_h <= 1000."""
        sol = self._make_solution(h=2000.0, chi2=0.1, sigma_h=1500.0)
        mask = np.ones((10, 10), dtype=bool)
        qa = compute_qa_flag(sol, mask)
        # sigma_h=1500 > 1000 threshold for low clouds -> level 1 only
        assert np.all(qa == 1.0)

    def test_sigma_h_height_dependent_mid_clouds(self):
        """Mid-level clouds (3000-7000) allow sigma_h <= 2000."""
        sol = self._make_solution(h=5000.0, chi2=0.1, sigma_h=1500.0)
        mask = np.ones((10, 10), dtype=bool)
        qa = compute_qa_flag(sol, mask)
        # sigma_h=1500 <= 2000 threshold for mid-level -> level 2
        assert np.all(qa == 2.0)

    def test_sigma_h_height_dependent_high_clouds(self):
        """High clouds (h>=7000) need sigma_h <= 1000."""
        sol = self._make_solution(h=12000.0, chi2=0.1, sigma_h=1500.0)
        mask = np.ones((10, 10), dtype=bool)
        qa = compute_qa_flag(sol, mask)
        # sigma_h=1500 > 1000 threshold for high clouds -> level 1 only
        assert np.all(qa == 1.0)

    def test_speed_over_100_stays_level1(self):
        sol = self._make_solution(chi2=0.1, sigma_h=500.0, u=80.0, v=80.0)
        mask = np.ones((10, 10), dtype=bool)
        qa = compute_qa_flag(sol, mask)
        # speed = sqrt(80^2+80^2) ~ 113 > 100 -> level 1
        assert np.all(qa == 1.0)

    def test_monotonic_levels(self):
        """Level 2 pixels must be a subset of level 1 pixels."""
        rng = np.random.default_rng(42)
        sol = {
            "h": rng.uniform(0, 15000, (30, 30)),
            "chi2": rng.uniform(0, 5, (30, 30)),
            "sigma_h": rng.uniform(100, 3000, (30, 30)),
            "u_wind": rng.uniform(-30, 30, (30, 30)),
            "v_wind": rng.uniform(-30, 30, (30, 30)),
        }
        mask = rng.random((30, 30)) > 0.1
        qa = compute_qa_flag(sol, mask)
        assert np.all((qa >= 2) <= (qa >= 1))

    def test_output_dtype(self):
        sol = self._make_solution()
        mask = np.ones((10, 10), dtype=bool)
        qa = compute_qa_flag(sol, mask)
        assert qa.dtype == np.float32
