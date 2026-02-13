"""Tests for pixel scan time models."""

import numpy as np

from stereo_winds.config import GOES16_CONFIG
from stereo_winds.time_model import abi_pixel_times, compute_scene_times

from datetime import datetime


class TestABIPixelTimes:
    def test_first_row_zero(self):
        """Row 0 should have zero offset (scan starts at top)."""
        t = abi_pixel_times(np.array([0]), GOES16_CONFIG)
        assert t[0] == 0.0

    def test_last_row_full_duration(self):
        """Last row should be close to full scan duration."""
        n = GOES16_CONFIG.n_rows
        t = abi_pixel_times(np.array([n]), GOES16_CONFIG)
        np.testing.assert_allclose(t[0], 600.0, atol=1.0)

    def test_monotonic(self):
        """Time offsets should increase monotonically with row."""
        rows = np.arange(GOES16_CONFIG.n_rows)
        t = abi_pixel_times(rows, GOES16_CONFIG)
        assert np.all(np.diff(t) > 0)

    def test_vectorized(self):
        """Should handle 2D arrays."""
        rows = np.arange(100).reshape(10, 10)
        t = abi_pixel_times(rows, GOES16_CONFIG)
        assert t.shape == (10, 10)


class TestComputeSceneTimes:
    def test_scene_time_offsets(self):
        """Check that scene times are correct relative to t0."""
        t0 = datetime(2024, 1, 1, 12, 0)
        times = compute_scene_times(t0, dt_minutes=10.0, sat_a=GOES16_CONFIG, sat_b=GOES16_CONFIG)

        assert times["A0"] == 0.0
        assert times["A_minus"] == -600.0
        assert times["A_plus"] == 600.0
        assert times["B_minus"] == -600.0
        assert times["B_plus"] == 600.0
