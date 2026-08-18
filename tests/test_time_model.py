"""Tests for pixel scan time models."""

import numpy as np

from stereo_winds.config import GOES16_CONFIG, SatelliteConfig
from stereo_winds.time_model import (
    abi_pixel_times,
    compute_scene_dt_fields,
    compute_scene_times,
    _to_unix,
)

from datetime import datetime, timedelta


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


def _small_sat(n=10, satellite_id="testsat", sweep="x"):
    return SatelliteConfig(
        satellite_id=satellite_id, sub_lon_deg=-75.0, sweep=sweep,
        n_rows=n, n_cols=n,
    )


class TestComputeSceneDtFields:
    """Per-pixel dt fields (relative to A0) for the design matrix."""

    def _time_info(self, t0, dt_min=10.0, pixel_time_b=None):
        d = timedelta(minutes=dt_min)
        mk = lambda t: {"t_nominal": t, "t_start": t,
                        "t_end": t + timedelta(seconds=600),
                        "pixel_time": None}
        info = {
            "A_minus": mk(t0 - d), "A0": mk(t0), "A_plus": mk(t0 + d),
            "B_minus": mk(t0 - d), "B_plus": mk(t0 + d),
        }
        if pixel_time_b is not None:
            info["B_minus"]["pixel_time"] = pixel_time_b - 600.0
            info["B_plus"]["pixel_time"] = pixel_time_b + 600.0
        return info

    def test_temporal_pairs_nominal(self):
        """Same-instrument A pairs: scan phase cancels → ±dt everywhere."""
        n = 10
        sat = _small_sat(n)
        t0 = datetime(2026, 1, 31, 13, 0)
        # Identity-ish LUT
        col_b, row_b = np.meshgrid(np.arange(n, dtype=float),
                                   np.arange(n, dtype=float))
        dts = compute_scene_dt_fields(self._time_info(t0), sat, sat, col_b, row_b)
        np.testing.assert_allclose(dts["A_minus"], -600.0, atol=1e-9)
        np.testing.assert_allclose(dts["A_plus"], 600.0, atol=1e-9)

    def test_b_linear_model_scan_phase(self):
        """B without native times: dt includes B-vs-A row scan phase."""
        n = 10
        sat = _small_sat(n)
        t0 = datetime(2026, 1, 31, 13, 0)
        # LUT maps every A pixel to B's LAST row (row n-1): B sees it late
        col_b = np.full((n, n), 5.0)
        row_b = np.full((n, n), float(n - 1))
        dts = compute_scene_dt_fields(self._time_info(t0), sat, sat, col_b, row_b)
        # At A row 0: t_B = -600 + (n-1)/n*600; t_A0 = 0 → dt = -600 + 540
        expected_row0 = -600.0 + (n - 1) / n * 600.0
        np.testing.assert_allclose(dts["B_minus"][0, :], expected_row0, atol=1e-6)
        # At A row n-1: t_A0 = (n-1)/n*600 → dt = -600 exactly
        np.testing.assert_allclose(dts["B_minus"][n - 1, :], -600.0, atol=1e-6)

    def test_b_native_pixel_time(self):
        """B with a native per-pixel time field: sampled through the LUT."""
        n = 10
        sat = _small_sat(n)
        t0 = datetime(2026, 1, 31, 13, 0)
        base = _to_unix(t0)
        # Native B pixel time: unique value per pixel
        pt = base + np.arange(n * n, dtype=np.float64).reshape(n, n)
        col_b, row_b = np.meshgrid(np.arange(n, dtype=float),
                                   np.arange(n, dtype=float))
        info = self._time_info(t0, pixel_time_b=pt)
        dts = compute_scene_dt_fields(info, sat, sat, col_b, row_b)
        # dt at A pixel (r,c) = pt[r,c] − 600 − (base + r/n*600)
        rows = np.arange(n, dtype=np.float64)[:, None]
        expected = (pt - 600.0) - (base + rows / n * 600.0)
        np.testing.assert_allclose(dts["B_minus"], expected, atol=1e-6)

    def test_sector_dims_b_linear_model(self):
        """Sector grids: A and B have different (CONUS-like) dimensions.

        The B-side linear model must run over B's sector rows/duration
        while the output stays on A's grid.
        """
        sat_a = SatelliteConfig(
            satellite_id="sat_a", sub_lon_deg=-75.0, sweep="x",
            n_rows=8, n_cols=12,
        )
        sat_b = SatelliteConfig(
            satellite_id="sat_b", sub_lon_deg=-137.0, sweep="x",
            n_rows=6, n_cols=10,
        )
        t0 = datetime(2026, 8, 11, 20, 0)
        scan = timedelta(seconds=150)  # CONUS-like duration
        mk = lambda t: {"t_nominal": t, "t_start": t, "t_end": t + scan,
                        "pixel_time": None}
        d = timedelta(minutes=10)
        info = {
            "A_minus": mk(t0 - d), "A0": mk(t0), "A_plus": mk(t0 + d),
            "B_minus": mk(t0 - d), "B_plus": mk(t0 + d),
        }
        # Every A pixel maps to B's last row (row 5 of 6)
        col_b = np.full((8, 12), 3.0)
        row_b = np.full((8, 12), 5.0)
        dts = compute_scene_dt_fields(info, sat_a, sat_b, col_b, row_b)

        for k in ("A_minus", "A_plus", "B_minus", "B_plus"):
            assert dts[k].shape == (8, 12), k

        # Equal-duration temporal pairs: scan phase cancels exactly
        np.testing.assert_allclose(dts["A_minus"], -600.0, atol=1e-9)
        np.testing.assert_allclose(dts["A_plus"], 600.0, atol=1e-9)

        # Cross pair at A row r: (−600 + 5/6·150) − (r/8·150)
        rows = np.arange(8, dtype=np.float64)[:, None]
        expected = (-600.0 + 5.0 / 6.0 * 150.0) - rows / 8.0 * 150.0
        np.testing.assert_allclose(dts["B_minus"], np.broadcast_to(expected, (8, 12)),
                                   atol=1e-6)

    def test_invalid_lut_falls_back_to_nominal(self):
        """Pixels B can't see get the nominal offset (finite design matrix)."""
        n = 10
        sat = _small_sat(n)
        t0 = datetime(2026, 1, 31, 13, 0)
        col_b = np.full((n, n), np.nan)
        row_b = np.full((n, n), np.nan)
        dts = compute_scene_dt_fields(self._time_info(t0), sat, sat, col_b, row_b)
        assert np.all(np.isfinite(dts["B_plus"]))
        np.testing.assert_allclose(dts["B_plus"], 600.0, atol=1e-9)
