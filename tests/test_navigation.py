"""Tests for geostationary projection math."""

import numpy as np
import pytest

from stereo_winds.config import GOES16_CONFIG, GOES18_CONFIG, HIMAWARI8_CONFIG
from stereo_winds.navigation import (
    fixed_grid_to_geodetic,
    geodetic_to_ecef,
    geodetic_to_fixed_grid,
    pixel_to_scanning_angle,
    scanning_angle_to_pixel,
    compute_zenith_angle,
)


class TestPixelScanningAngle:
    """Test pixel ↔ scanning angle conversions."""

    def test_round_trip(self):
        sat = GOES16_CONFIG
        col = np.array([0.0, 100.0, 2712.0, 5423.0])
        row = np.array([0.0, 100.0, 2712.0, 5423.0])
        x, y = pixel_to_scanning_angle(col, row, sat)
        col2, row2 = scanning_angle_to_pixel(x, y, sat)
        np.testing.assert_allclose(col2, col, atol=1e-10)
        np.testing.assert_allclose(row2, row, atol=1e-10)

    def test_center_pixel(self):
        """Center pixel should give scanning angles near zero."""
        sat = GOES16_CONFIG
        center_col = (sat.n_cols - 1) / 2.0
        center_row = (sat.n_rows - 1) / 2.0
        x, y = pixel_to_scanning_angle(center_col, center_row, sat)
        # Should be close to zero (sub-satellite point)
        assert abs(x) < 0.002
        assert abs(y) < 0.002


class TestFixedGridToGeodetic:
    """Test inverse projection: scanning angles → geodetic."""

    def test_subsatellite_point(self):
        """Scanning angles (0, 0) → sub-satellite point on equator."""
        sat = GOES16_CONFIG
        lat, lon = fixed_grid_to_geodetic(0.0, 0.0, sat)
        assert abs(lat) < 0.01  # equator
        assert abs(lon - sat.sub_lon_deg) < 0.01

    def test_off_earth_nan(self):
        """Large scanning angles should give NaN (off-Earth)."""
        sat = GOES16_CONFIG
        lat, lon = fixed_grid_to_geodetic(0.5, 0.5, sat)
        assert np.isnan(lat)
        assert np.isnan(lon)

    def test_vectorized(self):
        """Should handle 2D arrays."""
        sat = GOES16_CONFIG
        x = np.linspace(-0.1, 0.1, 50)
        y = np.linspace(-0.1, 0.1, 50)
        xx, yy = np.meshgrid(x, y)
        lat, lon = fixed_grid_to_geodetic(xx, yy, sat)
        assert lat.shape == (50, 50)
        # Central pixels should be valid
        assert not np.isnan(lat[25, 25])

    def test_ahi_sweep_y(self):
        """AHI (y-sweep) should also work correctly."""
        sat = HIMAWARI8_CONFIG
        lat, lon = fixed_grid_to_geodetic(0.0, 0.0, sat)
        assert abs(lat) < 0.01
        assert abs(lon - sat.sub_lon_deg) < 0.01


class TestGeodeticToFixedGrid:
    """Test forward projection: geodetic → scanning angles."""

    def test_subsatellite_point(self):
        """Sub-satellite point should give (0, 0) scanning angles."""
        sat = GOES16_CONFIG
        x, y = geodetic_to_fixed_grid(0.0, sat.sub_lon_deg, sat)
        assert abs(x) < 1e-10
        assert abs(y) < 1e-10

    def test_round_trip(self):
        """geodetic → fixed_grid → geodetic should be identity."""
        sat = GOES16_CONFIG
        # Generate test scanning angles on the disk
        x_test = np.linspace(-0.1, 0.1, 30)
        y_test = np.linspace(-0.1, 0.1, 30)
        xx, yy = np.meshgrid(x_test, y_test)

        lat, lon = fixed_grid_to_geodetic(xx, yy, sat)
        valid = ~np.isnan(lat)
        lat_v, lon_v = lat[valid], lon[valid]

        x_rt, y_rt = geodetic_to_fixed_grid(lat_v, lon_v, sat)
        lat_rt, lon_rt = fixed_grid_to_geodetic(x_rt, y_rt, sat)

        np.testing.assert_allclose(lat_rt, lat_v, atol=0.001)
        np.testing.assert_allclose(lon_rt, lon_v, atol=0.001)

    def test_elevated_point(self):
        """Forward projection with h > 0 should shift the scanning angle."""
        sat = GOES16_CONFIG
        lat, lon = 30.0, -90.0
        x0, y0 = geodetic_to_fixed_grid(lat, lon, sat, h_m=0.0)
        x1, y1 = geodetic_to_fixed_grid(lat, lon, sat, h_m=10000.0)
        # Elevated point should have different scanning angles (parallax)
        assert not np.isclose(x0, x1) or not np.isclose(y0, y1)

    def test_invisible_point(self):
        """Point on the far side of Earth should give NaN."""
        sat = GOES16_CONFIG
        # Point at 180° away from sub-satellite point
        x, y = geodetic_to_fixed_grid(0.0, sat.sub_lon_deg + 180.0, sat)
        assert np.isnan(x) and np.isnan(y)

    def test_goes18_round_trip(self):
        """Round-trip for GOES-18."""
        sat = GOES18_CONFIG
        x_test = np.linspace(-0.08, 0.08, 20)
        y_test = np.linspace(-0.08, 0.08, 20)
        xx, yy = np.meshgrid(x_test, y_test)

        lat, lon = fixed_grid_to_geodetic(xx, yy, sat)
        valid = ~np.isnan(lat)
        lat_v, lon_v = lat[valid], lon[valid]

        x_rt, y_rt = geodetic_to_fixed_grid(lat_v, lon_v, sat)
        lat_rt, lon_rt = fixed_grid_to_geodetic(x_rt, y_rt, sat)

        np.testing.assert_allclose(lat_rt, lat_v, atol=0.001)
        np.testing.assert_allclose(lon_rt, lon_v, atol=0.001)


class TestGeodeticToECEF:
    """Test ECEF coordinate conversion."""

    def test_equator_prime_meridian(self):
        """(0, 0, 0) should give (a, 0, 0)."""
        X, Y, Z = geodetic_to_ecef(0.0, 0.0, 0.0)
        np.testing.assert_allclose(X, 6378137.0, atol=1.0)
        np.testing.assert_allclose(Y, 0.0, atol=1.0)
        np.testing.assert_allclose(Z, 0.0, atol=1.0)

    def test_north_pole(self):
        """North pole should give (0, 0, b)."""
        X, Y, Z = geodetic_to_ecef(90.0, 0.0, 0.0)
        np.testing.assert_allclose(X, 0.0, atol=1.0)
        np.testing.assert_allclose(Y, 0.0, atol=1.0)
        np.testing.assert_allclose(Z, 6356752.31414, atol=1.0)


class TestZenithAngle:
    """Test satellite zenith angle computation."""

    def test_subsatellite(self):
        """Zenith angle at sub-satellite point should be ~0."""
        sat = GOES16_CONFIG
        z = compute_zenith_angle(0.0, sat.sub_lon_deg, sat)
        assert z < 1.0

    def test_limb(self):
        """Zenith angle far from sub-satellite should be large."""
        sat = GOES16_CONFIG
        z = compute_zenith_angle(70.0, sat.sub_lon_deg, sat)
        assert z > 60.0


class TestSyntheticSolver:
    """Test navigation round-trips with known synthetic geometry."""

    def test_parallax_direction(self):
        """Parallax between GOES-16 and GOES-18 should be mostly E-W."""
        sat_a = GOES16_CONFIG
        sat_b = GOES18_CONFIG

        lat, lon = 30.0, -100.0
        h = 10000.0

        # Project (lat, lon, 0) and (lat, lon, h) into both grids
        xa0, ya0 = geodetic_to_fixed_grid(lat, lon, sat_a, h_m=0.0)
        xa1, ya1 = geodetic_to_fixed_grid(lat, lon, sat_a, h_m=h)
        xb0, yb0 = geodetic_to_fixed_grid(lat, lon, sat_b, h_m=0.0)
        xb1, yb1 = geodetic_to_fixed_grid(lat, lon, sat_b, h_m=h)

        # Parallax in A's grid vs B's grid
        dxa = xa1 - xa0
        dxb = xb1 - xb0

        # Differential parallax (what the solver sees) should be nonzero
        diff_x = dxb - dxa
        assert abs(diff_x) > 1e-8, "Cross-satellite parallax should be nonzero"
