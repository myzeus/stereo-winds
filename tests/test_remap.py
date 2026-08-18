"""Tests for cross-satellite remapping."""

from dataclasses import replace

import numpy as np
import pytest

from stereo_winds.config import GOES16_CONFIG, GOES18_CONFIG, SatelliteConfig
from stereo_winds.remap import build_remap_lut, compute_valid_mask, remap_image


class TestBuildRemapLUT:
    """Test LUT construction."""

    def _small_config(self, base: SatelliteConfig, n: int = 64) -> SatelliteConfig:
        """Create a small-grid version of a config for fast testing."""
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

    def test_identity_remap(self):
        """Remapping a satellite to itself should give near-identity LUT."""
        sat = self._small_config(GOES16_CONFIG, n=32)
        col_b, row_b = build_remap_lut(sat, sat)

        rows = np.arange(sat.n_rows)
        cols = np.arange(sat.n_cols)
        col_grid, row_grid = np.meshgrid(cols, rows)

        # On-Earth pixels should map to themselves
        valid = np.isfinite(col_b)
        np.testing.assert_allclose(col_b[valid], col_grid[valid], atol=0.1)
        np.testing.assert_allclose(row_b[valid], row_grid[valid], atol=0.1)

    def test_cross_satellite_overlap(self):
        """GOES-16 to GOES-18 LUT should have valid overlap region."""
        sat_a = self._small_config(GOES16_CONFIG, n=32)
        sat_b = self._small_config(GOES18_CONFIG, n=32)
        col_b, row_b = build_remap_lut(sat_a, sat_b)

        # Some pixels should be valid (overlapping coverage)
        valid = np.isfinite(col_b) & np.isfinite(row_b)
        assert valid.sum() > 0, "Should have some overlap between GOES-16 and GOES-18"

    def test_lut_shape(self):
        """LUT should match A's grid dimensions."""
        sat_a = self._small_config(GOES16_CONFIG, n=16)
        sat_b = self._small_config(GOES18_CONFIG, n=16)
        col_b, row_b = build_remap_lut(sat_a, sat_b)
        assert col_b.shape == (16, 16)
        assert row_b.shape == (16, 16)


class TestComputeValidMaskSector:
    """B-grid in-bounds masking when satellite B is a sector product."""

    def _small_config(self, base: SatelliteConfig, n: int = 32) -> SatelliteConfig:
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

    def test_sector_b_bounds_masked(self):
        """A pixels mapping outside B's sector window must be invalid."""
        n = 32
        sat_a = self._small_config(GOES16_CONFIG, n)
        # B = same geometry as A but only the central 16x16 window: the
        # remap is then an identity shifted by the window origin (8, 8).
        sat_b = replace(
            sat_a,
            x_offset=sat_a.x_offset + 8 * sat_a.scale_x,
            y_offset=sat_a.y_offset + 8 * sat_a.scale_y,
            n_rows=16,
            n_cols=16,
        )
        col_b, row_b = build_remap_lut(sat_a, sat_b)
        mask = compute_valid_mask(col_b, row_b, sat_a, sat_b, max_zenith=80.0)

        assert mask.shape == (n, n)
        assert mask.sum() > 0, "central window pixels should be valid"
        # Outside the sector window the LUT coords are finite but out of
        # B's grid bounds — must be masked.
        assert not mask[:8, :].any()
        assert not mask[:, :8].any()
        assert not mask[24:, :].any()
        assert not mask[:, 24:].any()
        # Center of the window (disk center, both zeniths ~0) is valid
        assert mask[16, 16]

    def test_full_disk_unchanged(self):
        """Full-disk B: bounds check is a no-op vs finiteness + zenith."""
        n = 32
        sat_a = self._small_config(GOES16_CONFIG, n)
        sat_b = self._small_config(GOES18_CONFIG, n)
        col_b, row_b = build_remap_lut(sat_a, sat_b)
        mask = compute_valid_mask(col_b, row_b, sat_a, sat_b, max_zenith=70.0)
        assert mask.sum() > 0


class TestRemapImage:
    """Test image remapping via LUT."""

    def test_identity_remap_preserves_image(self):
        """Remapping with identity LUT should preserve the image."""
        n = 32
        image = np.random.rand(n, n).astype(np.float32)

        # Identity LUT
        cols = np.arange(n, dtype=np.float64)
        rows = np.arange(n, dtype=np.float64)
        col_b, row_b = np.meshgrid(cols, rows)

        remapped = remap_image(image, col_b, row_b)
        np.testing.assert_allclose(remapped, image, atol=1e-5)

    def test_nan_in_lut(self):
        """NaN LUT entries should produce NaN in output."""
        n = 32
        image = np.ones((n, n), dtype=np.float32)

        cols = np.arange(n, dtype=np.float64)
        rows = np.arange(n, dtype=np.float64)
        col_b, row_b = np.meshgrid(cols, rows)
        col_b[0, 0] = np.nan

        remapped = remap_image(image, col_b, row_b)
        assert np.isnan(remapped[0, 0])
