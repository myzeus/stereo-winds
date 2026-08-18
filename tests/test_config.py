"""Tests for configuration helpers (sector_config lattice snapping)."""

import logging
from dataclasses import replace

import numpy as np

from stereo_winds.config import GOES19_CONFIG, sector_config


class TestSectorConfig:
    """sector_config: adapt a canonical full-disk config to a sector grid."""

    def test_full_disk_identity(self):
        """Full-disk runtime dims return the canonical object unchanged."""
        runtime = replace(
            GOES19_CONFIG,
            x_offset=GOES19_CONFIG.x_offset + 1e-9,  # metadata round-off
        )
        assert sector_config(GOES19_CONFIG, runtime) is GOES19_CONFIG

    def test_sector_snaps_to_lattice(self):
        """Sector offsets snap to the canonical lattice, keep canonical scale."""
        canon = GOES19_CONFIG
        k_col, k_row = 902, 566  # CONUS-like window position (integer pixels)
        runtime = replace(
            canon,
            # 0.02-pixel round-off from the meters->radians conversion
            x_offset=canon.x_offset + (k_col + 0.02) * canon.scale_x,
            y_offset=canon.y_offset + (k_row - 0.02) * canon.scale_y,
            n_rows=1500,
            n_cols=2500,
        )
        out = sector_config(canon, runtime)

        assert (out.n_rows, out.n_cols) == (1500, 2500)
        assert out.scale_x == canon.scale_x
        assert out.scale_y == canon.scale_y
        assert out.sub_lon_deg == canon.sub_lon_deg
        assert out.satellite_height_m == canon.satellite_height_m
        np.testing.assert_allclose(
            out.x_offset, canon.x_offset + k_col * canon.scale_x, atol=1e-15)
        np.testing.assert_allclose(
            out.y_offset, canon.y_offset + k_row * canon.scale_y, atol=1e-15)

    def test_off_lattice_warns(self, caplog):
        """A sector window off the canonical lattice warns but still snaps."""
        canon = GOES19_CONFIG
        runtime = replace(
            canon,
            x_offset=canon.x_offset + 10.5 * canon.scale_x,
            n_rows=100,
            n_cols=100,
        )
        with caplog.at_level(logging.WARNING, logger="stereo_winds.config"):
            out = sector_config(canon, runtime)
        assert any("lattice" in r.message for r in caplog.records)
        # Still snapped onto an integer pixel of the canonical lattice
        col_off = (out.x_offset - canon.x_offset) / canon.scale_x
        np.testing.assert_allclose(col_off, round(col_off), atol=1e-9)
