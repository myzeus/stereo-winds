"""Tests for the standalone public-S3 GOES reader (no network)."""

import datetime as dt

import pytest

from stereo_winds.readers.goes import GOES, PRODUCT_TIMESTEPS


class FakeFS:
    """Records glob patterns; returns a single fake match."""

    def __init__(self):
        self.patterns = []

    def glob(self, pattern):
        self.patterns.append(pattern)
        return ["fake-key.nc"]


class TestSnapTime:
    def test_radf_10min(self):
        g = GOES(satellite="goes19", product="ABI-L1b-RadF")
        t = g._snap_time(dt.datetime(2026, 8, 11, 12, 34, 56))
        assert t == dt.datetime(2026, 8, 11, 12, 30)

    def test_radc_5min(self):
        g = GOES(satellite="goes19", product="ABI-L1b-RadC")
        assert g.step == 5
        assert (g._snap_time(dt.datetime(2026, 8, 11, 12, 34))
                == dt.datetime(2026, 8, 11, 12, 30))
        assert (g._snap_time(dt.datetime(2026, 8, 11, 12, 36))
                == dt.datetime(2026, 8, 11, 12, 35))

    def test_radm_1min(self):
        g = GOES(satellite="goes19", product="ABI-L1b-RadM")
        t = g._snap_time(dt.datetime(2026, 8, 11, 12, 34, 56))
        assert t == dt.datetime(2026, 8, 11, 12, 34)

    def test_product_timesteps(self):
        assert PRODUCT_TIMESTEPS["ABI-L1b-RadF"] == 10
        assert PRODUCT_TIMESTEPS["ABI-L1b-RadC"] == 5
        assert PRODUCT_TIMESTEPS["ABI-L1b-RadM"] == 1


class TestFindKey:
    """S3 glob pattern construction (RadC scans start +1 min past nominal)."""

    def _pattern(self, satellite, product, t, band="C14"):
        g = GOES(satellite=satellite, product=product, bands=[band])
        g._fs = FakeFS()
        g._find_key(t, band)
        assert len(g._fs.patterns) == 1
        return g._fs.patterns[0]

    def test_radf_pattern(self):
        p = self._pattern("goes19", "ABI-L1b-RadF",
                          dt.datetime(2026, 8, 11, 20, 0))
        assert p.startswith("noaa-goes19/ABI-L1b-RadF/2026/223/20/")
        assert "_G19_s20262232000" in p

    def test_radc_pattern_plus_one_minute(self):
        p = self._pattern("goes18", "ABI-L1b-RadC",
                          dt.datetime(2026, 8, 11, 20, 5))
        assert p.startswith("noaa-goes18/ABI-L1b-RadC/2026/223/20/")
        # nominal 20:05 slot -> scan start 20:06
        assert "_G18_s20262232006" in p

    def test_radc_snaps_then_offsets(self):
        # 20:07 snaps to the 20:05 slot, whose scan starts at 20:06
        p = self._pattern("goes19", "ABI-L1b-RadC",
                          dt.datetime(2026, 8, 11, 20, 7))
        assert "_G19_s20262232006" in p

    def test_no_match_raises(self):
        g = GOES(satellite="goes19", product="ABI-L1b-RadC", bands=["C14"])

        class EmptyFS:
            def glob(self, pattern):
                return []

        g._fs = EmptyFS()
        with pytest.raises(FileNotFoundError):
            g._find_key(dt.datetime(2026, 8, 11, 20, 5), "C14")


class TestRadToBT:
    def test_planck_inversion(self):
        """BT conversion inverts the forward Planck radiance calculation."""
        import numpy as np
        from stereo_winds.data_loading import _rad_to_bt

        # GOES-19 C14 planck constants (from a real L1b file)
        attrs = {"planck_fk1": 8510.22, "planck_fk2": 1286.67,
                 "planck_bc1": 0.18516, "planck_bc2": 0.99938}
        bt_true = np.array([[220.0, 250.0], [280.0, 300.0]])
        # forward: BT -> effective T -> radiance
        t_eff = attrs["planck_bc1"] + attrs["planck_bc2"] * bt_true
        rad = attrs["planck_fk1"] / (np.exp(attrs["planck_fk2"] / t_eff) - 1.0)
        bt = _rad_to_bt(rad.astype(np.float32), attrs)
        np.testing.assert_allclose(bt, bt_true, atol=1e-3)

    def test_missing_planck_raises(self):
        import numpy as np
        import pytest
        from stereo_winds.data_loading import _rad_to_bt

        with pytest.raises(ValueError, match="Planck"):
            _rad_to_bt(np.ones((2, 2), np.float32), {})
