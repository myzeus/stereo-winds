"""Integration tests for the stereo wind pipeline.

Tests the full pipeline with synthetic data (no network/GPU needed).
"""

from dataclasses import replace

import numpy as np

from stereo_winds.config import (
    GOES16_CONFIG,
    GOES18_CONFIG,
    GOES19_CONFIG,
    SatelliteConfig,
    StereoPairConfig,
    sector_config,
)
from stereo_winds.navigation import pixel_to_scanning_angle, fixed_grid_to_geodetic
from stereo_winds.pipeline import StereoWindPipeline
from stereo_winds.remap import build_remap_lut, remap_image
from stereo_winds.solver import (
    build_design_matrix,
    compute_parallax_vectors,
    pixels_to_wind_ms,
    solve_stereo_winds,
)
from stereo_winds.time_model import compute_scene_dt_fields, compute_scene_times
from stereo_winds.output import create_output_dataset
from datetime import datetime, timedelta


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
        # Most valid pixels should be finite (edge pixels may be NaN
        # due to half-pixel offset in per-pixel scale computation)
        finite_frac = np.isfinite(u_ms[valid]).mean()
        assert finite_frac > 0.8, f"Expected >80% finite valid pixels, got {finite_frac:.1%}"

        # 7. Create output dataset
        solution["u_wind"] = u_ms
        solution["v_wind"] = v_ms
        t0 = datetime(2024, 1, 15, 12, 0)
        ds = create_output_dataset(solution, sat_a, t0)

        assert "u_wind" in ds.data_vars
        assert "cloud_top_height" in ds.data_vars
        assert "quality_flag" in ds.data_vars
        assert ds.attrs["Conventions"] == "CF-1.8"

    def test_synthetic_sector_pipeline(self):
        """Sector (non-square, A/B grids of different size) end-to-end solve.

        Mimics a RadC run: both satellites see sub-windows of their disks,
        scan durations are CONUS-like (~150 s), and the solver runs on
        A's sector grid.
        """
        sat_a_full = _small_config(GOES16_CONFIG, 32)
        sat_b_full = _small_config(GOES18_CONFIG, 32)
        # A sector: rows 4..27 (24), cols 2..29 (28); B sector: rows
        # 6..25 (20), cols 4..27 (24) — windows on each disk's lattice.
        sat_a = replace(
            sat_a_full,
            x_offset=sat_a_full.x_offset + 2 * sat_a_full.scale_x,
            y_offset=sat_a_full.y_offset + 4 * sat_a_full.scale_y,
            n_rows=24, n_cols=28,
        )
        sat_b = replace(
            sat_b_full,
            x_offset=sat_b_full.x_offset + 4 * sat_b_full.scale_x,
            y_offset=sat_b_full.y_offset + 6 * sat_b_full.scale_y,
            n_rows=20, n_cols=24,
        )

        col_b, row_b = build_remap_lut(sat_a, sat_b)
        assert col_b.shape == (24, 28)

        w_u, w_v = compute_parallax_vectors(sat_a, sat_b)
        valid = np.isfinite(w_u) & np.isfinite(w_v) & (np.abs(w_u) + np.abs(w_v) > 0)
        assert valid.sum() > 0, "sector windows should overlap on-Earth"

        # CONUS-like timing: 150-s scans, ±10-min temporal offsets
        t0 = datetime(2026, 8, 11, 20, 0)
        scan = timedelta(seconds=150)
        mk = lambda t: {"t_nominal": t, "t_start": t, "t_end": t + scan,
                        "pixel_time": None}
        d = timedelta(minutes=10)
        time_info = {
            "A_minus": mk(t0 - d), "A0": mk(t0), "A_plus": mk(t0 + d),
            "B_minus": mk(t0 - d), "B_plus": mk(t0 + d),
        }
        dts = compute_scene_dt_fields(time_info, sat_a, sat_b, col_b, row_b)
        for k, v in dts.items():
            assert v.shape == (24, 28), k

        H_mat = build_design_matrix(
            w_u, w_v,
            dt_a_minus=dts["A_minus"], dt_a_plus=dts["A_plus"],
            dt_b_minus=dts["B_minus"], dt_b_plus=dts["B_plus"],
        )
        assert H_mat.shape == (24, 28, 8, 5)

        # Synthetic truth → observations → recovery
        x_true = np.array([8000.0, 0.2, -0.2, 3.0, 1.5])
        y = np.einsum("...ij,...j->...i", H_mat, x_true)
        y[~valid] = 0.0
        disparities = {
            "D1": np.stack([y[:, :, 0], y[:, :, 1]]),
            "D2": np.stack([y[:, :, 2], y[:, :, 3]]),
            "D3": np.stack([y[:, :, 4], y[:, :, 5]]),
            "D4": np.stack([y[:, :, 6], y[:, :, 7]]),
        }
        solution = solve_stereo_winds(disparities, H_mat)
        np.testing.assert_allclose(
            np.nanmean(solution["h"][valid]), 8000.0, atol=5.0)
        np.testing.assert_allclose(
            np.nanmean(solution["V_u"][valid]), 3.0, atol=0.1)

        # Output dataset carries the sector coordinates
        u_ms, v_ms = pixels_to_wind_ms(solution["V_u"], solution["V_v"], sat_a, 1.0)
        solution["u_wind"] = u_ms
        solution["v_wind"] = v_ms
        ds = create_output_dataset(solution, sat_a, t0)
        assert dict(ds.sizes) == {"y": 24, "x": 28}
        np.testing.assert_allclose(float(ds["x"][0]), sat_a.x_offset)
        np.testing.assert_allclose(float(ds["y"][0]), sat_a.y_offset)

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


class TestSectorCacheNaming:
    """Product/grid-aware cache and output naming (RadF names unchanged)."""

    def _pipeline(self, product: str) -> StereoWindPipeline:
        cfg = StereoPairConfig(
            sat_a=GOES19_CONFIG,
            sat_b=GOES18_CONFIG,
            product=product,
            model_ckpt_path="dummy.pt",
        )
        return StereoWindPipeline(cfg)

    def _radc_sector(self) -> SatelliteConfig:
        runtime = replace(
            GOES19_CONFIG,
            x_offset=GOES19_CONFIG.x_offset + 902 * GOES19_CONFIG.scale_x,
            y_offset=GOES19_CONFIG.y_offset + 566 * GOES19_CONFIG.scale_y,
            n_rows=1500,
            n_cols=2500,
        )
        return sector_config(GOES19_CONFIG, runtime)

    def test_full_disk_tag_empty(self):
        p = self._pipeline("ABI-L1b-RadF")
        assert p._grid_tag(GOES19_CONFIG) == ""

    def test_sector_tag(self):
        p = self._pipeline("ABI-L1b-RadC")
        assert p._grid_tag(self._radc_sector()) == "_radc_1500x2500"

    def test_radf_flow_cache_path_unchanged(self):
        """Pre-existing RadF flow cache filenames must stay byte-identical."""
        p = self._pipeline("ABI-L1b-RadF")
        t0 = datetime(2026, 8, 11, 20, 0)
        path = p._flow_cache_path(t0, "C14", "D1", "goes19", "goes18")
        assert path.name == "flow_goes19_goes18_2000z_C14_D1.npy"

    def test_radc_flow_cache_path_tagged(self):
        p = self._pipeline("ABI-L1b-RadC")
        t0 = datetime(2026, 8, 11, 20, 0)
        tag = p._grid_tag(self._radc_sector())
        path = p._flow_cache_path(t0, "C14", "D1", "goes19", "goes18", tag)
        assert path.name == "flow_goes19_goes18_2000z_C14_radc_1500x2500_D1.npy"

    def test_dt_tag(self):
        """Non-default pair offsets tag flow caches/outputs; default doesn't."""
        p10 = self._pipeline("ABI-L1b-RadC")
        assert p10._dt_tag() == ""
        cfg5 = StereoPairConfig(
            sat_a=GOES19_CONFIG, sat_b=GOES18_CONFIG,
            product="ABI-L1b-RadC", dt_minutes=5.0, model_ckpt_path="dummy.pt",
        )
        p5 = StereoWindPipeline(cfg5)
        assert p5._dt_tag() == "_dt5"
        t0 = datetime(2026, 8, 11, 20, 0)
        tag5 = p5._grid_tag(self._radc_sector()) + p5._dt_tag()
        path = p5._flow_cache_path(t0, "C14", "D1", "goes19", "goes18", tag5)
        assert path.name == "flow_goes19_goes18_2000z_C14_radc_1500x2500_dt5_D1.npy"

    def test_radc_lut_paths_distinct(self):
        """RadC remap/parallax cache paths differ from the RadF paths."""
        p = self._pipeline("ABI-L1b-RadC")
        sector = self._radc_sector()
        tag = p._grid_tag(sector)
        radc_lut = f"remap_lut_goes19_goes18{tag}.npz"
        radf_lut = "remap_lut_goes19_goes18.npz"
        assert radc_lut != radf_lut
        assert tag == "_radc_1500x2500"
