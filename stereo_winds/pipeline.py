"""End-to-end stereo wind retrieval pipeline.

Orchestrates: data loading → remapping → RAFT disparity →
5-state solver → wind conversion → output.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from .config import StereoPairConfig
from .data_loading import load_stereo_scenes
from .disparity import StereoDisparity
from .navigation import pixel_to_scanning_angle
from .output import create_output_dataset, write_netcdf
from .remap import (
    build_remap_lut,
    compute_valid_mask,
    load_remap_lut,
    remap_image,
    save_remap_lut,
)
from .solver import (
    build_design_matrix,
    compute_parallax_vectors,
    pixels_to_wind_ms,
    solve_stereo_winds,
)
from .time_model import compute_scene_times

logger = logging.getLogger(__name__)


class StereoWindPipeline:
    """Cross-satellite stereo wind retrieval pipeline.

    Parameters
    ----------
    config : StereoPairConfig
        Full pipeline configuration.
    """

    def __init__(self, config: StereoPairConfig):
        self.config = config
        self._remap_lut = None
        self._parallax_vectors = None
        self._valid_mask = None
        self._disparity_engine = None

    def _get_remap_lut(self, sat_a, sat_b):
        """Get or build the remap LUT, using disk cache if available."""
        if self._remap_lut is not None:
            return self._remap_lut

        cache_path = self.config.cache_dir / f"remap_lut_{sat_a.satellite_id}_{sat_b.satellite_id}.npz"
        if cache_path.exists():
            logger.info("Loading cached remap LUT from %s", cache_path)
            self._remap_lut = load_remap_lut(cache_path)
        else:
            logger.info("Building remap LUT (this may take a few seconds)...")
            self._remap_lut = build_remap_lut(sat_a, sat_b)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_remap_lut(*self._remap_lut, cache_path)
            logger.info("Remap LUT cached to %s", cache_path)

        return self._remap_lut

    def _get_parallax_vectors(self, sat_a, sat_b):
        """Get or compute parallax sensitivity vectors."""
        if self._parallax_vectors is not None:
            return self._parallax_vectors

        cache_path = self.config.cache_dir / f"parallax_{sat_a.satellite_id}_{sat_b.satellite_id}.npz"
        if cache_path.exists():
            logger.info("Loading cached parallax vectors from %s", cache_path)
            data = np.load(str(cache_path))
            self._parallax_vectors = (data["w_u"], data["w_v"])
        else:
            logger.info("Computing parallax vectors...")
            w_u, w_v = compute_parallax_vectors(sat_a, sat_b)
            self._parallax_vectors = (w_u, w_v)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(cache_path), w_u=w_u, w_v=w_v)
            logger.info("Parallax vectors cached to %s", cache_path)

        return self._parallax_vectors

    def _get_disparity_engine(self):
        """Lazily initialize RAFT disparity engine."""
        if self._disparity_engine is None:
            self._disparity_engine = StereoDisparity(
                model_ckpt_path=self.config.model_ckpt_path,
                tile_size=self.config.tile_size,
                overlap=self.config.overlap,
                batch_size=self.config.batch_size,
                device=self.config.device,
            )
        return self._disparity_engine

    def run(self, t0: datetime) -> xr.Dataset:
        """Run the full stereo wind retrieval for a single timestep.

        Parameters
        ----------
        t0 : center time for the retrieval

        Returns
        -------
        ds : xr.Dataset with wind vectors, height, and quality metrics
        """
        cfg = self.config
        logger.info("Starting stereo wind retrieval for %s", t0)

        # 1. Load 5 scenes in native fixed grid
        logger.info("Step 1: Loading scenes...")
        try:
            scenes = load_stereo_scenes(
                t0=t0,
                dt_minutes=cfg.dt_minutes,
                sat_a_id=cfg.sat_a.satellite_id,
                sat_b_id=cfg.sat_b.satellite_id,
                band_a=cfg.band,
                cache_dir=cfg.cache_dir,
            )
        except FileNotFoundError as e:
            logger.warning("B satellite data missing: %s. Falling back to temporal-only.", e)
            return self._run_temporal_only(t0)

        # Use actual satellite configs from file metadata
        sat_a = scenes["A0"][1]
        sat_b = scenes["B_minus"][1]

        # 2. Remap B scenes to A's grid
        logger.info("Step 2: Remapping B scenes to A's grid...")
        col_b, row_b = self._get_remap_lut(sat_a, sat_b)

        images = {
            "A_minus": scenes["A_minus"][0],
            "A0": scenes["A0"][0],
            "A_plus": scenes["A_plus"][0],
            "B_minus": remap_image(scenes["B_minus"][0], col_b, row_b),
            "B_plus": remap_image(scenes["B_plus"][0], col_b, row_b),
        }

        # 3. Run RAFT on 4 pairs
        logger.info("Step 3: Computing disparity fields (4 pairs)...")
        engine = self._get_disparity_engine()
        disparities = engine.compute_all(images)

        # 4. Build design matrix from parallax vectors + time offsets
        logger.info("Step 4: Building design matrix...")
        w_u, w_v = self._get_parallax_vectors(sat_a, sat_b)
        scene_times = compute_scene_times(t0, cfg.dt_minutes, sat_a, sat_b)
        H_matrix = build_design_matrix(
            w_u, w_v,
            dt_a_minus=scene_times["A_minus"],
            dt_a_plus=scene_times["A_plus"],
            dt_b_minus=scene_times["B_minus"],
            dt_b_plus=scene_times["B_plus"],
        )

        # 5. Solve 5-state WLS
        logger.info("Step 5: Solving 5-state system...")
        solution = solve_stereo_winds(disparities, H_matrix)

        # 6. Convert to m/s, apply QC flags
        logger.info("Step 6: Converting to physical units and applying QC...")
        dt_seconds = cfg.dt_minutes * 60.0
        u_ms, v_ms = pixels_to_wind_ms(solution["V_u"], solution["V_v"], sat_a, dt_seconds)
        solution["u_wind"] = u_ms
        solution["v_wind"] = v_ms

        # Apply chi2 threshold to quality flag
        solution["quality_flag"][solution["chi2"] > cfg.chi2_threshold] = 0.0

        # Apply valid mask
        valid_mask = compute_valid_mask(col_b, row_b, sat_a, sat_b, cfg.max_zenith_angle)
        solution["quality_flag"][~valid_mask] = 0.0

        # 7. Write output
        logger.info("Step 7: Writing output...")
        ds = create_output_dataset(solution, sat_a, t0, cfg)

        output_path = cfg.output_dir / f"stereo_winds_{t0:%Y%m%d_%H%M}.nc"
        write_netcdf(ds, output_path)
        logger.info("Output written to %s", output_path)

        return ds

    def _run_temporal_only(self, t0: datetime) -> xr.Dataset:
        """Fallback: temporal-only wind retrieval from A triplet.

        Uses only 2 pairs (A0→A_minus, A0→A_plus), solving for
        V_u and V_v only (no height retrieval).
        """
        cfg = self.config
        logger.info("Running temporal-only fallback for %s", t0)

        scenes = load_stereo_scenes(
            t0=t0,
            dt_minutes=cfg.dt_minutes,
            sat_a_id=cfg.sat_a.satellite_id,
            sat_b_id=cfg.sat_a.satellite_id,  # Both from A
            band_a=cfg.band,
            cache_dir=cfg.cache_dir,
        )

        sat_a = scenes["A0"][1]
        images = {
            "A_minus": scenes["A_minus"][0],
            "A0": scenes["A0"][0],
            "A_plus": scenes["A_plus"][0],
        }

        engine = self._get_disparity_engine()

        # Only 2 pairs
        d1 = engine._run_pair(images["A0"], images["A_minus"])
        d2 = engine._run_pair(images["A0"], images["A_plus"])

        # Simple average for velocity
        dt_seconds = cfg.dt_minutes * 60.0
        V_u = (d2[0] - d1[0]) / 2.0
        V_v = (d2[1] - d1[1]) / 2.0

        u_ms, v_ms = pixels_to_wind_ms(V_u, V_v, sat_a, dt_seconds)

        n_rows, n_cols = sat_a.n_rows, sat_a.n_cols
        solution = {
            "h": np.full((n_rows, n_cols), np.nan),
            "p_u": np.zeros((n_rows, n_cols)),
            "p_v": np.zeros((n_rows, n_cols)),
            "V_u": V_u,
            "V_v": V_v,
            "u_wind": u_ms,
            "v_wind": v_ms,
            "chi2": np.full((n_rows, n_cols), np.nan),
            "sigma_h": np.full((n_rows, n_cols), np.nan),
            "sigma_u": np.full((n_rows, n_cols), np.nan),
            "sigma_v": np.full((n_rows, n_cols), np.nan),
            "quality_flag": np.ones((n_rows, n_cols)),
        }

        ds = create_output_dataset(solution, sat_a, t0, cfg)
        ds.attrs["retrieval_mode"] = "temporal_only"

        output_path = cfg.output_dir / f"stereo_winds_temporal_{t0:%Y%m%d_%H%M}.nc"
        write_netcdf(ds, output_path)
        logger.info("Temporal-only output written to %s", output_path)

        return ds
