"""Tests for batch processing with Zarr storage."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from stereo_winds.batch import (
    ALL_BANDS,
    SPATIAL_CHUNK,
    ZARR_VARIABLES,
    BatchConfig,
    BatchProcessor,
    ZarrStore,
)
from stereo_winds.config import SatelliteConfig


_TEST_N = 100

_TEST_SAT_CONFIG = SatelliteConfig(
    satellite_id="test",
    sub_lon_deg=-75.0,
    n_rows=_TEST_N,
    n_cols=_TEST_N,
    scale_x=0.003 / _TEST_N,
    scale_y=-0.003 / _TEST_N,
    x_offset=-0.0015,
    y_offset=0.0015,
)


def _make_synthetic_dataset(t0: datetime, n_rows: int = _TEST_N, n_cols: int = _TEST_N):
    """Create a minimal dataset matching pipeline output format."""
    rng = np.random.default_rng(42)
    shape = (n_rows, n_cols)

    x_rad = np.arange(n_cols) * _TEST_SAT_CONFIG.scale_x + _TEST_SAT_CONFIG.x_offset
    y_rad = np.arange(n_rows) * _TEST_SAT_CONFIG.scale_y + _TEST_SAT_CONFIG.y_offset

    data_vars = {}
    for var in ZARR_VARIABLES:
        data_vars[var] = xr.Variable(
            ["y", "x"],
            rng.standard_normal(shape).astype(np.float32),
            attrs={"long_name": var, "units": "1", "grid_mapping": "goes_imager_projection"},
        )
    data_vars["position_correction_u"] = xr.Variable(
        ["y", "x"], np.zeros(shape, dtype=np.float32),
    )
    data_vars["goes_imager_projection"] = xr.Variable(
        [], np.int32(0),
        attrs={"grid_mapping_name": "geostationary", "longitude_of_projection_origin": -75.0},
    )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "x": xr.Variable(["x"], x_rad, attrs={"units": "rad", "axis": "X"}),
            "y": xr.Variable(["y"], y_rad, attrs={"units": "rad", "axis": "Y"}),
            "time": xr.Variable([], np.datetime64(t0), attrs={"axis": "T"}),
        },
        attrs={"Conventions": "CF-1.8", "title": "Test"},
    )
    return ds


def _init_store(store, t0, freq_minutes=720):
    """Initialize a store for tests."""
    store.initialize(t0, freq_minutes, _TEST_SAT_CONFIG)


# ---------------------------------------------------------------------------
# ZarrStore tests
# ---------------------------------------------------------------------------


class TestZarrStorePathRouting:
    """Test deterministic store path generation."""

    def test_monthly_for_12h_cadence(self):
        path = ZarrStore.store_path_for(
            Path("out"), "goes19_goes18", "C14",
            datetime(2025, 3, 15, 12, 0), freq_minutes=720,
        )
        assert path == Path("out/goes19_goes18/C14/2025-03.zarr")

    def test_monthly_for_6h_cadence(self):
        path = ZarrStore.store_path_for(
            Path("out"), "goes19_goes18", "C08",
            datetime(2025, 12, 31, 18, 0), freq_minutes=360,
        )
        assert path == Path("out/goes19_goes18/C08/2025-12.zarr")

    def test_daily_for_10min_cadence(self):
        path = ZarrStore.store_path_for(
            Path("out"), "goes19_goes18", "C14",
            datetime(2025, 6, 15, 14, 30), freq_minutes=10,
        )
        assert path == Path("out/goes19_goes18/C14/2025-06-15.zarr")

    def test_monthly_for_1h_cadence(self):
        path = ZarrStore.store_path_for(
            Path("out"), "goes19_goes18", "C04",
            datetime(2025, 1, 1, 0, 0), freq_minutes=60,
        )
        assert path == Path("out/goes19_goes18/C04/2025-01.zarr")


class TestZarrStoreIO:
    """Test Zarr store initialization, writing, and idempotency."""

    def test_init_and_write(self, tmp_path):
        """Pre-allocate store, then write one timestep via region."""
        store_path = tmp_path / "test.zarr"
        store = ZarrStore(store_path)

        assert not store.exists()

        t0 = datetime(2025, 1, 1, 0, 0)
        _init_store(store, t0)
        assert store.exists()

        ds = _make_synthetic_dataset(t0)
        store.write_timestep(ds)

        # Read back and verify
        ds_back = xr.open_zarr(str(store_path))
        assert "time" in ds_back.dims
        assert set(ds_back.data_vars) == set(ZARR_VARIABLES)
        np.testing.assert_array_almost_equal(
            ds_back["u_wind"].sel(time="2025-01-01T00:00").values, ds["u_wind"].values, decimal=5,
        )
        ds_back.close()

    def test_write_second_timestep(self, tmp_path):
        """Two writes go into correct time slots."""
        store_path = tmp_path / "test.zarr"
        store = ZarrStore(store_path)

        t1 = datetime(2025, 1, 1, 0, 0)
        t2 = datetime(2025, 1, 1, 12, 0)

        _init_store(store, t1)

        ds1 = _make_synthetic_dataset(t1)
        ds2 = _make_synthetic_dataset(t2)

        store.write_timestep(ds1)
        store.write_timestep(ds2)

        ds_back = xr.open_zarr(str(store_path))
        assert np.datetime64("2025-01-01T00:00") in ds_back.time.values
        assert np.datetime64("2025-01-01T12:00") in ds_back.time.values
        ds_back.close()

    def test_has_timestep(self, tmp_path):
        """Idempotency check: written slots are detected, empty slots are not."""
        store_path = tmp_path / "test.zarr"
        store = ZarrStore(store_path)

        t0 = datetime(2025, 1, 1, 0, 0)
        _init_store(store, t0)

        # Pre-allocated but not yet written — should be False
        assert not store.has_timestep(t0)

        ds = _make_synthetic_dataset(t0)
        store.write_timestep(ds)

        assert store.has_timestep(t0)
        assert not store.has_timestep(datetime(2025, 1, 1, 12, 0))

    def test_get_existing_times_empty(self, tmp_path):
        """Empty store returns empty array."""
        store = ZarrStore(tmp_path / "nonexistent.zarr")
        times = store.get_existing_times()
        assert len(times) == 0

    def test_excludes_position_corrections(self, tmp_path):
        """Position corrections and grid mapping are not in Zarr output."""
        store_path = tmp_path / "test.zarr"
        store = ZarrStore(store_path)

        t0 = datetime(2025, 1, 1, 0, 0)
        _init_store(store, t0)

        ds = _make_synthetic_dataset(t0)
        store.write_timestep(ds)

        ds_back = xr.open_zarr(str(store_path))
        assert "position_correction_u" not in ds_back.data_vars
        assert "goes_imager_projection" not in ds_back.data_vars
        ds_back.close()

    def test_preserves_grid_mapping_in_attrs(self, tmp_path):
        """Grid mapping metadata is stored in global attrs."""
        store_path = tmp_path / "test.zarr"
        store = ZarrStore(store_path)

        t0 = datetime(2025, 1, 1, 0, 0)
        _init_store(store, t0)

        ds_back = xr.open_zarr(str(store_path))
        assert "grid_mapping_name" in ds_back.attrs
        assert ds_back.attrs["grid_mapping_name"] == "geostationary"
        ds_back.close()


# ---------------------------------------------------------------------------
# BatchProcessor tests
# ---------------------------------------------------------------------------


class TestBatchProcessor:
    """Test batch processing logic."""

    def test_generate_times_12h(self):
        times = BatchProcessor._generate_times(
            datetime(2025, 1, 1, 0, 0),
            datetime(2025, 1, 2, 0, 0),
            freq_minutes=720,
        )
        assert len(times) == 3  # 00:00, 12:00, 00:00+1
        assert times[0] == datetime(2025, 1, 1, 0, 0)
        assert times[1] == datetime(2025, 1, 1, 12, 0)
        assert times[2] == datetime(2025, 1, 2, 0, 0)

    def test_generate_times_10min(self):
        times = BatchProcessor._generate_times(
            datetime(2025, 6, 15, 14, 0),
            datetime(2025, 6, 15, 15, 0),
            freq_minutes=10,
        )
        assert len(times) == 7  # 14:00, 14:10, ..., 15:00

    def test_generate_times_single(self):
        times = BatchProcessor._generate_times(
            datetime(2025, 1, 1, 0, 0),
            datetime(2025, 1, 1, 0, 0),
            freq_minutes=720,
        )
        assert len(times) == 1

    def test_process_single_skip_existing(self, tmp_path):
        """Already-processed timesteps are skipped."""
        config = BatchConfig(
            output_root=tmp_path / "zarr",
            cache_dir=tmp_path / "cache",
        )
        processor = BatchProcessor(config)

        t0 = datetime(2025, 1, 1, 0, 0)
        band = "C14"

        # Pre-populate the store
        store = processor._get_zarr_store(band, t0)
        _init_store(store, t0)
        ds = _make_synthetic_dataset(t0)
        store.write_timestep(ds)

        # Should skip without calling pipeline
        result = processor._process_single(t0, band)
        assert result is True

    @patch("stereo_winds.batch.SATELLITE_CONFIGS", {"goes16": _TEST_SAT_CONFIG})
    def test_process_single_writes_on_success(self, tmp_path):
        """Successful pipeline run writes to Zarr."""
        config = BatchConfig(
            output_root=tmp_path / "zarr",
            cache_dir=tmp_path / "cache",
        )
        processor = BatchProcessor(config)

        t0 = datetime(2025, 1, 1, 0, 0)
        band = "C14"
        ds = _make_synthetic_dataset(t0)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = ds
        processor._pipelines[band] = mock_pipeline

        result = processor._process_single(t0, band)
        assert result is True
        mock_pipeline.run.assert_called_once_with(t0, write_nc=False)

        # Verify written
        store = processor._get_zarr_store(band, t0)
        assert store.has_timestep(t0)

    @patch("stereo_winds.batch.SATELLITE_CONFIGS", {"goes16": _TEST_SAT_CONFIG})
    def test_process_single_handles_missing_data(self, tmp_path):
        """FileNotFoundError returns False without retry."""
        config = BatchConfig(
            output_root=tmp_path / "zarr",
            cache_dir=tmp_path / "cache",
            max_retries=3,
        )
        processor = BatchProcessor(config)

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = FileNotFoundError("No data on S3")
        processor._pipelines["C14"] = mock_pipeline

        result = processor._process_single(datetime(2025, 1, 1), "C14")
        assert result is False
        assert mock_pipeline.run.call_count == 1  # No retry

    @patch("stereo_winds.batch.SATELLITE_CONFIGS", {"goes16": _TEST_SAT_CONFIG})
    def test_process_single_retries_on_error(self, tmp_path):
        """Transient errors are retried up to max_retries."""
        config = BatchConfig(
            output_root=tmp_path / "zarr",
            cache_dir=tmp_path / "cache",
            max_retries=3,
            retry_delay_seconds=0.01,
        )
        processor = BatchProcessor(config)

        t0 = datetime(2025, 1, 1, 0, 0)
        ds = _make_synthetic_dataset(t0)

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = [
            RuntimeError("S3 timeout"),
            RuntimeError("S3 timeout"),
            ds,  # Third attempt succeeds
        ]
        processor._pipelines["C14"] = mock_pipeline

        result = processor._process_single(t0, "C14")
        assert result is True
        assert mock_pipeline.run.call_count == 3

    @patch("stereo_winds.batch.SATELLITE_CONFIGS", {"goes16": _TEST_SAT_CONFIG})
    def test_process_time_range_serial(self, tmp_path):
        """Serial time range processes all work units."""
        config = BatchConfig(
            output_root=tmp_path / "zarr",
            cache_dir=tmp_path / "cache",
            bands=["C14"],
            freq_minutes=720,
        )
        processor = BatchProcessor(config)

        ds1 = _make_synthetic_dataset(datetime(2025, 1, 1, 0, 0))
        ds2 = _make_synthetic_dataset(datetime(2025, 1, 1, 12, 0))

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = [ds1, ds2]
        processor._pipelines["C14"] = mock_pipeline

        results = processor.process_time_range(
            datetime(2025, 1, 1, 0, 0),
            datetime(2025, 1, 1, 12, 0),
            bands=["C14"],
        )

        assert len(results["succeeded"]) == 2
        assert len(results["failed"]) == 0


class TestBatchConfig:
    """Test batch configuration."""

    def test_sat_pair_id(self):
        config = BatchConfig(sat_a="goes19", sat_b="goes18")
        assert config.sat_pair_id == "goes19_goes18"

    def test_default_bands(self):
        config = BatchConfig()
        assert config.bands == ALL_BANDS
