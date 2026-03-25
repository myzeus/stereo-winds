"""Batch processing for stereo wind retrieval with Zarr storage.

Orchestrates multi-timestep, multi-band processing with MPI parallelism,
idempotent Zarr writes, and cron-compatible entry points.

Usage:
    # Backfill 2025 with 6 MPI ranks (one per band)
    mpirun -np 6 python scripts/run_batch.py \\
        --mode backfill --start "2025-01-01T00:00" --end "2025-12-31T12:00" \\
        --freq 720 --device cuda

    # Process latest timestep (cron mode)
    mpirun -np 6 python scripts/run_batch.py --mode latest --freq 720
"""

from __future__ import annotations

import calendar
import logging
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from filelock import FileLock

from .config import SATELLITE_CONFIGS, StereoPairConfig
from .pipeline import StereoWindPipeline

logger = logging.getLogger(__name__)

# Bands to process in default configuration
ALL_BANDS = ["C04", "C08", "C09", "C10", "C12", "C14"]

# Spatial chunk size: 5424 / 4 = 1356, evenly divides the ABI grid
SPATIAL_CHUNK = 1356

# Variables to store in Zarr (exclude position corrections and grid mapping)
ZARR_VARIABLES = [
    "u_wind", "v_wind", "cloud_top_height",
    "sigma_u", "sigma_v", "sigma_h",
    "chi_squared", "quality_flag",
]


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    sat_a: str = "goes16"
    sat_b: str = "goes18"
    bands: list[str] = field(default_factory=lambda: list(ALL_BANDS))
    freq_minutes: int = 720  # 12 hours
    output_root: Path = field(default_factory=lambda: Path("output/zarr"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    device: str = "cuda"
    max_retries: int = 3
    retry_delay_seconds: float = 30.0
    data_latency_minutes: int = 8

    # Arraylake cloud storage
    arraylake_repo: str = ""  # e.g. "zeus-ai/geo-stereo-winds", empty = disabled
    arraylake_branch: str = "main"

    # Pipeline settings forwarded to StereoPairConfig
    dt_minutes: float = 10.0
    n_iter: int = 3
    chi2_threshold: float = 10.0
    max_zenith_angle: float = 80.0
    stream: bool = True  # read GOES data directly from S3

    @property
    def sat_pair_id(self) -> str:
        return f"{self.sat_a}_{self.sat_b}"


class ZarrStore:
    """Manages a time-indexed Zarr store for one satellite pair and band.

    Handles initialization, idempotency checks, and append writes with
    file locking for multi-process safety.
    """

    def __init__(self, store_path: Path, sat_config=None):
        self.store_path = store_path
        self.sat_config = sat_config
        self._arraylake_branches_checked: set[str] = set()

    @staticmethod
    def store_path_for(
        output_root: Path,
        sat_pair: str,
        band: str,
        t0: datetime,
        freq_minutes: int,
    ) -> Path:
        """Determine the Zarr store path for a given timestep.

        Monthly stores for freq >= 360 (6 hours), daily otherwise.
        """
        if freq_minutes >= 360:
            period = t0.strftime("%Y-%m")
        else:
            period = t0.strftime("%Y-%m-%d")
        return output_root / sat_pair / band / f"{period}.zarr"

    def exists(self) -> bool:
        """Check if the store has been initialized."""
        # zarr v3 uses zarr.json; v2 uses .zmetadata or .zarray
        return (
            (self.store_path / "zarr.json").exists()
            or (self.store_path / ".zmetadata").exists()
            or (self.store_path / ".zattrs").exists()
        )

    def get_existing_times(self) -> np.ndarray:
        """Return array of datetime64 values already in the store."""
        if not self.exists():
            return np.array([], dtype="datetime64[ns]")
        ds = xr.open_zarr(str(self.store_path))
        times = ds.time.values
        ds.close()
        return times

    def has_timestep(self, t0: datetime) -> bool:
        """Check if a specific timestep has already been written.

        With pre-allocated stores, the time slot always exists. Instead
        check whether the data at that slot is non-NaN.
        """
        if not self.exists():
            return False
        ds = xr.open_zarr(str(self.store_path))
        t0_np = np.datetime64(t0, "ns")
        if t0_np not in ds.time.values:
            ds.close()
            return False
        # Check a single pixel of u_wind for non-NaN
        val = float(ds["u_wind"].sel(time=t0_np).isel(y=0, x=0).values)
        ds.close()
        return np.isfinite(val)

    @staticmethod
    def _build_encoding() -> dict:
        """Build per-variable encoding dict for zarr write."""
        encoding = {}
        for var in ZARR_VARIABLES:
            encoding[var] = {
                "chunks": (1, SPATIAL_CHUNK, SPATIAL_CHUNK),
                "dtype": "float32",
            }
        encoding["time"] = {"dtype": "int64", "units": "hours since 2000-01-01"}
        return encoding

    @staticmethod
    def _monthly_times(t0: datetime, freq_minutes: int) -> pd.DatetimeIndex:
        """Generate all time steps for the month containing t0."""
        start = datetime(t0.year, t0.month, 1)
        n_days = calendar.monthrange(t0.year, t0.month)[1]
        end = datetime(t0.year, t0.month, n_days, 23, 59)
        return pd.date_range(start, end, freq=f"{freq_minutes}min")

    def initialize(self, t0: datetime, freq_minutes: int, sat_config) -> None:
        """Pre-allocate the store with all time slots for the month.

        Creates a NaN-filled dataset with the full time axis so that
        individual timesteps can be written via region indexing.
        """
        times = self._monthly_times(t0, freq_minutes)
        n_times = len(times)
        n_rows, n_cols = sat_config.n_rows, sat_config.n_cols

        # Scanning-angle coordinates
        x_rad = np.arange(n_cols) * sat_config.scale_x + sat_config.x_offset
        y_rad = np.arange(n_rows) * sat_config.scale_y + sat_config.y_offset

        data_vars = {}
        for var in ZARR_VARIABLES:
            data_vars[var] = xr.Variable(
                ["time", "y", "x"],
                np.full((n_times, n_rows, n_cols), np.nan, dtype=np.float32),
                attrs={"_FillValue": np.float32(np.nan)},
            )

        ds = xr.Dataset(
            data_vars,
            coords={"time": times.values, "y": y_rad, "x": x_rad},
        )

        # Grid mapping metadata
        ds.attrs.update({
            "Conventions": "CF-1.8",
            "grid_mapping_name": "geostationary",
            "longitude_of_projection_origin": sat_config.sub_lon_deg,
            "perspective_point_height": sat_config.satellite_height_m,
            "semi_major_axis": sat_config.semi_major_m,
            "semi_minor_axis": sat_config.semi_minor_m,
            "sweep_angle_axis": sat_config.sweep,
            "latitude_of_projection_origin": 0.0,
        })

        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(
            str(self.store_path),
            mode="w",
            encoding=self._build_encoding(),
            compute=False,
        )
        logger.info(
            "Initialized store %s: %d time steps (%s to %s)",
            self.store_path, n_times, times[0], times[-1],
        )

    def _prepare_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Prepare a single-timestep dataset for Zarr storage.

        Promotes time from scalar coordinate to dimension,
        selects only ZARR_VARIABLES, copies grid metadata to attrs.
        """
        ds_3d = ds.expand_dims("time")
        ds_out = ds_3d[ZARR_VARIABLES]

        ds_out.attrs = ds.attrs.copy()
        if "goes_imager_projection" in ds.data_vars:
            ds_out.attrs.update(ds["goes_imager_projection"].attrs)

        return ds_out

    @staticmethod
    def arraylake_group_for(
        sat_pair: str, band: str, t0: datetime, freq_minutes: int,
    ) -> str:
        """Compute Arraylake group path for a given timestep."""
        if freq_minutes >= 60:
            freq_label = f"{freq_minutes // 60}h"
        else:
            freq_label = f"{freq_minutes}m"
        if freq_minutes >= 60:
            period = t0.strftime("%Y-%m")
        else:
            period = t0.strftime("%Y-%m-%d")
        return f"{sat_pair}/{freq_label}/{band}/{period}"

    def _compute_time_idx(self, t0_np, freq_minutes: int) -> int:
        """Compute the time index for a given timestep within its month."""
        times = self._monthly_times(
            datetime(
                pd.Timestamp(t0_np).year,
                pd.Timestamp(t0_np).month, 1,
            ),
            freq_minutes,
        )
        idx = np.searchsorted(times.values, t0_np)
        if idx >= len(times) or times[idx] != t0_np:
            raise ValueError(f"Time {t0_np} not in monthly grid")
        return int(idx)

    def write_timestep(
        self, ds: xr.Dataset, arraylake_repo_obj=None, arraylake_branch: str = "main",
        freq_minutes: int = 720, write_local: bool = True,
    ) -> None:
        """Write a single timestep to local Zarr and/or Arraylake.

        Parameters
        ----------
        ds : single-timestep dataset from pipeline
        arraylake_repo_obj : Arraylake repo object (None to skip)
        arraylake_branch : branch to write to
        freq_minutes : cadence for time index computation
        write_local : if True, write to local Zarr store
        """
        ds_3d = self._prepare_dataset(ds)
        t0_np = ds_3d.time.values[0]
        time_idx = self._compute_time_idx(t0_np, freq_minutes)

        if write_local:
            lock_path = self.store_path.parent / f".{self.store_path.name}.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)

            with FileLock(str(lock_path), timeout=300):
                if not self.exists():
                    raise RuntimeError(
                        f"Store {self.store_path} not initialized. "
                        "Call initialize() before writing timesteps."
                    )

                region = {
                    "time": slice(time_idx, time_idx + 1),
                    "y": slice(None),
                    "x": slice(None),
                }
                ds_3d.to_zarr(str(self.store_path), region=region)

            logger.info(
                "Wrote timestep %s (index %d) to %s",
                t0_np, time_idx, self.store_path,
            )

        # Write to Arraylake if enabled
        if arraylake_repo_obj is not None:
            self._write_to_arraylake(
                ds_3d, arraylake_repo_obj, arraylake_branch,
                time_idx=time_idx, freq_minutes=freq_minutes,
            )

    def _ensure_arraylake_branch(self, repo, branch: str) -> None:
        """Create the Arraylake branch if it doesn't exist yet.

        New branches are created from the repo's root (empty) snapshot
        so they start with a clean slate.
        """
        if branch == "main" or branch in self._arraylake_branches_checked:
            return
        try:
            repo.lookup_branch(branch)
            self._arraylake_branches_checked.add(branch)
        except Exception:
            # Walk ancestry to find the root snapshot
            root = None
            for snap in repo.ancestry(branch="main"):
                root = snap
            logger.info(
                "Creating Arraylake branch '%s' from root snapshot %s",
                branch, root.id,
            )
            repo.create_branch(branch, root.id)
            self._arraylake_branches_checked.add(branch)

    def _write_to_arraylake(
        self, ds_3d: xr.Dataset, repo, branch: str = "main",
        time_idx: int | None = None, freq_minutes: int = 720,
    ) -> None:
        """Write a single timestep into a pre-allocated Arraylake group."""
        from zeus.datasets.utils.zarr_utils import write_arraylake

        # Derive group path from store_path components
        parts = self.store_path.parts
        zarr_idx = next(i for i, p in enumerate(parts) if p.endswith(".zarr"))
        sat_pair = parts[zarr_idx - 2]
        band = parts[zarr_idx - 1]
        t0_dt = pd.Timestamp(ds_3d.time.values[0]).to_pydatetime()
        group = self.arraylake_group_for(sat_pair, band, t0_dt, freq_minutes)

        t_str = str(ds_3d.time.values[0])
        commit_msg = f"stereo-winds: {group} t={t_str}"

        # Write this timestep via region (group must be pre-initialized)
        region = {
            "time": slice(time_idx, time_idx + 1),
            "y": slice(None),
            "x": slice(None),
        }
        write_arraylake(
            ds_3d, repo, group=group, branch=branch,
            mode="r+", commit=commit_msg, region=region,
        )

        logger.info("Wrote timestep to Arraylake: %s/%s", repo, group)


class BatchProcessor:
    """Orchestrates batch stereo wind retrieval across times and bands.

    Supports MPI parallelism: distributes (timestep, band) work units
    across ranks. Reuses StereoWindPipeline instances per band to
    benefit from cached geometry and RAFT model weights.
    """

    def __init__(self, config: BatchConfig):
        self.config = config
        self._pipelines: dict[str, StereoWindPipeline] = {}
        self._al_repo = None

    @property
    def arraylake_repo(self):
        """Lazy-load Arraylake repo object (None if disabled)."""
        if self._al_repo is None and self.config.arraylake_repo:
            from arraylake import Client
            client = Client()
            self._al_repo = client.get_repo(self.config.arraylake_repo)
            logger.info("Connected to Arraylake repo: %s", self.config.arraylake_repo)
        return self._al_repo

    def _get_pipeline(self, band: str) -> StereoWindPipeline:
        """Get or create a pipeline for a specific band."""
        if band not in self._pipelines:
            pair_config = StereoPairConfig.from_satellites(
                sat_a=self.config.sat_a,
                sat_b=self.config.sat_b,
                band=band,
                device=self.config.device,
                cache_dir=self.config.cache_dir,
                dt_minutes=self.config.dt_minutes,
                n_iter=self.config.n_iter,
                chi2_threshold=self.config.chi2_threshold,
                max_zenith_angle=self.config.max_zenith_angle,
                stream=self.config.stream,
            )
            self._pipelines[band] = StereoWindPipeline(pair_config)
        return self._pipelines[band]

    def _init_arraylake_groups(
        self, times: list[datetime], bands: list[str],
    ) -> None:
        """Pre-create Arraylake branch and all needed groups (rank 0 only).

        Ensures the branch exists and creates pre-allocated empty groups
        for each unique (band, month) combination in the work list.
        """
        from zeus.datasets.utils.zarr_utils import group_exists_arraylake, write_arraylake

        repo = self.arraylake_repo
        branch = self.config.arraylake_branch
        freq = self.config.freq_minutes
        sat_config = SATELLITE_CONFIGS.get(self.config.sat_a)

        # Ensure branch exists
        store = self._get_zarr_store(bands[0], times[0])
        store._ensure_arraylake_branch(repo, branch)

        # Find unique (band, month) combinations
        seen_groups: set[str] = set()
        for t0 in times:
            for band in bands:
                group = ZarrStore.arraylake_group_for(
                    self.config.sat_pair_id, band, t0, freq,
                )
                if group in seen_groups:
                    continue
                seen_groups.add(group)

                if group_exists_arraylake(repo, group, branch):
                    logger.info("Arraylake group already exists: %s", group)
                    continue

                logger.info("Initializing Arraylake group: %s", group)
                import dask.array as da

                monthly_times = ZarrStore._monthly_times(t0, freq)
                n_rows, n_cols = sat_config.n_rows, sat_config.n_cols
                chunks = (1, SPATIAL_CHUNK, SPATIAL_CHUNK)
                shape = (len(monthly_times), n_rows, n_cols)
                empty_vars = {}
                for var in ZARR_VARIABLES:
                    empty_vars[var] = xr.Variable(
                        ["time", "y", "x"],
                        da.full(shape, np.nan, dtype=np.float32, chunks=chunks),
                    )
                x_rad = np.arange(n_cols) * sat_config.scale_x + sat_config.x_offset
                y_rad = np.arange(n_rows) * sat_config.scale_y + sat_config.y_offset
                ds_empty = xr.Dataset(
                    empty_vars,
                    coords={"time": monthly_times.values, "y": y_rad, "x": x_rad},
                )
                encoding = {}
                for var in ZARR_VARIABLES:
                    encoding[var] = {
                        "chunks": chunks,
                        "write_empty_chunks": False,
                    }
                write_arraylake(
                    ds_empty, repo, group=group, branch=branch,
                    mode="w", commit=f"Initialize {group}",
                    encoding=encoding,
                )
                del ds_empty

        logger.info("Arraylake initialization complete: %d groups", len(seen_groups))

    def _get_zarr_store(self, band: str, t0: datetime) -> ZarrStore:
        """Get a ZarrStore for a given band and time."""
        store_path = ZarrStore.store_path_for(
            self.config.output_root,
            self.config.sat_pair_id,
            band,
            t0,
            self.config.freq_minutes,
        )
        sat_config = SATELLITE_CONFIGS.get(self.config.sat_a)
        return ZarrStore(store_path, sat_config)

    def _ensure_store_initialized(self, band: str, t0: datetime) -> ZarrStore:
        """Ensure the Zarr store for this band/month is pre-allocated."""
        store = self._get_zarr_store(band, t0)
        if not store.exists():
            sat_config = SATELLITE_CONFIGS.get(self.config.sat_a)
            store.initialize(t0, self.config.freq_minutes, sat_config)
        return store

    def _process_single(self, t0: datetime, band: str) -> bool:
        """Process one timestep for one band.

        Returns True on success, False on failure.
        """
        write_local = not self.config.arraylake_repo
        store = self._get_zarr_store(band, t0)

        if write_local:
            store = self._ensure_store_initialized(band, t0)
            if store.has_timestep(t0):
                logger.info("Skipping %s %s -- already processed", t0, band)
                return True

        pipeline = self._get_pipeline(band)

        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.info(
                    "Processing %s %s (attempt %d/%d)",
                    t0, band, attempt, self.config.max_retries,
                )
                ds = pipeline.run(t0, write_nc=False)
                store.write_timestep(
                    ds,
                    arraylake_repo_obj=self.arraylake_repo,
                    arraylake_branch=self.config.arraylake_branch,
                    freq_minutes=self.config.freq_minutes,
                    write_local=write_local,
                )
                return True

            except FileNotFoundError as e:
                logger.warning("Data not available for %s %s: %s", t0, band, e)
                return False

            except Exception as e:
                logger.error(
                    "Attempt %d failed for %s %s: %s",
                    attempt, t0, band, e,
                    exc_info=True,
                )
                if attempt < self.config.max_retries:
                    _time.sleep(self.config.retry_delay_seconds)
                else:
                    logger.error(
                        "All %d attempts exhausted for %s %s",
                        self.config.max_retries, t0, band,
                    )
                    return False

        return False

    @staticmethod
    def _generate_times(
        start: datetime, end: datetime, freq_minutes: int,
    ) -> list[datetime]:
        """Generate timestep list from start to end (inclusive)."""
        freq = timedelta(minutes=freq_minutes)
        times = []
        t = start
        while t <= end:
            times.append(t)
            t += freq
        return times

    def run_mpi(
        self,
        start: datetime,
        end: datetime,
        bands: list[str] | None = None,
    ) -> dict[str, list]:
        """Process a time range with MPI parallelism.

        Distributes (timestep, band) work units round-robin across
        MPI ranks. Each rank processes its assigned units sequentially.

        Returns results dict on rank 0, empty dict on other ranks.
        """
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if bands is None:
            bands = self.config.bands

        times = self._generate_times(start, end, self.config.freq_minutes)

        # Build full work list and distribute round-robin
        all_work = [(t, b) for t in times for b in bands]
        my_work = all_work[rank::size]

        if rank == 0:
            logger.info(
                "MPI batch: %d total work units across %d ranks "
                "(%d times x %d bands)",
                len(all_work), size, len(times), len(bands),
            )

            # Rank 0 initializes Arraylake branch and groups before others start
            if self.config.arraylake_repo:
                self._init_arraylake_groups(times, bands)

        # Wait for rank 0 to finish initialization
        comm.Barrier()

        # All ranks: reopen repo to see rank 0's commits
        if self.config.arraylake_repo:
            from arraylake import Client
            self._al_repo = Client().get_repo(self.config.arraylake_repo)
            logger.info("Rank %d: refreshed Arraylake repo connection", rank)

        logger.info("Rank %d: %d work units assigned", rank, len(my_work))

        succeeded = []
        failed = []

        for i, (t0, band) in enumerate(my_work):
            ok = self._process_single(t0, band)
            if ok:
                succeeded.append((t0.isoformat(), band))
            else:
                failed.append((t0.isoformat(), band))

            if rank == 0 and (i + 1) % 10 == 0:
                logger.info(
                    "Rank 0 progress: %d/%d complete", i + 1, len(my_work),
                )

        # Gather results on rank 0
        all_succeeded = comm.gather(succeeded, root=0)
        all_failed = comm.gather(failed, root=0)

        if rank == 0:
            flat_succeeded = [item for sublist in all_succeeded for item in sublist]
            flat_failed = [item for sublist in all_failed for item in sublist]
            logger.info(
                "Batch complete: %d succeeded, %d failed",
                len(flat_succeeded), len(flat_failed),
            )
            if flat_failed:
                for t_str, band in flat_failed:
                    logger.warning("  FAILED: %s %s", t_str, band)
            return {"succeeded": flat_succeeded, "failed": flat_failed}

        return {"succeeded": [], "failed": []}

    def process_time_range(
        self,
        start: datetime,
        end: datetime,
        bands: list[str] | None = None,
    ) -> dict[str, list]:
        """Process a time range without MPI (single process).

        For testing and small jobs. For production use run_mpi().
        """
        if bands is None:
            bands = self.config.bands

        times = self._generate_times(start, end, self.config.freq_minutes)
        all_work = [(t, b) for t in times for b in bands]

        logger.info(
            "Serial batch: %d work units (%d times x %d bands)",
            len(all_work), len(times), len(bands),
        )

        succeeded = []
        failed = []

        for t0, band in all_work:
            ok = self._process_single(t0, band)
            if ok:
                succeeded.append((t0.isoformat(), band))
            else:
                failed.append((t0.isoformat(), band))

        logger.info(
            "Batch complete: %d succeeded, %d failed",
            len(succeeded), len(failed),
        )
        return {"succeeded": succeeded, "failed": failed}

    def process_latest(
        self,
        bands: list[str] | None = None,
        use_mpi: bool = False,
    ) -> dict[str, list]:
        """Process the most recent valid timestep (for cron operation).

        Snaps current time (minus latency buffer) to the frequency grid.
        """
        if bands is None:
            bands = self.config.bands

        now = datetime.utcnow()
        latency = timedelta(minutes=self.config.data_latency_minutes)
        available_time = now - latency

        # Snap to frequency grid (round down)
        freq = self.config.freq_minutes
        minutes_since_midnight = available_time.hour * 60 + available_time.minute
        snapped_minutes = (minutes_since_midnight // freq) * freq
        t0 = available_time.replace(
            hour=snapped_minutes // 60,
            minute=snapped_minutes % 60,
            second=0,
            microsecond=0,
        )

        logger.info(
            "Processing latest: now=%s, available=%s, snapped=%s",
            now.isoformat(), available_time.isoformat(), t0.isoformat(),
        )

        if use_mpi:
            return self.run_mpi(t0, t0, bands)
        return self.process_time_range(t0, t0, bands)
