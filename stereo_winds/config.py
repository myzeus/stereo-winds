"""Satellite and pipeline configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SatelliteConfig:
    """Per-satellite geostationary projection parameters.

    Scale/offset/dimensions are populated from netCDF file metadata at runtime;
    the defaults here are fallbacks for ABI 2 km full-disk.
    """

    satellite_id: str
    sub_lon_deg: float
    satellite_height_m: float = 35786023.0

    # GRS80 ellipsoid
    semi_major_m: float = 6378137.0
    semi_minor_m: float = 6356752.31414

    # Sweep axis: "x" for ABI, "y" for AHI
    sweep: str = "x"

    # Fixed-grid scale/offset (radians) — ABI 2 km full-disk defaults
    # pixel_to_scanning_angle: x = col * scale_x + x_offset
    scale_x: float = 5.6e-05
    scale_y: float = -5.6e-05
    x_offset: float = -0.151844
    y_offset: float = 0.151844  # north edge (row 0)

    # Grid dimensions (ABI 2 km full disk)
    n_rows: int = 5424
    n_cols: int = 5424


# ---------------------------------------------------------------------------
# Pre-built satellite configurations
# ---------------------------------------------------------------------------

GOES16_CONFIG = SatelliteConfig(
    satellite_id="goes16",
    sub_lon_deg=-75.0,
    sweep="x",
)

GOES18_CONFIG = SatelliteConfig(
    satellite_id="goes18",
    sub_lon_deg=-137.0,
    sweep="x",
)

HIMAWARI8_CONFIG = SatelliteConfig(
    satellite_id="himawari8",
    sub_lon_deg=140.7,
    sweep="y",
    # AHI 2 km full disk defaults
    scale_x=5.6e-05,
    scale_y=-5.6e-05,
    x_offset=-0.153719,
    y_offset=0.153719,
    n_rows=5500,
    n_cols=5500,
)

SATELLITE_CONFIGS = {
    "goes16": GOES16_CONFIG,
    "goes18": GOES18_CONFIG,
    "himawari8": HIMAWARI8_CONFIG,
}


@dataclass
class StereoPairConfig:
    """Configuration for the full stereo wind retrieval pipeline."""

    sat_a: SatelliteConfig
    sat_b: SatelliteConfig
    band: str = "C14"
    dt_minutes: float = 10.0

    # RAFT model settings
    model_ckpt_path: str = ""
    tile_size: int = 512
    overlap: int = 256
    batch_size: int = 8
    device: str = "cpu"

    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("output"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))

    # Quality control
    max_zenith_angle: float = 70.0
    chi2_threshold: float = 10.0
