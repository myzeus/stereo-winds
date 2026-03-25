"""Satellite and pipeline configuration dataclasses."""

from __future__ import annotations

import glob as _glob
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

GOES19_CONFIG = SatelliteConfig(
    satellite_id="goes19",
    sub_lon_deg=-75.0,
    sweep="x",
)

HIMAWARI9_CONFIG = SatelliteConfig(
    satellite_id="himawari9",
    sub_lon_deg=140.7,
    sweep="y",
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
    "goes19": GOES19_CONFIG,
    "himawari8": HIMAWARI8_CONFIG,
    "himawari9": HIMAWARI9_CONFIG,
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
    overlap: int = 128
    batch_size: int = 8
    device: str = "gpu"

    # Solver settings
    n_iter: int = 3
    product: str = "ABI-L1b-RadF"

    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("output"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))

    # Data loading
    stream: bool = False  # if True, read GOES data directly from S3

    # Quality control
    max_zenith_angle: float = 80.0
    chi2_threshold: float = 10.0

    @classmethod
    def from_satellites(
        cls,
        sat_a: str,
        sat_b: str,
        band: str = "C14",
        **kwargs,
    ) -> "StereoPairConfig":
        """Create a config from satellite ID strings.

        Looks up satellite configs from the registry and auto-finds
        the RAFT checkpoint in the zeus weights directory.

        Parameters
        ----------
        sat_a, sat_b : satellite identifiers (e.g., "goes19", "goes18")
        band : ABI/AHI band name (e.g., "C14", "B14")
        **kwargs : override any StereoPairConfig field
        """
        if sat_a not in SATELLITE_CONFIGS:
            raise ValueError(f"Unknown satellite '{sat_a}'. Known: {list(SATELLITE_CONFIGS)}")
        if sat_b not in SATELLITE_CONFIGS:
            raise ValueError(f"Unknown satellite '{sat_b}'. Known: {list(SATELLITE_CONFIGS)}")

        # Auto-find RAFT checkpoint if not provided
        if "model_ckpt_path" not in kwargs or not kwargs["model_ckpt_path"]:
            zeus_weights = Path(__file__).resolve().parent.parent / "zeus" / "zeus" / "networks" / "weights"
            ckpts = sorted(_glob.glob(str(zeus_weights / "raft-128.*.ckpt")))
            if ckpts:
                kwargs["model_ckpt_path"] = ckpts[-1]  # latest
            else:
                raise FileNotFoundError(
                    f"No RAFT checkpoint found in {zeus_weights}. "
                    "Pass model_ckpt_path explicitly."
                )

        defaults = dict(
            sat_a=SATELLITE_CONFIGS[sat_a],
            sat_b=SATELLITE_CONFIGS[sat_b],
            band=band,
            device="cuda",
            cache_dir=Path("cache"),
        )
        defaults.update(kwargs)
        return cls(**defaults)
