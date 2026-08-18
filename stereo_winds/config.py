"""Satellite and pipeline configuration dataclasses."""

from __future__ import annotations

import glob as _glob
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

logger = logging.getLogger(__name__)


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

# MTG-I1 FCI, FDHSI 2 km full-disk grid (5568², extent ±5568 km at
# perspective height 35786400 m -> 2000/35786400 rad/px). Fallbacks only —
# data_loading derives the exact values from the file's projection metadata.
MTG_I1_CONFIG = SatelliteConfig(
    satellite_id="mtg-i1",
    sub_lon_deg=0.0,
    satellite_height_m=35786400.0,
    sweep="y",  # Meteosat convention (only GOES ABI uses x-sweep)
    scale_x=5.58871e-05,
    scale_y=-5.58871e-05,
    x_offset=-0.155562,
    y_offset=0.155562,
    n_rows=5568,
    n_cols=5568,
)

SATELLITE_CONFIGS = {
    "goes16": GOES16_CONFIG,
    "goes18": GOES18_CONFIG,
    "goes19": GOES19_CONFIG,
    "himawari8": HIMAWARI8_CONFIG,
    "himawari9": HIMAWARI9_CONFIG,
    "mtg-i1": MTG_I1_CONFIG,
}


def sector_config(
    canonical: SatelliteConfig,
    runtime: SatelliteConfig,
    tol_px: float = 0.1,
) -> SatelliteConfig:
    """Adapt a canonical full-disk config to a sector (e.g. ABI CONUS) grid.

    ABI sector products (RadC/RadM) live on the same fixed-grid lattice as
    the full disk: same angular scale, offsets shifted by an integer number
    of pixels. Runtime configs derived from file metadata can carry small
    float round-off in the offsets (meters -> radians conversion), which
    would shift the remap LUT and inflate height errors, so instead of using
    the runtime offsets directly we snap the sector window onto the
    canonical lattice and keep the canonical scale/ellipsoid/sub-lon.

    Parameters
    ----------
    canonical : full-disk registry config (e.g. GOES19_CONFIG)
    runtime : config derived from the loaded scene's metadata
    tol_px : warn if the snap residual exceeds this many pixels

    Returns
    -------
    SatelliteConfig — ``canonical`` unchanged for a full-disk grid, else a
    copy with the sector's dimensions and lattice-snapped offsets.
    """
    if (runtime.n_rows, runtime.n_cols) == (canonical.n_rows, canonical.n_cols):
        return canonical

    col_off = (runtime.x_offset - canonical.x_offset) / canonical.scale_x
    row_off = (runtime.y_offset - canonical.y_offset) / canonical.scale_y
    residual = max(abs(col_off - round(col_off)), abs(row_off - round(row_off)))
    if residual > tol_px:
        logger.warning(
            "Sector grid for %s is %.3f px off the canonical lattice "
            "(col_off=%.3f, row_off=%.3f); snapping anyway",
            runtime.satellite_id, residual, col_off, row_off,
        )
    return replace(
        canonical,
        x_offset=canonical.x_offset + round(col_off) * canonical.scale_x,
        y_offset=canonical.y_offset + round(row_off) * canonical.scale_y,
        n_rows=runtime.n_rows,
        n_cols=runtime.n_cols,
    )

# ---------------------------------------------------------------------------
# Cross-instrument band equivalence (closest spectral centers)
# ---------------------------------------------------------------------------

# ABI band -> FCI (satpy fci_l1c_nc) channel. Used when one satellite of a
# stereo pair is an FCI: the retrieval is configured with the ABI band name
# and the FCI side loads its closest equivalent. ABI C09 (6.9 um) and
# C14 (11.2 um) have no FCI twin and are absent.
ABI_TO_FCI_BAND = {
    "C01": "vis_04",   # 0.47 / 0.444 um
    "C02": "vis_06",   # 0.64 / 0.64  um
    "C03": "vis_08",   # 0.865 / 0.865 um
    "C04": "nir_13",   # 1.378 / 1.38 um
    "C05": "nir_16",   # 1.61 / 1.61  um
    "C06": "nir_22",   # 2.25 / 2.25  um
    "C07": "ir_38",    # 3.90 / 3.80  um
    "C08": "wv_63",    # 6.19 / 6.30  um
    "C10": "wv_73",    # 7.34 / 7.35  um
    "C11": "ir_87",    # 8.44 / 8.70  um
    "C12": "ir_97",    # 9.61 / 9.66  um
    "C13": "ir_105",   # 10.35 / 10.50 um
    "C15": "ir_123",   # 12.30 / 12.30 um
    "C16": "ir_133",   # 13.30 / 13.30 um
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

    # Optional Carr et al. ABI per-pixel scan-time LUT for satellite A
    # (see time_model.load_abi_time_lut). Empty -> auto-discover
    # {cache_dir}/abi_time_model.nc, else fall back to the linear model.
    abi_time_lut_path: str = ""

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
