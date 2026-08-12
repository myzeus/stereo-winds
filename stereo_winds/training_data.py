"""Build Zarr dataset cubes from GOES ABI imagery for stereo wind training.

Downloads GOES imagery and writes to time-appendable Zarr stores with CF-1.8
metadata.  Each satellite/band combination gets its own store:

    {output_dir}/{satellite}_{band}.zarr     (time, y, x)

For the cross-satellite partner, a second "remapped" store is produced on the
reference satellite's fixed grid:

    {output_dir}/{satellite}_remap_{ref}_{band}.zarr   (time, y, x)

The Zarr stores are chunked for efficient spatial patch extraction via xbatcher
and support append-mode writes so new time steps can be added incrementally.

Usage
-----
# Build GOES-19 native cube
python -m stereo_winds.training_data \\
    --satellite goes19 --band C14 --monthly \\
    --start 2026-01-01T00:00 --end 2026-02-01T00:00 \\
    --output-dir data/stereo_training

# Build GOES-18 remapped onto GOES-19 grid
python -m stereo_winds.training_data \\
    --satellite goes18 --band C14 --monthly --remap-to goes19 \\
    --start 2026-01-01T00:00 --end 2026-02-01T00:00 \\
    --output-dir data/stereo_training
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time as _time
from pathlib import Path

import numpy as np
import xarray as xr

from .config import SATELLITE_CONFIGS, SatelliteConfig
from .data_loading import load_goes_scene
from .remap import (
    build_remap_lut,
    load_remap_lut,
    remap_image,
    save_remap_lut,
)

logger = logging.getLogger(__name__)

# Zarr chunking: 1 time step, 512x512 spatial tiles for xbatcher
CHUNKS = {"time": 1, "y": 512, "x": 512}


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------

def get_or_build_remap_lut(
    sat_a: SatelliteConfig,
    sat_b: SatelliteConfig,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or compute a remap LUT mapping sat_a's grid into sat_b's pixels."""
    a, b = sat_a.satellite_id, sat_b.satellite_id
    lut_path = cache_dir / f"remap_lut_{a}_{b}.npz"

    if lut_path.exists():
        logger.info("Loading cached remap LUT from %s", lut_path)
        return load_remap_lut(lut_path)

    # GOES-16 and GOES-19 share the same sub-longitude; reuse if available
    aliases = {"goes19": "goes16", "goes16": "goes19"}
    alias = aliases.get(a)
    if alias:
        alt_path = cache_dir / f"remap_lut_{alias}_{b}.npz"
        short_alias = alias.replace("goes", "g")
        short_b = b.replace("goes", "g")
        alt_path2 = cache_dir / f"remap_lut_{short_alias}_{short_b}.npz"
        for p in (alt_path, alt_path2):
            if p.exists():
                logger.info("Reusing compatible remap LUT from %s", p)
                col_b, row_b = load_remap_lut(p)
                save_remap_lut(col_b, row_b, lut_path)
                return col_b, row_b

    logger.info("Computing remap LUT (%s -> %s)...", a, b)
    t0 = _time.time()
    col_b, row_b = build_remap_lut(sat_a, sat_b)
    save_remap_lut(col_b, row_b, lut_path)
    logger.info("  Done in %.1fs", _time.time() - t0)
    return col_b, row_b


# ------------------------------------------------------------------
# Zarr cube construction
# ------------------------------------------------------------------

def _make_coords(sat: SatelliteConfig) -> dict:
    """Build y/x coordinate arrays (scanning angles in radians) for a satellite."""
    x_rad = np.arange(sat.n_cols, dtype=np.float64) * sat.scale_x + sat.x_offset
    y_rad = np.arange(sat.n_rows, dtype=np.float64) * sat.scale_y + sat.y_offset
    return {
        "x": xr.Variable(
            ["x"], x_rad,
            attrs={
                "long_name": "GOES fixed grid projection x-coordinate",
                "standard_name": "projection_x_coordinate",
                "units": "rad",
                "axis": "X",
            },
        ),
        "y": xr.Variable(
            ["y"], y_rad,
            attrs={
                "long_name": "GOES fixed grid projection y-coordinate",
                "standard_name": "projection_y_coordinate",
                "units": "rad",
                "axis": "Y",
            },
        ),
    }


def _make_grid_mapping(sat: SatelliteConfig) -> xr.Variable:
    """CF-1.8 grid mapping variable for geostationary projection."""
    return xr.Variable([], np.int32(0), attrs={
        "grid_mapping_name": "geostationary",
        "longitude_of_projection_origin": sat.sub_lon_deg,
        "perspective_point_height": sat.satellite_height_m,
        "semi_major_axis": sat.semi_major_m,
        "semi_minor_axis": sat.semi_minor_m,
        "sweep_angle_axis": sat.sweep,
        "latitude_of_projection_origin": 0.0,
    })


def _scene_to_dataset(
    data: np.ndarray,
    t: dt.datetime,
    sat: SatelliteConfig,
    band: str,
    long_name: str = "ABI L1b Radiance",
    units: str = "W m-2 sr-1 um-1",
) -> xr.Dataset:
    """Wrap a 2D radiance array as a single-timestep xarray Dataset."""
    coords = _make_coords(sat)
    coords["time"] = xr.Variable(
        ["time"],
        [np.datetime64(t, "ns")],
        attrs={"axis": "T"},
    )

    ds = xr.Dataset(
        data_vars={
            "Rad": xr.Variable(
                ["time", "y", "x"],
                data[np.newaxis, :, :].astype(np.float32),
                attrs={
                    "long_name": long_name,
                    "standard_name": "toa_outgoing_radiance_per_unit_wavelength",
                    "units": units,
                    "grid_mapping": "goes_imager_projection",
                },
            ),
            "goes_imager_projection": _make_grid_mapping(sat),
        },
        coords=coords,
        attrs={
            "Conventions": "CF-1.8",
            "satellite_id": sat.satellite_id,
            "band": band,
        },
    )
    return ds


def _zarr_encoding() -> dict:
    """Encoding options for Zarr writes."""
    return {
        "Rad": {"chunks": (1, CHUNKS["y"], CHUNKS["x"])},
        "time": {"units": "nanoseconds since 1970-01-01", "dtype": "int64"},
    }


def _existing_times(zarr_path: Path) -> set[np.datetime64]:
    """Return the set of times already in a Zarr store, or empty set."""
    if not zarr_path.exists():
        return set()
    try:
        ds = xr.open_zarr(zarr_path)
        times = set(ds.time.values)
        ds.close()
        return times
    except Exception:
        return set()


def append_to_zarr(ds: xr.Dataset, zarr_path: Path) -> None:
    """Append a single-timestep dataset to a Zarr store, creating if needed."""
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ds = ds.chunk(CHUNKS)

    if not zarr_path.exists():
        ds.to_zarr(str(zarr_path), mode="w", encoding=_zarr_encoding())
    else:
        ds.to_zarr(str(zarr_path), mode="a", append_dim="time")


# ------------------------------------------------------------------
# Triplet index
# ------------------------------------------------------------------

def _collect_times(zarr_paths: str | Path | list[str | Path]) -> set[np.datetime64]:
    """Collect all times from one or more Zarr stores."""
    if isinstance(zarr_paths, (str, Path)):
        zarr_paths = [zarr_paths]
    times: set[np.datetime64] = set()
    for p in zarr_paths:
        p = Path(p)
        if p.exists():
            ds = xr.open_zarr(str(p))
            times.update(ds.time.values)
            ds.close()
    return times


def build_triplet_index(
    sat_a_zarr: str | Path | list[str | Path],
    sat_b_zarr: str | Path | list[str | Path],
    dt_minutes: float = 10.0,
) -> list[np.datetime64]:
    """Find center times where a complete 5-scene stereo set exists.

    A valid triplet at time t0 requires:
      - sat_a store(s) have t0 - dt, t0, t0 + dt
      - sat_b store(s) have t0 - dt, t0 + dt

    Parameters
    ----------
    sat_a_zarr : path(s) to reference satellite Zarr store(s)
    sat_b_zarr : path(s) to partner satellite (remapped) Zarr store(s)
    dt_minutes : temporal offset between scenes

    Returns
    -------
    List of valid center times (sorted), suitable for random sampling.
    """
    times_a = _collect_times(sat_a_zarr)
    times_b = _collect_times(sat_b_zarr)

    delta = np.timedelta64(int(dt_minutes), "m")
    valid = []
    for t0 in sorted(times_a):
        t_minus = t0 - delta
        t_plus = t0 + delta
        if t_minus in times_a and t_plus in times_a and t_minus in times_b and t_plus in times_b:
            valid.append(t0)

    logger.info("Triplet index: %d valid center times out of %d in sat_a",
                len(valid), len(times_a))
    return valid


def load_stereo_triplet(
    t0: np.datetime64,
    ds_a: xr.Dataset,
    ds_b: xr.Dataset,
    dt_minutes: float = 10.0,
    y_slice: slice | None = None,
    x_slice: slice | None = None,
) -> dict[str, np.ndarray]:
    """Load a 5-scene stereo set from open Zarr datasets.

    Parameters
    ----------
    t0 : center time
    ds_a : reference satellite dataset (open)
    ds_b : partner satellite dataset, remapped to A's grid (open)
    dt_minutes : temporal offset
    y_slice, x_slice : optional pixel-index slices (e.g. slice(2000, 2512))

    Returns
    -------
    Dict with keys A_minus, A0, A_plus, B_minus, B_plus,
    each a 2D float32 numpy array.
    """
    delta = np.timedelta64(int(dt_minutes), "m")
    t_minus = t0 - delta
    t_plus = t0 + delta

    isel_kw = {}
    if y_slice is not None:
        isel_kw["y"] = y_slice
    if x_slice is not None:
        isel_kw["x"] = x_slice

    def _load(ds, t):
        arr = ds.Rad.sel(time=t)
        if isel_kw:
            arr = arr.isel(**isel_kw)
        return arr.values.astype(np.float32)

    return {
        "A_minus": _load(ds_a, t_minus),
        "A0":      _load(ds_a, t0),
        "A_plus":  _load(ds_a, t_plus),
        "B_minus": _load(ds_b, t_minus),
        "B_plus":  _load(ds_b, t_plus),
    }


# ------------------------------------------------------------------
# Processing
# ------------------------------------------------------------------

def _zarr_stem(satellite: str, band: str, remap_to: str | None = None) -> str:
    """Base name for a Zarr store (without month suffix or .zarr)."""
    if remap_to:
        return f"{satellite}_remap_{remap_to}_{band}"
    return f"{satellite}_{band}"


def _zarr_path_for(
    output_dir: Path,
    satellite: str,
    band: str,
    remap_to: str | None = None,
    month: str | None = None,
) -> Path:
    """Build the Zarr store path, optionally with a monthly suffix."""
    stem = _zarr_stem(satellite, band, remap_to)
    if month:
        return output_dir / f"{stem}_{month}.zarr"
    return output_dir / f"{stem}.zarr"


def process_satellite(
    satellite: str,
    band: str,
    timestamps: list[dt.datetime],
    output_dir: Path,
    data_cache_dir: str | None = None,
    product: str = "ABI-L1b-RadF",
    remap_to: str | None = None,
    remap_lut: tuple[np.ndarray, np.ndarray] | None = None,
    monthly: bool = False,
) -> tuple[int, int]:
    """Download scenes and write to a Zarr cube.

    Parameters
    ----------
    satellite : satellite id (e.g. "goes19", "goes18", "goes16")
    band : ABI band (e.g. "C14")
    timestamps : list of target datetimes
    output_dir : root output directory
    data_cache_dir : cache for raw GOES downloads
    product : ABI product type
    remap_to : if set, remap this satellite onto remap_to's grid
    remap_lut : (col_b, row_b) for remapping; required when remap_to is set
    monthly : if True, partition into monthly Zarr files (e.g. _202601.zarr)

    Returns
    -------
    (n_success, n_fail)
    """
    sat_cfg = SATELLITE_CONFIGS[satellite]
    ref_cfg = SATELLITE_CONFIGS[remap_to] if remap_to else sat_cfg

    # Pre-load existing times per store to avoid repeated opens
    existing_cache: dict[Path, set] = {}

    def _get_zarr(t: dt.datetime) -> Path:
        month = t.strftime("%Y%m") if monthly else None
        return _zarr_path_for(output_dir, satellite, band, remap_to, month)

    def _is_existing(t: dt.datetime) -> bool:
        zp = _get_zarr(t)
        if zp not in existing_cache:
            existing_cache[zp] = _existing_times(zp)
        return np.datetime64(t, "ns") in existing_cache[zp]

    n_success = 0
    n_fail = 0

    for i, t in enumerate(timestamps):
        if _is_existing(t):
            logger.debug("  %s: already in store, skipping", t)
            n_success += 1
            continue

        logger.info("[%d/%d] %s %s %s",
                    i + 1, len(timestamps), satellite, band, t.strftime("%Y-%m-%d %H:%M"))
        try:
            data, _ = load_goes_scene(
                t, band, satellite=satellite,
                cache_dir=data_cache_dir, product=product,
            )
        except Exception as e:
            logger.warning("  Failed to load %s %s at %s: %s", satellite, band, t, e)
            n_fail += 1
            continue

        # Remap if needed
        if remap_to and remap_lut is not None:
            col_b, row_b = remap_lut
            data = remap_image(data, col_b, row_b).astype(np.float32)

        ds = _scene_to_dataset(data, t, ref_cfg, band)
        zarr_path = _get_zarr(t)
        append_to_zarr(ds, zarr_path)
        n_success += 1

    # Summary per store
    stores = set(_get_zarr(t) for t in timestamps)
    for zp in sorted(stores):
        logger.info("  %s", zp.name)
    logger.info("%s: %d succeeded, %d failed",
                _zarr_stem(satellite, band, remap_to), n_success, n_fail)
    return n_success, n_fail


def process_fci_remap(
    bands: list[str],
    timestamps: list[dt.datetime],
    output_dir: Path,
    cache_dir: Path,
    data_cache_dir: str | None = None,
    satellite: str = "mtg-i1",
    remap_to: str = "goes19",
    monthly: bool = True,
    purge_cache: bool = False,
) -> tuple[int, int]:
    """Build FCI-remapped cubes on the reference satellite's grid.

    Every FCI chunk file carries all 16 channels, so each repeat cycle is
    downloaded and satpy-loaded ONCE for all requested bands (per-band loads
    would re-read the same ~6 GB cycle N times). Bands are given as ABI
    names ("C13") and translated via ABI_TO_FCI_BAND; stores are named like
    the GOES remap cubes (``mtg-i1_remap_goes19_C13_YYYYMM.zarr``) so the
    training triplet index and dataset code work unchanged.

    Parameters
    ----------
    bands : ABI band names with an FCI twin (e.g. ["C08", "C10", "C13"])
    timestamps : target datetimes (typically the paired A-side cube's
        time coordinate, so time coords match across stores)
    purge_cache : delete each repeat cycle's chunk files after processing
        (a cycle is ~6 GB; a month of sonde-aligned times is ~1 TB transient)

    Returns
    -------
    (n_success, n_fail) counted per (time, band) writes
    """
    raise NotImplementedError(
        "FCI training-data generation is not yet ported to the standalone "
        "build (requires satpy + eumdac). GOES-R ABI training data is "
        "supported.")

    import shutil  # noqa: F401  (deferred FCI port below)

    from zeus.datasets.core.base import DataSourceConfig
    from zeus.datasets.sources.mtg_fci import FCI, BANDS_VIS as FCI_BANDS_VIS

    from .config import ABI_TO_FCI_BAND
    from .data_loading import _block_mean

    ref_cfg = SATELLITE_CONFIGS[remap_to]
    sat_cfg = SATELLITE_CONFIGS[satellite]
    col_b, row_b = get_or_build_remap_lut(ref_cfg, sat_cfg, cache_dir)

    missing = [b for b in bands if b not in ABI_TO_FCI_BAND]
    if missing:
        raise ValueError(f"No FCI twin for bands {missing}; "
                         f"available: {sorted(ABI_TO_FCI_BAND)}")

    # Split into same-native-resolution groups so satpy never has to
    # cross-resample (mixed 1 km / 2 km loads through the native resampler
    # on the synchronous scheduler are pathologically slow). The 1 km
    # solar group is aggregated to the 2 km grid with a numpy block-mean.
    vis_group = [b for b in bands if ABI_TO_FCI_BAND[b] in FCI_BANDS_VIS]
    ir_group = [b for b in bands if ABI_TO_FCI_BAND[b] not in FCI_BANDS_VIS]
    groups: list[tuple[str, list[str], FCI]] = []
    for gname, gbands in (("ir", ir_group), ("vis", vis_group)):
        if gbands:
            groups.append((gname, gbands, FCI(
                config=DataSourceConfig(cache_dir=data_cache_dir),
                bands=[ABI_TO_FCI_BAND[b] for b in gbands],
            )))
    src = groups[0][2]  # any source: shared cycle cache dir for purge

    existing_cache: dict[Path, set] = {}

    def _store(band: str, t: dt.datetime) -> Path:
        month = t.strftime("%Y%m") if monthly else None
        return _zarr_path_for(output_dir, satellite, band, remap_to, month)

    def _is_existing(band: str, t: dt.datetime) -> bool:
        zp = _store(band, t)
        if zp not in existing_cache:
            existing_cache[zp] = _existing_times(zp)
        return np.datetime64(t, "ns") in existing_cache[zp]

    n_success = 0
    n_fail = 0

    for i, t in enumerate(timestamps):
        todo = [b for b in bands if not _is_existing(b, t)]
        if not todo:
            n_success += len(bands)
            continue

        logger.info("[%d/%d] %s %s (%d bands)", i + 1, len(timestamps),
                    satellite, t.strftime("%Y-%m-%d %H:%M"), len(todo))
        for gname, gbands, gsrc in groups:
            todo_g = [b for b in gbands if b in todo]
            if not todo_g:
                continue
            try:
                # One download + one same-area satpy load per group
                ds_fci = gsrc.data_at_time(t)
            except Exception as e:
                logger.warning("  Failed to load FCI %s group at %s: %s",
                               gname, t, e)
                n_fail += len(todo_g)
                continue

            gnames = [ABI_TO_FCI_BAND[b] for b in gbands]
            rad = ds_fci["Rad"].values[0]  # (band, y, x), y ascending
            for band in todo_g:
                arr = rad[gnames.index(ABI_TO_FCI_BAND[band])]
                if gname == "vis":
                    arr = _block_mean(arr, 2)  # 1 km solar -> 2 km grid
                arr = arr[::-1].astype(np.float32)  # north-up, LUT convention
                remapped = remap_image(arr, col_b, row_b).astype(np.float32)
                ds_out = _scene_to_dataset(
                    remapped, t, ref_cfg, band,
                    long_name="FCI L1c Effective Radiance (remapped)",
                    units=str(ds_fci["Rad"].attrs.get("units", "")),
                )
                zp = _store(band, t)
                append_to_zarr(ds_out, zp)
                existing_cache.setdefault(zp, set()).add(np.datetime64(t, "ns"))
                n_success += 1

        if purge_cache:
            cycle_dir = src._get_cache_dir(src._snap_time(t))
            if cycle_dir.exists():
                # Best-effort: panfs can report "directory not empty" on the
                # first pass (lazy entries); retry once, then leave leftovers.
                shutil.rmtree(cycle_dir, ignore_errors=True)
                if cycle_dir.exists():
                    shutil.rmtree(cycle_dir, ignore_errors=True)

    logger.info("%s_remap_%s [%s]: %d succeeded, %d failed",
                satellite, remap_to, ",".join(bands), n_success, n_fail)
    return n_success, n_fail


def generate_timestamps(
    start: dt.datetime,
    end: dt.datetime,
    cadence_minutes: float = 10.0,
) -> list[dt.datetime]:
    """Generate ABI full-disk timestamps in [start, end)."""
    timestamps = []
    t = start
    delta = dt.timedelta(minutes=cadence_minutes)
    while t < end:
        timestamps.append(t)
        t += delta
    return timestamps


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Zarr dataset cubes from GOES ABI imagery",
    )
    parser.add_argument("--satellite", required=True,
                        choices=list(SATELLITE_CONFIGS.keys()),
                        help="Satellite to process")
    parser.add_argument("--band", default="C14",
                        help="ABI band (default: C14). For an FCI satellite "
                             "a comma-separated list is allowed (e.g. "
                             "C07,C08,C10,C13) — all bands come from one "
                             "download/load per repeat cycle.")
    parser.add_argument("--purge-fci-cache", action="store_true",
                        help="FCI only: delete each repeat cycle's chunk files "
                             "after processing (~6 GB/cycle transient otherwise)")
    parser.add_argument("--start", required=True,
                        help="Start time (ISO format, e.g., 2026-03-01T00:00)")
    parser.add_argument("--end", required=True,
                        help="End time (ISO format, e.g., 2026-03-02T00:00)")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory for Zarr stores")
    parser.add_argument("--cache-dir", default="cache",
                        help="Cache directory for LUTs and downloaded data")
    parser.add_argument("--data-cache-dir", default=None,
                        help="Cache directory for raw GOES data (default: zeus default)")
    parser.add_argument("--cadence", type=float, default=10.0,
                        help="Time step in minutes (default: 10, ABI full-disk)")
    parser.add_argument("--product", default="ABI-L1b-RadF",
                        help="ABI product type (default: ABI-L1b-RadF)")
    parser.add_argument("--remap-to", default=None,
                        choices=list(SATELLITE_CONFIGS.keys()),
                        help="Remap onto this satellite's grid (e.g. goes19)")
    parser.add_argument("--monthly", action="store_true",
                        help="Partition into monthly Zarr files (e.g. _202601.zarr)")
    parser.add_argument("--times-file", default=None,
                        help="Path to a text file with one ISO timestamp per line. "
                             "If provided, only these timestamps are processed (overrides "
                             "--start/--end/--cadence).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    start = dt.datetime.fromisoformat(args.start)
    end = dt.datetime.fromisoformat(args.end)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.times_file:
        with open(args.times_file) as f:
            timestamps = [
                dt.datetime.fromisoformat(line.strip())
                for line in f if line.strip() and not line.startswith("#")
            ]
        # Filter to the requested month range so monthly partitioning works
        timestamps = [t for t in timestamps if start <= t < end]
    else:
        timestamps = generate_timestamps(start, end, args.cadence)
    logger.info("GOES Zarr Cube Builder")
    logger.info("  Satellite: %s", args.satellite)
    logger.info("  Band: %s", args.band)
    logger.info("  Time range: %s to %s (%d steps)", start, end, len(timestamps))
    logger.info("  Output: %s", output_dir)
    if args.remap_to:
        logger.info("  Remap onto: %s grid", args.remap_to)

    # FCI partner: multi-band per repeat cycle, always remapped
    if args.satellite.startswith("mtg"):
        if not args.remap_to:
            parser.error("FCI cubes are only produced remapped; pass --remap-to")
        process_fci_remap(
            bands=[b.strip() for b in args.band.split(",")],
            timestamps=timestamps,
            output_dir=output_dir,
            cache_dir=cache_dir,
            data_cache_dir=args.data_cache_dir,
            satellite=args.satellite,
            remap_to=args.remap_to,
            monthly=args.monthly,
            purge_cache=args.purge_fci_cache,
        )
        return

    # Build remap LUT if remapping
    remap_lut = None
    if args.remap_to:
        ref_cfg = SATELLITE_CONFIGS[args.remap_to]
        src_cfg = SATELLITE_CONFIGS[args.satellite]
        remap_lut = get_or_build_remap_lut(ref_cfg, src_cfg, cache_dir)

    process_satellite(
        satellite=args.satellite,
        band=args.band,
        timestamps=timestamps,
        output_dir=output_dir,
        data_cache_dir=args.data_cache_dir,
        product=args.product,
        remap_to=args.remap_to,
        remap_lut=remap_lut,
        monthly=args.monthly,
    )


if __name__ == "__main__":
    main()
