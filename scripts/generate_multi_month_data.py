#!/usr/bin/env python3
"""Generate GOES Zarr cubes and EarthCARE collocation labels for multiple months.

Orchestrates the existing training_data.py and collocate_earthcare.py CLIs
across a set of target months, then concatenates all collocation parquets
into a single combined file for training.

Usage:
    # Run everything (Zarr + collocation + combine)
    python scripts/generate_multi_month_data.py

    # Only generate GOES Zarr cubes
    python scripts/generate_multi_month_data.py --phase zarr

    # Only run EarthCARE collocation
    python scripts/generate_multi_month_data.py --phase collocation

    # Only combine existing parquets
    python scripts/generate_multi_month_data.py --phase combine

    # Process a subset of months
    python scripts/generate_multi_month_data.py --months 202507,202509
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
EARTHCARE_DIR = DATA_DIR / "earthcare"
TRAINING_DIR = Path("/home/ubuntu/earthnet-us-east-3/data/stereo_training")

# (start, end, label, YYYYMM key)
TARGET_MONTHS = [
    ("2024-01-01", "2024-02-01", "jan2024", "202401"),
    ("2024-02-01", "2024-03-01", "feb2024", "202402"),
    ("2024-03-01", "2024-04-01", "mar2024", "202403"),
    ("2024-04-01", "2024-05-01", "apr2024", "202404"),
    ("2024-05-01", "2024-06-01", "may2024", "202405"),
    ("2024-06-01", "2024-07-01", "jun2024", "202406"),
    ("2024-07-01", "2024-08-01", "jul2024", "202407"),
    ("2024-08-01", "2024-09-01", "aug2024", "202408"),
    ("2024-09-01", "2024-10-01", "sep2024", "202409"),
    ("2024-10-01", "2024-11-01", "oct2024", "202410"),
    ("2024-11-01", "2024-12-01", "nov2024", "202411"),
    ("2024-12-01", "2025-01-01", "dec2024", "202412"),
    ("2025-01-01", "2025-02-01", "jan2025", "202501"),
    ("2025-02-01", "2025-03-01", "feb2025", "202502"),
    ("2025-03-01", "2025-04-01", "mar2025", "202503"),
    ("2025-04-01", "2025-05-01", "apr2025", "202504"),
    ("2025-05-01", "2025-06-01", "may2025", "202505"),
    ("2025-06-01", "2025-07-01", "jun2025", "202506"),
    ("2025-07-01", "2025-08-01", "jul2025", "202507"),
    ("2025-08-01", "2025-09-01", "aug2025", "202508"),
    ("2025-09-01", "2025-10-01", "sep2025", "202509"),
    ("2025-10-01", "2025-11-01", "oct2025", "202510"),
    ("2025-11-01", "2025-12-01", "nov2025", "202511"),
    ("2025-12-01", "2026-01-01", "dec2025", "202512"),
    ("2026-01-01", "2026-02-01", "jan2026", "202601"),
    ("2026-02-01", "2026-03-01", "feb2026", "202602"),
]

DEFAULT_SAT_A = "goes19"
SATELLITE_B = "goes18"
BAND = "C14"
COMBINED_PARQUET = EARTHCARE_DIR / "all_months_collocation.parquet"


def run_cmd(cmd: list[str], description: str) -> bool:
    """Run a subprocess command, logging output. Returns True on success."""
    logger.info("Running: %s", description)
    logger.info("  %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(BASE))
    if result.returncode != 0:
        logger.error("  FAILED (exit code %d)", result.returncode)
        return False
    logger.info("  Done.")
    return True


def phase_zarr(
    months: list[tuple[str, str, str, str]],
    times_file: str | None = None,
    sat_a: str = DEFAULT_SAT_A,
    bands: list[str] | None = None,
) -> None:
    """Generate GOES Zarr cubes for each target month and band.

    If ``times_file`` is provided, only timestamps from that file (filtered to
    the month range) are processed. Useful for sparse sonde-supervised training
    where we only need 3 GOES scans per 12 hours.
    """
    extra = ["--times-file", times_file] if times_file else []
    band_list = bands if bands else [BAND]
    for band in band_list:
        for start, end, label, key in months:
            # Sat A native
            run_cmd(
                [
                    sys.executable, "-m", "stereo_winds.training_data",
                    "--satellite", sat_a,
                    "--band", band,
                    "--monthly",
                    "--start", f"{start}T00:00",
                    "--end", f"{end}T00:00",
                    "--output-dir", str(TRAINING_DIR),
                ] + extra,
                f"{sat_a} {band} {label}",
            )

            # GOES-18 remapped onto Sat A grid
            run_cmd(
                [
                    sys.executable, "-m", "stereo_winds.training_data",
                    "--satellite", SATELLITE_B,
                    "--band", band,
                    "--monthly",
                    "--remap-to", sat_a,
                    "--start", f"{start}T00:00",
                    "--end", f"{end}T00:00",
                    "--output-dir", str(TRAINING_DIR),
                ] + extra,
                f"{SATELLITE_B} remap {sat_a} {band} {label}",
            )


def phase_collocation(months: list[tuple[str, str, str, str]]) -> None:
    """Run EarthCARE collocation for each target month."""
    token = os.environ.get("MAAP_OFFLINE_TOKEN")
    if not token:
        logger.error(
            "MAAP_OFFLINE_TOKEN not set. "
            "Generate at https://portal.maap.eo.esa.int/ini/services/auth/token/90dToken.php"
        )
        sys.exit(1)

    EARTHCARE_DIR.mkdir(parents=True, exist_ok=True)

    for start, end, label, key in months:
        output = EARTHCARE_DIR / f"{label}_collocation.parquet"
        if output.exists():
            logger.info("Skipping %s — %s already exists", label, output.name)
            continue

        run_cmd(
            [
                sys.executable, str(BASE / "scripts" / "collocate_earthcare.py"),
                "--start", start,
                "--end", end,
                "--output", str(output),
            ],
            f"EarthCARE collocation {label}",
        )


def phase_combine(months: list[tuple[str, str, str, str]]) -> None:
    """Concatenate per-month collocation parquets into a single file."""
    EARTHCARE_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = []
    for _, _, label, _ in months:
        path = EARTHCARE_DIR / f"{label}_collocation.parquet"
        if path.exists():
            parquet_files.append((label, path))
        else:
            logger.warning("Missing %s — skipping", path.name)

    if not parquet_files:
        logger.error("No collocation parquets found. Run --phase collocation first.")
        sys.exit(1)

    frames = []
    for label, path in parquet_files:
        df = pd.read_parquet(path)
        df["source_month"] = label
        frames.append(df)
        logger.info("  %s: %d records", label, len(df))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(COMBINED_PARQUET, index=False)
    logger.info(
        "Combined %d records from %d months → %s",
        len(combined), len(frames), COMBINED_PARQUET.name,
    )
    logger.info("  GOES scan times: %d", combined["goes_time"].nunique())
    logger.info(
        "  CTH range: [%.0f, %.0f] m",
        combined["ec_cth_m"].min(), combined["ec_cth_m"].max(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["zarr", "collocation", "combine", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--months",
        default=None,
        help="Comma-separated YYYYMM keys to process (e.g. 202507,202509). Default: all.",
    )
    parser.add_argument(
        "--times-file",
        default=None,
        help="Path to a text file with one ISO timestamp per line. If provided, "
             "the zarr phase only fetches these timestamps (sparse sonde-aligned mode).",
    )
    parser.add_argument(
        "--sat-a",
        default=DEFAULT_SAT_A,
        help="Sat A satellite (goes19, goes16). GOES-East slot. Default: goes19.",
    )
    parser.add_argument(
        "--bands",
        default=BAND,
        help="Comma-separated list of bands to process (e.g. C08,C09,C10,C12,C14)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Filter months if requested
    months = TARGET_MONTHS
    if args.months:
        keys = set(args.months.split(","))
        months = [m for m in months if m[3] in keys]
        if not months:
            logger.error("No matching months for --months %s", args.months)
            sys.exit(1)

    logger.info("Target months: %s", [m[2] for m in months])

    bands_list = [b.strip() for b in args.bands.split(",")]

    if args.phase in ("zarr", "all"):
        phase_zarr(months, times_file=args.times_file, sat_a=args.sat_a, bands=bands_list)

    if args.phase in ("collocation", "all"):
        phase_collocation(months)

    if args.phase in ("combine", "all"):
        phase_combine(months)


if __name__ == "__main__":
    main()
