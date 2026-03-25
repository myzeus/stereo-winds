#!/usr/bin/env python
"""CLI entry point for batch stereo wind retrieval with Zarr output.

Usage:
    # Backfill all of 2024 at 12h cadence with 6 MPI ranks (default)
    mpirun -np 6 python scripts/run_batch.py --mode backfill --device cuda

    # Latest timestep (cron mode)
    mpirun -np 6 python scripts/run_batch.py --mode latest --freq 720

    # Single-process (no MPI, for testing)
    python scripts/run_batch.py --mode backfill \\
        --start "2024-01-01T00:00" --end "2024-01-01T12:00" \\
        --freq 720 --bands C14 --no-mpi
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from stereo_winds.batch import ALL_BANDS, BatchConfig, BatchProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Batch stereo wind retrieval with Zarr output",
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "latest"],
        required=True,
        help="backfill: process a time range; latest: process most recent timestep",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=720,
        help="Processing frequency in minutes (default: 720 = 12h)",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help=f"Bands to process (default: all {ALL_BANDS})",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2026-01-01T00:00",
        help="Start time for backfill (ISO format, default: 2024-01-01T00:00)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-01-31T23:00",
        help="End time for backfill (ISO format, default: 2024-12-31T12:00)",
    )
    parser.add_argument("--sat-a", default="goes19", help="Primary satellite")
    parser.add_argument("--sat-b", default="goes18", help="Secondary satellite")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/zarr"),
        help="Root directory for Zarr stores",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Cache directory for satellite data and flows",
    )
    parser.add_argument("--device", default="cuda", help="Compute device")
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max retries per timestep",
    )
    parser.add_argument(
        "--no-mpi",
        action="store_true",
        help="Run single-process without MPI",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Download GOES files to disk instead of streaming from S3",
    )
    parser.add_argument(
        "--arraylake-repo",
        default="zeus-ai/geo-stereo-winds",
        help="Arraylake repo for cloud storage (empty string to disable)",
    )
    parser.add_argument(
        "--arraylake-branch",
        default="main",
        help="Arraylake branch to write to (default: main)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging",
    )

    args = parser.parse_args()

    # Determine rank for log file naming
    rank_suffix = ""
    if not args.no_mpi:
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
            rank_suffix = f"_rank{rank}"
        except ImportError:
            args.no_mpi = True

    # Setup logging
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_handlers = [logging.StreamHandler()]
    log_file = args.output_root / f"batch{rank_suffix}.log"
    log_handlers.append(logging.FileHandler(str(log_file), mode="a"))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=f"%(asctime)s [%(levelname)s] rank{rank_suffix} %(name)s: %(message)s",
        handlers=log_handlers,
    )

    config = BatchConfig(
        sat_a=args.sat_a,
        sat_b=args.sat_b,
        bands=args.bands or list(ALL_BANDS),
        freq_minutes=args.freq,
        output_root=args.output_root,
        cache_dir=args.cache_dir,
        device=args.device,
        max_retries=args.max_retries,
        arraylake_repo=args.arraylake_repo,
        arraylake_branch=args.arraylake_branch,
        stream=not args.no_stream,
    )

    processor = BatchProcessor(config)

    if args.mode == "latest":
        results = processor.process_latest(
            bands=args.bands,
            use_mpi=not args.no_mpi,
        )
    elif args.mode == "backfill":
        start = datetime.fromisoformat(args.start)
        end = datetime.fromisoformat(args.end)

        if args.no_mpi:
            results = processor.process_time_range(start, end, args.bands)
        else:
            results = processor.run_mpi(start, end, args.bands)

    n_failed = len(results.get("failed", []))
    if n_failed > 0:
        logging.warning("%d timestep-band pairs failed", n_failed)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
