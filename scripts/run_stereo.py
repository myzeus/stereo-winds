#!/usr/bin/env python
"""CLI entry point for stereo wind retrieval."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from stereo_winds.config import SATELLITE_CONFIGS, StereoPairConfig
from stereo_winds.pipeline import StereoWindPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Cross-satellite stereo wind retrieval",
    )
    parser.add_argument(
        "time",
        type=str,
        help="Center time (ISO format, e.g. 2024-01-15T12:00)",
    )
    parser.add_argument(
        "--sat-a", default="goes16", choices=list(SATELLITE_CONFIGS.keys()),
        help="Primary satellite (default: goes16)",
    )
    parser.add_argument(
        "--sat-b", default="goes18", choices=list(SATELLITE_CONFIGS.keys()),
        help="Secondary satellite (default: goes18)",
    )
    parser.add_argument(
        "--band", default="C14",
        help="ABI/AHI band (default: C14)",
    )
    parser.add_argument(
        "--dt", type=float, default=10.0,
        help="Time offset in minutes (default: 10)",
    )
    parser.add_argument(
        "--model-ckpt", type=str, required=True,
        help="Path to RAFT model checkpoint",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device for RAFT (default: cpu)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache"),
        help="Cache directory (default: cache/)",
    )
    parser.add_argument(
        "--max-zenith", type=float, default=70.0,
        help="Max satellite zenith angle in degrees (default: 70)",
    )
    parser.add_argument(
        "--chi2-threshold", type=float, default=10.0,
        help="Chi-squared quality threshold (default: 10)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=512,
        help="RAFT tile size (default: 512)",
    )
    parser.add_argument(
        "--overlap", type=int, default=256,
        help="RAFT tile overlap (default: 256)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="RAFT batch size (default: 8)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t0 = datetime.fromisoformat(args.time)

    config = StereoPairConfig(
        sat_a=SATELLITE_CONFIGS[args.sat_a],
        sat_b=SATELLITE_CONFIGS[args.sat_b],
        band=args.band,
        dt_minutes=args.dt,
        model_ckpt_path=args.model_ckpt,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_zenith_angle=args.max_zenith,
        chi2_threshold=args.chi2_threshold,
    )

    pipeline = StereoWindPipeline(config)
    ds = pipeline.run(t0)

    print(f"Retrieval complete. Variables: {list(ds.data_vars)}")
    qf = ds["quality_flag"].values
    n_good = (qf > 0).sum()
    n_total = qf.size
    print(f"Quality: {n_good}/{n_total} pixels ({100*n_good/n_total:.1f}%)")


if __name__ == "__main__":
    main()
