#!/usr/bin/env python
"""Cache the 5 raw scenes for a single stereo timestep from NOAA S3.

Fetches goes-A and goes-B ABI L1b files for the 3 needed timestamps
(t0 - dt, t0, t0 + dt) using zeus, builds the B->A remap LUT if missing,
and writes the standard 5 .npy files in the same layout as
zarrs/C14/{YYYYMMDD_HHMM}/{A0,A_minus,A_plus,B_minus,B_plus}.npy used
by the pretrained inference comparison scripts.

Example
-------
    python scripts/cache_scene_from_s3.py \
        --time 2025-01-08T19:00 \
        --sat-a goes16 --sat-b goes18 --band C14 \
        --zeus-cache /explore/nobackup/people/tvandal/data/zeus_cache \
        --lut-dir   /explore/nobackup/people/tvandal/data/stereo-winds/cache \
        --out-dir   /explore/nobackup/people/tvandal/data/stereo-winds/zarrs/C14
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import numpy as np

from stereo_winds.config import SATELLITE_CONFIGS
from stereo_winds.data_loading import load_goes_scene
from stereo_winds.remap import (
    build_remap_lut,
    load_remap_lut,
    remap_image,
    save_remap_lut,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", required=True,
                        help="Center time (ISO format, e.g. 2025-01-08T19:00)")
    parser.add_argument("--sat-a", default="goes16",
                        choices=list(SATELLITE_CONFIGS.keys()))
    parser.add_argument("--sat-b", default="goes18",
                        choices=list(SATELLITE_CONFIGS.keys()))
    parser.add_argument("--band", default="C14")
    parser.add_argument("--dt-minutes", type=float, default=10.0)
    parser.add_argument("--zeus-cache", required=True,
                        help="Local cache dir for downloaded ABI files")
    parser.add_argument("--lut-dir", required=True,
                        help="Where to cache the B->A remap LUT npz")
    parser.add_argument("--out-dir", required=True,
                        help="Parent dir for cached scenes; output goes to "
                             "<out-dir>/<YYYYMMDD_HHMM>/")
    parser.add_argument("--stream", action="store_true",
                        help="Stream from S3 (no local cache)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    t0 = datetime.fromisoformat(args.time)
    dt = timedelta(minutes=args.dt_minutes)
    t_minus, t_plus = t0 - dt, t0 + dt

    out_dir = Path(args.out_dir) / t0.strftime("%Y%m%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing 5 scenes to %s", out_dir)

    # ----- Load sat-A scenes (no remap) -----
    logger.info("Loading %s %s scenes...", args.sat_a, args.band)
    a_minus, sat_a_cfg = load_goes_scene(
        t_minus, args.band, args.sat_a,
        cache_dir=args.zeus_cache, stream=args.stream,
    )
    a0, _ = load_goes_scene(
        t0, args.band, args.sat_a,
        cache_dir=args.zeus_cache, stream=args.stream,
    )
    a_plus, _ = load_goes_scene(
        t_plus, args.band, args.sat_a,
        cache_dir=args.zeus_cache, stream=args.stream,
    )

    # ----- Load sat-B scenes (will be remapped onto A's grid) -----
    logger.info("Loading %s %s scenes...", args.sat_b, args.band)
    b_minus_raw, sat_b_cfg = load_goes_scene(
        t_minus, args.band, args.sat_b,
        cache_dir=args.zeus_cache, stream=args.stream,
    )
    b_plus_raw, _ = load_goes_scene(
        t_plus, args.band, args.sat_b,
        cache_dir=args.zeus_cache, stream=args.stream,
    )

    # ----- Build/load remap LUT (B native grid -> A native grid) -----
    # Use canonical SATELLITE_CONFIGS (NOT the satpy-derived per-scene
    # configs) so the LUT matches the one used by training_data.py to build
    # the existing monthly zarrs. satpy returns sub_lon=-75.20 from ABI
    # metadata, but training_data.py uses the canonical -75.0, and the
    # downstream parallax/solver also uses canonical. Mixing satpy-derived
    # LUT with canonical solver geometry produces a constant cross-sat
    # translation that the WLS interprets as a bogus +12 km height bias.
    sat_a_lut = SATELLITE_CONFIGS[args.sat_a]
    sat_b_lut = SATELLITE_CONFIGS[args.sat_b]
    lut_dir = Path(args.lut_dir)
    lut_dir.mkdir(parents=True, exist_ok=True)
    lut_path = lut_dir / f"remap_lut_{args.sat_a}_{args.sat_b}.npz"
    if lut_path.exists():
        logger.info("Loading existing remap LUT: %s", lut_path)
        col_b, row_b = load_remap_lut(lut_path)
    else:
        logger.info("Building remap LUT %s -> %s (canonical configs)...",
                    args.sat_b, args.sat_a)
        col_b, row_b = build_remap_lut(sat_a_lut, sat_b_lut)
        save_remap_lut(col_b, row_b, lut_path)
        logger.info("Saved LUT to %s", lut_path)

    logger.info("Remapping %s onto %s grid...", args.sat_b, args.sat_a)
    b_minus = remap_image(b_minus_raw, col_b, row_b).astype(np.float32)
    b_plus = remap_image(b_plus_raw, col_b, row_b).astype(np.float32)

    # ----- Save scenes -----
    scenes = {
        "A_minus": a_minus, "A0": a0, "A_plus": a_plus,
        "B_minus": b_minus, "B_plus": b_plus,
    }
    for name, arr in scenes.items():
        path = out_dir / f"{name}.npy"
        np.save(path, arr)
        finite_frac = float(np.isfinite(arr).mean())
        logger.info("  %s: %s, finite=%.1f%%, range=[%.2g, %.2g]",
                    name, arr.shape, 100 * finite_frac,
                    np.nanmin(arr), np.nanmax(arr))

    # ----- Save per-scene sat configs so downstream uses the same geometry
    # as the LUT (avoids sub_lon / scale drift between static SatelliteConfig
    # defaults and satpy-derived per-file metadata, which propagates as a
    # constant cross-sat flow offset and bogus heights).
    import json
    from dataclasses import asdict
    config_path = out_dir / "sat_configs.json"
    with open(config_path, "w") as fh:
        json.dump(
            {"sat_a": asdict(sat_a_cfg), "sat_b": asdict(sat_b_cfg)},
            fh, indent=2,
        )
    logger.info("Saved per-scene sat configs: %s", config_path)
    logger.info("  sat_a sub_lon=%.4f, scale_x=%.4e, x_offset=%.4e",
                sat_a_cfg.sub_lon_deg, sat_a_cfg.scale_x, sat_a_cfg.x_offset)
    logger.info("  sat_b sub_lon=%.4f, scale_x=%.4e, x_offset=%.4e",
                sat_b_cfg.sub_lon_deg, sat_b_cfg.scale_x, sat_b_cfg.x_offset)

    logger.info("Done.")


if __name__ == "__main__":
    main()
