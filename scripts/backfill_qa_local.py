#!/usr/bin/env python
"""Backfill quality_flag in local zarr chunks using ``compute_qa_flag``.

The chunks already store ``u_wind``, ``v_wind``, ``cloud_top_height``,
``chi_squared`` and ``sigma_h``, so we have everything ``compute_qa_flag``
needs.  This script recomputes the multi-level QA (0/1/2) for each scene
and writes it back to ``quality_flag`` in place via xarray region writes.

This fixes the long-standing bug where ``cache_student_dataset.py`` was
writing only the solver's basic flag (``h ∈ [0, 20 km]``), so the
``qa_high=2`` upweighting in ``StudentXBatchDataset`` was a no-op — no
pixel ever had ``qf=2``.

Usage:
    python scripts/backfill_qa_local.py PATH1 [PATH2 ...]

Each PATH must be a zarr store written by ``cache_student_dataset.py``
(must contain u_wind, v_wind, cloud_top_height, chi_squared, sigma_h,
quality_flag).  Writes are in-place; the existing quality_flag is
overwritten.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time as _time
from pathlib import Path

import numpy as np
import xarray as xr

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from stereo_winds.qa import compute_qa_flag

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def backfill_chunk(chunk_path: str) -> dict[int, int]:
    """Backfill quality_flag in a single zarr chunk; return aggregate level counts."""
    ds = xr.open_zarr(chunk_path, chunks=None)
    n_times = ds.sizes["time"]
    H, W = ds.sizes["y"], ds.sizes["x"]
    # zenith masking was applied at write time (out-of-disk pixels are already NaN
    # in the fields), so a fully-True valid_mask is the right input.
    valid_mask = np.ones((H, W), dtype=bool)

    counts = {0: 0, 1: 0, 2: 0}
    for t in range(n_times):
        ds_t = ds.isel(time=t)
        h = ds_t["cloud_top_height"].values
        if not np.any(np.isfinite(h)):
            continue  # empty scene; skip

        sol = {
            "h": h,
            "chi2": ds_t["chi_squared"].values,
            "sigma_h": ds_t["sigma_h"].values,
            "u_wind": ds_t["u_wind"].values,
            "v_wind": ds_t["v_wind"].values,
        }
        qa = compute_qa_flag(sol, valid_mask)

        # Region-write just quality_flag for this time slot.
        qa_ds = xr.Dataset(
            {"quality_flag": (("time", "y", "x"), qa[np.newaxis].astype(np.float32))},
        )
        qa_ds.to_zarr(chunk_path, mode="r+", region={"time": slice(t, t + 1)})

        # Tally level counts.
        for lvl in (0, 1, 2):
            counts[lvl] += int(np.sum(qa == lvl))

    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="One or more chunk zarr paths.")
    args = ap.parse_args()

    total = {0: 0, 1: 0, 2: 0}
    for p in args.paths:
        t0 = _time.time()
        logger.info(f"=== {p}")
        c = backfill_chunk(p)
        dt = _time.time() - t0
        tot = sum(c.values())
        if tot == 0:
            logger.info(f"  empty (no finite scenes); {dt:.1f}s")
            continue
        pct = {k: 100.0 * v / tot for k, v in c.items()}
        logger.info(
            f"  level0={c[0]:>14,} ({pct[0]:5.1f}%)  "
            f"level1={c[1]:>14,} ({pct[1]:5.1f}%)  "
            f"level2={c[2]:>14,} ({pct[2]:5.1f}%)  in {dt:.1f}s"
        )
        for k in total:
            total[k] += c[k]

    grand = sum(total.values())
    if grand > 0:
        logger.info("=== GRAND TOTAL across all chunks ===")
        pct = {k: 100.0 * v / grand for k, v in total.items()}
        logger.info(
            f"  level0={total[0]:>14,} ({pct[0]:5.1f}%)  "
            f"level1={total[1]:>14,} ({pct[1]:5.1f}%)  "
            f"level2={total[2]:>14,} ({pct[2]:5.1f}%)"
        )


if __name__ == "__main__":
    main()
