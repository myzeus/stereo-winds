#!/usr/bin/env python
"""One-time backfill: recompute quality_flag for an existing Arraylake group.

Reads the stored fields (cloud_top_height, chi_squared, sigma_h, u_wind,
v_wind), applies compute_qa_flag, and writes the updated quality_flag back
to Arraylake using region-based writes per timestep.

Usage:
    python scripts/backfill_qa_flag.py \
        --repo zeus-ai/geo-stereo-winds \
        --branch solver-v4 \
        --group goes19_goes18/1h/C14/2026-01
"""

from __future__ import annotations

import argparse
import logging

import arraylake
import numpy as np
import xarray as xr

from stereo_winds.qa import compute_qa_flag

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def backfill_qa(repo_name: str, branch: str, group: str) -> None:
    client = arraylake.Client()
    repo = client.get_repo(repo_name)

    # Read existing data
    logger.info("Reading %s/%s (branch=%s)...", repo_name, group, branch)
    session = repo.readonly_session(branch)
    ds = xr.open_zarr(session.store, group=group, consolidated=False)

    n_times = ds.sizes["time"]
    logger.info("Found %d timesteps, grid shape (%d, %d)",
                n_times, ds.sizes["y"], ds.sizes["x"])

    # valid_mask: True everywhere (zenith masking was applied at write time,
    # those pixels are already NaN in the fields)
    valid_shape = (ds.sizes["y"], ds.sizes["x"])
    valid_mask = np.ones(valid_shape, dtype=bool)

    updated = 0
    for t in range(n_times):
        ds_t = ds.isel(time=t)
        h = ds_t["cloud_top_height"].values
        # Skip all-NaN timesteps (no data)
        if np.all(np.isnan(h)):
            continue

        solution = {
            "h": h,
            "chi2": ds_t["chi_squared"].values,
            "sigma_h": ds_t["sigma_h"].values,
            "u_wind": ds_t["u_wind"].values,
            "v_wind": ds_t["v_wind"].values,
        }

        qa = compute_qa_flag(solution, valid_mask)

        # Build single-timestep dataset with just quality_flag
        qa_ds = xr.Dataset(
            {"quality_flag": (["time", "y", "x"], qa[np.newaxis, :, :])},
            coords={
                "time": ds.time.values[t:t+1],
                "y": ds.y,
                "x": ds.x,
            },
        )

        region = {
            "time": slice(t, t + 1),
            "y": slice(None),
            "x": slice(None),
        }

        from zeus.datasets.utils.zarr_utils import write_arraylake
        t_str = str(ds.time.values[t])
        write_arraylake(
            qa_ds, repo, group=group, branch=branch,
            mode="r+", region=region,
            commit=f"backfill QA flag: {group} t={t_str}",
        )
        updated += 1

        # Log level counts
        counts = {int(v): int(np.sum(qa == v)) for v in [0, 1, 2]}
        logger.info("  t=%d %s  QA counts: %s", t, t_str, counts)

    logger.info("Done. Updated %d / %d timesteps.", updated, n_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill QA flags on Arraylake")
    parser.add_argument("--repo", default="zeus-ai/geo-stereo-winds")
    parser.add_argument("--branch", default="solver-v4")
    parser.add_argument("--group", default="goes19_goes18/1h/C14/2026-01")
    args = parser.parse_args()

    backfill_qa(args.repo, args.branch, args.group)
