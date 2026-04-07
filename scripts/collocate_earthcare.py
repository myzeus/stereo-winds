#!/usr/bin/env python3
"""Collocate EarthCARE CLP-2B cloud top height with GOES-18/19 stereo pairs.

Downloads EarthCARE granules for a date range, projects cloud top height
onto both GOES-18 and GOES-19 pixel grids, and saves a parquet catalog
for training the stereo wind RAFT model.

Usage:
    python scripts/collocate_earthcare.py \
        --start 2026-02-01 --end 2026-02-28 \
        --output data/earthcare/feb2026_collocation.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pystac_client import Client

from stereo_winds.config import GOES18_CONFIG, GOES19_CONFIG
from stereo_winds.navigation import geodetic_to_fixed_grid, scanning_angle_to_pixel

logger = logging.getLogger(__name__)

# GOES-18/19 overlapping FOV (longitude range visible to both)
LON_WEST = -170.0
LON_EAST = -30.0

# MAAP constants
MAAP_CATALOG_URL = "https://catalog.maap.eo.esa.int/catalogue/"
MAAP_TOKEN_URL = (
    "https://iam.maap.eo.esa.int/realms/esa-maap/"
    "protocol/openid-connect/token"
)
MAAP_CLIENT_ID = "offline-token"
MAAP_CLIENT_SECRET = "***REMOVED***"
MAAP_COLLECTIONS = ["JAXAL2Validated_MAAP", "EarthCAREL2Validated_MAAP"]
MAAP_PRODUCT_TYPES = {"ACM_CLP_2B", "AC__CLP_2B"}


def get_access_token() -> str:
    """Exchange MAAP offline token for a bearer access token."""
    offline_token = os.environ.get("MAAP_OFFLINE_TOKEN")
    if not offline_token:
        raise ValueError(
            "Set MAAP_OFFLINE_TOKEN env var. Generate at "
            "https://portal.maap.eo.esa.int/ini/services/auth/token/90dToken.php"
        )
    resp = requests.post(
        MAAP_TOKEN_URL,
        data={
            "client_id": MAAP_CLIENT_ID,
            "client_secret": MAAP_CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": offline_token,
            "scope": "offline_access openid",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def search_clp2b(start: str, end: str) -> list:
    """Search MAAP catalog for CLP-2B granules in a date range."""
    catalog = Client.open(MAAP_CATALOG_URL)
    all_matches = []
    seen_ids = set()
    for collection in MAAP_COLLECTIONS:
        search = catalog.search(
            collections=[collection],
            datetime=[f"{start}T00:00:00Z", f"{end}T23:59:59Z"],
            method="GET",
            max_items=1000,
        )
        for i in search.items():
            if (i.properties.get("product:type") in MAAP_PRODUCT_TYPES
                    and i.id not in seen_ids):
                all_matches.append(i)
                seen_ids.add(i.id)
    return all_matches


def download_granules(items: list, cache_dir: Path, token: str) -> list[Path]:
    """Download granules to local cache, skipping existing files."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in items:
        asset = item.assets.get("enclosure_h5") or next(iter(item.assets.values()))
        url = asset.href
        filename = url.rsplit("/", 1)[-1]
        local_path = cache_dir / filename

        if not local_path.exists():
            logger.info(f"Downloading {filename}")
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
        paths.append(local_path)
    return paths


def collocate_file(path: Path, ec) -> list[dict]:
    """Collocate a single EarthCARE file with GOES-18/19 grids."""
    ds = ec.load_file(path)

    lat = ds.lat.values
    lon = ds.lon.values
    cth = ds.cloud_top_height.values
    time = ds.time.values
    is_cloudy = ds.is_cloudy.values

    # Extract multi-layer cloud structure
    layer_tops = ds["layer_tops"].values if "layer_tops" in ds else None
    layer_bases = ds["layer_bases"].values if "layer_bases" in ds else None
    n_layers = ds["n_layers"].values if "n_layers" in ds else None

    # Filter to GOES FOV and cloudy pixels
    fov_mask = (lon >= LON_WEST) & (lon <= LON_EAST) & is_cloudy
    if fov_mask.sum() == 0:
        return []

    lat_f = lat[fov_mask]
    lon_f = lon[fov_mask]
    cth_f = cth[fov_mask]
    time_f = time[fov_mask]
    if layer_tops is not None:
        layer_tops_f = layer_tops[fov_mask]
        layer_bases_f = layer_bases[fov_mask]
        n_layers_f = n_layers[fov_mask]
    else:
        layer_tops_f = layer_bases_f = n_layers_f = None

    # Project to GOES-18 pixel grid
    x18, y18 = geodetic_to_fixed_grid(lat_f, lon_f, GOES18_CONFIG)
    col18, row18 = scanning_angle_to_pixel(x18, y18, GOES18_CONFIG)

    # Project to GOES-19 pixel grid
    x19, y19 = geodetic_to_fixed_grid(lat_f, lon_f, GOES19_CONFIG)
    col19, row19 = scanning_angle_to_pixel(x19, y19, GOES19_CONFIG)

    # Valid: both satellites see the point and within grid bounds
    finite = (
        np.isfinite(col18) & np.isfinite(row18)
        & np.isfinite(col19) & np.isfinite(row19)
    )
    row18_i = np.where(finite, np.round(row18), -1).astype(np.int32)
    col18_i = np.where(finite, np.round(col18), -1).astype(np.int32)
    row19_i = np.where(finite, np.round(row19), -1).astype(np.int32)
    col19_i = np.where(finite, np.round(col19), -1).astype(np.int32)

    valid = finite & (
        (row18_i >= 0) & (row18_i < GOES18_CONFIG.n_rows)
        & (col18_i >= 0) & (col18_i < GOES18_CONFIG.n_cols)
        & (row19_i >= 0) & (row19_i < GOES19_CONFIG.n_rows)
        & (col19_i >= 0) & (col19_i < GOES19_CONFIG.n_cols)
    )

    if valid.sum() == 0:
        return []

    # Snap to nearest GOES 10-min scan
    goes_time = pd.DatetimeIndex(time_f[valid]).round("10min")

    records = []
    for j, i in enumerate(np.where(valid)[0]):
        rec = {
            "ec_file": path.name,
            "ec_lat": lat_f[i],
            "ec_lon": lon_f[i],
            "ec_cth_m": cth_f[i],
            "ec_time": time_f[i],
            "goes_time": goes_time[j],
            "row_18": row18_i[i],
            "col_18": col18_i[i],
            "row_19": row19_i[i],
            "col_19": col19_i[i],
        }
        if layer_tops_f is not None:
            rec["ec_n_layers"] = int(n_layers_f[i])
            for k in range(layer_tops_f.shape[1]):
                rec[f"ec_layer{k+1}_top_m"] = layer_tops_f[i, k]
                rec[f"ec_layer{k+1}_base_m"] = layer_bases_f[i, k]
        records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument(
        "--cache-dir",
        default="cache/earthcare/CLP_2B",
        help="Local cache directory for HDF5 files",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Lazy import to avoid loading zeus at parse time
    from zeus.datasets.sources.earthcare import earthcare

    logger.info(f"Searching MAAP for CLP-2B: {args.start} to {args.end}")
    items = search_clp2b(args.start, args.end)
    logger.info(f"Found {len(items)} granules")

    if not items:
        logger.warning("No granules found. Exiting.")
        return

    token = get_access_token()
    cache_dir = Path(args.cache_dir)
    paths = download_granules(items, cache_dir, token)
    logger.info(f"Downloaded/cached {len(paths)} files")

    ec = earthcare()
    all_records = []
    for fi, path in enumerate(paths):
        records = collocate_file(path, ec)
        all_records.extend(records)
        if (fi + 1) % 20 == 0:
            logger.info(
                f"  {fi + 1}/{len(paths)}: {len(all_records)} collocations"
            )

    if not all_records:
        logger.warning("No collocations found. Exiting.")
        return

    df = pd.DataFrame(all_records)

    # Compact dtypes
    df["ec_lat"] = df["ec_lat"].astype(np.float32)
    df["ec_lon"] = df["ec_lon"].astype(np.float32)
    df["ec_cth_m"] = df["ec_cth_m"].astype(np.float32)
    df["row_18"] = df["row_18"].astype(np.int16)
    df["col_18"] = df["col_18"].astype(np.int16)
    df["row_19"] = df["row_19"].astype(np.int16)
    df["col_19"] = df["col_19"].astype(np.int16)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    logger.info(f"Saved {len(df)} collocations to {output}")
    logger.info(f"  GOES scan times: {df['goes_time'].nunique()}")
    logger.info(
        f"  Lat: [{df['ec_lat'].min():.1f}, {df['ec_lat'].max():.1f}]"
    )
    logger.info(
        f"  Lon: [{df['ec_lon'].min():.1f}, {df['ec_lon'].max():.1f}]"
    )
    logger.info(
        f"  CTH: [{df['ec_cth_m'].min():.0f}, {df['ec_cth_m'].max():.0f}] m"
    )


if __name__ == "__main__":
    main()
