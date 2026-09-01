"""Download and process IGRA data into obs-xvec ArrayLake.

Fetches a full year of IGRA soundings (Jan-Dec) for stereo wind training.

Not part of the standalone build: this script depends on a sibling ``zeus``
checkout (for ``scripts/igra_historical_processing.py``) and on ArrayLake
credentials, neither of which ships with this repository. It is kept for
reference and raises a clear error if those are unavailable.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

ZEUS_DIR = Path(__file__).resolve().parent.parent / "zeus"
sys.path.insert(0, str(ZEUS_DIR))

try:
    import arraylake
    from scripts.igra_historical_processing import year_to_xvec, download_year
except ImportError as exc:  # pragma: no cover - depends on external checkout
    raise ImportError(
        "process_igra_2025.py requires the private `zeus` checkout (for "
        "scripts/igra_historical_processing.py) and the `arraylake` package, "
        "neither of which is part of this standalone build. IGRA collocation "
        "for evaluation is handled by scripts/collocate_igra.py instead."
    ) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get(
            "IGRA_CACHE_DIR",
            str(Path(__file__).resolve().parent.parent / "cache" / "igra_raw"),
        ),
        help="Cache directory for raw IGRA station files",
    )
    parser.add_argument(
        "--data-dir",
        default="/tmp/igra_local_zarr",
        help="Local zarr fallback (unused when writing to ArrayLake)",
    )
    parser.add_argument(
        "--branch", default="main", help="ArrayLake branch"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip the NCEI download step (use cached files)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print(f"Downloading IGRA {args.year} station files to {cache_dir}", flush=True)
        download_year(args.year, str(cache_dir))

    print(f"Connecting to ArrayLake repo zeus-ai/obs-xvec", flush=True)
    client = arraylake.Client()
    repo = client.get_repo("zeus-ai/obs-xvec")

    print(f"Processing year {args.year} → obs-xvec/igra/{args.year}", flush=True)
    year_to_xvec(
        year=args.year,
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        repo=repo,
        group="igra",
        branch=args.branch,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
