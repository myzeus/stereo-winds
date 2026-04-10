"""Build a list of GOES timestamps needed for sonde-supervised training.

For each unique sonde launch time in an IGRA collocation parquet, emits the
3 GOES scans needed for the 5-scene stereo pipeline: t-10min, t, t+10min.

Usage:
    python scripts/build_sonde_times_file.py \\
        --parquets data/igra/igra_2025_collocation.parquet \\
                   data/igra/igra_2026_collocation.parquet \\
        --output data/igra/sonde_times.txt
"""

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquets", nargs="+", required=True,
                        help="Paths to IGRA collocation parquet files")
    parser.add_argument("--output", required=True,
                        help="Output text file path (one timestamp per line)")
    parser.add_argument("--dt-minutes", type=float, default=10.0,
                        help="Stereo time offset in minutes (default 10)")
    args = parser.parse_args()

    all_times = set()
    for p in args.parquets:
        df = pd.read_parquet(p)
        for t in df["goes_time"].unique():
            t0 = pd.Timestamp(t).to_pydatetime()
            for k in (-1, 0, 1):
                all_times.add(t0 + dt.timedelta(minutes=args.dt_minutes * k))

    sorted_times = sorted(all_times)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for t in sorted_times:
            f.write(t.strftime("%Y-%m-%dT%H:%M") + "\n")

    print(f"Wrote {len(sorted_times)} unique timestamps to {output}")
    print(f"  Date range: {sorted_times[0]} to {sorted_times[-1]}")


if __name__ == "__main__":
    main()
