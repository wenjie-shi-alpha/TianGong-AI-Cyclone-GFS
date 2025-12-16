"""Generate GFS GRIB2 (pgrb2b 0.25°) URLs for 00/12 UTC cycles.

Reads the cyclone catalogue to determine the date range, then emits URLs for
the operational GFS archive on S3 for forecast hours 0–240 (3h step) for each
day's 00Z and 12Z cycles.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd

BUCKET = "noaa-gfs-bdp-pds"
# 使用 pgrb2full 0.5° 版本以获取 prmsl/10m 风/位势高度全量变量
MODEL_PREFIX = "gfs_pgrb2full_0p50"


def _load_unique_days(csv_path: Path) -> List[datetime]:
    df = pd.read_csv(csv_path, usecols=["datetime"], parse_dates=["datetime"])
    df["datetime"] = df["datetime"].dt.floor("h")
    unique_days = sorted({dt.date() for dt in df["datetime"] if pd.notnull(dt)})
    return [datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc) for d in unique_days]


def _iter_cycles(days: Iterable[datetime], cycles: Iterable[int]) -> Iterable[datetime]:
    for day in days:
        for cycle in cycles:
            yield day + timedelta(hours=cycle)


def _forecast_hours(max_hour: int, step: int) -> List[int]:
    return list(range(0, max_hour + 1, step))


def _build_url(init_dt: datetime, fhour: int) -> str:
    day_key = init_dt.strftime("%Y%m%d")
    cycle = init_dt.strftime("%H")
    filename = f"gfs.t{cycle}z.pgrb2full.0p50.f{fhour:03d}"
    return f"s3://{BUCKET}/gfs.{day_key}/{cycle}/atmos/{filename}"


def generate_urls(
    track_csv: Path,
    output_csv: Path,
    max_hour: int = 240,
    step: int = 6,
    cycles: Iterable[int] = (0, 12),
) -> None:
    days = _load_unique_days(track_csv)
    if not days:
        raise SystemExit(f"No dates found in {track_csv}")

    rows = []
    hours = _forecast_hours(max_hour, step)
    for init_time in _iter_cycles(days, cycles):
        for fh in hours:
            rows.append(
                {
                    "day": init_time.strftime("%Y%m%d"),
                    "cycle": init_time.strftime("%H"),
                    "model_prefix": MODEL_PREFIX,
                    "init_time": init_time.isoformat(),
                    "forecast_hour": fh,
                    "s3_url": _build_url(init_time, fh),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["day", "cycle", "model_prefix", "init_time", "forecast_hour", "s3_url"]
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 00/12Z GFS GRIB2 URL list (pgrb2b 0.25°, f000–f240, 6h step)",
    )
    parser.add_argument(
        "--input",
        default="input/matched_cyclone_tracks_2021onwards.csv",
        help="Cyclone catalogue with 'datetime' column",
    )
    parser.add_argument(
        "--output",
        default="output/gfs_grib_urls_00_12_f000_f240.csv",
        help="Output CSV path",
    )
    parser.add_argument("--max-hour", type=int, default=240, help="Last forecast hour to include")
    parser.add_argument("--step", type=int, default=6, help="Forecast hour step")
    args = parser.parse_args()

    generate_urls(Path(args.input), Path(args.output), max_hour=args.max_hour, step=args.step)


if __name__ == "__main__":
    main()
