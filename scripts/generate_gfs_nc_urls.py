#!/usr/bin/env python3
"""Generate GFS netCDF URL list for the matched cyclone track date span.

The script reads `input/matched_cyclone_tracks.csv` to discover the date range,
clamps it to the GFS bucket coverage, and writes a CSV in the same schema as
`output/nc_file_urls_new.csv` containing the analysis file (`atmanl.nc`) and the
0-hour forecast file (`atmf000.nc`) for each 6-hour cycle.

Columns: day, model_prefix, key, s3_url, size_bytes, last_modified, init_time.
"""
import csv
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple

import boto3
from botocore import UNSIGNED
from botocore.client import Config

ROOT = Path(__file__).resolve().parents[1]
MATCHED = ROOT / "input" / "matched_cyclone_tracks.csv"
OUTPUT = ROOT / "output" / "gfs_nc_file_urls.csv"
BUCKET = "noaa-gfs-bdp-pds"
MODEL_PREFIX = "GFS_BDP"
CYCLE_HOURS = (0, 6, 12, 18)
BUCKET_START = dt.date(2021, 1, 1)  # earliest observed prefix in bucket


def _clean_datetime(value: str) -> dt.datetime:
    """Parse datetimes with overly long fractional seconds."""
    raw = value.strip()
    if "." in raw:
        base, frac = raw.split(".", 1)
        frac = (frac + "000000")[:6]
        raw = f"{base}.{frac}"
        return dt.datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f")
    return dt.datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")


def compute_date_range(path: Path) -> Tuple[dt.date, dt.date]:
    """Return the min/max dates present in the matched tracks file."""
    min_dt = None
    max_dt = None
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = _clean_datetime(row["datetime"]).date()
            if min_dt is None or parsed < min_dt:
                min_dt = parsed
            if max_dt is None or parsed > max_dt:
                max_dt = parsed
    if min_dt is None or max_dt is None:
        raise RuntimeError("No datetime values found in matched_cyclone_tracks.csv")
    return min_dt, max_dt


def build_targets(start: dt.date, end: dt.date) -> Iterable[Tuple[str, str, str]]:
    """Yield (day_str, key, init_time_iso) for each cycle in the range."""
    day = start
    while day <= end:
        day_str = day.strftime("%Y%m%d")
        for hour in CYCLE_HOURS:
            hh = f"{hour:02d}"
            init_time = f"{day.isoformat()}T{hh}:00:00"
            # analysis and 0-hour forecast
            yield day_str, f"gfs.{day_str}/{hh}/atmos/gfs.t{hh}z.atmanl.nc", init_time
            yield day_str, f"gfs.{day_str}/{hh}/atmos/gfs.t{hh}z.atmf000.nc", init_time
        day += dt.timedelta(days=1)


def main() -> None:
    start_dt, end_dt = compute_date_range(MATCHED)
    # clamp to bucket coverage
    if start_dt < BUCKET_START:
        start_dt = BUCKET_START
    today = dt.date.today()
    if end_dt > today:
        end_dt = today

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    targets = list(build_targets(start_dt, end_dt))
    results = []
    missing = []

    print(f"Scanning {len(targets)} keys from {start_dt} to {end_dt}...")

    def fetch(meta):
        day_str, key, init_time = meta
        try:
            obj = s3.head_object(Bucket=BUCKET, Key=key)
        except Exception as exc:  # noqa: BLE001
            return None, (day_str, key, init_time, str(exc))
        return {
            "day": day_str,
            "model_prefix": MODEL_PREFIX,
            "key": key,
            "s3_url": f"s3://{BUCKET}/{key}",
            "size_bytes": obj["ContentLength"],
            "last_modified": obj["LastModified"].isoformat(),
            "init_time": init_time,
        }, None

    with ThreadPoolExecutor(max_workers=24) as executor:
        future_to_target = {executor.submit(fetch, meta): meta for meta in targets}
        for idx, future in enumerate(as_completed(future_to_target), 1):
            data, miss = future.result()
            if data:
                results.append(data)
            elif miss:
                missing.append(miss)
            if idx % 500 == 0:
                print(f"Processed {idx}/{len(targets)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("day", "model_prefix", "key", "s3_url", "size_bytes", "last_modified", "init_time"),
        )
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: (r["day"], r["key"])))

    if missing:
        miss_path = OUTPUT.with_name(OUTPUT.stem + "_missing.log")
        with miss_path.open("w") as f:
            for day_str, key, init_time, err in missing:
                f.write(f"{day_str},{key},{init_time},{err}\n")
        print(f"{len(missing)} keys missing; see {miss_path}")
    print(f"Wrote {len(results)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
