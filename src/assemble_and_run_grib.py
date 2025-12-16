"""Assemble GFS GRIB (pgrb2full 0.5°) cycles into NetCDF and run tracking+environment extraction.

Usage example:
  python3 src/assemble_and_run_grib.py \
    --urls output/gfs_grib_urls_00_12_f000_f240_step6.csv \
    --cycles 20230830T00 20230830T12 \
    --max-cycles 1 --processes 1 --keep-nc

Steps per cycle:
1) Download forecast hours f000–f240 (step6h) GRIB files.
2) Extract prmsl/10u/10v/gh(700hPa) and stack along time.
3) Save a temporary NetCDF under data/grib_nc/.
4) Invoke the existing tracking+environment pipeline on that NetCDF (auto mode).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

from environment_extractor.pipeline import process_nc_files
from shared.grib_loader import open_grib_collection

def _download_file(s3, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if not url.startswith("s3://"):
        raise ValueError(f"Unsupported URL: {url}")
    _, _, bucket_key = url.partition("s3://")
    bucket, _, key = bucket_key.partition("/")
    s3.download_file(bucket, key, str(dest))


def _assemble_cycle(
    urls: List[str],
    workdir: Path,
    direct: bool = True,
    chunk_lat: int = 181,
    chunk_lon: int = 360,
    write_nc: bool = True,
) -> Path:
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    cache_dir = workdir / "grib_cache"
    local_urls: list[str] = []
    init_dt: datetime | None = None
    valid_times: list[pd.Timestamp] = []

    for url in urls:
        fname = url.split("/")[-1]
        local = cache_dir / fname
        _download_file(s3, url, local)
        local_urls.append(str(local))
        if init_dt is None:
            segs = url.strip("/").split("/")
            for i, seg in enumerate(segs):
                if seg.startswith("gfs.") and len(seg) == 12:
                    day_part = seg.replace("gfs.", "")
                    cycle_part = segs[i + 1] if i + 1 < len(segs) else "00"
                    try:
                        init_dt = datetime.strptime(day_part + cycle_part, "%Y%m%d%H").replace(tzinfo=timezone.utc)
                    except Exception:
                        init_dt = None
                    break
        try:
            fhour = int(fname.split("f")[-1])
        except Exception:
            fhour = None
        if init_dt and fhour is not None:
            valid_times.append(pd.to_datetime(init_dt + timedelta(hours=fhour)))

    if not local_urls:
        raise RuntimeError("No data loaded for cycle")

    out_dir = workdir / "grib_nc"
    out_dir.mkdir(parents=True, exist_ok=True)
    init_tag = "unknown"
    if valid_times:
        init_tag = pd.to_datetime(valid_times[0]).strftime("%Y-%m-%d%H")
    elif init_dt is not None:
        init_tag = init_dt.strftime("%Y-%m-%d%H")

    if not write_nc:
        list_path = out_dir / f"gfs_{init_tag}_f000_f240_6h.griblist"
        with list_path.open("w", encoding="utf-8") as fh:
            for p in local_urls:
                fh.write(f"{p}\n")
        return list_path

    if not direct:
        raise RuntimeError("Legacy assemble path disabled; use --direct")

    ds = open_grib_collection(local_urls, chunk_lat=chunk_lat, chunk_lon=chunk_lon, valid_times=valid_times if valid_times else None)
    out_path = out_dir / f"gfs_{init_tag}_f000_f240_6h.nc"

    encoding = {
        "msl": {"zlib": True, "complevel": 1, "chunksizes": (1, chunk_lat, chunk_lon)},
        "10u": {"zlib": True, "complevel": 1, "chunksizes": (1, chunk_lat, chunk_lon)},
        "10v": {"zlib": True, "complevel": 1, "chunksizes": (1, chunk_lat, chunk_lon)},
        "z": {"zlib": True, "complevel": 1, "chunksizes": (1, chunk_lat, chunk_lon)},
    }
    ds.to_netcdf(out_path, engine="h5netcdf", encoding=encoding)
    return out_path


def group_urls_by_cycle(csv_path: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(csv_path)
    required = {"day", "cycle", "s3_url"}
    if not required.issubset(df.columns):
        raise ValueError(f"URL CSV 缺少必要列: {required - set(df.columns)}")
    grouped: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        key = f"{int(row['day'])}T{int(row['cycle']):02d}"
        grouped[key].append(row["s3_url"])
    for key in grouped:
        grouped[key] = sorted(grouped[key])
    return grouped


def parse_cycle_list(raw: List[str]) -> List[str]:
    clean = []
    for s in raw:
        s = s.strip().replace("-", "").replace(":", "")
        if s.endswith(("00", "12")):
            clean.append(s)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble GFS GRIB cycles and run tracking")
    parser.add_argument("--urls", default="output/gfs_grib_urls_00_12_f000_f240_step6.csv")
    parser.add_argument("--cycles", nargs="*", help="指定 cycle, 例如 20230830T00 20230830T12")
    parser.add_argument("--max-cycles", type=int, default=1, help="最多处理的 cycles 数")
    parser.add_argument("--processes", type=int, default=1, help="并行进程数")
    parser.add_argument("--keep-nc", action="store_true", help="处理后保留 NC")
    parser.add_argument("--initials", default="input/western_pacific_typhoons_superfast.csv")
    parser.add_argument("--no-direct", dest="direct", action="store_false", help="禁用 cfgrib 流式读取 (不推荐)")
    parser.set_defaults(direct=True)
    parser.add_argument("--chunk-lat", type=int, default=181, help="cfgrib 分块纬度")
    parser.add_argument("--chunk-lon", type=int, default=360, help="cfgrib 分块经度")
    parser.add_argument("--no-nc", action="store_true", help="跳过 NC 写入，仅输出 .griblist")
    args = parser.parse_args()

    all_groups = group_urls_by_cycle(Path(args.urls))
    cycles = parse_cycle_list(args.cycles) if args.cycles else sorted(all_groups.keys())
    if args.max_cycles:
        cycles = cycles[: args.max_cycles]
    if not cycles:
        raise SystemExit("No cycles to process")

    built_nc: List[Path] = []
    for cyc in cycles:
        urls = all_groups.get(cyc)
        if not urls:
            print(f"⚠️ 无URL: {cyc}")
            continue
        print(f"⬇️ 组装 cycle {cyc} ({len(urls)} files)...")
        out_path = _assemble_cycle(urls, Path("data"), direct=args.direct, chunk_lat=args.chunk_lat, chunk_lon=args.chunk_lon, write_nc=not args.no_nc)
        built_nc.append(out_path)
        label = "GRIB列表" if args.no_nc else "NC"
        print(f"✅ 已生成 {label}: {out_path}")

    class _Args:
        def __init__(self, processes: int, keep_nc: bool, initials: str):
            self.processes = processes
            self.no_clean = keep_nc
            self.keep_nc = keep_nc
            self.auto = True
            self.initials = initials
            self.tracks = None

    run_args = _Args(args.processes, args.keep_nc, args.initials)
    if built_nc:
        process_nc_files(built_nc, run_args, concise_log=False, logs_root=Path("final_single_output/logs"))
    else:
        print("未生成任何 NC, 结束")


if __name__ == "__main__":
    main()
