"""Process GFS GRIB by individual TC lifetimes.

For each TC in the catalogue, find all 00Z/12Z forecast cycles that overlap
its active period, assemble those GRIBs into NC, then run tracking+extraction.

Usage:
  python src/process_by_tc_lifetime.py \
    --grib-urls output/gfs_grib_urls_00_12_f000_f240_step6.csv \
    --tc-tracks input/matched_cyclone_tracks.csv \
    --initials input/western_pacific_typhoons_superfast.csv \
    --processes 1
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter, sleep
from typing import Dict, List

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

from environment_extractor.pipeline import process_nc_files
from shared.grib_loader import open_grib_collection


def _download_file(s3, url: str, dest: Path, max_attempts: int = 3, backoff: float = 1.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if not url.startswith("s3://"):
        raise ValueError(f"Unsupported URL: {url}")
    _, _, bucket_key = url.partition("s3://")
    bucket, _, key = bucket_key.partition("/")
    for attempt in range(1, max_attempts + 1):
        try:
            s3.download_file(bucket, key, str(dest))
            return
        except Exception:
            if attempt >= max_attempts:
                raise
            sleep(backoff * attempt)


def load_grib_urls(csv_path: Path) -> pd.DataFrame:
    """Load GRIB URL CSV and parse init time + forecast hour."""
    df = pd.read_csv(csv_path)
    parsed = []
    for _, r in df.iterrows():
        try:
            day = int(r["day"])
            cycle = int(r["cycle"])
            url = r["s3_url"]
            fname = url.split("/")[-1]
            fhour = int(fname.split("f")[-1])
            init_dt = datetime.strptime(f"{day:08d}{cycle:02d}", "%Y%m%d%H").replace(tzinfo=timezone.utc)
            valid_dt = init_dt + timedelta(hours=fhour)
            parsed.append(
                {
                    "init_time": init_dt,
                    "fhour": fhour,
                    "valid_time": valid_dt,
                    "url": url,
                    "fname": fname,
                }
            )
        except Exception:
            continue
    return pd.DataFrame(parsed)


def load_tc_lifetimes(csv_path: Path) -> List[Dict]:
    """Load TC tracks and extract unique TC lifetimes."""
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    
    tcs = {}
    for _, row in df.iterrows():
        tc_id = row["storm_id"]
        dt = row["datetime"]
        if tc_id not in tcs:
            tcs[tc_id] = {"min_time": dt, "max_time": dt, "name": row.get("storm_name", tc_id)}
        else:
            tcs[tc_id]["min_time"] = min(tcs[tc_id]["min_time"], dt)
            tcs[tc_id]["max_time"] = max(tcs[tc_id]["max_time"], dt)
    
    result = []
    for tc_id, info in tcs.items():
        result.append({
            "storm_id": tc_id,
            "name": info["name"],
            "start": info["min_time"],
            "end": info["max_time"],
        })
    return result


def group_gribs_by_cycle(df: pd.DataFrame) -> Dict[datetime, pd.DataFrame]:
    grouped: Dict[datetime, pd.DataFrame] = {}
    for init_time, sub in df.groupby("init_time"):
        grouped[init_time] = sub.sort_values("fhour")
    return grouped


def select_cycles_for_tc(tc_start: datetime, tc_end: datetime, cycles: Dict[datetime, pd.DataFrame]) -> List[datetime]:
    """Select cycles with init_time (00/12Z) falling inside the TC lifetime."""
    selected = []
    for init_time in cycles.keys():
        if tc_start <= init_time <= tc_end:
            selected.append(init_time)
    return sorted(selected)


def assemble_and_track_cycle(
    tc_id: str,
    tc_name: str,
    init_time: datetime,
    grib_cycle_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    workdir: Path,
    initials_path: Path,
    processes: int = 1,
    keep_nc: bool = False,
    download_workers: int = 4,
    max_download_attempts: int = 3,
    s3_connect_timeout: int = 10,
    s3_read_timeout: int = 60,
) -> tuple[int, int]:
    """Assemble a single forecast cycle for one TC, then track/extract."""
    print(f"   ▶️ cycle {init_time.strftime('%Y-%m-%d %H:%M')}Z, 预报步数={len(grib_cycle_df)}")

    overall_start = perf_counter()

    # Download needed GRIBs for this cycle
    s3 = boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            connect_timeout=s3_connect_timeout,
            read_timeout=s3_read_timeout,
            retries={"max_attempts": max_download_attempts, "mode": "standard"},
        ),
    )
    cache_dir = workdir / "grib_cache"
    local_urls: List[str] = []
    valid_times: List[pd.Timestamp] = []

    download_start = perf_counter()
    download_workers = max(1, int(download_workers))
    futures = {}
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        for _, grib in grib_cycle_df.iterrows():
            local = cache_dir / grib["fname"]
            fut = pool.submit(
                _download_file,
                s3,
                grib["url"],
                local,
                max_download_attempts,
            )
            futures[fut] = (local, pd.to_datetime(grib["valid_time"]), grib["fname"])

        for fut in as_completed(futures):
            local, vtime, fname = futures[fut]
            try:
                fut.result()
                local_urls.append(str(local))
                valid_times.append(vtime)
            except Exception as e:
                print(f"      ⚠️  下载失败 {fname}: {e}")

    download_elapsed = perf_counter() - download_start
    print(f"      ⏱️ 下载耗时 {download_elapsed:.1f}s, 成功 {len(local_urls)}/{len(grib_cycle_df)}")

    if not local_urls:
        print("      ❌ 无可用预报文件，跳过该 cycle")
        return 0, 1

    out_dir = workdir / "grib_nc"
    out_dir.mkdir(parents=True, exist_ok=True)
    init_tag = init_time.strftime("%Y%m%dT%H")
    nc_name = f"gfs_{tc_id}_{init_tag}.nc"
    nc_path = out_dir / nc_name
    track_dir = Path("track_single")
    track_dir.mkdir(exist_ok=True)
    track_path = track_dir / f"track_{tc_id}_{nc_name.replace('.nc', '')}.csv"

    try:
        nc_start = perf_counter()
        ds = open_grib_collection(local_urls, valid_times=valid_times)
        encoding = {
            "msl": {"zlib": True, "complevel": 1, "chunksizes": (1, 181, 360)},
            "10u": {"zlib": True, "complevel": 1, "chunksizes": (1, 181, 360)},
            "10v": {"zlib": True, "complevel": 1, "chunksizes": (1, 181, 360)},
            "z": {"zlib": True, "complevel": 1, "chunksizes": (1, 181, 360)},
        }
        ds.to_netcdf(nc_path, engine="h5netcdf", encoding=encoding)
        ds.close()
        nc_elapsed = perf_counter() - nc_start
        print(f"      ✅ NC 已生成: {nc_name} (耗时 {nc_elapsed:.1f}s)")
    except Exception as e:
        print(f"      ❌ NC 合成失败: {e}")
        return 0, 1

    # 基于 matched_cyclone_tracks_2021onwards.csv 为当前 NC 构建轨迹文件
    try:
        tc_tracks = tracks_df[tracks_df["storm_id"] == tc_id].copy()
        tc_tracks["datetime"] = pd.to_datetime(tc_tracks["datetime"], utc=True, errors="coerce")
        rows = []
        for idx, vt in enumerate(valid_times):
            if tc_tracks.empty:
                break
            diffs = (tc_tracks["datetime"] - vt).abs()
            nearest_idx = diffs.idxmin()
            nearest = tc_tracks.loc[nearest_idx]
            rows.append(
                {
                    "time": vt,
                    "lat": nearest["latitude"],
                    "lon": nearest["longitude"],
                    "msl": nearest.get("min_pressure_wmo", None),
                    "wind": nearest.get("max_wind_wmo", None),
                    "particle": tc_id,
                    "time_idx": idx,
                }
            )

        if not rows:
            print("      ⚠️ 无匹配轨迹点 -> 跳过该 cycle")
            if not keep_nc:
                try:
                    nc_path.unlink()
                except Exception:
                    pass
            return 0, 1

        pd.DataFrame(rows).to_csv(track_path, index=False)
        print(f"      💾 轨迹文件: {track_path.name} (基于 matched_cyclone_tracks_2021onwards)")
    except Exception as e:
        print(f"      ❌ 生成轨迹失败: {e}")
        if not keep_nc:
            try:
                nc_path.unlink()
            except Exception:
                pass
        return 0, 1

    class _Args:
        def __init__(self, processes: int, keep_nc: bool, initials_path: Path):
            self.processes = processes
            self.no_clean = keep_nc
            self.keep_nc = keep_nc
            self.auto = True
            self.initials = str(initials_path)
            self.tracks = str(track_path)

    try:
        run_args = _Args(processes, keep_nc, initials_path)
        track_start = perf_counter()
        process_nc_files([nc_path], run_args, concise_log=True, logs_root=Path("final_single_output/logs"))
        track_elapsed = perf_counter() - track_start
        print(f"      ✅ 追踪和环境提取完成 (耗时 {track_elapsed:.1f}s)")
    except Exception as e:
        print(f"      ❌ 追踪失败: {e}")
        if not keep_nc:
            try:
                nc_path.unlink()
            except Exception:
                pass
        return 0, 1

    if not keep_nc:
        try:
            nc_path.unlink()
            print(f"      🧹 已删除 NC 以节省空间")
        except Exception:
            pass

    overall_elapsed = perf_counter() - overall_start
    print(f"      ⏱️ 本 cycle 总耗时 {overall_elapsed:.1f}s")

    return 1, 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Process GFS GRIB by TC lifetime")
    parser.add_argument("--grib-urls", default="output/gfs_grib_urls_00_12_f000_f240_step6.csv")
    parser.add_argument("--tc-tracks", default="input/matched_cyclone_tracks.csv")
    parser.add_argument("--initials", default="input/western_pacific_typhoons_superfast.csv")
    parser.add_argument("--max-tcs", type=int, default=None, help="最多处理的 TC 数")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--keep-nc", action="store_true")
    parser.add_argument("--download-workers", type=int, default=4, help="下载单个 cycle 时的并发下载线程数")
    parser.add_argument("--max-download-attempts", type=int, default=3, help="单个 GRIB 的最大下载重试次数")
    parser.add_argument("--s3-connect-timeout", type=int, default=10, help="S3 连接超时 (秒)")
    parser.add_argument("--s3-read-timeout", type=int, default=60, help="S3 读取超时 (秒)")
    args = parser.parse_args()
    
    print("📥 加载 GRIB URLs...")
    grib_df = load_grib_urls(Path(args.grib_urls))
    print(f"   找到 {len(grib_df)} 个预报时次")
    cycles = group_gribs_by_cycle(grib_df)
    print(f"   可用预报周期: {len(cycles)} (00/12Z)")
    
    print("📥 加载 TC 轨迹...")
    tracks_df = pd.read_csv(Path(args.tc_tracks))
    tc_lifetimes = load_tc_lifetimes(Path(args.tc_tracks))
    print(f"   找到 {len(tc_lifetimes)} 条台风")
    
    min_cycle = min(cycles.keys()) if cycles else None
    max_cycle = max(cycles.keys()) if cycles else None

    # 仅保留在可用预报时间范围内的台风
    if min_cycle and max_cycle:
        tc_lifetimes = [tc for tc in tc_lifetimes if tc["end"] >= min_cycle and tc["start"] <= max_cycle]

    if args.max_tcs:
        tc_lifetimes = tc_lifetimes[: args.max_tcs]

    processed = 0
    skipped = 0
    workdir = Path("data")
    initials_path = Path(args.initials)
    
    for tc in tc_lifetimes:
        tc_cycles = select_cycles_for_tc(tc["start"], tc["end"], cycles)
        if not tc_cycles:
            print(f"⏭️  [{tc['storm_id']}] 无 00/12Z 预报周期覆盖 -> 跳过")
            skipped += 1
            continue

        print(
            f"\n🌀 处理台风 [{tc['storm_id']}] {tc['name']} "
            f"({tc['start'].strftime('%Y-%m-%d')} - {tc['end'].strftime('%Y-%m-%d')}), 周期数 {len(tc_cycles)}"
        )
        for init_time in tc_cycles:
            cycle_df = cycles[init_time]
            p, s = assemble_and_track_cycle(
                tc["storm_id"],
                tc["name"],
                init_time,
                cycle_df,
                tracks_df,
                workdir,
                initials_path,
                processes=args.processes,
                keep_nc=args.keep_nc,
                download_workers=args.download_workers,
                max_download_attempts=args.max_download_attempts,
                s3_connect_timeout=args.s3_connect_timeout,
                s3_read_timeout=args.s3_read_timeout,
            )
            processed += p
            skipped += s
    
    print(f"\n{'='*60}")
    print(f"📊 处理完成:")
    print(f"   ✅ 已处理: {processed}")
    print(f"   ⏭️  已跳过: {skipped}")
    print(f"   📁 输出目录: final_single_output")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
