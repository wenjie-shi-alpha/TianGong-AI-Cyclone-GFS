"""台风驱动的 GFS GRIB 下载+分析流水线。

核心逻辑（正确顺序）:
  1. 先读台风CSV → 知道哪些时间段有台风
  2. 根据台风时间段自动计算需要哪些GFS预报cycle
  3. 动态生成S3 URL（不需要预生成的URL表）
  4. 只下载有台风的cycle数据
  5. 追踪 + 环境提取

用法:
  # 全量运行（自动从台风数据确定所有需要的GFS cycle）
  nohup python3 -u src/assemble_and_run_grib.py \\
    --tracks input/matched_cyclone_tracks.csv \\
    > run_fullscale.log 2>&1 &

  # 只处理指定台风
  python3 src/assemble_and_run_grib.py \\
    --tracks input/matched_cyclone_tracks.csv \\
    --storms 2025067S12085 2021093S10103

  # 指定cycle（手动模式）
  python3 src/assemble_and_run_grib.py \\
    --tracks input/matched_cyclone_tracks.csv \\
    --cycles 20250308T00 20250308T12

Pipeline架构（批次模式）:
  台风CSV → 计算需要的cycles → 分批处理:
    每批:
      A) 并行下载GRIB + cfgrib解析 → NC
      B) 追踪 + 环境提取
      C) 清理NC
    → 下一批

All defaults auto-tuned to CPU count (os.cpu_count()).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Set, Tuple

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

from environment_extractor.pipeline import process_nc_files
from shared.grib_loader import open_grib_collection, open_grib_collection_fast

import os as _os

_N_CPU = _os.cpu_count() or 4  # 自动检测 CPU 核数

# S3数据最早可用日期 (noaa-gfs-bdp-pds bucket)
_S3_DATA_START = pd.Timestamp("2021-03-23")

# GFS预报的时间步长和范围
_FORECAST_HOURS = list(range(0, 241, 6))  # f000, f006, ..., f240
_CYCLES_PER_DAY = [0, 12]  # 00Z 和 12Z

# Global print lock for clean multi-cycle output
_print_lock = Lock()
# netCDF4/HDF5 is NOT thread-safe — concurrent writes/reads from multiple
# threads deadlock inside the C library.  Serialize all NC write operations.
_nc_write_lock = Lock()


def _tprint(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 第一步: 台风数据驱动 — 从台风CSV确定需要哪些GFS cycle
# ─────────────────────────────────────────────────────────────────────────────

def compute_needed_cycles(
    tracks_csv: str | Path,
    lead_days: int = 10,
    storm_filter: list[str] | None = None,
) -> Tuple[list[datetime], Dict[str, list[str]]]:
    """从台风CSV读取所有台风的时间范围，计算需要的GFS forecast cycle。

    逻辑:
      - 每个台风有 start_time 和 end_time
      - 一个cycle的init_time如果在 [start - lead_days, end] 范围内，
        就说明该cycle的f000-f240预报期可能覆盖到这个台风
      - 只生成2021-03-23之后的cycle（S3上有数据的时间段）

    Parameters
    ----------
    tracks_csv : 台风轨迹CSV路径
    lead_days : 最早允许的提前天数 (默认10天 = f240)
    storm_filter : 只处理指定的storm_id列表 (None=全部)

    Returns
    -------
    cycles : 时间排序的cycle datetime列表
    cycle_storms : {cycle_tag: [storm_id, ...]} 每个cycle关联的台风ID
    """
    df = pd.read_csv(tracks_csv)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 检测列名
    storm_col = "storm_id" if "storm_id" in df.columns else "SID"

    # 可选: 只处理指定的台风
    if storm_filter:
        df = df[df[storm_col].isin(storm_filter)]
        if df.empty:
            raise SystemExit(f"❌ 指定的storm_id在CSV中找不到: {storm_filter}")

    # 计算每个台风的时间范围
    storm_ranges = df.groupby(storm_col)["datetime"].agg(["min", "max"])
    print(f"📊 台风CSV: {len(storm_ranges)} 个台风, "
          f"时间范围 {storm_ranges['min'].min().strftime('%Y-%m-%d')} → "
          f"{storm_ranges['max'].max().strftime('%Y-%m-%d')}")

    # 计算每个cycle对应哪些台风
    cycle_storms: Dict[str, list[str]] = defaultdict(list)

    for storm_id, row in storm_ranges.iterrows():
        s_start = pd.Timestamp(row["min"])
        s_end = pd.Timestamp(row["max"])

        # cycle init_time 范围: [storm_start - lead_days, storm_end]
        earliest_init = s_start - timedelta(days=lead_days)
        latest_init = s_end

        # 不早于S3数据起始日期
        if latest_init < _S3_DATA_START:
            continue  # 这个台风完全在S3数据之前，跳过
        earliest_init = max(earliest_init, _S3_DATA_START)

        # 生成所有 00Z 和 12Z 的cycle
        cur_day = earliest_init.normalize()  # 当天0点
        while cur_day <= latest_init:
            for hour in _CYCLES_PER_DAY:
                cyc_dt = cur_day + timedelta(hours=hour)
                if earliest_init <= cyc_dt <= latest_init:
                    tag = cyc_dt.strftime("%Y%m%dT%H")
                    cycle_storms[tag].append(str(storm_id))
            cur_day += timedelta(days=1)

    # 排序
    sorted_tags = sorted(cycle_storms.keys())
    sorted_dts = [datetime.strptime(t, "%Y%m%dT%H").replace(tzinfo=timezone.utc)
                  for t in sorted_tags]

    print(f"🎯 需要下载的GFS cycles: {len(sorted_dts)} 个")
    if sorted_dts:
        print(f"   时间范围: {sorted_tags[0]} → {sorted_tags[-1]}")
        print(f"   GRIB文件数: {len(sorted_dts) * len(_FORECAST_HOURS)} 个")

    return sorted_dts, dict(cycle_storms)


def generate_cycle_urls(cycle_dt: datetime) -> List[str]:
    """根据cycle的init时间动态生成S3 URL列表。

    URL模式: s3://noaa-gfs-bdp-pds/gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2full.0p50.fXXX
    """
    date_str = cycle_dt.strftime("%Y%m%d")
    hour_str = f"{cycle_dt.hour:02d}"
    urls = []
    for fh in _FORECAST_HOURS:
        url = (f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/{hour_str}/atmos/"
               f"gfs.t{hour_str}z.pgrb2full.0p50.f{fh:03d}")
        urls.append(url)
    return urls


# ─────────────────────────────────────────────────────────────────────────────
# 第二步: 下载和解析
# ─────────────────────────────────────────────────────────────────────────────

def _download_file(s3, url: str, dest: Path) -> bool:
    """Download a single S3 file. Returns True on success, False on 404/missing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return True
    if not url.startswith("s3://"):
        raise ValueError(f"Unsupported URL: {url}")
    _, _, bucket_key = url.partition("s3://")
    bucket, _, key = bucket_key.partition("/")
    try:
        s3.download_file(bucket, key, str(dest))
        return True
    except Exception as exc:
        err_str = str(exc)
        # Handle 404 / NoSuchKey gracefully
        if hasattr(exc, "response"):
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey"):
                return False
        if "404" in err_str or "Not Found" in err_str or "NoSuchKey" in err_str:
            return False
        raise


def _url_to_local(url: str, cache_dir: Path) -> Path:
    """Compute the local cache path for a given S3 URL (date/cycle/filename)."""
    fname = url.split("/")[-1]
    segs = url.strip("/").split("/")
    date_seg = next((s for s in segs if s.startswith("gfs.") and len(s) == 12), None)
    cyc_seg = segs[segs.index(date_seg) + 1] if date_seg and date_seg in segs else "00"
    return cache_dir / (date_seg or "unknown") / cyc_seg / fname


def _wait_for_shm_space(cache_dir: Path, max_gb: float = 70.0) -> None:
    """Block until /dev/shm has enough free space. Prevents tmpfs OOM."""
    import shutil as _shutil, time as _time
    shm = Path("/dev/shm")
    if not shm.is_dir():
        return
    # Only gate on /dev/shm usage when cache is actually on tmpfs.
    # If cache_dir is on disk (e.g. data/grib_cache), waiting on shm is unnecessary.
    try:
        cache_dir.resolve().relative_to(shm.resolve())
    except Exception:
        return
    while True:
        used_gb = _shutil.disk_usage(str(shm)).used / 1e9
        if used_gb < max_gb:
            return
        _tprint(f"⏸️  /dev/shm 已用 {used_gb:.1f}GB ≥ {max_gb}GB 上限，等待清理...")
        _time.sleep(15)


def _download_files_parallel(
    urls: List[str],
    cache_dir: Path,
    n_threads: int = 8,
) -> List[str]:
    """Download a list of S3 URLs in parallel using ThreadPoolExecutor.
    Returns list of successfully downloaded local paths."""
    # 下载前先检查 /dev/shm 空间，超过 70GB 则等待
    _wait_for_shm_space(cache_dir)
    local_paths: list[str | None] = [None] * len(urls)
    n_skipped = 0

    def _worker(idx_url: tuple[int, str]) -> tuple[int, str | None]:
        idx, url = idx_url
        local = _url_to_local(url, cache_dir)
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=50))
        ok = _download_file(s3, url, local)
        return idx, str(local) if ok else None

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {ex.submit(_worker, (i, url)): i for i, url in enumerate(urls)}
        for fut in as_completed(futures):
            idx, local = fut.result()
            if local is None:
                n_skipped += 1
            else:
                local_paths[idx] = local

    # Filter out failed downloads
    result = [p for p in local_paths if p is not None]
    if n_skipped:
        _tprint(f"⚠️ {n_skipped}/{len(urls)} 个文件下载失败 (404/不存在), "
                f"成功 {len(result)}/{len(urls)}")
    return result


def _assemble_cycle(
    cycle_dt: datetime,
    urls: List[str],
    workdir: Path,
    download_threads: int = 8,
    parse_workers: int = 8,
    grib_dir: Path | None = None,
    keep_grib: bool = False,
    executor=None,
) -> Path:
    """Download + parse one GFS forecast cycle into a NetCDF file.

    Parameters
    ----------
    cycle_dt : init datetime of this cycle
    urls : list of S3 URLs for this cycle
    download_threads : parallel S3 download threads
    parse_workers : ProcessPoolExecutor workers for cfgrib
    grib_dir : GRIB cache root (default: /dev/shm/grib_cache or workdir/grib_cache)
    keep_grib : keep GRIB files after NC write
    executor : shared ProcessPoolExecutor for cfgrib parsing
    """
    cache_dir = grib_dir if grib_dir is not None else workdir / "grib_cache"

    init_tag = cycle_dt.strftime("%Y-%m-%d%H")
    out_dir = workdir / "grib_nc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gfs_{init_tag}_f000_f240_6h.nc"

    # Check if NC already exists and is valid
    if out_path.exists():
        try:
            sz = out_path.stat().st_size
            if sz > 10_000_000:
                with _nc_write_lock:
                    import netCDF4 as _nc4
                    with _nc4.Dataset(str(out_path)) as _ds:
                        _vars = set(_ds.variables.keys())
                        dims_map = {
                            name: tuple(_ds.variables[name].dimensions)
                            for name in ("z", "u", "v", "t")
                            if name in _ds.variables
                        }
                required = {"msl", "10u", "10v", "z", "u", "v", "t"}
                has_required = required.issubset(_vars)
                has_levels = False
                if has_required:
                    level_dim = "isobaricInhPa"
                    z_dims = dims_map.get("z", ())
                    u_dims = dims_map.get("u", ())
                    v_dims = dims_map.get("v", ())
                    t_dims = dims_map.get("t", ())
                    has_levels = all(level_dim in dims for dims in (z_dims, u_dims, v_dims, t_dims))
                if has_required and has_levels:
                    _tprint(f"⏭️ NC 已存在且完整，跳过: {out_path.name} ({sz/1e6:.0f}MB)")
                    return out_path
            _tprint(f"⚠️ NC 存在但不完整 ({sz} bytes)，将重建: {out_path.name}")
            out_path.unlink(missing_ok=True)
        except Exception as exc:
            _tprint(f"⚠️ NC 校验失败 ({exc})，将重建: {out_path.name}")
            out_path.unlink(missing_ok=True)

    # ── Step 1: Parallel S3 download ──────────────────────────────────────────
    local_urls = _download_files_parallel(urls, cache_dir, n_threads=download_threads)
    if not local_urls:
        raise RuntimeError("No data downloaded for cycle (all 404?)")
    if len(local_urls) < 5:
        raise RuntimeError(
            f"Only {len(local_urls)}/{len(urls)} GRIB files downloaded — "
            f"too few for valid NC, skipping cycle"
        )

    # Compute valid_times from actually downloaded files
    valid_times = []
    for p in local_urls:
        fname = Path(p).name
        try:
            fh = int(fname.split("f")[-1])
        except (ValueError, IndexError):
            continue
        valid_times.append(pd.to_datetime(cycle_dt + timedelta(hours=fh)))

    # ── Step 2: Fast per-file parallel cfgrib extraction ──────────────────────
    ds = open_grib_collection_fast(
        local_urls,
        n_workers=parse_workers,
        valid_times=valid_times if valid_times else None,
        executor=executor,
    )

    # ── Step 3: Write NetCDF ──────────────────────────────────────────────────
    lat_dim, lon_dim = ds.sizes["latitude"], ds.sizes["longitude"]
    encoding = {}
    for var_name in ds.data_vars:
        data_var = ds[var_name]
        var_encoding = {"zlib": True, "complevel": 1}
        if data_var.ndim == 3:
            var_encoding["chunksizes"] = (1, lat_dim, lon_dim)
        elif data_var.ndim == 4:
            var_encoding["chunksizes"] = (1, 1, lat_dim, lon_dim)
        encoding[var_name] = var_encoding
    _tprint(f"🔒 等待 NC 写入锁 ({out_path.name})...")
    with _nc_write_lock:
        ds.to_netcdf(out_path, engine="netcdf4", encoding=encoding)
    ds.close()
    _tprint(f"📝 NC 已写入: {out_path.name} ({out_path.stat().st_size/1e6:.0f}MB)")

    # ── Step 4: Delete GRIB files to free disk/RAM ────────────────────────────
    if not keep_grib and local_urls:
        import shutil as _shutil
        cyc_dir = Path(local_urls[0]).parent
        try:
            _shutil.rmtree(cyc_dir, ignore_errors=True)
            _tprint(f"🧹 已删除 GRIB 目录: {cyc_dir}")
        except Exception as exc:
            _tprint(f"⚠️ 删除 GRIB 目录失败: {exc}")

    return out_path


def _assemble_only(
    cycle_dt: datetime,
    cyc_tag: str,
    urls: List[str],
    args,
    executor=None,
) -> Path | None:
    """Phase A: Download + cfgrib parse → write NetCDF. Thread-safe."""
    import time as _time
    t0 = _time.time()
    n_files = len(urls)
    _tprint(f"⬇️  [{cyc_tag}] 开始下载 {n_files} 个 GRIB 文件...")
    try:
        out_path = _assemble_cycle(
            cycle_dt,
            urls,
            Path("data"),
            download_threads=args.download_threads,
            parse_workers=args.parse_workers,
            grib_dir=Path(args.grib_dir) if getattr(args, "grib_dir", None) else None,
            keep_grib=getattr(args, "keep_grib", False),
            executor=executor,
        )
        elapsed = _time.time() - t0
        _tprint(f"✅ [{cyc_tag}] NC 就绪 ({elapsed:.0f}s): {out_path.name}")
        return out_path
    except Exception as exc:
        import traceback
        _tprint(f"❌ [{cyc_tag}] 下载/解析失败: {exc}\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="台风驱动的 GFS GRIB 下载+分析流水线\n"
                    "先看台风CSV → 确定需要的GFS cycle → 下载 → 追踪 → 提取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── 核心输入: 台风数据 ────────────────────────────────────────────────────
    parser.add_argument("--tracks", required=True,
                        help="台风轨迹CSV路径 (必须, 例如 input/matched_cyclone_tracks.csv)")
    parser.add_argument("--storms", nargs="*", default=None,
                        help="只处理指定的storm_id (不指定=全部)")
    parser.add_argument("--cycles", nargs="*", default=None,
                        help="手动指定cycle (覆盖自动计算, 例如 20250308T00 20250308T12)")
    parser.add_argument("--lead-days", type=int, default=10,
                        help="最大提前天数 (默认10天=f240)")

    # ── 并行参数 (根据 CPU 核数自动调整) ─────────────────────────────────────
    _def_cycle   = max(2, _N_CPU // 4)       # 24核→6
    _def_dl      = max(4, _N_CPU // 2)       # 24核→12
    _def_parse   = max(4, _N_CPU - 4)        # 24核→20
    _def_extract = max(1, _N_CPU // 2)       # 24核→12

    parser.add_argument("--cycle-workers",    type=int, default=_def_cycle,
                        help=f"同时处理的 cycle 数量 (自动={_def_cycle})")
    parser.add_argument("--download-threads", type=int, default=_def_dl,
                        help=f"每个 cycle 的 S3 并发下载线程数 (自动={_def_dl})")
    parser.add_argument("--parse-workers",    type=int, default=_def_parse,
                        help=f"cfgrib 解析进程数 (自动={_def_parse})")
    parser.add_argument("--processes",        type=int, default=_def_extract,
                        help=f"环境提取并行进程数 (自动={_def_extract})")

    # ── 其他参数 ─────────────────────────────────────────────────────────────
    parser.add_argument("--batch-size", type=int, default=0,
                        help="每批处理的 cycle 数量 (0=自动)")
    parser.add_argument("--keep-nc",   action="store_true", help="处理后保留 NC 文件")
    parser.add_argument("--keep-grib", action="store_true", help="写 NC 后保留 GRIB 文件")
    parser.add_argument("--grib-dir",  default=None,
                        help="GRIB 缓存目录 (默认自动: /dev/shm 或 data/)")
    args = parser.parse_args()

    # ── 自动选择 GRIB 缓存目录 ────────────────────────────────────────────────
    if args.grib_dir is None:
        import shutil as _shutil
        shm = Path("/dev/shm")
        if shm.is_dir():
            try:
                free_gb = _shutil.disk_usage(str(shm)).free / 1e9
                if free_gb > 20:
                    args.grib_dir = str(shm / "grib_cache")
                    print(f"📁 GRIB 目录: {args.grib_dir}  ({free_gb:.0f} GB 可用, RAM tmpfs)")
            except Exception:
                pass
        if args.grib_dir is None:
            args.grib_dir = "data/grib_cache"
            free_gb = _shutil.disk_usage("data").free / 1e9
            print(f"📁 GRIB 目录: {args.grib_dir}  ({free_gb:.0f} GB 可用, 磁盘)")

    # ── 第1步: 从台风数据确定需要的GFS cycles ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"🌀 第1步: 读取台风数据, 确定需要的GFS预报cycle")
    print(f"{'='*70}")

    if args.cycles:
        # 手动指定cycle — 不需要计算
        cycle_tags = []
        cycle_dts = []
        for s in args.cycles:
            s = s.strip().replace("-", "").replace(":", "")
            try:
                dt = datetime.strptime(s, "%Y%m%dT%H").replace(tzinfo=timezone.utc)
                cycle_tags.append(dt.strftime("%Y%m%dT%H"))
                cycle_dts.append(dt)
            except ValueError:
                print(f"⚠️ 无法解析cycle: {s}, 跳过")
        cycle_storms = {tag: ["manual"] for tag in cycle_tags}
        print(f"📋 手动指定 {len(cycle_dts)} 个 cycles")
    else:
        # 自动计算: 从台风CSV确定需要哪些cycle
        cycle_dts, cycle_storms = compute_needed_cycles(
            args.tracks,
            lead_days=args.lead_days,
            storm_filter=args.storms,
        )
        cycle_tags = [dt.strftime("%Y%m%dT%H") for dt in cycle_dts]

    if not cycle_dts:
        raise SystemExit("❌ 没有需要处理的cycle")

    # ── 检查已有结果, 跳过已完成的cycle ───────────────────────────────────────
    print(f"\n🔍 检查已有分析结果...")
    existing_jsons = set()
    json_dir = Path("final_single_output")
    if json_dir.exists():
        for jf in json_dir.glob("*.json"):
            if jf.name.startswith("_"):
                continue
            existing_jsons.add(jf.stem)

    already_done = []
    todo_dts = []
    todo_tags = []
    for dt, tag in zip(cycle_dts, cycle_tags):
        init_tag = dt.strftime("%Y-%m-%d%H")
        nc_stem = f"gfs_{init_tag}_f000_f240_6h"
        # Check if ANY json contains this NC stem pattern
        if any(nc_stem in js for js in existing_jsons):
            already_done.append(tag)
        else:
            todo_dts.append(dt)
            todo_tags.append(tag)

    if already_done:
        print(f"⏭️ 已有分析结果的cycles: {len(already_done)} 个, 跳过")
    print(f"📋 待处理cycles: {len(todo_dts)} 个")

    if not todo_dts:
        print("🎉 所有cycle都已完成!")
        return

    # ── 第2步: 分批下载+处理 ──────────────────────────────────────────────────
    import time as _time
    t_start = _time.time()
    total = len(todo_dts)
    batch_size = args.batch_size if args.batch_size > 0 else max(12, args.cycle_workers * 3)

    print(f"\n{'='*70}")
    print(f"🚀 第2步: 分批下载+追踪+提取")
    print(f"{'='*70}")
    print(f"   总cycles: {total}  |  CPU={_N_CPU}核  |  RAM={_os.sysconf('SC_PAGE_SIZE')*_os.sysconf('SC_PHYS_PAGES')/1e9:.0f}GB")
    print(f"   下载+解析: cycle-workers={args.cycle_workers}, "
          f"download-threads={args.download_threads}, parse-workers={args.parse_workers}")
    print(f"   追踪+提取: processes={args.processes}")
    print(f"   每 {batch_size} 个 cycles 一批, 共 {(total + batch_size - 1) // batch_size} 批\n")

    # 共享cfgrib解析进程池
    import multiprocessing as _mp
    shared_executor = __import__("concurrent.futures", fromlist=["ProcessPoolExecutor"]).ProcessPoolExecutor(
        max_workers=args.parse_workers,
        mp_context=_mp.get_context("spawn"),
    )

    class _RunArgs:
        def __init__(self, a):
            self.processes = a.processes
            self.no_clean  = a.keep_nc
            self.keep_nc   = a.keep_nc
            self.auto      = True
            self.initials  = a.tracks   # 用同一个台风CSV做追踪
            self.tracks    = None

    total_processed = 0
    total_skipped   = 0
    total_errors    = 0

    for batch_idx in range(0, total, batch_size):
        batch_dts = todo_dts[batch_idx : batch_idx + batch_size]
        batch_tags = todo_tags[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        n_batches = (total + batch_size - 1) // batch_size
        batch_t0 = _time.time()

        # 显示这批次涉及的台风
        batch_storm_ids: Set[str] = set()
        for tag in batch_tags:
            batch_storm_ids.update(cycle_storms.get(tag, []))

        print(f"\n{'='*70}")
        print(f"📦 批次 {batch_num}/{n_batches}  |  cycles {batch_idx+1}-{batch_idx+len(batch_dts)}/{total}")
        print(f"   时间: {batch_tags[0]} → {batch_tags[-1]}")
        print(f"   关联台风: {', '.join(sorted(batch_storm_ids)[:5])}"
              f"{'...' if len(batch_storm_ids) > 5 else ''}")
        print(f"{'='*70}")

        # ── 阶段A: 动态生成URL + 并行下载 + cfgrib解析 → NC ─────────────────
        n_workers = min(args.cycle_workers, len(batch_dts))
        built_nc: List[Path] = []

        if n_workers <= 1:
            for dt, tag in zip(batch_dts, batch_tags):
                urls = generate_cycle_urls(dt)
                result = _assemble_only(dt, tag, urls, args, executor=shared_executor)
                if result:
                    built_nc.append(result)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as tex:
                fut_to_tag = {}
                for dt, tag in zip(batch_dts, batch_tags):
                    urls = generate_cycle_urls(dt)
                    fut = tex.submit(_assemble_only, dt, tag, urls, args, shared_executor)
                    fut_to_tag[fut] = tag

                done_in_batch = 0
                for fut in as_completed(fut_to_tag):
                    tag = fut_to_tag[fut]
                    done_in_batch += 1
                    try:
                        result = fut.result()
                        if result:
                            built_nc.append(result)
                    except Exception as exc:
                        total_errors += 1
                        _tprint(f"❌ [{tag}] 异常: {exc}")
                    overall_done = batch_idx + done_in_batch
                    elapsed = _time.time() - t_start
                    avg = elapsed / overall_done if overall_done else 0
                    _tprint(f"📦 进度: {overall_done}/{total}  "
                            f"已用时 {elapsed:.0f}s  平均 {avg:.1f}s/cycle  "
                            f"预计剩余 {avg*(total-overall_done)/60:.0f}min")

        batch_dl_t = _time.time() - batch_t0
        print(f"   ⬇️  批次{batch_num} 下载完成: {len(built_nc)} NCs, 用时 {batch_dl_t:.0f}s")

        # ── 阶段B: 追踪 + 环境提取 ──────────────────────────────────────────
        if built_nc:
            print(f"   🔍 批次{batch_num} 开始追踪+提取 ({len(built_nc)} NCs)...")
            try:
                processed, skipped = process_nc_files(
                    built_nc,
                    _RunArgs(args),
                    concise_log=True,
                    logs_root=Path("final_single_output/logs"),
                )
                total_processed += processed
                total_skipped += skipped
            except Exception as exc:
                print(f"   ❌ 批次{batch_num} 追踪提取异常: {exc}")
                import traceback; traceback.print_exc()

        batch_elapsed = _time.time() - batch_t0
        overall_elapsed = _time.time() - t_start
        overall_done = min(batch_idx + len(batch_dts), total)
        remaining = total - overall_done
        eta_s = overall_elapsed / overall_done * remaining if overall_done > 0 else 0
        print(f"   ✅ 批次{batch_num} 完成: {batch_elapsed:.0f}s | "
              f"总进度 {overall_done}/{total} | "
              f"已用 {overall_elapsed:.0f}s | "
              f"预计剩余 {eta_s/60:.0f}min")

    shared_executor.shutdown(wait=True)

    total_elapsed = _time.time() - t_start
    print(f"\n{'='*70}")
    print(f"🎉 全部完成: {total} cycles")
    print(f"   ✅ 已分析: {total_processed}")
    print(f"   ⏭️ 跳过: {total_skipped}")
    print(f"   ❌ 错误: {total_errors}")
    print(f"   ⏱️ 总用时: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    print(f"   📊 平均: {total_elapsed/max(total,1):.1f}s/cycle")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
