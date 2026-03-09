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
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    TimeoutError as FuturesTimeout,
    as_completed,
)
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Set, Tuple
import math

import boto3
import pandas as pd
from boto3.s3.transfer import TransferConfig
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

# Keep per-file S3 downloads single-threaded so outer download_threads remains
# the true concurrency limit. The boto3 default uses its own thread pool, which
# multiplies request fan-out and can overwhelm shared HPC egress/S3 throttling.
_S3_TRANSFER_CONFIG = TransferConfig(max_concurrency=1, use_threads=False)
_TRACK_WINDOW_TOLERANCE = pd.Timedelta(hours=6, minutes=1)
_MIN_GRIB_FILES_PER_CYCLE = int(_os.getenv("MIN_GRIB_FILES_PER_CYCLE", "30"))
_MIN_FORECAST_SPAN_HOURS = int(_os.getenv("MIN_FORECAST_SPAN_HOURS", "120"))
try:
    _MIN_GRIB_SUCCESS_RATIO = float(_os.getenv("MIN_GRIB_SUCCESS_RATIO", "0.60"))
except ValueError:
    _MIN_GRIB_SUCCESS_RATIO = 0.60
_MIN_GRIB_SUCCESS_RATIO = min(max(_MIN_GRIB_SUCCESS_RATIO, 0.0), 1.0)

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
    # Be tolerant to UTF-8 BOM and surrounding spaces in CSV headers.
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]

    if "datetime" not in df.columns:
        raise SystemExit(
            f"❌ tracks CSV missing required column 'datetime': {tracks_csv}"
        )
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 检测列名
    if "storm_id" in df.columns:
        storm_col = "storm_id"
    elif "SID" in df.columns:
        storm_col = "SID"
    else:
        raise SystemExit(
            f"❌ tracks CSV missing storm id column ('storm_id' or 'SID'): {tracks_csv}"
        )

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


def _load_track_times_utc(tracks_csv: str | Path, storm_filter: list[str] | None = None) -> pd.DatetimeIndex:
    """Load all observed cyclone timestamps in UTC for cycle pre-filtering."""
    df = pd.read_csv(tracks_csv)
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    if "datetime" not in df.columns:
        return pd.DatetimeIndex([])

    storm_col = "storm_id" if "storm_id" in df.columns else ("SID" if "SID" in df.columns else None)
    if storm_filter and storm_col is not None:
        df = df[df[storm_col].astype(str).isin([str(s) for s in storm_filter])]
    if df.empty:
        return pd.DatetimeIndex([])

    ts = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    ts = ts.dropna()
    if ts.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(ts).sort_values()


def _filter_cycles_with_observed_points(
    cycle_dts: list[datetime],
    cycle_tags: list[str],
    track_times: pd.DatetimeIndex,
) -> tuple[list[datetime], list[str], int]:
    """Keep only cycles with backward-available observations at cycle init."""
    if not cycle_dts or track_times.empty:
        return cycle_dts, cycle_tags, 0

    track_ns = track_times.asi8
    kept_dts: list[datetime] = []
    kept_tags: list[str] = []
    dropped = 0

    for dt, tag in zip(cycle_dts, cycle_tags):
        dt_ts = pd.Timestamp(dt)
        if dt_ts.tzinfo is None:
            dt_ts = dt_ts.tz_localize("UTC")
        else:
            dt_ts = dt_ts.tz_convert("UTC")
        start = dt_ts - _TRACK_WINDOW_TOLERANCE
        end = dt_ts
        left = track_ns.searchsorted(start.value, side="left")
        right = track_ns.searchsorted(end.value, side="right")
        if right > left:
            kept_dts.append(dt)
            kept_tags.append(tag)
        else:
            dropped += 1
    return kept_dts, kept_tags, dropped


# ─────────────────────────────────────────────────────────────────────────────
# 第二步: 下载和解析
# ─────────────────────────────────────────────────────────────────────────────

def _download_file(
    s3,
    url: str,
    dest: Path,
    *,
    max_retries: int = 6,
    base_backoff_sec: float = 0.6,
) -> bool:
    """Download one S3 object with retry/backoff.

    Returns True on success, False on permanent 404/missing.
    Raises on repeated transient failures so the caller can decide whether to
    skip the cycle.
    """
    import random
    import time as _time

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return True
    if not url.startswith("s3://"):
        raise ValueError(f"Unsupported URL: {url}")
    _, _, bucket_key = url.partition("s3://")
    bucket, _, key = bucket_key.partition("/")

    transient_codes = {
        "SlowDown",
        "Throttling",
        "ThrottlingException",
        "RequestTimeout",
        "RequestTimeTooSkewed",
        "RequestExpired",
        "InternalError",
        "ServiceUnavailable",
        "500",
        "503",
    }
    transient_tokens = (
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "broken pipe",
        "too many requests",
        "slowdown",
        "throttl",
        "service unavailable",
    )

    for attempt in range(max_retries + 1):
        try:
            s3.download_file(bucket, key, str(dest), Config=_S3_TRANSFER_CONFIG)
            return True
        except Exception as exc:
            err_str = str(exc)
            err_lower = err_str.lower()
            code = ""
            if hasattr(exc, "response"):
                code = str(exc.response.get("Error", {}).get("Code", "")).strip()

            # 404 / object not found -> treat as normal miss
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            if "404" in err_str or "not found" in err_lower or "nosuchkey" in err_lower:
                return False

            transient = False
            if code in transient_codes:
                transient = True
            elif any(tok in err_lower for tok in transient_tokens):
                transient = True

            # Keep only complete files; partial writes may happen on interrupted transfers.
            if dest.exists():
                try:
                    dest.unlink()
                except Exception:
                    pass

            if transient and attempt < max_retries:
                sleep_sec = min(base_backoff_sec * (2**attempt), 20.0) + random.uniform(0.0, 0.4)
                _time.sleep(sleep_sec)
                continue
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
    overall_timeout_sec: int | None = None,
) -> List[str]:
    """Download a list of S3 URLs in parallel using ThreadPoolExecutor.
    Returns list of successfully downloaded local paths."""
    # 下载前先检查 /dev/shm 空间，超过 70GB 则等待
    _wait_for_shm_space(cache_dir)
    local_paths: list[str | None] = [None] * len(urls)
    n_skipped = 0

    n_threads = max(1, int(n_threads))
    if overall_timeout_sec is None:
        overall_timeout_sec = int(_os.getenv("GFS_DOWNLOAD_TIMEOUT_SEC", "1200"))
    s3 = boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            max_pool_connections=max(32, n_threads * 2),
            connect_timeout=10,
            read_timeout=180,
            retries={"max_attempts": 6, "mode": "standard"},
        ),
    )

    def _worker(idx_url: tuple[int, str]) -> tuple[int, str | None]:
        idx, url = idx_url
        local = _url_to_local(url, cache_dir)
        ok = _download_file(s3, url, local)
        return idx, str(local) if ok else None

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {ex.submit(_worker, (i, url)): i for i, url in enumerate(urls)}
        try:
            for fut in as_completed(futures, timeout=overall_timeout_sec):
                try:
                    idx, local = fut.result()
                except Exception as exc:
                    idx = futures[fut]
                    local = None
                    _tprint(f"⚠️ 下载失败(线程异常): idx={idx}, err={exc}")
                if local is None:
                    n_skipped += 1
                else:
                    local_paths[idx] = local
        except FuturesTimeout:
            pending = [f for f in futures if not f.done()]
            for fut in pending:
                fut.cancel()
            n_skipped += len(pending)
            _tprint(
                f"⚠️ 下载批次超时 ({overall_timeout_sec}s)，已取消 {len(pending)} 个未完成文件"
            )

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
                    # 清理可能残留的 GRIB 文件（上次崩溃时可能未删除）
                    if not keep_grib and grib_dir is not None:
                        import shutil as _shutil_skip
                        init_tag_short = cycle_dt.strftime("%Y%m%d")
                        for _cyc_h in ("00", "06", "12", "18"):
                            _grib_sub = grib_dir / init_tag_short / _cyc_h
                            if _grib_sub.is_dir():
                                try:
                                    _shutil_skip.rmtree(_grib_sub, ignore_errors=True)
                                    _tprint(f"🧹 清理残留 GRIB: {_grib_sub}")
                                except Exception:
                                    pass
                    return out_path
            _tprint(f"⚠️ NC 存在但不完整 ({sz} bytes)，将重建: {out_path.name}")
            out_path.unlink(missing_ok=True)
        except Exception as exc:
            _tprint(f"⚠️ NC 校验失败 ({exc})，将重建: {out_path.name}")
            out_path.unlink(missing_ok=True)

    local_urls: list[str] = []
    # One cycle should map to one GRIB subdir, but keep a set for safety.
    expected_grib_dirs = {Path(_url_to_local(url, cache_dir)).parent for url in urls}
    ds = None
    try:
        # ── Step 1: Parallel S3 download ──────────────────────────────────────
        local_urls = _download_files_parallel(urls, cache_dir, n_threads=download_threads)
        if not local_urls:
            raise RuntimeError("No data downloaded for cycle (all 404?)")

        downloaded_fhours: list[int] = []
        for p in local_urls:
            try:
                downloaded_fhours.append(int(Path(p).name.split("f")[-1]))
            except (ValueError, IndexError):
                continue
        downloaded_fhours = sorted(set(downloaded_fhours))
        has_f000 = 0 in downloaded_fhours
        forecast_span_h = (
            downloaded_fhours[-1] - downloaded_fhours[0]
            if downloaded_fhours
            else 0
        )
        min_files_by_ratio = math.ceil(len(urls) * _MIN_GRIB_SUCCESS_RATIO)
        min_files_required = max(_MIN_GRIB_FILES_PER_CYCLE, min_files_by_ratio)

        if (
            len(local_urls) < min_files_required
            or not has_f000
            or forecast_span_h < _MIN_FORECAST_SPAN_HOURS
        ):
            raise RuntimeError(
                "Insufficient GRIB coverage for reliable tracking: "
                f"downloaded={len(local_urls)}/{len(urls)}, "
                f"unique_fhours={len(downloaded_fhours)}, "
                f"has_f000={has_f000}, "
                f"forecast_span_h={forecast_span_h}, "
                f"thresholds(min_files={min_files_required}, "
                f"min_span_h={_MIN_FORECAST_SPAN_HOURS}, "
                f"min_ratio={_MIN_GRIB_SUCCESS_RATIO:.2f})"
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

        # ── Step 2: Fast per-file parallel cfgrib extraction ──────────────────
        ds = open_grib_collection_fast(
            local_urls,
            n_workers=parse_workers,
            valid_times=valid_times if valid_times else None,
            executor=executor,
        )

        # ── Step 3: Write NetCDF ──────────────────────────────────────────────
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
        _tprint(f"📝 NC 已写入: {out_path.name} ({out_path.stat().st_size/1e6:.0f}MB)")

        # ── Step 4: Delete GRIB files to free disk/RAM ────────────────────────
        if not keep_grib and local_urls:
            import shutil as _shutil

            cyc_dir = Path(local_urls[0]).parent
            try:
                _shutil.rmtree(cyc_dir, ignore_errors=True)
                _tprint(f"🧹 已删除 GRIB 目录: {cyc_dir}")
            except Exception as exc:
                _tprint(f"⚠️ 删除 GRIB 目录失败: {exc}")

        return out_path
    except Exception:
        import shutil as _shutil

        # Remove partially written NC so next retry doesn't read a corrupt file.
        if out_path.exists():
            try:
                out_path.unlink()
                _tprint(f"🧹 已删除残缺 NC: {out_path.name}")
            except Exception:
                pass

        if not keep_grib:
            cleanup_dirs = expected_grib_dirs
            if local_urls:
                cleanup_dirs = {Path(p).parent for p in local_urls}
            for cyc_dir in sorted(cleanup_dirs):
                try:
                    _shutil.rmtree(cyc_dir, ignore_errors=True)
                    _tprint(f"🧹 失败后清理 GRIB 目录: {cyc_dir}")
                except Exception:
                    pass
        raise
    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass


def _assemble_only(
    cycle_dt: datetime,
    cyc_tag: str,
    urls: List[str],
    args,
    executor=None,
) -> tuple[Path | None, str]:
    """Phase A: Download + cfgrib parse → write NetCDF. Thread-safe.

    Returns (nc_path, error_msg).  nc_path is None on failure.
    """
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
        return out_path, ""
    except Exception as exc:
        import traceback
        err_short = str(exc)[:200]
        _tprint(f"❌ [{cyc_tag}] 下载/解析失败: {exc}\n{traceback.format_exc()}")
        return None, err_short


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
    parser.add_argument(
        "--skip-head-cycles",
        type=int,
        default=int(_os.getenv("SKIP_HEAD_CYCLES", "0")),
        help="启动时跳过待处理列表最前面的 N 个 cycle",
    )
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

    # Secondary guard: keep only cycles with backward-available observations at init time.
    # This prevents downloading cycles that cannot start tracking without future leakage.
    track_times = _load_track_times_utc(args.tracks, args.storms)
    todo_dts, todo_tags, dropped_no_points = _filter_cycles_with_observed_points(
        todo_dts,
        todo_tags,
        track_times,
    )
    if dropped_no_points:
        print(
            f"⏭️ 预检跳过 {dropped_no_points} 个无 backward 观测点的 cycle "
            f"(窗口=init-{_TRACK_WINDOW_TOLERANCE}~init)"
        )
        print(f"📋 过滤后待处理cycles: {len(todo_dts)} 个")

    skip_head_cycles = max(0, int(getattr(args, "skip_head_cycles", 0)))
    if skip_head_cycles > 0 and todo_dts:
        n_drop = min(skip_head_cycles, len(todo_dts))
        dropped_tags = todo_tags[:n_drop]
        todo_dts = todo_dts[n_drop:]
        todo_tags = todo_tags[n_drop:]
        print(
            f"⏭️ 配置 skip-head-cycles={skip_head_cycles}, "
            f"额外跳过前 {n_drop} 个 cycle: {dropped_tags[0]} → {dropped_tags[-1]}"
        )
        print(f"📋 跳过后待处理cycles: {len(todo_dts)} 个")

    if not todo_dts:
        print("🎉 没有剩余 cycle 需要执行。")
        return

    # ── 第2步: 逐cycle下载+处理 (流式, 避免资源争抢) ────────────────────────
    import time as _time
    import shutil as _shutil_main
    t_start = _time.time()
    total = len(todo_dts)

    # 当 cycle_workers=1 时，使用 batch_size=1 实现逐 cycle 处理+清理
    # 避免大批次下载占满磁盘/内存后再统一处理
    if args.cycle_workers <= 1:
        batch_size = 1
    elif args.batch_size > 0:
        batch_size = args.batch_size
    else:
        batch_size = max(4, args.cycle_workers * 2)

    print(f"\n{'='*70}")
    print(f"🚀 第2步: 下载+追踪+提取 (batch_size={batch_size})")
    print(f"{'='*70}")
    print(f"   总cycles: {total}  |  CPU={_N_CPU}核  |  RAM={_os.sysconf('SC_PAGE_SIZE')*_os.sysconf('SC_PHYS_PAGES')/1e9:.0f}GB")
    print(f"   下载+解析: cycle-workers={args.cycle_workers}, "
          f"download-threads={args.download_threads}, parse-workers={args.parse_workers}")
    print(f"   追踪+提取: processes={args.processes}")
    if batch_size == 1:
        print(f"   模式: 逐cycle流式处理 (下载→解析→追踪→提取→清理)\n")
    else:
        print(f"   每 {batch_size} 个 cycles 一批, 共 {(total + batch_size - 1) // batch_size} 批\n")

    # 共享cfgrib解析进程池 (带自动恢复)
    import multiprocessing as _mp
    _spawn_ctx = _mp.get_context("spawn")

    class _ResilientExecutor:
        """Wrapper that auto-recreates the ProcessPool when a child crashes."""
        def __init__(self, max_workers, mp_context):
            self._max_workers = max_workers
            self._mp_context = mp_context
            self._lock = Lock()
            self._pool: ProcessPoolExecutor | None = None
            self._create_pool()

        def _create_pool(self):
            self._pool = ProcessPoolExecutor(
                max_workers=self._max_workers,
                mp_context=self._mp_context,
            )
            _tprint(f"🔄 cfgrib 进程池已创建 (workers={self._max_workers})")

        def submit(self, fn, *a, **kw):
            with self._lock:
                try:
                    return self._pool.submit(fn, *a, **kw)
                except BrokenProcessPool:
                    _tprint("⚠️ 进程池已损坏，正在重建...")
                    try:
                        self._pool.shutdown(wait=False)
                    except Exception:
                        pass
                    self._create_pool()
                    return self._pool.submit(fn, *a, **kw)

        def shutdown(self, wait=True):
            if self._pool:
                self._pool.shutdown(wait=wait)

    shared_executor = _ResilientExecutor(
        max_workers=args.parse_workers,
        mp_context=_spawn_ctx,
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
    _last_error_msg = ""       # 最近一次失败的描述
    _failed_cycles: list[str] = []  # 所有失败的 cycle tag

    # 进度状态文件: 写入共享目录以供实时监控
    progress_file = Path("final_single_output") / "_progress.txt"
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    def _write_progress(current: int, tag: str, status: str) -> None:
        """Write human-readable progress to shared dir for real-time monitoring."""
        elapsed = _time.time() - t_start
        avg = elapsed / current if current else 0
        remaining = avg * (total - current)
        failed_summary = ",".join(_failed_cycles[-10:])  # 最近10个失败cycle
        try:
            progress_file.write_text(
                f"progress={current}/{total}\n"
                f"current_cycle={tag}\n"
                f"status={status}\n"
                f"elapsed_s={elapsed:.0f}\n"
                f"avg_s_per_cycle={avg:.1f}\n"
                f"eta_s={remaining:.0f}\n"
                f"processed={total_processed}\n"
                f"skipped={total_skipped}\n"
                f"errors={total_errors}\n"
                f"last_error={_last_error_msg}\n"
                f"recent_failed_cycles={failed_summary}\n"
                f"updated_utc={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def _cleanup_grib_nc_dir() -> None:
        """删除批次结束后残留的 GRIB 缓存目录，立即释放磁盘空间。"""
        # NC 文件由 _run_environment_analysis 自动删除 (keep_nc=False)
        # 这里额外清理 GRIB 缓存残留（含非空目录）
        if not args.keep_grib and hasattr(args, 'grib_dir') and args.grib_dir:
            import shutil as _shutil_cleanup
            grib_root = Path(args.grib_dir)
            if grib_root.exists():
                for sub in sorted(grib_root.iterdir(), reverse=True):
                    if sub.is_dir():
                        try:
                            _shutil_cleanup.rmtree(sub, ignore_errors=True)
                        except Exception:
                            pass

    for batch_idx in range(0, total, batch_size):
        batch_dts = todo_dts[batch_idx : batch_idx + batch_size]
        batch_tags = todo_tags[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        n_batches = (total + batch_size - 1) // batch_size
        batch_t0 = _time.time()
        batch_had_failures = False

        # 显示这批次涉及的台风
        batch_storm_ids: Set[str] = set()
        for tag in batch_tags:
            batch_storm_ids.update(cycle_storms.get(tag, []))

        print(f"\n{'─'*70}")
        print(f"📦 [{batch_num}/{n_batches}]  cycles {batch_idx+1}-{batch_idx+len(batch_dts)}/{total}")
        print(f"   时间: {batch_tags[0]} → {batch_tags[-1]}")
        print(f"   关联台风: {', '.join(sorted(batch_storm_ids)[:5])}"
              f"{'...' if len(batch_storm_ids) > 5 else ''}")

        # ── 阶段A: 动态生成URL + 并行下载 + cfgrib解析 → NC ─────────────────
        n_workers = min(args.cycle_workers, len(batch_dts))
        built_nc: List[Path] = []

        _write_progress(batch_idx + 1, batch_tags[0], "downloading")

        if n_workers <= 1:
            done_in_batch = 0
            for dt, tag in zip(batch_dts, batch_tags):
                urls = generate_cycle_urls(dt)
                result, err_msg = _assemble_only(dt, tag, urls, args, executor=shared_executor)
                if result:
                    built_nc.append(result)
                else:
                    total_errors += 1
                    batch_had_failures = True
                    _failed_cycles.append(tag)
                    _tprint(f"⚠️ [{tag}] 下载/解析失败，跳过此 cycle: {err_msg}")
                    _last_error_msg = f"{tag}: {err_msg or 'download/parse failed'}"
                done_in_batch += 1
                _write_progress(batch_idx + done_in_batch, tag, "downloading")
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
                        result, err_msg = fut.result()
                        if result:
                            built_nc.append(result)
                        elif err_msg:
                            total_errors += 1
                            batch_had_failures = True
                            _failed_cycles.append(tag)
                            _last_error_msg = f"{tag}: {err_msg}"
                    except Exception as exc:
                        total_errors += 1
                        batch_had_failures = True
                        _failed_cycles.append(tag)
                        _last_error_msg = f"{tag}: {exc}"
                        _tprint(f"❌ [{tag}] 异常: {exc}")
                    overall_done = batch_idx + done_in_batch
                    elapsed = _time.time() - t_start
                    avg = elapsed / overall_done if overall_done else 0
                    _tprint(f"📦 进度: {overall_done}/{total}  "
                            f"已用时 {elapsed:.0f}s  平均 {avg:.1f}s/cycle  "
                            f"预计剩余 {avg*(total-overall_done)/60:.0f}min")
                    _write_progress(overall_done, tag, "downloading")

        batch_dl_t = _time.time() - batch_t0
        batch_status = "completed_batch"

        # ── 阶段B: 追踪 + 环境提取 ──────────────────────────────────────────
        if built_nc:
            _write_progress(batch_idx + 1, batch_tags[0], "tracking+extracting")
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
                batch_status = "analysis_failed"
                _last_error_msg = f"{batch_tags[0]}: {exc}"
                print(f"   ❌ 批次{batch_num} 追踪提取异常: {exc}")
                import traceback; traceback.print_exc()

            # ── 阶段C: 立即清理残留 GRIB/NC (process_nc_files 已删除 NC) ─────
            _cleanup_grib_nc_dir()
        else:
            batch_status = "failed_batch"
            print(
                f"   ⚠️ [{batch_num}/{n_batches}] cycle {batch_tags[0]} 失败 (无NC产出): "
                f"{_last_error_msg or 'download/parse failed'}"
            )

        if built_nc and batch_had_failures:
            batch_status = "partial_batch"

        overall_elapsed = _time.time() - t_start
        overall_done = min(batch_idx + len(batch_dts), total)
        remaining = total - overall_done
        eta_s = overall_elapsed / overall_done * remaining if overall_done > 0 else 0
        print(f"   ✅ [{batch_num}/{n_batches}] {batch_dl_t:.0f}s | "
              f"总进度 {overall_done}/{total} | "
              f"已用 {overall_elapsed:.0f}s | "
              f"剩余 ~{eta_s/60:.0f}min")

        _write_progress(overall_done, batch_tags[-1], batch_status)

    shared_executor.shutdown(wait=True)

    _write_progress(total, "done", "finished")

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
