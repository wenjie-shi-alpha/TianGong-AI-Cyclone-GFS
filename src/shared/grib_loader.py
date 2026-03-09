from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List
import os

import pandas as pd
import xarray as xr
import numpy as np

try:
    import dask  # noqa: F401

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


# ---------------------------------------------------------------------------
# Per-file parallel extraction worker (module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _extract_one_grib_file_worker(args: tuple) -> tuple:
    """
    Worker process: extract mandatory + optional fields from ONE GRIB file.

    Uses a shared cfgrib index file (.idx) stored alongside the GRIB file.
    First call per file builds the index (~2s); subsequent calls read the
    index directly (~0.006s) — ~400x speedup on re-runs or when the same
    file is accessed for different variables.

    Returns:
      (
        path,
        {
          'msl': arr2d, '10u': arr2d, '10v': arr2d,
          'z': arr3d(level,lat,lon), 'u': arr3d, 'v': arr3d, 't': arr3d,
          'q'?: arr3d, 'w'?: arr3d, 't2m'?: arr2d, 'sst'?: arr2d
        },
        lats,
        lons,
        valid_time,
        {'z': levels, 'u': levels, ...}
      )
    """
    path, idx_dir = args
    import cfgrib as _cfgrib
    import numpy as _np
    import pandas as _pd

    os.makedirs(idx_dir, exist_ok=True)
    idx_path = os.path.join(idx_dir, os.path.basename(path) + ".idx")

    QUERIES: list[tuple[str, dict, bool]] = [
        ("msl", {"shortName": "prmsl", "typeOfLevel": "meanSea"}, True),
        ("10u", {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10}, True),
        ("10v", {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10}, True),
        ("z", {"shortName": "gh", "typeOfLevel": "isobaricInhPa"}, True),
        ("u", {"shortName": "u", "typeOfLevel": "isobaricInhPa"}, True),
        ("v", {"shortName": "v", "typeOfLevel": "isobaricInhPa"}, True),
        ("t", {"shortName": "t", "typeOfLevel": "isobaricInhPa"}, True),
        ("q", {"shortName": "q", "typeOfLevel": "isobaricInhPa"}, False),
        ("w", {"shortName": "w", "typeOfLevel": "isobaricInhPa"}, False),
        ("t2m", {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2}, False),
        ("tsfc", {"shortName": "t", "typeOfLevel": "surface"}, False),
        ("sst", {"shortName": "sst", "typeOfLevel": "surface"}, False),
    ]

    out: dict[str, _np.ndarray] = {}
    lats = lons = None
    valid_time = None
    isobaric_levels: dict[str, _np.ndarray] = {}

    for var_name, fkeys, required in QUERIES:
        try:
            ds = _cfgrib.open_dataset(path, filter_by_keys=fkeys, indexpath=idx_path)
        except Exception:
            if required:
                raise
            continue

        # cfgrib may return an empty Dataset (no data_vars) when the requested
        # shortName/typeOfLevel combination is absent in a GRIB message. This is
        # common for optional fields such as SST over some forecast files.
        if len(ds.data_vars) == 0:
            if required:
                raise RuntimeError(
                    f"Required GRIB field empty after filter: var={var_name}, keys={fkeys}, file={path}"
                )
            continue

        if var_name == "z" and "gh" in ds.data_vars:
            arr = _np.asarray(ds["gh"].values)
        elif var_name in {"u", "v", "t", "q", "w"} and var_name in ds.data_vars:
            arr = _np.asarray(ds[var_name].values)
        else:
            orig = list(ds.data_vars)[0]
            arr = _np.asarray(ds[orig].values)

        if "isobaricInhPa" in ds.coords:
            levels = _np.asarray(ds["isobaricInhPa"].values, dtype=float).reshape(-1)
            isobaric_levels[var_name] = levels
            if arr.ndim == 2:
                arr = arr[_np.newaxis, :, :]
        else:
            arr = arr.squeeze()

        out[var_name] = arr

        if lats is None and "latitude" in ds.coords:
            lats = _np.asarray(ds["latitude"])
            lons = _np.asarray(ds["longitude"])
        if valid_time is None:
            for cn in ("valid_time", "time"):
                if cn in ds.coords:
                    try:
                        valid_time = _pd.Timestamp(ds[cn].values)
                        break
                    except Exception:
                        pass

    return path, out, lats, lons, valid_time, isobaric_levels


def open_grib_collection_fast(
    paths: Iterable[str],
    n_workers: int = 8,
    valid_times: list[pd.Timestamp] | None = None,
    executor=None,
) -> xr.Dataset:
    """
    Fast parallel GRIB → xarray Dataset.

    Each of the N GRIB files is processed ONCE in a dedicated worker process
    (extracting mandatory multi-level fields + optional auxiliaries together).
    A shared cfgrib index file is
    written next to each GRIB file on first access so subsequent runs skip
    the full-file scan entirely.

    Speed vs open_grib_collection (serial, no index):
      First run  : ~12s  (8 workers, 41 files, builds idx)
      Cached run : ~1s   (idx files exist, direct seek)
      Old method : ~347s (4x full scan per file, no index)

    Parameters
    ----------
    paths     : iterable of GRIB file paths (must all be from the same cycle)
    n_workers : ProcessPoolExecutor worker count (default 8)
    valid_times: optional override for the time axis
    executor  : optional shared ProcessPoolExecutor (recommended for multi-cycle
                parallel runs to avoid spawning multiple pools simultaneously).
                If None, a local pool is created and destroyed per call.
    """
    from concurrent.futures import ProcessPoolExecutor

    path_list = list(paths)
    if not path_list:
        raise ValueError("No GRIB files provided")

    idx_dir = str(Path(path_list[0]).parent / ".cfgrib_idx")
    work_items = [(p, idx_dir) for p in path_list]

    if executor is not None:
        # 使用调用方提供的共享池 — 用 submit+as_completed 替代 map 以支持超时
        from concurrent.futures import as_completed as _as_completed
        from concurrent.futures.process import BrokenProcessPool as _BrokenPool
        timeout_sec = int(os.getenv("GFS_CFGRIB_TIMEOUT_SEC", "900"))
        futs = {executor.submit(_extract_one_grib_file_worker, item): i
                for i, item in enumerate(work_items)}
        raw_ordered: list = [None] * len(work_items)
        try:
            for fut in _as_completed(futs, timeout=timeout_sec):
                raw_ordered[futs[fut]] = fut.result()
        except _BrokenPool:
            raise RuntimeError(
                "cfgrib 子进程崩溃 (可能 OOM)，进程池已损坏"
            )
        except TimeoutError:
            raise RuntimeError(
                f"cfgrib 解析超时 (>{timeout_sec}s), {sum(1 for r in raw_ordered if r is None)}/{len(work_items)} 个文件未完成"
            )
        raw = raw_ordered
    else:
        # 单 cycle 模式：创建本地临时池
        # spawn: 使用 subprocess.Popen (posix_spawn)，线程安全；
        # 不用 forkserver（forkserver 在多线程进程中调用 os.fork() 会死锁）
        with ProcessPoolExecutor(
            max_workers=min(n_workers, len(path_list)),
            mp_context=__import__("multiprocessing").get_context("spawn"),
        ) as ex:
            raw = list(ex.map(_extract_one_grib_file_worker, work_items))

    p2r = {r[0]: r for r in raw}
    ordered = [p2r[p] for p in path_list]

    lats = next(r[2] for r in ordered if r[2] is not None)
    lons = next(r[3] for r in ordered if r[3] is not None)

    if valid_times is None:
        extracted = [r[4] for r in ordered]
        if all(t is not None for t in extracted):
            valid_times = extracted
        else:
            valid_times = _build_valid_times(path_list)

    time_idx = pd.DatetimeIndex(pd.to_datetime(valid_times))
    if time_idx.tz is not None:
        time_idx = time_idx.tz_convert("UTC").tz_localize(None)

    common_vars = set(ordered[0][1].keys())
    for rec in ordered[1:]:
        common_vars &= set(rec[1].keys())

    required = {"msl", "10u", "10v", "z", "u", "v", "t"}
    missing_required = sorted(required - common_vars)
    if missing_required:
        raise RuntimeError(
            f"GRIB fields missing after extraction: {missing_required}. "
            f"Available common fields: {sorted(common_vars)}"
        )

    isobaric_vars = {"z", "u", "v", "t", "q", "w"}
    required_isobaric_vars = {"z", "u", "v", "t"}
    levels = None
    for rec in ordered:
        level_map = rec[5]
        if "z" in level_map and level_map["z"].size:
            levels = np.asarray(level_map["z"], dtype=float).reshape(-1)
            break
    if levels is None:
        raise RuntimeError("Missing isobaric coordinate for mandatory field 'z'")

    data_vars: dict[str, tuple[list[str], np.ndarray]] = {}
    for var in sorted(common_vars):
        stacked = np.stack([np.asarray(rec[1][var]) for rec in ordered], axis=0)
        if var in isobaric_vars:
            if stacked.ndim != 4:
                if var in required_isobaric_vars:
                    raise RuntimeError(
                        f"Field '{var}' should be 4D after stacking (time,level,lat,lon), got shape={stacked.shape}"
                    )
                # Optional isobaric variable (e.g. q/w) with malformed shape.
                continue
            if stacked.shape[1] != levels.size:
                if var in required_isobaric_vars:
                    raise RuntimeError(
                        f"Field '{var}' level size mismatch: {stacked.shape[1]} vs {levels.size}"
                    )
                # Optional isobaric variable may have a reduced pressure-level set.
                # Skip it to keep the primary mandatory fields usable.
                continue
            data_vars[var] = (["time", "isobaricInhPa", "latitude", "longitude"], stacked)
        else:
            if stacked.ndim != 3:
                raise RuntimeError(
                    f"Field '{var}' should be 3D after stacking (time,lat,lon), got shape={stacked.shape}"
                )
            data_vars[var] = (["time", "latitude", "longitude"], stacked)

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": time_idx,
            "latitude": lats,
            "longitude": lons,
            "isobaricInhPa": levels,
        },
    )
    return ds.sortby("time")


def _parse_init_and_fhour(path: str) -> tuple[datetime | None, int | None]:
    fname = Path(path).name
    init_dt = None
    fhour = None

    segs = path.strip("/").split("/")
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
    return init_dt, fhour


def _build_valid_times(paths: List[str]) -> list[pd.Timestamp]:
    times: list[pd.Timestamp] = []
    for p in paths:
        init_dt, fhour = _parse_init_and_fhour(p)
        if init_dt and fhour is not None:
            times.append(pd.to_datetime(init_dt + timedelta(hours=fhour)))
    return times


def _open_mf_field(
    paths: List[str],
    filter_keys: dict,
    rename: dict | None = None,
    chunk_lat: int = 181,
    chunk_lon: int = 360,
    prefer_dask: bool = True,
):
    """
    Open multiple GRIB files with a safe fallback when dask is unavailable/misconfigured.

    prefer_dask=True keeps prior behavior; if the environment lacks a chunk manager
    it retries without dask chunks to avoid runtime failures.
    """
    chunks = {"time": 1, "latitude": chunk_lat, "longitude": chunk_lon} if HAS_DASK and prefer_dask else None
    try:
        ds = xr.open_mfdataset(
            paths,
            engine="cfgrib",
            combine="nested",
            concat_dim="time",
            parallel=False,
            coords="minimal",
            compat="override",
            data_vars="minimal",
            backend_kwargs={"filter_by_keys": filter_keys, "indexpath": ""},
            chunks=chunks,
        )
    except ValueError as exc:
        # Common when dask is partially installed ("unrecognized chunk manager dask")
        if chunks is not None and "chunk manager dask" in str(exc):
            ds = xr.open_mfdataset(
                paths,
                engine="cfgrib",
                combine="nested",
                concat_dim="time",
                parallel=False,
                coords="minimal",
                compat="override",
                data_vars="minimal",
                backend_kwargs={"filter_by_keys": filter_keys, "indexpath": ""},
                chunks=None,
            )
        else:
            raise

    if rename:
        ds = ds.rename(rename)
    return ds


def open_grib_collection(
    paths: Iterable[str],
    chunk_lat: int = 181,
    chunk_lon: int = 360,
    valid_times: list[pd.Timestamp] | None = None,
    prefer_dask: bool = True,
) -> xr.Dataset:
    """Open a list of GFS pgrb2 files as a merged Dataset with correct time axis."""
    path_list = list(paths)
    if not path_list:
        raise ValueError("No GRIB files provided")

    if valid_times is None:
        valid_times = _build_valid_times(path_list)

    def _rename_first_var(ds_in: xr.Dataset, new_name: str) -> xr.Dataset:
        first = list(ds_in.data_vars)[0]
        return ds_in.rename({first: new_name})

    fields: list[xr.Dataset] = []

    mandatory_queries = [
        ("msl", {"shortName": "prmsl", "typeOfLevel": "meanSea"}),
        ("10u", {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10}),
        ("10v", {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10}),
        ("z", {"shortName": "gh", "typeOfLevel": "isobaricInhPa"}),
        ("u", {"shortName": "u", "typeOfLevel": "isobaricInhPa"}),
        ("v", {"shortName": "v", "typeOfLevel": "isobaricInhPa"}),
        ("t", {"shortName": "t", "typeOfLevel": "isobaricInhPa"}),
    ]
    optional_queries = [
        ("q", {"shortName": "q", "typeOfLevel": "isobaricInhPa"}),
        ("w", {"shortName": "w", "typeOfLevel": "isobaricInhPa"}),
        ("t2m", {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2}),
        ("tsfc", {"shortName": "t", "typeOfLevel": "surface"}),
        ("sst", {"shortName": "sst", "typeOfLevel": "surface"}),
    ]

    for out_name, keys in mandatory_queries:
        ds_field = _open_mf_field(
            path_list,
            keys,
            chunk_lat=chunk_lat,
            chunk_lon=chunk_lon,
            prefer_dask=prefer_dask,
        )
        fields.append(_rename_first_var(ds_field, out_name))

    for out_name, keys in optional_queries:
        try:
            ds_field = _open_mf_field(
                path_list,
                keys,
                chunk_lat=chunk_lat,
                chunk_lon=chunk_lon,
                prefer_dask=prefer_dask,
            )
        except Exception:
            continue
        fields.append(_rename_first_var(ds_field, out_name))

    ds = xr.merge(fields, compat="override")
    if valid_times and "time" in ds.sizes and len(valid_times) == ds.sizes["time"]:
        ds = ds.assign_coords(time=("time", pd.to_datetime(valid_times)))
    elif valid_times and "time" in ds.sizes:
        steps = [i * 6 for i in range(ds.sizes["time"])]
        times = [valid_times[0] + pd.Timedelta(hours=h) for h in steps]
        ds = ds.assign_coords(time=("time", times))
    if "time" in ds.coords:
        ds = ds.sortby("time")
    return ds


def load_paths_from_griblist(list_path: Path) -> list[str]:
    with Path(list_path).open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def is_griblist(path: Path) -> bool:
    return path.suffix == ".griblist"


__all__ = ["open_grib_collection", "open_grib_collection_fast", "load_paths_from_griblist", "is_griblist"]
