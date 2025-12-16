from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import xarray as xr

try:
    import dask  # noqa: F401

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


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

    msl = _open_mf_field(
        path_list, {"shortName": "prmsl", "typeOfLevel": "meanSea"}, rename={"prmsl": "msl"}, chunk_lat=chunk_lat, chunk_lon=chunk_lon, prefer_dask=prefer_dask
    )
    u10 = _open_mf_field(
        path_list, {"shortName": "10u", "typeOfLevel": "heightAboveGround"}, rename={"u10": "10u"}, chunk_lat=chunk_lat, chunk_lon=chunk_lon, prefer_dask=prefer_dask
    )
    v10 = _open_mf_field(
        path_list, {"shortName": "10v", "typeOfLevel": "heightAboveGround"}, rename={"v10": "10v"}, chunk_lat=chunk_lat, chunk_lon=chunk_lon, prefer_dask=prefer_dask
    )
    gh = _open_mf_field(
        path_list, {"shortName": "gh", "typeOfLevel": "isobaricInhPa"}, chunk_lat=chunk_lat, chunk_lon=chunk_lon, prefer_dask=prefer_dask
    )
    gh = gh.sel(isobaricInhPa=700, method="nearest").drop_vars("isobaricInhPa", errors="ignore").rename({"gh": "z"})

    ds = xr.merge([msl, u10, v10, gh], compat="override")
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


__all__ = ["open_grib_collection", "load_paths_from_griblist", "is_griblist"]
