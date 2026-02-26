"""Utilities for loading and filtering cyclone initial positions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_all_points(csv_path: Path) -> pd.DataFrame:
    """Read the cyclone catalogue and normalise time information."""
    df = pd.read_csv(csv_path)
    required = {"storm_id", "datetime", "latitude", "longitude", "max_wind_usa", "min_pressure_usa"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV 缺少必要列: {required - set(df.columns)}")
    df["dt"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["max_wind_usa"] = pd.to_numeric(df["max_wind_usa"], errors="coerce")
    df["min_pressure_usa"] = pd.to_numeric(df["min_pressure_usa"], errors="coerce")
    df = df.dropna(subset=["dt"]).copy()
    return df


def _load_initial_points(csv_path: Path) -> pd.DataFrame:
    """Backward-compatible alias used by downstream code."""
    return _load_all_points(csv_path)


def _select_initials_for_time(
    df_all: pd.DataFrame,
    target_time: pd.Timestamp,
    tol_hours: int = 6,
) -> pd.DataFrame:
    """Select the best matching initial point for each storm near a target time."""
    if df_all.empty:
        return pd.DataFrame(columns=["storm_id", "init_time", "init_lat", "init_lon"])
    # Slightly widen the time window to be robust to catalogue microsecond offsets
    delta = pd.Timedelta(hours=tol_hours) + pd.Timedelta(seconds=60)
    sub = df_all.loc[(df_all["dt"] >= target_time - delta) & (df_all["dt"] <= target_time + delta)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["storm_id", "init_time", "init_lat", "init_lon"])
    sub["time_diff"] = (sub["dt"] - target_time).abs()
    idx = sub.groupby("storm_id")["time_diff"].idxmin()
    pick = sub.loc[idx].copy()
    pick = pick.rename(columns={"latitude": "init_lat", "longitude": "init_lon"})
    pick["init_time"] = pick["dt"].values
    cols = [
        "storm_id",
        "init_time",
        "init_lat",
        "init_lon",
        "max_wind_usa",
        "min_pressure_usa",
    ]
    return pick[cols].reset_index(drop=True)


def _select_initials_for_window(
    df_all: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    tol_hours: int = 6,
) -> pd.DataFrame:
    """Select one initial point per storm within a forecast window.

    Strategy
    --------
    1) Prefer the earliest point at/after `start_time` and within
       `[start_time - tol, end_time + tol]`.
    2) If no such point exists for a storm, fallback to the closest point
       to `start_time` within the same widened window.
    """
    if df_all.empty:
        return pd.DataFrame(columns=["storm_id", "init_time", "init_lat", "init_lon"])

    delta = pd.Timedelta(hours=tol_hours) + pd.Timedelta(seconds=60)
    window = df_all.loc[
        (df_all["dt"] >= start_time - delta) & (df_all["dt"] <= end_time + delta)
    ].copy()
    if window.empty:
        return pd.DataFrame(columns=["storm_id", "init_time", "init_lat", "init_lon"])

    after_start = window.loc[window["dt"] >= start_time].copy()
    chosen_parts: list[pd.DataFrame] = []

    if not after_start.empty:
        idx_after = after_start.groupby("storm_id")["dt"].idxmin()
        chosen_parts.append(after_start.loc[idx_after])

    missing_storms = set(window["storm_id"].astype(str).unique())
    if chosen_parts:
        picked_storms = set(chosen_parts[0]["storm_id"].astype(str).unique())
        missing_storms = missing_storms - picked_storms

    if missing_storms:
        fallback = window.loc[window["storm_id"].astype(str).isin(missing_storms)].copy()
        if not fallback.empty:
            fallback["time_diff"] = (fallback["dt"] - start_time).abs()
            idx_fallback = fallback.groupby("storm_id")["time_diff"].idxmin()
            chosen_parts.append(fallback.loc[idx_fallback])

    if not chosen_parts:
        return pd.DataFrame(columns=["storm_id", "init_time", "init_lat", "init_lon"])

    pick = pd.concat(chosen_parts, ignore_index=True)
    pick = pick.drop_duplicates(subset=["storm_id"], keep="first")
    pick = pick.rename(columns={"latitude": "init_lat", "longitude": "init_lon"})
    pick["init_time"] = pick["dt"].values
    cols = [
        "storm_id",
        "init_time",
        "init_lat",
        "init_lon",
        "max_wind_usa",
        "min_pressure_usa",
    ]
    return pick[cols].reset_index(drop=True)


__all__ = [
    "_load_all_points",
    "_load_initial_points",
    "_select_initials_for_time",
    "_select_initials_for_window",
]
