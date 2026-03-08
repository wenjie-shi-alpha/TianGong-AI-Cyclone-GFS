"""Google Earth Engine based processing of historical GFS forecasts.

This module keeps every heavy computation inside GEE (reduceRegion /
sampleRectangle) so only storm-level summaries are downloaded locally.
The implementation mirrors the logic from ``initial_tracker`` and
``environment_extractor`` but runs against the `NOAA/GFS0P25`
ImageCollection directly on the server.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence

import ee
import pandas as pd

from initial_tracker.initials import _load_all_points, _select_initials_for_time

DEFAULT_BAND_MAP = {
    "msl": "mean_sea_level_pressure",
    "u10": "u_component_of_wind_10m_above_ground",
    "v10": "v_component_of_wind_10m_above_ground",
    "precipitable_water": "precipitable_water_entire_atmosphere",
    "temp2m": "temperature_2m_above_ground",
    "rh2m": "relative_humidity_2m_above_ground",
}

EE_MAX_PIXELS = 1_000_000_000
KNOTS_TO_MS = 0.514444


def _to_0360(lon: float) -> float:
    wrapped = lon % 360.0
    return wrapped if wrapped >= 0 else wrapped + 360.0


def _to_180(lon: float) -> float:
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    return wrapped


def _hours_to_millis(hours: float) -> int:
    return int(hours * 3600 * 1000)


def _to_utc_iso(ts: pd.Timestamp | str) -> str:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_cycle_prefix(ts: pd.Timestamp | str) -> str:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.strftime("%Y%m%d%H")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class TrackerOptions:
    search_radii_deg: tuple[float, ...] = (5.0, 4.0, 3.0, 2.0, 1.5)
    reduction_scale_m: int = 25000
    annulus_inner_km: float = 550.0
    annulus_outer_km: float = 750.0
    snap_tolerance_pa: float = 150.0
    max_steps: int = 60
    max_consecutive_failures: int = 4
    failure_window_hours: float = 18.0
    wind_core_radius_km: float = 150.0
    lsm_threshold: float = 0.5  # Threshold for Land-Sea Mask (0=Ocean, 1=Land)
    wind_snap_tolerance_ms: float = 0.8


@dataclass
class EnvironmentOptions:
    moisture_radius_km: float = 500.0
    thermodynamic_radius_km: float = 300.0
    summary_precision: int = 2


@dataclass
class PipelineOptions:
    dataset_id: str = "NOAA/GFS0P25"
    analysis_only: bool = True
    max_forecast_hour: int = 120
    time_window_hours: int = 6
    temporal_span_hours: int = 120
    spatial_pad_deg: float = 12.0
    tracker: TrackerOptions = field(default_factory=TrackerOptions)
    environment: EnvironmentOptions = field(default_factory=EnvironmentOptions)


@dataclass
class OutputOptions:
    output_root: Path
    save_csv: bool = True
    save_json: bool = True


class GEERemotePipeline:
    """Coordinator wrapping tracking + environment extraction on top of GEE."""

    def __init__(
        self,
        pipeline_cfg: PipelineOptions,
        output_cfg: OutputOptions,
        *,
        ee_project: str | None = None,
        ee_service_account_key: Path | None = None,
        request_sleep_sec: float = 0.5,
        band_map: dict[str, str] | None = None,
    ) -> None:
        self.cfg = pipeline_cfg
        self.output_cfg = output_cfg
        self.ee_project = ee_project
        self.ee_service_account_key = ee_service_account_key
        self.request_sleep_sec = max(float(request_sleep_sec), 0.0)
        self.request_retry_count = 3
        self.request_retry_base_sec = 1.0
        self.band_map = band_map or DEFAULT_BAND_MAP
        self._image_cache: dict[str, ee.Image] = {}
        self._ensure_ee_initialized()
        self.available_bands = self._discover_bands()
        self.active_band_map = self._build_active_band_map()
        if not any("isobaric" in name for name in self.available_bands):
            logging.warning(
                "Dataset %s does not expose pressure-level bands in GEE; "
                "this pipeline will output tracking + near-surface environmental summaries only.",
                self.cfg.dataset_id,
            )
        # Load Land-Sea Mask (1=Land, 0=Water). MOD44W is an ImageCollection in GEE.
        # Use the newest available image; fallback to all-ocean mask if unavailable.
        try:
            lsm_src = (
                ee.ImageCollection("MODIS/006/MOD44W")
                .select("water_mask")
                .sort("system:time_start", False)
                .first()
            )
            self.lsm_image = ee.Image(lsm_src).unmask(0).Not().rename("lsm")
        except Exception:  # noqa: BLE001
            logging.warning("Failed to load MOD44W mask; fallback to ocean-only mask.")
            self.lsm_image = ee.Image.constant(0).rename("lsm")

    def _initialize_with_service_account(self) -> None:
        key_path = self.ee_service_account_key
        if key_path is None:
            return
        if not key_path.exists():
            raise FileNotFoundError(f"service account key not found: {key_path}")
        info = json.loads(key_path.read_text(encoding="utf-8"))
        client_email = info.get("client_email")
        if not client_email:
            raise RuntimeError(f"service account key missing client_email: {key_path}")
        credentials = ee.ServiceAccountCredentials(client_email, str(key_path))
        if self.ee_project:
            ee.Initialize(credentials=credentials, project=self.ee_project)
        else:
            ee.Initialize(credentials=credentials)

    def _ensure_ee_initialized(self) -> None:
        if self.ee_service_account_key is not None:
            self._initialize_with_service_account()
            return

        try:
            if self.ee_project:
                ee.Initialize(project=self.ee_project)
            else:
                ee.Initialize()
        except Exception as exc:  # noqa: BLE001
            if not sys.stdin.isatty():
                raise RuntimeError(
                    "Earth Engine auth failed in non-interactive mode. "
                    "Provide --service-account-key-json for headless runs."
                ) from exc
            logging.info("Earth Engine authentication required. Launching flow ...")
            ee.Authenticate()
            if self.ee_project:
                ee.Initialize(project=self.ee_project)
            else:
                ee.Initialize()

    def _request_with_retry(
        self,
        fetch_fn: Any,
        *,
        context: str,
        required: bool,
    ) -> Any | None:
        delay = self.request_retry_base_sec
        for attempt in range(1, self.request_retry_count + 2):
            try:
                return fetch_fn()
            except Exception as exc:  # noqa: BLE001
                if attempt > self.request_retry_count:
                    if required:
                        logging.error(
                            "GEE request failed after %d attempts (%s): %s",
                            attempt,
                            context,
                            exc,
                        )
                        raise
                    logging.warning(
                        "GEE request failed after %d attempts (%s): %s",
                        attempt,
                        context,
                        exc,
                    )
                    return None
                logging.warning(
                    "GEE request error (%s), retry %d/%d in %.1fs: %s",
                    context,
                    attempt,
                    self.request_retry_count,
                    delay,
                    exc,
                )
                time.sleep(delay)
                delay = min(delay * 2.0, 8.0)

    def _discover_bands(self) -> set[str]:
        sample = ee.ImageCollection(self.cfg.dataset_id).first()
        if sample is None:
            raise RuntimeError(f"Dataset {self.cfg.dataset_id} returned no images.")
        names = self._request_with_retry(
            lambda: sample.bandNames().getInfo(),
            context=f"discover bands for {self.cfg.dataset_id}",
            required=True,
        )
        if names is None:
            raise RuntimeError(f"Unable to discover bands for dataset {self.cfg.dataset_id}.")
        return set(names)

    def _build_active_band_map(self) -> dict[str, str]:
        active: dict[str, str] = {}
        missing: dict[str, str] = {}
        for key, band in self.band_map.items():
            if band in self.available_bands:
                active[key] = band
            else:
                missing[key] = band
        if "u10" not in active or "v10" not in active:
            raise RuntimeError(
                "GFS collection is missing mandatory bands (u10, v10). "
                f"Available: {sorted(self.available_bands)}"
            )
        if "msl" not in active:
            logging.warning(
                "Band %s is unavailable in %s. Fallback tracking will use low-wind eye localization.",
                self.band_map.get("msl"),
                self.cfg.dataset_id,
            )
        if missing:
            logging.warning("Skipping unavailable bands: %s", ", ".join(missing.values()))
        return active

    def _augment_image(self, image: ee.Image) -> ee.Image:
        cfg = self.cfg
        band_map = self.active_band_map
        forecast_hours = ee.Number(ee.Algorithms.If(image.get("forecast_hours"), image.get("forecast_hours"), 0))
        valid_time = ee.Date(image.get("system:time_start")).advance(forecast_hours, "hour")
        u10 = image.select(band_map["u10"])
        v10 = image.select(band_map["v10"])
        wind = u10.hypot(v10).rename("wind_speed_10m")
        augmented = image.addBands([wind], overwrite=True)
        if "msl" in band_map:
            msl = image.select(band_map["msl"]).rename("msl_pa")
            augmented = augmented.addBands([msl], overwrite=True)
        if "precipitable_water" in band_map:
            augmented = augmented.addBands(
                image.select(band_map["precipitable_water"]).rename("pw_entire"),
                overwrite=True,
            )
        if "temp2m" in band_map:
            augmented = augmented.addBands(image.select(band_map["temp2m"]).rename("temp2m_c"), overwrite=True)
        if "rh2m" in band_map:
            augmented = augmented.addBands(image.select(band_map["rh2m"]).rename("rh2m_pct"), overwrite=True)
        
        # Add LSM
        augmented = augmented.addBands(self.lsm_image, overwrite=True)

        return augmented.set(
            {
                "forecast_hours": forecast_hours,
                "valid_time": valid_time.millis(),
                "valid_time_iso": valid_time.format("YYYY-MM-dd HH:mm"),
                "valid_time_millis": valid_time.millis(),
            }
        )

    def _build_collection(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        center_lat: float,
        center_lon0360: float,
        forecast_init: pd.Timestamp | None = None,
    ) -> ee.ImageCollection:
        pad_buffer_m = self.cfg.spatial_pad_deg * 111_000
        point = ee.Geometry.Point([_to_180(center_lon0360), center_lat])
        region = point.buffer(pad_buffer_m, 1000)
        
        collection = ee.ImageCollection(self.cfg.dataset_id)
        
        if forecast_init:
            # Use cycle prefix (YYYYMMDDHH) on system:index to include all Fxxx
            # members for one forecast initialization (e.g. 2021040912F000..F120).
            cycle_prefix = _to_cycle_prefix(forecast_init)
            collection = collection.filter(ee.Filter.stringStartsWith("system:index", cycle_prefix))
        else:
            # Fallback to window around valid time (approximate)
            collection = collection.filterDate(_to_utc_iso(start), _to_utc_iso(end))
            
        collection = (
            collection
            .filterBounds(region)
            .map(self._augment_image)
        )
        if self.cfg.analysis_only:
            collection = collection.filter(ee.Filter.eq("forecast_hours", 0))
        else:
            collection = collection.filter(ee.Filter.lte("forecast_hours", self.cfg.max_forecast_hour))
        return collection.sort("valid_time_millis")

    def _calc_guess(self, track: list[dict[str, Any]]) -> dict[str, float]:
        if len(track) < 2:
            last = track[-1]
            return {"lat": last["lat"], "lon": last["lon_0360"]}
        last = track[-1]
        prev = track[-2]
        lat = last["lat"] + (last["lat"] - prev["lat"])
        lon = last["lon_0360"] + (last["lon_0360"] - prev["lon_0360"])
        lat = max(min(lat, 90.0), -90.0)
        lon = _to_0360(lon)
        return {"lat": lat, "lon": lon}

    def _square_geometry(self, lat: float, lon360: float, delta_deg: float) -> ee.Geometry:
        radius_m = delta_deg * 111_000
        point = ee.Geometry.Point([_to_180(lon360), lat])
        return point.buffer(radius_m, 1000).bounds()

    def _circle_geometry(self, lat: float, lon360: float, radius_km: float) -> ee.Geometry:
        point = ee.Geometry.Point([_to_180(lon360), lat])
        return point.buffer(radius_km * 1000.0, 1000)

    def _annulus_geometry(self, lat: float, lon360: float, inner_km: float, outer_km: float) -> ee.Geometry:
        point = ee.Geometry.Point([_to_180(lon360), lat])
        outer = point.buffer(outer_km * 1000.0, 1000)
        inner = point.buffer(inner_km * 1000.0, 1000)
        return outer.difference(inner, 1000)

    def _reduce_region(
        self,
        image: ee.Image,
        band_name: str,
        geometry: ee.Geometry,
        reducer: ee.Reducer,
    ) -> dict[str, Any] | None:
        data = self._request_with_retry(
            lambda: image.select(band_name)
            .reduceRegion(
                reducer=reducer,
                geometry=geometry,
                scale=self.cfg.tracker.reduction_scale_m,
                bestEffort=True,
                maxPixels=EE_MAX_PIXELS,
            )
            .getInfo(),
            context=f"reduceRegion:{band_name}",
            required=False,
        )
        return data

    def _search_eye(
        self,
        image: ee.Image,
        guess_lat: float,
        guess_lon0360: float,
    ) -> dict[str, Any] | None:
        tracker_cfg = self.cfg.tracker
        has_msl = "msl" in self.active_band_map
        for delta in tracker_cfg.search_radii_deg:
            bbox = self._square_geometry(guess_lat, guess_lon0360, delta)
            
            # LSM Check: Ensure the search box is over water
            # lsm band: 1=Land, 0=Water. We want max < threshold (i.e. no land)
            lsm_stats = self._reduce_region(image, "lsm", bbox, ee.Reducer.max())
            if lsm_stats and "lsm" in lsm_stats:
                if lsm_stats["lsm"] >= tracker_cfg.lsm_threshold:
                    # Land detected in the search box, skip this radius
                    continue

            center_pa = None
            if has_msl:
                stats = self._reduce_region(image, "msl_pa", bbox, ee.Reducer.min())
                if not stats or "msl_pa" not in stats:
                    continue
                center_pa = float(stats["msl_pa"])
                snap_mask = image.select("msl_pa").lte(center_pa + tracker_cfg.snap_tolerance_pa)
            else:
                wind_min_stats = self._reduce_region(image, "wind_speed_10m", bbox, ee.Reducer.min())
                if not wind_min_stats or "wind_speed_10m" not in wind_min_stats:
                    continue
                min_wind_ms = float(wind_min_stats["wind_speed_10m"])
                snap_mask = image.select("wind_speed_10m").lte(min_wind_ms + tracker_cfg.wind_snap_tolerance_ms)
            lonlat_stats = self._request_with_retry(
                lambda: ee.Image.pixelLonLat()
                .updateMask(snap_mask)
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=bbox,
                    scale=tracker_cfg.reduction_scale_m,
                    bestEffort=True,
                    maxPixels=EE_MAX_PIXELS,
                )
                .getInfo(),
                context="reduceRegion:pixelLonLat",
                required=False,
            )
            if not lonlat_stats or "latitude" not in lonlat_stats or "longitude" not in lonlat_stats:
                continue
            lat = float(lonlat_stats["latitude"])
            lon_geom = float(lonlat_stats["longitude"])
            lon0360 = _to_0360(lon_geom)
            core = self._circle_geometry(lat, lon0360, tracker_cfg.wind_core_radius_km)
            wind_stats = self._reduce_region(image, "wind_speed_10m", core, ee.Reducer.max())
            wind_max = wind_stats.get("wind_speed_10m") if wind_stats else None
            drop_hpa = None
            if has_msl and center_pa is not None:
                annulus = self._annulus_geometry(lat, lon0360, tracker_cfg.annulus_inner_km, tracker_cfg.annulus_outer_km)
                periphery_stats = self._reduce_region(image, "msl_pa", annulus, ee.Reducer.mean())
                if periphery_stats and "msl_pa" in periphery_stats:
                    drop_hpa = max((periphery_stats["msl_pa"] - center_pa) / 100.0, 0.0)
            return {
                "lat": lat,
                "lon_0360": lon0360,
                "lon_-180": lon_geom,
                "msl_pa": center_pa,
                "msl_hpa": (center_pa / 100.0) if center_pa is not None else None,
                "wind_speed_ms": wind_max,
                "pressure_drop_hpa": drop_hpa,
                "search_radius_deg": delta,
            }
        return None

    def _compute_environment(self, image: ee.Image, lat: float, lon0360: float) -> dict[str, Any]:
        env_cfg = self.cfg.environment
        results: dict[str, Any] = {}
        core = self._circle_geometry(lat, lon0360, self.cfg.tracker.wind_core_radius_km)
        if "precipitable_water" in self.active_band_map:
            moisture = self._reduce_region(
                image,
                "pw_entire",
                self._circle_geometry(lat, lon0360, env_cfg.moisture_radius_km),
                ee.Reducer.mean(),
            )
            if moisture and "pw_entire" in moisture:
                results["pw_kgm2"] = moisture["pw_entire"]
        if "temp2m" in self.active_band_map:
            temp = self._reduce_region(
                image,
                "temp2m_c",
                self._circle_geometry(lat, lon0360, env_cfg.thermodynamic_radius_km),
                ee.Reducer.mean(),
            )
            if temp and "temp2m_c" in temp:
                results["temp2m_c"] = temp["temp2m_c"]
        if "rh2m" in self.active_band_map:
            rh = self._reduce_region(
                image,
                "rh2m_pct",
                self._circle_geometry(lat, lon0360, env_cfg.thermodynamic_radius_km),
                ee.Reducer.mean(),
            )
            if rh and "rh2m_pct" in rh:
                results["rh2m_pct"] = rh["rh2m_pct"]
        wind_mean = self._reduce_region(image, "wind_speed_10m", core, ee.Reducer.mean())
        wind_max = self._reduce_region(image, "wind_speed_10m", core, ee.Reducer.max())
        if wind_mean and "wind_speed_10m" in wind_mean:
            results["wind_core_mean_ms"] = wind_mean["wind_speed_10m"]
        if wind_max and "wind_speed_10m" in wind_max:
            results["wind_core_max_ms"] = wind_max["wind_speed_10m"]
        summary_parts: list[str] = []
        if "pw_kgm2" in results:
            summary_parts.append(f"水汽 {results['pw_kgm2']:.{env_cfg.summary_precision}f} kg/m²")
        if "temp2m_c" in results:
            summary_parts.append(f"2米气温 {results['temp2m_c']:.{env_cfg.summary_precision}f} °C")
        if "rh2m_pct" in results:
            summary_parts.append(f"相对湿度 {results['rh2m_pct']:.{env_cfg.summary_precision}f}%")
        if "wind_core_mean_ms" in results:
            summary_parts.append(f"核心平均风 {results['wind_core_mean_ms']:.{env_cfg.summary_precision}f} m/s")
        if summary_parts:
            results["environment_summary"] = "，".join(summary_parts)
        return results

    def _fetch_metadata(self, image: ee.Image) -> dict[str, Any]:
        props = self._request_with_retry(
            lambda: image.toDictionary(
                ["system:index", "valid_time_iso", "valid_time_millis", "forecast_hours"]
            ).getInfo(),
            context="fetch image metadata",
            required=True,
        )
        if props is None:
            raise RuntimeError("Failed to fetch GEE image metadata.")
        return {
            "image_id": props["system:index"],
            "valid_time_iso": props["valid_time_iso"],
            "valid_time_millis": props["valid_time_millis"],
            "forecast_hours": props["forecast_hours"],
        }

    def _track_single_storm(
        self,
        storm_id: str,
        init_row: pd.Series,
        forecast_init: pd.Timestamp | None = None,
    ) -> list[dict[str, Any]]:
        init_time = pd.Timestamp(init_row["init_time"])
        start = init_time - pd.Timedelta(hours=self.cfg.time_window_hours)
        end = init_time + pd.Timedelta(hours=self.cfg.temporal_span_hours)
        init_lat = float(init_row["init_lat"])
        init_lon0360 = _to_0360(float(init_row["init_lon"]))
        collection = self._build_collection(start, end, init_lat, init_lon0360, forecast_init=forecast_init)
        count_obj = self._request_with_retry(
            lambda: collection.size().getInfo(),
            context=f"collection size storm={storm_id}",
            required=True,
        )
        count = int(count_obj or 0)
        if count == 0:
            logging.warning("[%s] No GFS images for requested window.", storm_id)
            return []
        if forecast_init is not None and not self.cfg.analysis_only and count <= 1:
            logging.warning(
                "[%s] Only %d image(s) found for forecast cycle %s; tracking may be truncated.",
                storm_id,
                count,
                pd.Timestamp(forecast_init).strftime("%Y-%m-%d %H:%M"),
            )
        image_list = collection.toList(min(count, self.cfg.tracker.max_steps * 2))
        history: list[dict[str, Any]] = []
        fails = 0
        last_success_time = None
        peak_drop = 0.0
        peak_wind = 0.0
        init_pressure_pa = None
        if pd.notna(init_row.get("min_pressure_usa")):
            init_pressure_pa = float(init_row["min_pressure_usa"]) * 100.0
        init_wind = None
        if pd.notna(init_row.get("max_wind_usa")):
            init_wind = float(init_row["max_wind_usa"]) * KNOTS_TO_MS
        seed_entry = {
            "storm_id": storm_id,
            "time_iso": init_time.strftime("%Y-%m-%d %H:%M"),
            "time_millis": int(init_time.timestamp() * 1000),
            "lat": init_lat,
            "lon_0360": init_lon0360,
            "lon_-180": _to_180(init_lon0360),
            "msl_hpa": init_pressure_pa / 100.0 if init_pressure_pa else None,
            "wind_speed_ms": init_wind,
            "pressure_drop_hpa": None,
            "search_radius_deg": None,
            "image_id": None,
            "forecast_hours": None,
            "snap_success": False,
        }
        history.append(seed_entry)
        for idx in range(min(count, self.cfg.tracker.max_steps)):
            image = ee.Image(image_list.get(idx))
            meta = self._fetch_metadata(image)
            guess = self._calc_guess(history)
            snap = self._search_eye(image, guess["lat"], guess["lon"])
            if snap:
                fails = 0
                last_success_time = meta["valid_time_millis"]
                peak_drop = max(peak_drop, snap.get("pressure_drop_hpa") or 0.0)
                peak_wind = max(peak_wind, snap.get("wind_speed_ms") or 0.0)
                lat = snap["lat"]
                lon0360 = snap["lon_0360"]
            else:
                fails += 1
                lat = guess["lat"]
                lon0360 = guess["lon"]
            entry = {
                "storm_id": storm_id,
                "time_iso": meta["valid_time_iso"],
                "time_millis": meta["valid_time_millis"],
                "lat": lat,
                "lon_0360": lon0360,
                "lon_-180": _to_180(lon0360),
                "msl_hpa": snap.get("msl_hpa") if snap else None,
                "wind_speed_ms": snap.get("wind_speed_ms") if snap else None,
                "pressure_drop_hpa": snap.get("pressure_drop_hpa") if snap else None,
                "search_radius_deg": snap.get("search_radius_deg") if snap else None,
                "image_id": meta["image_id"],
                "forecast_hours": meta["forecast_hours"],
                "snap_success": bool(snap),
            }
            self._image_cache[meta["image_id"]] = image
            if snap:
                env_metrics = self._compute_environment(image, lat, lon0360)
                entry.update(env_metrics)
            history.append(entry)
            if fails >= self.cfg.tracker.max_consecutive_failures:
                logging.info("[%s] stopping after %d consecutive failures.", storm_id, fails)
                break
            if snap and entry["pressure_drop_hpa"] is not None and peak_drop > 0:
                if entry["pressure_drop_hpa"] < max(1.0, 0.25 * peak_drop):
                    logging.info("[%s] pressure drop collapsed below threshold, stopping.", storm_id)
                    break
            if last_success_time and entry["time_millis"] - last_success_time > _hours_to_millis(
                self.cfg.tracker.failure_window_hours
            ):
                logging.info("[%s] no successful snaps for %.1f hours, stopping.", storm_id, self.cfg.tracker.failure_window_hours)
                break
        return history

    def _write_outputs(self, storm_id: str, history: list[dict[str, Any]]) -> None:
        out_dir = self.output_cfg.output_root / storm_id
        _ensure_dir(out_dir)
        if self.output_cfg.save_csv:
            df = pd.json_normalize(history)
            csv_path = out_dir / f"{storm_id}_gee_track.csv"
            df.to_csv(csv_path, index=False)
            logging.info("[%s] track saved to %s", storm_id, csv_path)
        if self.output_cfg.save_json:
            json_path = out_dir / f"{storm_id}_gee_track.json"
            with json_path.open("w", encoding="utf-8") as fp:
                json.dump(history, fp, ensure_ascii=False, indent=2)
            logging.info("[%s] JSON summary saved to %s", storm_id, json_path)

    def _export_analysis_only(self, storm_id: str, history: list[dict[str, Any]]) -> None:
        """Persist only analyzed results; no raw downloads (GEE-only workflow)."""
        if not history:
            return
        # Track CSV/JSON already written by _write_outputs; this is a hook for future summaries.
        logging.info("[%s] Skipping raw downloads; analysis kept on server.", storm_id)

    def run(
        self,
        initials_csv: Path,
        start_time: str | None = None,
        storm_filter: list[str] | None = None,
    ) -> None:
        df_all = _load_all_points(initials_csv)

        if storm_filter:
            df_all = df_all[df_all["storm_id"].isin(storm_filter)]

        if start_time:
            target_time = pd.to_datetime(start_time, utc=True).tz_convert(None)
            forecast_id = target_time.strftime("%Y%m%d_%H%M")
            logging.info("Processing single forecast cycle %s UTC ...", target_time.strftime("%Y-%m-%d %H:%M"))
            for storm_id, group in df_all.groupby("storm_id"):
                candidates = _select_initials_for_time(group, target_time, tol_hours=max(self.cfg.time_window_hours, 1))
                if candidates.empty:
                    continue
                row = candidates.iloc[0]
                logging.info("Processing storm %s @ cycle %s", storm_id, target_time.strftime("%Y-%m-%d %H:%M"))
                history = self._track_single_storm(str(storm_id), row, forecast_init=target_time)
                if not history:
                    continue
                output_id = f"{storm_id}_{forecast_id}"
                self._write_outputs(output_id, history)
                self._export_analysis_only(output_id, history)
                if self.request_sleep_sec > 0:
                    time.sleep(self.request_sleep_sec)
            return

        for storm_id, group in df_all.groupby("storm_id"):
            logging.info("Processing storm %s ...", storm_id)

            storm_start = pd.to_datetime(group["dt"]).min()
            storm_end = pd.to_datetime(group["dt"]).max()

            # Align forecast cycles to 6-hourly synoptic times, mirroring the notebook logic.
            freq = "6h"
            init_times = pd.date_range(
                storm_start.floor(freq),
                storm_end.ceil(freq),
                freq=freq,
            )

            for init_time in init_times:
                init_str = init_time.strftime("%Y-%m-%d %H:%M")
                forecast_id = init_time.strftime("%Y%m%d_%H%M")

                tracking_start = max(init_time, storm_start)

                candidates = _select_initials_for_time(group, tracking_start, tol_hours=3)
                if candidates.empty:
                    continue

                row = candidates.iloc[0]
                logging.info("  Forecast %s (Start Track: %s)", init_str, tracking_start)

                history = self._track_single_storm(str(storm_id), row, forecast_init=init_time)
                if not history:
                    continue

                self._write_outputs(f"{storm_id}_{forecast_id}", history)

                # Only persist analyzed results; no raw downloads.
                self._export_analysis_only(f"{storm_id}_{forecast_id}", history)

                if self.request_sleep_sec > 0:
                    time.sleep(self.request_sleep_sec)



def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track tropical cyclones with the TianGong algorithms directly on Google Earth Engine."
    )
    parser.add_argument(
        "--initials-csv",
        type=Path,
        default=Path("input/matched_cyclone_tracks.csv"),
        help="CSV file containing catalogue points (default: input/matched_cyclone_tracks.csv)",
    )
    parser.add_argument(
        "--storm-id",
        nargs="*",
        help="Optional list of storm IDs to process (default: all storms in the selected window).",
    )
    parser.add_argument(
        "--start-time",
        required=False,
        help="Reference datetime (UTC) used to pick the initial seeds. If omitted, processes all storms in CSV.",
    )
    parser.add_argument(
        "--time-window-hours",
        type=int,
        default=6,
        help="Tolerance window (hours) when matching seeds (default: 6).",
    )
    parser.add_argument(
        "--temporal-span-hours",
        type=int,
        default=120,
        help="How long to continue tracking after the initial time (default: 120 hours).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=40,
        help="Maximum number of forecast frames to inspect per storm (default: 40).",
    )
    parser.add_argument(
        "--max-forecast-hour",
        type=int,
        default=120,
        help="Maximum forecast lead hour to keep when analysis_only is False (default: 120).",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Use only analysis frames (forecast_hour == 0). If omitted, a full forecast cycle is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("colab_outputs/gee_pipeline"),
        help="Directory where processed tracks will be stored (default: colab_outputs/gee_pipeline).",
    )
    parser.add_argument(
        "--ee-project",
        "--project",
        dest="ee_project",
        default=None,
        help="Optional Earth Engine project ID (needed when your account is tied to a GCP project).",
    )
    parser.add_argument(
        "--service-account-key-json",
        type=Path,
        default=None,
        help="Optional service account key JSON for headless Earth Engine auth.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.5,
        help="Sleep seconds between forecast jobs to reduce GEE throttling.",
    )
    parser.add_argument(
        "--spatial-pad-deg",
        type=float,
        default=12.0,
        help="Half-width of the bounding area (degrees) used when filtering the ImageCollection.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tracker_cfg = TrackerOptions(
        search_radii_deg=(5.0, 4.0, 3.0, 2.0, 1.5),
        reduction_scale_m=25000,
        annulus_inner_km=550.0,
        annulus_outer_km=750.0,
        snap_tolerance_pa=150.0,
        max_steps=args.max_steps,
        max_consecutive_failures=4,
        failure_window_hours=18.0,
        wind_core_radius_km=150.0,
    )
    env_cfg = EnvironmentOptions(moisture_radius_km=500.0, thermodynamic_radius_km=300.0)
    pipeline_cfg = PipelineOptions(
        dataset_id="NOAA/GFS0P25",
        analysis_only=args.analysis_only,
        max_forecast_hour=args.max_forecast_hour,
        time_window_hours=args.time_window_hours,
        temporal_span_hours=args.temporal_span_hours,
        spatial_pad_deg=args.spatial_pad_deg,
        tracker=tracker_cfg,
        environment=env_cfg,
    )
    output_cfg = OutputOptions(output_root=args.output_dir)
    pipeline = GEERemotePipeline(
        pipeline_cfg,
        output_cfg,
        ee_project=args.ee_project,
        ee_service_account_key=args.service_account_key_json,
        request_sleep_sec=args.sleep_sec,
    )
    pipeline.run(args.initials_csv, start_time=args.start_time, storm_filter=args.storm_id)


if __name__ == "__main__":
    main()
