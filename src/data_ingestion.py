"""Fetches raw weather and pollutant data from AQICN and OpenWeather APIs."""

import time
import logging
from datetime import datetime, timezone

import requests
import pandas as pd
from dateutil import parser as date_parser

from src.config import (
    AQICN_API_KEY,
    AQICN_BASE_URL,
    OPENWEATHER_API_KEY,
    OPENWEATHER_BASE_URL,
    CITIES,
)

STALE_THRESHOLD_HOURS = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Verified correct AQICN stations for cities where the matched station
# name doesn't literally contain the city name (checked manually against
# aqicn.org station pages) — avoids false-positive mismatch warnings.
KNOWN_GOOD_STATIONS = {
    "quetta": ["irrigation directorate"],
    "faisalabad": ["fda-01"],
}


def _resolve_aqi(raw_aqi, pm25_value):
    """
    AQICN sometimes returns aqi='-' when it hasn't computed a composite
    index but individual pollutant sub-indices (like pm25) are present.
    Fall back to the PM2.5 reading as the best available proxy for AQI.
    """
    if raw_aqi not in (None, "-", ""):
        try:
            return float(raw_aqi)
        except (ValueError, TypeError):
            pass
    if pm25_value is not None:
        try:
            return float(pm25_value)
        except (ValueError, TypeError):
            pass
    return None


def fetch_current_aqi(city_name: str) -> dict | None:
    """
    Fetch current AQI + pollutant breakdown for a single city from AQICN,
    using the configured station slug/ID for that city.

    Returns a flat dict of fields, or None if the request failed / no data.
    """
    station = CITIES[city_name]["station"]
    url = f"{AQICN_BASE_URL}/{station}/"
    params = {"token": AQICN_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as e:
        logger.error(f"[{city_name}] AQICN request failed: {e}")
        return None

    if payload.get("status") != "ok":
        logger.warning(f"[{city_name}] AQICN returned non-ok status: {payload}")
        return None

    data = payload["data"]
    iaqi = data.get("iaqi", {})

    record = {
        "city": city_name,
        "matched_station": data.get("city", {}).get("name"),
        "timestamp": data.get("time", {}).get("iso", datetime.now(timezone.utc).isoformat()),
        "aqi": _resolve_aqi(data.get("aqi"), iaqi.get("pm25", {}).get("v")),
        "dominant_pollutant": data.get("dominentpol"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        # AQICN sometimes includes weather readings inline too (t, h, p, w)
        "aqicn_temp": iaqi.get("t", {}).get("v"),
        "aqicn_humidity": iaqi.get("h", {}).get("v"),
        "aqicn_pressure": iaqi.get("p", {}).get("v"),
        "aqicn_wind": iaqi.get("w", {}).get("v"),
    }
    return record


def fetch_current_weather(city_name: str) -> dict | None:
    """
    Fetch current weather (temp, humidity, wind, pressure) for a city
    from OpenWeather, using lat/lon (more reliable than city name matching).
    """
    coords = CITIES[city_name]
    params = {
        "lat": coords["lat"],
        "lon": coords["lon"],
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        resp = requests.get(f"{OPENWEATHER_BASE_URL}/weather", params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as e:
        logger.error(f"[{city_name}] OpenWeather request failed: {e}")
        return None

    return {
        "city": city_name,
        "temp": payload.get("main", {}).get("temp"),
        "humidity": payload.get("main", {}).get("humidity"),
        "pressure": payload.get("main", {}).get("pressure"),
        "wind_speed": payload.get("wind", {}).get("speed"),
        "wind_deg": payload.get("wind", {}).get("deg"),
    }


def fetch_current_snapshot(city_name: str) -> dict | None:
    """
    Merge AQICN pollutant data + OpenWeather weather data into a single
    record for one city at the current moment. This is the function the
    hourly feature pipeline will call for each city.
    """
    aqi_record = fetch_current_aqi(city_name)
    if aqi_record is None:
        return None

    weather_record = fetch_current_weather(city_name)

    merged = dict(aqi_record)
    if weather_record:
        # Prefer OpenWeather's dedicated weather fields; fall back to
        # AQICN's inline weather readings if OpenWeather call failed.
        merged["temp"] = weather_record.get("temp") if weather_record.get("temp") is not None else aqi_record.get("aqicn_temp")
        merged["humidity"] = weather_record.get("humidity") if weather_record.get("humidity") is not None else aqi_record.get("aqicn_humidity")
        merged["pressure"] = weather_record.get("pressure") if weather_record.get("pressure") is not None else aqi_record.get("aqicn_pressure")
        merged["wind_speed"] = weather_record.get("wind_speed") if weather_record.get("wind_speed") is not None else aqi_record.get("aqicn_wind")
    else:
        merged["temp"] = aqi_record.get("aqicn_temp")
        merged["humidity"] = aqi_record.get("aqicn_humidity")
        merged["pressure"] = aqi_record.get("aqicn_pressure")
        merged["wind_speed"] = aqi_record.get("aqicn_wind")

    # Flag stale readings: AQICN stations sometimes return a cached
    # last-known reading rather than a fresh one. We keep the row (useful
    # for training) but mark it so downstream steps can filter/weight it.
    merged["is_stale"] = _is_reading_stale(merged.get("timestamp"))
    if merged["is_stale"]:
        logger.warning(f"[{city_name}] Stale reading detected (timestamp: {merged.get('timestamp')}).")

    matched = (merged.get("matched_station") or "").lower()
    city_key = city_name.lower()
    is_known_good = any(alias in matched for alias in KNOWN_GOOD_STATIONS.get(city_key, []))
    if city_key not in matched and not is_known_good:
        logger.warning(
            f"[{city_name}] Matched station name ('{merged.get('matched_station')}') "
            f"does not mention the expected city — verify this isn't a mismatched station."
        )

    return merged


def _is_reading_stale(timestamp_str: str | None) -> bool:
    """
    Return True if the AQICN reading's timestamp is older than
    STALE_THRESHOLD_HOURS compared to now.
    """
    if not timestamp_str:
        return True  # missing timestamp is treated as stale/untrustworthy

    try:
        reading_time = date_parser.parse(timestamp_str)
        if reading_time.tzinfo is None:
            reading_time = reading_time.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - reading_time).total_seconds() / 3600
        return age_hours > STALE_THRESHOLD_HOURS
    except (ValueError, TypeError):
        return True


def fetch_historical_pollution(city_name: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Fetch historical hourly pollutant concentrations for a city from
    OpenWeather's Air Pollution History API (free tier, hourly
    resolution, available back to 2020-11-27).

    Note: this returns raw pollutant concentrations (µg/m³), not a
    pre-computed AQI — see src/aqi_utils.py for how we derive AQI from
    the PM2.5 value to stay consistent with AQICN's live AQI scale.
    """
    coords = CITIES[city_name]
    params = {
        "lat": coords["lat"],
        "lon": coords["lon"],
        "start": int(start_dt.timestamp()),
        "end": int(end_dt.timestamp()),
        "appid": OPENWEATHER_API_KEY,
    }

    try:
        resp = requests.get(f"{OPENWEATHER_BASE_URL}/air_pollution/history", params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as e:
        logger.error(f"[{city_name}] OpenWeather historical pollution request failed: {e}")
        return pd.DataFrame()

    entries = payload.get("list", [])
    if not entries:
        logger.warning(f"[{city_name}] No historical pollution entries returned for this range.")
        return pd.DataFrame()

    records = []
    for entry in entries:
        components = entry.get("components", {})
        records.append({
            "city": city_name,
            "timestamp": datetime.fromtimestamp(entry["dt"], tz=timezone.utc).isoformat(),
            "pm25": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "co": components.get("co"),
        })

    df = pd.DataFrame(records)
    logger.info(f"[{city_name}] Fetched {len(df)} historical pollution records.")
    return df


def fetch_all_cities_snapshot(sleep_between_calls: float = 1.0) -> pd.DataFrame:
    """
    Fetch current snapshot for all configured cities and return as a DataFrame.
    Used by the hourly feature pipeline.
    """
    records = []
    for city_name in CITIES:
        record = fetch_current_snapshot(city_name)
        if record:
            records.append(record)
        else:
            logger.warning(f"[{city_name}] Skipped — no data returned this cycle.")
        time.sleep(sleep_between_calls)  # be polite to free-tier rate limits

    df = pd.DataFrame(records)
    logger.info(f"Fetched snapshot for {len(df)}/{len(CITIES)} cities.")
    return df


if __name__ == "__main__":
    # Quick manual test: run `python -m src.data_ingestion` from repo root
    # after setting AQICN_API_KEY and OPENWEATHER_API_KEY in your .env
    df = fetch_all_cities_snapshot()
    print(df.to_string())