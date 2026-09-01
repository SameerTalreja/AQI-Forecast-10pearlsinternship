"""Backfills historical data so we have enough history for real model
training — run once (or occasionally) rather than on the hourly cron.

Uses OpenWeather's Air Pollution History API (free, hourly resolution,
available back to 2020-11-27) for pollutant concentrations, since
AQICN's free tier doesn't expose deep historical data. AQI is derived
from the historical PM2.5 concentration via src/aqi_utils.py, keeping
it on the same 0-500 scale as AQICN's live AQI.

Known limitation (documented for the write-up): historical weather
fields (temp/humidity/wind/pressure) are not backfilled here, since
OpenWeather's historical weather API requires a paid subscription tier.
Backfilled rows will have NaN weather fields; only live pipeline rows
(going forward) have real weather. This is a reasonable scope cut for
project timeline — noted as a "future work" item.
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import CITIES
from src.data_ingestion import fetch_historical_pollution
from src.aqi_utils import compute_aqi_from_pm25
from src.feature_engineering import engineer_features
from src.feature_pipeline import (
    get_feature_store,
    get_or_create_raw_feature_group,
    get_or_create_engineered_feature_group,
    _prepare_for_hopsworks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def backfill_city(city_name: str, days_back: int) -> pd.DataFrame:
    """Fetch and prepare historical data for a single city."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)

    df = fetch_historical_pollution(city_name, start_dt, end_dt)
    if df.empty:
        return df

    df["aqi"] = df["pm25"].apply(compute_aqi_from_pm25)
    df["dominant_pollutant"] = "pm25"  # approximation: we only compute AQI from PM2.5
    df["matched_station"] = f"{city_name} (OpenWeather historical)"
    df["is_stale"] = False  # historical rows are treated as valid-for-their-time, not stale

    # Weather fields not available historically on the free tier — left
    # as NaN here; see module docstring.
    for col in ["temp", "humidity", "pressure", "wind_speed"]:
        df[col] = None
    for col in ["aqicn_temp", "aqicn_humidity", "aqicn_pressure", "aqicn_wind"]:
        df[col] = None

    return df


def run_backfill(days_back: int = 30):
    """
    Backfill historical data for all configured cities, engineer
    features on the combined history, and write to Hopsworks.
    """
    logger.info(f"Starting backfill for {len(CITIES)} cities, {days_back} days back...")

    all_records = []
    for city_name in CITIES:
        logger.info(f"Backfilling {city_name}...")
        city_df = backfill_city(city_name, days_back)
        if city_df.empty:
            logger.warning(f"[{city_name}] No historical data returned — skipping.")
            continue
        all_records.append(city_df)

    if not all_records:
        logger.error("No historical data fetched for any city — aborting backfill.")
        return

    raw_df = pd.concat(all_records, ignore_index=True)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True, errors="coerce")
    raw_df = raw_df.dropna(subset=["timestamp", "aqi"])
    raw_df = raw_df.drop_duplicates(subset=["city", "timestamp"], keep="last")
    logger.info(f"Total backfilled rows across all cities: {len(raw_df)}")

    fs = get_feature_store()
    raw_fg = get_or_create_raw_feature_group(fs)
    engineered_fg = get_or_create_engineered_feature_group(fs)

    logger.info("Writing raw backfill data to Hopsworks...")
    raw_fg.insert(_prepare_for_hopsworks(raw_df))

    logger.info("Engineering features on backfilled history...")
    engineered = engineer_features(raw_df)

    logger.info(f"Writing {len(engineered)} engineered rows to Hopsworks...")
    engineered_fg.insert(_prepare_for_hopsworks(engineered))

    logger.info("Backfill complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical AQI data into Hopsworks.")
    parser.add_argument(
        "--days", type=int, default=30,
        help="How many days of history to backfill (default: 30).",
    )
    args = parser.parse_args()
    run_backfill(days_back=args.days)