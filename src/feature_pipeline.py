"""Orchestrates: fetch raw data -> engineer features -> write to
Hopsworks Feature Store. This is the script the hourly GitHub Actions
workflow runs.

Design: we maintain TWO feature groups —
  1. `aqi_raw` — every raw snapshot ever fetched, deduped by
     (city, timestamp). This is the source of truth / audit trail.
  2. `aqi_features` — engineered features, fully recomputed from the
     complete raw history on every run.

Recomputing engineered features from full history each run (rather
than incrementally) guarantees lag/rolling features are always
correct, at the cost of a bit of redundant compute — an acceptable
tradeoff at this data volume (a handful of cities, hourly cadence).
"""

import logging

import hopsworks
import pandas as pd

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_GROUP_PRIMARY_KEY,
)
from src.data_ingestion import fetch_all_cities_snapshot
from src.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_FEATURE_GROUP_NAME = "aqi_raw"
RAW_FEATURE_GROUP_VERSION = 1

import socket
import streamlit as st

try:
    ip = socket.gethostbyname("c.app.hopsworks.ai")
    st.success(f"Hopsworks DNS works: {ip}")
except Exception as e:
    st.error(f"Hopsworks DNS failed: {e}")

def get_hopsworks_project():
    """Log in to Hopsworks and return the Project handle (needed for
    both the Feature Store and the Model Registry)."""
    return hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)


def get_feature_store(project=None):
    """Return the Feature Store handle. Pass an existing `project` to
    avoid logging in twice; otherwise logs in fresh."""
    if project is None:
        project = get_hopsworks_project()
    return project.get_feature_store()


def _prepare_for_hopsworks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dtypes before writing to Hopsworks: convert timezone-aware
    timestamps to naive UTC (Hopsworks' online store doesn't reliably
    handle tz-aware datetimes), and ensure object columns with all-None
    values don't break schema inference.
    """
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    # Hopsworks needs consistent numeric dtypes; force lag/rolling/raw
    # numeric columns to float64 explicitly. pd.to_numeric() alone can
    # infer int64 when every value in a batch happens to be a whole
    # number (e.g. a single snapshot with no decimals), which then
    # mismatches a feature group schema already registered as 'double'
    # from an earlier batch that did have decimals. Always forcing
    # float64 avoids that inconsistency regardless of batch content.
    numeric_like_prefixes = ("lag_", "rolling_", "aqi_change_rate")
    for col in df.columns:
        if col.startswith(numeric_like_prefixes) or col in (
            "aqi", "pm25", "pm10", "no2", "o3", "so2", "co",
            "temp", "humidity", "pressure", "wind_speed",
            "aqicn_temp", "aqicn_humidity", "aqicn_pressure", "aqicn_wind",
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    return df


def get_or_create_raw_feature_group(fs):
    return fs.get_or_create_feature_group(
        name=RAW_FEATURE_GROUP_NAME,
        version=RAW_FEATURE_GROUP_VERSION,
        primary_key=FEATURE_GROUP_PRIMARY_KEY,
        event_time="timestamp",
        description="Raw AQI + weather snapshots per city, deduped by (city, timestamp).",
        online_enabled=False,
        time_travel_format="HUDI",
    )


def get_or_create_engineered_feature_group(fs):
    return fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=FEATURE_GROUP_PRIMARY_KEY,
        event_time="timestamp",
        description="Engineered AQI forecasting features: time, lag, rolling, change-rate.",
        online_enabled=False,
        time_travel_format="HUDI",
    )


def run_feature_pipeline():
    """
    Full hourly run:
      1. Fetch current snapshot for all cities.
      2. Read existing raw history from Hopsworks (if any).
      3. Merge + dedupe, write updated raw history back.
      4. Recompute engineered features on the full raw history.
      5. Write engineered features to the engineered feature group.
    """
    fs = get_feature_store()
    raw_fg = get_or_create_raw_feature_group(fs)
    engineered_fg = get_or_create_engineered_feature_group(fs)

    logger.info("Fetching new snapshot for all cities...")
    new_raw = fetch_all_cities_snapshot()
    if new_raw.empty:
        logger.error("No data fetched this run — aborting to avoid writing an empty update.")
        return

    logger.info("Reading existing raw history from Hopsworks...")
    try:
        existing_raw = raw_fg.read()
    except Exception as e:
        logger.warning(f"Could not read existing raw history (likely first run): {e}")
        existing_raw = pd.DataFrame()

    if not existing_raw.empty:
        combined_raw = pd.concat([existing_raw, new_raw], ignore_index=True)
    else:
        combined_raw = new_raw

    combined_raw["timestamp"] = pd.to_datetime(combined_raw["timestamp"], utc=True, errors="coerce")
    combined_raw = combined_raw.drop_duplicates(subset=["city", "timestamp"], keep="last")

    logger.info(f"Writing {len(new_raw)} new row(s) to raw feature group (raw table now has {len(combined_raw)} total rows)...")
    raw_fg.insert(_prepare_for_hopsworks(new_raw))

    logger.info("Recomputing engineered features on full raw history...")
    engineered = engineer_features(combined_raw)

    logger.info(f"Writing {len(engineered)} engineered rows to feature group...")
    engineered_fg.insert(_prepare_for_hopsworks(engineered))

    logger.info("Feature pipeline run complete.")


if __name__ == "__main__":
    run_feature_pipeline()