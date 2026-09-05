"""Exploratory data analysis over the full raw AQI history across all
cities — city comparisons, time trends, daily/weekly patterns, and a
data-quality summary (which stations are actually reporting fresh
data, since we found real gaps in this during the project).

Functions here are pure pandas logic, deliberately separated from the
Hopsworks-loading step so they can be tested without a live connection.
"""

import logging

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Common cleanup: parse timestamp, coerce aqi to numeric, dedupe."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    df = df.dropna(subset=["timestamp", "aqi", "city"])
    df = df.drop_duplicates(subset=["city", "timestamp"], keep="last")
    return df


def city_summary_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Per-city AQI summary: mean, median, min, max, std, reading count."""
    df = _prep(raw_df)
    if df.empty:
        return pd.DataFrame(columns=["city", "mean", "median", "min", "max", "std", "count"])

    summary = df.groupby("city")["aqi"].agg(
        mean="mean", median="median", min="min", max="max", std="std", count="count",
    ).reset_index()
    return summary.sort_values("mean", ascending=False).reset_index(drop=True)


def daily_trend(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Per-city daily mean AQI over time — for a multi-line trend chart."""
    df = _prep(raw_df)
    if df.empty:
        return pd.DataFrame(columns=["city", "day", "aqi"])

    df["day"] = df["timestamp"].dt.normalize()
    daily = df.groupby(["city", "day"], as_index=False)["aqi"].mean()
    return daily.sort_values(["city", "day"]).reset_index(drop=True)


def hourly_pattern(raw_df: pd.DataFrame, city: str | None = None) -> pd.DataFrame:
    """
    Mean AQI by hour of day (0-23) — reveals whether air quality is
    worse at particular times, e.g. traffic hours or nighttime
    inversion. Filters to one city if given, otherwise aggregates
    across all cities.
    """
    df = _prep(raw_df)
    if city:
        df = df[df["city"] == city]
    if df.empty:
        return pd.DataFrame(columns=["hour", "aqi"])

    df["hour"] = df["timestamp"].dt.hour
    pattern = df.groupby("hour", as_index=False)["aqi"].mean()
    # Ensure all 24 hours are present even if some have no data (avoids
    # a chart with gaps that could be misread as "zero AQI" at that hour)
    full_hours = pd.DataFrame({"hour": range(24)})
    pattern = full_hours.merge(pattern, on="hour", how="left")
    return pattern


def weekday_pattern(raw_df: pd.DataFrame, city: str | None = None) -> pd.DataFrame:
    """Mean AQI by day of week (0=Monday..6=Sunday)."""
    df = _prep(raw_df)
    if city:
        df = df[df["city"] == city]
    if df.empty:
        return pd.DataFrame(columns=["day_of_week", "aqi"])

    df["day_of_week"] = df["timestamp"].dt.dayofweek
    pattern = df.groupby("day_of_week", as_index=False)["aqi"].mean()
    full_days = pd.DataFrame({"day_of_week": range(7)})
    pattern = full_days.merge(pattern, on="day_of_week", how="left")
    return pattern


def data_freshness_summary(raw_df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    For each city: how many readings we have, when the latest one is,
    and how many hours old that latest reading is right now. This
    surfaces the real station-freshness gaps we found during the
    project (some cities update near-hourly, others every few days) —
    an honest, useful data-quality view rather than hiding the issue.
    """
    df = _prep(raw_df)
    if now is None:
        now = pd.Timestamp.now(tz="UTC")

    if df.empty:
        return pd.DataFrame(columns=["city", "reading_count", "latest_reading", "hours_since_latest"])

    summary = df.groupby("city").agg(
        reading_count=("timestamp", "count"),
        latest_reading=("timestamp", "max"),
    ).reset_index()
    summary["hours_since_latest"] = (now - summary["latest_reading"]).dt.total_seconds() / 3600
    return summary.sort_values("hours_since_latest").reset_index(drop=True)


# ---------------------------------------------------------------------
# Hopsworks-dependent loading (kept thin and separate)
# ---------------------------------------------------------------------

def load_all_history(fs) -> pd.DataFrame:
    """Read the full raw AQI history across all cities from Hopsworks."""
    from src.feature_pipeline import RAW_FEATURE_GROUP_NAME, RAW_FEATURE_GROUP_VERSION

    raw_fg = fs.get_feature_group(name=RAW_FEATURE_GROUP_NAME, version=RAW_FEATURE_GROUP_VERSION)
    df = raw_fg.read()
    logger.info(f"Loaded {len(df)} raw rows across {df['city'].nunique()} cities for EDA.")
    return df


if __name__ == "__main__":
    # Quick manual test: print the summary tables using live data.
    # Run from repo root: python -m src.eda
    from src.feature_pipeline import get_hopsworks_project, get_feature_store

    project = get_hopsworks_project()
    fs = get_feature_store(project)
    raw_df = load_all_history(fs)

    print("\n=== City summary ===")
    print(city_summary_stats(raw_df).to_string())

    print("\n=== Data freshness ===")
    print(data_freshness_summary(raw_df).to_string())

    print("\n=== Hourly pattern (all cities) ===")
    print(hourly_pattern(raw_df).to_string())