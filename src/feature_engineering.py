"""Computes engineered features from raw AQI/weather data: time-based
features (hour, day, month) and derived features (lags, rolling stats,
AQI change rate).

"""

import logging

import pandas as pd

from src.config import LAG_HOURS, ROLLING_WINDOWS_HOURS, POLLUTANT_FIELDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


LAG_TOLERANCE_FRACTION = 0.2


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add hour, day_of_week, month, is_weekend from the timestamp column."""
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek  # 0=Monday
    df["month"] = ts.dt.month
    df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    return df


def _dedupe_by_city_timestamp(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    before = len(df)
    df = df.drop_duplicates(subset=["city", "timestamp"], keep="last")
    dropped = before - len(df)
    if dropped:
        logger.info(f"Deduped {dropped} repeated (city, timestamp) rows.")
    return df


def _add_lag_for_city(city_df: pd.DataFrame, lag_hours: int, target_col: str = "aqi") -> pd.Series:
    """
    For a single city's time-sorted series, find the value of target_col
    from approximately `lag_hours` hours before each row's timestamp,
    using nearest matching with a tolerance window (handles irregular
    sampling — there usually isn't a reading at exactly t-N).
    """
    city_df = city_df.sort_values("timestamp").reset_index(drop=True)
    lookup_col = f"lag_{lag_hours}h_{target_col}"

    lookup = city_df[["timestamp", target_col]].rename(columns={target_col: lookup_col})
    lookup = lookup.sort_values("timestamp")

    query = city_df[["timestamp"]].copy()
    query["orig_order"] = range(len(query))
    query["search_time"] = query["timestamp"] - pd.Timedelta(hours=lag_hours)
    query_sorted = query.sort_values("search_time").rename(columns={"timestamp": "row_timestamp"})
    query_sorted = query_sorted.rename(columns={"search_time": "timestamp"})

    tolerance = pd.Timedelta(hours=max(lag_hours * LAG_TOLERANCE_FRACTION, 0.25))

    merged = pd.merge_asof(
        query_sorted[["timestamp", "orig_order"]],
        lookup,
        on="timestamp",
        direction="nearest",
        tolerance=tolerance,
    )
    merged = merged.sort_values("orig_order")
    return merged[lookup_col].values


def add_lag_features(df: pd.DataFrame, target_col: str = "aqi") -> pd.DataFrame:
    """Add lag_{N}h_aqi columns for each N in LAG_HOURS, computed per city."""
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    for lag_hours in LAG_HOURS:
        col_name = f"lag_{lag_hours}h_{target_col}"
        df[col_name] = None
        for city in df["city"].unique():
            mask = df["city"] == city
            city_df = df.loc[mask]
            df.loc[mask, col_name] = _add_lag_for_city(city_df, lag_hours, target_col)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = "aqi") -> pd.DataFrame:
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    pieces = []
    for city in df["city"].unique():
        city_df = df[df["city"] == city].sort_values("timestamp").set_index("timestamp")
        for window_hours in ROLLING_WINDOWS_HOURS:
            window_str = f"{window_hours}h"
            city_df[f"rolling_{window_hours}h_mean_{target_col}"] = (
                city_df[target_col].rolling(window_str, min_periods=1).mean()
            )
            city_df[f"rolling_{window_hours}h_std_{target_col}"] = (
                city_df[target_col].rolling(window_str, min_periods=1).std()
            )
        pieces.append(city_df.reset_index())

    return pd.concat(pieces, ignore_index=True)


def add_aqi_change_rate(df: pd.DataFrame, target_col: str = "aqi") -> pd.DataFrame:

    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    df["aqi_change_rate"] = None

    for city in df["city"].unique():
        mask = df["city"] == city
        city_series = df.loc[mask, target_col].astype(float)
        prev = city_series.shift(1)
        change_rate = (city_series - prev) / prev.replace(0, pd.NA)
        df.loc[mask, "aqi_change_rate"] = change_rate.values

    return df


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        logger.warning("engineer_features received an empty DataFrame.")
        return raw_df

    df = _dedupe_by_city_timestamp(raw_df)
    df = add_time_features(df)
    df = add_lag_features(df, target_col="aqi")
    df = add_rolling_features(df, target_col="aqi")
    df = add_aqi_change_rate(df)

    logger.info(f"Feature engineering complete: {len(df)} rows, {len(df.columns)} columns.")
    return df


if __name__ == "__main__":
    from src.data_ingestion import fetch_all_cities_snapshot

    raw = fetch_all_cities_snapshot()
    features = engineer_features(raw)
    pd.set_option("display.max_columns", None)
    print(features.to_string())