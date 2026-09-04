"""Loads the best-performing registered model and generates a 72-hour
(3-day) AQI forecast per city.

Core design — recursive forecasting:
Our models are trained as single-step predictors (predict AQI "now"
from lag/rolling features computed from the past). To project 72 hours
ahead, we predict hour+1, append that prediction to the city's history,
then use it (along with real history) to compute the lag/rolling
features needed to predict hour+2, and so on. Error compounds across
this chain — the model's single-step RMSE (measured in
training_pipeline.py) is NOT the same as its 72-hour-ahead accuracy.
This is a known, documented limitation of the recursive approach and
worth calling out explicitly in the project write-up.
"""

import json
import logging
import os
import shutil
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CITIES, LAG_HOURS, ROLLING_WINDOWS_HOURS, TARGET_COLUMN,
    HAZARDOUS_AQI_THRESHOLD, FORECAST_HOURS_AHEAD,
)
from src.feature_pipeline import get_hopsworks_project, get_feature_store, RAW_FEATURE_GROUP_NAME, RAW_FEATURE_GROUP_VERSION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAMES = ["aqi_ridge", "aqi_random_forest", "aqi_xgboost"]
DOWNLOAD_DIR = "models/_inference_download"

# How far back (hours) to seed the recursive forecast's starting history
# — needs to cover the largest lag/rolling window (24h) with margin.
HISTORY_SEED_HOURS = 72


def aqi_to_category(aqi: float) -> str:
    """Standard US EPA AQI category labels."""
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


# ---------------------------------------------------------------------
# Core recursive forecasting logic — no Hopsworks dependency, so it can
# be unit tested with a plain scikit-learn-style model and synthetic
# history.
# ---------------------------------------------------------------------

def _lookup_lag(history: dict, target_time: pd.Timestamp, lag_hours: int, tolerance_hours: float):
    """Find the history value nearest to (target_time - lag_hours),
    within tolerance_hours. `history` is {pd.Timestamp: aqi_value}."""
    search_time = target_time - timedelta(hours=lag_hours)
    best_ts, best_diff = None, None
    for ts, val in history.items():
        diff = abs((ts - search_time).total_seconds()) / 3600
        if diff <= tolerance_hours and (best_diff is None or diff < best_diff):
            best_ts, best_diff = ts, diff
    return history[best_ts] if best_ts is not None else None


def _rolling_stats(history: dict, target_time: pd.Timestamp, window_hours: int):
    """Mean/std of all history values within window_hours before target_time."""
    window_start = target_time - timedelta(hours=window_hours)
    values = [v for ts, v in history.items() if window_start < ts <= target_time]
    if not values:
        return None, None
    mean = float(np.mean(values))
    std = float(np.std(values)) if len(values) > 1 else None
    return mean, std


def recursive_forecast(
    model, medians: dict, feature_columns: list, city_name: str,
    history: dict, last_timestamp: pd.Timestamp,
    horizon_hours: int = FORECAST_HOURS_AHEAD,
) -> pd.DataFrame:
    """
    Generate `horizon_hours` of forward predictions for one city.

    Args:
        model: fitted model with .predict(X)
        medians: dict of column -> fallback value (from training)
        feature_columns: exact column order the model expects
        city_name: for the city_* one-hot columns
        history: {pd.Timestamp: aqi_value} seed history (mutated/extended in place)
        last_timestamp: the most recent known real timestamp
        horizon_hours: how many hours ahead to forecast

    Returns: DataFrame with columns [city, timestamp, predicted_aqi, aqi_category, is_hazardous]
    """
    history = dict(history)  # don't mutate caller's copy
    results = []

    for step in range(1, horizon_hours + 1):
        forecast_time = last_timestamp + timedelta(hours=step)

        row = {}
        row["hour"] = forecast_time.hour
        row["day_of_week"] = forecast_time.dayofweek
        row["month"] = forecast_time.month
        row["is_weekend"] = int(forecast_time.dayofweek in [5, 6])
        row["is_stale"] = 0

        for lag_hours in LAG_HOURS:
            tolerance = max(lag_hours * 0.2, 0.25)
            val = _lookup_lag(history, forecast_time, lag_hours, tolerance)
            row[f"lag_{lag_hours}h_aqi"] = val

        for window_hours in ROLLING_WINDOWS_HOURS:
            mean, std = _rolling_stats(history, forecast_time, window_hours)
            row[f"rolling_{window_hours}h_mean_aqi"] = mean
            row[f"rolling_{window_hours}h_std_aqi"] = std

        # aqi_change_rate: change from the most recent known/predicted point
        latest_ts = max(history.keys())
        latest_val = history[latest_ts]
        prev_lag = row.get("lag_1h_aqi")
        if prev_lag not in (None, 0):
            row["aqi_change_rate"] = (latest_val - prev_lag) / prev_lag
        else:
            row["aqi_change_rate"] = None

        for city in CITIES:
            row[f"city_{city}"] = 1 if city == city_name else 0

        # Assemble into the exact column order the model expects,
        # filling any missing/unseen columns with training medians.
        X_row = {}
        for col in feature_columns:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = medians.get(col, 0)
            X_row[col] = val
        X = pd.DataFrame([X_row])[feature_columns]

        pred = float(model.predict(X)[0])
        pred = max(pred, 0.0)  # AQI can't be negative

        history[forecast_time] = pred
        results.append({
            "city": city_name,
            "timestamp": forecast_time,
            "predicted_aqi": round(pred, 1),
            "aqi_category": aqi_to_category(pred),
            "is_hazardous": pred >= HAZARDOUS_AQI_THRESHOLD,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Hopsworks-dependent orchestration
# ---------------------------------------------------------------------

def get_best_model_info(project):
    """
    Compare the LATEST version of each model type by test RMSE (lower
    is better) and return the winner's metadata.

    Deliberately does NOT rely on mr.get_model(name=name) with no
    version argument — that call silently defaults to VERSION 1 (the
    oldest registered version), not the latest, which resurrected our
    already-fixed pm25 leakage bug once before. Instead we explicitly
    list every version via mr.get_models(name=name) and pick the one
    with the highest version number ourselves.
    """
    mr = project.get_model_registry()
    candidates = []
    for name in MODEL_NAMES:
        try:
            versions = mr.get_models(name=name)
            if not versions:
                logger.warning(f"No versions found for {name}.")
                continue
            latest = max(versions, key=lambda m: m.version)
            logger.info(f"{name}: latest is v{latest.version} (RMSE={latest.training_metrics.get('rmse')})")
            candidates.append(latest)
        except Exception as e:
            logger.warning(f"Could not fetch versions for {name}: {e}")

    if not candidates:
        raise RuntimeError("No registered models found — run training_pipeline.py first.")

    winner = min(candidates, key=lambda m: m.training_metrics.get("rmse", float("inf")))
    logger.info(f"Best model: {winner.name} v{winner.version} (RMSE={winner.training_metrics.get('rmse')})")
    return winner


def load_model_artifacts(model_meta):
    """Download the winning model + its preprocessing artifacts, return (model, medians, feature_columns)."""
    if os.path.exists(DOWNLOAD_DIR):
        shutil.rmtree(DOWNLOAD_DIR)
    local_dir = model_meta.download(DOWNLOAD_DIR)

    model = joblib.load(os.path.join(local_dir, "model.pkl"))
    with open(os.path.join(local_dir, "preprocessing.json")) as f:
        prep = json.load(f)
    medians, feature_columns = prep["medians"], prep["feature_columns"]

    # Defensive guard: refuse to run inference with a model trained on
    # concurrent pollutant readings (pm25, pm10, etc.) — these are the
    # known leakage columns from an earlier bug. Catching this here
    # means a future version-selection mistake fails loudly instead of
    # silently producing flat, meaningless predictions again.
    leaky_cols = {"pm25", "pm10", "no2", "o3", "so2", "co"} & set(feature_columns)
    if leaky_cols:
        raise RuntimeError(
            f"Loaded model '{model_meta.name}' v{model_meta.version} was trained on "
            f"concurrent pollutant column(s) {leaky_cols} — this is a known data-leakage "
            f"signature (see training_pipeline.py NON_FEATURE_COLUMNS). Refusing to run "
            f"inference with this model. Retrain with training_pipeline.py and use the "
            f"resulting new version instead."
        )

    return model, medians, feature_columns


def load_city_history(fs, city_name: str, hours_back: int = HISTORY_SEED_HOURS) -> dict:
    """Read recent raw AQI history for one city from Hopsworks."""
    raw_fg = fs.get_feature_group(
        name=RAW_FEATURE_GROUP_NAME,
        version=RAW_FEATURE_GROUP_VERSION
    )

    # Use Hive instead of the Hopsworks Query Service.
    df = raw_fg.read(
        read_options={"use_hive": True}
    )

    if df.empty:
        return {}

    df = df[df["city"] == city_name].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    df = (
        df.dropna(subset=["timestamp", "aqi"])
          .sort_values("timestamp")
    )

    cutoff = df["timestamp"].max() - timedelta(hours=hours_back)
    df = df[df["timestamp"] >= cutoff]

    history = {
        row["timestamp"]: float(row["aqi"])
        for _, row in df.iterrows()
    }

    return history


def generate_all_forecasts() -> pd.DataFrame:
    """Full pipeline: load best model, forecast 72h ahead for every city."""
    project = get_hopsworks_project()
    fs = get_feature_store(project)

    model_meta = get_best_model_info(project)
    model, medians, feature_columns = load_model_artifacts(model_meta)

    all_forecasts = []
    for city_name in CITIES:
        logger.info(f"Forecasting for {city_name}...")
        history = load_city_history(fs, city_name)
        if not history:
            logger.warning(f"[{city_name}] No history available — skipping.")
            continue
        last_timestamp = max(history.keys())
        city_forecast = recursive_forecast(
            model, medians, feature_columns, city_name, history, last_timestamp,
        )
        all_forecasts.append(city_forecast)

    if not all_forecasts:
        logger.error("No forecasts generated for any city.")
        return pd.DataFrame()

    return pd.concat(all_forecasts, ignore_index=True)


if __name__ == "__main__":
    forecasts = generate_all_forecasts()
    pd.set_option("display.max_rows", None)
    print(forecasts.to_string())