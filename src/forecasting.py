"""Unified forecasting entry point for the dashboard — supports all 4
registered models (Ridge, Random Forest, XGBoost, LSTM) behind one
consistent interface, since each family needs different loading and
recursive-forecast logic (flat feature vectors for the classical
models vs. windowed sequences for the LSTM).
"""

import json
import logging
import os
import shutil
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

from src.config import CITIES, HAZARDOUS_AQI_THRESHOLD, FORECAST_HOURS_AHEAD
from src.feature_pipeline import get_hopsworks_project, get_feature_store, RAW_FEATURE_GROUP_NAME, RAW_FEATURE_GROUP_VERSION
from src.inference import recursive_forecast, aqi_to_category, load_city_history, HISTORY_SEED_HOURS
from src.train_lstm import SEQ_LENGTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Display name -> registered Hopsworks model name
MODEL_CHOICES = {
    "Random Forest": "aqi_random_forest",
    "XGBoost": "aqi_xgboost",
    "Ridge Regression": "aqi_ridge",
    "LSTM (Deep Learning)": "aqi_lstm",
}

CLASSICAL_DOWNLOAD_DIR = "models/_dashboard_download"
LSTM_DOWNLOAD_DIR = "models/_dashboard_lstm_download"

LEAKY_COLUMNS = {"pm25", "pm10", "no2", "o3", "so2", "co"}


def get_latest_model_version(project, model_name: str):
    """Explicitly find the highest-version registered model with this
    name — never rely on any SDK 'default version' behavior (see
    inference.py's get_best_model_info for why)."""
    mr = project.get_model_registry()
    versions = mr.get_models(name=model_name)
    if not versions:
        raise RuntimeError(f"No registered versions found for model '{model_name}'.")
    return max(versions, key=lambda m: m.version)


def load_classical_model(project, model_name: str):
    """Load a specific classical (sklearn-style) model + its preprocessing artifacts."""
    model_meta = get_latest_model_version(project, model_name)

    if os.path.exists(CLASSICAL_DOWNLOAD_DIR):
        shutil.rmtree(CLASSICAL_DOWNLOAD_DIR)
    local_dir = model_meta.download(CLASSICAL_DOWNLOAD_DIR)

    model = joblib.load(os.path.join(local_dir, "model.pkl"))
    with open(os.path.join(local_dir, "preprocessing.json")) as f:
        prep = json.load(f)
    medians, feature_columns = prep["medians"], prep["feature_columns"]

    leaky = LEAKY_COLUMNS & set(feature_columns)
    if leaky:
        raise RuntimeError(
            f"Model '{model_name}' v{model_meta.version} was trained on concurrent "
            f"pollutant column(s) {leaky} — refusing to use (data leakage signature)."
        )

    return model, medians, feature_columns, model_meta.version


def load_lstm_model(project):
    """Load the LSTM model + its scalers/metadata."""
    import tensorflow as tf

    model_meta = get_latest_model_version(project, "aqi_lstm")

    if os.path.exists(LSTM_DOWNLOAD_DIR):
        shutil.rmtree(LSTM_DOWNLOAD_DIR)
    local_dir = model_meta.download(LSTM_DOWNLOAD_DIR)

    model = tf.keras.models.load_model(os.path.join(local_dir, "lstm_model.keras"))
    scalers = joblib.load(os.path.join(local_dir, "scalers.pkl"))
    with open(os.path.join(local_dir, "metadata.json")) as f:
        metadata = json.load(f)

    return model, scalers, metadata["seq_length"], metadata["cities"], model_meta.version


def lstm_recursive_forecast(
    model, scalers: dict, seq_length: int, city_order: list, city_name: str,
    history: dict, last_timestamp: pd.Timestamp, horizon_hours: int = FORECAST_HOURS_AHEAD,
) -> pd.DataFrame:

    if city_name not in scalers:
        raise ValueError(f"No scaler found for city '{city_name}' — was it in the LSTM training data?")
    scaler = scalers[city_name]

    sorted_ts = sorted(history.keys())
    if len(sorted_ts) < seq_length:
        raise ValueError(
            f"[{city_name}] Not enough history ({len(sorted_ts)} points) for "
            f"LSTM seq_length={seq_length} — need at least {seq_length}."
        )

    recent_ts = sorted_ts[-seq_length:]
    recent_values = np.array([history[ts] for ts in recent_ts]).reshape(-1, 1)
    scaled_window = scaler.transform(recent_values).flatten().tolist()

    city_onehot = np.zeros((1, len(city_order)), dtype="float32")
    if city_name in city_order:
        city_onehot[0, city_order.index(city_name)] = 1.0

    results = []
    for step in range(1, horizon_hours + 1):
        forecast_time = last_timestamp + timedelta(hours=step)

        X_seq = np.array(scaled_window[-seq_length:]).reshape(1, seq_length, 1)
        pred_scaled = float(model.predict([X_seq, city_onehot], verbose=0)[0, 0])
        pred_scaled = min(max(pred_scaled, 0.0), 1.0)  # keep in valid scaler range

        pred_real = float(scaler.inverse_transform([[pred_scaled]])[0, 0])
        pred_real = max(pred_real, 0.0)

        scaled_window.append(pred_scaled)

        results.append({
            "city": city_name,
            "timestamp": forecast_time,
            "predicted_aqi": round(pred_real, 1),
            "aqi_category": aqi_to_category(pred_real),
            "is_hazardous": pred_real >= HAZARDOUS_AQI_THRESHOLD,
        })

    return pd.DataFrame(results)


def forecast_city(city_name: str, model_choice: str, project=None) -> pd.DataFrame:

    if model_choice not in MODEL_CHOICES:
        raise ValueError(f"Unknown model choice '{model_choice}'. Options: {list(MODEL_CHOICES.keys())}")

    project = project or get_hopsworks_project()
    fs = get_feature_store(project)

    history = load_city_history(fs, city_name, hours_back=HISTORY_SEED_HOURS)
    if not history:
        raise RuntimeError(f"No history available for {city_name}.")
    last_timestamp = max(history.keys())

    if model_choice == "LSTM (Deep Learning)":
        model, scalers, seq_length, city_order, version = load_lstm_model(project)
        logger.info(f"Using aqi_lstm v{version}")
        return lstm_recursive_forecast(model, scalers, seq_length, city_order, city_name, history, last_timestamp)
    else:
        model_name = MODEL_CHOICES[model_choice]
        model, medians, feature_columns, version = load_classical_model(project, model_name)
        logger.info(f"Using {model_name} v{version}")
        return recursive_forecast(model, medians, feature_columns, city_name, history, last_timestamp)


if __name__ == "__main__":

    project = get_hopsworks_project()
    for choice in MODEL_CHOICES:
        print(f"\n=== {choice} ===")
        try:
            df = forecast_city("Quetta", choice, project=project)
            print(df.head(5).to_string())
        except Exception as e:
            print(f"FAILED: {e}")