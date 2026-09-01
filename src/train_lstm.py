"""TensorFlow LSTM training pipeline — the "deep learning" entry in the
project's statistical-to-deep-learning model lineup.

Design: a single global LSTM (not one per city), matching the same
"city as a feature" strategy used for the classical models. Each
training sample is a sequence of the past `SEQ_LENGTH` hours of AQI
(scaled per-city, since cities sit at very different AQI levels) plus
that sample's city as a one-hot side-input, predicting the next hour's
AQI. This is a single-step model — 72-hour forecasts are produced by
feeding predictions back in recursively, the same conceptual approach
as the classical models in inference.py (a separate LSTM-specific
recursive loop is added when this is wired into the dashboard).
"""

import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import CITIES
from src.feature_pipeline import get_hopsworks_project, get_feature_store, RAW_FEATURE_GROUP_NAME, RAW_FEATURE_GROUP_VERSION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEQ_LENGTH = 24  # past 24 hours used to predict the next hour
MODEL_DIR = "models/lstm"
TEST_FRACTION = 0.2


# ---------------------------------------------------------------------
# Data prep — pure pandas/numpy/sklearn, testable without TensorFlow.
# ---------------------------------------------------------------------

def load_raw_history(project=None) -> pd.DataFrame:
    """Read the full raw AQI history (all cities) from Hopsworks."""
    project = project or get_hopsworks_project()
    fs = get_feature_store(project)
    raw_fg = fs.get_feature_group(name=RAW_FEATURE_GROUP_NAME, version=RAW_FEATURE_GROUP_VERSION)
    df = raw_fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "aqi"]).sort_values(["city", "timestamp"])
    df = df.drop_duplicates(subset=["city", "timestamp"], keep="last")
    logger.info(f"Loaded {len(df)} raw rows across {df['city'].nunique()} cities.")
    return df


def fit_city_scalers(df: pd.DataFrame, city_col: str = "city", value_col: str = "aqi") -> dict:
    """Fit one MinMaxScaler per city (cities sit at very different AQI
    levels, so a single global scaler would bias the model toward the
    dominant city's scale)."""
    scalers = {}
    for city in df[city_col].unique():
        values = df.loc[df[city_col] == city, value_col].values.reshape(-1, 1).astype(float)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(values)
        scalers[city] = scaler
    return scalers


def build_sequences(df: pd.DataFrame, scalers: dict, seq_length: int = SEQ_LENGTH):
    """
    Build (X, y, city_ids, target_timestamps) from a per-city sorted
    AQI series: X[i] = scaled AQI values for the seq_length hours
    before target_timestamps[i]; y[i] = scaled AQI at target_timestamps[i].

    Only genuinely consecutive hourly readings are used for a given
    window — if there's a gap in a city's data (missing hour), that
    window is skipped rather than silently bridging a gap with
    mismatched time spacing.
    """
    all_X, all_y, all_city_ids, all_timestamps = [], [], [], []

    for city in df["city"].unique():
        city_df = df[df["city"] == city].sort_values("timestamp").reset_index(drop=True)
        if len(city_df) < seq_length + 1:
            logger.warning(f"[{city}] Not enough history ({len(city_df)} rows) for seq_length={seq_length} — skipping.")
            continue

        scaler = scalers[city]
        scaled_values = scaler.transform(city_df[["aqi"]].values.astype(float)).flatten()
        timestamps = city_df["timestamp"].values

        for i in range(len(city_df) - seq_length):
            window_ts = timestamps[i:i + seq_length + 1]
            # Check the window (seq_length+1 points) is genuinely hourly
            # spaced — skip windows straddling a data gap.
            deltas = np.diff(window_ts).astype("timedelta64[m]").astype(int)
            if not np.all(np.abs(deltas - 60) <= 15):  # allow +/-15 min jitter
                continue

            all_X.append(scaled_values[i:i + seq_length])
            all_y.append(scaled_values[i + seq_length])
            all_city_ids.append(city)
            all_timestamps.append(pd.Timestamp(timestamps[i + seq_length]))

    X = np.array(all_X).reshape(-1, seq_length, 1)
    y = np.array(all_y)
    logger.info(f"Built {len(X)} sequences (seq_length={seq_length}) across {df['city'].nunique()} cities.")
    return X, y, np.array(all_city_ids), np.array(all_timestamps)


def city_ids_to_onehot(city_ids: np.ndarray) -> np.ndarray:
    """One-hot encode city id array using the fixed CITIES order from
    config, so column order is consistent and reproducible."""
    city_list = list(CITIES.keys())
    onehot = np.zeros((len(city_ids), len(city_list)), dtype="float32")
    for i, city in enumerate(city_ids):
        if city in city_list:
            onehot[i, city_list.index(city)] = 1.0
    return onehot


def time_based_split_sequences(X, y, city_ids, timestamps, test_fraction: float = TEST_FRACTION):
    """
    Time-based split done PER CITY (since sequences from different
    cities interleave in time but each city's own series must stay
    chronologically ordered within its own split) — then combined.
    """
    train_idx, test_idx = [], []
    for city in np.unique(city_ids):
        city_mask = np.where(city_ids == city)[0]
        # city_mask indices are already in chronological order since
        # build_sequences iterates each city's rows in time order.
        split_point = int(len(city_mask) * (1 - test_fraction))
        train_idx.extend(city_mask[:split_point])
        test_idx.extend(city_mask[split_point:])

    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    return (
        X[train_idx], y[train_idx], city_ids[train_idx],
        X[test_idx], y[test_idx], city_ids[test_idx],
    )


# ---------------------------------------------------------------------
# Model — requires TensorFlow. Kept separate from the data-prep
# functions above so those can be unit tested without a TF install.
# ---------------------------------------------------------------------

def build_lstm_model(seq_length: int, n_cities: int):
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    aqi_input = layers.Input(shape=(seq_length, 1), name="aqi_sequence")
    city_input = layers.Input(shape=(n_cities,), name="city_onehot")

    x = layers.LSTM(32, return_sequences=False)(aqi_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Concatenate()([x, city_input])
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1, activation="linear", name="next_aqi")(x)

    model = Model(inputs=[aqi_input, city_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def inverse_transform_per_city(scaled_values, city_ids, scalers):
    """Convert scaled [0,1] predictions/targets back to real AQI scale,
    using each sample's own city-specific scaler."""
    result = np.zeros_like(scaled_values, dtype=float)
    for city in np.unique(city_ids):
        mask = city_ids == city
        result[mask] = scalers[city].inverse_transform(scaled_values[mask].reshape(-1, 1)).flatten()
    return result


def evaluate_real_scale(y_true_scaled, y_pred_scaled, city_ids, scalers) -> dict:
    y_true = inverse_transform_per_city(y_true_scaled, city_ids, scalers)
    y_pred = inverse_transform_per_city(y_pred_scaled, city_ids, scalers)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_lstm_training(epochs: int = 30, batch_size: int = 32):
    project = get_hopsworks_project()
    raw_df = load_raw_history(project)
    if raw_df.empty:
        logger.error("No data available — run backfill.py first.")
        return

    scalers = fit_city_scalers(raw_df)
    X, y, city_ids, timestamps = build_sequences(raw_df, scalers, seq_length=SEQ_LENGTH)
    if len(X) == 0:
        logger.error("No valid sequences built — check data has enough consecutive hourly history.")
        return

    X_train, y_train, city_train, X_test, y_test, city_test = time_based_split_sequences(
        X, y, city_ids, timestamps, test_fraction=TEST_FRACTION
    )
    city_train_oh = city_ids_to_onehot(city_train)
    city_test_oh = city_ids_to_onehot(city_test)

    logger.info(f"Train sequences: {len(X_train)}, Test sequences: {len(X_test)}")

    model = build_lstm_model(seq_length=SEQ_LENGTH, n_cities=len(CITIES))

    import tensorflow as tf
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        [X_train, city_train_oh], y_train,
        validation_split=0.1,
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    y_pred_scaled = model.predict([X_test, city_test_oh]).flatten()
    metrics = evaluate_real_scale(y_test, y_pred_scaled, city_test, scalers)
    logger.info(f"LSTM test metrics (real AQI scale): RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f} R2={metrics['r2']:.3f}")

    # Save model + scalers + metadata
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))
    joblib.dump(scalers, os.path.join(MODEL_DIR, "scalers.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump({"seq_length": SEQ_LENGTH, "cities": list(CITIES.keys())}, f)

    mr = project.get_model_registry()
    hw_model = mr.python.create_model(
        name="aqi_lstm",
        metrics=metrics,
        description="AQI forecasting LSTM (TensorFlow) — sequence model over past 24h AQI, city as side-input.",
    )
    hw_model.save(MODEL_DIR)
    logger.info("LSTM registered in Hopsworks Model Registry.")

    print("\n=== LSTM Model ===")
    print(f"RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R2={metrics['r2']:.3f}")


if __name__ == "__main__":
    run_lstm_training()