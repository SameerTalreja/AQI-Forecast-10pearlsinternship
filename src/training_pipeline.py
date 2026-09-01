"""Training pipeline: pulls engineered features from Hopsworks, trains
multiple models (Ridge, Random Forest, XGBoost), evaluates with
RMSE/MAE/R², and registers all of them in the Hopsworks Model Registry.

Run manually or via the daily GitHub Actions cron:
    python -m src.training_pipeline
"""

import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from src.config import TARGET_COLUMN
from src.feature_pipeline import get_hopsworks_project, get_feature_store, get_or_create_engineered_feature_group

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
MAX_MISSING_FRACTION = 0.5  # drop a feature column if more than this fraction is missing

# Columns that are identifiers, targets, or redundant with other columns
# — never used as model input features.
#
# IMPORTANT — leakage guard: raw concurrent pollutant readings (pm25,
# pm10, no2, o3, so2, co) are EXCLUDED even though they're numeric and
# would look like reasonable features. AQI is directly derived from
# PM2.5 (either by AQICN internally, or by our own compute_aqi_from_pm25
# formula for backfilled rows) — so including the same-moment pm25
# lets the model trivially reverse-engineer that formula instead of
# genuinely forecasting. At real inference time we never know the
# future PM2.5 any more than we know the future AQI, so training on it
# produces a model with unrealistically perfect offline metrics that
# cannot actually forecast anything. Only LAGGED/rolling versions of
# these signals (computed from strictly past readings) are legitimate
# predictive features.
NON_FEATURE_COLUMNS = [
    "timestamp", "city", TARGET_COLUMN,
    "matched_station", "dominant_pollutant",
    "aqicn_temp", "aqicn_humidity", "aqicn_pressure", "aqicn_wind",  # redundant with temp/humidity/pressure/wind_speed
    "pm25", "pm10", "no2", "o3", "so2", "co",  # concurrent readings — leakage risk, see note above
]


def load_training_data(project) -> pd.DataFrame:
    """Read the full engineered feature group from Hopsworks."""
    fs = get_feature_store(project)
    fg = get_or_create_engineered_feature_group(fs)
    df = fg.read()
    logger.info(f"Loaded {len(df)} rows from engineered feature group.")
    return df


def build_feature_matrix(df: pd.DataFrame, fit_medians: dict | None = None, fit_columns: list | None = None):
    """
    Prepare X, y from the raw engineered DataFrame:
      - drop rows with missing target
      - drop feature columns that are mostly missing
      - one-hot encode `city`
      - impute remaining NaNs with column medians (computed on TRAIN only
        when fit_medians is None; reused on test/inference when passed in)

    Returns (X, y, medians_used, feature_columns) so the same
    preprocessing can be replayed exactly at inference time.
    """
    df = df.copy()
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])

    y = df[TARGET_COLUMN].astype(float)

    candidate_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    feature_df = df[candidate_cols].copy()

    # Coerce everything except is_stale/city-related to numeric; is_stale
    # is boolean already, keep as int.
    if "is_stale" in feature_df.columns:
        feature_df["is_stale"] = feature_df["is_stale"].astype(int)

    for col in feature_df.columns:
        if col == "is_stale":
            continue
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    # One-hot encode city BEFORE column reconciliation below, so both
    # the "fit" (train) and "reuse fit_columns" (test/inference)
    # branches operate on a DataFrame that already includes city_*
    # columns — otherwise the reconciliation step and a second
    # concat would each add their own copy, producing duplicates.
    city_dummies = pd.get_dummies(df["city"], prefix="city")
    feature_df = pd.concat([feature_df.reset_index(drop=True), city_dummies.reset_index(drop=True)], axis=1)

    if fit_columns is None:
        # Drop columns that are mostly missing (fit phase only)
        missing_fraction = feature_df.isna().mean()
        keep_cols = missing_fraction[missing_fraction <= MAX_MISSING_FRACTION].index.tolist()
        dropped = set(feature_df.columns) - set(keep_cols)
        if dropped:
            logger.info(f"Dropping mostly-missing feature columns: {sorted(dropped)}")
        feature_df = feature_df[keep_cols]
    else:
        # Inference/test phase: force exact same columns as training.
        # Columns missing entirely at this stage are a schema mismatch
        # (e.g. a city one-hot category absent from this slice) rather
        # than a genuine sensor gap, so fill with 0, not NaN/median.
        for col in fit_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        feature_df = feature_df[fit_columns]

    # Impute remaining NaNs with medians (fit on train, reused on test)
    if fit_medians is None:
        medians = feature_df.median(numeric_only=True).to_dict()
    else:
        medians = fit_medians
    for col in feature_df.columns:
        if feature_df[col].isna().any():
            feature_df[col] = feature_df[col].fillna(medians.get(col, 0))

    feature_columns = feature_df.columns.tolist()
    return feature_df, y.reset_index(drop=True), medians, feature_columns


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Sort by timestamp and split chronologically — never shuffle time series data."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    logger.info(f"Time-based split: {len(train_df)} train rows, {len(test_df)} test rows.")
    return train_df, test_df


def evaluate(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_all_models(X_train, y_train, X_test, y_test) -> dict:
    """Train Ridge, Random Forest, and XGBoost; return {name: (model, metrics)}."""
    results = {}

    logger.info("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    results["ridge"] = (ridge, evaluate(y_test, ridge.predict(X_test)))

    logger.info("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["random_forest"] = (rf, evaluate(y_test, rf.predict(X_test)))

    logger.info("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    results["xgboost"] = (xgb_model, evaluate(y_test, xgb_model.predict(X_test)))

    return results


def save_and_register_models(results: dict, medians: dict, feature_columns: list, project):
    """Save each model + its preprocessing artifacts locally, then
    register in the Hopsworks Model Registry with its evaluation metrics."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    mr = project.get_model_registry()

    for name, (model, metrics) in results.items():
        model_dir = os.path.join(MODELS_DIR, name)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        joblib.dump(model, os.path.join(model_dir, "model.pkl"))
        with open(os.path.join(model_dir, "preprocessing.json"), "w") as f:
            json.dump({"medians": medians, "feature_columns": feature_columns}, f)

        logger.info(f"[{name}] RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f} R2={metrics['r2']:.3f}")

        hw_model = mr.python.create_model(
            name=f"aqi_{name}",
            metrics=metrics,
            description=f"AQI forecasting model ({name}) for 6 Pakistani cities.",
        )
        hw_model.save(model_dir)
        logger.info(f"[{name}] Registered in Hopsworks Model Registry.")


def run_training_pipeline():
    project = get_hopsworks_project()
    raw_df = load_training_data(project)
    if raw_df.empty:
        logger.error("No training data available — run backfill.py first.")
        return

    train_df, test_df = time_based_split(raw_df, test_fraction=0.2)

    X_train, y_train, medians, feature_columns = build_feature_matrix(train_df)
    X_test, y_test, _, _ = build_feature_matrix(test_df, fit_medians=medians, fit_columns=feature_columns)

    logger.info(f"Training on {X_train.shape[0]} rows, {X_train.shape[1]} features.")

    results = train_all_models(X_train, y_train, X_test, y_test)

    print("\n=== Model Comparison ===")
    print(f"{'Model':<15} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    for name, (_, metrics) in results.items():
        print(f"{name:<15} {metrics['rmse']:>8.2f} {metrics['mae']:>8.2f} {metrics['r2']:>8.3f}")

    save_and_register_models(results, medians, feature_columns, project)


if __name__ == "__main__":
    run_training_pipeline()