from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

# Allow importing `src.*` when Streamlit runs this from inside app/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CITIES as _SRC_CITIES, HAZARDOUS_AQI_THRESHOLD
from src.feature_pipeline import get_hopsworks_project, get_feature_store
from src.inference import load_city_history
from src.forecasting import MODEL_CHOICES


CITIES: dict[str, tuple[float, float]] = {
    name: (info["lat"], info["lon"]) for name, info in _SRC_CITIES.items()
}

MODELS: list[str] = list(MODEL_CHOICES.keys())

HAZARD_THRESHOLD = HAZARDOUS_AQI_THRESHOLD



CATEGORIES = [
    (50, "good", "Good", "clean"),
    (100, "moderate", "Moderate", "fair"),
    (150, "sensitive", "Unhealthy for sensitive groups", "slightly rough"),
    (200, "unhealthy", "Unhealthy", "unhealthy"),
    (300, "very-unhealthy", "Very unhealthy", "very unhealthy"),
    (10**6, "hazardous", "Hazardous", "hazardous"),
]

CATEGORY_COLOR = {
    "good": "#22C58B",
    "moderate": "#FFC93C",
    "sensitive": "#FF7A59",
    "unhealthy": "#E23F5D",
    "very-unhealthy": "#A855C9",
    "hazardous": "#8E1B33",
}

HERO_GRADIENT = {
    "good": "linear-gradient(140deg, rgba(47,155,240,0.30) 0%, rgba(34,197,139,0.34) 55%, rgba(255,255,255,0.72) 100%)",
    "moderate": "linear-gradient(140deg, rgba(47,155,240,0.22) 0%, rgba(255,201,60,0.42) 55%, rgba(255,255,255,0.72) 100%)",
    "sensitive": "linear-gradient(140deg, rgba(255,201,60,0.36) 0%, rgba(255,122,89,0.40) 60%, rgba(255,255,255,0.70) 100%)",
    "unhealthy": "linear-gradient(140deg, rgba(255,122,89,0.42) 0%, rgba(226,63,93,0.38) 60%, rgba(255,255,255,0.68) 100%)",
    "very-unhealthy": "linear-gradient(140deg, rgba(226,63,93,0.36) 0%, rgba(168,85,201,0.44) 60%, rgba(255,255,255,0.66) 100%)",
    "hazardous": "linear-gradient(140deg, rgba(168,85,201,0.38) 0%, rgba(142,27,51,0.48) 60%, rgba(255,255,255,0.62) 100%)",
}

GUIDANCE = {
    "good": "A great day to be outside — the air won't hold you back.",
    "moderate": "Fine for almost everyone, though a long run might feel heavier than usual.",
    "sensitive": "If you have asthma, are older, or are out with small children, take it easy outdoors today.",
    "unhealthy": "Keep outdoor time short, and a good mask helps if you have to be out for a while.",
    "very-unhealthy": "Stay indoors where you can, keep windows shut, and run a purifier if you have one.",
    "hazardous": "This is dangerous air for everyone — stay inside, seal up windows, and skip outdoor plans.",
}


def category_key(aqi: float) -> str:
    for limit, key, _label, _phrase in CATEGORIES:
        if aqi <= limit:
            return key
    return "hazardous"


def category_label(aqi: float) -> str:
    key = category_key(aqi)
    return next(label for _l, k, label, _p in CATEGORIES if k == key)


def category_phrase(aqi: float) -> str:
    key = category_key(aqi)
    return next(phrase for _l, k, _lab, phrase in CATEGORIES if k == key)


# ---------------------------------------------------------------------
# Real data + forecasting, backed by Hopsworks
# ---------------------------------------------------------------------

@dataclass
class CityAir:
    city: str
    history: pd.DataFrame  # local Asia/Karachi time, tz-naive — for display only
    history_utc: dict  # {utc_timestamp: aqi} — the real feature-computation input
    api_forecast: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["time", "aqi"]))
    # ^ kept only for interface compatibility with the original design;
    # unused now that all 4 models come from our own registry rather
    # than a third-party "physical model" forecast.


@st.cache_resource(ttl=3600, show_spinner=False)
def _get_project():
    return get_hopsworks_project()


def fetch_city_air(city: str, past_days: int = 7, forecast_days: int = 3) -> CityAir:
    """Pull real recent AQI history for a city from the Hopsworks feature store."""
    project = _get_project()
    fs = get_feature_store(project)

    history_dict = load_city_history(fs, city, hours_back=past_days * 24)
    if not history_dict:
        raise RuntimeError(f"No recent data available for {city}.")

    items = sorted(history_dict.items())
    history_local = pd.DataFrame(items, columns=["time", "aqi"])

    # Display in local Pakistan time rather than UTC — but only this
    # DISPLAY copy. The underlying history_dict stays in its original
    # (UTC) form and gets passed to forecast() unchanged, because the
    # classical models were trained on UTC-derived hour/day-of-week
    # features. Shifting to local time before computing those features
    # would quietly feed the model an hour-of-day 5 hours off from what
    # it learned during training.
    if pd.api.types.is_datetime64tz_dtype(history_local["time"]):
        history_local["time"] = history_local["time"].dt.tz_convert("Asia/Karachi").dt.tz_localize(None)

    return CityAir(city=city, history=history_local, history_utc=history_dict)


@st.cache_resource(ttl=3600, show_spinner=False)
def _load_model_artifacts(model_choice: str):
    """
    Load + cache model artifacts PER MODEL, independent of city. This is
    the key fix for perceived slowness: previously, switching cities
    while keeping the same model selected still re-downloaded that
    model's files from Hopsworks every time, because caching only
    happened at the full per-(city, model) forecast result level. Now
    the model itself loads once per hour and is reused across every
    city that uses it.
    """
    from src.forecasting import load_classical_model, load_lstm_model, MODEL_CHOICES as _MC

    project = _get_project()
    if model_choice == "LSTM (Deep Learning)":
        model, scalers, seq_length, city_order, version = load_lstm_model(project)
        return {"kind": "lstm", "model": model, "scalers": scalers,
                "seq_length": seq_length, "city_order": city_order, "version": version}
    else:
        model_name = _MC[model_choice]
        model, medians, feature_columns, version = load_classical_model(project, model_name)
        return {"kind": "classical", "model": model, "medians": medians,
                "feature_columns": feature_columns, "version": version}


def forecast(air: CityAir, model_name: str, horizon: int = 72) -> pd.DataFrame:
    """Run our real, already-trained model and return a time/aqi
    DataFrame. Reuses air.history_utc (already fetched by
    fetch_city_air) instead of hitting Hopsworks a second time."""
    from src.inference import recursive_forecast
    from src.forecasting import lstm_recursive_forecast

    artifacts = _load_model_artifacts(model_name)

    history = air.history_utc
    last_timestamp = max(history.keys())

    if artifacts["kind"] == "lstm":
        result = lstm_recursive_forecast(
            artifacts["model"], artifacts["scalers"], artifacts["seq_length"],
            artifacts["city_order"], air.city, history, last_timestamp,
            horizon_hours=horizon,
        )
    else:
        result = recursive_forecast(
            artifacts["model"], artifacts["medians"], artifacts["feature_columns"],
            air.city, history, last_timestamp, horizon_hours=horizon,
        )

    result = result.rename(columns={"timestamp": "time", "predicted_aqi": "aqi"})
    if pd.api.types.is_datetime64tz_dtype(result["time"]):
        result["time"] = result["time"].dt.tz_convert("Asia/Karachi").dt.tz_localize(None)

    return result[["time", "aqi"]].head(horizon).reset_index(drop=True)


def explain_current(air: CityAir, model_name: str, top_n: int = 6):
    """
    Explain the model's next-hour prediction using SHAP — which features
    pushed the forecast up or down, and by how much. Returns None for
    the LSTM model (see src.explainability's module docstring for why
    that's a deliberate scope decision, not a bug) or if anything about
    the explanation fails — callers should treat None as "no
    explanation available right now" and hide the panel rather than error.
    """
    artifacts = _load_model_artifacts(model_name)
    if artifacts["kind"] == "lstm":
        return None

    from src.explainability import explain_prediction


    specific_kind = MODEL_CHOICES[model_name].removeprefix("aqi_")

    history = air.history_utc
    last_timestamp = max(history.keys())
    forecast_time = last_timestamp + pd.Timedelta(hours=1)

    try:
        return explain_prediction(
            specific_kind, artifacts["model"], air.city, history, forecast_time,
            artifacts["medians"], artifacts["feature_columns"], top_n=top_n,
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"SHAP explanation failed: {e}")
        return None


def daily_means(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["day"] = out["time"].dt.normalize()
    return out.groupby("day", as_index=False)["aqi"].mean().round()


@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_all_history() -> pd.DataFrame:
    """Full raw AQI history across all cities, for the Trends & insights
    tab. Cached longer (30 min) than the live forecast data — trend
    charts don't need to be as fresh as the current-conditions hero."""
    from src.eda import load_all_history as _load_all_history

    project = _get_project()
    fs = get_feature_store(project)
    return _load_all_history(fs)