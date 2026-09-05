"""SHAP explainability for the classical models (Ridge, Random Forest,
XGBoost), explaining a specific prediction — normally the model's
current "next hour" forecast — in terms of which features pushed it up
or down.

Scope note: LSTM is intentionally NOT covered here. SHAP support for
sequence models (DeepExplainer/GradientExplainer) is significantly
slower and more fragile than TreeExplainer/LinearExplainer, and would
need its own background-sequence sampling strategy. Given the project
timeline, explaining the three classical models — which is also where
SHAP is fastest and most reliable — was judged the better use of time.
This is a deliberate scope decision, not an oversight.
"""

import logging

import numpy as np
import pandas as pd

from src.inference import build_feature_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Human-readable labels for engineered feature names, used when
# displaying explanations to end users rather than raw column names.
FEATURE_LABELS = {
    "hour": "Time of day",
    "day_of_week": "Day of the week",
    "month": "Month",
    "is_weekend": "Weekend",
    "is_stale": "Reading freshness",
    "lag_1h_aqi": "AQI an hour ago",
    "lag_3h_aqi": "AQI three hours ago",
    "lag_24h_aqi": "AQI this time yesterday",
    "rolling_3h_mean_aqi": "3-hour average AQI",
    "rolling_3h_std_aqi": "3-hour AQI variability",
    "rolling_24h_mean_aqi": "24-hour average AQI",
    "rolling_24h_std_aqi": "24-hour AQI variability",
    "aqi_change_rate": "Recent rate of change",
}


def _label_for(feature_name: str, city_name: str) -> str | None:
    """
    Return a human-readable label for a feature column, or None if this
    feature shouldn't be shown (inactive city one-hot columns — only
    the active city's own dummy is meaningful to display).
    """
    if feature_name.startswith("city_"):
        this_city = feature_name[len("city_"):]
        if this_city != city_name:
            return None  # inactive dummy, not meaningful to show individually
        return f"Being in {city_name}"
    return FEATURE_LABELS.get(feature_name, feature_name)


def _build_explainer(model_kind: str, model, background_df: pd.DataFrame):
    """
    Construct the appropriate SHAP explainer for the model family.
    Tree models get the fast, exact TreeExplainer (no background data
    needed, using tree_path_dependent perturbation). Ridge gets a
    LinearExplainer with a small background sample as its baseline.
    """
    import shap

    if model_kind in ("random_forest", "xgboost"):
        return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    elif model_kind == "ridge":
        masker = shap.maskers.Independent(background_df)
        return shap.LinearExplainer(model, masker)
    else:
        raise ValueError(f"SHAP explanation not supported for model_kind='{model_kind}'.")


def explain_prediction(
    model_kind: str, model, city_name: str, history: dict,
    forecast_time: pd.Timestamp, medians: dict, feature_columns: list,
    top_n: int = 6,
) -> list[dict]:
    """
    Explain a single prediction (by default, the model's next-hour
    forecast) using SHAP. Returns a list of up to top_n dicts, sorted
    by |SHAP value| descending:

        {"feature": raw column name, "label": human-readable label,
         "value": the feature's actual value, "shap_value": contribution
         (positive = pushed AQI up, negative = pushed AQI down)}

    Inactive city one-hot columns are filtered out before ranking, so
    top_n reflects genuinely meaningful features rather than wasting
    slots on ~0-contribution dummy columns.
    """
    X = build_feature_row(history, forecast_time, city_name, medians, feature_columns)

    # Background sample for LinearExplainer: use the training medians
    # as a single representative "typical" baseline row. Cheap and
    # reasonably interpretable — contributions are relative to "what if
    # every feature were at its typical training value".
    background_df = pd.DataFrame([medians])[feature_columns]

    explainer = _build_explainer(model_kind, model, background_df)
    shap_values = explainer.shap_values(X)

    # TreeExplainer/LinearExplainer both return shape (1, n_features)
    # for a single-output regressor; flatten to a 1D array of per-feature
    # contributions for this one row.
    shap_row = np.array(shap_values).reshape(-1)

    entries = []
    for col, shap_val in zip(feature_columns, shap_row):
        label = _label_for(col, city_name)
        if label is None:
            continue
        entries.append({
            "feature": col,
            "label": label,
            "value": float(X[col].iloc[0]),
            "shap_value": float(shap_val),
        })

    entries.sort(key=lambda e: abs(e["shap_value"]), reverse=True)
    return entries[:top_n]


if __name__ == "__main__":
    # Quick manual test: explain the current live prediction for one
    # city using the best registered model. Run from repo root:
    #   python -m src.explainability
    from src.feature_pipeline import get_hopsworks_project
    from src.forecasting import load_classical_model, MODEL_CHOICES
    from src.inference import load_city_history
    from src.feature_pipeline import get_feature_store
    from datetime import timedelta

    project = get_hopsworks_project()
    fs = get_feature_store(project)

    city = "Lahore"
    model_name = "aqi_random_forest"
    model, medians, feature_columns, version = load_classical_model(project, model_name)
    print(f"Using {model_name} v{version}")

    history = load_city_history(fs, city, hours_back=72)
    last_timestamp = max(history.keys())
    forecast_time = last_timestamp + timedelta(hours=1)

    explanation = explain_prediction(
        "random_forest", model, city, history, forecast_time, medians, feature_columns,
    )
    print(f"\nWhy is the model predicting what it predicts for {city}?")
    for e in explanation:
        direction = "pushes AQI up" if e["shap_value"] > 0 else "pushes AQI down"
        print(f"  {e['label']:<28} value={e['value']:.1f}  {direction} by {abs(e['shap_value']):.2f}")