"""Pakistan AQI Forecast — Streamlit dashboard.

Run locally (from the project root):
    streamlit run app/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aqi_data import (
    CATEGORY_COLOR,
    CITIES,
    GUIDANCE,
    HAZARD_THRESHOLD,
    HERO_GRADIENT,
    MODELS,
    category_key,
    category_label,
    category_phrase,
    daily_means,
    fetch_city_air,
    forecast,
)

st.set_page_config(
    page_title="Pakistan AQI Forecast",
    page_icon="🌤️",
    layout="centered",
)

CSS = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------- background
st.markdown(
    """
    <div class="bg-sky-canvas" aria-hidden="true">
      <div class="bg-blob bg-blob--1"></div>
      <div class="bg-blob bg-blob--2"></div>
      <div class="bg-blob bg-blob--3"></div>
      <div class="bg-blob bg-blob--4"></div>
    </div>
    <div class="topbar">🌤️ Pakistan AQI Forecast</div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- state
st.session_state.setdefault("city", "Quetta")
st.session_state.setdefault("model", "Random Forest")


def _fmt_day(ts: pd.Timestamp) -> str:
    """Format as e.g. 'Tue 1 Sep' — cross-platform. Windows' strftime
    doesn't support the Unix-only '%-d' (no leading zero) directive
    that Python's f-string format spec would otherwise use, so we
    zero-pad with '%d' and strip the leading zero manually instead."""
    return ts.strftime("%a %d %b").replace(" 0", " ", 1)


@st.cache_data(ttl=10 * 60, show_spinner=False)
def load_air(city: str):
    return fetch_city_air(city)


@st.cache_data(ttl=10 * 60, show_spinner=False)
def load_forecast(city: str, model: str):
    return forecast(load_air(city), model)


city = st.session_state["city"]
model = st.session_state["model"]

loading_slot = st.empty()
loading_slot.markdown(
    f"""
    <div class="loading-overlay">
      <div class="loading-card glass">
        <div class="loading-ring"></div>
        <div class="loading-text">Reading the air over {city}…</div>
        <div class="loading-subtext">Fetching the latest {model} forecast</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    air = load_air(city)
    fc = load_forecast(city, model)
except Exception as exc:  # noqa: BLE001 - surface pipeline issues plainly
    loading_slot.empty()
    st.error(f"Couldn't reach the air quality pipeline just now: {exc}")
    st.stop()

loading_slot.empty()

current = int(round(air.history["aqi"].iloc[-1]))
key = category_key(current)
color = CATEGORY_COLOR[key]

peak_row = fc.loc[fc["aqi"].idxmax()]
peak_aqi = int(round(peak_row["aqi"]))
peak_time: pd.Timestamp = peak_row["time"]
peak_key = category_key(peak_aqi)

# ---------------------------------------------------------------- hero
st.markdown(
    f"""
    <section class="hero glass" style="background: {HERO_GRADIENT[key]}">
      <h1 class="hero__headline">{city} is breathing {category_phrase(current)} air right now</h1>
      <div class="hero__value" style="color: {color}">{current}</div>
      <p class="hero__label">Air quality index</p>
      <span class="pill-badge" style="background: {color}">{category_label(current)}</span>
      <p class="hero__guidance">{GUIDANCE[key]}</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ---------------------------------------------------------------- chips
def chip_row(options: list[str], state_key: str) -> None:
    columns = st.columns(len(options))
    for column, option in zip(columns, options):
        with column:
            st.button(
                option,
                key=f"{state_key}-{option}",
                type="primary" if st.session_state[state_key] == option else "secondary",
                on_click=lambda o=option: st.session_state.update({state_key: o}),
            )


chip_row(list(CITIES), "city")
chip_row(MODELS, "model")

# ---------------------------------------------------------------- alert
if peak_aqi >= HAZARD_THRESHOLD:
    st.markdown(
        f"""
        <div class="alert-card">
          <span style="font-size:1.3rem">⚠️</span>
          <p style="margin:0">Heads up — the air in {city} is expected to reach
          <strong>{peak_aqi}</strong> ({category_label(peak_aqi).lower()}) around
          {peak_time:%H:%M} on {_fmt_day(peak_time)}. Plan indoor time around that window if you can.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------------- chart
st.markdown(
    f"""
    <div class="card-pad">
      <div class="section-title">How the air's been, and where it's headed</div>
      <div class="section-sub">The solid line is the last seven days we measured.
      The dashed coral line is what {model} expects over the next three.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

hist_daily = daily_means(air.history)
fc_daily = daily_means(fc)
bridge = pd.concat([hist_daily.tail(1), fc_daily], ignore_index=True)

figure = go.Figure()
figure.add_trace(
    go.Scatter(
        x=hist_daily["day"],
        y=hist_daily["aqi"],
        mode="lines+markers",
        name="Measured",
        line=dict(color="#2F9BF0", width=3, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(47,155,240,0.18)",
        hovertemplate="%{x|%a %-d %b}: %{y:.0f} AQI<extra>Measured</extra>",
    )
)
figure.add_trace(
    go.Scatter(
        x=bridge["day"],
        y=bridge["aqi"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#FF7A59", width=3, dash="dash", shape="spline"),
        hovertemplate="%{x|%a %-d %b}: %{y:.0f} AQI<extra>Forecast</extra>",
    )
)
figure.add_hline(
    y=HAZARD_THRESHOLD,
    line=dict(color="rgba(226,63,93,0.55)", width=1, dash="dot"),
    annotation_text="Hazardous line",
    annotation_font=dict(color="rgba(226,63,93,0.85)", size=11),
)
figure.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=20, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Manrope, sans-serif", color="#1B2733"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    hovermode="x unified",
    transition=dict(duration=600, easing="cubic-in-out"),
)
figure.update_xaxes(showgrid=False, tickformat="%a %-d")
figure.update_yaxes(gridcolor="rgba(27,39,51,0.08)", zeroline=False)

st.plotly_chart(figure, width='stretch', config={"displayModeBar": False})

# ---------------------------------------------------------------- summary
st.markdown(
    f"""
    <div class="glass summary-card">
      The dirtiest stretch ahead looks like <strong>{peak_time:%A} at {peak_time:%H:%M}</strong>,
      when {city} should touch around
      <strong style="color:{CATEGORY_COLOR[peak_key]}">{peak_aqi} AQI</strong> —
      {category_label(peak_aqi).lower()} air.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---------------------------------------------------------------- table
with st.expander("See the hour-by-hour forecast"):
    table = fc.copy()
    table["Day"] = table["time"].apply(_fmt_day)
    table["Time"] = table["time"].dt.strftime("%H:%M")
    table["AQI"] = table["aqi"].round().astype(int)
    table["How it feels"] = table["AQI"].map(category_label)
    st.dataframe(
        table[["Day", "Time", "AQI", "How it feels"]],
        width=800,
        hide_index=True,
        height=420,
    )

# ---------------------------------------------------------------- footer
st.markdown(
    """
    <div class="site-footer">
      <p>Readings come from AQICN and OpenWeather, processed through an hourly feature
      pipeline into a Hopsworks feature store. Forecasts come from models trained and
      registered on that pipeline: Random Forest, XGBoost, Ridge Regression, and an
      LSTM neural network.</p>
      <p>Forecasts get fuzzier the further out they go, and nothing here is medical advice —
      if you're unwell, please talk to a doctor.</p>
    </div>
    """,
    unsafe_allow_html=True,
)