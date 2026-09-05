"""Pakistan AQI Forecast — Streamlit dashboard.
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
    explain_current,
    fetch_city_air,
    forecast,
    load_all_history,
)
from src.eda import (
    city_summary_stats,
    daily_trend,
    hourly_pattern,
    weekday_pattern,
    data_freshness_summary,
)

st.set_page_config(
    page_title="Pakistan AQI Forecast",
    page_icon="🌤️",
    layout="centered",
)

CSS = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
st.markdown(
        f"""
        <style>
            {CSS}
            [data-testid="stExpander"] summary,
            [data-testid="stExpander"] summary:hover,
            [data-testid="stExpander"] summary:focus,
            [data-testid="stExpander"] summary:focus-visible,
            [data-testid="stExpander"] summary:active,
            [data-testid="stExpander"] summary:has(+ *) {{
                color: inherit !important;
                background-color: transparent !important;
                outline: none !important;
                box-shadow: none !important;
            }}
            [data-testid="stExpander"] summary * {{
                color: inherit !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
)

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# background
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

# state
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


@st.cache_data(ttl=10 * 60, show_spinner=False)
def load_explanation(city: str, model: str):
    return explain_current(load_air(city), model)


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
    explanation = load_explanation(city, model)
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

#  hero
st.markdown(
    f"""
    <section class="hero glass" style="background: {HERO_GRADIENT[key]}">
      <h1 class="hero__headline">{city} is breathing {category_phrase(current)} air right now</h1>
      <div class="hero__value" style="color: {color}">{current}</div>
      <p class="hero__label">Air quality index</p>
      <span class="pill-badge" style="background: {color}">{category_label(current)}</span>
            <div class="hero__guidance" style="width: 100%; text-align: center;">{GUIDANCE[key]}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write("")


# chips
def chip_row(options: list[str], state_key: str) -> None:
    columns = st.columns(len(options))
    for column, option in zip(columns, options):
        with column:
            st.button(
                option,
                key=f"{state_key}-{option}",
                width='stretch',
                type="primary" if st.session_state[state_key] == option else "secondary",
                on_click=lambda o=option: st.session_state.update({state_key: o}),
            )


chip_row(list(CITIES), "city")
chip_row(MODELS, "model")

# alert
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

#  tabs
tab_forecast, tab_trends = st.tabs(["Forecast", "Trends & insights"])

with tab_forecast:
    #  chart
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(color="#1B2733"),
        ),
        hovermode="x unified",
        transition=dict(duration=600, easing="cubic-in-out"),
    )
    figure.update_xaxes(showgrid=False, tickformat="%a %-d")
    figure.update_yaxes(gridcolor="rgba(27,39,51,0.08)", zeroline=False)

    st.plotly_chart(figure, width='stretch', config={"displayModeBar": False})

    #  summary
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

    #  why
    st.markdown(
        """
        <div class="card-pad">
          <div class="section-title">What's driving this forecast</div>
          <div class="section-sub">The biggest reasons behind the very next hour's prediction.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if explanation:
        max_abs = max(abs(e["shap_value"]) for e in explanation) or 1

        rows = []
        for e in explanation:
            v = e["shap_value"]
            is_up = v > 0
            arrow = "▲" if is_up else "▼"
            driver_color = (
                CATEGORY_COLOR["unhealthy"]
                if is_up
                else CATEGORY_COLOR["good"]
            )
            verb = "Adds about" if is_up else "Cuts about"
            width_pct = min(abs(v) / max_abs * 100, 100)

            rows.append(
                f'<div class="driver-row">'
                f'<div class="driver-icon" style="color:{driver_color}">{arrow}</div>'
                f'<div class="driver-text">'
                f'<div class="driver-label">{e["label"]}</div>'
                f'<div class="driver-bar-track">'
                f'<div class="driver-bar-fill" style="--w:{width_pct:.0f}%; background:{driver_color};"></div>'
                f'</div>'
                f'</div>'
                f'<div class="driver-value" style="color:{driver_color}">'
                f'{verb} {abs(v):.0f}'
                f'</div>'
                f'</div>'
            )

        rows_html = "".join(rows)

        st.markdown(
            f'<div class="driver-list">{rows_html}</div>',
            unsafe_allow_html=True,
        )
        st.caption("This explains the next hour's number specifically — not the whole 3-day forecast.")
    else:
        st.markdown(
    
            unsafe_allow_html=True,
        )

    st.write("")

    #  table
    with st.expander("See the hour-by-hour forecast"):
        table = fc.copy()
        table["Day"] = table["time"].apply(_fmt_day)
        table["Time"] = table["time"].dt.strftime("%H:%M")
        table["AQI"] = table["aqi"].round().astype(int)
        table["How it feels"] = table["AQI"].map(category_label)
        table_view = table[["Day", "Time", "AQI", "How it feels"]].style.set_properties(
            **{
                "background-color": "#eef7fb",
                "color": "#17324d",
                "border-color": "#c6dce5",
            }
        ).set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#d8edf4"),
                        ("color", "#17324d"),
                        ("font-weight", "600"),
                    ],
                }
            ]
        )
        st.dataframe(
            table_view,
            width='stretch',
            hide_index=True,
            height=420,
        )

with tab_trends:
    try:
        all_history = load_all_history()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't load trend data just now: {exc}")
        all_history = pd.DataFrame()

    if all_history.empty:
        st.markdown(
            '<div class="glass summary-card">No historical data available yet.</div>',
            unsafe_allow_html=True,
        )
    else:
        #  city comparison
        st.markdown(
            """
            <div class="card-pad">
              <div class="section-title">How the six cities compare</div>
              <div class="section-sub">Average AQI across all the data we've collected so far.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        summary = city_summary_stats(all_history)
        summary_colors = [CATEGORY_COLOR[category_key(v)] for v in summary["mean"]]

        cmp_fig = go.Figure(go.Bar(
            x=summary["city"], y=summary["mean"],
            marker_color=summary_colors,
            hovertemplate="%{x}: %{y:.0f} average AQI<extra></extra>",
        ))
        cmp_fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Manrope, sans-serif", color="#1B2733"),
            yaxis=dict(title="Average AQI", gridcolor="rgba(27,39,51,0.08)"),
            xaxis=dict(title=None),
        )
        st.plotly_chart(cmp_fig, width='stretch', config={"displayModeBar": False})

        st.write("")

        # trend over time
        st.markdown(
            """
            <div class="card-pad">
              <div class="section-title">The trend over time</div>
              <div class="section-sub">Daily average AQI for each city since we started collecting data.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        trend = daily_trend(all_history)
        trend_fig = go.Figure()
        palette = ["#2F9BF0", "#FF7A59", "#22C58B", "#FFC93C", "#A855C9", "#E23F5D"]
        for i, c in enumerate(sorted(trend["city"].unique())):
            city_trend = trend[trend["city"] == c]
            trend_fig.add_trace(go.Scatter(
                x=city_trend["day"], y=city_trend["aqi"],
                mode="lines", name=c,
                line=dict(color=palette[i % len(palette)], width=2),
            ))
        trend_fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Manrope, sans-serif", color="#1B2733"),
            yaxis=dict(title="AQI", gridcolor="rgba(27,39,51,0.08)"),
            xaxis=dict(
                title=None,
                gridcolor="rgba(27,39,51,0.08)",
                tickfont=dict(color="#000000"),
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, x=0,
                font=dict(color="#000000"),
            ),
            hovermode="x unified",
        )
        st.plotly_chart(trend_fig, width='stretch', config={"displayModeBar": False})

        st.write("")

        #  hourly + weekday pattern for selected city
        st.markdown(
            f"""
            <div class="card-pad">
              <div class="section-title">When {city}'s air is at its worst</div>
              <div class="section-sub">Average AQI by hour of day and by day of the week.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_hourly, col_weekday = st.columns(2)

        with col_hourly:
            hourly = hourly_pattern(all_history, city=city)
            hourly_fig = go.Figure(go.Bar(
                x=hourly["hour"], y=hourly["aqi"],
                marker_color="#2F9BF0",
                hovertemplate="%{x}:00 — %{y:.0f} AQI<extra></extra>",
            ))
            hourly_fig.update_layout(
                height=280, margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Manrope, sans-serif", color="#1B2733", size=11),
                title=dict(text="By hour of day", font=dict(size=13)),
                yaxis=dict(title=None, gridcolor="rgba(27,39,51,0.08)"),
                xaxis=dict(title=None, dtick=4),
            )
            st.plotly_chart(hourly_fig, width='stretch', config={"displayModeBar": False})

        with col_weekday:
            weekday = weekday_pattern(all_history, city=city)
            weekday_fig = go.Figure(go.Bar(
                x=WEEKDAY_NAMES, y=weekday["aqi"],
                marker_color="#FF7A59",
                hovertemplate="%{x}: %{y:.0f} AQI<extra></extra>",
            ))
            weekday_fig.update_layout(
                height=280, margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Manrope, sans-serif", color="#1B2733", size=11),
                title=dict(text="By day of week", font=dict(size=13)),
                yaxis=dict(title=None, gridcolor="rgba(27,39,51,0.08)"),
                xaxis=dict(title=None),
            )
            st.plotly_chart(weekday_fig, width='stretch', config={"displayModeBar": False})

        st.write("")

        #  data freshness
        with st.expander("How fresh is each city's data?"):
            st.caption(
                "Some monitoring stations report near-hourly; others update every few days. "
                "This is a real limitation of the free station data this project relies on, "
                "not a bug in the pipeline."
            )
            freshness = data_freshness_summary(all_history)
            freshness_display = freshness.copy()
            freshness_display["latest_reading"] = freshness_display["latest_reading"].dt.strftime("%d %b, %H:%M UTC")
            freshness_display["hours_since_latest"] = freshness_display["hours_since_latest"].round(1)
            freshness_display = freshness_display.rename(columns={
                "city": "City", "reading_count": "Readings collected",
                "latest_reading": "Latest reading", "hours_since_latest": "Hours ago",
            })
            st.dataframe(freshness_display, width='stretch', hide_index=True)

#  footer
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