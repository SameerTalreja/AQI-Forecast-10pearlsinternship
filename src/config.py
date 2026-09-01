"""Central configuration: cities, API endpoints, feature/model constants."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys (set these in your .env file) ---
AQICN_API_KEY = os.getenv("AQICN_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")

# --- API Base URLs ---
AQICN_BASE_URL = "https://api.waqi.info/feed"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# --- Cities (lat/lon used for AQICN geo-lookup and OpenWeather) ---
CITIES = {
    "Quetta": {"station": "A544306", "lat": 30.1798, "lon": 66.9750},
    "Lahore": {"station": "lahore", "lat": 31.5497, "lon": 74.3436},
    "Karachi": {"station": "karachi", "lat": 24.8607, "lon": 67.0011},
    "Islamabad": {"station": "islamabad", "lat": 33.6844, "lon": 73.0479},
    "Faisalabad": {"station": "A545326", "lat": 31.4504, "lon": 73.1350},
    "Peshawar": {"station": "peshawar", "lat": 34.0151, "lon": 71.5249},
}

DEFAULT_CITY = "Quetta"

# --- Feature engineering constants ---
POLLUTANT_FIELDS = ["pm25", "pm10", "no2", "o3", "so2", "co"]
WEATHER_FIELDS = ["temp", "humidity", "wind_speed", "pressure"]

LAG_HOURS = [1, 3, 24]
ROLLING_WINDOWS_HOURS = [3, 24]

TARGET_COLUMN = "aqi"

# --- Alerting ---
HAZARDOUS_AQI_THRESHOLD = 150  # "Unhealthy" and above on US EPA scale

# --- Hopsworks ---
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_forecast")
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_GROUP_PRIMARY_KEY = ["city", "timestamp"]

# --- Forecast horizon ---
FORECAST_HOURS_AHEAD = 72  # 3 days