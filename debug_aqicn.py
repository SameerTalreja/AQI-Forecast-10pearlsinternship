"""One-off debug script: print raw AQICN JSON for Quetta and Faisalabad
so we can see exactly why pollutant data is missing.
Run from project root: python debug_aqicn.py
"""

import json
import requests
from src.config import AQICN_API_KEY, AQICN_BASE_URL, CITIES

for city_name in ["Quetta", "Faisalabad"]:
    station = CITIES[city_name]["station"]
    url = f"{AQICN_BASE_URL}/{station}/"
    resp = requests.get(url, params={"token": AQICN_API_KEY}, timeout=15)
    print(f"\n===== {city_name} ({station}) =====")
    print(json.dumps(resp.json(), indent=2))