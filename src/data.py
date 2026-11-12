"""Download hourly data from Open-Meteo and aggregate it to daily values.

Data sources (both CC BY 4.0, attribution in the README):
- Air quality: Open-Meteo air-quality API. This is CAMS (Copernicus
  Atmosphere Monitoring Service) global atmospheric composition MODEL
  output at ~45 km resolution - modelled concentrations, not
  ground-station measurements.
- Weather: Open-Meteo historical archive.

Nothing is bundled: the first run downloads JSON for the pinned window
and caches it in data/, which is git-ignored. The window is pinned so
the numbers in the README reproduce exactly.
"""

import json
import os
import urllib.parse
import urllib.request

import numpy as np

LATITUDE = 18.52   # Pune, India
LONGITUDE = 73.86
CITY = "Pune, India"
TIMEZONE = "Asia/Kolkata"
WINDOW = ("2025-11-13", "2026-11-12")  # pinned evaluation window

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://archive-api.open-meteo.com/v1/archive"

MIN_VALID_HOURS = 18  # a day needs at least this many hourly readings


def _fetch(url, params, cache_name, cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, cache_name)
    if not os.path.exists(path):
        qs = urllib.parse.urlencode(params)
        print(f"downloading {cache_name} from Open-Meteo (one time only)...")
        with urllib.request.urlopen(f"{url}?{qs}", timeout=60) as r:
            payload = json.loads(r.read().decode())
        with open(path, "w") as f:
            json.dump(payload, f)
    with open(path) as f:
        return json.load(f)


def _daily_means(times, values, agg=np.mean):
    """Aggregate hourly values to per-date values; NaN-aware."""
    by_date = {}
    for t, v in zip(times, values):
        by_date.setdefault(t[:10], []).append(np.nan if v is None else float(v))
    dates = sorted(by_date)
    out, hours = [], []
    for d in dates:
        arr = np.array(by_date[d])
        hours.append(int((~np.isnan(arr)).sum()))
        with np.errstate(all="ignore"):
            out.append(float(agg(arr)) if hours[-1] else np.nan)
    return dates, np.array(out), np.array(hours)


def load_daily(cache_dir="data"):
    """Return a dict of daily series for the pinned window."""
    aq = _fetch(AQ_URL, {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "hourly": "pm2_5,pm10", "start_date": WINDOW[0], "end_date": WINDOW[1],
        "timezone": TIMEZONE,
    }, f"aq_{WINDOW[0]}_{WINDOW[1]}.json", cache_dir)
    wx = _fetch(WX_URL, {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "start_date": WINDOW[0], "end_date": WINDOW[1],
        "timezone": TIMEZONE,
    }, f"wx_{WINDOW[0]}_{WINDOW[1]}.json", cache_dir)

    dates, pm25, aq_hours = _daily_means(aq["hourly"]["time"], aq["hourly"]["pm2_5"])
    _, pm25_max, _ = _daily_means(aq["hourly"]["time"], aq["hourly"]["pm2_5"], np.max)
    _, pm25_min, _ = _daily_means(aq["hourly"]["time"], aq["hourly"]["pm2_5"], np.min)
    _, temp, _ = _daily_means(wx["hourly"]["time"], wx["hourly"]["temperature_2m"])
    _, humidity, _ = _daily_means(wx["hourly"]["time"], wx["hourly"]["relative_humidity_2m"])
    _, wind, _ = _daily_means(wx["hourly"]["time"], wx["hourly"]["wind_speed_10m"])

    return {
        "dates": dates,
        "pm25_mean": pm25,
        "pm25_max": pm25_max,
        "pm25_min": pm25_min,
        "temp_mean": temp,
        "humidity_mean": humidity,
        "wind_mean": wind,
        "valid": aq_hours >= MIN_VALID_HOURS,
    }
