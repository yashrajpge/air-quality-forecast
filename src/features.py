"""Build the supervised learning matrix from the daily series.

One row per day D; every feature uses only data from day D or earlier,
and the target is the mean PM2.5 of day D+1. Rows touching invalid days
(too few hourly readings) are dropped.
"""

import datetime

import numpy as np

FEATURE_NAMES = [
    "pm25_today", "pm25_yesterday", "pm25_2days_ago",
    "pm25_mean_last3", "pm25_mean_last7",
    "pm25_max_today", "pm25_min_today",
    "temp_mean", "humidity_mean", "wind_mean",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
]


def build_rows(daily):
    """Return (X, y, target_dates, skipped_days)."""
    dates = daily["dates"]
    n = len(dates)
    pm = daily["pm25_mean"]
    valid = daily["valid"] & ~np.isnan(pm)

    X, y, target_dates, skipped = [], [], [], 0
    for i in range(6, n - 1):
        window_ok = valid[i - 6 : i + 2].all()  # features need i-6..i, target needs i+1
        if not window_ok:
            skipped += 1
            continue
        d = datetime.date.fromisoformat(dates[i])
        dow = d.weekday()
        doy = d.timetuple().tm_yday
        X.append([
            pm[i], pm[i - 1], pm[i - 2],
            pm[i - 2 : i + 1].mean(), pm[i - 6 : i + 1].mean(),
            daily["pm25_max"][i], daily["pm25_min"][i],
            daily["temp_mean"][i], daily["humidity_mean"][i], daily["wind_mean"][i],
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        ])
        y.append(pm[i + 1])
        target_dates.append(dates[i + 1])
    return np.array(X), np.array(y), target_dates, skipped
