"""Walk-forward backtest of next-day PM2.5 forecasts.

For each of the last N days in the pinned window, the forecast uses only
data from earlier days: the model is retrained every REFIT_EVERY test
days on all rows before the current one.

Issuance rule: the forecast for day D+1 is issued at the end of day D
(IST), so every feature is a realised value from day D or earlier. No
future weather or future pollution is ever used as an input.

The model predicts the CHANGE from today's PM2.5 (a persistence-anchored
correction). Both the delta formulation and the configuration below were
chosen by an internal walk-forward on the training slice only
(src/sweep.py) before the test block was ever scored.

Two baselines are scored the same way: persistence (tomorrow = today)
and weekly climatology (mean of the previous 7 daily means).

Run: python src/backtest.py   (from the repo root)
"""

import csv
import json
import os

import numpy as np

from data import load_daily, CITY, WINDOW
from features import build_rows, FEATURE_NAMES
from gbdt import GradientBoostingRegressor

TEST_DAYS = 120
REFIT_EVERY = 7
CONFIG = dict(n_estimators=200, learning_rate=0.03, max_depth=2,
              min_samples_split=20, subsample=0.8, seed=42)


def metrics(actual, predicted):
    err = predicted - actual
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
    }


def fit_model(X, y):
    """Fit the persistence-anchored model on rows X[:], y[:]."""
    model = GradientBoostingRegressor(**CONFIG)
    model.fit(X, y - X[:, 0])  # learn the change from pm25_today
    return model


def predict_model(model, row):
    return float(row[0] + model.predict(row.reshape(1, -1))[0])


def main():
    daily = load_daily()
    X, y, target_dates, skipped = build_rows(daily)
    print(f"{CITY}, pinned window {WINDOW[0]}..{WINDOW[1]}")
    print(f"{len(y)} usable days after dropping {skipped} with missing readings")

    test_start = len(y) - TEST_DAYS
    preds = np.zeros(len(y))
    refits = 0
    for j in range(test_start, len(y)):
        if (j - test_start) % REFIT_EVERY == 0:
            model = fit_model(X[:j], y[:j])
            refits += 1
        preds[j] = predict_model(model, X[j])

    actual = y[test_start:]
    persistence = X[test_start:, 0]   # pm25_today
    weekly = X[test_start:, 4]        # pm25_mean_last7

    results = {
        "city": CITY,
        "window": WINDOW,
        "test_days": int(len(actual)),
        "model": {"config": "delta target, 200 trees, lr 0.03, depth 2, "
                            "min_samples_split 20, subsample 0.8, seed 42",
                  **metrics(actual, preds[test_start:])},
        "persistence": metrics(actual, persistence),
        "weekly_climatology": metrics(actual, weekly),
    }
    print(f"walk-forward over {len(actual)} test days ({refits} refits):")
    for name in ("model", "persistence", "weekly_climatology"):
        r = results[name]
        print(f"  {name:<19} MAE {r['mae']:6.2f}  RMSE {r['rmse']:6.2f}  bias {r['bias']:+5.2f}")

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("results/backtest_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "actual_pm25", "model", "persistence", "weekly_climatology"])
        for k in range(len(actual)):
            w.writerow([target_dates[test_start + k], f"{actual[k]:.2f}",
                        f"{preds[test_start + k]:.2f}", f"{persistence[k]:.2f}",
                        f"{weekly[k]:.2f}"])
    print("wrote results/summary.json and results/backtest_predictions.csv")


if __name__ == "__main__":
    main()
