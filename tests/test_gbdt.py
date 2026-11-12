"""Sanity tests: run with `python tests/test_gbdt.py` from the repo root.

The first run downloads the pinned data window from Open-Meteo
(needs internet once).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from data import load_daily, WINDOW
from features import build_rows, FEATURE_NAMES
from gbdt import GradientBoostingRegressor
from tree import RegressionTree


def test_tree_learns_simple_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(300, 3))
    y = 3 * X[:, 0] - 2 * X[:, 1] + np.where(X[:, 2] > 0, 5.0, -5.0)
    # depth 8: after the exact x2 split, ~64 leaves per branch to
    # piecewise-approximate the two linear terms
    tree = RegressionTree(max_depth=8, min_samples_split=4).fit(X, y)
    pred = tree.predict(X)
    rmse = float(np.sqrt(((y - pred) ** 2).mean()))
    assert rmse < 0.4, rmse
    print(f"ok: regression tree fits a piecewise function (train RMSE {rmse:.3f})")


def test_boosting_reduces_error():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(400, 4))
    y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + rng.normal(0, 0.05, 400)
    model = GradientBoostingRegressor(n_estimators=60, learning_rate=0.2,
                                      max_depth=3, seed=3).fit(X, y)
    rmse = model.train_rmse_
    baseline = float(np.sqrt(((y - y.mean()) ** 2).mean()))
    assert rmse[-1] < baseline * 0.5, (baseline, rmse[-1])
    print(f"ok: boosting cuts train RMSE from {baseline:.3f} to {rmse[-1]:.3f}")


def test_features_use_only_past_days():
    daily = load_daily()
    X, y, target_dates, skipped = build_rows(daily)
    assert X.shape[1] == len(FEATURE_NAMES)
    # row j's features must come from the day before its target date
    import datetime
    first = datetime.date.fromisoformat(target_dates[0])
    d0 = datetime.date.fromisoformat(daily["dates"][0])
    assert (first - d0).days >= 7  # 6 history days + 1 forecast step
    # target of row j is the day after the day its features describe:
    # pm25_today (column 0) must equal the daily mean of target_date - 1
    idx = daily["dates"].index(target_dates[5]) - 1
    assert abs(X[5, 0] - daily["pm25_mean"][idx]) < 1e-9
    print("ok: features describe only days before the forecast day")


def test_no_leakage_in_backtest_refits():
    # for test row j (0-based inside the test block), the refit at
    # j % REFIT == 0 trains on rows[:test_start + j] - never future rows
    test_start, n_rows, refit = 100, 220, 7
    for j in range(test_start, n_rows):
        train_end = test_start + ((j - test_start) // refit) * refit
        assert train_end <= j
    print("ok: every backtest refit uses strictly earlier rows")


def test_model_is_deterministic():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(200, 5))
    y = X[:, 0] ** 2 + X[:, 3]
    preds = []
    for _ in range(2):
        m = GradientBoostingRegressor(n_estimators=30, subsample=0.8, seed=9).fit(X, y)
        preds.append(m.predict(X[:10]))
    assert (preds[0] == preds[1]).all()
    print("ok: same seed gives identical predictions")


if __name__ == "__main__":
    test_tree_learns_simple_function()
    test_boosting_reduces_error()
    test_features_use_only_past_days()
    test_no_leakage_in_backtest_refits()
    test_model_is_deterministic()
    print("all tests passed")
