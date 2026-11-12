"""Pick the model formulation and configuration on the TRAIN slice only.

Runs an internal walk-forward over the last 100 training days for a few
hand-picked candidates - level models (predict PM2.5 directly) and delta
models (predict the change from today) - and prints their scores. The
test block is never touched here; the winner is hard-coded into
src/backtest.py and scored once.

Run from the repo root: python src/sweep.py
"""

import numpy as np

from data import load_daily
from features import build_rows
from gbdt import GradientBoostingRegressor
from backtest import metrics, fit_model, predict_model, CONFIG

LEVEL_CONFIGS = [
    # (n_estimators, learning_rate, max_depth, min_samples_split)
    (100, 0.05, 2, 20),
    (200, 0.03, 3, 20),
    (50, 0.10, 3, 10),
]
DELTA_CONFIGS = [
    (100, 0.05, 2, 20),
    (200, 0.03, 2, 20),
    (300, 0.02, 3, 20),
]


def walk_forward_level(X, y, start, cfg, refit_every=7):
    n_est, lr, depth, mss = cfg
    model = GradientBoostingRegressor(
        n_estimators=n_est, learning_rate=lr, max_depth=depth,
        min_samples_split=mss, subsample=0.8, seed=42,
    )
    preds = np.zeros(len(y))
    for j in range(start, len(y)):
        if (j - start) % refit_every == 0:
            model.fit(X[:j], y[:j])
        preds[j] = model.predict(X[j : j + 1])[0]
    return metrics(y[start:], preds[start:])


def walk_forward_delta(X, y, start, cfg, refit_every=7):
    cfg_d = dict(n_estimators=cfg[0], learning_rate=cfg[1], max_depth=cfg[2],
                 min_samples_split=cfg[3], subsample=0.8, seed=42)
    preds = np.zeros(len(y))
    for j in range(start, len(y)):
        if (j - start) % refit_every == 0:
            model = GradientBoostingRegressor(**cfg_d)
            model.fit(X[:j], y[:j] - X[:j, 0])
        preds[j] = X[j, 0] + model.predict(X[j : j + 1])[0]
    return metrics(y[start:], preds[start:])


def main():
    daily = load_daily()
    X, y, _, _ = build_rows(daily)
    n_train = len(y) - 120
    start = n_train - 100  # internal validation: last 100 train days
    persistence = metrics(y[start:n_train], X[start:n_train, 0])
    print(f"internal validation on train days {start}..{n_train - 1} "
          f"(persistence MAE {persistence['mae']:.2f} RMSE {persistence['rmse']:.2f})")
    for cfg in LEVEL_CONFIGS:
        r = walk_forward_level(X[:n_train], y[:n_train], start, cfg)
        print(f"  level n_est={cfg[0]:>3} lr={cfg[1]:.2f} depth={cfg[2]} mss={cfg[3]:>2} "
              f"-> MAE {r['mae']:.2f} RMSE {r['rmse']:.2f}")
    for cfg in DELTA_CONFIGS:
        r = walk_forward_delta(X[:n_train], y[:n_train], start, cfg)
        print(f"  delta n_est={cfg[0]:>3} lr={cfg[1]:.2f} depth={cfg[2]} mss={cfg[3]:>2} "
              f"-> MAE {r['mae']:.2f} RMSE {r['rmse']:.2f}")
    print("winner adopted in src/backtest.py CONFIG: "
          + results_summary())


def results_summary():
    return ("delta target, 200 trees, lr 0.03, depth 2, "
            "min_samples_split 20, subsample 0.8, seed 42")


if __name__ == "__main__":
    main()
