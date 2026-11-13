# air-quality-forecast

Next-day PM2.5 forecasting for Pune, India, with gradient-boosted
regression trees written from scratch in NumPy - no scikit-learn, no
XGBoost. Includes a leakage-safe walk-forward backtest against two
baselines, with every number below produced by the committed code.

## What the model does

- **Target:** the daily mean PM2.5 for *tomorrow* in Pune
  (18.52° N, 73.86° E).
- **Issuance rule:** the forecast for day D+1 is issued at the end of
  day D (IST). Every feature is a realised value from day D or earlier -
  no future weather or pollution is ever used as an input.
- **Model:** gradient boosting over hand-written regression trees
  (`src/tree.py`, `src/gbdt.py`). Splits are chosen by a vectorized
  squared-error scan; each boosting stage fits the residuals. The final
  configuration predicts the *change* from today's PM2.5 (a
  persistence-anchored correction): 200 trees, learning rate 0.03,
  depth 2, min_samples_split 20, subsample 0.8, seed 42.
- **Uncertainty:** every forecast also carries a nominal 90% prediction
  interval sized by a split-conformal-style quantile (from scratch,
  `src/intervals.py`) over a trailing 28-day window of genuine
  out-of-sample residuals, with a Gaussian-residual interval scored as
  a comparison. Caveat: the classic conformal coverage guarantee
  assumes exchangeable calibration scores; daily pollution residuals
  are autocorrelated, so here 90% is a nominal target and the measured
  coverage below is the honest evidence, not a guaranteed rate.
- **Features (14):** PM2.5 today/yesterday/2 days ago, 3- and 7-day
  rolling means, today's max and min, daily mean temperature, relative
  humidity and wind speed, and cyclic day-of-week / day-of-year
  encodings.

## Data provenance (read this before citing the results)

All data comes from [Open-Meteo](https://open-meteo.com) (free
non-commercial API, CC BY 4.0):

- **Air quality:** the Open-Meteo air-quality API serves **CAMS
  (Copernicus Atmosphere Monitoring Service) atmospheric composition
  model output** - at Pune's location the ~45 km global CAMS forecasts.
  These are **modelled concentrations, not ground-station
  measurements**. This project therefore forecasts a modelled pollution
  field (the same kind of field many public AQI dashboards display); it
  is **not** evidence about observed street-level pollution or public
  health. Validation against reference ground stations (e.g. CPCB
  monitors in Pune) is future work.
- **Weather covariates:** the Open-Meteo historical archive.
- Nothing is bundled. The first run downloads the pinned window
  (2025-11-13 to 2026-11-12, Asia/Kolkata) and caches it in `data/`
  (git-ignored). The window is pinned so the numbers below reproduce
  exactly.
- Note on the issuance rule: the backtest fixes the *information set*
  (only day-D-or-earlier values), not API latency - the Open-Meteo
  archive publishes with a lag, so a live system would need same-day
  sources for the current day's weather.

## Results (exact, reproducible)

Walk-forward backtest over the last **120 days** of the pinned window
(2026-07-16 to 2026-11-12): the model is retrained every 7 test days on
all earlier rows only, so no future data ever leaks into a forecast.
Baselines are scored on the same days. 358 of 365 days were usable (0
dropped for missing readings); 18 refits; seed 42.

| Forecaster         | MAE (µg/m³) | RMSE (µg/m³) | Bias  |
|--------------------|------------:|-------------:|------:|
| Model (NumPy GBDT) |        3.83 |         5.92 | +0.39 |
| Persistence        |        3.76 |         5.98 | -0.18 |
| Weekly climatology |        4.86 |         7.18 | -0.78 |

Persistence = "tomorrow equals today"; weekly climatology = mean of the
previous 7 daily means.

**Honest reading:** the model roughly matches persistence (slightly
higher MAE, slightly lower RMSE) and clearly beats weekly climatology.
At the daily horizon this CAMS modelled field is strongly persistent,
and the learned correction adds little - a result reported as measured,
not dressed up. Both the delta formulation and the configuration were
chosen by an internal walk-forward on the training slice only
(`src/sweep.py`) before the test block was scored once.

**Nominal 90% prediction intervals** (scored on the 92 test days that
follow a full 28-day calibration window of strictly earlier
out-of-sample residuals):

| Interval                    | Nominal | Measured coverage | Mean width |
|-----------------------------|--------:|------------------:|-----------:|
| Conformal-style quantile    |    90%  |             88.0% | 18.49 µg/m³ |
| Gaussian residuals          |    90%  |             82.6% | 15.79 µg/m³ |

The conformal-style interval came closer to its nominal level than the
Gaussian one over these 92 days. A plausible explanation is
heavy-tailed residuals, but that is a hypothesis - it was not tested,
and 92 days is a modest sample, so treat the gap as observed, not
explained. Neither interval carries a coverage guarantee here: the
finite-sample conformal guarantee needs exchangeable scores, which
autocorrelated daily residuals do not satisfy. Both half-widths are
committed per day in the results CSV.

Per-day forecasts: `results/backtest_predictions.csv` (committed).
Machine-readable summary: `results/summary.json` (committed).

## Repository layout

```
src/tree.py      regression tree from scratch (vectorized split search)
src/gbdt.py      gradient boosting over those trees (residual fitting)
src/intervals.py split-conformal + Gaussian prediction intervals
src/data.py      Open-Meteo download + hourly-to-daily aggregation
src/features.py  supervised matrix; features use only past days
src/sweep.py     train-slice-only model selection
src/backtest.py  walk-forward evaluation vs baselines
tests/           sanity tests, leakage checks, determinism check
results/         committed backtest output
```

## Run it

```bash
pip install -r requirements.txt   # numpy only
python tests/test_gbdt.py         # first run downloads data (needs internet once)
python src/sweep.py               # train-slice model selection
python src/backtest.py            # final walk-forward on the test block
```

The tests cover: tree fitting on a noise-free function, boosting
monotonically reducing training error, features using only days before
the forecast day, backtest refits never seeing future rows, exact
determinism under a fixed seed, and conformal quantile mechanics
producing ~90% coverage on iid draws (that test validates the
implementation in an iid setting only - it says nothing about coverage
on this autocorrelated time series).

## Attribution

- Air-quality data: **CAMS / Copernicus Atmosphere Monitoring Service**,
  served by Open-Meteo. Per Open-Meteo's terms, credit belongs to the
  CAMS ENSEMBLE data providers (see
  [Open-Meteo air-quality docs](https://open-meteo.com/en/docs/air-quality-api)).
- Weather data: [Open-Meteo](https://open-meteo.com) historical archive.
- Both datasets are CC BY 4.0; the API is free for non-commercial use.

## License

MIT © Yashraj Pangave
