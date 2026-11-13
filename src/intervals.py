"""Prediction intervals from scratch.

Split-conformal-style quantile: the half-width of a nominal (1-alpha)
interval is a high quantile of absolute residuals on a calibration
window of genuine out-of-sample predictions. The classic finite-sample
coverage guarantee (at least 1-alpha) requires exchangeable calibration
scores; on autocorrelated time series like daily pollution residuals
that assumption fails, so the levels here are nominal targets and
coverage is reported as measured, not guaranteed. A Gaussian interval
(mean +- z * std of trailing residuals) is scored as a comparison; it
additionally assumes normality.
"""

import math

import numpy as np

Z = {0.20: 1.2816, 0.10: 1.6449}  # two-sided z for 80% and 90% intervals


def conformal_radius(abs_residuals, alpha):
    """Conformal-style quantile of calibration scores.

    Uses the standard rank ceil((n+1)*(1-alpha)) of the sorted scores,
    clipped to the sample range. The finite-sample coverage guarantee
    attached to this rank assumes exchangeable scores; with rolling
    time-series calibration treat the level as nominal.
    """
    scores = np.sort(np.asarray(abs_residuals, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("empty calibration window")
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    idx = min(n - 1, int(math.ceil(level * n)) - 1)
    return float(scores[idx])


def gaussian_radius(residuals, alpha):
    """z * sample std of the calibration residuals."""
    r = np.asarray(residuals, dtype=float)
    if len(r) < 2:
        raise ValueError("need at least 2 residuals")
    return Z[alpha] * float(np.std(r, ddof=1))


def coverage(actual, predicted, radius):
    """Fraction of days where |actual - predicted| <= radius."""
    return float((np.abs(np.asarray(actual) - np.asarray(predicted)) <= radius).mean())
