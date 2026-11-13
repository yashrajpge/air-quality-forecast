"""Distribution-free prediction intervals from scratch.

Split conformal: the radius of a (1-alpha) interval is the right
quantile of absolute residuals on a calibration window of genuine
out-of-sample predictions. No distribution assumption. Compared here
against a Gaussian interval (mean +- z * std of trailing residuals),
which does assume normality.
"""

import math

import numpy as np

Z = {0.20: 1.2816, 0.10: 1.6449}  # two-sided z for 80% and 90% intervals


def conformal_radius(abs_residuals, alpha):
    """Conformal quantile of calibration scores.

    Uses the standard finite-sample level ceil((n+1)*(1-alpha))/n so
    coverage is guaranteed at least 1-alpha on exchangeable data.
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
