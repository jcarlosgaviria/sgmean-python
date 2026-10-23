"""Winsorized mean and winsorized sigma, compatible with Statgraphics.

Statgraphics Centurion (Summary Statistics) was found, on seven data sets
(n = 8..20, trims of 5 % and 10 %), to behave as follows:

* ``k = floor(trim * n)`` observations are replaced at *each* tail
  (integer truncation: no fractional treatment, unlike its trimmed mean).
* Each of the ``k`` smallest values is replaced by the (k+1)-th smallest
  and each of the ``k`` largest by the (k+1)-th largest.
* Winsorized mean = mean of the winsorized sample.
* Winsorized sigma = sample standard deviation (ddof=1) of the winsorized
  sample times ``sqrt((n - 1) / (n - 2k - 1))``.
"""
import math
from fractions import Fraction
import numpy as np


def _k(n, trim):
    if not 0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5)")
    return math.floor(n * Fraction(trim).limit_denominator(10**9))


def _winsorize(x, trim):
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    if n == 0 or np.isnan(x).any():
        raise ValueError("x must be non-empty and free of NaN")
    k = _k(n, trim)
    w = x.copy()
    if k > 0:
        w[:k] = x[k]
        w[n - k:] = x[n - k - 1]
    return w, k


def winsorized_mean(x, trim=0.05):
    """Mean of ``x`` after replacing ``floor(trim*n)`` values at each tail."""
    w, _ = _winsorize(x, trim)
    return float(w.mean())


def winsorized_sigma(x, trim=0.05):
    """Winsorized standard deviation as reported by Statgraphics."""
    w, k = _winsorize(x, trim)
    n = w.size
    if n - 2 * k - 1 <= 0:
        raise ValueError("not enough observations for this trim")
    return float(w.std(ddof=1) * math.sqrt((n - 1) / (n - 2 * k - 1)))
