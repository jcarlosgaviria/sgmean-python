"""Descriptive statistics compatible with Statgraphics Centurion (Summary Statistics).

Every definition below was checked against the Statgraphics output for seven data
sets (n = 8, 9, 10, 11, 12, 14, 20); see ``tests/test_descriptive.py``.
Only NumPy is required.
"""
import math
from collections import Counter
import numpy as np

from .core import sgmean
from .quantiles import quartiles, quantile
from .winsor import winsorized_mean, winsorized_sigma
from .report import SummaryResult


def _clean(x, min_n=1):
    x = np.asarray(x, dtype=float).ravel()
    if x.size < min_n:
        raise ValueError(f"at least {min_n} observation(s) required")
    if np.isnan(x).any():
        raise ValueError("x must be free of NaN")
    return x


def geometric_mean(x):
    """exp(mean(log x)); all observations must be strictly positive."""
    x = _clean(x)
    if (x <= 0).any():
        raise ValueError("geometric mean requires strictly positive data")
    return float(np.exp(np.log(x).mean()))


def dam(x):
    """Median absolute deviation about the median, *not* rescaled (Statgraphics 'DAM')."""
    x = _clean(x)
    return float(np.median(np.abs(x - np.median(x))))


def sbi(x, c=9.0):
    """Tukey biweight scale (square root of the biweight midvariance).

    Centre = median; scale = unscaled MAD; tuning constant ``c = 9``.
    Returns ``nan`` when the MAD is zero (Statgraphics leaves it blank; ``summary()``
    reports ``None``).
    """
    x = _clean(x)
    m = np.median(x)
    mad = np.median(np.abs(x - m))
    if mad == 0:
        return float("nan")
    u = (x - m) / (c * mad)
    inside = np.abs(u) < 1
    num = np.sum(inside * (x - m) ** 2 * (1 - u ** 2) ** 4)
    den = abs(np.sum(inside * (1 - u ** 2) * (1 - 5 * u ** 2)))
    return float(math.sqrt(x.size * num) / den)


def skewness(x, standardized=False):
    """Adjusted Fisher-Pearson skewness G1 (needs n >= 3).

    ``standardized=True`` divides by ``sqrt(6/n)``.
    """
    x = _clean(x, 3)
    n = x.size
    z = (x - x.mean()) / x.std(ddof=1)
    g1 = n / ((n - 1) * (n - 2)) * np.sum(z ** 3)
    return float(g1 / math.sqrt(6 / n)) if standardized else float(g1)


def kurtosis(x, standardized=False):
    """Adjusted excess kurtosis G2 (needs n >= 4).

    ``standardized=True`` divides by ``sqrt(24/n)``.
    """
    x = _clean(x, 4)
    n = x.size
    z = (x - x.mean()) / x.std(ddof=1)
    g2 = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(z ** 4)
          - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return float(g2 / math.sqrt(24 / n)) if standardized else float(g2)


def cv(x):
    """Coefficient of variation in percent: 100 * sd / mean."""
    x = _clean(x, 2)
    return float(100 * x.std(ddof=1) / x.mean())


def std_error(x):
    """Standard error of the mean: sd / sqrt(n)."""
    x = _clean(x, 2)
    return float(x.std(ddof=1) / math.sqrt(x.size))


def mode(x):
    """Most frequent value, or ``None`` unless exactly one value is the most frequent.

    Statgraphics shows the mode only when it is unique (also when it occurs just
    twice) and leaves it blank when every value is different or when several values
    tie for the highest frequency.
    """
    x = _clean(x)
    counts = Counter(x.tolist())
    top = max(counts.values())
    winners = [v for v, k in counts.items() if k == top]
    if top == 1 or len(winners) > 1:
        return None
    return winners[0]


def summary(x, trim=0.05, quartile_type=2, lang=None):
    """All Statgraphics 'Summary Statistics' results as an ordered dict.

    ``trim`` applies to the trimmed mean (proportional, via ``sgmean``) and to the
    winsorized mean and sigma (integer ``floor(trim*n)``, as Statgraphics does).
    Skewness needs n >= 3 and kurtosis n >= 4; they are ``None`` otherwise, as in
    Statgraphics (n = 3 leaves the kurtosis blank).

    Returns a ``SummaryResult`` (a dict that prints as an aligned table). ``lang`` selects
    the language of the printed labels: ``"en"`` (default), ``"es"`` or ``"auto"``; see
    ``sgmean.set_language``.
    """
    x = _clean(x, 2)
    n = x.size
    q1, _, q3 = quartiles(x, quartile_type)
    s1, s5 = quantile(x, [1 / 6, 5 / 6], quartile_type)
    return SummaryResult({
        "count": n,
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "mode": mode(x),
        "geometric_mean": geometric_mean(x) if (x > 0).all() else None,
        "trimmed_mean": sgmean(x, trim=trim),
        "winsorized_mean": winsorized_mean(x, trim),
        "variance": float(x.var(ddof=1)),
        "std": float(x.std(ddof=1)),
        "cv": cv(x),
        "std_error": std_error(x),
        "winsorized_sigma": winsorized_sigma(x, trim),
        "dam": dam(x),
        "sbi": None if math.isnan(sbi(x)) else sbi(x),
        "min": float(x.min()),
        "max": float(x.max()),
        "range": float(x.max() - x.min()),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "sextile_1_6": float(s1),
        "sextile_5_6": float(s5),
        "intersextile_range": float(s5 - s1),
        "skewness": skewness(x) if n >= 3 else None,
        "std_skewness": skewness(x, True) if n >= 3 else None,
        "kurtosis": kurtosis(x) if n >= 4 else None,
        "std_kurtosis": kurtosis(x, True) if n >= 4 else None,
        "sum": float(x.sum()),
        "sum_squares": float((x ** 2).sum()),
    }, trim=trim, lang=lang)
