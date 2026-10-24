"""Descriptive statistics compatible with Statgraphics Centurion (Summary Statistics).

Every definition below was checked against the Statgraphics output for seven data
sets (n = 8, 9, 10, 11, 12, 14, 20); see ``tests/test_descriptive.py``.
Only NumPy is required.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from numpy.typing import ArrayLike
import math
import warnings
from collections import Counter
import numpy as np

from ._input import to_array
from .core import sgmean
from .quantiles import quartiles, quantile
from .winsor import winsorized_mean, winsorized_sigma
from .report import SummaryResult, MultiSummaryResult


def _clean(x, min_n=1, nan_policy="raise"):
    return to_array(x, nan_policy, min_n)


def _as_columns(x):
    """Dict name -> 1-D data when ``x`` holds several variables, else ``None``."""
    if isinstance(x, dict):
        return {str(k): v for k, v in x.items()}
    if hasattr(x, "columns"):                       # pandas DataFrame
        cols = {}
        skipped = []
        for c in x.columns:
            col = x[c]
            kind = getattr(getattr(col, "dtype", None), "kind", "f")
            if kind in "biuf" or str(getattr(col, "dtype", "")) in (
                    "Int64", "Float64", "Int32", "Float32", "boolean"):
                cols[str(c)] = col
            else:
                skipped.append(str(c))
        if skipped:
            warnings.warn("non-numeric columns skipped: " + ", ".join(skipped))
        if not cols:
            raise ValueError("the DataFrame has no numeric columns")
        return cols
    arr = x if isinstance(x, np.ndarray) else None
    if arr is not None and arr.ndim == 2 and arr.shape[1] > 1:
        return {f"x{j + 1}": arr[:, j] for j in range(arr.shape[1])}
    return None


def geometric_mean(x: ArrayLike, nan_policy: str = "raise") -> float:
    """exp(mean(log x)); all observations must be strictly positive."""
    x = _clean(x, 1, nan_policy)
    if (x <= 0).any():
        raise ValueError("geometric mean requires strictly positive data")
    return float(np.exp(np.log(x).mean()))


def dam(x: ArrayLike, nan_policy: str = "raise") -> float:
    """Median absolute deviation about the median, *not* rescaled (Statgraphics 'DAM')."""
    x = _clean(x, 1, nan_policy)
    return float(np.median(np.abs(x - np.median(x))))


def sbi(x: ArrayLike, c: float = 9.0, nan_policy: str = "raise") -> float:
    """Tukey biweight scale (square root of the biweight midvariance).

    Centre = median; scale = unscaled MAD; tuning constant ``c = 9``.
    Returns ``nan`` when the MAD is zero (Statgraphics leaves it blank; ``summary()``
    reports ``None``).
    """
    x = _clean(x, 1, nan_policy)
    m = np.median(x)
    mad = np.median(np.abs(x - m))
    if mad == 0:
        return float("nan")
    u = (x - m) / (c * mad)
    inside = np.abs(u) < 1
    num = np.sum(inside * (x - m) ** 2 * (1 - u ** 2) ** 4)
    den = abs(np.sum(inside * (1 - u ** 2) * (1 - 5 * u ** 2)))
    return float(math.sqrt(x.size * num) / den)


def skewness(x: ArrayLike, standardized: bool = False,
             nan_policy: str = "raise") -> float:
    """Adjusted Fisher-Pearson skewness G1 (needs n >= 3).

    ``standardized=True`` divides by ``sqrt(6/n)``.
    """
    x = _clean(x, 3, nan_policy)
    n = x.size
    z = (x - x.mean()) / x.std(ddof=1)
    g1 = n / ((n - 1) * (n - 2)) * np.sum(z ** 3)
    return float(g1 / math.sqrt(6 / n)) if standardized else float(g1)


def kurtosis(x: ArrayLike, standardized: bool = False,
             nan_policy: str = "raise") -> float:
    """Adjusted excess kurtosis G2 (needs n >= 4).

    ``standardized=True`` divides by ``sqrt(24/n)``.
    """
    x = _clean(x, 4, nan_policy)
    n = x.size
    z = (x - x.mean()) / x.std(ddof=1)
    g2 = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(z ** 4)
          - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return float(g2 / math.sqrt(24 / n)) if standardized else float(g2)


def cv(x: ArrayLike, nan_policy: str = "raise") -> float:
    """Coefficient of variation in percent: 100 * sd / mean."""
    x = _clean(x, 2, nan_policy)
    return float(100 * x.std(ddof=1) / x.mean())


def std_error(x: ArrayLike, nan_policy: str = "raise") -> float:
    """Standard error of the mean: sd / sqrt(n)."""
    x = _clean(x, 2, nan_policy)
    return float(x.std(ddof=1) / math.sqrt(x.size))


def mode(x: ArrayLike, nan_policy: str = "raise") -> float:
    """Most frequent value, or ``None`` unless exactly one value is the most frequent.

    Statgraphics shows the mode only when it is unique (also when it occurs just
    twice) and leaves it blank when every value is different or when several values
    tie for the highest frequency.
    """
    x = _clean(x, 1, nan_policy)
    counts = Counter(x.tolist())
    top = max(counts.values())
    winners = [v for v, k in counts.items() if k == top]
    if top == 1 or len(winners) > 1:
        return None
    return winners[0]


def summary(x: Any, trim: float = 0.05, quartile_type: int = 2,
            lang: Optional[str] = None, nan_policy: str = "raise") -> Any:
    """All Statgraphics 'Summary Statistics' results as an ordered dict.

    ``x`` can be a list, array or pandas Series (one variable) or a pandas
    DataFrame / dict of columns / 2-D array (several variables: a table with one
    column per variable is returned, see ``MultiSummaryResult``). ``nan_policy``
    is ``"raise"`` (default) or ``"omit"`` (each variable drops its own NaN).

    ``trim`` applies to the trimmed mean (proportional, via ``sgmean``) and to the
    winsorized mean and sigma (integer ``floor(trim*n)``, as Statgraphics does).
    Skewness needs n >= 3 and kurtosis n >= 4; they are ``None`` otherwise, as in
    Statgraphics (n = 3 leaves the kurtosis blank).

    Returns a ``SummaryResult`` (a dict that prints as an aligned table). ``lang`` selects
    the language of the printed labels: ``"en"`` (default), ``"es"`` or ``"auto"``; see
    ``sgmean.set_language``.
    """
    multi = _as_columns(x)
    if multi is not None:
        return MultiSummaryResult(
            {name: summary(col, trim, quartile_type, lang, nan_policy)
             for name, col in multi.items()}, trim=trim, lang=lang)
    x = _clean(x, 2, nan_policy)
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
