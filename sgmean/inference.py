"""Standard error, confidence interval and one-sample test for the
proportional trimmed mean (``sgmean``) when ``k = trim * n`` is not an integer.

Method (practical proposal, "D")
--------------------------------
Let ``T = sgmean(x, trim)``, ``k = trim * n``, ``g = floor(k)``,
``delta = k - g`` and ``h = n - 2k`` (effective sample size, possibly
fractional).  For an integer ``m`` let ``SSW(m)`` be the sum of squared
deviations of the sample winsorized with ``m`` values replaced at each tail
(the ``m`` smallest by the (m+1)-th smallest and the ``m`` largest by the
(m+1)-th largest).  The classical Tukey-McLaughlin standard error for an
integer trim is ``sqrt(SSW(k) / (h (h - 1)))``.  For fractional ``k`` we use

    SE = (1 - delta) * SE(g) + delta * SE(g + 1),

with ``SE(m) = sqrt(SSW(m) / (h (h - 1)))`` and the *fractional* ``h``.  The
interval and the test use a Student t distribution with ``h - 1`` degrees of
freedom (Tukey & McLaughlin, 1963; Yuen, 1974).

Properties: continuous in ``trim`` and identical to the classical SE when
``k`` is an integer.  For non-integer ``k`` the interpolation is a
**heuristic**: it is supported by Monte Carlo pilot simulations (see
``docs/``) but has no theoretical proof; treat it as a practical proposal.

Requires SciPy (``pip install sgmean[inference]``) for the t distribution.
"""
import math
from collections import namedtuple
from fractions import Fraction

import numpy as np

from .core import sgmean

TTestResult = namedtuple("TTestResult",
                         ["statistic", "pvalue", "df", "estimate", "se"])


def _prepare(x, trim):
    if not 0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5)")
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0 or np.isnan(x).any():
        raise ValueError("x must be non-empty and free of NaN")
    x = np.sort(x)
    n = x.size
    k = Fraction(trim).limit_denominator(10**9) * n   # exact k
    g = math.floor(k)
    delta = float(k - g)
    h = float(n - 2 * k)
    if h <= 1:
        raise ValueError("not enough observations for this trim "
                         "(n - 2k must be > 1)")
    if delta > 0 and 2 * (g + 1) >= n:
        raise ValueError("trim too large for this n (need 2*ceil(k) < n)")
    return x, n, g, delta, h


def _ssw(x, g):
    """Sum of squares of the sample winsorized with g values per tail."""
    n = x.size
    y = x.copy()
    if g > 0:
        y[:g] = x[g]
        y[n - g:] = x[n - g - 1]
    return float(((y - y.mean()) ** 2).sum())


def se_trimmed(x, trim=0.05):
    """Standard error of the proportional trimmed mean (method D).

    Equals the classical Tukey-McLaughlin SE when ``trim * n`` is an integer.
    """
    x, n, g, delta, h = _prepare(x, trim)
    den = h * (h - 1.0)
    se = (1 - delta) * math.sqrt(_ssw(x, g) / den)
    if delta > 0:
        se += delta * math.sqrt(_ssw(x, g + 1) / den)
    return se


def _t():
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover
        raise ImportError("sgmean inference needs SciPy: "
                          "pip install sgmean[inference]") from exc
    return stats.t


def ci_trimmed(x, trim=0.05, conf=0.95):
    """Confidence interval ``(lower, upper)`` for the trimmed mean."""
    if not 0 < conf < 1:
        raise ValueError("conf must be in (0, 1)")
    _, n, _, _, h = _prepare(x, trim)
    est = sgmean(x, trim)
    se = se_trimmed(x, trim)
    q = float(_t().ppf(0.5 + conf / 2.0, h - 1.0))
    return est - q * se, est + q * se


def ttest_trimmed(x, mu0=0.0, trim=0.05):
    """Two-sided one-sample t test of ``H0: trimmed mean == mu0``.

    Returns a namedtuple ``(statistic, pvalue, df, estimate, se)``.
    """
    _, n, _, _, h = _prepare(x, trim)
    est = sgmean(x, trim)
    se = se_trimmed(x, trim)
    if se == 0:
        raise ValueError("zero standard error (constant winsorized sample)")
    tstat = (est - mu0) / se
    df = h - 1.0
    p = float(2 * _t().sf(abs(tstat), df))
    return TTestResult(float(tstat), p, df, est, se)


TwoSampleResult = namedtuple(
    "TwoSampleResult",
    ["statistic", "pvalue", "df", "difference", "se", "ci"])


def ttest_ind_trimmed(x, y, trim=0.05, conf=0.95):
    """Yuen-type two-sample test (unequal variances) with proportional trimming.

    ``t = (T_x - T_y) / sqrt(se_x**2 + se_y**2)`` with the method-D standard
    errors and Welch-Satterthwaite degrees of freedom based on ``h - 1``.
    With integer ``k`` it reduces to Yuen's (1974) test as implemented in
    ``scipy.stats.ttest_ind(..., equal_var=False, trim=...)``.  For fractional
    ``k`` it is a **heuristic** extension (size and power studied by
    simulation, see ``docs/sim_power_twosample.csv``; no proof).
    Returns ``(statistic, pvalue, df, difference, se, ci)``.
    """
    if not 0 < conf < 1:
        raise ValueError("conf must be in (0, 1)")
    _, _, _, _, h1 = _prepare(x, trim)
    _, _, _, _, h2 = _prepare(y, trim)
    s1 = se_trimmed(x, trim)
    s2 = se_trimmed(y, trim)
    v = s1 ** 2 + s2 ** 2
    if v == 0:
        raise ValueError("zero standard error")
    diff = sgmean(x, trim) - sgmean(y, trim)
    se = math.sqrt(v)
    df = v ** 2 / (s1 ** 4 / (h1 - 1.0) + s2 ** 4 / (h2 - 1.0))
    t = _t()
    stat = diff / se
    p = float(2 * t.sf(abs(stat), df))
    q = float(t.ppf(0.5 + conf / 2.0, df))
    return TwoSampleResult(float(stat), p, float(df), float(diff), se,
                           (diff - q * se, diff + q * se))
