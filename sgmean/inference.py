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

Also: ``alternative`` (two-sided / less / greater), a paired test on the
differences (``ttest_rel_trimmed``) and a robust effect size
(``effect_size_trimmed``, Algina-Keselman-Penfield type).

Requires SciPy (``pip install sgmean[inference]``) for the t distribution.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from numpy.typing import ArrayLike
import math
import warnings
from collections import namedtuple
from fractions import Fraction

import numpy as np

from ._input import to_array
from .core import sgmean
from . import _tables as _t_

_ALTS = ("two-sided", "less", "greater")
_PLABEL = {
    "two-sided": ("P-value (two-sided)", "Valor P (dos colas)"),
    "less": ("P-value (one-sided, less)", "Valor P (una cola, menor)"),
    "greater": ("P-value (one-sided, greater)", "Valor P (una cola, mayor)"),
}


def _check_alt(alternative):
    if alternative not in _ALTS:
        raise ValueError(f"alternative must be one of {_ALTS}")


def _t():
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover
        raise ImportError("sgmean inference needs SciPy: "
                          "pip install sgmean[inference]") from exc
    return stats.t


def _pvalue(tstat, df, alternative):
    t = _t()
    if alternative == "two-sided":
        return float(2 * t.sf(abs(tstat), df))
    if alternative == "greater":
        return float(t.sf(tstat, df))
    return float(t.cdf(tstat, df))


def _interval(est, se, df, conf, alternative):
    t = _t()
    if alternative == "two-sided":
        q = float(t.ppf(0.5 + conf / 2.0, df))
        return est - q * se, est + q * se
    q = float(t.ppf(conf, df))
    if alternative == "greater":
        return est - q * se, math.inf
    return -math.inf, est + q * se


# ---------------------------------------------------------------- results

_TTestBase = namedtuple("TTestResult",
                        ["statistic", "pvalue", "df", "estimate", "se"])


class TTestResult(_TTestBase):
    """One-sample (or paired) result. Unpacks like a tuple; ``print(r)`` shows a
    table (``r.table(lang="es")`` for Spanish)."""

    kind = "one-sample"
    alternative = "two-sided"

    def table(self, lang=None):
        lg = _t_.lang_of(lang)
        n, tr, mu0, conf = self.n, self.trim, self.mu0, self.conf
        lo, hi = self.ci
        paired = self.kind == "paired"
        rows = [
            ("Pairs" if paired else "Sample size",
             "Pares" if paired else "Tamaño de muestra", str(n)),
            ("Trim per tail", "Recorte por cola",
             f"{_t_.pct(tr, lg)} (k = {_t_.num(tr * n, lg, 4)})"),
            ("Trimmed mean of differences" if paired else "Trimmed mean",
             "Media recortada de las diferencias" if paired else "Media recortada",
             _t_.num(self.estimate, lg)),
            ("Hypothesized value" if paired else "Hypothesized mean",
             "Valor hipotético" if paired else "Media hipotética", _t_.num(mu0, lg)),
            ("Standard error", "Error estándar", _t_.num(self.se, lg)),
            ("Degrees of freedom", "Grados de libertad", _t_.num(self.df, lg)),
            ("t statistic", "Estadístico t", _t_.num(self.statistic, lg)),
            _PLABEL[self.alternative] + (_t_.pval(self.pvalue, lg),),
            (f"{_t_.num(conf * 100, lg)}% confidence interval",
             f"Intervalo de confianza {_t_.num(conf * 100, lg)}%",
             f"({_t_.num(lo, lg)}, {_t_.num(hi, lg)})"),
        ]
        title = (("Paired trimmed-mean t test (proportional trimming)",
                  "Prueba t pareada para medias recortadas (recorte proporcional)")
                 if paired else
                 ("Trimmed-mean t test (proportional trimming)",
                  "Prueba t para la media recortada (recorte proporcional)"))
        return _t_.render(title, rows, lg)

    __str__ = table

    def __repr__(self):
        return self.table()


class CIResult(tuple):
    """Confidence interval; unpacks as ``lower, upper``. ``print`` shows a table."""

    def __new__(cls, lower, upper, estimate, se, conf, trim, n):
        self = tuple.__new__(cls, (lower, upper))
        self.lower, self.upper, self.estimate = lower, upper, estimate
        self.se, self.conf, self.trim, self.n = se, conf, trim, n
        return self

    def table(self, lang=None):
        lg = _t_.lang_of(lang)
        rows = [
            ("Sample size", "Tamaño de muestra", str(self.n)),
            ("Trim per tail", "Recorte por cola",
             f"{_t_.pct(self.trim, lg)} (k = {_t_.num(self.trim * self.n, lg, 4)})"),
            ("Trimmed mean", "Media recortada", _t_.num(self.estimate, lg)),
            ("Standard error", "Error estándar", _t_.num(self.se, lg)),
            ("Confidence level", "Nivel de confianza", _t_.pct(self.conf, lg)),
            ("Lower limit", "Límite inferior", _t_.num(self.lower, lg)),
            ("Upper limit", "Límite superior", _t_.num(self.upper, lg)),
        ]
        return _t_.render(("Confidence interval for the trimmed mean",
                           "Intervalo de confianza para la media recortada"), rows, lg)

    __str__ = table

    def __repr__(self):
        return self.table()


_TwoBase = namedtuple("TwoSampleResult",
                      ["statistic", "pvalue", "df", "difference", "se", "ci"])


class TwoSampleResult(_TwoBase):
    """Two-sample result. Unpacks like a tuple; ``print(r)`` shows a table."""

    alternative = "two-sided"
    effect_size = float("nan")

    def table(self, lang=None):
        lg = _t_.lang_of(lang)
        lo, hi = self.ci
        rows = [
            ("Sample sizes (x, y)", "Tamaños de muestra (x, y)", f"{self.nx}, {self.ny}"),
            ("Trim per tail", "Recorte por cola", _t_.pct(self.trim, lg)),
            ("Trimmed mean of x", "Media recortada de x", _t_.num(self.tx, lg)),
            ("Trimmed mean of y", "Media recortada de y", _t_.num(self.ty, lg)),
            ("Difference (x - y)", "Diferencia (x - y)", _t_.num(self.difference, lg)),
            ("Standard error", "Error estándar", _t_.num(self.se, lg)),
            ("Degrees of freedom", "Grados de libertad", _t_.num(self.df, lg)),
            ("t statistic", "Estadístico t", _t_.num(self.statistic, lg)),
            _PLABEL[self.alternative] + (_t_.pval(self.pvalue, lg),),
            (f"{_t_.num(self.conf * 100, lg)}% CI for the difference",
             f"IC {_t_.num(self.conf * 100, lg)}% de la diferencia",
             f"({_t_.num(lo, lg)}, {_t_.num(hi, lg)})"),
            ("Robust effect size", "Tamaño del efecto robusto",
             _t_.num(self.effect_size, lg, 4)),
        ]
        return _t_.render(("Two-sample trimmed-mean test (Yuen type, proportional trimming)",
                           "Prueba de dos muestras para medias recortadas (tipo Yuen, recorte proporcional)"),
                          rows, lg)

    __str__ = table

    def __repr__(self):
        return self.table()


# ---------------------------------------------------------------- core

def _prepare(x, trim, nan_policy="raise"):
    if not 0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5)")
    x = np.sort(to_array(x, nan_policy))
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
    if h < 8:
        warnings.warn(f"very small effective sample size (n - 2k = {h:.3g}); "
                      "intervals and p-values may be unreliable", UserWarning,
                      stacklevel=3)
    return x, n, g, delta, h


def _ssw(x, g):
    """Sum of squares of the sample winsorized with g values per tail."""
    n = x.size
    y = x.copy()
    if g > 0:
        y[:g] = x[g]
        y[n - g:] = x[n - g - 1]
    return float(((y - y.mean()) ** 2).sum())


def _se(x, g, delta, h):
    den = h * (h - 1.0)
    se = (1 - delta) * math.sqrt(_ssw(x, g) / den)
    if delta > 0:
        se += delta * math.sqrt(_ssw(x, g + 1) / den)
    return se


def se_trimmed(x: ArrayLike, trim: float = 0.05, nan_policy: str = "raise") -> float:
    """Standard error of the proportional trimmed mean (method D).

    Equals the classical Tukey-McLaughlin SE when ``trim * n`` is an integer.
    """
    xs, n, g, delta, h = _prepare(x, trim, nan_policy)
    return _se(xs, g, delta, h)


def ci_trimmed(x: ArrayLike, trim: float = 0.05, conf: float = 0.95,
               nan_policy: str = "raise") -> "CIResult":
    """Two-sided confidence interval ``(lower, upper)`` for the trimmed mean."""
    if not 0 < conf < 1:
        raise ValueError("conf must be in (0, 1)")
    xs, n, g, delta, h = _prepare(x, trim, nan_policy)
    est = sgmean(xs, trim)
    se = _se(xs, g, delta, h)
    lo, hi = _interval(est, se, h - 1.0, conf, "two-sided")
    return CIResult(lo, hi, est, se, conf, trim, n)


def _one_sample(xs, n, g, delta, h, trim, mu0, conf, alternative):
    est = sgmean(xs, trim)
    se = _se(xs, g, delta, h)
    if se == 0:
        raise ValueError("zero standard error (constant winsorized sample)")
    tstat = (est - mu0) / se
    df = h - 1.0
    r = TTestResult(float(tstat), _pvalue(tstat, df, alternative), df, est, se)
    r.n, r.trim, r.mu0, r.conf, r.alternative = n, trim, mu0, conf, alternative
    r.ci = _interval(est, se, df, conf, alternative)
    return r


def ttest_trimmed(x: ArrayLike, mu0: float = 0.0, trim: float = 0.05,
                  conf: float = 0.95, alternative: str = "two-sided",
                  nan_policy: str = "raise") -> "TTestResult":
    """One-sample t test of ``H0: trimmed mean == mu0``.

    ``alternative``: ``"two-sided"`` (default), ``"less"`` or ``"greater"``
    (one-sided tests give a one-sided interval, as ``scipy.stats``).
    Returns a ``TTestResult`` ``(statistic, pvalue, df, estimate, se)``.
    """
    _check_alt(alternative)
    if not 0 < conf < 1:
        raise ValueError("conf must be in (0, 1)")
    xs, n, g, delta, h = _prepare(x, trim, nan_policy)
    return _one_sample(xs, n, g, delta, h, trim, mu0, conf, alternative)


def ttest_rel_trimmed(x: ArrayLike, y: ArrayLike, trim: float = 0.05,
                      conf: float = 0.95, alternative: str = "two-sided",
                      nan_policy: str = "raise") -> "TTestResult":
    """Paired test: one-sample test on the differences ``x - y``.

    Tests ``H0: trimmed mean of (x - y) == 0``. Note this is the trimmed mean of
    the differences, which in general is *not* the difference of the trimmed
    means. With ``nan_policy="omit"`` a pair is dropped when either value is NaN.
    """
    _check_alt(alternative)
    if not 0 < conf < 1:
        raise ValueError("conf must be in (0, 1)")
    a = to_array(x, "propagate", 0, "x")
    b = to_array(y, "propagate", 0, "y")
    if a.size != b.size:
        raise ValueError("x and y must have the same length")
    d = a - b
    xs, n, g, delta, h = _prepare(d, trim, nan_policy)
    r = _one_sample(xs, n, g, delta, h, trim, 0.0, conf, alternative)
    r.kind = "paired"
    return r


def _wvar(xs, g, delta):
    """Sample variance of the winsorized sample, interpolated like the SE."""
    n = xs.size

    def v(m):
        return _ssw(xs, m) / (n - 1.0)
    return (1 - delta) * v(g) + (delta * v(g + 1) if delta > 0 else 0.0)


def _akp_constant(trim):
    """sd of the winsorized standard normal: rescales the winsorized sd so that
    it estimates sigma under normality (0.642 for 20 % trimming)."""
    if trim == 0:
        return 1.0
    from scipy import stats
    z = float(stats.norm.ppf(1 - trim))
    phi = float(stats.norm.pdf(z))
    return math.sqrt(1 - 2 * trim - 2 * z * phi + 2 * trim * z * z)


def effect_size_trimmed(x: ArrayLike, y: ArrayLike, trim: float = 0.05,
                        nan_policy: str = "raise") -> float:
    """Robust standardized difference of trimmed means (Algina, Keselman &
    Penfield, 2005, generalised to fractional k).

    ``d = c(trim) * (T_x - T_y) / S_w`` with ``S_w`` the pooled winsorized sd and
    ``c(trim)`` the sd of the winsorized standard normal (0.642 at 20 %); for
    ``trim = 0`` it is Cohen's d with the pooled sd. For fractional ``k`` the
    winsorized variance is interpolated between ``floor(k)`` and ``floor(k)+1``
    (heuristic, like the standard error).
    """
    xs, n1, g1, d1, _ = _prepare(x, trim, nan_policy)
    ys, n2, g2, d2, _ = _prepare(y, trim, nan_policy)
    v1, v2 = _wvar(xs, g1, d1), _wvar(ys, g2, d2)
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if pooled == 0:
        raise ValueError("zero pooled winsorized variance")
    return float(_akp_constant(trim) * (sgmean(xs, trim) - sgmean(ys, trim))
                 / math.sqrt(pooled))


def ttest_ind_trimmed(x: ArrayLike, y: ArrayLike, trim: float = 0.05,
                      conf: float = 0.95, alternative: str = "two-sided",
                      nan_policy: str = "raise") -> "TwoSampleResult":
    """Yuen-type two-sample test (unequal variances) with proportional trimming.

    ``t = (T_x - T_y) / sqrt(se_x**2 + se_y**2)`` with the method-D standard
    errors and Welch-Satterthwaite degrees of freedom based on ``h - 1``.
    With integer ``k`` it reduces to Yuen's (1974) test as implemented in
    ``scipy.stats.ttest_ind(..., equal_var=False, trim=...)``.  For fractional
    ``k`` it is a **heuristic** extension (size and power studied by
    simulation, see ``docs/sim_power_twosample.csv``; no proof).
    ``alternative`` is ``"two-sided"``, ``"less"`` or ``"greater"`` (for
    ``x - y``). The result carries a robust ``effect_size``.
    Returns ``(statistic, pvalue, df, difference, se, ci)``.
    """
    _check_alt(alternative)
    if not 0 < conf < 1:
        raise ValueError("conf must be in (0, 1)")
    xs, n1, g1, d1, h1 = _prepare(x, trim, nan_policy)
    ys, n2, g2, d2, h2 = _prepare(y, trim, nan_policy)
    s1 = _se(xs, g1, d1, h1)
    s2 = _se(ys, g2, d2, h2)
    v = s1 ** 2 + s2 ** 2
    if v == 0:
        raise ValueError("zero standard error")
    tx, ty = sgmean(xs, trim), sgmean(ys, trim)
    diff = tx - ty
    se = math.sqrt(v)
    df = v ** 2 / (s1 ** 4 / (h1 - 1.0) + s2 ** 4 / (h2 - 1.0))
    stat = diff / se
    r = TwoSampleResult(float(stat), _pvalue(stat, df, alternative), float(df),
                        float(diff), se, _interval(diff, se, df, conf, alternative))
    r.nx, r.ny, r.trim, r.conf, r.alternative = n1, n2, trim, conf, alternative
    r.tx, r.ty = tx, ty
    try:
        r.effect_size = effect_size_trimmed(xs, ys, trim)
    except ValueError:
        pass
    return r
