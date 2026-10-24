"""Sample quantiles with the nine Hyndman & Fan (1996) definitions.

``type`` follows the numbering of R's ``quantile()``. Statgraphics'
"Summary Statistics" quartiles are reproduced by ``type=2``.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from numpy.typing import ArrayLike
import math
from fractions import Fraction
import numpy as np

from ._input import to_array

_METHOD = {1: "inverted_cdf", 2: "averaged_inverted_cdf",
           3: "closest_observation", 4: "interpolated_inverted_cdf",
           5: "hazen", 6: "weibull", 7: "linear",
           8: "median_unbiased", 9: "normal_unbiased"}

# Continuous types 4-9: h = a + (n + 1 - a - b) * p, Q = x[floor(h)] + g*(x[floor(h)+1]-x[floor(h)])
_AB = {4: (0.0, 1.0), 5: (0.5, 0.5), 6: (0.0, 0.0), 7: (1.0, 1.0),
       8: (1 / 3, 1 / 3), 9: (3 / 8, 3 / 8)}


def _exact(p):
    """Exact rational value of a probability (0.1 -> 1/10, not 0.1000000000000000055...)."""
    return Fraction(p).limit_denominator(10**9)


def _one(xs, n, p, type):
    if type == 2:
        # Exact integer arithmetic: no floating-point tolerance is needed
        # to decide whether n*p is an integer. See quantile() for the rule.
        np_ = n * _exact(p)
        if np_.denominator == 1:                       # n*p integer -> average
            j = int(np_)
            lo, hi = min(max(j, 1), n), min(max(j + 1, 1), n)
            return 0.5 * (xs[lo - 1] + xs[hi - 1])
        return xs[min(max(math.ceil(np_), 1), n) - 1]  # otherwise x[ceil(n*p)]
    if type in (1, 3):
        np_ = n * p
        eps = 4 * np.finfo(float).eps
        j = math.floor(np_ + eps)
        g = np_ - j
        if abs(g) < eps * max(1.0, np_):
            g = 0.0
        if type == 1:
            idx = j if g == 0 else j + 1
            return xs[min(max(idx, 1), n) - 1]
        # type 3: nearest even order statistic (R: m = -0.5)
        h = np_ - 0.5
        j3 = math.floor(h)
        g3 = h - j3
        idx = j3 if (g3 == 0 and j3 % 2 == 0) else j3 + 1
        return xs[min(max(idx, 1), n) - 1]
    a, b = _AB[type]
    h = a + (n + 1 - a - b) * p
    j = math.floor(h + 4 * np.finfo(float).eps)
    g = h - j
    lo = xs[min(max(j, 1), n) - 1]
    hi = xs[min(max(j + 1, 1), n) - 1]
    return lo + g * (hi - lo)


def quantile(x: ArrayLike, probs: Any = (0.25, 0.5, 0.75), type: int = 2,
             nan_policy: str = "raise") -> Any:
    """Quantiles of ``x`` using Hyndman-Fan definition ``type`` (1-9).

    Default ``type=2`` (R: ``quantile(x, type=2)``; NumPy:
    ``method='averaged_inverted_cdf'``) reproduces the quartiles of
    Statgraphics' Summary Statistics procedure in every case tested
    (see the test-suite).

    Position rule for ``type=2`` (Gaviria-Chaverra's formulation). With the
    sorted sample ``x[1] <= ... <= x[n]`` and a probability ``p``, the position
    of the quantile is::

        pos = n*p + 1/2

    * ``pos`` integer        -> the quantile is ``x[pos]``.
    * ``pos`` ends in ``.5`` -> the mean of ``x[floor(pos)]`` and ``x[floor(pos)+1]``.
    * otherwise              -> the observation at the nearest position
                                (``3.4`` -> ``x[3]``; ``3.6`` -> ``x[4]``).

    For the quartiles (``p = Q/4``, ``Q`` = 1, 2, 3) this is ``(n*Q + 2)/4``;
    for the sextiles (``p = S/6``, ``S`` = 1..5) it is ``(n*S + 3)/6``.
    The constant in the numerator is **not** universal: it is ``denominator/2``
    (2/4 and 3/6), because the offset is always 1/2 in position units.
    Copying the quartile form with a different denominator, ``(n*S + 2)/6``,
    is wrong and contradicts Statgraphics (see the example below).

    Worked example, 5/6 sextil (Statgraphics gives 6.902 and 14.07)::

        n = 11: x = 0.125 0.480 0.731 1.205 1.610 2.034 2.388 3.147 4.560 6.902 9.318
                pos = (11*5 + 3)/6 = 9.667  -> nearest position 10 -> x[10] = 6.902
                ((11*5 + 2)/6 = 9.5 would average x[9], x[10] = 5.731: wrong)
        n = 12: x = 2.35 3.10 4.72 5.05 6.48 7.91 8.26 9.63 11.07 12.84 15.30 21.75
                pos = (12*5 + 3)/6 = 10.5   -> mean of x[10] and x[11]
                    = (12.84 + 15.30)/2 = 14.07
                ((12*5 + 2)/6 = 10.333 would give x[10] = 12.84: wrong)

    Equivalently, this is the Hyndman & Fan type-2 definition: ``x[ceil(n*p)]``
    when ``n*p`` is not an integer, and the mean of ``x[n*p]`` and
    ``x[n*p + 1]`` when it is. The equality was checked with exact arithmetic
    for n = 1..300 and Q = 1, 2, 3, and the rule was checked against the
    quartiles and sextiles of Statgraphics Centurion (Summary Statistics) on
    seven data sets (n = 8..20). That shows Statgraphics behaves *consistently
    with* type 2; it does not prove which formula the program uses internally,
    and other Statgraphics procedures were not tested.

    Note: R and NumPy default to ``type=7``, which gives different values.
    """
    if type not in _METHOD:
        raise ValueError("type must be an integer between 1 and 9")
    xs = np.sort(to_array(x, nan_policy))
    p = np.atleast_1d(np.asarray(probs, dtype=float))
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probs must be in [0, 1]")
    out = np.array([_one(xs, xs.size, float(q), type) for q in p])
    return out if np.ndim(probs) else float(out[0])


def quartiles(x: ArrayLike, type: int = 2, nan_policy: str = "raise") -> Any:
    """Return (Q1, median, Q3). Default ``type=2`` matches Statgraphics."""
    q = quantile(x, (0.25, 0.5, 0.75), type, nan_policy)
    return float(q[0]), float(q[1]), float(q[2])


def iqr(x: ArrayLike, type: int = 2, nan_policy: str = "raise") -> float:
    q1, _, q3 = quartiles(x, type, nan_policy)
    return q3 - q1
