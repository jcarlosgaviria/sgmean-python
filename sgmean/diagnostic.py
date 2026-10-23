"""Truncation diagnostic and trim-profile plot.

``diagnose_truncation`` reports how much the usual integer rule
``k = floor(trim * n)`` (R ``mean``, SciPy ``trim_mean``) differs from the
proportional trimmed mean.  ``plot_trim_profile`` draws both estimators as a
function of ``trim`` (needs matplotlib, optional).
"""
import math
from collections import namedtuple
from fractions import Fraction

import numpy as np

from .core import sgmean

TruncationDiagnostic = namedtuple(
    "TruncationDiagnostic",
    ["n", "trim", "k", "k_floor", "delta", "effective_trim",
     "proportional", "floor_mean", "difference", "relative_difference",
     "truncated"])


def _floor_mean(x, trim):
    """Trimmed mean with k = floor(trim*n) (SciPy / R convention)."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    g = math.floor(Fraction(trim).limit_denominator(10**9) * n)
    if n - 2 * g <= 0:
        raise ValueError("not enough observations for this trim")
    return float(x[g:n - g].mean())


def diagnose_truncation(x, trim=0.05):
    """Compare the proportional trimmed mean with the floor(k) one.

    ``truncated`` is True when ``trim * n`` is not an integer, i.e. when the
    integer rule silently trims less than requested (``effective_trim <
    trim``).  ``relative_difference`` is ``difference`` over the proportional
    value (nan if that value is 0).
    """
    if not 0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5)")
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0 or np.isnan(x).any():
        raise ValueError("x must be non-empty and free of NaN")
    n = x.size
    kk = Fraction(trim).limit_denominator(10**9) * n
    g = math.floor(kk)
    delta = float(kk - g)
    prop = sgmean(x, trim)
    fl = _floor_mean(x, trim)
    diff = fl - prop
    rel = diff / prop if prop != 0 else float("nan")
    return TruncationDiagnostic(n, trim, float(kk), g, delta, g / n, prop, fl,
                                diff, rel, delta > 0)


def plot_trim_profile(x, trim=None, trims=None, ax=None):
    """Plot proportional (smooth) vs floor (step) trimmed mean against trim.

    Parameters
    ----------
    x : array-like
    trim : float, optional
        Chosen trim; drawn as a vertical line.
    trims : array-like, optional
        Grid of trims in [0, 0.5); default 200 points.
    ax : matplotlib Axes, optional

    Returns the Axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot_trim_profile needs matplotlib: "
                          "pip install sgmean[plots]") from exc
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 3 or np.isnan(x).any():
        raise ValueError("need at least 3 non-NaN observations")
    if trims is None:
        trims = np.linspace(0, 0.5, 201)[:-1]
    trims = np.asarray(trims, dtype=float)
    # keep a grid that is valid for this n (at least one obs. left)
    trims = trims[trims * n < (n - 1) / 2.0 + 1e-12]
    prop = [sgmean(x, t) for t in trims]
    flo = [_floor_mean(x, t) for t in trims]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.6))
    ax.step(trims, flo, where="post", color="#c0504d", lw=1.6,
            label="floor(k): R / SciPy")
    ax.plot(trims, prop, color="#1f4e79", lw=1.8, label="proportional (sgmean)")
    if trim is not None:
        ax.axvline(trim, color="grey", ls=":", lw=1)
    ax.set_xlabel("trim (proportion per tail)")
    ax.set_ylabel("trimmed mean")
    ax.set_title("Trimmed mean vs trim, n = %d" % n)
    ax.legend(frameon=False)
    return ax
