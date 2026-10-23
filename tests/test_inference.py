import math
import numpy as np
import pytest
from sgmean import sgmean
from sgmean.inference import se_trimmed, ci_trimmed, ttest_trimmed

X15 = np.array([850, 920, 980, 1050, 1120, 1180, 1250, 1320, 1400,
                1480, 1550, 1700, 1850, 2100, 8500], float)


def classical_se(x, k):
    """Independent Tukey-McLaughlin SE for integer k."""
    x = np.sort(np.asarray(x, float)); n = len(x)
    y = x.copy()
    if k:
        y[:k] = x[k]; y[n - k:] = x[n - k - 1]
    h = n - 2 * k
    return math.sqrt(((y - y.mean()) ** 2).sum() / (h * (h - 1)))


@pytest.mark.parametrize("n,trim,k", [(20, 0.1, 2), (20, 0.2, 4), (30, 0.1, 3),
                                      (25, 0.2, 5), (12, 0.0, 0)])
def test_equals_classical_when_k_integer(n, trim, k):
    x = np.random.default_rng(n).normal(size=n)
    assert se_trimmed(x, trim) == pytest.approx(classical_se(x, k), rel=1e-12)


def test_continuity_in_trim():
    x = np.random.default_rng(1).lognormal(size=40)
    for a in (0.05, 0.1, 0.2):
        eps = 1e-7
        assert se_trimmed(x, a + eps) == pytest.approx(se_trimmed(x, a - eps), rel=1e-4)


def test_between_A_and_B():
    x = np.random.default_rng(2).normal(size=15)
    se = se_trimmed(x, 0.05)  # k = 0.75
    a, b = classical_se(x, 0), classical_se(x, 1)
    h = 15 - 1.5
    f = math.sqrt((13.0 * 12.0) / (h * (h - 1)))  # converts h-denominators
    seA = classical_se(x, 0) * math.sqrt(15 * 14) / math.sqrt(h * (h - 1))
    seB = classical_se(x, 1) * math.sqrt(13 * 12) / math.sqrt(h * (h - 1))
    assert se == pytest.approx(0.25 * seA + 0.75 * seB, rel=1e-12)


def test_ci_and_ttest_consistent():
    lo, hi = ci_trimmed(X15, 0.05)
    est = sgmean(X15, 0.05)
    assert lo < est < hi and est - lo == pytest.approx(hi - est)
    r = ttest_trimmed(X15, mu0=est, trim=0.05)
    assert r.pvalue == pytest.approx(1.0) and r.statistic == pytest.approx(0.0)
    r2 = ttest_trimmed(X15, mu0=hi, trim=0.05)
    assert r2.pvalue == pytest.approx(0.05, abs=1e-9)


def test_matches_scipy_t_interval_integer_k():
    from scipy import stats
    x = np.random.default_rng(3).normal(5, 2, 30)
    lo, hi = ci_trimmed(x, 0.1, 0.9)
    h = 24
    q = stats.t.ppf(0.95, h - 1)
    est, se = sgmean(x, 0.1), classical_se(x, 3)
    assert (lo, hi) == pytest.approx((est - q * se, est + q * se))


def test_errors():
    with pytest.raises(ValueError):
        se_trimmed([1, 2, 3], 0.5)
    with pytest.raises(ValueError):
        se_trimmed([1, 2, np.nan, 4], 0.1)
    with pytest.raises(ValueError):
        se_trimmed([1, 2], 0.45)
    with pytest.raises(ValueError):
        ci_trimmed(X15, 0.05, conf=1.5)
