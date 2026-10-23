import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from scipy import stats
from sgmean import (sgmean, diagnose_truncation, plot_trim_profile,
                    ttest_ind_trimmed)

X15 = np.array([850, 920, 980, 1050, 1120, 1180, 1250, 1320, 1400,
                1480, 1550, 1700, 1850, 2100, 8500], float)


def test_diag_truncated_case():
    d = diagnose_truncation(X15, 0.05)       # k = 0.75, floor -> 0
    assert d.truncated and d.k_floor == 0 and d.effective_trim == 0
    assert d.floor_mean == pytest.approx(X15.mean())
    assert d.proportional == pytest.approx(sgmean(X15, 0.05))
    assert d.difference == pytest.approx(d.floor_mean - d.proportional)


def test_diag_integer_case_no_difference():
    x = np.random.default_rng(0).normal(size=20)
    d = diagnose_truncation(x, 0.1)
    assert not d.truncated and d.difference == pytest.approx(0, abs=1e-12)


def test_diag_floor_matches_scipy():
    x = np.random.default_rng(1).normal(size=23)
    assert diagnose_truncation(x, 0.1).floor_mean == pytest.approx(stats.trim_mean(x, 0.1))


def test_plot_runs():
    ax = plot_trim_profile(X15, trim=0.05)
    assert len(ax.lines) >= 2


@pytest.mark.parametrize("n1,n2,trim", [(20, 30, 0.1), (30, 30, 0.2), (40, 30, 0.1)])
def test_yuen_matches_scipy_integer_k(n1, n2, trim):
    r = np.random.default_rng(n1)
    x, y = r.normal(0, 1, n1), r.normal(0.5, 2, n2)
    res = ttest_ind_trimmed(x, y, trim)
    ref = stats.ttest_ind(x, y, equal_var=False, trim=trim)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-9)
    assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-6)
    assert res.df == pytest.approx(ref.df, rel=1e-9)


def test_twosample_fractional_runs_and_ci():
    r = np.random.default_rng(5)
    x, y = r.lognormal(0, 1, 17), r.lognormal(0.3, 1, 22)
    res = ttest_ind_trimmed(x, y, 0.05)
    assert res.ci[0] < res.difference < res.ci[1]
    assert 0 <= res.pvalue <= 1
