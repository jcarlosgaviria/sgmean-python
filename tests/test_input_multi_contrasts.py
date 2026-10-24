import math
import warnings
import numpy as np
import pytest
from scipy import stats
import sgmean as sg
from sgmean import (sgmean, summary, quantile, winsorized_mean, ttest_trimmed,
                    ttest_ind_trimmed, ttest_rel_trimmed, effect_size_trimmed,
                    ci_trimmed, se_trimmed, diagnose_truncation)

X = np.array([850, 920, 980, 1050, 1120, 1180, 1250, 1320, 1400,
              1480, 1550, 1700, 1850, 2100, 8500], float)
Y = np.array([900, 1010, 1100, 1180, 1230, 1300, 1390, 1450, 1520, 1600, 1680,
              1760, 1900, 2050, 2200, 2400, 2650, 3000, 3400, 4100, 5200, 9800], float)


# ----- input: pandas, lists, NaN -----------------------------------------
def test_pandas_series_and_list_equal_numpy():
    pd = pytest.importorskip("pandas")
    s = pd.Series(X)
    assert sgmean(s, 0.05) == sgmean(X, 0.05) == sgmean(list(X), 0.05)
    assert se_trimmed(s, 0.05) == se_trimmed(X, 0.05)
    assert winsorized_mean(s, 0.1) == winsorized_mean(X, 0.1)
    assert quantile(s, 0.5) == quantile(X, 0.5)


def test_pandas_nullable_and_dataframe_column():
    pd = pytest.importorskip("pandas")
    s = pd.Series([1, 2, None, 4, 5, 6, 7, 8, 9, 10], dtype="Int64")
    assert sgmean(s, 0.1, nan_policy="omit") == sgmean([1, 2, 4, 5, 6, 7, 8, 9, 10], 0.1)
    df = pd.DataFrame({"a": X})
    assert sgmean(df, 0.05) == sgmean(X, 0.05)
    with pytest.raises(TypeError, match="columns"):
        sgmean(pd.DataFrame({"a": X, "b": X}), 0.05)


def test_nan_policies():
    xn = np.append(X, np.nan)
    assert math.isnan(sgmean(xn, 0.05))                       # default propagate
    assert sgmean(xn, 0.05, nan_policy="omit") == sgmean(X, 0.05)
    with pytest.raises(ValueError, match="NaN"):
        sgmean(xn, 0.05, nan_policy="raise")
    with pytest.raises(ValueError, match="NaN"):
        se_trimmed(xn, 0.05)                                  # default raise
    assert se_trimmed(xn, 0.05, nan_policy="omit") == se_trimmed(X, 0.05)
    assert diagnose_truncation(xn, 0.05, nan_policy="omit").n == 15
    with pytest.raises(ValueError):
        sgmean(X, 0.05, nan_policy="bogus")


# ----- summary of several variables --------------------------------------
def test_summary_dataframe_dict_and_2d():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": X, "y": X[::-1] * 2, "txt": ["a"] * 15})
    with pytest.warns(UserWarning, match="non-numeric"):
        m = summary(df, trim=0.05)
    assert list(m) == ["x", "y"]
    assert m["x"]["mean"] == pytest.approx(X.mean())
    assert m["y"]["mean"] == pytest.approx(X.mean() * 2)
    t = str(m)
    assert "Summary Statistics (2 variables)" in t and "Trimmed Mean 5%" in t
    assert "Resumen Estadístico (2 variables)" in m.table(lang="es")
    assert m.to_dataframe().loc["count", "x"] == 15
    m2 = summary({"a": X, "b": X[:10]})
    assert m2["b"]["count"] == 10
    m3 = summary(np.column_stack([X, X + 1]))
    assert list(m3) == ["x1", "x2"]
    lines = str(m).splitlines()
    assert len({len(l) for l in lines[1:]}) == 1


def test_summary_nan_omit_per_variable():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, np.nan], "b": [2.0, 3, 4, 5, 6, 7]})
    with pytest.raises(ValueError):
        summary(df)
    m = summary(df, nan_policy="omit")
    assert m["a"]["count"] == 5 and m["b"]["count"] == 6


def test_single_variable_summary_unchanged():
    assert type(summary(X)).__name__ == "SummaryResult"


# ----- alternative --------------------------------------------------------
def test_alternative_one_sample():
    r2 = ttest_trimmed(X, 1000, 0.05)
    rg = ttest_trimmed(X, 1000, 0.05, alternative="greater")
    rl = ttest_trimmed(X, 1000, 0.05, alternative="less")
    assert r2.statistic > 0
    assert rg.pvalue == pytest.approx(r2.pvalue / 2)
    assert rl.pvalue == pytest.approx(1 - r2.pvalue / 2)
    assert rg.ci[1] == math.inf and rl.ci[0] == -math.inf
    assert "one-sided, greater" in str(rg)
    with pytest.raises(ValueError):
        ttest_trimmed(X, 1000, 0.05, alternative="bigger")


def test_alternative_matches_scipy_integer_k():
    r = np.random.default_rng(1)
    x, y = r.normal(0, 1, 30), r.normal(0.4, 2, 30)
    for alt in ("two-sided", "less", "greater"):
        mine = ttest_ind_trimmed(x, y, 0.1, alternative=alt)
        ref = stats.ttest_ind(x, y, equal_var=False, trim=0.1, alternative=alt)
        assert mine.statistic == pytest.approx(ref.statistic)
        assert mine.pvalue == pytest.approx(ref.pvalue)
        assert mine.df == pytest.approx(ref.df)


# ----- paired -------------------------------------------------------------
def test_paired_equals_one_sample_on_differences():
    a = X
    b = X * 0.9 + np.arange(15)
    r = ttest_rel_trimmed(a, b, 0.1)
    ref = ttest_trimmed(a - b, 0.0, 0.1)
    assert tuple(r) == pytest.approx(tuple(ref))
    assert "Paired" in str(r) and "Pares" in r.table(lang="es")


def test_paired_trim0_matches_scipy_and_nan_pairs():
    r = np.random.default_rng(3)
    a, b = r.normal(size=25), r.normal(0.3, 1, 25)
    mine = ttest_rel_trimmed(a, b, 0.0)
    ref = stats.ttest_rel(a, b)
    assert mine.statistic == pytest.approx(ref.statistic)
    assert mine.pvalue == pytest.approx(ref.pvalue)
    a2, b2 = a.copy(), b.copy(); a2[3] = np.nan; b2[7] = np.nan
    om = ttest_rel_trimmed(a2, b2, 0.0, nan_policy="omit")
    keep = np.ones(25, bool); keep[[3, 7]] = False
    assert om.statistic == pytest.approx(stats.ttest_rel(a[keep], b[keep]).statistic)
    with pytest.raises(ValueError, match="same length"):
        ttest_rel_trimmed(a[:5], b[:6])


# ----- effect size --------------------------------------------------------
def test_effect_size_trim0_is_cohens_d():
    r = np.random.default_rng(5)
    a, b = r.normal(1, 2, 40), r.normal(0, 2, 35)
    sp = math.sqrt(((39) * a.var(ddof=1) + 34 * b.var(ddof=1)) / 73)
    assert effect_size_trimmed(a, b, 0.0) == pytest.approx((a.mean() - b.mean()) / sp)


def test_effect_size_constant_20pct_is_0642():
    from sgmean.inference import _akp_constant
    assert _akp_constant(0.2) == pytest.approx(0.6419, abs=1e-3)
    assert _akp_constant(0.0) == 1.0


def test_effect_size_in_result_and_normal_large_n():
    r = np.random.default_rng(6)
    a, b = r.normal(1, 1, 4000), r.normal(0, 1, 4000)
    d = effect_size_trimmed(a, b, 0.2)
    assert d == pytest.approx(1.0, abs=0.1)                  # estimates delta/sigma = 1
    res = ttest_ind_trimmed(a, b, 0.2)
    assert res.effect_size == pytest.approx(d)
    assert "Robust effect size" in str(res)


# ----- warnings -----------------------------------------------------------
def test_small_effective_sample_warns_but_normal_does_not():
    with pytest.warns(UserWarning, match="small effective sample"):
        se_trimmed([1, 2, 3, 4, 5, 6, 7], 0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        se_trimmed(X, 0.05)
