import numpy as np
import pytest
from sgmean import (ci_trimmed, ttest_trimmed, ttest_ind_trimmed,
                    diagnose_truncation, sgmean)

X = np.array([850, 920, 980, 1050, 1120, 1180, 1250, 1320, 1400,
              1480, 1550, 1700, 1850, 2100, 8500], float)
Y = np.array([900, 1010, 1100, 1180, 1230, 1300, 1390, 1450, 1520, 1600, 1680,
              1760, 1900, 2050, 2200, 2400, 2650, 3000, 3400, 4100, 5200, 9800], float)


def test_tuple_behaviour_preserved():
    lo, hi = ci_trimmed(X, 0.05)
    assert lo == pytest.approx(1018.6064048126743) and hi == pytest.approx(1979.541743335474)
    assert len(ci_trimmed(X, 0.05)) == 2 and ci_trimmed(X, 0.05)[0] == lo
    t, p, df, est, se = ttest_trimmed(X, 1000, 0.05)
    assert t == pytest.approx(2.2531889971243326) and df == 12.5
    assert est == pytest.approx(sgmean(X, 0.05))
    r = ttest_ind_trimmed(X, Y, 0.05)
    assert r.statistic == pytest.approx(-1.6219980625502974)
    assert r.ci[0] < r.difference < r.ci[1]


def test_tables_english():
    s = str(ttest_trimmed(X, 1000, 0.05))
    for word in ("Trimmed-mean t test", "Sample size", "P-value", "5% (k = 0.75)", "1499.07", "0.0429"):
        assert word in s
    assert repr(ttest_trimmed(X, 1000, 0.05)) == s
    assert "Confidence interval" in str(ci_trimmed(X, 0.05))
    d = str(diagnose_truncation(X, 0.05))
    assert "1816.67" in d and "truncates" in d and "21.19%" in d
    assert "Yuen type" in str(ttest_ind_trimmed(X, Y, 0.05))


def test_tables_spanish_and_decimal_comma():
    s = ttest_trimmed(X, 1000, 0.05).table(lang="es")
    assert "Prueba t" in s and "1499,07" in s and "Grados de libertad" in s
    assert "Veredicto" in diagnose_truncation(X, 0.05).table(lang="es")
    assert "Tamaños de muestra" in ttest_ind_trimmed(X, Y, 0.05).table(lang="es")


def test_alignment_and_integer_k_verdict():
    lines = str(ttest_ind_trimmed(X, Y, 0.05)).splitlines()
    assert len(set(len(l) for l in lines[1:])) == 1       # underline and rows same width
    x = np.random.default_rng(0).normal(size=20)
    assert "agree" in str(diagnose_truncation(x, 0.1))


def test_conf_in_ttest_and_ci():
    r90 = ttest_trimmed(X, 1000, 0.05, conf=0.90)
    assert "90% confidence interval" in str(r90)
    lo90, hi90 = ci_trimmed(X, 0.05, 0.90)
    assert (lo90, hi90) == pytest.approx(r90.ci)
