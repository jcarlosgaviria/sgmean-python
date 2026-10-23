import numpy as np, pytest
from sgmean.quantiles import quantile, quartiles, iqr, _METHOD

A=[2,5,7,11,13,20,24,31,45,80]; B=[3,8,9,14,17,25,33,41,90]; C=[4,6,9,12,15,22,30,71]

D=[2.35,3.10,4.72,5.05,6.48,7.91,8.26,9.63,11.07,12.84,15.30,21.75]
E=[0.125,0.480,0.731,1.205,1.610,2.034,2.388,3.147,4.560,6.902,9.318]
F=[7.2,3.4,3.4,9.8,5.5,5.5,5.5,12.1,8.6,4.9,15.7,6.3,6.3,10.2]
G=[1.8,2.4,2.9,3.3,3.7,4.1,4.6,5.2,5.8,6.5,7.3,8.2,9.4,10.9,12.7,15.1,18.6,23.4,31.9,52.3]

@pytest.mark.parametrize("data,expected", [
    (A,(7,16.5,31)),(B,(9,17,33)),(C,(7.5,13.5,26)),          # Statgraphics, verified 2026-10-20
    (D,(4.885,8.085,11.955)),(E,(0.731,2.034,4.56)),
    (F,(5.5,6.3,9.8)),(G,(3.9,6.9,13.9))])
def test_statgraphics(data, expected):
    assert quartiles(data) == pytest.approx(expected)

@pytest.mark.parametrize("t", range(1,10))
@pytest.mark.parametrize("n", [1,2,3,4,5,8,9,10,11,12,25,100])
def test_matches_numpy(t, n):
    rng = np.random.default_rng(n*10+t)
    x = rng.normal(size=n)
    p = [0,0.05,0.1,0.25,0.333,0.5,0.75,0.9,0.95,1]
    assert quantile(x,p,type=t) == pytest.approx(np.percentile(x,[q*100 for q in p],method=_METHOD[t]))

def test_ties_and_iqr():
    assert iqr(A) == 24
    assert quantile([5,5,5,5],0.3,type=2) == 5

def test_statgraphics_percentiles_n20():
    # Statgraphics Centurion, Percentiles table, n=20 (verified 2026-10-21)
    p = [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]
    want = [1.8,2.1,2.65,3.9,6.9,13.9,27.65,42.1,52.3]
    assert quantile(G, p, type=2) == pytest.approx(want)


from fractions import Fraction

def _user_rule(xs, Q):
    """pos = (n*Q + 2)/4: integer -> x[pos]; ends in .5 -> mean of neighbours; else nearest."""
    n = len(xs); pos = Fraction(n*Q + 2, 4)
    if pos.denominator == 1: return xs[int(pos)-1]
    if pos.denominator == 2:
        k = int(pos); return (xs[k-1] + xs[k]) / 2
    return xs[round(pos)-1]

def test_user_position_rule_equals_type2():
    rng = np.random.default_rng(7)
    for n in range(1, 301):
        xs = sorted(rng.normal(size=n).round(3).tolist())
        for Q in (1, 2, 3):
            assert quantile(xs, Q/4, type=2) == pytest.approx(_user_rule(xs, Q), abs=1e-12)

def test_type2_exact_when_float_product_is_inexact():
    # 30 * 0.1 == 3.0000000000000004 in floating point; n*p is exactly 3.
    xs = list(range(1, 31))
    assert quantile(xs, 0.1, type=2) == 3.5
    # every p = k/100 must follow the exact rule, whatever the float rounding
    for n in (10, 20, 30, 50, 70, 100):
        xs = list(range(1, n+1))
        for k in range(1, 100):
            e = Fraction(n*k, 100)
            want = 0.5*(e + e + 1) if e.denominator == 1 else float(-(-e.numerator // e.denominator))
            assert quantile(xs, k/100, type=2) == pytest.approx(want)


# Statgraphics Centurion, Summary Statistics, sextiles 1/6 and 5/6 (verified 2026-10-21)
@pytest.mark.parametrize("data,expected", [
    (A,(5,45)),(B,(8,41)),(C,(6,30)),(D,(3.91,14.07)),(E,(0.48,6.902)),
    (F,(4.9,10.2)),(G,(3.3,18.6))])
def test_statgraphics_sextiles(data, expected):
    assert quantile(data, [1/6, 5/6], type=2) == pytest.approx(expected)


def test_docstring_sextile_examples_n11_n12():
    # (n*S+3)/6 is the correct position; (n*S+2)/6 would contradict Statgraphics.
    assert quantile(E, 5/6, type=2) == pytest.approx(6.902)      # not 5.731
    assert quantile(D, 5/6, type=2) == pytest.approx(14.07)      # not 12.84
    assert (11*5 + 3) / 6 == pytest.approx(9.6667, abs=1e-4)
    assert (12*5 + 3) / 6 == 10.5


@pytest.mark.parametrize("data,want", [
    (E,[0.125,0.125,0.48,0.731,2.034,4.56,6.902,9.318,9.318]),                  # n=11
    (D,[2.35,2.35,3.1,4.885,8.085,11.955,15.3,21.75,21.75])])                   # n=12
def test_statgraphics_percentiles_n11_n12(data, want):
    # Statgraphics Centurion, Percentiles table (verified 2026-10-21)
    p = [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]
    assert quantile(data, p, type=2) == pytest.approx(want)
