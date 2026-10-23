import pytest
from sgmean.winsor import winsorized_mean, winsorized_sigma

DATA = {
 20:[1.8,2.4,2.9,3.3,3.7,4.1,4.6,5.2,5.8,6.5,7.3,8.2,9.4,10.9,12.7,15.1,18.6,23.4,31.9,52.3],
 14:[7.2,3.4,3.4,9.8,5.5,5.5,5.5,12.1,8.6,4.9,15.7,6.3,6.3,10.2],
 12:[2.35,3.10,4.72,5.05,6.48,7.91,8.26,9.63,11.07,12.84,15.30,21.75],
 11:[0.125,0.480,0.731,1.205,1.610,2.034,2.388,3.147,4.560,6.902,9.318],
 10:[2,5,7,11,13,20,24,31,45,80], 9:[3,8,9,14,17,25,33,41,90], 8:[4,6,9,12,15,22,30,71]}

# Statgraphics Centurion, Summary Statistics, 2026-10-21: (n, trim%) -> value.
MEAN = {(20,5):10.515,(20,10):9.715,(14,5):7.45714,(14,10):7.2,(12,5):9.03833,(12,10):8.56333,(11,5):2.95455,(11,10):2.76718,
        (10,5):23.8,(10,10):20.6,(9,5):26.6667,(9,10):26.6667,(8,5):21.125,(8,10):21.125}
# n=12 means: first reported by mistake from the *trimmed* mean row; re-read and corrected 2026-10-21.
SIGMA = {(20,5):9.76141,(20,10):8.20051,(14,5):3.48087,(14,10):3.16515,(12,5):5.58164,
         (12,10):4.80909,(11,5):2.89485,(11,10):2.66038,(10,5):23.7243,(10,10):17.4536,
         (9,5):26.7722,(9,10):26.7722,(8,5):21.8987,(8,10):21.8987}

@pytest.mark.parametrize("key,want", MEAN.items())
def test_winsorized_mean_statgraphics(key, want):
    n, p = key
    assert winsorized_mean(DATA[n], p/100) == pytest.approx(want, rel=2e-6)

@pytest.mark.parametrize("key,want", SIGMA.items())
def test_winsorized_sigma_statgraphics(key, want):
    n, p = key
    assert winsorized_sigma(DATA[n], p/100) == pytest.approx(want, rel=2e-6)

def test_exact_k_with_float_trim():
    x = list(range(1, 31))             # 30*0.1 = 3.0000000000000004 in floats
    assert winsorized_mean(x, 0.1) == pytest.approx(15.5)   # symmetric: unchanged mean
    assert winsorized_mean([1,2,3,4,100,5,6,7,8,9], 0.1) == pytest.approx(5.5)

def test_errors():
    with pytest.raises(ValueError): winsorized_mean([1,2,3], 0.5)
    with pytest.raises(ValueError): winsorized_mean([], 0.1)

def test_n12_reported_means_are_the_trimmed_mean():
    # The two n=12 'winsorized mean' cells (8.7037 and 8.40417) are exactly sgmean's
    # proportional trimmed mean, so they were copied from the trimmed-mean row.
    from sgmean import sgmean
    assert sgmean(DATA[12], trim=0.05) == pytest.approx(8.7037, rel=2e-6)
    assert sgmean(DATA[12], trim=0.10) == pytest.approx(8.40417, rel=2e-6)

@pytest.mark.parametrize("trim,k_expected", [(0.13, 2), (0.19, 3), (0.28, 5), (0.375, 7)])
def test_only_integer_part_of_k_is_used(trim, k_expected):
    # Rule described by the author (k = 2.6, 3.8, 5.6, 7.5 -> 2, 3, 5, 7 per tail).
    # Measured in Statgraphics for k = 2.6, 3.8 and 5.6 (n=20) and for k = 1.5, 2.5, 3.5
    # (n=10); k = 7.5 follows the author's description only (the program accepts only
    # integer percentages). n = 20 so that k = trim * 20.
    import numpy as np
    x = np.sort(np.array(DATA[20]))
    w = x.copy()
    w[:k_expected] = x[k_expected]
    w[20 - k_expected:] = x[20 - k_expected - 1]
    assert winsorized_mean(x, trim) == pytest.approx(w.mean())


@pytest.mark.parametrize("trim,mean,sigma", [(0.13, 9.715, 8.20051), (0.19, 9.055, 7.12055)])
def test_statgraphics_fractional_k_ge_half(trim, mean, sigma):
    # Statgraphics Centurion, n=20, 2026-10-21: k = 2.6 -> 2 and k = 3.8 -> 3 (truncation,
    # not rounding; rounding would give 9.055/7.12055 and 8.435/6.14525).
    assert winsorized_mean(DATA[20], trim) == pytest.approx(mean, rel=2e-6)
    assert winsorized_sigma(DATA[20], trim) == pytest.approx(sigma, rel=2e-6)


@pytest.mark.parametrize("n,trim,mean,sigma", [
    (20, 0.28, 7.935, 5.38625),       # k = 5.6 -> 5 (rounding would give 7.545 / 4.67622)
    (10, 0.15, 20.6, 17.4536),        # k = 1.5 -> 1 (rounding up: 18.2 / 14.02569)
    (10, 0.25, 18.2, 14.0257),        # k = 2.5 -> 2 (rounding up: 17.3 / 11.01665)
    (10, 0.35, 17.3, 11.0167)])       # k = 3.5 -> 3 (round half to even: 16.5 / 11.06797)
def test_statgraphics_truncation_not_rounding(n, trim, mean, sigma):
    # Statgraphics Centurion, 2026-10-21. Sigma is displayed with 6 significant digits.
    assert winsorized_mean(DATA[n], trim) == pytest.approx(mean, rel=1e-5)
    assert winsorized_sigma(DATA[n], trim) == pytest.approx(sigma, rel=1e-5)
