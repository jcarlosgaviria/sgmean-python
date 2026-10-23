import numpy as np
import pytest
from sgmean import sgmean


def reference(x, trim):
    """Independent definition: mean of the empirical quantile function between trim and 1-trim.

    Observation i (1-based, sorted) occupies [i-1, i] on a scale of n; the kept region is
    [k, n-k] with k = trim*n. Weight = length of the overlap; divide by n - 2k.
    """
    x = np.sort(np.asarray(x, float)); n = x.size; k = trim * n
    i = np.arange(1, n + 1)
    w = np.clip(np.minimum(i, n - k) - np.maximum(i - 1, k), 0, 1)
    return float(np.sum(w * x) / (n - 2 * k))


@pytest.mark.parametrize("n", list(range(1, 41)) + [50, 99, 100])
def test_matches_reference_on_the_whole_trim_range(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n) * 3 + 1
    for trim in list(np.arange(0, 0.5, 0.0125)) + [0.4999, 0.499, 0.4975]:
        assert sgmean(x, trim=float(trim)) == pytest.approx(reference(x, float(trim)), rel=1e-9, abs=1e-9)


def test_odd_n_near_half_returns_median_not_nonsense():
    # Regression: n = 9, trim = 0.45 used to return 10.5556 (n = 9, trim = 0.4999 gave 2782.8)
    x = np.arange(1, 10)
    for trim in (0.40, 0.43, 0.45, 0.49, 0.4999):
        assert sgmean(x, trim=trim) == pytest.approx(5.0)
    assert sgmean(np.arange(1, 12), trim=0.47) == pytest.approx(6.0)


def test_result_always_inside_the_data_range():
    rng = np.random.default_rng(0)
    for n in range(1, 31):
        x = rng.standard_normal(n)
        for trim in np.arange(0, 0.5, 0.01):
            v = sgmean(x, trim=float(trim))
            assert x.min() - 1e-12 <= v <= x.max() + 1e-12


def test_single_observation():
    assert sgmean([3.5], trim=0.1) == 3.5
