import numpy as np
import pytest
from sgmean import sgmean

x_wages = np.array([850, 920, 980, 1050, 1120, 1180, 1250,
                    1320, 1400, 1480, 1550, 1700, 1850, 2100, 8500])
x_test = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 120])

def test_type_a():
    """Type A: k < 1, sgmean != scipy.trim_mean"""
    result = sgmean(x_wages, trim=0.05)
    assert abs(result - 1499.074) < 0.001

def test_type_b_wages():
    """Type B: k=1.50, matches Statgraphics"""
    result = sgmean(x_wages, trim=0.10)
    assert abs(result - 1365.833) < 0.001

def test_type_b_k150():
    """Type B: k=1.50 with symmetric data"""
    result = sgmean(x_test, trim=0.15)
    assert abs(result - 11.0) < 0.001

def test_type_b_k250():
    """Type B: k=2.50 with symmetric data"""
    result = sgmean(x_test, trim=0.25)
    assert abs(result - 11.0) < 0.001

def test_control_integer_k():
    """Control: k exact integer, identical to scipy"""
    from scipy import stats
    result = sgmean(x_test, trim=0.10)
    expected = stats.trim_mean(x_test, 0.10)
    assert abs(result - expected) < 1e-10

def test_no_trim():
    """trim=0 returns arithmetic mean"""
    result = sgmean(x_test, trim=0.0)
    assert abs(result - np.mean(x_test)) < 1e-10

def test_invalid_trim():
    """trim >= 0.5 raises ValueError"""
    with pytest.raises(ValueError):
        sgmean(x_test, trim=0.5)
