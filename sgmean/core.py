import numpy as np

def sgmean(x, trim=0.05):
    """
    Proportional trimmed mean compatible with Statgraphics.
    
    Resolves the integer truncation problem of scipy.stats.trim_mean.
    When k = trim * n is non-integer, applies a fractional discount
    (1 - delta) to boundary observations, where delta = k - floor(k).
    
    Parameters
    ----------
    x    : array-like
        Numeric data vector.
    trim : float, optional
        Fraction of observations to trim from each tail. Must be
        in [0, 0.5). Default is 0.05.
    
    Returns
    -------
    float
        Proportional trimmed mean.
    
    Examples
    --------
    >>> import numpy as np
    >>> from sgmean import sgmean
    >>> x = np.array([850, 920, 980, 1050, 1120, 1180, 1250,
    ...               1320, 1400, 1480, 1550, 1700, 1850, 2100, 8500])
    >>> sgmean(x, trim=0.05)   # Type A: k=0.75
    1499.0740740740741
    >>> sgmean(x, trim=0.10)   # Type B: k=1.50
    1365.8333333333333
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if not (0 <= trim < 0.5):
        raise ValueError("trim must be in [0, 0.5)")
    k = trim * n
    kf = int(np.floor(k))
    delta = k - kf
    if k == 0:
        return float(np.mean(x))
    if kf == 0:
        suma = (1-k)*x[0] + np.sum(x[1:-1]) + (1-k)*x[-1]
    elif delta == 0:
        suma = np.sum(x[kf:n-kf])
    else:
        suma = ((1-delta)*x[kf] +
                np.sum(x[kf+1:n-kf-1]) +
                (1-delta)*x[n-kf-1])
    return float(suma / (n - 2*k))
