"""Monte Carlo: integer truncation floor(k) vs proportional trimming (sgmean).

Both estimators use the SAME simulated samples (paired design). Target: 0, the centre of
the clean component, so bias measures the influence of contamination.
Run: python monte_carlo_truncation.py   (about a minute)
"""
import math, sys, csv, json
import numpy as np

SEED = 20261021
R = 20000
NS = [10, 15, 20, 25, 30, 50, 100]
ALPHAS = [0.05, 0.10, 0.20]

def dist_sampler(name):
    if name == "normal":
        return lambda rng, size: rng.standard_normal(size)
    if name == "t3":
        return lambda rng, size: rng.standard_t(3, size)
    if name == "contam_sym":          # 90% N(0,1) + 10% N(0,9^2)
        def f(rng, size):
            z = rng.standard_normal(size)
            m = rng.random(size) < 0.10
            return np.where(m, 9 * z, z)
        return f
    if name == "contam_right":        # 90% N(0,1) + 10% N(10,1): outliers on one side
        def f(rng, size):
            z = rng.standard_normal(size)
            m = rng.random(size) < 0.10
            return np.where(m, z + 10, z)
        return f
    raise ValueError(name)

DISTS = ["normal", "t3", "contam_sym", "contam_right"]

def _kfrac(n, a):
    """k = a*n guarding against float noise (e.g. 0.29*100 = 28.999999999999996)."""
    k = round(a * n, 9)
    return k

def weights_proportional(n, a):
    """Proportional trimmed mean as an overlap: the weight of sorted observation i
    (i = 1..n) is the length of [i-1, i] intersected with [k, n-k], divided by n-2k.
    Independent of the branches used in sgmean.core (validated against it in validate())."""
    k = _kfrac(n, a)
    i = np.arange(1, n + 1)
    w = np.clip(np.minimum(i, n - k) - np.maximum(i - 1, k), 0.0, None)
    return w / (n - 2 * k)

def weights_floor(n, a):
    """R's mean(x, trim) / scipy.stats.trim_mean: k = floor(n * trim) observations per tail."""
    kf = int(math.floor(_kfrac(n, a) + 1e-9))
    w = np.zeros(n); w[kf:n - kf] = 1.0
    return w / (n - 2 * kf)

def validate():
    """Weights must reproduce sgmean (proportional) and scipy.stats.trim_mean (floor)."""
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from sgmean import sgmean
    from scipy.stats import trim_mean
    rng = np.random.default_rng(1)
    bad = 0; cnt = 0
    for n in list(range(3, 61)) + [100]:
        for a in [0, 0.01, 0.05, 0.1, 0.13, 0.2, 0.25, 0.3, 0.35, 0.45, 0.4999]:
            x = np.sort(rng.standard_normal(n))
            cnt += 1
            if abs(x @ weights_proportional(n, a) - sgmean(x, trim=a)) > 1e-9: bad += 1
            if abs(x @ weights_floor(n, a) - trim_mean(x, a)) > 1e-9: bad += 1
    assert bad == 0, f"{bad} mismatches"
    print("validation OK:", cnt, "combinations (n x trim) vs sgmean and scipy.stats.trim_mean")

def cell(rng, dist, n, a):
    x = np.sort(dist_sampler(dist)(rng, (R, n)), axis=1)
    ef = x @ weights_floor(n, a)          # estimate, floor(k)
    ep = x @ weights_proportional(n, a)   # estimate, proportional
    sf, sp = ef ** 2, ep ** 2             # squared error w.r.t. target 0
    d = sf - sp
    se = d.std(ddof=1) / math.sqrt(R)
    k = _kfrac(n, a)
    return dict(dist=dist, n=n, alpha=a, k=round(k, 4), kfloor=int(math.floor(k + 1e-9)),
                kind=("integer" if abs(k - round(k)) < 1e-9 else ("A (k<1)" if k < 1 else "B (k>1)")),
                bias_floor=ef.mean(), bias_prop=ep.mean(),
                rmse_floor=math.sqrt(sf.mean()), rmse_prop=math.sqrt(sp.mean()),
                mse_ratio=sf.mean() / sp.mean(), z=(d.mean() / se) if se > 0 else 0.0)

def main():
    validate()
    rng = np.random.default_rng(SEED)
    rows = []
    for dist in DISTS:
        for n in NS:
            for a in ALPHAS:
                rows.append(cell(rng, dist, n, a))
    with open("mc_cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # trim grid at n = 20, one-sided contamination and normal (for the step-vs-smooth figure)
    grid = []
    for dist in ("contam_right", "normal"):
        for a in np.round(np.arange(0.0, 0.3001, 0.005), 4):
            c = cell(rng, dist, 20, float(a)); grid.append(c)
    with open("mc_grid_n20.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0])); w.writeheader(); w.writerows(grid)
    print("done", len(rows), len(grid))

if __name__ == "__main__":
    main()
