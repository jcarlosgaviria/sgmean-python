"""Paired test (size, power) and robust effect size (bias, spread), vectorized.

Part A. Paired test = one-sample test of the differences. Methods on the SAME
samples of differences: mean-t, trimmed-t with floor(k) (SciPy/R rule) and
proportional D (sgmean.ttest_rel_trimmed). Each tests its OWN target (mean or
trimmed mean of the difference distribution, computed from a 2e6 sample) so size
is the rejection rate at alpha = .05 (two-sided); power: differences shifted by
d * scale (scale = IQR/1.349).
Part B. Effect size d = c(trim) (Tx-Ty)/Sw (sgmean.effect_size_trimmed): mean,
sd and rmse of the estimate around its population value (same formula applied to
a 2e6 sample) vs Cohen's d.
Usage: python sim_paired_effect.py [R] [prefix]"""
import os, sys, csv, math, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fractions import Fraction
import numpy as np
from scipy import stats
from sgmean import ttest_rel_trimmed, effect_size_trimmed
from sgmean.inference import _akp_constant

R = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
PRE = sys.argv[2] if len(sys.argv) > 2 else "sim_paired_effect"
rng = np.random.default_rng(20261024)

def cn(size):
    z = rng.normal(size=size); s = np.where(rng.random(size) < 0.1, 10.0, 1.0)
    return z * s
DISTS = {"normal": lambda s: rng.normal(size=s),
         "t3": lambda s: rng.standard_t(3, size=s),
         "lognormal": lambda s: rng.lognormal(size=s),
         "exponential": lambda s: rng.exponential(size=s),
         "contam_normal": cn}

def kk(n, a): return Fraction(a).limit_denominator(10**9) * n
def weights(n, a):
    k = float(kk(n, a)); i = np.arange(1, n + 1)
    w = np.clip(np.minimum(i, n - k) - np.maximum(i - 1, k), 0, None)
    return w / (n - 2 * k)
def ssw(x, g):
    n = x.shape[1]; y = x.copy()
    if g: y[:, :g] = x[:, [g]]; y[:, n - g:] = x[:, [n - g - 1]]
    return ((y - y.mean(1, keepdims=True)) ** 2).sum(1)
def prop(x, a):
    n = x.shape[1]; k = kk(n, a); g = math.floor(k); d = float(k - g); h = float(n - 2 * k)
    T = x @ weights(n, a); den = h * (h - 1)
    se = (1 - d) * np.sqrt(ssw(x, g) / den)
    if d > 0: se = se + d * np.sqrt(ssw(x, g + 1) / den)
    return T, se, h
def flo(x, a):
    n = x.shape[1]; g = math.floor(kk(n, a)); h = n - 2 * g
    return x[:, g:n - g].mean(1), np.sqrt(ssw(x, g) / (h * (h - 1))), h
def mean_(x, a):
    n = x.shape[1]; return x.mean(1), x.std(1, ddof=1) / math.sqrt(n), n
METHODS = {"mean_t": mean_, "floor": flo, "D": prop}

def pop_trimmed(z, a):          # population nominal-trim mean from a huge sample
    z = np.sort(z); m = z.size; k = a * m
    return z[int(k):m - int(k)].mean()

# sanity: vectorized D == package function
_r = np.random.default_rng(1)
for n, a in [(15, .05), (12, .1), (23, .2)]:
    d = _r.lognormal(size=(2, n)); T, se, h = prop(np.sort(d, 1), a)
    for i in range(2):
        r = ttest_rel_trimmed(d[i], np.zeros(n), a)
        assert abs(r.se - se[i]) < 1e-10 and abs(r.estimate - T[i]) < 1e-10
print("vectorized == package: OK", flush=True)

NS = [10, 15, 20, 30, 45, 60]
TRIMS = [0.1, 0.2]
big = {k: f((2_000_000,)) for k, f in DISTS.items()}
scale = {k: (np.percentile(z, 75) - np.percentile(z, 25)) / 1.349 for k, z in big.items()}
rows = []
for name, f in DISTS.items():
    for a in TRIMS:
        target = {"mean_t": big[name].mean(), "floor": pop_trimmed(big[name], a),
                  "D": pop_trimmed(big[name], a)}
        for n in NS:
            d0 = np.sort(f((R, n)), axis=1)
            for dsh in (0.0, 0.5):
                # shifted sample keeps order: add constant
                x = d0 + dsh * scale[name]
                res = {}
                for mn, m in METHODS.items():
                    T, se, h = m(x, a)
                    t = (T - (target[mn] if dsh == 0 else target[mn])) / se
                    # H0 target is the unshifted population value; shift is the alternative
                    p = 2 * stats.t.sf(np.abs(t), h - 1)
                    res[mn] = float((p < 0.05).mean())
                rows.append(dict(part="paired", dist=name, trim=a, n=n,
                                 k=float(kk(n, a)), shift=dsh, **res))
    print("paired done", name, flush=True)

# Part B: effect size
eff = []
for name, f in DISTS.items():
    for a in TRIMS:
        for delta in (0.0, 0.5):
            zx = np.sort(f((2_000_000,)))
            zy = np.sort(f((2_000_000,))) + delta * scale[name]
            def tm(z): return pop_trimmed(z, a)
            def wv(z):
                m = z.size; g = int(a * m); y = z.copy(); y[:g] = z[g]; y[m - g:] = z[m - g - 1]
                return y.var(ddof=1)
            sw = math.sqrt((wv(zx) + wv(zy)) / 2)
            pop_d = _akp_constant(a) * (tm(zx) - tm(zy)) / sw
            pop_cohen = (zx.mean() - zy.mean()) / math.sqrt((zx.var(ddof=1) + zy.var(ddof=1)) / 2) if np.isfinite(zx.var()) else np.nan
            for n in (10, 20, 40, 80):
                x = f((1500, n)); y = f((1500, n)) + delta * scale[name]
                est = np.array([effect_size_trimmed(x[i], y[i], a) for i in range(1500)])
                eff.append(dict(dist=name, trim=a, n=n, shift=delta, pop_d=pop_d,
                                mean_est=est.mean(), sd_est=est.std(ddof=1),
                                rmse=math.sqrt(((est - pop_d) ** 2).mean()),
                                pop_cohen=pop_cohen))
    print("effect done", name, flush=True)

for fn, data in ((PRE + "_paired.csv", rows), (PRE + "_effect.csv", eff)):
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
print("written")
