"""Does proportional trimming (method D) beat simply rounding k?
Same samples, five two-sample tests (alpha = 0.05, two-sided):
  welch | yuen_floor | yuen_ceil | yuen_round (k=floor(k+.5)) | sgmean_D
Integer-k Yuen uses g trimmed values per tail in each sample, h=n-2g.
Shift alternative: y + d*scale (scale = IQR/1.349).
Usage: python sim_rounding_twosample.py [R] [out.csv]"""
import os, sys, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fractions import Fraction
import numpy as np
from scipy import stats

R = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
OUT = sys.argv[2] if len(sys.argv) > 2 else "sim_rounding_twosample.csv"
rng = np.random.default_rng(20261024)

def cn(size):
    z = rng.normal(size=size); s = np.where(rng.random(size) < 0.1, 10.0, 1.0); return z * s
DISTS = {"normal": lambda s: rng.normal(size=s), "t3": lambda s: rng.standard_t(3, size=s),
         "lognormal": lambda s: rng.lognormal(size=s), "exponential": lambda s: rng.exponential(size=s),
         "contam_normal": cn}
def scale_of(f):
    z = f((2_000_000,)); q1, q3 = np.percentile(z, [25, 75]); return (q3 - q1) / 1.349
SCALE = {k: scale_of(f) for k, f in DISTS.items()}

def kk(n, a): return Fraction(a).limit_denominator(10**9) * n
def weights(n, a):
    k = float(kk(n, a)); i = np.arange(1, n + 1)
    return np.clip(np.minimum(i, n - k) - np.maximum(i - 1, k), 0, None) / (n - 2 * k)
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
def integer_rule(rule):
    def f(x, a):
        n = x.shape[1]; k = kk(n, a)
        g = {"floor": math.floor(k), "ceil": math.ceil(k), "round": math.floor(k + Fraction(1, 2))}[rule]
        h = n - 2 * g
        return x[:, g:n - g].mean(1), np.sqrt(ssw(x, g) / (h * (h - 1))), h
    return f
def mean_(x, a):
    n = x.shape[1]; return x.mean(1), x.std(1, ddof=1) / math.sqrt(n), n
METHODS = {"welch": mean_, "yuen_floor": integer_rule("floor"), "yuen_ceil": integer_rule("ceil"),
           "yuen_round": integer_rule("round"), "sgmean_D": prop}
def pval(m, x, y, a):
    T1, s1, h1 = m(x, a); T2, s2, h2 = m(y, a)
    v = s1**2 + s2**2; t = (T1 - T2) / np.sqrt(v)
    return 2 * stats.t.sf(np.abs(t), v**2 / (s1**4 / (h1 - 1) + s2**4 / (h2 - 1)))

SETTINGS = [(15, 22, .05), (20, 20, .10), (15, 22, .10), (30, 45, .15), (12, 30, .10), (25, 25, .05), (10, 25, .15)]
def effg(n, a):
    k = kk(n, a); return (math.floor(k), math.ceil(k), math.floor(k + Fraction(1, 2)))
rows = []; head = ["dist","n1","n2","trim","k1","k2","d"] + list(METHODS)
print(",".join(head))
for name, f in DISTS.items():
    sc = SCALE[name]
    for n1, n2, a in SETTINGS:
        for d in (0.0, 0.5, 1.0):
            x = np.sort(f((R, n1)), 1); y = np.sort(f((R, n2)) + d * sc, 1)
            r = [round(float((pval(m, x, y, a) < 0.05).mean()), 4) for m in METHODS.values()]
            row = [name, n1, n2, a, round(a * n1, 2), round(a * n2, 2), d] + r
            rows.append(row); print(",".join(map(str, row)), flush=True)
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(head); w.writerows(rows)
