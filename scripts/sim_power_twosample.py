"""Size and power of two-sample tests, vectorized.
Methods on the SAME samples: Welch (means), Yuen floor(k) (SciPy rule),
proportional D (sgmean.ttest_ind_trimmed).  Shift alternative: y + d*scale,
scale = IQR/1.349 of the base distribution.  Usage: python sim_power_twosample.py [R] [out.csv]"""
import os, sys, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fractions import Fraction
import numpy as np
from scipy import stats
from sgmean import ttest_ind_trimmed

R = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
OUT = sys.argv[2] if len(sys.argv) > 2 else "sim_power_twosample.csv"
rng = np.random.default_rng(20261023)

def cn(size):  # contaminated normal 90% N(0,1) + 10% N(0,10^2)
    z = rng.normal(size=size); s = np.where(rng.random(size) < 0.1, 10.0, 1.0)
    return z * s
DISTS = {"normal": lambda s: rng.normal(size=s),
         "t3": lambda s: rng.standard_t(3, size=s),
         "lognormal": lambda s: rng.lognormal(size=s),
         "exponential": lambda s: rng.exponential(size=s),
         "contam_normal": cn}
def scale_of(f):
    z = f((2_000_000,)); q1, q3 = np.percentile(z, [25, 75]); return (q3 - q1) / 1.349
SCALE = {k: scale_of(f) for k, f in DISTS.items()}

def kk(n, a): return Fraction(a).limit_denominator(10**9) * n
def weights(n, a):
    k = float(kk(n, a)); i = np.arange(1, n + 1)
    w = np.clip(np.minimum(i, n - k) - np.maximum(i - 1, k), 0, None)
    return w / (n - 2 * k)
def ssw(x, g):
    n = x.shape[1]; y = x.copy()
    if g: y[:, :g] = x[:, [g]]; y[:, n - g:] = x[:, [n - g - 1]]
    return ((y - y.mean(1, keepdims=True)) ** 2).sum(1)

def prop(x, a):               # x sorted; returns T, se, h
    n = x.shape[1]; k = kk(n, a); g = math.floor(k); d = float(k - g); h = float(n - 2 * k)
    T = x @ weights(n, a); den = h * (h - 1)
    se = (1 - d) * np.sqrt(ssw(x, g) / den)
    if d > 0: se = se + d * np.sqrt(ssw(x, g + 1) / den)
    return T, se, h
def flo(x, a):
    n = x.shape[1]; g = math.floor(kk(n, a)); h = n - 2 * g
    T = x[:, g:n - g].mean(1); se = np.sqrt(ssw(x, g) / (h * (h - 1)))
    return T, se, h
def mean_(x, a):
    n = x.shape[1]; return x.mean(1), x.std(1, ddof=1) / math.sqrt(n), n
def pval(m, x, y, a):
    T1, s1, h1 = m(x, a); T2, s2, h2 = m(y, a)
    v = s1**2 + s2**2; t = (T1 - T2) / np.sqrt(v)
    df = v**2 / (s1**4 / (h1 - 1) + s2**4 / (h2 - 1))
    return 2 * stats.t.sf(np.abs(t), df)

# check vectorized D against the package function
_r = np.random.default_rng(1)
for n1, n2, a in [(15, 22, .05), (12, 30, .1)]:
    x, y = _r.lognormal(size=(3, n1)), _r.lognormal(size=(3, n2))
    pv = pval(prop, np.sort(x, 1), np.sort(y, 1), a)
    for j in range(3):
        assert abs(pv[j] - ttest_ind_trimmed(x[j], y[j], a).pvalue) < 1e-10
print("vectorized D == package function: OK", flush=True)

SETTINGS = [(15, 22, .05), (20, 20, .10), (15, 22, .10), (30, 45, .15), (12, 30, .10)]
rows = []
print("dist,n1,n2,trim,k1,k2,d,welch,yuen_floor,sgmean_D")
for name, f in DISTS.items():
    sc = SCALE[name]
    for n1, n2, a in SETTINGS:
        for d in (0.0, 0.5, 1.0):
            x = np.sort(f((R, n1)), 1); y = np.sort(f((R, n2)) + d * sc, 1)
            r = [float((pval(m, x, y, a) < 0.05).mean()) for m in (mean_, flo, prop)]
            row = [name, n1, n2, a, round(a * n1, 2), round(a * n2, 2), d] + [round(v, 4) for v in r]
            rows.append(row); print(",".join(map(str, row)), flush=True)
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["dist","n1","n2","trim","k1","k2","d","welch","yuen_floor","sgmean_D"]); w.writerows(rows)
