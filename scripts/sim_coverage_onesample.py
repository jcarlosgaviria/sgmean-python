"""One-sample 95% CI coverage for the trimmed mean: proportional (sgmean D) vs rounded-k Yuen vs mean.
Two targets: NOMINAL = theta(alpha) = integral of Q over [alpha,1-alpha]/(1-2alpha) (what the user asked for)
             SELF    = the parameter each rule actually estimates (theta(g/n); theta(alpha) for D; mean for t).
Usage: python sim_coverage_onesample.py [R] [out.csv]"""
import os, sys, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fractions import Fraction
import numpy as np
from scipy import stats, integrate

R = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
OUT = sys.argv[2] if len(sys.argv) > 2 else "sim_coverage_onesample.csv"
rng = np.random.default_rng(20261025)

def cn(size):
    z = rng.normal(size=size); s = np.where(rng.random(size) < 0.1, 10.0, 1.0); return z * s
DISTS = {  # name: (sampler, ppf or None if symmetric about 0)
    "normal": (lambda s: rng.normal(size=s), None),
    "t3": (lambda s: rng.standard_t(3, size=s), None),
    "contam_normal": (cn, None),
    "lognormal": (lambda s: rng.lognormal(size=s), lambda u: np.exp(stats.norm.ppf(u))),
    "exponential": (lambda s: rng.exponential(size=s), lambda u: -np.log1p(-u)),
    "chisq3": (lambda s: rng.chisquare(3, size=s), lambda u: stats.chi2.ppf(u, 3)),
}
def theta(ppf, a):
    if ppf is None: return 0.0
    if a == 0: return float(integrate.quad(ppf, 1e-12, 1 - 1e-12, limit=200)[0])
    return float(integrate.quad(ppf, a, 1 - a, limit=200)[0] / (1 - 2 * a))
def true_mean(name, ppf):
    return {"lognormal": math.exp(.5), "exponential": 1.0, "chisq3": 3.0}.get(name, 0.0)

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
    return T, se, h, a
def integer_rule(rule):
    def f(x, a):
        n = x.shape[1]; k = kk(n, a)
        g = {"floor": math.floor(k), "ceil": math.ceil(k), "round": math.floor(k + Fraction(1, 2))}[rule]
        h = n - 2 * g
        return x[:, g:n - g].mean(1), np.sqrt(ssw(x, g) / (h * (h - 1))), h, g / n
    return f
def mean_(x, a):
    n = x.shape[1]; return x.mean(1), x.std(1, ddof=1) / math.sqrt(n), n, 0.0
METHODS = {"mean_t": mean_, "yuen_floor": integer_rule("floor"), "yuen_ceil": integer_rule("ceil"),
           "yuen_round": integer_rule("round"), "sgmean_D": prop}

NS = [15, 20, 22, 30, 45, 60]; TRIMS = [0.05, 0.10, 0.20]
head = ["dist", "n", "trim", "k", "method", "eff_trim", "bias_vs_nominal", "cov_nominal", "cov_self", "mean_width"]
rows = []; print(",".join(head))
for name, (f, ppf) in DISTS.items():
    for n in NS:
        for a in TRIMS:
            x = np.sort(f((R, n)), 1)
            tn = theta(ppf, a)
            for mname, m in METHODS.items():
                T, se, h, eff = m(x, a)
                q = stats.t.ppf(0.975, h - 1); lo, hi = T - q * se, T + q * se
                ts = true_mean(name, ppf) if mname == "mean_t" else theta(ppf, eff)
                row = [name, n, a, round(a * n, 2), mname, round(eff, 4), round(float(T.mean() - tn), 4),
                       round(float(((lo <= tn) & (tn <= hi)).mean()), 4),
                       round(float(((lo <= ts) & (ts <= hi)).mean()), 4), round(float((hi - lo).mean()), 4)]
                rows.append(row); print(",".join(map(str, row)), flush=True)
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(head); w.writerows(rows)
