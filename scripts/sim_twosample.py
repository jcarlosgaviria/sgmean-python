"""Pilot: type I error of ttest_ind_trimmed (alpha=0.05), H0 true (equal trimmed means).
Uses symmetric / lognormal / exponential with the same distribution in both groups."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from sgmean import ttest_ind_trimmed
R = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
rng = np.random.default_rng(2026)
dists = {"normal": lambda n: rng.normal(size=n),
         "lognormal": lambda n: rng.lognormal(size=n),
         "exponential": lambda n: rng.exponential(size=n)}
print("dist n1 n2 trim  k1   k2   rejection_rate")
for name, f in dists.items():
    for n1, n2, a in [(15, 22, 0.05), (20, 20, 0.1), (15, 22, 0.1), (30, 45, 0.15), (12, 30, 0.1)]:
        rej = 0
        for _ in range(R):
            if ttest_ind_trimmed(f(n1), f(n2), a).pvalue < 0.05:
                rej += 1
        print(f"{name:11s} {n1:3d} {n2:3d} {a:.2f} {a*n1:4.2f} {a*n2:4.2f}  {rej/R:.3f}")
