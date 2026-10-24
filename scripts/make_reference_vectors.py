"""Generate tests/reference_vectors.csv: shared reference values used by both the
Python and the R test suites (keeps the two packages in agreement).

Columns: id, kind (one|paired|two), x, y, trim, mu0, alternative, then outputs
est (trimmed mean / difference), se, t, df, p, ci_lo, ci_hi, effect.
Vectors are ';'-separated. Run:  python scripts/make_reference_vectors.py
"""
import csv, os, warnings
import numpy as np
from sgmean import (ttest_trimmed, ttest_rel_trimmed, ttest_ind_trimmed,
                    effect_size_trimmed)

warnings.simplefilter("ignore")
rng = np.random.default_rng(20261024)
A = np.round(rng.normal(10, 3, 23), 2)
B = np.round(rng.lognormal(2, 0.8, 31), 2)
C = np.round(rng.normal(12, 5, 23) + (rng.random(23) < .1) * 40, 2)   # contaminated
D = np.round(A + rng.normal(0.8, 2, 23), 2)
E = np.round(rng.normal(0, 1, 16), 3)

cases = []
for tr in (0.0, 0.1, 0.2, 0.25):
    cases.append(("one", A, None, tr, 9.0, "two-sided"))
    cases.append(("one", B, None, tr, 7.0, "greater"))
    cases.append(("one", C, None, tr, 10.0, "less"))
    cases.append(("paired", A, D, tr, 0.0, "two-sided"))
    cases.append(("paired", D, A, tr, 0.0, "greater"))
    cases.append(("two", A, B, tr, 0.0, "two-sided"))
    cases.append(("two", C, A, tr, 0.0, "less"))
    cases.append(("two", B, E, tr, 0.0, "greater"))

fmt = lambda v: ";".join(repr(float(u)) for u in v)
out = os.path.join(os.path.dirname(__file__), "..", "tests", "reference_vectors.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "kind", "x", "y", "trim", "mu0", "alternative", "est", "se",
                "t", "df", "p", "ci_lo", "ci_hi", "effect"])
    for i, (kind, x, y, tr, mu0, alt) in enumerate(cases, 1):
        if kind == "one":
            r = ttest_trimmed(x, mu0, tr, alternative=alt)
            row = [r.estimate, r.se, r.statistic, r.df, r.pvalue, *r.ci, ""]
        elif kind == "paired":
            r = ttest_rel_trimmed(x, y, tr, alternative=alt)
            row = [r.estimate, r.se, r.statistic, r.df, r.pvalue, *r.ci, ""]
        else:
            r = ttest_ind_trimmed(x, y, tr, alternative=alt)
            row = [r.difference, r.se, r.statistic, r.df, r.pvalue, *r.ci,
                   effect_size_trimmed(x, y, tr)]
        row = ["" if v == "" else repr(float(v)) for v in row]
        w.writerow([i, kind, fmt(x), fmt(y) if y is not None else "", tr, mu0, alt] + row)
print("written", out, len(cases), "cases")
