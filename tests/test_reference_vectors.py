"""Check the shared reference vectors (the same file is used by the R tests)."""
import csv
import os
import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")
from sgmean import (ttest_trimmed, ttest_rel_trimmed, ttest_ind_trimmed,
                    effect_size_trimmed)

PATH = os.path.join(os.path.dirname(__file__), "reference_vectors.csv")
with open(PATH, newline="") as f:
    ROWS = list(csv.DictReader(f))


def vec(s):
    return np.array([float(u) for u in s.split(";")])


@pytest.mark.parametrize("row", ROWS, ids=[r["id"] for r in ROWS])
def test_vector(row):
    x, trim, alt = vec(row["x"]), float(row["trim"]), row["alternative"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if row["kind"] == "one":
            r = ttest_trimmed(x, float(row["mu0"]), trim, alternative=alt)
            est = r.estimate
        elif row["kind"] == "paired":
            r = ttest_rel_trimmed(x, vec(row["y"]), trim, alternative=alt)
            est = r.estimate
        else:
            y = vec(row["y"])
            r = ttest_ind_trimmed(x, y, trim, alternative=alt)
            est = r.difference
            assert effect_size_trimmed(x, y, trim) == pytest.approx(
                float(row["effect"]), rel=1e-10)
    for got, name in ((est, "est"), (r.se, "se"), (r.statistic, "t"),
                      (r.df, "df"), (r.pvalue, "p"),
                      (r.ci[0], "ci_lo"), (r.ci[1], "ci_hi")):
        assert got == pytest.approx(float(row[name]), rel=1e-10, abs=1e-12)
