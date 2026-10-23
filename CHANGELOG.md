# Changelog

## 0.3.0 (2026-10-23)
Also includes everything in 0.2.0 below (the truncation fix, descriptive statistics, MIT licence).

- New `sgmean.inference`: `se_trimmed`, `ci_trimmed`, `ttest_trimmed` (SE, t confidence interval and one-sample test for the proportional trimmed mean).
- SE "D": `(1-delta)*SE(floor k) + delta*SE(floor k + 1)`, with `SE(m)=sqrt(SSW(m)/(h(h-1)))`, `h=n-2k` fractional, t with `h-1` df. Equals the classical Tukey-McLaughlin SE when `k` is an integer and is continuous in `trim`.
- **Practical proposal / heuristic**: supported by pilot Monte Carlo (symmetric, lognormal, exponential), no theoretical proof yet.
- Optional dependency: `pip install sgmean[inference]` (SciPy).
- New `diagnose_truncation(x, trim)`: compares the proportional trimmed mean with the floor(k) rule (R/SciPy), reporting k, delta, effective trim, both means, difference and whether truncation occurred.
- New `plot_trim_profile(x, trim)`: proportional (smooth) vs floor (step) trimmed mean against trim (optional matplotlib: `pip install sgmean[plots]`).
- New `ttest_ind_trimmed(x, y, trim)`: Yuen-type two-sample test with proportional trimming (method-D SEs, Welch-Satterthwaite df, CI for the difference). Matches `scipy.stats.ttest_ind(trim=..., equal_var=False)` when k is an integer in both samples. Simulation (20000 reps, 5 distributions incl. t3 and contaminated normal, 5 sample-size/trim settings, shifts 0/0.5/1 robust-sd): type I error 0.024-0.052 at alpha 0.05 (never liberal; most conservative under contaminated normal at small n); power compared with Welch and Yuen floor(k) in `docs/sim_power_twosample.csv` (script `scripts/sim_power_twosample.py`). Heuristic, not proven.

## 0.2.0 (2026-10-21, never published to PyPI; included in 0.3.0)
### Fixed
- `sgmean()` returned absurd values for odd `n` when `trim * n` was between `(n-1)/2` and `n/2`
  (e.g. `n = 9`, `trim = 0.45` returned 10.56 for data 1..9; `trim = 0.4999` returned 2782.8
  instead of the median 5). The only observation left was counted as the boundary of both
  tails. It now returns that observation (the median), as the documented boundary behaviour
  requires. Also fixes `n = 1`. Typical use (`trim` <= 0.25) was not affected.
- New tests compare `sgmean()` with an independent definition over the whole trim range.

### Changed
- Licence changed from GPL-3.0-or-later (0.1.1 on PyPI) to MIT, matching the R package on CRAN.
  Copies already obtained under 0.1.1 remain under the GPL.

### Added
- `sgmean.quantile`, `sgmean.quartiles`, `sgmean.iqr`: sample quantiles with the
  nine Hyndman & Fan definitions. Default `type=2`, which reproduces the
  quartiles, sextiles and percentiles of Statgraphics Centurion in every case
  tested (R and NumPy default to type 7). Type 2 uses exact rational
  arithmetic, so no floating-point tolerance is involved.
- `sgmean.winsorized_mean` and `sgmean.winsorized_sigma`, reproducing Statgraphics
  Centurion (Summary Statistics): `k = floor(trim*n)` values replaced at each tail;
  sigma = sd(ddof=1) of the winsorized sample * sqrt((n-1)/(n-2k-1)). Checked on
  seven data sets (5 % and 10 %): 14 means and 14 sigmas.
- `sgmean.geometric_mean`, `dam`, `sbi`, `skewness`, `kurtosis`, `cv`, `std_error`, `mode`
  and `summary()`, which returns the 29 results of Statgraphics' Summary Statistics
  (checked against its output on seven data sets: 190 values). It returns a `SummaryResult`
  (a dict that prints as an aligned table) with English labels by default, Spanish with
  `lang="es"`, `sgmean.set_language()` or the `SGMEAN_LANG` environment variable.
- `scripts/monte_carlo_truncation.py` and `docs/fig_step_vs_smooth_n20.png`: Monte Carlo comparison of
  `floor(k)` and proportional trimming (see README).
- Tests against Statgraphics results (seven data sets, n = 8 to 20) and against
  `numpy.percentile` for all nine types.

## 0.1.1
- Licence metadata set to GPL-3.0-or-later (PyPI).

## 0.1.0
- First release: `sgmean`, the proportional trimmed mean (fractional discount of
  the boundary observations; no integer truncation).
