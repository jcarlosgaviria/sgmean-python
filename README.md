# sgmean for Python

**Proportional trimmed mean compatible with Statgraphics.**

Resolves the integer truncation problem of `scipy.stats.trim_mean`.

## Installation

```bash
pip install sgmean
```

## The problem

`scipy.stats.trim_mean` computes the trimmed mean via integer 
truncation: it calculates `k = proportiontocut * n` and silently
reduces it to the nearest integer below. This generates two types
of silent distortion:

- **Type A** (k < 1): returns the arithmetic mean without any trimming
- **Type B** (k > 1, non-integer): trims fewer observations than requested

## The solution

`sgmean` applies a fractional discount `(1 - delta)` to boundary
observations, where `delta = k - floor(k)`:

```python
from sgmean import sgmean
import numpy as np

x = np.array([850, 920, 980, 1050, 1120, 1180, 1250,
              1320, 1400, 1480, 1550, 1700, 1850, 2100, 8500])

# Type A distortion (k = 0.05 * 15 = 0.75)
from scipy import stats
stats.trim_mean(x, 0.05)  # 1816.667 — no trimming applied
sgmean(x, trim=0.05)      # 1499.074 — correct proportional discount

# Type B distortion (k = 0.10 * 15 = 1.50)
stats.trim_mean(x, 0.10)  # 1376.923 — 33% trimming loss
sgmean(x, trim=0.10)      # 1365.833 — matches Statgraphics exactly
```

## Reference

Gaviria-Chaverra, J.C. (2026). sgmean: A Proportional Trimmed Mean
for R Compatible with Statgraphics. *The R Journal*.
DOI: 10.32614/CRAN.package.sgmean

## License

MIT
