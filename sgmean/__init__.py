from .core import sgmean
from .quantiles import quantile, quartiles, iqr
from .winsor import winsorized_mean, winsorized_sigma
from .inference import (se_trimmed, ci_trimmed, ttest_trimmed, ttest_ind_trimmed,
                        ttest_rel_trimmed, effect_size_trimmed)
from .diagnostic import diagnose_truncation, plot_trim_profile
from .report import set_language, SummaryResult, MultiSummaryResult
from .descriptive import (geometric_mean, dam, sbi, skewness, kurtosis,
                          cv, std_error, mode, summary)
__version__ = "0.4.0"
__author__ = "Juan C. Gaviria-Chaverra"
__all__ = ["sgmean", "quantile", "quartiles", "iqr",
           "winsorized_mean", "winsorized_sigma",
           "geometric_mean", "dam", "sbi", "skewness", "kurtosis",
           "se_trimmed", "ci_trimmed", "ttest_trimmed", "ttest_ind_trimmed",
           "ttest_rel_trimmed", "effect_size_trimmed",
           "diagnose_truncation", "plot_trim_profile",
           "cv", "std_error", "mode", "summary", "set_language", "SummaryResult", "MultiSummaryResult"]
