"""Input handling shared by every public function.

Accepts lists, NumPy arrays, pandas Series (including nullable dtypes with
``pd.NA``) and one-column DataFrames.  pandas is *not* required.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from numpy.typing import ArrayLike
import numpy as np

NAN_POLICIES = ("raise", "omit", "propagate")


def to_array(x: ArrayLike, nan_policy: str = "raise", min_n: int = 1,
             name: str = "x") -> np.ndarray:
    """Return ``x`` as a flat float array, applying ``nan_policy``.

    ``"raise"``     -> ValueError if NaN is present (default for most functions);
    ``"omit"``      -> drop NaN values;
    ``"propagate"`` -> keep NaN (only meaningful for ``sgmean``; the caller
                       returns NaN).
    A DataFrame with several columns is rejected: flattening it would silently
    mix variables (use ``summary(df)`` or pass one column).
    """
    if nan_policy not in NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of {NAN_POLICIES}")
    if hasattr(x, "columns"):                      # pandas DataFrame
        if len(x.columns) != 1:
            raise TypeError(
                f"{name} has {len(x.columns)} columns; pass one column "
                f"(e.g. df['col']) or use summary(df) for all of them")
        x = x.iloc[:, 0]
    if hasattr(x, "to_numpy"):                     # pandas Series / Index
        try:
            arr = x.to_numpy(dtype=float, na_value=np.nan)
        except TypeError:
            arr = x.to_numpy(dtype=float)
    else:
        arr = np.asarray(x, dtype=float)
    arr = arr.ravel()
    has_nan = bool(np.isnan(arr).any())
    if has_nan:
        if nan_policy == "raise":
            raise ValueError(f"{name} must be free of NaN "
                             "(use nan_policy='omit' to drop them)")
        if nan_policy == "omit":
            arr = arr[~np.isnan(arr)]
    if arr.size < min_n and not (has_nan and nan_policy == "propagate"):
        raise ValueError(f"at least {min_n} observation(s) required"
                         + (" after removing NaN" if has_nan else ""))
    return arr
