"""Laguerre basis construction for LSM continuation-value regression."""

from __future__ import annotations

import numpy as np


def laguerre_basis(
    stock_prices: np.ndarray,
    strike: float | None = None,
    normalize_by_strike: bool = True,
) -> np.ndarray:
    """Build constant + first three weighted Laguerre terms for one state variable.

    Basis columns are:
        1,
        exp(-x / 2),
        exp(-x / 2) * (1 - x),
        exp(-x / 2) * (1 - 2x + 0.5x^2)

    where x = S / K if normalize_by_strike is True, otherwise x = S.
    """
    s = np.asarray(stock_prices, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError("stock_prices must be a 1D array")
    if np.any(s <= 0.0):
        raise ValueError("stock_prices must contain strictly positive values")

    if normalize_by_strike:
        if strike is None:
            raise ValueError("strike is required when normalize_by_strike=True")
        if strike <= 0.0:
            raise ValueError("strike must be positive")
        x = s / strike
    else:
        x = s

    e = np.exp(-0.5 * x)
    l0 = e
    l1 = e * (1.0 - x)
    l2 = e * (1.0 - 2.0 * x + 0.5 * x * x)
    return np.column_stack((np.ones_like(x), l0, l1, l2))
