"""Shared helpers for option payoff and discounting logic."""

from __future__ import annotations

import numpy as np


def put_intrinsic_value(spot: np.ndarray | float, strike: float) -> np.ndarray | float:
    """Return intrinsic value max(K - S, 0)."""
    value = np.maximum(strike - np.asarray(spot), 0.0)
    return float(value) if np.isscalar(spot) else value


def discount(values: np.ndarray | float, rate: float, dt: float, steps: int = 1) -> np.ndarray | float:
    """Apply continuous discounting over `steps * dt` years."""
    factor = np.exp(-rate * dt * steps)
    out = np.asarray(values) * factor
    return float(out) if np.isscalar(values) else out
