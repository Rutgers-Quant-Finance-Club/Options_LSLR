"""Risk-neutral geometric Brownian motion simulation utilities."""

from __future__ import annotations

import numpy as np


def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    maturity: float,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate GBM paths under the risk-neutral measure using exact discretization.

    Dynamics:
        dS_t = r S_t dt + sigma S_t dW_t

    Exact Euler step under lognormal solution:
        S_{t+dt} = S_t * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)

    Returns:
        Array with shape (n_paths, n_steps + 1), including S0 in column 0.
    """
    if s0 <= 0.0:
        raise ValueError("s0 must be positive")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    dt = maturity / n_steps
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_paths, n_steps))

    growth = np.exp((r - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z)

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.cumprod(growth, axis=1)
    return paths
