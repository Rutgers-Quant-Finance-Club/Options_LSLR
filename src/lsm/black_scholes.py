"""European put pricing utilities for Week 3 baselining."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from .utils import put_intrinsic_value


def black_scholes_put_price(
    spot: float | np.ndarray,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
) -> float | np.ndarray:
    """Price a European put with the Black-Scholes closed form.

    Args:
        spot: Current underlying spot price(s).
        strike: Strike price.
        rate: Continuously compounded risk-free rate.
        sigma: Volatility.
        maturity: Time-to-maturity in years.
    """
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")

    s = np.asarray(spot, dtype=np.float64)
    if np.any(s <= 0.0):
        raise ValueError("spot must be positive")

    if maturity <= 0.0:
        payoff = put_intrinsic_value(s, strike)
        return float(payoff) if np.isscalar(spot) else payoff

    if sigma == 0.0:
        forward_terminal = s * np.exp(rate * maturity)
        deterministic_payoff = put_intrinsic_value(forward_terminal, strike)
        value = np.exp(-rate * maturity) * deterministic_payoff
        return float(value) if np.isscalar(spot) else value

    vol_sqrt_t = sigma * np.sqrt(maturity)
    d1 = (np.log(s / strike) + (rate + 0.5 * sigma * sigma) * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    discounted_strike = strike * np.exp(-rate * maturity)
    put = discounted_strike * norm.cdf(-d2) - s * norm.cdf(-d1)
    return float(put) if np.isscalar(spot) else put


def european_put_mc_price_from_paths(
    paths: np.ndarray,
    strike: float,
    rate: float,
    maturity: float,
) -> float:
    """Estimate European put value by discounting terminal path payoffs."""
    if paths.ndim != 2:
        raise ValueError("paths must be a 2D array")
    terminal = paths[:, -1]
    payoff = put_intrinsic_value(terminal, strike)
    return float(np.exp(-rate * maturity) * payoff.mean())


def compare_european_put_prices(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    paths: np.ndarray,
) -> dict[str, float]:
    """Return analytic/MC comparison diagnostics for Week 3 checks."""
    analytic = float(black_scholes_put_price(spot, strike, rate, sigma, maturity))
    mc = european_put_mc_price_from_paths(paths, strike, rate, maturity)
    abs_error = abs(mc - analytic)
    rel_error = abs_error / analytic if analytic != 0.0 else math.nan
    return {
        "analytic": analytic,
        "mc": mc,
        "abs_error": abs_error,
        "rel_error": rel_error,
    }
