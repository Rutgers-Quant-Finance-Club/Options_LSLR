"""Cross-sectional least-squares continuation estimation for one LSM time slice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import laguerre_basis
from .utils import discount, put_intrinsic_value


@dataclass(frozen=True)
class RegressionResult:
    """Output bundle for one continuation-value regression step."""

    coefficients: np.ndarray
    continuation_values: np.ndarray
    intrinsic_values: np.ndarray
    itm_mask: np.ndarray
    discounted_targets_itm: np.ndarray
    n_itm: int
    rank: int
    condition_number: float | None
    r_squared: float | None
    status: str


def _fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """Fit least-squares coefficients with NumPy's SVD-based solver."""
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta, int(rank)


def estimate_continuation_value(
    stock_prices_t: np.ndarray,
    realized_future_cashflows: np.ndarray,
    strike: float,
    rate: float,
    dt: float,
    discount_steps: int = 1,
    normalize_by_strike: bool = True,
) -> RegressionResult:
    """Estimate continuation values at one exercise date using LSM regression.

    This function implements the Week 4 regression logic only (single time slice):
      1. Compute intrinsic values and identify ITM paths.
      2. Build Laguerre features on current state (ITM paths only).
      3. Build regression target Y as discounted realized future cash flows.
      4. Fit cross-sectional OLS and return fitted continuation values.

    The caller is responsible for generating `realized_future_cashflows` using a
    valid stopping policy from later dates (Week 5 backward induction workflow).
    """
    s_t = np.asarray(stock_prices_t, dtype=np.float64)
    cf_future = np.asarray(realized_future_cashflows, dtype=np.float64)

    if s_t.ndim != 1 or cf_future.ndim != 1:
        raise ValueError("stock_prices_t and realized_future_cashflows must be 1D arrays")
    if s_t.shape[0] != cf_future.shape[0]:
        raise ValueError("stock_prices_t and realized_future_cashflows must have same length")
    if np.any(s_t <= 0.0):
        raise ValueError("stock_prices_t must contain strictly positive values")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if discount_steps <= 0:
        raise ValueError("discount_steps must be positive")

    intrinsic = put_intrinsic_value(s_t, strike)
    itm_mask = intrinsic > 0.0
    n_paths = s_t.shape[0]
    continuation = np.zeros(n_paths, dtype=np.float64)

    if not np.any(itm_mask):
        return RegressionResult(
            coefficients=np.zeros(4, dtype=np.float64),
            continuation_values=continuation,
            intrinsic_values=intrinsic,
            itm_mask=itm_mask,
            discounted_targets_itm=np.array([], dtype=np.float64),
            n_itm=0,
            rank=0,
            condition_number=None,
            r_squared=None,
            status="no_itm_paths",
        )

    y_discounted_full = discount(cf_future, rate=rate, dt=dt, steps=discount_steps)
    y_itm = y_discounted_full[itm_mask]
    x_itm = laguerre_basis(
        s_t[itm_mask],
        strike=strike,
        normalize_by_strike=normalize_by_strike,
    )

    beta, rank = _fit_ols(x_itm, y_itm)
    y_hat_itm = x_itm @ beta
    continuation[itm_mask] = y_hat_itm

    cond_number = float(np.linalg.cond(x_itm)) if x_itm.size else None

    y_var = float(np.var(y_itm)) if y_itm.size else 0.0
    if y_itm.size <= 1 or y_var == 0.0:
        r_sq = None
    else:
        residual = y_itm - y_hat_itm
        r_sq = 1.0 - float(np.var(residual) / y_var)

    status = "ok" if rank == x_itm.shape[1] else "rank_deficient"

    return RegressionResult(
        coefficients=beta,
        continuation_values=continuation,
        intrinsic_values=intrinsic,
        itm_mask=itm_mask,
        discounted_targets_itm=y_itm,
        n_itm=int(itm_mask.sum()),
        rank=rank,
        condition_number=cond_number,
        r_squared=r_sq,
        status=status,
    )
