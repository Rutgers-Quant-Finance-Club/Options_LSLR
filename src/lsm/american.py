"""Week 5: Longstaff-Schwartz backward induction for American put options."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import laguerre_basis
from .utils import put_intrinsic_value


@dataclass(frozen=True)
class LSMSliceDiagnostics:
    """Regression diagnostics for one exercise date in backward induction."""

    time_index: int
    n_itm: int
    rank: int
    condition_number: float | None
    status: str


@dataclass(frozen=True)
class LSMPolicy:
    """Fitted continuation models across exercise times."""

    strike: float
    rate: float
    maturity: float
    n_steps: int
    normalize_by_strike: bool
    min_itm_paths: int
    coefficients_by_time: dict[int, np.ndarray]
    diagnostics_by_time: dict[int, LSMSliceDiagnostics]


@dataclass(frozen=True)
class LSMValuationResult:
    """Pathwise valuation output under an LSM stopping rule."""

    price: float
    std_error: float
    discounted_cashflows: np.ndarray
    cashflows_at_exercise: np.ndarray
    exercise_time_index: np.ndarray
    exercise_counts_by_time: dict[int, int]
    policy: LSMPolicy


def _validate_paths(paths: np.ndarray) -> tuple[int, int]:
    if paths.ndim != 2:
        raise ValueError("paths must be a 2D array")
    if np.any(paths <= 0.0):
        raise ValueError("paths must be strictly positive")

    n_paths, n_cols = paths.shape
    if n_paths == 0:
        raise ValueError("paths must contain at least one path")
    if n_cols < 2:
        raise ValueError("paths must have at least two columns (t0 and maturity)")
    return n_paths, n_cols - 1


def _fit_slice_regression(
    stock_itm: np.ndarray,
    target_itm: np.ndarray,
    strike: float,
    normalize_by_strike: bool,
    min_itm_paths: int,
) -> tuple[np.ndarray, int, float | None, str]:
    """Fit one cross-sectional continuation regression with robust fallbacks."""
    n_features = 4
    n_itm = stock_itm.shape[0]

    if n_itm == 0:
        return np.zeros(n_features, dtype=np.float64), 0, None, "no_itm_paths"

    if n_itm < min_itm_paths or n_itm < n_features:
        # Low-sample fallback: constant continuation estimate E[Y | ITM].
        beta = np.zeros(n_features, dtype=np.float64)
        beta[0] = float(target_itm.mean())
        return beta, 1, None, "insufficient_itm_constant_fallback"

    x = laguerre_basis(stock_itm, strike=strike, normalize_by_strike=normalize_by_strike)
    beta, _, rank, _ = np.linalg.lstsq(x, target_itm, rcond=None)
    cond = float(np.linalg.cond(x))
    status = "ok" if rank == x.shape[1] else "rank_deficient"
    return beta, int(rank), cond, status


def _evaluate_policy_forward(
    paths: np.ndarray,
    strike: float,
    rate: float,
    maturity: float,
    coefficients_by_time: dict[int, np.ndarray],
    normalize_by_strike: bool,
    allow_immediate_exercise: bool,
    min_itm_paths: int,
    diagnostics_by_time: dict[int, LSMSliceDiagnostics],
) -> LSMValuationResult:
    """Apply a fixed LSM policy pathwise (typically for out-of-sample valuation)."""
    n_paths, n_steps = _validate_paths(paths)
    dt = maturity / n_steps

    intrinsic = put_intrinsic_value(paths, strike)
    exercise_time = np.full(n_paths, n_steps, dtype=np.int32)
    cashflow = np.zeros(n_paths, dtype=np.float64)
    alive = np.ones(n_paths, dtype=bool)

    for t in range(1, n_steps):
        beta = coefficients_by_time.get(t)
        if beta is None:
            continue

        itm_alive = alive & (intrinsic[:, t] > 0.0)
        if not np.any(itm_alive):
            continue

        x = laguerre_basis(
            paths[itm_alive, t],
            strike=strike,
            normalize_by_strike=normalize_by_strike,
        )
        continuation = x @ beta
        immediate = intrinsic[itm_alive, t]
        exercise_now_local = immediate >= continuation

        itm_alive_idx = np.flatnonzero(itm_alive)
        chosen_idx = itm_alive_idx[exercise_now_local]

        if chosen_idx.size > 0:
            exercise_time[chosen_idx] = t
            cashflow[chosen_idx] = intrinsic[chosen_idx, t]
            alive[chosen_idx] = False

    # Maturity decision for surviving paths.
    if np.any(alive):
        cashflow[alive] = intrinsic[alive, n_steps]

    discounted = cashflow * np.exp(-rate * dt * exercise_time)

    option_price = float(discounted.mean())
    if allow_immediate_exercise:
        immediate0 = float(max(strike - paths[0, 0], 0.0))
        if immediate0 >= option_price:
            discounted = np.full(n_paths, immediate0, dtype=np.float64)
            cashflow = np.full(n_paths, immediate0, dtype=np.float64)
            exercise_time = np.zeros(n_paths, dtype=np.int32)
            option_price = immediate0

    std_error = float(discounted.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else 0.0

    unique_t, counts = np.unique(exercise_time, return_counts=True)
    counts_by_time = {int(t): int(c) for t, c in zip(unique_t, counts, strict=True)}

    policy = LSMPolicy(
        strike=strike,
        rate=rate,
        maturity=maturity,
        n_steps=n_steps,
        normalize_by_strike=normalize_by_strike,
        min_itm_paths=min_itm_paths,
        coefficients_by_time=coefficients_by_time,
        diagnostics_by_time=diagnostics_by_time,
    )

    return LSMValuationResult(
        price=option_price,
        std_error=std_error,
        discounted_cashflows=discounted,
        cashflows_at_exercise=cashflow,
        exercise_time_index=exercise_time,
        exercise_counts_by_time=counts_by_time,
        policy=policy,
    )


def price_american_put_lsm(
    paths: np.ndarray,
    strike: float,
    rate: float,
    maturity: float,
    normalize_by_strike: bool = True,
    min_itm_paths: int = 8,
    allow_immediate_exercise: bool = False,
) -> LSMValuationResult:
    """Fit and value an American put using in-sample LSM backward induction.

    The regression target at time t is the discounted realized future cash flow
    implied by the currently-estimated stopping policy at later times.
    """
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if min_itm_paths <= 0:
        raise ValueError("min_itm_paths must be positive")

    n_paths, n_steps = _validate_paths(paths)
    dt = maturity / n_steps

    intrinsic = put_intrinsic_value(paths, strike)

    # Start from maturity where stopping decision is known.
    exercise_time = np.full(n_paths, n_steps, dtype=np.int32)
    cashflow = intrinsic[:, n_steps].astype(np.float64)

    coefficients_by_time: dict[int, np.ndarray] = {}
    diagnostics_by_time: dict[int, LSMSliceDiagnostics] = {}

    for t in range(n_steps - 1, 0, -1):
        itm = intrinsic[:, t] > 0.0
        alive = exercise_time > t
        candidates = itm & alive
        n_itm = int(candidates.sum())

        if n_itm == 0:
            beta = np.zeros(4, dtype=np.float64)
            coefficients_by_time[t] = beta
            diagnostics_by_time[t] = LSMSliceDiagnostics(
                time_index=t,
                n_itm=0,
                rank=0,
                condition_number=None,
                status="no_itm_paths",
            )
            continue

        candidate_idx = np.flatnonzero(candidates)
        future_steps = exercise_time[candidate_idx] - t
        y = cashflow[candidate_idx] * np.exp(-rate * dt * future_steps)

        beta, rank, cond, status = _fit_slice_regression(
            stock_itm=paths[candidate_idx, t],
            target_itm=y,
            strike=strike,
            normalize_by_strike=normalize_by_strike,
            min_itm_paths=min_itm_paths,
        )
        coefficients_by_time[t] = beta
        diagnostics_by_time[t] = LSMSliceDiagnostics(
            time_index=t,
            n_itm=n_itm,
            rank=rank,
            condition_number=cond,
            status=status,
        )

        x = laguerre_basis(
            paths[candidate_idx, t],
            strike=strike,
            normalize_by_strike=normalize_by_strike,
        )
        continuation_hat = x @ beta
        immediate = intrinsic[candidate_idx, t]
        exercise_now = immediate >= continuation_hat

        chosen_idx = candidate_idx[exercise_now]
        if chosen_idx.size > 0:
            exercise_time[chosen_idx] = t
            cashflow[chosen_idx] = intrinsic[chosen_idx, t]

    discounted = cashflow * np.exp(-rate * dt * exercise_time)
    option_price = float(discounted.mean())

    if allow_immediate_exercise:
        immediate0 = float(max(strike - paths[0, 0], 0.0))
        if immediate0 >= option_price:
            discounted = np.full(n_paths, immediate0, dtype=np.float64)
            cashflow = np.full(n_paths, immediate0, dtype=np.float64)
            exercise_time = np.zeros(n_paths, dtype=np.int32)
            option_price = immediate0

    std_error = float(discounted.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else 0.0
    unique_t, counts = np.unique(exercise_time, return_counts=True)
    counts_by_time = {int(t): int(c) for t, c in zip(unique_t, counts, strict=True)}

    policy = LSMPolicy(
        strike=strike,
        rate=rate,
        maturity=maturity,
        n_steps=n_steps,
        normalize_by_strike=normalize_by_strike,
        min_itm_paths=min_itm_paths,
        coefficients_by_time=coefficients_by_time,
        diagnostics_by_time=diagnostics_by_time,
    )

    return LSMValuationResult(
        price=option_price,
        std_error=std_error,
        discounted_cashflows=discounted,
        cashflows_at_exercise=cashflow,
        exercise_time_index=exercise_time,
        exercise_counts_by_time=counts_by_time,
        policy=policy,
    )


def price_american_put_with_policy(
    paths: np.ndarray,
    policy: LSMPolicy,
    allow_immediate_exercise: bool = False,
) -> LSMValuationResult:
    """Value an American put out-of-sample by applying a pre-fitted LSM policy."""
    _, n_steps = _validate_paths(paths)
    if n_steps != policy.n_steps:
        raise ValueError(
            "paths step count does not match policy.n_steps: "
            f"{n_steps} vs {policy.n_steps}"
        )
    return _evaluate_policy_forward(
        paths=paths,
        strike=policy.strike,
        rate=policy.rate,
        maturity=policy.maturity,
        coefficients_by_time=policy.coefficients_by_time,
        normalize_by_strike=policy.normalize_by_strike,
        allow_immediate_exercise=allow_immediate_exercise,
        min_itm_paths=policy.min_itm_paths,
        diagnostics_by_time=policy.diagnostics_by_time,
    )
