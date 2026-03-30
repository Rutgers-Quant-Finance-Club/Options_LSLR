"""Week 6 validation utilities for benchmarking and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .american import price_american_put_lsm
from .black_scholes import black_scholes_put_price
from .simulation import simulate_gbm_paths


@dataclass(frozen=True)
class Table1Case:
    """One parameter case from the American put table-style benchmark."""

    s0: float
    sigma: float
    maturity: float
    paper_simulated_american: float
    paper_european: float


@dataclass(frozen=True)
class Table1ValidationRow:
    """Aggregated LSM results against a paper benchmark row."""

    case: Table1Case
    n_paths: int
    seeds: tuple[int, ...]
    lsm_american_mean: float
    lsm_american_std_across_seeds: float
    lsm_european_bs: float
    lsm_early_exercise_mean: float
    paper_early_exercise: float
    american_abs_error_vs_paper: float
    early_exercise_abs_error_vs_paper: float


@dataclass(frozen=True)
class NormalizationStats:
    """Summary metrics for one normalization setting."""

    normalize_by_strike: bool
    mean_price: float
    std_price_across_seeds: float
    median_condition_number: float | None
    max_condition_number: float | None
    rank_deficient_slices: int
    ok_slices: int


@dataclass(frozen=True)
class NormalizationDiagnosticResult:
    """Comparison of normalized vs unnormalized basis behavior."""

    normalized: NormalizationStats
    unnormalized: NormalizationStats


def default_table1_subset_cases() -> tuple[Table1Case, ...]:
    """Subset of Table 1 rows visible in the project paper copy.

    Parameters are for strike=40, rate=0.06.
    """
    return (
        Table1Case(s0=36.0, sigma=0.20, maturity=1.0, paper_simulated_american=4.472, paper_european=3.844),
        Table1Case(s0=36.0, sigma=0.20, maturity=2.0, paper_simulated_american=4.821, paper_european=3.763),
        Table1Case(s0=36.0, sigma=0.40, maturity=1.0, paper_simulated_american=7.091, paper_european=6.711),
        Table1Case(s0=36.0, sigma=0.40, maturity=2.0, paper_simulated_american=8.488, paper_european=7.700),
        Table1Case(s0=38.0, sigma=0.20, maturity=1.0, paper_simulated_american=3.244, paper_european=2.852),
        Table1Case(s0=38.0, sigma=0.20, maturity=2.0, paper_simulated_american=3.735, paper_european=2.991),
        Table1Case(s0=38.0, sigma=0.40, maturity=1.0, paper_simulated_american=6.139, paper_european=5.834),
        Table1Case(s0=38.0, sigma=0.40, maturity=2.0, paper_simulated_american=7.669, paper_european=6.979),
        Table1Case(s0=40.0, sigma=0.20, maturity=1.0, paper_simulated_american=2.313, paper_european=2.066),
        Table1Case(s0=40.0, sigma=0.20, maturity=2.0, paper_simulated_american=2.879, paper_european=2.356),
        Table1Case(s0=40.0, sigma=0.40, maturity=1.0, paper_simulated_american=5.308, paper_european=5.060),
        Table1Case(s0=40.0, sigma=0.40, maturity=2.0, paper_simulated_american=6.921, paper_european=6.326),
    )


def run_table1_subset_validation(
    cases: tuple[Table1Case, ...] | None = None,
    *,
    strike: float = 40.0,
    rate: float = 0.06,
    n_paths: int = 50_000,
    seeds: tuple[int, ...] = (11, 17, 23),
    min_itm_paths: int = 8,
    normalize_by_strike: bool = True,
) -> list[Table1ValidationRow]:
    """Run Table-1-style LSM validation across a benchmark subset."""
    if cases is None:
        cases = default_table1_subset_cases()
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if len(seeds) == 0:
        raise ValueError("seeds must be non-empty")

    rows: list[Table1ValidationRow] = []

    for case in cases:
        american_prices: list[float] = []
        for seed in seeds:
            n_steps = int(round(50 * case.maturity))
            paths = simulate_gbm_paths(
                s0=case.s0,
                r=rate,
                sigma=case.sigma,
                maturity=case.maturity,
                n_paths=n_paths,
                n_steps=n_steps,
                seed=seed,
            )
            result = price_american_put_lsm(
                paths=paths,
                strike=strike,
                rate=rate,
                maturity=case.maturity,
                normalize_by_strike=normalize_by_strike,
                min_itm_paths=min_itm_paths,
            )
            american_prices.append(result.price)

        american_arr = np.asarray(american_prices, dtype=np.float64)
        lsm_american_mean = float(american_arr.mean())
        lsm_american_std = float(american_arr.std(ddof=1)) if american_arr.size > 1 else 0.0

        lsm_european_bs = float(
            black_scholes_put_price(
                spot=case.s0,
                strike=strike,
                rate=rate,
                sigma=case.sigma,
                maturity=case.maturity,
            )
        )
        lsm_early = lsm_american_mean - lsm_european_bs
        paper_early = case.paper_simulated_american - case.paper_european

        rows.append(
            Table1ValidationRow(
                case=case,
                n_paths=n_paths,
                seeds=seeds,
                lsm_american_mean=lsm_american_mean,
                lsm_american_std_across_seeds=lsm_american_std,
                lsm_european_bs=lsm_european_bs,
                lsm_early_exercise_mean=lsm_early,
                paper_early_exercise=paper_early,
                american_abs_error_vs_paper=abs(lsm_american_mean - case.paper_simulated_american),
                early_exercise_abs_error_vs_paper=abs(lsm_early - paper_early),
            )
        )

    return rows


def _summarize_normalization_mode(
    *,
    s0: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    n_paths: int,
    n_steps: int,
    seeds: tuple[int, ...],
    normalize_by_strike: bool,
    min_itm_paths: int,
) -> NormalizationStats:
    prices: list[float] = []
    condition_numbers: list[float] = []
    rank_deficient = 0
    ok = 0

    for seed in seeds:
        paths = simulate_gbm_paths(
            s0=s0,
            r=rate,
            sigma=sigma,
            maturity=maturity,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
        )
        result = price_american_put_lsm(
            paths=paths,
            strike=strike,
            rate=rate,
            maturity=maturity,
            normalize_by_strike=normalize_by_strike,
            min_itm_paths=min_itm_paths,
        )
        prices.append(result.price)

        for diag in result.policy.diagnostics_by_time.values():
            if diag.status == "ok":
                ok += 1
            elif diag.status == "rank_deficient":
                rank_deficient += 1
            if diag.condition_number is not None and np.isfinite(diag.condition_number):
                condition_numbers.append(diag.condition_number)

    p = np.asarray(prices, dtype=np.float64)
    median_cond = float(np.median(condition_numbers)) if condition_numbers else None
    max_cond = float(np.max(condition_numbers)) if condition_numbers else None

    return NormalizationStats(
        normalize_by_strike=normalize_by_strike,
        mean_price=float(p.mean()),
        std_price_across_seeds=float(p.std(ddof=1)) if p.size > 1 else 0.0,
        median_condition_number=median_cond,
        max_condition_number=max_cond,
        rank_deficient_slices=rank_deficient,
        ok_slices=ok,
    )


def run_normalization_diagnostics(
    *,
    s0: float = 400.0,
    strike: float = 400.0,
    rate: float = 0.06,
    sigma: float = 0.20,
    maturity: float = 1.0,
    n_paths: int = 40_000,
    n_steps: int = 50,
    seeds: tuple[int, ...] = (5, 13, 29),
    min_itm_paths: int = 8,
) -> NormalizationDiagnosticResult:
    """Compare normalized vs unnormalized Laguerre basis stability and pricing."""
    normalized = _summarize_normalization_mode(
        s0=s0,
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        n_paths=n_paths,
        n_steps=n_steps,
        seeds=seeds,
        normalize_by_strike=True,
        min_itm_paths=min_itm_paths,
    )
    unnormalized = _summarize_normalization_mode(
        s0=s0,
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        n_paths=n_paths,
        n_steps=n_steps,
        seeds=seeds,
        normalize_by_strike=False,
        min_itm_paths=min_itm_paths,
    )

    return NormalizationDiagnosticResult(normalized=normalized, unnormalized=unnormalized)
