"""Core modules for Weeks 1-5 of a Longstaff-Schwartz LSM project."""

from .american import (
    LSMPolicy,
    LSMSliceDiagnostics,
    LSMValuationResult,
    price_american_put_lsm,
    price_american_put_with_policy,
)
from .black_scholes import black_scholes_put_price, european_put_mc_price_from_paths
from .basis import laguerre_basis
from .config import LSMConfig
from .regression import RegressionResult, estimate_continuation_value
from .simulation import simulate_gbm_paths
from .validation import (
    NormalizationDiagnosticResult,
    NormalizationStats,
    Table1Case,
    Table1ValidationRow,
    default_table1_subset_cases,
    run_normalization_diagnostics,
    run_table1_subset_validation,
)

__all__ = [
    "LSMConfig",
    "LSMPolicy",
    "RegressionResult",
    "LSMSliceDiagnostics",
    "LSMValuationResult",
    "black_scholes_put_price",
    "european_put_mc_price_from_paths",
    "estimate_continuation_value",
    "laguerre_basis",
    "NormalizationDiagnosticResult",
    "NormalizationStats",
    "price_american_put_lsm",
    "price_american_put_with_policy",
    "run_normalization_diagnostics",
    "run_table1_subset_validation",
    "simulate_gbm_paths",
    "Table1Case",
    "Table1ValidationRow",
    "default_table1_subset_cases",
]
