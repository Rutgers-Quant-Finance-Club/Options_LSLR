import numpy as np

from lsm.validation import (
    default_table1_subset_cases,
    run_normalization_diagnostics,
    run_table1_subset_validation,
)


def test_table1_subset_validation_returns_finite_rows() -> None:
    cases = default_table1_subset_cases()[:2]
    rows = run_table1_subset_validation(cases=cases, n_paths=4_000, seeds=(1, 2))

    assert len(rows) == 2
    for row in rows:
        assert np.isfinite(row.lsm_american_mean)
        assert np.isfinite(row.lsm_early_exercise_mean)
        assert row.american_abs_error_vs_paper >= 0.0
        assert row.early_exercise_abs_error_vs_paper >= 0.0


def test_normalization_diagnostics_detects_stability_gap() -> None:
    diag = run_normalization_diagnostics(
        s0=400.0,
        strike=400.0,
        n_paths=5_000,
        n_steps=50,
        seeds=(1, 2),
    )

    assert diag.normalized.median_condition_number is not None
    assert diag.unnormalized.median_condition_number is not None
    assert diag.unnormalized.median_condition_number > diag.normalized.median_condition_number
    assert diag.unnormalized.rank_deficient_slices >= diag.normalized.rank_deficient_slices
