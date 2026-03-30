import numpy as np
import pytest

from lsm.basis import laguerre_basis
from lsm.regression import estimate_continuation_value


def test_regression_recovers_synthetic_coefficients() -> None:
    strike = 100.0
    rate = 0.05
    dt = 0.5

    stock_t = np.array([75.0, 80.0, 85.0, 90.0, 95.0, 105.0, 110.0])
    itm = stock_t < strike
    x_itm = laguerre_basis(stock_t[itm], strike=strike, normalize_by_strike=True)

    beta_true = np.array([0.1, 0.2, -0.15, 0.05])
    y_discounted_itm = x_itm @ beta_true

    future_cf = np.zeros_like(stock_t)
    future_cf[itm] = y_discounted_itm * np.exp(rate * dt)

    result = estimate_continuation_value(
        stock_prices_t=stock_t,
        realized_future_cashflows=future_cf,
        strike=strike,
        rate=rate,
        dt=dt,
        discount_steps=1,
        normalize_by_strike=True,
    )

    assert result.status == "ok"
    assert result.coefficients == pytest.approx(beta_true, abs=1e-10)
    assert result.continuation_values[itm] == pytest.approx(y_discounted_itm, abs=1e-10)


def test_regression_handles_zero_itm_paths() -> None:
    stock_t = np.array([101.0, 105.0, 130.0])
    future_cf = np.array([1.0, 2.0, 3.0])

    result = estimate_continuation_value(
        stock_prices_t=stock_t,
        realized_future_cashflows=future_cf,
        strike=100.0,
        rate=0.05,
        dt=0.1,
    )

    assert result.status == "no_itm_paths"
    assert result.n_itm == 0
    assert np.allclose(result.continuation_values, 0.0)


def test_regression_handles_rank_deficiency() -> None:
    # Only one ITM point for four features => rank-deficient by construction.
    stock_t = np.array([95.0, 101.0, 103.0, 104.0])
    future_cf = np.array([2.0, 0.0, 0.0, 0.0])

    result = estimate_continuation_value(
        stock_prices_t=stock_t,
        realized_future_cashflows=future_cf,
        strike=100.0,
        rate=0.01,
        dt=0.25,
    )

    assert result.n_itm == 1
    assert result.status == "rank_deficient"
    assert np.isfinite(result.coefficients).all()
