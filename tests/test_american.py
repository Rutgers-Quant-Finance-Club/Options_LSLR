import numpy as np
import pytest

from lsm.american import price_american_put_lsm, price_american_put_with_policy
from lsm.black_scholes import black_scholes_put_price, european_put_mc_price_from_paths
from lsm.simulation import simulate_gbm_paths


def test_american_lsm_price_exceeds_european_baseline() -> None:
    s0, k, r, sigma, t = 40.0, 40.0, 0.06, 0.2, 1.0
    paths = simulate_gbm_paths(s0, r, sigma, t, n_paths=60_000, n_steps=50, seed=321)

    result = price_american_put_lsm(paths=paths, strike=k, rate=r, maturity=t)
    euro_mc = european_put_mc_price_from_paths(paths, strike=k, rate=r, maturity=t)
    euro_bs = black_scholes_put_price(spot=s0, strike=k, rate=r, sigma=sigma, maturity=t)

    assert result.price > euro_mc
    assert result.price > euro_bs


def test_week5_outputs_have_consistent_shapes_and_indices() -> None:
    paths = simulate_gbm_paths(40.0, 0.06, 0.2, 1.0, n_paths=20_000, n_steps=40, seed=7)
    result = price_american_put_lsm(paths=paths, strike=40.0, rate=0.06, maturity=1.0)

    assert result.discounted_cashflows.shape == (20_000,)
    assert result.cashflows_at_exercise.shape == (20_000,)
    assert result.exercise_time_index.shape == (20_000,)
    assert np.all(result.exercise_time_index >= 1)
    assert np.all(result.exercise_time_index <= 40)
    assert sum(result.exercise_counts_by_time.values()) == 20_000


def test_policy_can_be_applied_out_of_sample() -> None:
    s0, k, r, sigma, t = 40.0, 40.0, 0.06, 0.2, 1.0
    train = simulate_gbm_paths(s0, r, sigma, t, n_paths=50_000, n_steps=50, seed=11)
    test = simulate_gbm_paths(s0, r, sigma, t, n_paths=50_000, n_steps=50, seed=22)

    fit = price_american_put_lsm(paths=train, strike=k, rate=r, maturity=t)
    out = price_american_put_with_policy(paths=test, policy=fit.policy)

    assert abs(out.price - fit.price) < 0.15


def test_allow_immediate_exercise_switch() -> None:
    # Deterministic deep ITM put with zero rates/vol has same value at all dates.
    # With allow_immediate_exercise=True, model should choose t0 exercise.
    s0, k, r, sigma, t = 60.0, 100.0, 0.0, 0.0, 1.0
    paths = simulate_gbm_paths(s0, r, sigma, t, n_paths=5_000, n_steps=20, seed=5)

    default_result = price_american_put_lsm(paths=paths, strike=k, rate=r, maturity=t)
    immediate_result = price_american_put_lsm(
        paths=paths,
        strike=k,
        rate=r,
        maturity=t,
        allow_immediate_exercise=True,
    )

    assert immediate_result.price == pytest.approx(40.0, abs=1e-12)
    assert np.all(immediate_result.exercise_time_index == 0)
    assert default_result.price == pytest.approx(immediate_result.price, abs=1e-12)
