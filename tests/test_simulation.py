import numpy as np

from lsm.simulation import simulate_gbm_paths


def test_simulation_shape_and_initial_column() -> None:
    paths = simulate_gbm_paths(
        s0=100.0,
        r=0.03,
        sigma=0.2,
        maturity=1.0,
        n_paths=128,
        n_steps=20,
        seed=123,
    )
    assert paths.shape == (128, 21)
    assert np.allclose(paths[:, 0], 100.0)


def test_simulation_deterministic_seed() -> None:
    a = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 500, 10, seed=7)
    b = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 500, 10, seed=7)
    assert np.array_equal(a, b)


def test_prices_are_positive() -> None:
    paths = simulate_gbm_paths(100.0, 0.05, 0.5, 2.0, 2000, 50, seed=42)
    assert np.all(paths > 0.0)


def test_risk_neutral_terminal_mean_behavior() -> None:
    s0 = 100.0
    r = 0.04
    sigma = 0.25
    maturity = 1.5

    paths = simulate_gbm_paths(
        s0=s0,
        r=r,
        sigma=sigma,
        maturity=maturity,
        n_paths=100_000,
        n_steps=30,
        seed=999,
    )

    empirical = paths[:, -1].mean()
    theoretical = s0 * np.exp(r * maturity)

    rel_error = abs(empirical - theoretical) / theoretical
    assert rel_error < 0.01
