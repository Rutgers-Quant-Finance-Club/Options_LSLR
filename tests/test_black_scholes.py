import pytest

from lsm.black_scholes import black_scholes_put_price, european_put_mc_price_from_paths
from lsm.simulation import simulate_gbm_paths


def test_black_scholes_known_value_sanity() -> None:
    # Canonical parameter set used in LSM put examples.
    value = black_scholes_put_price(
        spot=40.0,
        strike=40.0,
        rate=0.06,
        sigma=0.2,
        maturity=1.0,
    )
    assert value == pytest.approx(2.066, abs=0.02)


def test_put_monotonicity_in_spot_and_vol() -> None:
    low_spot = black_scholes_put_price(spot=35.0, strike=40.0, rate=0.03, sigma=0.2, maturity=1.0)
    high_spot = black_scholes_put_price(spot=45.0, strike=40.0, rate=0.03, sigma=0.2, maturity=1.0)
    assert low_spot > high_spot

    low_vol = black_scholes_put_price(spot=40.0, strike=40.0, rate=0.03, sigma=0.1, maturity=1.0)
    high_vol = black_scholes_put_price(spot=40.0, strike=40.0, rate=0.03, sigma=0.4, maturity=1.0)
    assert high_vol > low_vol


def test_mc_converges_toward_black_scholes() -> None:
    s0 = 40.0
    k = 40.0
    r = 0.06
    sigma = 0.2
    t = 1.0

    paths = simulate_gbm_paths(
        s0=s0,
        r=r,
        sigma=sigma,
        maturity=t,
        n_paths=200_000,
        n_steps=50,
        seed=11,
    )
    mc = european_put_mc_price_from_paths(paths, strike=k, rate=r, maturity=t)
    bs = black_scholes_put_price(spot=s0, strike=k, rate=r, sigma=sigma, maturity=t)

    assert mc == pytest.approx(bs, abs=0.06)
