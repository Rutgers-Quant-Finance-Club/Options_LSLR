import numpy as np
import pytest

from lsm.basis import laguerre_basis


def test_basis_shape_and_constant_column() -> None:
    x = np.array([80.0, 90.0, 110.0])
    b = laguerre_basis(x, strike=100.0, normalize_by_strike=True)
    assert b.shape == (3, 4)
    assert np.allclose(b[:, 0], 1.0)


def test_basis_known_values_without_normalization() -> None:
    x = np.array([1.0])
    b = laguerre_basis(x, normalize_by_strike=False)
    e = np.exp(-0.5)

    assert b[0, 0] == pytest.approx(1.0)
    assert b[0, 1] == pytest.approx(e)
    assert b[0, 2] == pytest.approx(0.0)
    assert b[0, 3] == pytest.approx(-0.5 * e)


def test_basis_requires_strike_if_normalizing() -> None:
    x = np.array([100.0, 110.0])
    with pytest.raises(ValueError):
        laguerre_basis(x, strike=None, normalize_by_strike=True)
