"""Parameter containers for simulation and pricing experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LSMConfig:
    """Model and simulation inputs for Weeks 2-4 components.

    Attributes:
        s0: Initial underlying price.
        k: Option strike.
        r: Continuously-compounded risk-free rate.
        sigma: Volatility of underlying returns.
        maturity: Time to maturity in years.
        n_paths: Number of Monte Carlo paths.
        n_steps: Number of simulation time steps / exercise dates.
        seed: Random seed for reproducibility.
    """

    s0: float
    k: float
    r: float
    sigma: float
    maturity: float
    n_paths: int
    n_steps: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.s0 <= 0.0:
            raise ValueError("s0 must be positive")
        if self.k <= 0.0:
            raise ValueError("k must be positive")
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive")
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")

    @property
    def dt(self) -> float:
        """Simulation step size in years."""
        return self.maturity / self.n_steps
