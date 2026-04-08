# Options LSM

This repository is a staged implementation of Longstaff-Schwartz least-squares Monte Carlo (LSM) for American puts, built as a week-by-week quant research project rather than a one-shot script.


## Why this project exists

American options are not hard because of payoff math, they are hard because of stopping decisions.

At every exercise date, you have to decide between:
- immediate exercise value,
- continuation value (a conditional expectation under the risk-neutral measure).

That continuation value is not available in closed form in general, so Longstaff-Schwartz estimates it from simulated paths via cross-sectional regression. This repo implements that pipeline in a controlled sequence.

## What is currently implemented

- Week 1: literature and mathematical breakdown from Sections 1-3 of Longstaff-Schwartz.
- Week 2: risk-neutral GBM simulation engine.
- Week 3: European put baseline with Black-Scholes and Monte Carlo cross-check.
- Week 4: single-time-slice continuation regression (ITM-only).
- Week 5: full backward induction and stopping-policy recursion.
- Week 6: validation against a Table-1 subset and explicit normalization diagnostics.

## Repository map

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── docs/
│   ├── week1_math_breakdown.md
│   ├── week5_backward_induction.md
│   └── week6_validation_report.md
├── notebooks/
│   ├── 01_lsm_paper_math_walkthrough.ipynb
│   ├── 02_gbm_and_european_baseline.ipynb
│   ├── 03_regression_engine_demo.ipynb
│   └── 04_backward_induction_demo.ipynb
├── scripts/
│   └── run_week6_validation.py
├── src/
│   └── lsm/
│       ├── __init__.py
│       ├── american.py
│       ├── basis.py
│       ├── black_scholes.py
│       ├── config.py
│       ├── regression.py
│       ├── simulation.py
│       ├── utils.py
│       └── validation.py
└── tests/
    ├── test_american.py
    ├── test_basis.py
    ├── test_black_scholes.py
    ├── test_regression.py
    ├── test_simulation.py
    └── test_validation.py
```

## Week-by-week development log:

## Week 1: Paper first, code second

What happened:
- Extracted the algorithmic logic from Sections 1-3.
- Wrote a structured memo in `docs/week1_math_breakdown.md`.
- Documented common failure modes before implementation.

Why this week mattered:
- LSM failures are usually conceptual, not syntactic.
- The most common production bug is regressing the wrong target.
- Getting definitions right up front avoids a quiet, biased model later.

Core decisions locked in:
- Regression target is discounted realized future continuation cash flow.
- ITM-only regression is the default.
- Backward induction is mandatory, not optional.
- Laguerre basis includes normalization option (`x = S/K`) to address scale issues noted in the paper.

## Week 2: Risk-neutral GBM simulation

What happened:
- Built vectorized path simulation in `src/lsm/simulation.py`.
- Used exact lognormal discretization.
- Added deterministic seed support.

Why this week mattered:
- If the risk-neutral engine is wrong, every downstream result is contaminated.
- Exact discretization reduces avoidable discretization bias in the state process.

Validation added:
- Path shape and initial-column checks.
- Positivity checks.
- Reproducibility under fixed seed.
- Terminal mean sanity check versus `S0 * exp(rT)`.

## Week 3: European put baseline

What happened:
- Implemented Black-Scholes put in `src/lsm/black_scholes.py`.
- Implemented MC European estimator from simulated terminal payoffs.
- Added comparison utility for analytic vs MC error tracking.

Why this week mattered:
- American pricing should only be trusted after the European baseline is correct.
- This baseline isolates simulation quality from stopping-policy logic.

Validation added:
- Known-value sanity (canonical `S0=K=40, r=0.06, sigma=0.2, T=1`).
- Monotonicity in spot and volatility.
- MC convergence toward closed-form price at larger `N`.

## Week 4: Continuation regression engine

What happened:
- Implemented weighted Laguerre basis in `src/lsm/basis.py`.
- Implemented single-slice LSM continuation estimator in `src/lsm/regression.py`.
- Added ITM filtering and robust edge-case handling.

Why this week mattered:
- This is the statistical core of LSM.
- It must estimate `E[continuation | state]`, not fit noise from unrelated paths.

Validation added:
- Basis shape/value checks.
- Synthetic-coefficient recovery test for regression.
- Edge-case tests for no-ITM and rank-deficient slices.

## Week 5: Full backward induction

What happened:
- Built backward induction engine in `src/lsm/american.py`.
- Added policy object (`LSMPolicy`) and valuation result object (`LSMValuationResult`).
- Added out-of-sample policy application path (`price_american_put_with_policy`).

Why this week mattered:
- Week 4 by itself estimates one slice.
- A real American pricer needs recursive cash-flow updates and stopping-time overwrites from maturity backward.

Implementation details:
- Start with maturity payoff.
- At each earlier date, restrict to ITM + alive paths.
- Build regression target by discounting currently realized future cashflows back to current time.
- Exercise when immediate payoff >= fitted continuation.
- Overwrite stopping time and cashflow for exercised paths.

Validation added:
- American value exceeds European baseline in canonical cases.
- Exercise index and cashflow arrays are consistent and bounded.
- Out-of-sample policy valuation is stable relative to in-sample fit.

## Week 6: Validation and scaling diagnostics

What happened:
- Added benchmarking utilities in `src/lsm/validation.py`.
- Added reproducible report script in `scripts/run_week6_validation.py`.
- Generated `docs/week6_validation_report.md`.
- Added normalization diagnostics comparing `x=S/K` vs `x=S`.

Why this week mattered:
- “Model works” is not enough; we need quantified error against known references.
- Weighted Laguerre polynomials are sensitive to scale; normalization must be tested, not assumed.

Week 6 outcomes (current run):

- Table-1 subset comparison across 12 cases (3 seeds, 30k paths/seed):
- Mean absolute error vs paper American values: `0.0182`.
- Max absolute error vs paper American values: `0.0357`.
- Mean absolute error vs paper early-exercise values: `0.0182`.
- Normalization diagnostics: `x=S/K` had `0` rank-deficient slices vs `x=S` with `147` rank-deficient slices.

## Data flow

`Model parameters -> GBM paths -> European baseline checks -> ITM continuation regressions -> backward induction stopping updates -> American valuation -> Week 6 benchmark/diagnostic reports`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest -q
```

## Run Week 6 validation report

```bash
PYTHONPATH=src python scripts/run_week6_validation.py
```

This writes:
- `docs/week6_validation_report.md`

## Minimal usage examples

```python
from lsm.simulation import simulate_gbm_paths
from lsm.black_scholes import black_scholes_put_price
from lsm.american import price_american_put_lsm

paths = simulate_gbm_paths(
    s0=40.0,
    r=0.06,
    sigma=0.2,
    maturity=1.0,
    n_paths=100_000,
    n_steps=50,
    seed=7,
)

euro_bs = black_scholes_put_price(
    spot=40.0,
    strike=40.0,
    rate=0.06,
    sigma=0.2,
    maturity=1.0,
)

american = price_american_put_lsm(
    paths=paths,
    strike=40.0,
    rate=0.06,
    maturity=1.0,
)

print(euro_bs, american.price, american.std_error)
```

## Current assumptions and boundaries

- No dividends.
- Constant `r` and `sigma`.
- Discrete exercise dates (Bermudan approximation to continuous exercise).
- Single-factor stock process in this phase.

## What is not done yet (Week 7+)

- Speed and memory optimization focused runs.
- Full paper table replication with larger path budgets and confidence-interval analysis per regime.
- Systematic basis-family sensitivity studies beyond Laguerre.
- Visualization layer for exercise boundary evolution.
