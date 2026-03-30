# Week 5 Backward Induction: LSM American Put Engine

## Objective
Implement the full Longstaff-Schwartz backward induction recursion for a Bermudan-discretized American put:
- learn stopping policy backwards,
- update pathwise cash flows and stopping times,
- produce a price from discounted realized exercise cash flows.

## Implementation Summary
Core implementation is in `src/lsm/american.py`:
- `price_american_put_lsm(...)`: in-sample fit + valuation via backward induction.
- `price_american_put_with_policy(...)`: out-of-sample policy application.

### State tracked in backward recursion
At each backward step `t`:
- `exercise_time[i]`: currently optimal future stopping time for path `i` given decisions from `t+1..T`.
- `cashflow[i]`: payoff realized at `exercise_time[i]`.

Regression target for candidate ITM/alive paths:
\[
Y_i = C_i \exp\{-r\Delta t (\tau_i - t)\}
\]
where:
- `C_i` is current realized future cash flow on path `i`,
- `\tau_i` is current stopping time index,
- `t` is current exercise index.

This ensures target is discounted to the current time slice before fitting.

### Exercise decision at time `t`
For ITM and alive paths:
1. Build Laguerre basis from current stock level.
2. Fit continuation model by OLS (or fallback if too few ITM points).
3. Compare immediate payoff vs fitted continuation estimate.
4. If immediate >= continuation, overwrite `exercise_time` and `cashflow` at `t`.

## Numerical Guardrails
- ITM-only regression.
- Robust fallback when ITM sample is too small: constant continuation estimate `E[Y|ITM]`.
- Optional immediate-exercise check at `t0` via `allow_immediate_exercise=True`.
- Policy object stores coefficients and per-time diagnostics for auditability.

## Output Objects
- `LSMPolicy`: strike/rate/maturity, basis settings, coefficients by time, diagnostics by time.
- `LSMValuationResult`: price, standard error, discounted path cash flows, exercise times, exercise counts.

## Week 5 Validation Performed
- American price > European baseline for canonical put parameters.
- Exercise-time/cash-flow arrays shape and range checks.
- Out-of-sample policy application sanity check.
- Deterministic edge case with optional `t0` exercise.

## Deferred to Week 6+
- Direct Table 1 replication and confidence intervals across seeds.
- In-sample vs out-of-sample statistical diagnostics grid.
- Basis sensitivity and normalization stress tests.
