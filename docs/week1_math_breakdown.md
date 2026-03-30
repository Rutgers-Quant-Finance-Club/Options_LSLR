# Week 1 Math Breakdown: Longstaff-Schwartz LSM (Sections 1-3)

## Objective and Scope
This memo captures the mathematical logic required before implementation. The current codebase covers Weeks 1-4 only:
- Week 2: risk-neutral GBM simulation
- Week 3: European put baseline (analytic + MC)
- Week 4: single-time-slice LSM continuation regression

Full backward induction and policy rollout are intentionally deferred to Week 5.

## Section 1-3 Summary in Plain Quant Terms

### 1. Why American options are harder than European options
A European option has a fixed exercise date, so valuation reduces to one discounted expectation under the risk-neutral measure. An American option introduces an optimal stopping problem: at each exercise date, the holder chooses between:
1. immediate exercise value (intrinsic value), and
2. continuation value (conditional expectation of discounted future cash flows if not exercised now).

That local compare-and-decide rule must be solved jointly across time because later exercise decisions feed back into earlier continuation values.

### 2. Core LSM insight
The continuation value is a conditional expectation that is generally unavailable in closed form for complex dynamics/payoffs. LSM approximates this conditional expectation cross-sectionally:
- simulate many paths under the risk-neutral measure,
- at a given time slice, regress discounted realized future cash flows on basis functions of current state variables,
- use fitted values as continuation-value estimates.

This turns dynamic programming into repeated least-squares problems.

### 3. Why ITM-only regression is the default
For a put, exercise is relevant only when intrinsic value is positive. Paths that are out-of-the-money are not candidates for immediate exercise, so including them in regression broadens the approximation domain unnecessarily and typically increases noise. The paper emphasizes ITM-only regression for efficiency and accuracy in the exercise-relevant region.

### 4. Why backward induction is required
Continuation cash flows at time \(t_k\) depend on whether exercise was optimal at \(t_{k+1}, t_{k+2}, \ldots\). Therefore, one must solve from maturity backward:
- at maturity, payoff is known,
- move to prior date, estimate continuation from future realized cash flows,
- update stopping decisions,
- continue backward until earliest exercise date.

### 5. Section 3 put-option assumptions for this project
The simple benchmark setting is:
- risk-neutral GBM for stock under \(Q\):
  \[
  dS_t = r S_t dt + \sigma S_t dW_t
  \]
- no dividends,
- discrete exercise dates (Bermudan approximation to continuous American exercise),
- put payoff \(\max(K-S_t, 0)\),
- risk-neutral discounting with continuously compounded rate \(r\).

## Translation to Implementation Checklist
- [x] Represent market/simulation parameters in a typed config object.
- [x] Simulate risk-neutral GBM paths with exact lognormal step.
- [x] Validate European baseline before touching American exercise logic.
- [x] Implement intrinsic value and discounting primitives.
- [x] Implement Laguerre basis (constant + first three weighted terms).
- [x] Add explicit normalization option \(x = S/K\) to manage scaling.
- [x] Implement ITM mask at a given time slice.
- [x] Regress discounted realized future continuation cash flows \(Y\) on basis \(X\).
- [x] Return continuation estimates only as regression outputs, not exercise policy yet.
- [x] Add tests for shape, reproducibility, numerical sanity, and edge cases.

## Equations and Objects Needed in Code

### Core equations
1. **GBM exact discretization**
   \[
   S_{t+\Delta t} = S_t \exp\left((r-\tfrac12\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z\right), \quad Z\sim\mathcal{N}(0,1)
   \]

2. **European put baseline**
   \[
   P_{BS} = K e^{-rT} \Phi(-d_2) - S_0 \Phi(-d_1)
   \]
   with standard \(d_1,d_2\).

3. **Intrinsic value (put)**
   \[
   h_t = (K - S_t)^+
   \]

4. **Week 4 regression target**
   \[
   Y_i = e^{-r\Delta t \cdot m} \cdot C^{\text{realized}}_{i,\text{future}}
   \]
   where \(m\) is the number of time steps between current slice and cash-flow realization.

5. **Continuation approximation**
   \[
   \hat{F}(S_t) = \beta_0 + \beta_1 L_0(x) + \beta_2 L_1(x) + \beta_3 L_2(x), \quad x = S_t/K\ \text{(default)}
   \]

6. **Weighted Laguerre basis used here**
   \[
   L_0(x)=e^{-x/2},\quad
   L_1(x)=e^{-x/2}(1-x),\quad
   L_2(x)=e^{-x/2}(1-2x+x^2/2)
   \]

### Code objects (Weeks 1-4)
- `LSMConfig`: parameter container and `dt` helper.
- `simulate_gbm_paths(...)`: risk-neutral path engine.
- `black_scholes_put_price(...)`: analytic European put.
- `european_put_mc_price_from_paths(...)`: MC estimator from simulated terminal prices.
- `laguerre_basis(...)`: feature matrix builder.
- `estimate_continuation_value(...)`: ITM filtering + OLS continuation fit for one time slice.

## Pseudocode for Full Backward Induction (Week 5 target)
```text
Inputs: simulated paths S[path, t], strike K, rate r, dt, basis functions phi(.)

Initialize cashflows at maturity:
  CF[path] = max(K - S[path, T], 0)
  exercise_time[path] = T if CF[path] > 0 else None

For t from T-1 down to 1:
  intrinsic[path] = max(K - S[path, t], 0)
  itm = intrinsic > 0

  if any(itm):
    Y = discounted realized future CF for itm paths
    X = phi(S[itm, t])
    beta = OLS(X, Y)
    continuation_hat = X @ beta

    exercise_now = itm and (intrinsic[itm] >= continuation_hat)
    update CF[path] for exercised paths to intrinsic at t
    zero-out later CF contributions for paths exercised now

Price at t=0:
  discount each path CF from its exercise time back to 0
  return path average
```

## Common Implementation Failure Modes
1. **Regressing on all paths rather than ITM-only**
   Inflates estimation noise in regions where exercise is irrelevant.

2. **Wrong regression target**
   Using same-time intrinsic payoff (or fitted continuation itself) as `Y` instead of discounted realized future continuation cash flow.

3. **Forgetting discounting in `Y`**
   Produces inconsistent time units and biased exercise decisions.

4. **Mixing normalized and unnormalized quantities**
   Example: basis uses `S/K` while targets/intrinsic are treated as if unnormalized without documentation.

5. **Shape misalignment in design matrix**
   Path filtering errors commonly create mismatched `(n_itm, n_features)` and `(n_itm,)` arrays.

6. **Confusing continuation estimate vs realized future cash flow**
   `Y` is a realized pathwise quantity from future decisions; continuation estimate is regression fitted value at current state.

7. **Accidental look-ahead / cash-flow overwrite bug**
   Incorrectly using future information unavailable at time `t` or overwriting already-exercised cash-flow states.

8. **Upward-bias bug from wrong max-discount pattern**
   Discounting `max(intrinsic, noisy continuation estimate)` pathwise without first deriving the stopping rule from regression on realized continuation cash flows.

## Current Validation Boundary (end of Week 4)
Already validated:
- path generator mechanics and reproducibility,
- European analytic vs Monte Carlo baseline,
- basis shape and regression edge handling for one slice.

Not yet validated (Week 5+):
- full backward induction recursion,
- full American put valuation and comparison to paper table values,
- in-sample vs out-of-sample stopping-policy diagnostics.
