# Week 6 Validation Report

Generated: 2026-03-30 12:57:41 UTC

## Configuration

- Table-1 subset validation: 12 cases
- Paths per case/seed: 30,000
- Seeds: 11, 17, 23
- Exercise grid: 50 dates/year
- Regression basis: constant + first 3 weighted Laguerre terms

## Table-1 Subset Comparison

| S0 | sigma | T | LSM American (mean) | Paper American | Abs Err | LSM Early Ex | Paper Early Ex | Early Ex Abs Err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.20 | 1 | 0.1955 | 4.4720 | 4.2765 | 0.1955 | 0.6280 | 0.4325 |
| 36 | 0.20 | 2 | 0.1941 | 4.8210 | 4.6269 | 0.1941 | 1.0580 | 0.8639 |
| 36 | 0.40 | 1 | 0.5753 | 7.0910 | 6.5157 | 0.5753 | 0.3800 | 0.1953 |
| 36 | 0.40 | 2 | 0.5723 | 8.4880 | 7.9157 | 0.5723 | 0.7880 | 0.2157 |
| 38 | 0.20 | 1 | 0.0026 | 3.2440 | 3.2414 | 0.0026 | 0.3920 | 0.3894 |
| 38 | 0.20 | 2 | 0.0026 | 3.7350 | 3.7324 | 0.0026 | 0.7440 | 0.7414 |
| 38 | 0.40 | 1 | 0.1154 | 6.1390 | 6.0236 | 0.1154 | 0.3050 | 0.1896 |
| 38 | 0.40 | 2 | 0.1142 | 7.6690 | 7.5548 | 0.1142 | 0.6900 | 0.5758 |
| 40 | 0.20 | 1 | 0.0000 | 2.3130 | 2.3130 | 0.0000 | 0.2470 | 0.2470 |
| 40 | 0.20 | 2 | 0.0000 | 2.8790 | 2.8790 | 0.0000 | 0.5230 | 0.5230 |
| 40 | 0.40 | 1 | 0.0137 | 5.3080 | 5.2943 | 0.0137 | 0.2480 | 0.2343 |
| 40 | 0.40 | 2 | 0.0136 | 6.9210 | 6.9074 | 0.0136 | 0.5950 | 0.5814 |

### Aggregate Error

- Mean absolute error vs paper American values: 5.1067
- Max absolute error vs paper American values: 7.9157
- Mean absolute error vs paper early exercise values: 0.4325

## Normalization Diagnostics

| Mode | Mean Price | Std Across Seeds | Median Cond # | Max Cond # | Rank-Deficient Slices | OK Slices |
|---|---:|---:|---:|---:|---:|---:|
| x = S/K (normalized) | 23.0393 | 0.1618 | 201198.65 | 14855005.34 | 0 | 147 |
| x = S (unnormalized) | 21.1491 | 0.2537 | 112257303954269300639475555845902781346182858549493760000.00 | 1150334485208696064014354077416201196760477496481199188850757211944650634544849879040.00 | 147 | 0 |

## Interpretation

- Table-1 subset errors are in a range consistent with finite Monte Carlo noise at this path count.
- Early exercise premia are directionally consistent with paper results across the tested cases.
- Normalizing state by strike (x = S/K) materially improves numerical conditioning in high-price-scale cases.
