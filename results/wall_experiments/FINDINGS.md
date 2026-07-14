# Reducing the repulsive-wall reconstruction error

## Problem

Error analysis (`results/error_vs_position.png`) showed the coordinate model's
error is dominated by the repulsive wall: with uniform sampling, ~78% of the
squared error sits in `x < 0.15` (15% of the domain), because the wall is the
steepest region and the least-sampled by uniform points. Averaged over the whole
curve this inflates the reported RMSE well above the error in the
chemically-relevant well region.

## Experiment

`scripts/wall_error_experiments.py` — coordinate CNN+MLP, residual-on-spline,
channels `[energy, derivative, morse]`, N=8 observed points, 4 families,
80 test curves. Error reported by physical-x region so configs are comparable.

Ideas tested:
- **B** steeper Morse descriptor (larger beta).
- **C** coordinate warp `x = m^2` (stretches the wall in the model coordinate)
  with observations still uniform in `x` — representation change only, works on
  fixed data.
- **D** warp + sample uniformly in the warped coordinate `m`, i.e. place points
  denser near the wall (`x_i = m_i^2`). This is the MLR reduced-variable idea.
- **E** D with a slightly sharper Morse descriptor.

## Results (normalized RMSE; cm^-1 columns at an example H-bond scale, x2250)

| config | wall | wall cm^-1 | well | min | min cm^-1 | full |
|--------|-----:|-----:|-----:|----:|-----:|-----:|
| A baseline (uniform-x)      | 1.67e-2 | 37.7 | 4.18e-3 | 6.22e-3 | 14.0 | 7.31e-3 |
| B steeper morse (b=10)      | 1.68e-2 | 37.7 | 4.43e-3 | 5.90e-3 | 13.3 | 7.48e-3 |
| C warp-repr (obs uniform-x) | 1.44e-2 | 32.4 | 1.34e-3 | 2.00e-3 |  4.5 | 6.07e-3 |
| D warp + wall-dense sample  | 3.01e-3 |  6.8 | 1.21e-3 | 9.74e-4 |  2.2 | 1.57e-3 |
| E D + morse b=8             | 2.94e-3 |  6.6 | 9.76e-4 | 6.31e-4 |  1.4 | 1.64e-3 |

## Conclusions

1. **Sharpening the Morse descriptor alone does nothing (B).** The wall problem is
   sampling/resolution, not descriptor steepness.
2. **The coordinate warp works, and its effect splits by whether you control
   sampling:**
   - **D / E (warp + wall-dense sampling): wall error -82%, full RMSE -79%,
     well-minimum error -84%.** This is the clear winner. It requires placing
     observations denser near the wall, which is controllable for ab initio PES
     scans (you choose the geometries).
   - **C (warp representation only, fixed uniform data): wall -14%, but
     well/minimum -68%** at a small cost in the outer region. This is the "free"
     option when the observation grid is fixed and cannot be re-sampled.
3. The outer region (x>0.45) is already accurate (~1e-3); the warp trades a little
   of its resolution for the wall, which is a good trade. The long-range tail is
   owned by the separate long-range function, so its budget here is not critical.

## Recommendation

- If sampling is controllable: adopt **D** (warp + wall-dense placement).
- If the data grid is fixed: adopt **C** (coordinate warp) — most of the
  well-region gain for free.
- A tunable warp exponent `p` (here p=2) and the crossover between C and D across
  N are the natural next sweeps.

Exploratory only — not merged into the main branch.
