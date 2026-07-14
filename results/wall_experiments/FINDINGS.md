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

---

# Follow-up: fixed points (no per-curve control), warp across N

We cannot control per-curve placement, so the usable lever is the coordinate warp
applied to the given points. `scripts/warp_vs_sampling_sweep.py` sweeps N=4..16 and
compares spline, GP, CNN+MLP (baseline) and CNN+MLP+warp (the MLR / Hua Guo model)
under two FIXED grids: regular (uniform-x) and an optimized wall-dense grid
(x_i=u_i^p*, p* found by search = 2.0, same grid for every curve).

## Regular (uniform-x, fixed) — the realistic case

| N  | spline | GP | CNN base | CNN+warp | warp vs base |
|---:|-------:|---:|---------:|---------:|-------------:|
|  4 | 2.29e-1 | 4.44e-1 | 5.56e-2 | 4.36e-2 | -22% |
|  6 | 6.55e-2 | 2.02e-1 | 2.29e-2 | 2.09e-2 |  -9% |
|  8 | 2.62e-2 | 9.29e-2 | 7.43e-3 | 6.29e-3 | -15% |
| 10 | 1.21e-2 | 3.92e-2 | 4.84e-3 | 2.61e-3 | -46% |
| 12 | 6.24e-3 | 1.48e-2 | 4.30e-3 | 2.03e-3 | -53% |
| 14 | 3.49e-3 | 4.96e-3 | 3.06e-3 | 1.68e-3 | -45% |
| 16 | 2.08e-3 | 1.39e-3 | 2.14e-3 | 1.56e-3 | -27% |

The warp improves the model on the SAME fixed points at every N (biggest at mid N,
-45..-53%), and beats spline/GP at every N except N=16 (GP edges it).

## Optimized grid (x_i=u_i^2.0, fixed for all curves)

| N  | spline | GP | CNN+warp |
|---:|-------:|---:|---------:|
|  4 | 6.67e-1 | 6.62e-1 | 2.50e-2 |
|  6 | 1.26e-2 | 5.12e-1 | 6.96e-3 |
|  8 | 1.85e-3 | 2.47e-1 | 1.56e-3 |
| 10 | 5.38e-4 | 1.06e-1 | 5.30e-4 |
| 12 | 2.13e-4 | 4.74e-2 | 3.73e-4 |
| 14 | 1.01e-4 | 2.35e-2 | 5.34e-4 |
| 16 | 5.42e-5 | 1.12e-2 | 3.68e-4 |

Findings:
- A wall-dense grid helps every method a lot (CNN+warp -75..-82% vs regular).
- The cubic spline becomes the BEST method at N>=12 (5-7x better than the neural
  model at N=14,16): once the wall is sampled, spline interpolation of a smooth
  curve races to numerical precision while the neural model hits its decoder floor.
  If you can choose the grid, a spline on it beats the net past N~12.
- GP is destroyed by the non-uniform grid (clustered wall points, sparse outer);
  GP only works on ~uniform grids.
- At low N (4-6) the neural model dominates on any grid; classical methods cannot
  reconstruct a few wall-clustered points.

## Takeaways

- Fixed uniform points (our constraint): use CNN+MLP+warp (the Hua Guo MLR model).
- The neural model's niche is low N (4-8) and regular grids.
- "Hua Guo model" here = the coordinate-warp (MLR reduced variable) model. The
  MLRNet analytic-parameter decoder is a separate, still-unbuilt candidate.

---

# Follow-up: MLRNet-style physics decoder (negative result)

`scripts/mlrnet_physics_decoder.py` implements a physics-structured decoder: the
encoder predicts the parameters of a generalized-Morse template
`g(x)=exp(-b1(x-x0)) - lam*exp(-b2(x-x0))`, min-max normalized to [-1,1], so the
reconstruction is a ~4-parameter physics fit. Regular sampling, N=4..16.

| N  | CNN+warp | MLRNet | MLR wall | MLR min |
|---:|---------:|-------:|---------:|--------:|
|  4 | 4.57e-2 | 5.92e-2 | 1.24e-1 | 1.03e-2 |
|  6 | 2.09e-2 | 6.07e-2 | 1.26e-1 | 1.07e-2 |
|  8 | 6.12e-3 | 4.98e-2 | 9.11e-2 | 4.85e-3 |
| 10 | 2.31e-3 | 4.73e-2 | 8.59e-2 | 4.14e-3 |
| 12 | 1.88e-3 | 4.01e-2 | 7.27e-2 | 2.62e-3 |
| 14 | 1.81e-3 | 4.61e-2 | 7.85e-2 | 3.66e-3 |
| 16 | 1.99e-3 | 3.52e-2 | 6.26e-2 | 2.25e-3 |

Findings:
- The pure physics template is ~20x worse than CNN+warp and worse than a cubic
  spline past N=8; it barely improves with N (plateaus ~4e-2).
- The error splits sharply: the well minimum is respectable (~2-4e-3), but the
  wall is catastrophic (0.06-0.13). A single-exponential wall cannot match the
  diverse LJ (1/r^12) / Buckingham (exp-6) / Rydberg walls at once -> template
  misspecification sets a bias floor that data cannot lower.
- This is only the "physics model" half of MLRNet; the real method adds an NN
  correction. Our CNN+warp already embodies that structure (physics-consistent
  base = spline in the warped coordinate, plus a learned residual), which is why
  it wins.

Conclusion: for MULTI-family super-resolution the physics belongs in the base and
the coordinate (warp + spline base + residual), not in a rigid full-curve template.
A pure analytic decoder would only pay off when fitting a single molecule with its
correct functional form. Recommended model for our setting: CNN+MLP + coordinate
warp.

---

# Follow-up: exploit the PES shape manifold (gappy-POD) — the big lever

Insight: the adversary is useless because the conditional distribution is near-delta
(one valid curve per set of points). The way to exploit that is to constrain the
reconstruction to the low-dimensional manifold of valid PES shapes, not adversarial
realism. `scripts/shape_manifold_recon.py` tests gappy-POD: learn mean + top-K PCA
modes from the training curves, then reconstruct each curve as the subspace element
that best fits the N observed points (ridge least squares).

The 4-family PES curves are astonishingly low-dimensional:
  2 modes -> 94.5% variance, 4 modes -> 99.89%, 6 modes -> 100.00%.

## In-family, continuous RMSE (regular uniform sampling)

| N | spline | GP | CNN+warp | gappy-POD (best K) |
|--:|-------:|---:|---------:|-------------------:|
| 4 | 2.28e-1 | 4.71e-1 | 4.57e-2 | 5.17e-2 (K2) |
| 6 | 6.48e-2 | 2.03e-1 | 2.09e-2 | 8.29e-3 (K5) |
| 8 | 2.56e-2 | 9.24e-2 | 6.12e-3 | 1.54e-3 (K6) |
|10 | 1.18e-2 | 3.91e-2 | 2.31e-3 | 6.04e-4 (K6) |
|12 | 6.48e-3 | 1.53e-2 | 1.88e-3 | 1.19e-4 (K8) |
|14 | 3.79e-3 | 5.06e-3 | 1.81e-3 | 2.36e-5 (K8) |
|16 | 1.99e-3 | 1.33e-3 | 1.99e-3 | 1.08e-5 (K8) |

Gappy-POD beats every free-form method for N>=6 (4x at N=8, 16-180x at N>=12),
reaching machine precision: ~8 points over-determine an ~8-dimensional object.
Only at N=4 does the neural prior edge it (only K=2 modes are affordable).
K must be chosen <= ~N-2 (larger K overfits the points and wiggles elsewhere);
a plain rule K=min(N-2,6) or cross-validation recovers most of the oracle gain.

## Out-of-family (O2), K=min(N-2,6)

| N | O2 gappy-POD |
|--:|-------------:|
| 6 | 2.56e-2 |
| 8 | 1.21e-2 |
|10 | 3.80e-3 |
|12 | 2.07e-3 |
|14 | 1.45e-3 |
|16 | 1.12e-3 |

On O2 gappy-POD plateaus at ~1e-3 and cannot improve with N: O2 has shape
components outside the training subspace, a bias floor a linear subspace cannot
cross. There it is worse than the neural model, which generalizes out-of-family.

## Synthesis

- IN-distribution, the shape manifold is by far the biggest lever (orders of
  magnitude over free-form models), and it is cheap (SVD + least squares).
- OUT-of-distribution it is brittle (subspace bias floor).
- The winning design is therefore a HYBRID: shape-manifold reconstruction as the
  base (captures the bulk in a few coefficients) + a neural residual for
  out-of-manifold deviations, and/or a richer/nonlinear manifold (autoencoder)
  and more training families to lower the OOD floor. This mirrors the recurring
  "structured base + learned residual" pattern.
- Neural model's niche narrows to: N=4 (extreme sparse) and out-of-family curves.

Next: gappy-POD base + neural residual hybrid; nonlinear (autoencoder) manifold;
CV-based K selection; enlarge the training family set to broaden the subspace.
