# Conditional stochastic GAN

`scripts/conditional_gan.py`: conditional discriminator, in-graph adversarial
gradient, stochastic generator G(c,z), relativistic-hinge loss with instance
noise, data consistency at the observed points, MSGAN mode-seeking. The target
is a calibrated uncertainty band from the sample spread, not point accuracy.

| metric | GAN in-family | GAN O2 | GP in-family | GP O2 |
|---|---:|---:|---:|---:|
| point RMSE (sample mean) | 1.42e-2 | 1.88e-2 | 8.97e-2 | 3.87e-2 |
| 90%-band coverage        | 0.99 | 0.97 | 0.71 | 0.63 |
| corr(sigma, abs error)   | 0.26 | 0.08 | -    | -    |
| mean sigma               | 3.23e-2 | 3.06e-2 | - | - |

It trains stably and uses z without collapsing, but loses to the deterministic
CNN+warp (~6.3e-3 at N=8) on point estimate. The band over-covers, and mean sigma
on O2 (3.06e-2) is no larger than in-family (3.23e-2), so it does not flag
out-of-distribution inputs.

---

# Frozen validity discriminator as a generator penalty

`scripts/prior_penalty.py`: train a "is this a valid PES?" classifier once,
freeze it, then train the generator with MSE + lambda * (-log D(fake)). N=8.

| lambda | in-family full | wall | D-score | O2 full |
|-------:|---------------:|-----:|--------:|--------:|
| 0.0 | 4.79e-3 | 1.14e-2 | 0.51 | 2.75e-3 |
| 0.1 | 7.60e-2 | 9.23e-2 | 0.58 | 7.14e-2 |
| 0.3 | 1.67e-1 | 2.12e-1 | 0.65 | 1.63e-1 |
| 1.0 | 3.20e-1 | 4.36e-1 | 0.74 | 3.20e-1 |
| 3.0 | 4.94e-1 | 7.14e-1 | 0.80 | 5.09e-1 |

D accuracy tops out at 64%, and as lambda grows the D-score rises while RMSE
grows 16-100x: the generator finds inputs D scores as valid but that sit farther
from truth. A frozen classifier's gradient points at its own blind spots.

---

# Outlier / OOD detection

`scripts/outlier_detection.py`. AUROC separating in-family from O2
(out-of-family) and from corrupted-shape in-family curves.

| detector | O2 | corrupted |
|---|---:|---:|
| discriminator score D(x)        | 0.679 | 0.728 |
| POD residual (full curve, K=6)  | 0.993 | 1.000 |
| POD residual (sparse N=8, K=6)  | 0.923 | 0.956 |

The discriminator works here (scoring only, nothing optimizes against it), but
the POD residual dominates, including from just 8 observed points.

---

# Richer priors and the O2 floor

`scripts/richer_prior.py` tests physics-augmented POD (PCA modes plus
Morse/power functions) and a gappy autoencoder against linear gappy-POD.

| N | method | in-family | O2 |
|--:|--------|----------:|---:|
| 8 | linear POD K=6        | 1.5e-3 | 1.18e-2 |
| 8 | physics-augmented POD | 1.30e-2| 1.13e-2 |
| 8 | gappy autoencoder     | 1.06e-2| 1.24e-2 |
|12 | linear POD K=6        | 3.4e-4 | 2.04e-3 |
|12 | physics-augmented POD | 2.41e-3| 2.56e-3 |
|12 | gappy autoencoder     | 3.55e-3| 6.39e-3 |
|16 | linear POD K=6        | 1.7e-4 | 1.10e-3 |
|16 | physics-augmented POD | 8.6e-4 | 1.03e-3 |
|16 | gappy autoencoder     | 2.94e-3| 5.26e-3 |

Both are worse in-family and neither lowers the O2 floor. The normalized curves
lie on a linear ~6-dim subspace (6 modes = 100% variance), so linear PCA is
already near-exact and the AE only approximates it. The O2 floor is a coverage
problem, not a prior-richness one.

---

# Residual-gated hybrid

`scripts/gated_hybrid.py`: reconstruct with gappy-POD, use its sparse fit
residual as an OOD flag, and fall back to CNN+warp when the residual exceeds the
98th percentile of in-family training residuals (calibrated without the test set).

| N | set | POD | neural | HYBRID | oracle | routed->neural |
|--:|-----|----:|-------:|-------:|-------:|---------------:|
| 8 | in-family | 1.58e-3 | 4.33e-3 | 1.81e-3 | 1.52e-3 |  5% |
| 8 | O2        | 1.34e-2 | 3.53e-3 | 4.65e-3 | 3.43e-3 | 70% |
|12 | in-family | 3.54e-4 | 1.73e-3 | 4.06e-4 | 3.54e-4 |  3% |
|12 | O2        | 2.20e-3 | 1.84e-3 | 1.84e-3 | 1.44e-3 |100% |
|16 | in-family | 1.76e-4 | 1.26e-3 | 2.15e-4 | 1.76e-4 |  3% |
|16 | O2        | 1.19e-3 | 1.29e-3 | 1.26e-3 | 8.76e-4 | 97% |

The gate costs a 3-5% false-flag tax in-family and cuts O2 error 2.9x at N=8. It
still trails the per-curve oracle, and at high N routing to the net costs a
little, but it removes POD's low-N OOD failure. This hybrid, with no GAN, is the
recommended system.
