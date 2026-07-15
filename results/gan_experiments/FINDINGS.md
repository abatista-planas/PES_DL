# A correct conditional stochastic GAN — what it can and cannot do

`scripts/conditional_gan.py` is a correctly-coded conditional stochastic GAN,
built after auditing/deleting the broken legacy gan*.py drafts. It fixes every
structural flaw of the old pipeline:
- CONDITIONAL discriminator (sees candidate + conditioning), spectral norm, no BatchNorm.
- In-graph adversarial gradient (no numpy/detach breaks) -> D actually reaches G.
- STOCHASTIC generator G(c,z): latent z so different z -> different plausible curve.
- Relativistic-hinge loss + instance noise; data-consistency at observed points;
  weak mean anchor; MSGAN mode-seeking so z is used (no collapse).
- Good channels [spline, derivative, morse] + coordinate warp; NOT inverse_derivative.

Purpose: not point accuracy (proven the adversary cannot beat MSE here) but a
calibrated UNCERTAINTY band from the sample spread.

## Results (N=8)

| metric | GAN in-family | GAN O2 | GP in-family | GP O2 |
|---|---:|---:|---:|---:|
| point RMSE (sample mean) | 1.42e-2 | 1.88e-2 | 8.97e-2 | 3.87e-2 |
| 90%-band coverage        | 0.99 | 0.97 | 0.71 | 0.63 |
| corr(sigma, abs error)   | 0.26 | 0.08 | -    | -    |
| mean sigma               | 3.23e-2 | 3.06e-2 | - | - |

## Verdict

- It TRAINS correctly (the old GAN's problem was coding/training, not concept):
  stable, z used, no collapse.
- Point estimate: worse than the deterministic CNN+warp (~6.3e-3 at N=8) and only
  better than GP. The adversary/stochasticity does not help accuracy, as proven.
- Uncertainty: it produces a band that is CONSERVATIVE (over-covers, 0.99 vs
  nominal 0.90), which is safer than the GP's OVER-confident band (0.71, under-covers).
- BUT the band is crude and, critically, OOD-BLIND: mean sigma on O2 (3.06e-2) is
  NOT larger than in-family (3.23e-2) even though O2 error is higher. A vanilla GAN
  cannot flag out-of-distribution inputs (its discriminator only ever saw the
  training families) -- the single most important thing uncertainty is needed for.
  The band-vs-error correlation is also weak (0.26 in, 0.08 OOD): roughly uniform
  inflation, not a targeted "unsure in the gaps/wall" signal.

## Conclusion

A correct conditional stochastic GAN is feasible and trains, but even correct it is
not compelling for this problem: its point estimate loses to the CNN, and its
uncertainty is uninformative and OOD-blind. For calibrated, OOD-aware uncertainty a
deep ensemble or a gappy-POD-residual model is the better route. The GAN remains the
right tool ONLY if the goal is generating diverse plausible in-distribution curves.

---

# Frozen PES-validity discriminator as a fixed generator penalty (negative result)

`scripts/prior_penalty.py`: train a strong "is this a valid PES?" classifier once
(spectral-normed, negatives = sparse-spline recons + shape distortions), FREEZE it,
then train the CNN+MLP+warp generator with  MSE + lambda * (-log D(fake)).  Not a GAN
(D never chases G). N=8.

| lambda | in-family full | wall | D-score | O2 full |
|-------:|---------------:|-----:|--------:|--------:|
| 0.0 | 4.79e-3 | 1.14e-2 | 0.51 | 2.75e-3 |
| 0.1 | 7.60e-2 | 9.23e-2 | 0.58 | 7.14e-2 |
| 0.3 | 1.67e-1 | 2.12e-1 | 0.65 | 1.63e-1 |
| 1.0 | 3.20e-1 | 4.36e-1 | 0.74 | 3.20e-1 |
| 3.0 | 4.94e-1 | 7.14e-1 | 0.80 | 5.09e-1 |

Two compounding failures:
1. D accuracy is only 64% -- a validity classifier CANNOT resolve the errors we
   care about, because sparse-spline recons and slightly-off curves are, after
   normalization, almost-valid PES shapes. The mistakes live ON the manifold
   boundary; a global classifier cannot see them.
2. Even that weak signal is a BAD gradient: as lambda grows the D-score rises
   (0.51->0.80) while RMSE explodes (16x-100x). The generator finds adversarial
   examples that D scores as "valid" but are farther from truth. A frozen
   classifier's gradient points toward its blind spots, not toward the manifold.
   This is exactly why a GAN co-trains D (to keep patching those holes); freezing
   D removes the safeguard.

## The correct reading

The PRIOR is right and powerful (gappy-POD proved it, orders of magnitude). The
DISCRIMINATOR is the wrong vehicle for it. A discriminative, scalar validity score
gives an unreliable adversarial gradient. The prior must be injected CONSTRUCTIVELY:
- gappy-POD / POD subspace: reconstruction is BUILT from valid shape modes -> always
  on-manifold, projection points the right way. (proven winner)
- a learned nonlinear manifold (autoencoder decoder) fit to the observed points.
- a DENOISER / projection prior (Regularization-by-Denoising / plug-and-play): a net
  trained to PROJECT any curve onto the PES manifold; its residual (x - denoise(x))
  IS a reliable "move toward valid PES" gradient -- the correct version of this
  penalty idea, using a denoiser instead of a discriminator.

Rule: for guiding reconstruction, a GENERATIVE/projection prior beats a
DISCRIMINATIVE (classifier) prior. Constrain to the manifold; do not penalize with a
classifier.
