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
