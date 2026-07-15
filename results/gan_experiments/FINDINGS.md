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

---

# Outlier / OOD detection: discriminator vs shape-manifold residual (a legitimate use)

`scripts/outlier_detection.py`. Detection only READS a score on a given input (no
optimization against D), so the adversarial-gaming failure does not apply -- this is
a legitimate use of the discriminator. AUROC separating NORMAL (in-family) from
OUTLIERS (O2 out-of-family; corrupted-shape in-family):

| detector | O2 | corrupted |
|---|---:|---:|
| discriminator score D(x)        | 0.679 | 0.728 |
| POD residual (full curve, K=6)  | 0.993 | 1.000 |
| POD residual (sparse N=8, K=6)  | 0.923 | 0.956 |

Findings:
- YES, the discriminator works for outlier detection (0.68-0.73 AUROC, clearly above
  chance, and un-gameable since we only read the score). First genuinely valid use of
  the discriminator in this project.
- BUT the shape-manifold (POD) residual dominates: near-perfect on full curves
  (0.99/1.00) and excellent even from just 8 observed points (0.92/0.96).
- Reason: an outlier here IS a curve far from the PES shape manifold. The POD residual
  measures exactly that distance; the discriminator is only an indirect, noisy proxy
  for it (same fuzzy-boundary weakness that capped it at ~70% as a classifier).

Unifying insight: everything reduces to the shape manifold.
- Reconstruction  = PROJECT onto it (gappy-POD)  -> orders-of-magnitude accuracy.
- Outlier/OOD     = DISTANCE from it (POD residual) -> near-perfect AUROC, even sparse.
The discriminator is a weak proxy for the manifold in both tasks.

Practical payoff: the sparse POD residual (0.92 AUROC from N=8) flags off-distribution
/ unreliable inputs at reconstruction time -- the exact OOD-awareness the GAN's
uncertainty lacked. The shape manifold gives us reconstruction AND trustworthy OOD
detection; recommend it as the outlier detector. The discriminator works but is
dominated.

---

# Richer priors do NOT lower the O2 floor (negative result)

`scripts/richer_prior.py`. Tested whether a richer prior lowers the out-of-family
(O2) floor of linear gappy-POD: physics-augmented POD (PCA modes + Morse/power
functions) and a gappy AUTOENCODER (nonlinear manifold, the literature's gappy-POD-AE).

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

AE full-curve fidelity (encode/decode of the TRUE curve): in-family 2.2e-3, O2 4.5e-3.

Findings:
- Richer manifold priors HURT. Both are worse in-family and neither lowers the O2
  floor. The nonlinear AE is worst everywhere.
- Reason: the normalized, aligned PES curves genuinely live on a LINEAR ~6-dim
  subspace (6 modes = 100% variance). Linear PCA is therefore EXACT in-family; a
  nonlinear AE can only approximate it (lossy, ~2e-3 floor). And the AE's curved
  manifold is FARTHER from O2 (full-recon 4.5e-3) than the flat linear subspace
  (gappy O2 ~1.1e-3). Linear POD is already near-optimal.
- So the O2 floor is NOT a prior-richness problem. It is genuine OUT-OF-DISTRIBUTION:
  O2's shape is outside the span of ANY model built from the 4 families (linear or
  nonlinear). More manifold expressiveness cannot help.

The real levers for the O2 floor (not tested here, recommended):
- COVERAGE: add training FAMILIES that span O2-like shapes (more data diversity, not
  more observed points). Widens the subspace where it matters.
- NEURAL OOD-fallback: the CNN+warp generalizes out-of-family better than POD at low
  N (O2 ~2.6e-3 vs POD 1.2e-2 at N=8). Gate it with the POD residual (the OOD
  detector): use POD when the residual is small (in-family, machine precision), fall
  back to the net when the residual is large (O2). This is the residual-gated hybrid.

---

# Residual-gated hybrid: gappy-POD + CNN+warp fallback (the working endpoint)

`scripts/gated_hybrid.py`. Reconstruct with gappy-POD; use its sparse fit residual
as an OOD flag; if the residual exceeds a threshold (98th percentile of in-family
TRAIN residuals -- calibrated WITHOUT touching the test set) fall back to the
CNN+warp neural reconstruction. Fully automatic.

| N | set | POD | neural | HYBRID | oracle | routed->neural |
|--:|-----|----:|-------:|-------:|-------:|---------------:|
| 8 | in-family | 1.58e-3 | 4.33e-3 | 1.81e-3 | 1.52e-3 |  5% |
| 8 | O2        | 1.34e-2 | 3.53e-3 | 4.65e-3 | 3.43e-3 | 70% |
|12 | in-family | 3.54e-4 | 1.73e-3 | 4.06e-4 | 3.54e-4 |  3% |
|12 | O2        | 2.20e-3 | 1.84e-3 | 1.84e-3 | 1.44e-3 |100% |
|16 | in-family | 1.76e-4 | 1.26e-3 | 2.15e-4 | 1.76e-4 |  3% |
|16 | O2        | 1.19e-3 | 1.29e-3 | 1.26e-3 | 8.76e-4 | 97% |

What it delivers, automatically, from the same sparse points:
- In-family: near machine precision (POD), with only a 3-5% false-flag tax.
- OOD detection: the residual flags O2 (70-100% routed to the neural fallback).
- OOD reconstruction: 2.9x better than POD-alone at low N (O2 N=8: 1.34e-2 -> 4.65e-3)
  by falling back to the net, which generalizes off-manifold. Approaches the oracle.
- The gate threshold is a tunable knob (90th pct -> better O2 but higher in-family
  tax; 98th -> near-zero in-family tax, slightly more O2 missed).

Honest limits: the oracle (per-curve best) is still a bit lower -- imperfect gating
misroutes a few curves; at high N POD is marginally better than the net even on O2,
so routing there costs a hair. But the hybrid captures the main win: it removes
POD's catastrophic low-N OOD failure while preserving in-family precision.

# Overall conclusion of the GAN/prior thread
- The adversary is inert for reconstruction (near-delta distribution); a correct GAN
  trains but doesn't beat MSE and its uncertainty is OOD-blind.
- A discriminator penalty is worse (adversarial-gradient gaming), a better D does not
  fix it, and richer manifold priors do not lower the OOD floor (curves are a genuine
  linear ~6-dim subspace, so linear POD is already near-optimal).
- Everything reduces to the shape manifold: PROJECT onto it (gappy-POD) for
  reconstruction, DISTANCE from it (POD residual) for OOD detection, and a neural
  fallback for off-manifold curves. That residual-gated hybrid -- no GAN -- is the
  recommended system.
