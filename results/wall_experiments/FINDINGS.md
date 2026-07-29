# Repulsive-wall reconstruction error

With uniform sampling about 78% of the squared error sits in `x < 0.15`, the
steepest and least-sampled part of the curve. `scripts/wall_error_experiments.py`
tests a steeper Morse descriptor (B), a coordinate warp `x = m^2` (C), the warp
with observations placed uniformly in `m` (D), and D with a sharper Morse (E).
Coordinate CNN+MLP, residual-on-spline, N=8, 4 families, 80 test curves.

Normalized RMSE; cm^-1 columns at an example H-bond scale (x2250).

| config | wall | wall cm^-1 | well | min | min cm^-1 | full |
|--------|-----:|-----:|-----:|----:|-----:|-----:|
| A baseline (uniform-x)      | 1.67e-2 | 37.7 | 4.18e-3 | 6.22e-3 | 14.0 | 7.31e-3 |
| B steeper morse (b=10)      | 1.68e-2 | 37.7 | 4.43e-3 | 5.90e-3 | 13.3 | 7.48e-3 |
| C warp-repr (obs uniform-x) | 1.44e-2 | 32.4 | 1.34e-3 | 2.00e-3 |  4.5 | 6.07e-3 |
| D warp + wall-dense sample  | 3.01e-3 |  6.8 | 1.21e-3 | 9.74e-4 |  2.2 | 1.57e-3 |
| E D + morse b=8             | 2.94e-3 |  6.6 | 9.76e-4 | 6.31e-4 |  1.4 | 1.64e-3 |

A sharper descriptor alone changes nothing; the limit is sampling, not descriptor
steepness. Use D/E when observation placement is controllable, C when the grid is
fixed.

---

# Warp across N under fixed grids

`scripts/warp_vs_sampling_sweep.py` sweeps N=4..16 and compares spline, GP,
CNN+MLP and CNN+MLP+warp under two fixed grids.

Regular (uniform-x, fixed):

| N  | spline | GP | CNN base | CNN+warp | warp vs base |
|---:|-------:|---:|---------:|---------:|-------------:|
|  4 | 2.29e-1 | 4.44e-1 | 5.56e-2 | 4.36e-2 | -22% |
|  6 | 6.55e-2 | 2.02e-1 | 2.29e-2 | 2.09e-2 |  -9% |
|  8 | 2.62e-2 | 9.29e-2 | 7.43e-3 | 6.29e-3 | -15% |
| 10 | 1.21e-2 | 3.92e-2 | 4.84e-3 | 2.61e-3 | -46% |
| 12 | 6.24e-3 | 1.48e-2 | 4.30e-3 | 2.03e-3 | -53% |
| 14 | 3.49e-3 | 4.96e-3 | 3.06e-3 | 1.68e-3 | -45% |
| 16 | 2.08e-3 | 1.39e-3 | 2.14e-3 | 1.56e-3 | -27% |

Optimized grid (x_i=u_i^2.0, fixed for all curves):

| N  | spline | GP | CNN+warp |
|---:|-------:|---:|---------:|
|  4 | 6.67e-1 | 6.62e-1 | 2.50e-2 |
|  6 | 1.26e-2 | 5.12e-1 | 6.96e-3 |
|  8 | 1.85e-3 | 2.47e-1 | 1.56e-3 |
| 10 | 5.38e-4 | 1.06e-1 | 5.30e-4 |
| 12 | 2.13e-4 | 4.74e-2 | 3.73e-4 |
| 14 | 1.01e-4 | 2.35e-2 | 5.34e-4 |
| 16 | 5.42e-5 | 1.12e-2 | 3.68e-4 |

The warp helps at every N on the same fixed points. On a wall-dense grid the
cubic spline overtakes the neural model at N>=12, and GP degrades badly on the
non-uniform grid. The neural model's niche is low N and regular grids.

---

# MLRNet-style physics decoder

`scripts/mlrnet_physics_decoder.py` predicts the parameters of a
generalized-Morse template `g(x)=exp(-b1(x-x0)) - lam*exp(-b2(x-x0))`, min-max
normalized to [-1,1], so the reconstruction is a ~4-parameter physics fit.
Regular sampling, N=4..16.

| N  | CNN+warp | MLRNet | MLR wall | MLR min |
|---:|---------:|-------:|---------:|--------:|
|  4 | 4.57e-2 | 5.92e-2 | 1.24e-1 | 1.03e-2 |
|  6 | 2.09e-2 | 6.07e-2 | 1.26e-1 | 1.07e-2 |
|  8 | 6.12e-3 | 4.98e-2 | 9.11e-2 | 4.85e-3 |
| 10 | 2.31e-3 | 4.73e-2 | 8.59e-2 | 4.14e-3 |
| 12 | 1.88e-3 | 4.01e-2 | 7.27e-2 | 2.62e-3 |
| 14 | 1.81e-3 | 4.61e-2 | 7.85e-2 | 3.66e-3 |
| 16 | 1.99e-3 | 3.52e-2 | 6.26e-2 | 2.25e-3 |

The template plateaus around 4e-2 and is worst on the wall (0.06-0.13): one
exponential cannot match the LJ, Buckingham and Rydberg walls at once. Only the
physics-model half of MLRNet is implemented here, without the NN correction.
