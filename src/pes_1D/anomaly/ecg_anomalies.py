"""Synthetic corruptions of normal ECG beats (for the train-on-normal track).

This mirrors the PES project's idea (``data_generator.generate_bad_samples`` /
``utils.NoiseFunctions``): given only *normal* examples, synthesize anomalies by
deforming them, train a discriminator normal-vs-synthetic-anomaly, then test on *real*
anomalies. It keeps the setup label-free w.r.t. real anomalies, so it is comparable to
BeatGAN, while exercising the discriminator. The deformations are adapted to ECG beats
(z-normalized, fixed length) rather than radial PES curves.
"""

from __future__ import annotations

import numpy as np


def _spikes(beat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = beat.copy()
    k = rng.integers(1, 4)
    idx = rng.integers(0, len(beat), size=k)
    out[idx] += rng.uniform(3.0, 6.0, size=k) * rng.choice([-1.0, 1.0], size=k)
    return out


def _oscillation(beat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 1.0, len(beat))
    n = rng.integers(4, 20)
    amp = rng.uniform(0.3, 0.8)
    return beat + amp * np.sin(2.0 * np.pi * n * t + rng.uniform(0, 2 * np.pi))


def _local_warp(beat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = beat.copy()
    w = rng.integers(len(beat) // 8, len(beat) // 3)
    start = rng.integers(0, len(beat) - w)
    seg = out[start : start + w]
    out[start : start + w] = seg * rng.uniform(1.8, 3.0) + rng.uniform(-1.0, 1.0)
    return out


def _flatten(beat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = beat.copy()
    w = rng.integers(len(beat) // 6, len(beat) // 2)
    start = rng.integers(0, len(beat) - w)
    out[start : start + w] = rng.uniform(-0.2, 0.2)
    return out


def _noise_burst(beat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = beat.copy()
    w = rng.integers(len(beat) // 6, len(beat) // 2)
    start = rng.integers(0, len(beat) - w)
    out[start : start + w] += rng.normal(0.0, rng.uniform(0.5, 1.2), size=w)
    return out


DEFORMATIONS = (_spikes, _oscillation, _local_warp, _flatten, _noise_burst)


def corrupt_beat(beat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply one randomly chosen deformation to a single normal beat."""
    fn = DEFORMATIONS[rng.integers(0, len(DEFORMATIONS))]
    out = fn(np.asarray(beat, dtype=np.float64), rng)
    return out.astype(np.float32)


def corrupt_batch(X_normal: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return a synthetic-anomaly batch, one deformation per input normal beat."""
    rng = np.random.default_rng(seed)
    return np.stack([corrupt_beat(b, rng) for b in X_normal])
