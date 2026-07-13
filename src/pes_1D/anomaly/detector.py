"""Thin wrapper to train the reused CnnDiscriminator and score beats.

The discriminator is reused unchanged from ``pes_1D.discriminator``. Its ``test_model``
only evaluates the first batch, so ``evaluate_scores`` here runs the whole test set and
returns per-beat anomaly scores = sigmoid(logit), suitable for AUROC/AUPRC.

Beats arrive as raw ``(N, L)`` arrays. With ``channels=3`` the detector augments each
beat with its first and second derivatives (morphology cues: QRS slopes/curvature),
each z-normalized, giving a ``(N, 3, L)`` input; ``channels=1`` uses the raw beat.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pes_1D.discriminator import CnnDiscriminator  # type: ignore

DEFAULT_HIDDEN = [32, 64]


def build_discriminator(
    grid_size: int = 320, in_channels: int = 3, hidden_channels=None
) -> CnnDiscriminator:
    params = {
        "in_channels": in_channels,
        "grid_size": grid_size,
        "hidden_channels": list(hidden_channels or DEFAULT_HIDDEN),
        "kernel_size": [3, 3],
        "pool_size": [2, 2],
    }
    return CnnDiscriminator(params)


def _znorm_rows(a: np.ndarray) -> np.ndarray:
    m = a.mean(axis=-1, keepdims=True)
    s = a.std(axis=-1, keepdims=True)
    return (a - m) / (s + 1e-8)


def make_channels(X: np.ndarray, channels: int) -> np.ndarray:
    """Turn raw beats (N, L) into (N, channels, L). channels in {1, 3}."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:  # already channelled
        return X
    if channels == 1:
        return X[:, None, :]
    d1 = np.gradient(X, axis=1)
    d2 = np.gradient(d1, axis=1)
    return np.stack([X, _znorm_rows(d1), _znorm_rows(d2)], axis=1).astype(np.float32)


def _to_tensors(X: np.ndarray, y: np.ndarray, device: str):
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    if Xt.dim() == 2:
        Xt = Xt.unsqueeze(1)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device).view(-1, 1)
    return Xt, yt


def train_discriminator(
    X: np.ndarray,
    y: np.ndarray,
    grid_size: int = 320,
    channels: int = 3,
    hidden_channels=None,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str | None = None,
    seed: int = 0,
) -> CnnDiscriminator:
    """Train a discriminator on labeled beats (y in {0,1}, 1 = anomalous)."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_discriminator(grid_size, channels, hidden_channels).to(device)
    Xt, yt = _to_tensors(make_channels(X, channels), y, device)
    loader = DataLoader(
        TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True, drop_last=False
    )
    # weight the positive (anomaly) class to counter imbalance (~1.0 when balanced)
    n_pos = float((yt > 0.5).sum().item())
    n_neg = float((yt <= 0.5).sum().item())
    pos_weight = torch.tensor(
        [n_neg / n_pos if n_pos > 0 else 1.0], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train_model(loader, criterion, optimizer, epochs)
    return model


@torch.no_grad()
def evaluate_scores(
    model: CnnDiscriminator,
    X: np.ndarray,
    channels: int = 3,
    device: str | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """Per-beat anomaly scores sigmoid(logit) over the full set (higher = anomalous)."""
    device = device or next(model.parameters()).device.type
    model.eval()
    Xc = make_channels(X, channels)
    Xt, _ = _to_tensors(Xc, np.zeros(len(Xc)), device)
    scores = []
    for i in range(0, len(Xt), batch_size):
        logits = model.forward(Xt[i : i + batch_size])
        scores.append(torch.sigmoid(logits).view(-1).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty((0,), np.float32)
