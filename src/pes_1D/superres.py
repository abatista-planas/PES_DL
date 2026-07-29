"""1D PES super-resolution: data generation, models, training, baselines.

The discriminator is conditional: it sees the low-resolution observation (as
spline/linear interpolation channels) next to the candidate high-resolution
curve, so "real vs fake" means "consistent with these observed points" rather
than just "PES-shaped".

Training uses an SRGAN-style objective, a dominant pointwise reconstruction
loss plus a small adversarial term, with the discriminator updated every step.
The generator interpolates first (cubic baseline) and refines with a dilated
residual CNN, avoiding ConvTranspose checkerboard artifacts and BatchNorm.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline  # type: ignore
from sklearn.exceptions import ConvergenceWarning  # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.gaussian_process.kernels import (  # type: ignore
    RBF,
    ConstantKernel,
    Matern,
    WhiteKernel,
)

# ---------------------------------------------------------------------------
# Analytic PES families (dimensionless: well depth = 1)
# ---------------------------------------------------------------------------


def lennard_jones(r: np.ndarray, sigma: float) -> np.ndarray:
    """LJ potential with epsilon = 1 (depth 1 at r = 2^(1/6) sigma)."""
    x = sigma / r
    return 4.0 * (x**12 - x**6)


def morse(r: np.ndarray, a: float, r0: float) -> np.ndarray:
    """Morse potential with D_e = 1, shifted so the minimum is -1."""
    return (1.0 - np.exp(-a * (r - r0))) ** 2 - 1.0


def buckingham_exp6(r: np.ndarray, alpha: float, rm: float) -> np.ndarray:
    """Reduced Buckingham exp-6 potential (depth 1 at rm).

    Only valid to the right of the spurious short-range turnover; the
    sampler clips the window there.
    """
    x = r / rm
    return (6.0 / (alpha - 6.0)) * np.exp(alpha * (1.0 - x)) - (
        alpha / (alpha - 6.0)
    ) * x**-6


def extended_rydberg(r: np.ndarray, a1: float, re: float, b2: float, b3: float) -> np.ndarray:
    """Extended Rydberg potential in reduced form (depth ~1 near re).

    V(u) = -(1 + u + b2 u^2 + b3 u^3) exp(-u) with u = a1 (r - re).
    """
    u = a1 * (r - re)
    return -(1.0 + u + b2 * u**2 + b3 * u**3) * np.exp(-u)


def reudenberg_o2(r: np.ndarray) -> np.ndarray:
    """Reudenberg O2 potential (out-of-family test case), long-range shifted."""
    coeff = [
        42030.046, -2388.564169, 18086.977116, -71760.197585, 154738.09175,
        -215074.85646, 214799.54567, -148395.4285, 73310.781453,
    ]
    c1, c2, c3 = 219.47463, 0.785, 1.307
    pes = np.ones_like(r, dtype=np.float64) * coeff[0]
    for i in range(len(coeff) - 1):
        pes = pes + coeff[i + 1] * np.exp(-(c3**i) * c2 * r**2) * c1
    zero = np.ones(1) * coeff[0]
    for i in range(len(coeff) - 1):
        zero = zero + coeff[i + 1] * np.exp(-(c3**i) * c2 * 100.0**2) * c1
    return pes - zero[0]


def _valid_single_well(v: np.ndarray) -> bool:
    """True if v is a repulsive wall + single well + monotonic tail."""
    i_min = int(np.argmin(v))
    if i_min == 0 or i_min == len(v) - 1:
        return False
    span = float(v.max() - v.min())
    if v[0] < 3.05 * abs(v[i_min]):  # wall must clear the largest wall_factor
        return False
    tol = 1e-9 * span
    return bool(
        np.all(np.diff(v[: i_min + 1]) <= tol)
        and np.all(np.diff(v[i_min:]) >= -tol)
    )


def _crossing(r: np.ndarray, v: np.ndarray, level: float) -> float:
    """First r where v crosses `level` (linear interp between grid points)."""
    sign = (v - level)[:-1] * (v - level)[1:]
    idx = np.nonzero(sign <= 0)[0]
    if len(idx) == 0:
        return float(r[-1])
    i = idx[0]
    f = (level - v[i]) / (v[i + 1] - v[i] + 1e-300)
    return float(r[i] + f * (r[i + 1] - r[i]))


def sample_pes_curve(
    rng: np.random.Generator,
    family: str,
    hr_size: int,
    wall_factor: float | None = None,
    tail_factor: float | None = None,
) -> np.ndarray:
    """One PES on `hr_size` uniform points, min-max normalized to [-1, 1].

    The r-window follows the original code's logic, made relative: the left
    edge is where V reaches wall_factor x well-depth above dissociation and
    the right edge is where |V| decays to tail_factor x well-depth.
    """
    if wall_factor is None:
        wall_factor = float(rng.uniform(1.0, 3.0))
    if tail_factor is None:
        tail_factor = float(rng.uniform(0.02, 0.40))

    if family == "lennard_jones":
        sigma = float(rng.uniform(1.2, 10.0))
        r_dense = np.linspace(0.70 * sigma, 5.0 * sigma, 4096)
        pes = lambda r: lennard_jones(r, sigma)  # noqa: E731
    elif family == "morse":
        a = float(rng.uniform(2.5, 10.0))
        r0 = float(rng.uniform(1.2, 10.0))
        r_dense = np.linspace(max(0.02, r0 - 3.0 / a), r0 + 8.0 / a, 4096)
        pes = lambda r: morse(r, a, r0)  # noqa: E731
    elif family == "buckingham_exp6":
        alpha = float(rng.uniform(10.5, 16.0))
        rm = float(rng.uniform(1.2, 10.0))
        pes = lambda r: buckingham_exp6(r, alpha, rm)  # noqa: E731
        # clip the window to the right of the spurious short-range maximum
        probe = np.linspace(0.15 * rm, 5.0 * rm, 4096)
        i_top = int(np.argmax(pes(probe)))
        r_dense = np.linspace(probe[i_top], 5.0 * rm, 4096)
    elif family == "extended_rydberg":
        for _ in range(100):
            a1 = float(rng.uniform(2.5, 6.0))
            re = float(rng.uniform(1.5, 8.0))
            b2 = float(rng.uniform(0.0, 0.5))
            b3 = float(rng.uniform(0.0, 0.2))
            pes = lambda r: extended_rydberg(r, a1, re, b2, b3)  # noqa: E731
            r_dense = np.linspace(max(0.02, re - 4.0 / a1), re + 10.0 / a1, 4096)
            if _valid_single_well(pes(r_dense)):
                break
        else:  # always-valid fallback: plain Rydberg (1 + u) exp(-u)
            b2 = b3 = 0.0
            pes = lambda r: extended_rydberg(r, a1, re, 0.0, 0.0)  # noqa: E731
            r_dense = np.linspace(max(0.02, re - 4.0 / a1), re + 10.0 / a1, 4096)
    elif family == "reudenberg_o2":
        r_dense = np.linspace(0.6, 6.0, 4096)
        pes = reudenberg_o2
    else:
        raise ValueError(f"unknown family {family}")

    v = pes(r_dense)
    i_min = int(np.argmin(v))
    depth = abs(float(v[i_min]))

    # keep the wall level below the highest available point on the left branch
    wall = min(wall_factor * depth, 0.9 * float(v[0]))
    r_lo = _crossing(r_dense[: i_min + 1], v[: i_min + 1], wall)
    r_hi = _crossing(r_dense[i_min:], -v[i_min:], tail_factor * depth)

    r_grid = np.linspace(r_lo, r_hi, hr_size)
    e = pes(r_grid)
    e_min, e_max = e.min(), e.max()
    return (2.0 * (e - e_min) / (e_max - e_min + 1e-12) - 1.0).astype(np.float32)


DEFAULT_FAMILIES = ("lennard_jones", "morse")


def sample_dataset(
    rng: np.random.Generator,
    n_per_family: int,
    hr_size: int,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized curves from the given families.

    Returns (curves [B, hr_size], family labels [B]), jointly shuffled.
    """
    curves = [
        sample_pes_curve(rng, fam, hr_size)
        for fam in families
        for _ in range(n_per_family)
    ]
    labels = np.array([fam for fam in families for _ in range(n_per_family)])
    out = np.stack(curves)
    perm = rng.permutation(len(out))
    return out[perm], labels[perm]


def lr_indices(hr_size: int, n_lr: int) -> np.ndarray:
    """Indices of the n_lr observed points (uniform, endpoints included)."""
    return np.round(np.linspace(0, hr_size - 1, n_lr)).astype(int)


def random_lr_indices(
    rng: np.random.Generator, hr_size: int, n_lr: int, n_curves: int
) -> np.ndarray:
    """Per-curve irregular placements [n_curves, n_lr]: endpoints kept,
    interior points drawn uniformly without replacement (no minimum
    spacing, so clusters and gaps occur, the realistic stress case)."""
    out = np.empty((n_curves, n_lr), dtype=int)
    for i in range(n_curves):
        interior = rng.choice(np.arange(1, hr_size - 1), size=n_lr - 2, replace=False)
        out[i] = np.sort(np.concatenate([[0, hr_size - 1], interior]))
    return out


def _idx_rows(idx: np.ndarray, n_curves: int) -> np.ndarray:
    """Broadcast a shared [n] index set to per-curve [n_curves, n] form."""
    idx = np.asarray(idx)
    if idx.ndim == 1:
        return np.broadcast_to(idx, (n_curves, idx.shape[0]))
    return idx


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------


def predict_linear(e_hr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    x = np.linspace(0.0, 1.0, e_hr.shape[-1])
    rows_idx = _idx_rows(idx, e_hr.shape[0])
    return np.stack(
        [np.interp(x, x[ix], row[ix]) for row, ix in zip(e_hr, rows_idx)]
    )


def predict_cubic_spline(e_hr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    x = np.linspace(0.0, 1.0, e_hr.shape[-1])
    rows_idx = _idx_rows(idx, e_hr.shape[0])
    return np.stack(
        [CubicSpline(x[ix], row[ix])(x) for row, ix in zip(e_hr, rows_idx)]
    )


def predict_gp(
    e_hr: np.ndarray, idx: np.ndarray, kernel_name: str = "rbf"
) -> np.ndarray:
    """GP regression per curve on the observed points (the reference method)."""
    x = np.linspace(0.0, 1.0, e_hr.shape[-1])
    if kernel_name == "rbf":
        base = RBF(length_scale=0.2, length_scale_bounds=(1e-3, 2.0))
    else:
        base = Matern(length_scale=0.2, length_scale_bounds=(1e-3, 2.0), nu=2.5)
    preds = []
    rows_idx = _idx_rows(idx, e_hr.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for row, ix in zip(e_hr, rows_idx):
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(
                1e-8, (1e-12, 1e-4)
            )
            gp = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0
            )
            gp.fit(x[ix][:, None], row[ix])
            preds.append(gp.predict(x[:, None]))
    return np.stack(preds)


# ---------------------------------------------------------------------------
# Conditioning tensors shared by the generator and the discriminator
# ---------------------------------------------------------------------------


def build_conditioning(e_hr: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    """[B, 3, H]: cubic-spline interp, linear interp, distance-to-known-point.

    `idx` may be a shared [n] index set or per-curve [B, n] placements
    (irregular sampling); the distance channel tells the network where
    the observation gaps are in either case.
    """
    spline = predict_cubic_spline(e_hr, idx)
    linear = predict_linear(e_hr, idx)
    x = np.linspace(0.0, 1.0, e_hr.shape[-1])
    rows_idx = _idx_rows(idx, e_hr.shape[0])
    dist = np.stack(
        [np.min(np.abs(x[:, None] - x[ix][None, :]), axis=1) for ix in rows_idx]
    )
    cond = np.stack([spline, linear, dist], axis=1).astype(np.float32)
    return torch.from_numpy(cond)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class _ResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        return x + self.conv2(h)


class RefineGenerator(nn.Module):
    """Interpolate-then-refine generator.

    Input: conditioning [B, 3, H] (spline, linear, distance channels).
    Output: [B, 1, H] = spline baseline + learned residual.  Working on the
    full high-res grid with dilated convolutions avoids the per-segment
    stamping artifacts of ConvTranspose1d(kernel=stride=upscale).
    """

    def __init__(self, channels: int = 48, dilations=(1, 2, 4, 8, 4, 2, 1)):
        super().__init__()
        self.head = nn.Conv1d(3, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[_ResBlock(channels, d) for d in dilations])
        self.tail = nn.Conv1d(channels, 1, kernel_size=3, padding=1)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.head(cond), 0.2)
        residual = self.tail(self.blocks(h))
        return cond[:, :1, :] + residual


class ConditionalDiscriminator(nn.Module):
    """Judges (candidate high-res curve, low-res observation) pairs.

    The candidate is concatenated with the conditioning channels, so the
    discriminator can penalize curves inconsistent with the observed points.
    Spectral norm keeps its Lipschitz constant bounded so its gradients stay
    informative.
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        sn = nn.utils.spectral_norm

        def block(c_in, c_out):
            return sn(nn.Conv1d(c_in, c_out, kernel_size=4, stride=2, padding=1))

        self.net = nn.ModuleList(
            [block(4, channels), block(channels, 2 * channels), block(2 * channels, 4 * channels)]
        )
        self.fc = sn(nn.Linear(4 * channels, 1))

    def forward(self, candidate: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = torch.cat([candidate, cond], dim=1)
        for layer in self.net:
            h = F.leaky_relu(layer(h), 0.2)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        return self.fc(h)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class GanLog:
    d_loss: list = field(default_factory=list)
    g_adv: list = field(default_factory=list)
    g_recon: list = field(default_factory=list)
    d_real_acc: list = field(default_factory=list)
    d_fake_acc: list = field(default_factory=list)


def _batches(n: int, batch_size: int, generator: torch.Generator):
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n - batch_size + 1, batch_size):
        yield perm[i : i + batch_size]


def train_supervised(
    gen: RefineGenerator,
    cond: torch.Tensor,
    target: torch.Tensor,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = False,
    device: str = "cpu",
) -> list[float]:
    """Plain MSE super-resolution training (the non-GAN neural baseline)."""
    gen = gen.to(device)
    cond = cond.to(device)
    target = target.to(device)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g_cpu = torch.Generator().manual_seed(seed)
    losses = []
    gen.train()
    for epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for sel in _batches(cond.shape[0], batch_size, g_cpu):
            sel = sel.to(device)
            fake = gen(cond[sel])
            loss = F.mse_loss(fake, target[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        sched.step()
        losses.append(epoch_loss / max(n_batches, 1))
        if verbose and epoch % 10 == 0:
            print(f"  [supervised] epoch {epoch:3d} mse {losses[-1]:.3e}")
    return losses


def train_gan(
    gen: RefineGenerator,
    disc: ConditionalDiscriminator,
    cond: torch.Tensor,
    target: torch.Tensor,
    epochs: int = 25,
    batch_size: int = 64,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    adv_weight: float = 1e-2,
    seed: int = 0,
    verbose: bool = False,
    device: str = "cpu",
) -> GanLog:
    """Adversarial fine-tuning: reconstruction-dominated, D updated every step.

    Both players update on every batch (no accuracy gating), the adversarial
    term is a small fraction of the generator loss, and the fake curve fed to
    the discriminator is the live generator output, so gradients flow.
    """
    gen = gen.to(device)
    disc = disc.to(device)
    cond = cond.to(device)
    target = target.to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    g_cpu = torch.Generator().manual_seed(seed)
    log = GanLog()

    gen.train()
    disc.train()
    for epoch in range(epochs):
        for sel in _batches(cond.shape[0], batch_size, g_cpu):
            sel = sel.to(device)
            c, real = cond[sel], target[sel]
            bs = c.shape[0]
            ones = torch.ones(bs, 1, device=device)
            zeros = torch.zeros(bs, 1, device=device)

            fake = gen(c)

            # --- discriminator step (every batch; 0.9 = label smoothing) ---
            d_real = disc(real, c)
            d_fake = disc(fake.detach(), c)
            loss_d = 0.5 * (bce(d_real, 0.9 * ones) + bce(d_fake, zeros))
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # --- generator step: reconstruction + small adversarial term ---
            adv = bce(disc(fake, c), ones)
            recon = F.mse_loss(fake, real)
            loss_g = recon + adv_weight * adv
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            log.d_loss.append(loss_d.item())
            log.g_adv.append(adv.item())
            log.g_recon.append(recon.item())
            log.d_real_acc.append((d_real > 0).float().mean().item())
            log.d_fake_acc.append((d_fake < 0).float().mean().item())
        if verbose and epoch % 5 == 0:
            k = max(len(log.d_loss) - 20, 0)
            print(
                f"  [gan] epoch {epoch:3d} recon {np.mean(log.g_recon[k:]):.3e} "
                f"adv {np.mean(log.g_adv[k:]):.3e} "
                f"D(real) {np.mean(log.d_real_acc[k:]):.2f} "
                f"D(fake) {np.mean(log.d_fake_acc[k:]):.2f}"
            )
    return log


@torch.no_grad()
def predict_nn(gen: RefineGenerator, cond: torch.Tensor) -> np.ndarray:
    gen.eval()
    device = next(gen.parameters()).device
    return gen(cond.to(device)).squeeze(1).cpu().numpy()


def rmse_per_curve(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - truth) ** 2, axis=-1))
