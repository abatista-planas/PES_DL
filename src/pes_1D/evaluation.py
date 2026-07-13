"""Physically-uniform error metrics and observable (bound-state) validation for 1D PES.

The problem this solves
-----------------------
A 1D interaction potential ``V(R)`` spans a huge dynamic range: a repulsive wall
at ~+1e4..1e5 cm^-1, an attractive well at ~-1e1..1e4 cm^-1, and an asymptote
that decays to 0. A plain energy RMSE is dominated by the wall and blind to the
asymptote, so "100 cm^-1 of error" is negligible on the wall but ruins the PES
near the asymptote. The tools here measure error by *physical significance*
instead:

  1. ``headline_metric``  -- a tolerance-normalized, scale-free error (the headline
                             number that matches the wall-vs-asymptote intuition).
  2. ``asinh_target``     -- the matching training-target transform + loss geometry.
  3. ``banded_rmse``      -- honest per-region absolute RMSE in cm^-1.
  4. ``sinc_dvr_levels`` / ``observable_error`` -- the gold standard: error in the
                             bound-state spectrum, validated against the analytic
                             Morse spectrum (``morse_spectrum``).
  5. ``descriptors`` / ``shape_checks`` -- the quantitative "does it look like a PES".

Units everywhere: R in Angstrom, energy in cm^-1, reduced mass mu in amu,
a in Angstrom^-1. Potentials are referenced so the asymptote is 0.
"""

from __future__ import annotations

import numpy as np

# hbar^2 / 2 expressed in cm^-1 * amu * Angstrom^2 (CODATA 2018; = 16.85762917).
# Single master constant for the DVR kinetic matrix, centrifugal term, B_e and
# omega_e-from-curvature -- no stray 2*pi*c anywhere.
K_CONST = 16.857629
V_INF = 0.0  # asymptote reference


def K(mu: float) -> float:
    """Kinetic prefactor hbar^2/(2 mu) in cm^-1 * Angstrom^2."""
    return K_CONST / mu


# --------------------------------------------------------------------------- #
# Analytic reference potentials (raw cm^-1, asymptote shifted to 0)
# --------------------------------------------------------------------------- #
def lennard_jones(R, sigma, epsilon):
    """V(R) = 4 eps [(sigma/R)^12 - (sigma/R)^6]; well depth = epsilon, V(inf)=0."""
    x = sigma / R
    return 4.0 * epsilon * (x**12 - x**6)


def morse(R, De, a, R0):
    """Morse with asymptote at 0 and well bottom at -De (V(R0) = -De)."""
    return De * (1.0 - np.exp(-a * (R - R0))) ** 2 - De


# --------------------------------------------------------------------------- #
# 1. Headline scale-aware metric
# --------------------------------------------------------------------------- #
def headline_metric(R, V_true, V_hat, tau0=1.0):
    """Tolerance-normalized relative error, RMS-aggregated.

    ``e(R) = |V_hat - V_true| / sqrt(V_true^2 + tau0^2)``.

    The soft denominator is C-infinity through the zero-crossing V=0 and uses the
    *true* V only, so a wrong prediction cannot inflate its own denominator. For
    ``|V| >> tau0`` this is a relative (scale-free) error; for ``|V| < tau0`` it is
    an absolute cm^-1 tolerance. Recommend ``tau0 = 1`` cm^-1 (0.1 for near
    spectroscopic targets).
    """
    R = np.asarray(R, float)
    V_true = np.asarray(V_true, float)
    V_hat = np.asarray(V_hat, float)
    e = np.abs(V_hat - V_true) / np.sqrt(V_true**2 + tau0**2)
    return dict(
        E_rms=float(np.sqrt(np.mean(e**2))),
        P95=float(np.percentile(e, 95)),
        E_max=float(e.max()),
        e=e,
    )


# --------------------------------------------------------------------------- #
# 2. Training-target transform (same scale s = tau0 as the headline floor)
# --------------------------------------------------------------------------- #
def asinh_target(V, s=1.0):
    """Signed, scale-free training target: linear for |V|<s, logarithmic for |V|>s."""
    return np.arcsinh(np.asarray(V, float) / s)


def asinh_invert(y, s=1.0):
    """Inverse of :func:`asinh_target`."""
    return s * np.sinh(np.asarray(y, float))


def logcosh(residual):
    """Robust loss on the (transformed) residual; stable large-|x| form."""
    x = np.abs(np.asarray(residual, float))
    return x + np.log1p(np.exp(-2.0 * x)) - np.log(2.0)


# --------------------------------------------------------------------------- #
# 3. Automatic region split + per-region absolute RMSE
# --------------------------------------------------------------------------- #
def region_masks(R, V):
    """Split the curve into wall / well / shoulder from the true curve itself."""
    R = np.asarray(R, float)
    V = np.asarray(V, float)
    i_min = int(np.argmin(V))
    Re, Vmin = R[i_min], V[i_min]
    De = V_INF - Vmin
    well = V <= Vmin + 0.25 * De          # deep well (bottom quartile by depth)
    wall = (V > 0) & (R < Re)             # repulsive branch
    shoulder = ~well & ~wall              # near-threshold band
    return dict(well=well, shoulder=shoulder, wall=wall, Re=Re, De=De, Vmin=Vmin)


def banded_rmse(R, V_true, V_hat, W=None):
    """Unweighted RMSE (cm^-1) per region + overall; optional weighted RMSE."""
    R = np.asarray(R, float)
    V_true = np.asarray(V_true, float)
    V_hat = np.asarray(V_hat, float)
    m = region_masks(R, V_true)
    out = {}
    for name in ("well", "shoulder", "wall"):
        mask = m[name]
        d = (V_hat - V_true)[mask]
        out[name] = dict(
            N=int(mask.sum()),
            RMSE=(float(np.sqrt(np.mean(d**2))) if mask.any() else None),
        )
    d = V_hat - V_true
    out["overall_RMSE"] = float(np.sqrt(np.mean(d**2)))
    if W is not None:
        W = np.asarray(W, float)
        out["RMSWE"] = float(np.sqrt(np.sum(W * d**2) / np.sum(W)))
    return out


# --------------------------------------------------------------------------- #
# 4. Observable gold standard: sinc-DVR + analytic Morse validation
# --------------------------------------------------------------------------- #
def sinc_dvr_levels(Vfun, mu, R_min, R_max, N, J=0):
    """Colbert-Miller sinc-DVR bound levels (E<0) on a uniform grid.

    Returns ``(E_bound, psi_bound, R)``. Place ``R_min`` deep in the wall
    (V >> energies of interest) and ``R_max`` a few Angstrom past R_e; both act as
    infinite walls (psi=0 at the edges). Converge in N and R_max.
    """
    k = K(mu)
    R = np.linspace(R_min, R_max, N)
    dR = R[1] - R[0]
    i = np.arange(N)
    diff = i[:, None] - i[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        off = 2.0 * ((-1.0) ** diff) / (diff**2)
    np.fill_diagonal(off, np.pi**2 / 3.0)
    T = k * off / dR**2
    Vdiag = Vfun(R) + k * J * (J + 1) / R**2
    H = T + np.diag(Vdiag)
    E, psi = np.linalg.eigh(H)
    bound = E < V_INF
    return E[bound], psi[:, bound], R


def morse_spectrum(De, a, mu):
    """Exact Morse bound spectrum (measured from the well bottom).

    ``omega_e = 2 a sqrt(K De)``, ``omega_e x_e = a^2 K`` (K = 16.857629/mu).
    ``E_v = omega_e (v+1/2) - omega_e x_e (v+1/2)^2`` for v = 0..v_max.
    """
    k = K(mu)
    we = 2.0 * a * np.sqrt(k * De)
    wexe = a**2 * k
    vmax = int(np.floor(we / (2.0 * wexe) - 0.5))
    v = np.arange(0, vmax + 1)
    Ev = we * (v + 0.5) - wexe * (v + 0.5) ** 2
    return Ev, we, wexe, vmax


def observable_error(Vtrue, Vrecon, mu, R_min, R_max, N, Jlist=(0,)):
    """RMS error over bound levels (cm^-1) between two potentials on one box.

    Levels are matched by (order-within-J); bound set is defined by the TRUE
    potential. Also reports the count mismatch and per-level residuals.
    """
    residuals = []
    n_true_total = 0
    n_recon_total = 0
    for J in Jlist:
        Et, _, _ = sinc_dvr_levels(Vtrue, mu, R_min, R_max, N, J=J)
        Er, _, _ = sinc_dvr_levels(Vrecon, mu, R_min, R_max, N, J=J)
        n_true_total += len(Et)
        n_recon_total += len(Er)
        n = min(len(Et), len(Er))
        residuals.extend((Er[:n] - Et[:n]).tolist())
    residuals = np.asarray(residuals, float)
    return dict(
        E_obs=float(np.sqrt(np.mean(residuals**2))) if residuals.size else None,
        dN_bound=int(n_recon_total - n_true_total),
        N_true=n_true_total,
        residuals=residuals,
    )


# --------------------------------------------------------------------------- #
# 5. Shape descriptors + hard pass/fail checks
# --------------------------------------------------------------------------- #
def _second_derivative_at_min(R, V):
    i = int(np.argmin(V))
    i = min(max(i, 2), len(R) - 3)
    # local quadratic fit on a 5-point stencil around the minimum
    sl = slice(i - 2, i + 3)
    c = np.polyfit(R[sl], V[sl], 2)
    return 2.0 * c[0]


def descriptors(R, V, mu):
    """Spectroscopic shape descriptors: R_e, D_e, omega_e, B_e."""
    R = np.asarray(R, float)
    V = np.asarray(V, float)
    i = int(np.argmin(V))
    Re = R[i]
    De = V_INF - V[i]
    Vpp = _second_derivative_at_min(R, V)
    we = float(np.sqrt(2.0 * K(mu) * Vpp)) if Vpp > 0 else float("nan")
    Be = K(mu) / Re**2
    return dict(Re=float(Re), De=float(De), we=we, Be=float(Be))


def shape_checks(R, V, slope_tol=0.0):
    """Hard pass/fail for the physical shape of a single-well 1D PES.

    A physically valid curve is strictly decreasing on the repulsive/inner branch
    (R < R_e), strictly increasing on the outer branch (R > R_e), and therefore has
    exactly one interior minimum. Spurious wiggles break monotonicity on one of the
    branches, so ``monotone_wall`` and ``monotone_tail`` together are a robust
    wiggle detector that does not false-positive on a genuinely steep smooth wall.
    ``slope_tol`` (>=0) allows a small tolerance against grid noise.
    """
    R = np.asarray(R, float)
    V = np.asarray(V, float)
    dV = np.gradient(V, R)
    i_min = int(np.argmin(V))
    interior_minima = int(np.sum((dV[:-1] < 0) & (dV[1:] >= 0)))
    single_well = interior_minima == 1
    monotone_wall = bool(np.all(dV[:i_min] <= slope_tol)) if i_min > 0 else True
    monotone_tail = (
        bool(np.all(dV[i_min + 1:] >= -slope_tol)) if i_min < len(R) - 1 else True
    )
    return dict(
        single_well=single_well,
        interior_minima=interior_minima,
        monotone_wall=monotone_wall,
        monotone_tail=monotone_tail,
        valid_shape=single_well and monotone_wall and monotone_tail,
    )
