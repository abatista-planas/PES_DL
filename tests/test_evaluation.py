import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

from pes_1D.evaluation import (  # type: ignore
    banded_rmse,
    descriptors,
    headline_metric,
    lennard_jones,
    morse,
    morse_spectrum,
    observable_error,
    region_masks,
    shape_checks,
    sinc_dvr_levels,
)


# --------------------------------------------------------------------------- #
# 1. Headline scale-aware metric
# --------------------------------------------------------------------------- #
def test_headline_wall_error_is_negligible():
    # 100 cm^-1 error on a 5e4 cm^-1 wall is fractionally tiny
    m = headline_metric([1.0], [5.0e4], [5.0e4 + 100.0], tau0=1.0)
    assert m["E_rms"] == pytest.approx(0.002, abs=1e-4)


def test_headline_asymptote_error_is_catastrophic():
    # the same 100 cm^-1 error near the asymptote (|V|=5) is enormous
    m = headline_metric([1.0], [5.0], [5.0 + 100.0], tau0=1.0)
    assert m["E_rms"] > 10.0


def test_headline_finite_and_smooth_at_zero_crossing():
    # at V=0 the soft denominator equals tau0, so e = |dV| / tau0 (no blow-up)
    m = headline_metric([1.0], [0.0], [100.0], tau0=1.0)
    assert m["E_rms"] == pytest.approx(100.0, abs=1e-9)


def test_headline_wall_vs_asymptote_separation():
    wall = headline_metric([1.0], [5.0e4], [5.0e4 + 100.0], tau0=1.0)["E_rms"]
    asy = headline_metric([1.0], [5.0], [5.0 + 100.0], tau0=1.0)["E_rms"]
    assert asy / wall > 1000.0


# --------------------------------------------------------------------------- #
# 2. sinc-DVR validated against the exact Morse spectrum
# --------------------------------------------------------------------------- #
def test_dvr_matches_analytic_morse_levels():
    De, a, mu, R0 = 2000.0, 1.2, 4.0, 3.0
    Ev, we, wexe, vmax = morse_spectrum(De, a, mu)
    E_dvr, _, _ = sinc_dvr_levels(
        lambda R: morse(R, De, a, R0), mu, R_min=1.0, R_max=15.0, N=1400, J=0
    )
    n = min(6, len(Ev), len(E_dvr))
    # DVR is referenced to the asymptote (0); analytic Ev is from the well bottom
    diff = np.abs((E_dvr[:n] + De) - Ev[:n])
    assert diff.max() < 1e-2


def test_dvr_bound_state_count_matches_morse():
    De, a, mu, R0 = 2000.0, 1.2, 4.0, 3.0
    _, _, _, vmax = morse_spectrum(De, a, mu)
    E_dvr, _, _ = sinc_dvr_levels(
        lambda R: morse(R, De, a, R0), mu, R_min=1.0, R_max=15.0, N=1400, J=0
    )
    assert len(E_dvr) == vmax + 1


def test_morse_spectrum_identities():
    # omega_e*x_e == a^2 K and == omega_e^2 / (4 De)
    De, a, mu = 2000.0, 1.2, 4.0
    _, we, wexe, _ = morse_spectrum(De, a, mu)
    assert wexe == pytest.approx(we**2 / (4.0 * De), rel=1e-10)


# --------------------------------------------------------------------------- #
# 3. Region split, banded RMSE, descriptors, shape checks on a true LJ curve
# --------------------------------------------------------------------------- #
@pytest.fixture
def lj_curve():
    sigma, epsilon = 3.4, 100.0
    R = np.linspace(3.0, 8.0, 800)
    return R, lennard_jones(R, sigma, epsilon), sigma, epsilon


def test_region_masks_cover_grid_without_overlap(lj_curve):
    R, V, _, _ = lj_curve
    m = region_masks(R, V)
    total = m["well"].astype(int) + m["shoulder"].astype(int) + m["wall"].astype(int)
    assert np.all(total == 1)  # every point in exactly one region
    assert m["De"] == pytest.approx(100.0, abs=0.1)


def test_banded_rmse_zero_for_perfect_reconstruction(lj_curve):
    R, V, _, _ = lj_curve
    b = banded_rmse(R, V, V.copy())
    assert b["overall_RMSE"] == pytest.approx(0.0, abs=1e-9)
    for name in ("well", "shoulder", "wall"):
        assert b[name]["RMSE"] == pytest.approx(0.0, abs=1e-9)


def test_descriptors_recover_lj_minimum(lj_curve):
    R, V, sigma, epsilon = lj_curve
    d = descriptors(R, V, mu=20.0)
    assert d["Re"] == pytest.approx(2.0 ** (1.0 / 6.0) * sigma, abs=1e-2)
    assert d["De"] == pytest.approx(epsilon, abs=0.1)
    assert d["we"] > 0.0


def test_shape_checks_pass_on_true_potential(lj_curve):
    R, V, _, _ = lj_curve
    s = shape_checks(R, V)
    assert s["valid_shape"]
    assert s["single_well"] and s["monotone_wall"] and s["monotone_tail"]


def test_shape_checks_flag_spurious_wiggle(lj_curve):
    R, V, _, _ = lj_curve
    V_bad = V + 5.0 * np.sin(40.0 * R)  # inject unphysical oscillation
    s = shape_checks(R, V_bad)
    assert not s["valid_shape"]


# --------------------------------------------------------------------------- #
# 4. Reconstruction: parametric (correct form) beats spline at few points
# --------------------------------------------------------------------------- #
def test_parametric_beats_spline_and_is_exact_at_few_points():
    sigma, epsilon = 3.4, 100.0
    Rmin_fit, Rcut = 3.0, 8.0
    R_dense = np.linspace(Rmin_fit, Rcut, 800)
    V_true = lennard_jones(R_dense, sigma, epsilon)

    R_s = np.linspace(Rmin_fit, Rcut, 6)
    V_s = lennard_jones(R_s, sigma, epsilon)
    V_spline = CubicSpline(R_s, V_s)(R_dense)
    popt, _ = curve_fit(
        lambda R, s, e: lennard_jones(R, s, e), R_s, V_s, p0=[sigma, epsilon]
    )
    V_param = lennard_jones(R_dense, *popt)

    h_spline = headline_metric(R_dense, V_true, V_spline, tau0=1.0)["E_rms"]
    h_param = headline_metric(R_dense, V_true, V_param, tau0=1.0)["E_rms"]
    assert h_param < h_spline
    assert h_param < 1e-2


def test_observable_error_zero_when_potentials_match():
    sigma, epsilon, mu = 3.4, 100.0, 20.0

    def true_fun(R):
        return lennard_jones(R, sigma, epsilon)

    obs = observable_error(true_fun, true_fun, mu, R_min=2.6, R_max=30.0, N=1000)
    assert obs["E_obs"] == pytest.approx(0.0, abs=1e-6)
    assert obs["dN_bound"] == 0
