"""Self-test for evaluation.py: validate the DVR against the analytic Morse
spectrum, reproduce the headline-metric wall-vs-asymptote cases, and demonstrate
the full metric suite on a few-point reconstruction (spline vs parametric fit).

Run: python -m pes_1D.validate_evaluation   (or python src/pes_1D/validate_evaluation.py)
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

from pes_1D.evaluation import (
    banded_rmse,
    descriptors,
    headline_metric,
    lennard_jones,
    morse,
    morse_spectrum,
    observable_error,
    shape_checks,
    sinc_dvr_levels,
)


def check(label, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def test_headline_two_cases():
    print("\n== 1. Headline metric: wall vs asymptote (both |dV|=100, tau0=1) ==")
    # wall: |V|=5e4 -> negligible ; asymptote: |V|=5 -> catastrophic
    wall = headline_metric([1.0], [5.0e4], [5.0e4 + 100.0], tau0=1.0)["E_rms"]
    asy = headline_metric([1.0], [5.0], [5.0 + 100.0], tau0=1.0)["E_rms"]
    zero = headline_metric([1.0], [0.0], [100.0], tau0=1.0)["E_rms"]
    print(f"  wall e={wall:.5f} (~0.2%) | asymptote e={asy:.4f} (~1960%) | "
          f"ratio={asy/wall:.0f} | zero-crossing e={zero:.3f}")
    ok = check("wall negligible (<0.01)", wall < 0.01)
    ok &= check("asymptote catastrophic (>10)", asy > 10)
    ok &= check("finite & smooth at V=0 (e==100)", abs(zero - 100.0) < 1e-9)
    return ok


def test_dvr_vs_morse():
    print("\n== 2. sinc-DVR vs analytic Morse spectrum ==")
    De, a, mu, R0 = 2000.0, 1.2, 4.0, 3.0
    Ev, we, wexe, vmax = morse_spectrum(De, a, mu)
    print(f"  omega_e={we:.4f}  omega_e*x_e={wexe:.5f}  v_max={vmax}")
    def Vfun(R):
        return morse(R, De, a, R0)

    E_dvr, _, _ = sinc_dvr_levels(Vfun, mu, R_min=1.0, R_max=15.0, N=3000, J=0)
    # DVR levels are referenced to the asymptote (0); analytic Ev is from the
    # well bottom, so compare (E_dvr + De) with Ev.
    n = min(len(Ev), len(E_dvr))
    diff = np.abs((E_dvr[:n] + De) - Ev[:n])
    for v in range(min(n, 6)):
        print(f"    v={v}: analytic={Ev[v]:10.4f}  DVR={E_dvr[v] + De:10.4f}  "
              f"|d|={diff[v]:.2e} cm^-1")
    ok = check(f"low-v levels match to <1e-2 cm^-1 (max |d|={diff[:min(n,6)].max():.1e})",
               diff[: min(n, 6)].max() < 1e-2)
    ok &= check(f"bound-state count matches analytic (DVR={len(E_dvr)}, exact={vmax + 1})",
                len(E_dvr) == vmax + 1)
    return ok


def test_reconstruction_demo():
    print("\n== 3. Few-point reconstruction: spline vs parametric (LJ) ==")
    # Ar2-like van der Waals well
    sigma, epsilon, mu = 3.4, 100.0, 20.0
    Rmin_fit, Rcut = 3.0, 8.0                    # interaction region actually fit
    R_dense = np.linspace(Rmin_fit, Rcut, 800)
    V_true = lennard_jones(R_dense, sigma, epsilon)

    def true_fun(R):
        return lennard_jones(R, sigma, epsilon)

    def grafted(recon_cs):
        # graft the analytic wall/tail outside the fitted window (as one does in
        # practice); inside the window use the reconstruction. Isolates the
        # interaction-region fit quality in the observable error.
        def fun(R):
            R = np.asarray(R, float)
            out = true_fun(R)
            inside = (R >= Rmin_fit) & (R <= Rcut)
            out = np.where(inside, recon_cs(np.clip(R, Rmin_fit, Rcut)), out)
            return out
        return fun

    def run(n_pts):
        R_s = np.linspace(Rmin_fit, Rcut, n_pts)
        V_s = lennard_jones(R_s, sigma, epsilon)
        V_spline = CubicSpline(R_s, V_s)                 # baseline A: cubic spline
        popt, _ = curve_fit(lambda R, s, e: lennard_jones(R, s, e), R_s, V_s,
                            p0=[sigma, epsilon])          # baseline B: parametric fit
        def V_param(R):
            return lennard_jones(R, *popt)

        return {"spline": V_spline, "parametric": V_param}

    print(f"  {'N':>3} | {'method':>10} | {'E_rms':>8} | {'well':>7} {'shoulder':>8} "
          f"{'wall':>9} | {'E_obs(cm^-1)':>12} {'dN':>3}")
    ok = True
    for n_pts in (6, 10, 16):
        recons = run(n_pts)
        for name, fun in recons.items():
            V_hat = fun(R_dense)
            h = headline_metric(R_dense, V_true, V_hat, tau0=1.0)
            b = banded_rmse(R_dense, V_true, V_hat)
            obs = observable_error(true_fun, grafted(fun), mu,
                                   R_min=2.6, R_max=30.0, N=2000)
            print(f"  {n_pts:>3} | {name:>10} | {h['E_rms']:8.4f} | "
                  f"{b['well']['RMSE']:7.3f} {b['shoulder']['RMSE']:8.3f} "
                  f"{b['wall']['RMSE']:9.2f} | {fmt(obs['E_obs']):>12} {obs['dN_bound']:>3}")
    # sanity: parametric should be near-exact everywhere; spline trails at low N
    r6 = run(6)
    h_sp = headline_metric(R_dense, V_true, r6["spline"](R_dense), tau0=1.0)["E_rms"]
    h_pa = headline_metric(R_dense, V_true, r6["parametric"](R_dense), tau0=1.0)["E_rms"]
    o_pa = observable_error(true_fun, grafted(r6["parametric"]), mu,
                            R_min=2.6, R_max=30.0, N=2000)["E_obs"]
    print(f"  descriptors(true): {short(descriptors(R_dense, V_true, mu))}")
    print(f"  shape_checks(true): {shape_checks(R_dense, V_true)}")
    ok &= check("parametric beats spline at N=6 (headline)", h_pa < h_sp)
    ok &= check("parametric near-exact at N=6 (E_rms<1e-2)", h_pa < 1e-2)
    ok &= check("parametric observable error ~0 with grafted wall/tail (<1e-2)",
                o_pa < 1e-2)
    return ok


def fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


def short(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


if __name__ == "__main__":
    results = [
        test_headline_two_cases(),
        test_dvr_vs_morse(),
        test_reconstruction_demo(),
    ]
    print("\n" + ("ALL TESTS PASSED" if all(results) else "SOME TESTS FAILED"))
