"""Exploit the PES shape manifold: gappy-POD (PCA subspace) reconstruction.

All PES curves share a shape (single well, monotone wall, smooth tail), so they
live on a low-dimensional subspace. Learn it from training curves (mean + top-K
PCA modes), then reconstruct any curve as the subspace element that best fits the
N observed points (ridge least squares). The subspace forbids unphysical wiggles
and needs very few points.

Compared on REGULAR (uniform) sampling, N=4..16, vs spline / GP / (CNN+warp ref).
"""
import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

from pes_1D.superres import lr_indices, sample_pes_curve

FAMS = ("lennard_jones", "morse", "buckingham_exp6", "extended_rydberg")
DENSE, GG = 1024, 128
np.random.seed(0)
x128, xe = np.linspace(0, 1, GG), np.linspace(0, 1, 2000)
def truthfn(seed, fam):
    d = sample_pes_curve(np.random.default_rng(seed), fam, DENSE)
    return CubicSpline(np.linspace(0, 1, DENSE), d)

TRAIN = np.stack([truthfn(1000+i, FAMS[i%4])(x128) for i in range(400)])   # [n,128]
TEST_F = [truthfn(90000+i, FAMS[i%4]) for i in range(80)]

# --- learn the shape subspace ---
mu = TRAIN.mean(0)
U, S, Vt = np.linalg.svd(TRAIN - mu, full_matrices=False)
PHI = Vt.T                                   # [128, n_modes], columns = shape modes
var = (S**2) / np.sum(S**2)
print("variance captured by first modes:",
      " ".join(f"K{k}:{np.sum(var[:k])*100:.2f}%" for k in (2, 4, 6, 8, 12)))

def gappy(test_curve, idx, K, lam=1e-6):
    A = PHI[idx, :K]; b = test_curve[idx] - mu[idx]
    a = np.linalg.solve(A.T @ A + lam*np.eye(K), A.T @ b)
    return mu + PHI[:, :K] @ a                # [128]

def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))

def classical(N):
    xa = x128[lr_indices(GG, N)]; sp, gp = [], []
    for t in TEST_F:
        ya = t(xa); yv = t(xe); sp.append(rmse(CubicSpline(xa, ya)(xe), yv))
        k = C(1.0, (1e-3, 1e3))*RBF(0.2, (1e-2, 1.0))
        g = GaussianProcessRegressor(kernel=k, alpha=1e-10, normalize_y=False).fit(xa[:, None], ya)
        gp.append(rmse(g.predict(xe[:, None]), yv))
    return np.mean(sp), np.mean(gp)

KS = [2, 3, 4, 5, 6, 8, 10, 12]
CNN_WARP = {4: 4.57e-2, 6: 2.09e-2, 8: 6.12e-3, 10: 2.31e-3, 12: 1.88e-3, 14: 1.81e-3, 16: 1.99e-3}
# out-of-family test set: O2 curves (varied window), NOT in the training families
O2 = [truthfn(70000+i, "reudenberg_o2") for i in range(20)]

print(f"\n{'N':>3} | {'spline':>9} | {'GP':>9} | {'CNN+warp':>9} | in-family best | O2 out-of-family (K=min(N-2,6))")
for N in [4, 6, 8, 10, 12, 14, 16]:
    idx = lr_indices(GG, N)
    perK = {}
    for K in KS:
        if K > N: continue
        perK[K] = np.mean([rmse(CubicSpline(x128, gappy(t(x128), idx, K))(xe), t(xe)) for t in TEST_F])
    bestK = min(perK, key=perK.get)
    Kcv = max(2, min(N - 2, 6))                      # a plain non-oracle rule
    o2e = np.mean([rmse(CubicSpline(x128, gappy(t(x128), idx, Kcv))(xe), t(xe)) for t in O2])
    sp, gp = classical(N)
    print(f"{N:>3} | {sp:.3e} | {gp:.3e} | {CNN_WARP[N]:.3e} | {perK[bestK]:.2e} (K={bestK}) | "
          f"O2 {o2e:.2e} (K={Kcv})", flush=True)
