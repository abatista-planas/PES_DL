"""Richer priors to lower the out-of-family (O2) floor, without more points.

Linear gappy-POD hits a bias floor on O2 (shape outside the 6-mode subspace).
Test three richer priors on the SAME sparse points:
  1. linear POD (K=6)                       -- baseline
  2. physics-augmented POD                  -- 6 PCA modes + physical functions
  3. gappy autoencoder (nonlinear manifold) -- the literature's "gappy POD-AE"

Report in-family and O2 RMSE at N = 8, 12, 16. Also report each manifold's
full-curve reconstruction floor (encode/decode of the TRUE curve) -- the best the
manifold could ever do, i.e. does it even CONTAIN O2's shape.
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
warnings.simplefilter("ignore")
from pes_1D.superres import lr_indices, sample_pes_curve

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
FAMS = ("lennard_jones", "morse", "buckingham_exp6", "extended_rydberg")
DENSE, GG = 1024, 128
x = np.linspace(0, 1, GG)
def tf(seed, fam):
    d = sample_pes_curve(np.random.default_rng(seed), fam, DENSE)
    return CubicSpline(np.linspace(0, 1, DENSE), d)(x).astype(np.float32)
def nrm(a):
    a = np.asarray(a, float); return 2*(a-a.min())/(a.max()-a.min()+1e-10)-1
TRAIN = np.stack([tf(1000+i, FAMS[i%4]) for i in range(500)])
TESTF = np.stack([tf(90000+i, FAMS[i%4]) for i in range(60)])
O2 = np.stack([tf(70000+i, "reudenberg_o2") for i in range(20)])
def rmse_set(R, T): return float(np.mean(np.sqrt(np.mean((R-T)**2, 1))))

# ---- 1) linear POD ----
mu = TRAIN.mean(0); U, S, Vt = np.linalg.svd(TRAIN-mu, full_matrices=False); PHI = Vt.T
def pod_recon(T, idx, K, basis=PHI, lam=1e-6):
    A = basis[idx, :K]; out = []
    for c in T:
        a = np.linalg.solve(A.T@A + lam*np.eye(K), A.T@(c[idx]-mu[idx]))
        out.append(mu + basis[:, :K]@a)
    return np.stack(out)

# ---- 2) physics-augmented basis: PCA modes + physical functions ----
phys = np.stack([nrm(np.exp(-b*x)) for b in (2, 4, 8)] + [nrm(x), nrm(x**2), nrm(x**3)]).T  # [128,6]
PHI_AUG = np.concatenate([PHI[:, :6], phys], axis=1).astype(np.float32)                     # [128,12]
def augpod_recon(T, idx, lam=1e-4):
    A = PHI_AUG[idx]; out = []
    for c in T:
        a = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@(c[idx]-mu[idx]))
        out.append(mu + PHI_AUG@a)
    return np.stack(out)

# ---- 3) gappy autoencoder (nonlinear manifold) ----
D = 8
class AE(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(GG, 256), nn.SiLU(), nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, D))
        s.dec = nn.Sequential(nn.Linear(D, 64), nn.SiLU(), nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, GG))
    def forward(s, z): return s.dec(z)
ae = AE().to(DEV)
Xt = torch.tensor(TRAIN, device=DEV)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)
gc = torch.Generator().manual_seed(0)
for ep in range(600):
    for sel in torch.randperm(len(Xt), generator=gc).split(64):
        sel = sel.to(DEV); z = ae.enc(Xt[sel]); rec = ae.dec(z)
        loss = F.mse_loss(rec, Xt[sel]); opt.zero_grad(); loss.backward(); opt.step()
    sch.step()
ae.eval()
for p in ae.parameters(): p.requires_grad_(False)

def ae_full_floor(T):                                   # encode/decode TRUE curve = manifold fidelity
    with torch.no_grad():
        return rmse_set(ae.dec(ae.enc(torch.tensor(T, device=DEV))).cpu().numpy(), T)

def gappy_ae(T, idx):                                   # optimize latent z to fit observed points
    Tt = torch.tensor(T, device=DEV); yobs = Tt[:, idx]
    z = ae.enc(Tt).clone().detach().requires_grad_(True)  # init from full-encode (good start)
    oz = torch.optim.Adam([z], lr=5e-2)
    for _ in range(400):
        rec = ae.dec(z); l = F.mse_loss(rec[:, idx], yobs) + 1e-4*(z**2).mean()
        oz.zero_grad(); l.backward(); oz.step()
    with torch.no_grad():
        return ae.dec(z).cpu().numpy()

print(f"AE manifold fidelity (full-curve encode/decode):  in-family {ae_full_floor(TESTF):.2e} | O2 {ae_full_floor(O2):.2e}")
print(f"\n{'N':>3} | {'method':22} | {'in-family':>10} | {'O2 (OOD)':>10}")
for N in [8, 12, 16]:
    idx = lr_indices(GG, N); K = min(N-2, 6)
    for name, Rf in [
        ("linear POD (K=6)", lambda T: pod_recon(T, idx, K)),
        ("physics-augmented POD", lambda T: augpod_recon(T, idx)),
        ("gappy autoencoder", lambda T: gappy_ae(T, idx)),
    ]:
        print(f"{N:>3} | {name:22} | {rmse_set(Rf(TESTF), TESTF):.3e} | {rmse_set(Rf(O2), O2):.3e}", flush=True)
    print()
