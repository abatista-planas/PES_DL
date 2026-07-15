"""Outlier / OOD detection: discriminator score vs shape-manifold (POD) residual.

Unlike using D as a reconstruction penalty (which gets gamed), detection only READS
a score on a given input -- no optimization against D, so no adversarial gaming.
This is a legitimate use of the adversarial network.

We score each test curve three ways and measure how well each separates NORMAL
(in-family) from OUTLIERS (O2 out-of-family; and corrupted in-family curves), via
AUROC (1.0 = perfect detector, 0.5 = chance):
  (a) discriminator score  D(curve)          (low = anomalous)
  (b) POD residual, full   ||x - proj_K(x)|| (high = off-manifold)
  (c) POD residual, sparse  fit K modes to N observed points, residual there
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
from sklearn.metrics import roc_auc_score
warnings.simplefilter("ignore")
from pes_1D.superres import lr_indices, sample_pes_curve

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
FAMS = ("lennard_jones", "morse", "buckingham_exp6", "extended_rydberg")
DENSE, GG, N = 1024, 128, 8
x128 = np.linspace(0, 1, GG); idxN = lr_indices(GG, N)
def truthfn(seed, fam):
    d = sample_pes_curve(np.random.default_rng(seed), fam, DENSE)
    return CubicSpline(np.linspace(0, 1, DENSE), d)
def nrm(a):
    a = np.asarray(a, float); return 2*(a-a.min())/(a.max()-a.min()+1e-10)-1
TRAIN = np.stack([truthfn(1000+i, FAMS[i%4])(x128) for i in range(600)]).astype(np.float32)

# ---- test sets ----
normal = np.stack([truthfn(90000+i, FAMS[i%4])(x128) for i in range(60)]).astype(np.float32)   # in-family
o2 = np.stack([truthfn(70000+i, "reudenberg_o2")(x128) for i in range(30)]).astype(np.float32)  # out-of-family
rng = np.random.default_rng(3)
def corrupt(c):
    cc, w, a = rng.uniform(.15, .85), rng.uniform(.03, .10), rng.uniform(.2, .5)*rng.choice([-1, 1])
    return nrm(c + a*np.exp(-((x128-cc)/w)**2)).astype(np.float32)
corr = np.stack([corrupt(truthfn(80000+i, FAMS[i%4])(x128)) for i in range(30)]).astype(np.float32)  # anomalous shape

# ---- (b,c) POD shape-manifold residual ----
mu = TRAIN.mean(0); U, S, Vt = np.linalg.svd(TRAIN - mu, full_matrices=False); PHI = Vt.T
K = 6
def pod_full_resid(X):                                  # ||x - proj_K(x)||
    a = (X - mu) @ PHI[:, :K]; rec = mu + a @ PHI[:, :K].T
    return np.sqrt(np.mean((X - rec) ** 2, axis=1))
def pod_sparse_resid(X):                                # fit K modes to N observed pts, residual there
    A = PHI[idxN, :K]; out = []
    for x in X:
        a = np.linalg.solve(A.T @ A + 1e-6*np.eye(K), A.T @ (x[idxN] - mu[idxN]))
        rec = mu[idxN] + A @ a
        out.append(np.sqrt(np.mean((x[idxN] - rec) ** 2)))
    return np.array(out)

# ---- (a) discriminator: train "valid PES vs realistic fakes" ----
def neg_of(c):
    m = rng.integers(0, 3)
    if m == 0:
        Nn = int(rng.choice([4, 5, 6, 8, 10])); idx = lr_indices(GG, Nn)
        return nrm(CubicSpline(x128[idx], c[idx])(x128))
    if m == 1:
        cc, w, a = rng.uniform(.1, .9), rng.uniform(.03, .12), rng.uniform(.15, .5)*rng.choice([-1, 1])
        return nrm(c + a*np.exp(-((x128-cc)/w)**2))
    return nrm(c + rng.uniform(.05, .2)*np.sin(rng.uniform(1, 5)*np.pi*x128 + rng.uniform(0, 6)))
neg = np.stack([neg_of(c) for c in TRAIN]).astype(np.float32)
class Dnet(nn.Module):
    def __init__(s, c=64):
        super().__init__(); sn = nn.utils.spectral_norm
        s.net = nn.ModuleList([sn(nn.Conv1d(1, c, 4, 2, 1)), sn(nn.Conv1d(c, 2*c, 4, 2, 1)),
                               sn(nn.Conv1d(2*c, 4*c, 4, 2, 1)), sn(nn.Conv1d(4*c, 4*c, 4, 2, 1))])
        s.fc = nn.Sequential(sn(nn.Linear(4*c, 128)), nn.LeakyReLU(.2), sn(nn.Linear(128, 1)))
    def forward(s, x):
        h = x.unsqueeze(1)
        for l in s.net: h = F.leaky_relu(l(h), .2)
        return s.fc(F.adaptive_avg_pool1d(h, 1).squeeze(-1)).squeeze(-1)
D = Dnet().to(DEV)
Xd = torch.tensor(np.concatenate([TRAIN, neg]), device=DEV)
yd = torch.tensor(np.concatenate([np.ones(len(TRAIN)), np.zeros(len(neg))]).astype(np.float32), device=DEV)
opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9)); gc = torch.Generator().manual_seed(0)
for ep in range(300):
    for sel in torch.randperm(len(Xd), generator=gc).split(256):
        sel = sel.to(DEV); l = F.binary_cross_entropy_with_logits(D(Xd[sel]), yd[sel])
        opt.zero_grad(); l.backward(); opt.step()
D.eval()
def dscore(X):                                          # higher logit = more "valid"; anomaly = -logit
    with torch.no_grad():
        return D(torch.tensor(X, device=DEV)).cpu().numpy()

# ---- AUROC: normal (label 0) vs outliers (label 1) ----
def auroc(score_fn, sign, outliers, name):
    sn_, so = score_fn(normal), score_fn(outliers)
    y = np.r_[np.zeros(len(sn_)), np.ones(len(so))]
    return roc_auc_score(y, sign*np.r_[sn_, so])

print("AUROC (1.0 = perfect outlier detector, 0.5 = chance)\n")
print(f"{'detector':32} | {'O2 (out-of-family)':>18} | {'corrupted shape':>16}")
rows = [
    ("discriminator score D(x)", dscore, -1.0),                # low D = anomalous
    ("POD residual (full curve, K=6)", pod_full_resid, +1.0),  # high resid = anomalous
    ("POD residual (sparse N=8, K=6)", pod_sparse_resid, +1.0),
]
for name, fn, sign in rows:
    a1 = auroc(fn, sign, o2, name); a2 = auroc(fn, sign, corr, name)
    print(f"{name:32} | {a1:>18.3f} | {a2:>16.3f}")
