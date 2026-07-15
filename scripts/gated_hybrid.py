"""Residual-gated hybrid: gappy-POD  +  CNN+warp fallback, gated by the POD residual.

- gappy-POD gives machine-precision reconstruction in-family, and its sparse fit
  residual is a near-perfect OOD detector (large residual = off-manifold).
- the CNN+warp generalizes off-manifold better than POD at low N.
Gate: if the POD residual exceeds a threshold (learned on in-family data only),
the curve is flagged OOD and we fall back to the neural reconstruction.

Reports, at N = 8, 12, 16, in-family and O2:
  POD alone | neural alone | residual-gated hybrid | oracle (per-curve best)
plus how the gate routes curves.
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
DENSE, GG, G, P = 1024, 128, 384, 2.0
x128, mg, xe = np.linspace(0, 1, GG), np.linspace(0, 1, G), np.linspace(0, 1, 2000)
inv = lambda x: x**(1/P); phi = lambda m: m**P
def tf(seed, fam):
    d = sample_pes_curve(np.random.default_rng(seed), fam, DENSE)
    return CubicSpline(np.linspace(0, 1, DENSE), d)
def nrm(a):
    a = np.asarray(a, float); return 2*(a-a.min())/(a.max()-a.min()+1e-10)-1
TRAINf = [tf(1000+i, FAMS[i%4]) for i in range(400)]
TRAIN = np.stack([t(x128) for t in TRAINf]).astype(np.float32)
TESTf = [tf(90000+i, FAMS[i%4]) for i in range(60)]
O2f = [tf(70000+i, "reudenberg_o2") for i in range(30)]

# ---- POD basis + gappy fit (returns full-curve recon on xe, and sparse residual) ----
mu = TRAIN.mean(0); U, S, Vt = np.linalg.svd(TRAIN-mu, full_matrices=False); PHI = Vt.T
mu_e = CubicSpline(x128, mu)(xe); PHI_e = np.stack([CubicSpline(x128, PHI[:, k])(xe) for k in range(12)]).T
def pod(tfun, idx, K, lam=1e-6):
    c = tfun(x128); A = PHI[idx, :K]
    a = np.linalg.solve(A.T@A + lam*np.eye(K), A.T@(c[idx]-mu[idx]))
    recon_e = mu_e + PHI_e[:, :K]@a
    resid = float(np.sqrt(np.mean((c[idx] - (mu[idx] + PHI[idx, :K]@a))**2)))   # OOD signal
    return recon_e, resid

# ---- CNN+warp neural model ----
class Enc(nn.Module):
    def __init__(s, c=48):
        super().__init__(); s.h = nn.Conv1d(3, c, 3, padding=1)
        s.b = nn.ModuleList([nn.Conv1d(c, c, 3, padding=d, dilation=d) for d in (2, 4, 8)])
    def forward(s, x):
        h = F.leaky_relu(s.h(x), .2)
        for cv in s.b: h = F.leaky_relu(h+cv(h), .2)
        return h
class Dec(nn.Module):
    def __init__(s, c=48, fr=(1, 2, 4, 8)):
        super().__init__(); s.fr = torch.tensor([f*np.pi for f in fr], dtype=torch.float32)
        s.n = nn.Sequential(nn.Linear(c+2*len(fr)+1, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 1))
    def g(s, f, x):
        p = x*(f.shape[-1]-1); i0 = p.floor().clamp(0, f.shape[-1]-2).long(); w = (p-i0).unsqueeze(1)
        a = f.gather(2, i0.unsqueeze(1).expand(-1, f.shape[1], -1)); b = f.gather(2, (i0+1).unsqueeze(1).expand(-1, f.shape[1], -1))
        return ((1-w)*a+w*b).transpose(1, 2)
    def forward(s, f, x, base):
        fx = s.g(f, x); ang = x.unsqueeze(-1)*s.fr.to(x.device)
        return base + s.n(torch.cat([fx, torch.sin(ang), torch.cos(ang), base.unsqueeze(-1)], -1)).squeeze(-1)
def cond_of(tfun, xa, ma):
    e = tfun(phi(ma)); Sp = CubicSpline(ma, e)
    return np.stack([Sp(mg), nrm(Sp(mg, 1)), nrm(np.exp(-4*mg))]).astype(np.float32), Sp
def train_neural(N):
    xa = np.linspace(0, 1, N); ma = inv(xa)
    condT = torch.tensor(np.stack([cond_of(t, xa, ma)[0] for t in TRAINf]), dtype=torch.float32, device=DEV)
    spl = [cond_of(t, xa, ma)[1] for t in TRAINf]
    K = 256; mq = np.random.rand(len(TRAINf), K).astype(np.float32)
    yt = torch.tensor(np.stack([TRAINf[i](phi(mq[i])) for i in range(len(TRAINf))]).astype(np.float32), device=DEV)
    yb = torch.tensor(np.stack([spl[i](mq[i]) for i in range(len(TRAINf))]).astype(np.float32), device=DEV)
    mqt = torch.tensor(mq, device=DEV); enc, dec = Enc().to(DEV), Dec().to(DEV)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=250)
    for _ in range(250):
        loss = F.mse_loss(dec(enc(condT), mqt, yb), yt); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    enc.eval(); dec.eval()
    def predict(tfun):
        xa2 = np.linspace(0, 1, N); ma2 = inv(xa2); c, Sp = cond_of(tfun, xa2, ma2); me = inv(xe)
        with torch.no_grad():
            f = enc(torch.tensor(c[None], dtype=torch.float32, device=DEV))
            bb = torch.tensor(Sp(me)[None].astype(np.float32), device=DEV)
            return dec(f, torch.tensor(me[None], dtype=torch.float32, device=DEV), bb)[0].cpu().numpy()
    return predict, xa, ma

def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))
print(f"{'N':>3} | {'set':10} | {'POD':>9} | {'neural':>9} | {'HYBRID':>9} | {'oracle':>9} | routing")
for N in [8, 12, 16]:
    K = min(N-2, 6); idx = lr_indices(GG, N)
    predict, xa, ma = train_neural(N)
    # threshold from in-family TRAIN residuals only (no test peeking): 90th percentile
    train_resid = np.array([pod(t, idx, K)[1] for t in TRAINf])
    tau = np.percentile(train_resid, 98)
    for setname, curves in [("in-family", TESTf), ("O2", O2f)]:
        ep, en, eh, eo, routed = [], [], [], [], 0
        for t in curves:
            yv = t(xe); rp, r = pod(t, idx, K); rn = predict(t)
            p_err, n_err = rmse(rp, yv), rmse(rn, yv)
            use_neural = r > tau
            routed += int(use_neural)
            eh.append(n_err if use_neural else p_err)
            ep.append(p_err); en.append(n_err); eo.append(min(p_err, n_err))
        frac = routed/len(curves)*100
        print(f"{N:>3} | {setname:10} | {np.mean(ep):.3e} | {np.mean(en):.3e} | {np.mean(eh):.3e} | {np.mean(eo):.3e} | {frac:.0f}% -> neural", flush=True)
    print()
