"""Improve the model on FIXED points: coordinate warp vs baseline, across N.

Constraint: we do NOT control per-curve placement. So the usable improvement is
the MLR/Hua-Guo coordinate warp applied to whatever points we are given.

Compares, for N = 4..16 (4 families, 80 test curves, continuous RMSE):
  spline, GP, CNN+MLP (baseline coord=x), CNN+MLP+warp (coord=m, x=m^model_p)
under two fixed sampling grids:
  - regular   : points uniform in x
  - optimized : the single wall-dense grid x_i=u_i^p* that minimizes model error
                (p* found by a search at N=8); same grid for every curve.

model_p = warp exponent of the model coordinate; grid_p = exponent of the point grid.
When grid_p == model_p the points are uniform in the model coordinate.
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

from pes_1D.superres import sample_pes_curve

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
FAMS = ("lennard_jones", "morse", "buckingham_exp6", "extended_rydberg")
DENSE, G = 1024, 384
xe = np.linspace(0, 1, 2000)
def truthfn(seed, fam):
    d = sample_pes_curve(np.random.default_rng(seed), fam, DENSE)
    return CubicSpline(np.linspace(0, 1, DENSE), d)
def nrm(a):
    a = np.asarray(a, float); return 2*(a-a.min())/(a.max()-a.min()+1e-10)-1
TRAIN = [truthfn(1000+i, FAMS[i%4]) for i in range(256)]
TEST = [truthfn(90000+i, FAMS[i%4]) for i in range(80)]

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

def anchors(N, grid_p):
    return np.linspace(0, 1, N)**grid_p            # points in physical x

def run_nn(N, grid_p, model_p, beta=4.0, epochs=250):
    xa = anchors(N, grid_p); mg = np.linspace(0, 1, G); morse = nrm(np.exp(-beta*mg))
    phi = lambda m: m**model_p; inv = lambda x: x**(1.0/model_p)
    ma = inv(xa)
    def cond(t):
        e = t(phi(ma)); S = CubicSpline(ma, e)
        return np.stack([S(mg), nrm(S(mg, 1)), morse]).astype(np.float32), S
    Xtr = torch.tensor(np.stack([cond(t)[0] for t in TRAIN]), dtype=torch.float32, device=DEV)
    spl = [cond(t)[1] for t in TRAIN]
    K = 256; mq = np.random.rand(len(TRAIN), K).astype(np.float32)
    yt = torch.tensor(np.stack([TRAIN[i](phi(mq[i])) for i in range(len(TRAIN))]).astype(np.float32), device=DEV)
    yb = torch.tensor(np.stack([spl[i](mq[i]) for i in range(len(TRAIN))]).astype(np.float32), device=DEV)
    mqt = torch.tensor(mq, device=DEV)
    enc, dec = Enc().to(DEV), Dec().to(DEV)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        loss = F.mse_loss(dec(enc(Xtr), mqt, yb), yt); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    enc.eval(); dec.eval(); me = inv(xe); sq = np.zeros(len(xe)); mn = []
    with torch.no_grad():
        for t in TEST:
            yv = t(xe); imin = int(np.argmin(yv)); c, S = cond(t)
            f = enc(torch.tensor(c[None], dtype=torch.float32, device=DEV))
            bb = torch.tensor(S(me)[None].astype(np.float32), device=DEV)
            yc = dec(f, torch.tensor(me[None], dtype=torch.float32, device=DEV), bb)[0].cpu().numpy()
            sq += (yc-yv)**2; mn.append(abs(yc[imin]-yv[imin]))
    return {"full": float(np.sqrt(np.mean(sq/len(TEST)))),
            "wall": float(np.sqrt(np.mean(sq[xe < .15]/len(TEST)))),
            "min": float(np.mean(mn))}

def run_classical(N, grid_p):
    xa = anchors(N, grid_p); out = {"spline": [], "GP": []}
    for t in TEST:
        ya = t(xa); yv = t(xe)
        out["spline"].append(np.sqrt(np.mean((CubicSpline(xa, ya)(xe)-yv)**2)))
        k = C(1.0, (1e-3, 1e3))*RBF(0.2, (1e-2, 1.0))
        gp = GaussianProcessRegressor(kernel=k, alpha=1e-10, normalize_y=False).fit(xa[:, None], ya)
        out["GP"].append(np.sqrt(np.mean((gp.predict(xe[:, None])-yv)**2)))
    return {m: float(np.mean(v)) for m, v in out.items()}

# --- find the optimized grid exponent p* at N=8 (points uniform in model coord) ---
print("=== optimized-grid search at N=8 (grid_p = model_p = p) ===", flush=True)
best_p, best = 1.0, 1e9
for p in (1.0, 1.5, 2.0, 2.5, 3.0):
    r = run_nn(8, p, p)["full"]
    print(f"  p={p:.1f}  full RMSE {r:.3e}", flush=True)
    if r < best: best, best_p = r, p
print(f"  -> optimized grid exponent p* = {best_p}\n", flush=True)

N_SWEEP = [4, 6, 8, 10, 12, 14, 16]
print("=== regular sampling (uniform-x, fixed): continuous RMSE ===", flush=True)
print(f"{'N':>3} | {'spline':>9} | {'GP':>9} | {'CNN base':>9} | {'CNN+warp':>9} | warp min")
for N in N_SWEEP:
    cl = run_classical(N, 1.0); b = run_nn(N, 1.0, 1.0); w = run_nn(N, 1.0, 2.0)
    print(f"{N:>3} | {cl['spline']:.3e} | {cl['GP']:.3e} | {b['full']:.3e} | {w['full']:.3e} | {w['min']:.2e}", flush=True)

print(f"\n=== optimized grid (x_i=u_i^{best_p}, same for all curves): continuous RMSE ===", flush=True)
print(f"{'N':>3} | {'spline':>9} | {'GP':>9} | {'CNN+warp':>9} | warp wall | warp min")
for N in N_SWEEP:
    cl = run_classical(N, best_p); w = run_nn(N, best_p, best_p)
    print(f"{N:>3} | {cl['spline']:.3e} | {cl['GP']:.3e} | {w['full']:.3e} | {w['wall']:.2e} | {w['min']:.2e}", flush=True)
