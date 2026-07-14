"""MLRNet-style physics decoder vs baseline / warp / spline / GP.

Instead of predicting free per-point values, the encoder digests the observed
points and predicts the parameters of a physics template (generalized Morse):

    g(x) = exp(-b1 (x-x0)) - lam * exp(-b2 (x-x0)),   b1 > b2 > 0

which is then min-max normalized to [-1,1] to match the target convention. So the
reconstruction is a ~4-parameter physics fit rather than a free-form curve -- the
idea being that this is better-posed at very low N.

Compared on REGULAR (fixed uniform) sampling, N=4..16, 4 families, 80 test curves.
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

class MLRNet(nn.Module):
    def __init__(s, c=48):
        super().__init__(); s.enc = Enc(c)
        s.head = nn.Sequential(nn.Linear(c, 64), nn.SiLU(), nn.Linear(64, 4))
        s.register_buffer("xr", torch.linspace(0, 1, 256))
    def par(s, cond):
        z = s.enc(cond).mean(-1); r = s.head(z)
        x0 = 0.05 + 0.55 * torch.sigmoid(r[:, 0:1])
        b2 = 1.0 + 9 * torch.sigmoid(r[:, 1:2])
        b1 = b2 + 0.5 + 9 * torch.sigmoid(r[:, 2:3])
        lam = 0.2 + 3 * torch.sigmoid(r[:, 3:4])
        return x0, b1, b2, lam
    def tmpl(s, x, p):
        x0, b1, b2, lam = p; u = x - x0
        return torch.exp(torch.clamp(-b1 * u, max=30.)) - lam * torch.exp(torch.clamp(-b2 * u, max=30.))
    def forward(s, cond, x):
        p = s.par(cond)
        gr = s.tmpl(s.xr.expand(cond.shape[0], -1), p)
        gmn = gr.min(1, keepdim=True).values; gmx = gr.max(1, keepdim=True).values
        return 2 * (s.tmpl(x, p) - gmn) / (gmx - gmn + 1e-6) - 1

def cond_x(t, xa, mg, beta=4.0):
    S = CubicSpline(xa, t(xa))
    return np.stack([S(mg), nrm(S(mg, 1)), nrm(np.exp(-beta * mg))]).astype(np.float32), S

def run_mlrnet(N, epochs=300):
    xa = np.linspace(0, 1, N); mg = np.linspace(0, 1, G)
    Xtr = torch.tensor(np.stack([cond_x(t, xa, mg)[0] for t in TRAIN]), dtype=torch.float32, device=DEV)
    K = 256; xq = np.random.rand(len(TRAIN), K).astype(np.float32)
    yt = torch.tensor(np.stack([TRAIN[i](xq[i]) for i in range(len(TRAIN))]).astype(np.float32), device=DEV)
    xqt = torch.tensor(xq, device=DEV)
    m = MLRNet().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        loss = F.mse_loss(m(Xtr, xqt), yt); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval(); sq = np.zeros(len(xe)); mn = []
    xet = torch.tensor(xe[None], dtype=torch.float32, device=DEV)
    with torch.no_grad():
        for t in TEST:
            c = torch.tensor(cond_x(t, xa, mg)[0][None], dtype=torch.float32, device=DEV)
            yc = m(c, xet)[0].cpu().numpy(); yv = t(xe); imin = int(np.argmin(yv))
            sq += (yc - yv) ** 2; mn.append(abs(yc[imin] - yv[imin]))
    return {"full": float(np.sqrt(np.mean(sq/len(TEST)))),
            "wall": float(np.sqrt(np.mean(sq[xe < .15]/len(TEST)))), "min": float(np.mean(mn))}

def run_coord(N, model_p, epochs=250):
    xa = np.linspace(0, 1, N); mg = np.linspace(0, 1, G)
    phi = lambda m: m**model_p; inv = lambda x: x**(1.0/model_p); ma = inv(xa)
    def cc(t):
        e = t(phi(ma)); S = CubicSpline(ma, e)
        return np.stack([S(mg), nrm(S(mg, 1)), nrm(np.exp(-4.0*mg))]).astype(np.float32), S
    Xtr = torch.tensor(np.stack([cc(t)[0] for t in TRAIN]), dtype=torch.float32, device=DEV)
    spl = [cc(t)[1] for t in TRAIN]; K = 256; mq = np.random.rand(len(TRAIN), K).astype(np.float32)
    yt = torch.tensor(np.stack([TRAIN[i](phi(mq[i])) for i in range(len(TRAIN))]).astype(np.float32), device=DEV)
    yb = torch.tensor(np.stack([spl[i](mq[i]) for i in range(len(TRAIN))]).astype(np.float32), device=DEV)
    mqt = torch.tensor(mq, device=DEV); enc, dec = Enc().to(DEV), Dec().to(DEV)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        loss = F.mse_loss(dec(enc(Xtr), mqt, yb), yt); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    enc.eval(); dec.eval(); me = inv(xe); sq = np.zeros(len(xe))
    with torch.no_grad():
        for t in TEST:
            yv = t(xe); c, S = cc(t)
            f = enc(torch.tensor(c[None], dtype=torch.float32, device=DEV))
            bb = torch.tensor(S(me)[None].astype(np.float32), device=DEV)
            yc = dec(f, torch.tensor(me[None], dtype=torch.float32, device=DEV), bb)[0].cpu().numpy()
            sq += (yc - yv) ** 2
    return float(np.sqrt(np.mean(sq/len(TEST))))

def classical(N):
    xa = np.linspace(0, 1, N); sp, gp = [], []
    for t in TEST:
        ya = t(xa); yv = t(xe); sp.append(np.sqrt(np.mean((CubicSpline(xa, ya)(xe)-yv)**2)))
        k = C(1.0, (1e-3, 1e3))*RBF(0.2, (1e-2, 1.0))
        g = GaussianProcessRegressor(kernel=k, alpha=1e-10, normalize_y=False).fit(xa[:, None], ya)
        gp.append(np.sqrt(np.mean((g.predict(xe[:, None])-yv)**2)))
    return float(np.mean(sp)), float(np.mean(gp))

print("=== REGULAR sampling — continuous RMSE (adds MLRNet physics decoder) ===", flush=True)
print(f"{'N':>3} | {'spline':>9} | {'GP':>9} | {'CNN base':>9} | {'CNN+warp':>9} | {'MLRNet':>9} | MLR wall  MLR min")
for N in [4, 6, 8, 10, 12, 14, 16]:
    sp, gp = classical(N); b = run_coord(N, 1.0); w = run_coord(N, 2.0); m = run_mlrnet(N)
    print(f"{N:>3} | {sp:.3e} | {gp:.3e} | {b:.3e} | {w:.3e} | {m['full']:.3e} | {m['wall']:.2e}  {m['min']:.2e}", flush=True)
