"""Use a frozen, accurate PES-validity discriminator as a fixed generator penalty.

NOT a GAN: D is trained once to high accuracy ("is this a valid PES?"), then FROZEN.
The generator (CNN+MLP + warp) trains with  MSE + lambda * (-log D(fake)), pushing
its output toward D's "valid PES" region. D never chases G (no min-max).

D negatives are realistic reconstruction errors (sparse-spline recons + shape
distortions), so D learns to reject exactly the mistakes the generator makes.
D is spectral-normed so its gradient is smooth (harder to game).

Reports: D accuracy; then generator RMSE (full/wall/min, in-family + O2) vs lambda,
plus the mean D-score of the generator output (to detect gaming: score up, RMSE not).
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
DENSE, GG, G, P, N = 1024, 128, 384, 2.0, 8
x128, mg, xe = np.linspace(0, 1, GG), np.linspace(0, 1, G), np.linspace(0, 1, 2000)
phi = lambda m: m**P; inv = lambda x: x**(1/P)
xa = np.linspace(0, 1, N); ma = inv(xa)
def truthfn(seed, fam):
    d = sample_pes_curve(np.random.default_rng(seed), fam, DENSE)
    return CubicSpline(np.linspace(0, 1, DENSE), d)
def nrm(a):
    a = np.asarray(a, float); return 2*(a-a.min())/(a.max()-a.min()+1e-10)-1
TRAIN = [truthfn(1000+i, FAMS[i%4]) for i in range(400)]
TEST = [truthfn(90000+i, FAMS[i%4]) for i in range(60)]
O2 = [truthfn(70000+i, "reudenberg_o2") for i in range(20)]

# ---------- 1) build D training data ----------
pos = np.stack([t(x128) for t in TRAIN]).astype(np.float32)          # true PES (already ~[-1,1])
rng = np.random.default_rng(1)
def make_neg(c):
    m = rng.integers(0, 3)
    if m == 0:                                                       # sparse-spline recon (gap errors)
        Nn = int(rng.choice([4, 5, 6, 8, 10])); idx = lr_indices(GG, Nn)
        neg = CubicSpline(x128[idx], c[idx])(x128)
    elif m == 1:                                                     # spurious bump
        cc, w, a = rng.uniform(.1, .9), rng.uniform(.03, .12), rng.uniform(.15, .5)*rng.choice([-1, 1])
        neg = c + a*np.exp(-((x128-cc)/w)**2)
    else:                                                            # spurious oscillation
        neg = c + rng.uniform(.05, .2)*np.sin(rng.uniform(1, 5)*np.pi*x128 + rng.uniform(0, 6))
    return nrm(neg).astype(np.float32)                              # renormalize so D judges SHAPE not range
neg = np.stack([make_neg(c) for c in pos])

class Dnet(nn.Module):
    def __init__(s, c=32):
        super().__init__(); sn = nn.utils.spectral_norm
        s.net = nn.ModuleList([sn(nn.Conv1d(1, c, 4, 2, 1)), sn(nn.Conv1d(c, 2*c, 4, 2, 1)), sn(nn.Conv1d(2*c, 4*c, 4, 2, 1))])
        s.fc = sn(nn.Linear(4*c, 1))
    def forward(s, x):                                              # x[B,GG]
        h = x.unsqueeze(1)
        for l in s.net: h = F.leaky_relu(l(h), .2)
        return s.fc(F.adaptive_avg_pool1d(h, 1).squeeze(-1)).squeeze(-1)

D = Dnet().to(DEV)
Xd = torch.tensor(np.concatenate([pos, neg]), device=DEV)
yd = torch.tensor(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32), device=DEV)
optd = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9)); gc = torch.Generator().manual_seed(0)
for ep in range(200):
    for sel in torch.randperm(len(Xd), generator=gc).split(128):
        sel = sel.to(DEV); loss = F.binary_cross_entropy_with_logits(D(Xd[sel]), yd[sel])
        optd.zero_grad(); loss.backward(); optd.step()
D.eval()
with torch.no_grad():
    acc = (((D(Xd) > 0).float() == yd).float().mean()).item()
print(f"Discriminator accuracy (valid PES vs realistic fakes): {acc*100:.1f}%", flush=True)
for p in D.parameters(): p.requires_grad_(False)

# ---------- 2) generator (CNN+MLP+warp), trained MSE + lambda * penalty(frozen D) ----------
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

def cond_of(t):
    e = t(phi(ma)); S = CubicSpline(ma, e)
    return np.stack([S(mg), nrm(S(mg, 1)), nrm(np.exp(-4*mg))]).astype(np.float32), S
condT = torch.tensor(np.stack([cond_of(t)[0] for t in TRAIN]), dtype=torch.float32, device=DEV)
splT = [cond_of(t)[1] for t in TRAIN]
m128 = inv(x128)                                                    # eval D on the 128 grid (model coord)
y128 = torch.tensor(np.stack([TRAIN[i](x128) for i in range(len(TRAIN))]).astype(np.float32), device=DEV)
base128 = torch.tensor(np.stack([splT[i](m128) for i in range(len(TRAIN))]).astype(np.float32), device=DEV)
m128T = torch.tensor(m128[None].repeat(len(TRAIN), 0), dtype=torch.float32, device=DEV)

def train_gen(lam):
    enc, dec = Enc().to(DEV), Dec().to(DEV)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=250)
    for _ in range(250):
        fake = dec(enc(condT), m128T, base128)                     # [B,128] on grid
        mse = F.mse_loss(fake, y128)
        pen = F.softplus(-D(fake)).mean() if lam > 0 else torch.zeros((), device=DEV)
        (mse + lam*pen).backward(); opt.step(); opt.zero_grad(); sch.step()
    return enc.eval(), dec.eval()

def evaluate(enc, dec, curves):
    me = inv(xe); sq = np.zeros(len(xe)); mn = []; dsc = []
    with torch.no_grad():
        for t in curves:
            c, S = cond_of(t)
            f = enc(torch.tensor(c[None], dtype=torch.float32, device=DEV))
            bb = torch.tensor(S(me)[None].astype(np.float32), device=DEV)
            yc = dec(f, torch.tensor(me[None], dtype=torch.float32, device=DEV), bb)[0].cpu().numpy()
            yv = t(xe); imin = int(np.argmin(yv)); sq += (yc-yv)**2; mn.append(abs(yc[imin]-yv[imin]))
            # D score of the reconstruction (on 128 grid)
            f128 = dec(enc(torch.tensor(c[None], dtype=torch.float32, device=DEV)),
                       torch.tensor(m128[None], dtype=torch.float32, device=DEV), torch.tensor(S(m128)[None].astype(np.float32), device=DEV))
            dsc.append(torch.sigmoid(D(f128)).item())
    reg = lambda lo, hi: float(np.sqrt(np.mean(sq[(xe >= lo) & (xe < hi)]/len(curves))))
    return {"full": float(np.sqrt(np.mean(sq/len(curves)))), "wall": reg(0, .15), "min": float(np.mean(mn)), "Dscore": float(np.mean(dsc))}

print(f"\n{'lambda':>7} | {'in: full':>9} {'wall':>9} {'min':>9} {'Dscore':>7} | {'O2 full':>9} {'O2 Dscore':>9}")
for lam in [0.0, 0.1, 0.3, 1.0, 3.0]:
    enc, dec = train_gen(lam)
    r = evaluate(enc, dec, TEST); o = evaluate(enc, dec, O2)
    print(f"{lam:>7.1f} | {r['full']:.3e} {r['wall']:.3e} {r['min']:.3e} {r['Dscore']:.3f} | {o['full']:.3e} {o['Dscore']:.3f}", flush=True)
