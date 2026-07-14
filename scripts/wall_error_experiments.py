"""Exploration: reduce the repulsive-wall reconstruction error.

Diagnosis (see results/error_vs_position): ~78% of the CNN+MLP squared error
sits in the wall region x < 0.15 (only 15% of the domain), because the wall is
the steepest part of the curve and the least-sampled by uniform points.

This script ablates three wall-targeted ideas on the same harness (coordinate
CNN+MLP, residual-on-spline, channels [energy, derivative, morse]):

  A baseline          uniform-x anchors, model coordinate = x, morse beta=4
  B steeper-morse     baseline + morse beta=10 (sharper wall descriptor)
  C warp-repr         model works in warped coord m (x=m^2 stretches the wall),
                      but observations stay uniform in x  -> representation only
  D warp+sample       anchors uniform in m -> x_i=m_i^2 (denser at the wall),
                      model works in m                    -> MLR-style y_p idea
  E warp+sample+morse D + morse beta=8

Metric: RMSE by physical-x region, especially the wall (x<0.15), plus the well
minimum. Everything is measured in physical x so configs are comparable.

Note: C needs no change to sampling (works on fixed uniform data); D/E assume you
control where the PES is sampled (denser near the wall) -- true for ab initio scans.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline

from pes_1D.superres import lr_indices, sample_pes_curve

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
FAMS = ("lennard_jones", "morse", "buckingham_exp6", "extended_rydberg")
DENSE, N, G = 1024, 8, 384
xe = np.linspace(0, 1, 2000)


def truthfn(seed, fam):
    d = sample_pes_curve(np.random.default_rng(seed), fam, DENSE)
    return CubicSpline(np.linspace(0, 1, DENSE), d)
def nrm(a):
    a = np.asarray(a, float); return 2 * (a - a.min()) / (a.max() - a.min() + 1e-10) - 1

TRAIN = [truthfn(1000 + i, FAMS[i % 4]) for i in range(320)]
TEST = [truthfn(90000 + i, FAMS[i % 4]) for i in range(80)]


class Enc(nn.Module):
    def __init__(s, c=48):
        super().__init__(); s.h = nn.Conv1d(3, c, 3, padding=1)
        s.b = nn.ModuleList([nn.Conv1d(c, c, 3, padding=d, dilation=d) for d in (2, 4, 8)])
    def forward(s, x):
        h = F.leaky_relu(s.h(x), .2)
        for cv in s.b: h = F.leaky_relu(h + cv(h), .2)
        return h
class Dec(nn.Module):
    def __init__(s, c=48, fr=(1, 2, 4, 8)):
        super().__init__(); s.fr = torch.tensor([f * np.pi for f in fr], dtype=torch.float32)
        s.n = nn.Sequential(nn.Linear(c + 2 * len(fr) + 1, 96), nn.SiLU(),
                            nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 1))
    def g(s, f, x):
        p = x * (f.shape[-1] - 1); i0 = p.floor().clamp(0, f.shape[-1] - 2).long(); w = (p - i0).unsqueeze(1)
        a = f.gather(2, i0.unsqueeze(1).expand(-1, f.shape[1], -1)); b = f.gather(2, (i0 + 1).unsqueeze(1).expand(-1, f.shape[1], -1))
        return ((1 - w) * a + w * b).transpose(1, 2)
    def forward(s, f, x, base):
        fx = s.g(f, x); ang = x.unsqueeze(-1) * s.fr.to(x.device)
        return base + s.n(torch.cat([fx, torch.sin(ang), torch.cos(ang), base.unsqueeze(-1)], -1)).squeeze(-1)


def run(p_warp=1.0, place="x", beta=4.0):
    """p_warp: phi(m)=m**p mapping model-coord m -> physical x. place: anchors uniform in 'x' or 'm'."""
    phi = lambda m: m ** p_warp
    inv = lambda x: x ** (1.0 / p_warp)
    mg = np.linspace(0, 1, G)                                   # model-coord grid
    if place == "m":
        m_anchor = np.linspace(0, 1, N)
    else:
        m_anchor = inv(np.linspace(0, 1, N))
    morse = nrm(np.exp(-beta * mg))

    def cond(t):
        e = t(phi(m_anchor)); S = CubicSpline(m_anchor, e)
        return np.stack([S(mg), nrm(S(mg, 1)), morse]).astype(np.float32), S

    Xtr = torch.tensor(np.stack([cond(t)[0] for t in TRAIN]), dtype=torch.float32, device=DEV)
    splines = [cond(t)[1] for t in TRAIN]
    K = 256; mq = np.random.rand(len(TRAIN), K).astype(np.float32)
    yt = torch.tensor(np.stack([TRAIN[i](phi(mq[i])) for i in range(len(TRAIN))]).astype(np.float32), device=DEV)
    yb = torch.tensor(np.stack([splines[i](mq[i]) for i in range(len(TRAIN))]).astype(np.float32), device=DEV)
    mqt = torch.tensor(mq, device=DEV)
    enc, dec = Enc().to(DEV), Dec().to(DEV)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    for _ in range(300):
        loss = F.mse_loss(dec(enc(Xtr), mqt, yb), yt); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    enc.eval(); dec.eval()
    me = inv(xe)                                                # eval at physical xe -> model coord
    sq = np.zeros(len(xe)); minerr = []
    with torch.no_grad():
        for t in TEST:
            yv = t(xe); imin = int(np.argmin(yv))
            c, S = cond(t)
            f = enc(torch.tensor(c[None], dtype=torch.float32, device=DEV))
            bb = torch.tensor(S(me)[None].astype(np.float32), device=DEV)
            yc = dec(f, torch.tensor(me[None], dtype=torch.float32, device=DEV), bb)[0].cpu().numpy()
            sq += (yc - yv) ** 2; minerr.append(abs(yc[imin] - yv[imin]))
    reg = lambda lo, hi: float(np.sqrt(np.mean(sq[(xe >= lo) & (xe < hi)] / len(TEST))))
    return {"wall": reg(0, .15), "well": reg(.30, .45), "outer": reg(.45, 1.0),
            "full": float(np.sqrt(np.mean(sq / len(TEST)))), "min": float(np.mean(minerr))}


CONFIGS = [
    ("A baseline (uniform-x, coord=x, b=4)", dict(p_warp=1.0, place="x", beta=4.0)),
    ("B steeper-morse (b=10)",               dict(p_warp=1.0, place="x", beta=10.0)),
    ("C warp-repr (coord=m, obs uniform-x)", dict(p_warp=2.0, place="x", beta=4.0)),
    ("D warp+sample (obs denser at wall)",   dict(p_warp=2.0, place="m", beta=4.0)),
    ("E warp+sample+morse (b=8)",            dict(p_warp=2.0, place="m", beta=8.0)),
]
print(f"{'config':40s} | {'wall':>9s} | {'well':>9s} | {'outer':>9s} | {'full':>9s} | {'min':>9s}")
base = None
for name, kw in CONFIGS:
    r = run(**kw)
    if base is None: base = r
    dw = 100 * (r["wall"] - base["wall"]) / base["wall"]
    print(f"{name:40s} | {r['wall']:.3e} | {r['well']:.3e} | {r['outer']:.3e} | {r['full']:.3e} | {r['min']:.3e}   wall {dw:+.0f}%", flush=True)
