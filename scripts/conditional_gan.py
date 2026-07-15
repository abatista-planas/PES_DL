"""A correct conditional STOCHASTIC GAN for PES super-resolution.

Purpose (established by prior analysis): the adversary cannot beat MSE on point
RMSE (near-delta conditional distribution). The one thing a GAN can add is a
DISTRIBUTION of plausible curves -> a calibrated uncertainty band. So:

  G(c, z): warp-encoder + coordinate decoder, residual on the spline base, with a
           latent z so different z -> different plausible curve.
  D(x, c): CONDITIONAL (sees candidate + conditioning), spectral norm, no BatchNorm,
           in-graph so gradients reach G. Relativistic-hinge loss + instance noise.
  L_G    : adversarial + data-consistency at observed points + weak mean anchor +
           MSGAN mode-seeking (so z is used, not ignored).

Evaluated on: point RMSE (of the sample mean), and — the real test — uncertainty
CALIBRATION (does the true curve fall in the predicted band at the right rate?),
band-vs-error correlation, and whether uncertainty grows out-of-family (O2).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from pes_1D.superres import lr_indices, sample_pes_curve

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
FAMS = ("lennard_jones", "morse", "buckingham_exp6", "extended_rydberg")
DENSE, G, DZ, N = 1024, 192, 8, 8
P = 2.0                                            # warp exponent
phi = lambda m: m ** P; inv = lambda x: x ** (1.0 / P)
mg = np.linspace(0, 1, G); xe = np.linspace(0, 1, 2000)
xa = np.linspace(0, 1, N); ma = inv(xa)            # anchors (uniform in x)
obs_idx = np.clip(np.round(ma * (G - 1)).astype(int), 0, G - 1)

def truthfn(seed, fam):
    d = sample_pes_curve(np.random.default_rng(seed), fam, DENSE)
    return CubicSpline(np.linspace(0, 1, DENSE), d)
def nrm(a):
    a = np.asarray(a, float); return 2*(a-a.min())/(a.max()-a.min()+1e-10)-1
morse_g = nrm(np.exp(-4.0 * mg))
def cond_of(t):
    e = t(phi(ma)); S = CubicSpline(ma, e)
    return np.stack([S(mg), nrm(S(mg, 1)), morse_g]).astype(np.float32), S

TRAIN = [truthfn(1000+i, FAMS[i%4]) for i in range(256)]
TEST = [truthfn(90000+i, FAMS[i%4]) for i in range(60)]
O2 = [truthfn(70000+i, "reudenberg_o2") for i in range(20)]

class Enc(nn.Module):
    def __init__(s, c=48):
        super().__init__(); s.h = nn.Conv1d(3, c, 3, padding=1)
        s.b = nn.ModuleList([nn.Conv1d(c, c, 3, padding=d, dilation=d) for d in (2, 4, 8)])
    def forward(s, x):
        h = F.leaky_relu(s.h(x), .2)
        for cv in s.b: h = F.leaky_relu(h + cv(h), .2)
        return h

class Gen(nn.Module):
    def __init__(s, c=48, fr=(1, 2, 4, 8)):
        super().__init__(); s.enc = Enc(c)
        s.fr = torch.tensor([f*np.pi for f in fr], dtype=torch.float32)
        s.net = nn.Sequential(nn.Linear(c + 2*len(fr) + 1 + DZ, 96), nn.SiLU(),
                              nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 1))
    def gather(s, feat, m):                        # feat[B,C,G], m[B,M]
        p = m*(feat.shape[-1]-1); i0 = p.floor().clamp(0, feat.shape[-1]-2).long(); w = (p-i0).unsqueeze(1)
        a = feat.gather(2, i0.unsqueeze(1).expand(-1, feat.shape[1], -1)); b = feat.gather(2, (i0+1).unsqueeze(1).expand(-1, feat.shape[1], -1))
        return ((1-w)*a + w*b).transpose(1, 2)
    def forward(s, cond, m, base, z):              # cond[B,3,G], m[B,M], base[B,M], z[B,DZ]
        feat = s.enc(cond); fx = s.gather(feat, m)
        ang = m.unsqueeze(-1)*s.fr.to(m.device)
        zb = z.unsqueeze(1).expand(-1, m.shape[1], -1)
        inp = torch.cat([fx, torch.sin(ang), torch.cos(ang), base.unsqueeze(-1), zb], -1)
        return base + s.net(inp).squeeze(-1)

class Disc(nn.Module):
    def __init__(s, c=32):
        super().__init__(); sn = nn.utils.spectral_norm
        s.net = nn.ModuleList([sn(nn.Conv1d(4, c, 4, 2, 1)), sn(nn.Conv1d(c, 2*c, 4, 2, 1)), sn(nn.Conv1d(2*c, 4*c, 4, 2, 1))])
        s.fc = sn(nn.Linear(4*c, 1))
    def forward(s, cand, cond):                    # cand[B,G], cond[B,3,G]
        h = torch.cat([cand.unsqueeze(1), cond], 1)
        for l in s.net: h = F.leaky_relu(l(h), .2)
        return s.fc(F.adaptive_avg_pool1d(h, 1).squeeze(-1))

# ---- training ----
condT = torch.tensor(np.stack([cond_of(t)[0] for t in TRAIN]), dtype=torch.float32, device=DEV)
baseT = condT[:, 0, :]                              # spline base on grid
realT = torch.tensor(np.stack([TRAIN[i](phi(mg)) for i in range(len(TRAIN))]).astype(np.float32), device=DEV)
mgT = torch.tensor(mg[None].repeat(len(TRAIN), 0), dtype=torch.float32, device=DEV)
gen, disc = Gen().to(DEV), Disc().to(DEV)
og = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.9))
od = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.9))
gcpu = torch.Generator().manual_seed(0)
B = len(TRAIN); EP = 500
for ep in range(EP):
    sig = 0.10 * max(0.0, 1 - ep/(0.7*EP))
    z1 = torch.randn(B, DZ, device=DEV); z2 = torch.randn(B, DZ, device=DEV)
    f1 = gen(condT, mgT, baseT, z1)
    # D step (relativistic hinge, instance noise)
    rn = realT + sig*torch.randn_like(realT); fn = f1.detach() + sig*torch.randn_like(f1)
    dr, df = disc(rn, condT), disc(fn, condT)
    ld = F.relu(1 - (dr - df.mean())).mean() + F.relu(1 + (df - dr.mean())).mean()
    od.zero_grad(); ld.backward(); od.step()
    # G step
    f1 = gen(condT, mgT, baseT, z1); f2 = gen(condT, mgT, baseT, z2)
    dr, df = disc(realT + sig*torch.randn_like(realT), condT), disc(f1, condT)
    adv = F.relu(1 - (df - dr.mean())).mean() + F.relu(1 + (dr - df.mean())).mean()
    obs = F.mse_loss(f1[:, obs_idx], realT[:, obs_idx])         # data consistency
    mean_anchor = F.mse_loss(0.5*(f1+f2), realT)                # keep mean accurate (weak)
    ms = (f1-f2).abs().mean() / ((z1-z2).abs().mean() + 1e-3)   # MSGAN: reward z-dependence
    lg = 1.0*adv + 20.0*obs + 0.5*mean_anchor - 2.0*ms
    og.zero_grad(); lg.backward(); og.step()
    if ep % 100 == 0:
        print(f"  ep {ep:3d}  adv {adv.item():.3f}  obs {obs.item():.2e}  meanMSE {mean_anchor.item():.2e}  diversity {ms.item():.3e}", flush=True)

# ---- evaluation: point RMSE + uncertainty calibration ----
S = 40; me = inv(xe)
def gp_std(t):
    xaa = xa; ya = t(xaa); k = C(1.0, (1e-3, 1e3))*RBF(0.2, (1e-2, 1.0))
    g = GaussianProcessRegressor(kernel=k, alpha=1e-10).fit(xaa[:, None], ya)
    m, sd = g.predict(xe[:, None], return_std=True); return m, sd

def evaluate(curves, tag):
    gen.eval(); rmse_mu, cover, corr, meansig, gp_rmse, gp_cover = [], [], [], [], [], []
    with torch.no_grad():
        for t in curves:
            c, S_ = cond_of(t)
            cc = torch.tensor(c[None], dtype=torch.float32, device=DEV).expand(S, -1, -1)
            bb = torch.tensor(S_(me)[None].astype(np.float32), device=DEV).expand(S, -1)
            mm = torch.tensor(me[None], dtype=torch.float32, device=DEV).expand(S, -1)
            z = torch.randn(S, DZ, device=DEV)
            samp = gen(cc, mm, bb, z).cpu().numpy()               # [S, 2000]
            yv = t(xe); mu = samp.mean(0); sd = samp.std(0) + 1e-9
            rmse_mu.append(np.sqrt(np.mean((mu-yv)**2)))
            cover.append(np.mean(np.abs(yv-mu) <= 1.645*sd))       # nominal 90%
            corr.append(np.corrcoef(sd, np.abs(yv-mu))[0, 1])
            meansig.append(sd.mean())
            gm, gs = gp_std(t); gp_rmse.append(np.sqrt(np.mean((gm-yv)**2)))
            gp_cover.append(np.mean(np.abs(yv-gm) <= 1.645*(gs+1e-9)))
    print(f"\n[{tag}]  N={N}, {len(curves)} curves, {S} samples")
    print(f"  GAN  mean-RMSE {np.mean(rmse_mu):.3e} | 90%-band coverage {np.mean(cover):.2f} "
          f"| corr(sigma,|err|) {np.nanmean(corr):.2f} | mean sigma {np.mean(meansig):.3e}")
    print(f"  GP   mean-RMSE {np.mean(gp_rmse):.3e} | 90%-band coverage {np.mean(gp_cover):.2f}")

evaluate(TEST, "in-family")
evaluate(O2, "O2 out-of-family")
