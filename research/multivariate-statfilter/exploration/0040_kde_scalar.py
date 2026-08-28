"""Probe 0040 -- learn the log-scale MARGINAL (KDE) vs commit to a fixed-s Gaussian.

The q-study (0039) showed the reach requirement is a minimax burst-envelope commitment: no fixed
(phi, s) can be both calm-optimal (small s, smooth) and burst-optimal (large s, reaches).  Theorem B
(optimality-proof) locates the rigidity: a single Gaussian AR(1) can hold a heavy-tailed scale law
only by committing phi->0 and one s; the reach lives in gamma0 (the marginal SHAPE), the persistence
in gamma1 (phi).  So the different way is to LEARN gamma0 -- the heavy-tailed log-scale marginal --
and keep gamma1.  Then a burst is reached because the learned marginal HAS tail mass to jump to, and
calm stays smooth because the bulk is at 0; no committed s.

Clean scalar test.  A local-level state x (random walk) with ONE sensor whose log-scale eta_t is 0
on calm and jumps to B on rare bursts (the pot-hot analogue: sensor noise bursts, coast the level).
A grid-over-eta filter, identical in every part EXCEPT the scale prior:
  * transition (shared): a stationary process with lag-1 autocorr phi and a given marginal --
    pi_pred = phi * pi + (1-phi) * marginal   (persistence phi, else resample from the marginal;
    E[eta_t eta_{t-1}] = phi*gamma0, so gamma1 = phi gamma0, an AR(1)-consistent gamma1).
  * FIXED-s: marginal = N(eta; 0, s^2)  (the current class -- one Gaussian width).
  * KDE:     marginal = a running estimate of the posterior over eta (learns gamma0 online);
    bounded to the grid so E[e^eta] < infinity is automatic (well-posedness, optimality/0024).
Measure RMSE(x) / oracle on CALM steps and BURST steps separately.  If the learned marginal matches
small-s on calm AND large-s on bursts, it dissolves the interior optimum that no fixed s can.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))

T = 3000
QX = 1e-4                      # process (level) variance -- small, so the sensor bursts matter
R0 = 1.0                       # nominal sensor variance
B = 4.0                        # burst log-scale (e^4 ~ 55x noise variance)
PHI = 0.9
GRID = np.linspace(-4.0, 10.0, 48)     # eta grid (bounded -> E[e^eta] < inf)
_LOG2PI = math.log(2.0 * math.pi)


def gen(seed):
    rng = np.random.default_rng(seed)
    eta = np.zeros(T)
    # a few rare bursts (calm-dominated -> s is under-identified toward small)
    for c in range(3):
        a = 300 + c * 850 + int(rng.integers(0, 200)); eta[a:a + 150] = B
    x = np.zeros(T); xt = 0.0; y = np.zeros(T)
    for t in range(T):
        xt += math.sqrt(QX) * rng.standard_normal(); x[t] = xt
        y[t] = xt + math.sqrt(R0 * math.exp(eta[t])) * rng.standard_normal()
    return x, y, eta


def oracle(y, eta):
    m = 0.0; P = 1.0; out = np.zeros(T)
    for t in range(T):
        Pp = P + QX; S = Pp + R0 * math.exp(eta[t]); K = Pp / S
        m = m + K * (y[t] - m); P = (1 - K) * Pp; out[t] = m
    return out


def grid_filter(y, mode, s=0.5, learn=0.02):
    G = len(GRID); Rg = R0 * np.exp(GRID)
    if mode == "fixed":
        marg = np.exp(-0.5 * (GRID / s) ** 2); marg /= marg.sum()
    else:
        marg = np.ones(G) / G                       # KDE: start flat, learn online
    pi = marg.copy()
    m = np.zeros(G); P = np.ones(G); out = np.zeros(T)
    for t in range(T):
        pi = PHI * pi + (1 - PHI) * marg            # persistence phi + resample from marginal
        Pp = P + QX
        e = y[t] - m                                # per-node innovation (shared prev mean m_g)
        S = Pp + Rg
        ll = -0.5 * (e * e / S + np.log(S))
        w = pi * np.exp(ll - ll.max()); w /= w.sum()
        K = Pp / S
        m = m + K * e; P = (1 - K) * Pp
        mbar = float(w @ m)
        out[t] = mbar
        pi = w
        if mode == "kde":
            marg = (1 - learn) * marg + learn * w    # learn the log-scale marginal (gamma0)
            marg = np.maximum(marg, 1e-6); marg /= marg.sum()
    return out


def rms(a, b, mask):
    d = (a - b)[mask]
    return math.sqrt(float((d * d).mean()))


def main(nseed=8):
    calm = np.ones(T, bool); burst = np.zeros(T, bool)
    _, _, eta0 = gen(0); burst = eta0 > 1.0; calm = ~burst   # same schedule-ish; recompute per seed below
    print(f"scalar rig: local level + bursting sensor (B={B}), {nseed} seeds")
    print(f"  {'filter':16s} {'calm/orc':>9} {'burst/orc':>10} {'all/orc':>9}")
    rows = {}
    configs = [("fixed s=0.3", "fixed", 0.3), ("fixed s=0.8", "fixed", 0.8),
              ("fixed s=1.5", "fixed", 1.5), ("fixed s=2.5", "fixed", 2.5),
              ("KDE (learned)", "kde", None)]
    for name, mode, s in configs:
        c = []; b = []; a = []
        for seed in range(nseed):
            x, y, eta = gen(seed); bmask = eta > 1.0; cmask = ~bmask
            oc = oracle(y, eta)
            est = grid_filter(y, mode, s=s or 0.5)
            oc_c = rms(oc, x, cmask); oc_b = rms(oc, x, bmask); oc_a = rms(oc, x, np.ones(T, bool))
            c.append(rms(est, x, cmask) / oc_c); b.append(rms(est, x, bmask) / oc_b)
            a.append(rms(est, x, np.ones(T, bool)) / oc_a)
        print(f"  {name:16s} {np.mean(c):9.3f} {np.mean(b):10.3f} {np.mean(a):9.3f}")


if __name__ == "__main__":
    main()


def walk_point(y, s=0.5, phi=PHI):
    """Single-point finding-18 walk on the log-scale (the production mechanism): mu += K*(target-mu),
    K* = (1-phi)/4, target = log(resid/R0), resid = C0 - state var (EMA C0)."""
    Kstar = (1 - phi) / 4.0; beta = 0.05
    m = 0.0; P = 1.0; mu = 0.0; C0 = R0; out = np.zeros(T)
    for t in range(T):
        Pp = P + QX; e = y[t] - m; C0 = (1 - beta) * C0 + beta * e * e
        resid = max(C0 - Pp, 1e-6); target = math.log(resid / R0)
        mu += Kstar * (target - mu); mu = min(max(mu, -6), 12)
        S = Pp + R0 * math.exp(mu); K = Pp / S; m = m + K * e; P = (1 - K) * Pp
        out[t] = m
    return out


def compare_point(nseed=8):
    print("\nsingle-POINT walk (production mechanism) vs the KDE posterior:")
    print(f"  {'filter':16s} {'calm/orc':>9} {'burst/orc':>10} {'all/orc':>9}")
    for name, s in [("point s=0.3", 0.3), ("point s=0.8", 0.8), ("point s=1.5", 1.5)]:
        c=[];b=[];a=[]
        for seed in range(nseed):
            x,y,eta=gen(seed); bm=eta>1.0; cm=~bm; oc=oracle(y,eta); est=walk_point(y,s=s)
            c.append(rms(est,x,cm)/rms(oc,x,cm)); b.append(rms(est,x,bm)/rms(oc,x,bm)); a.append(rms(est,x,np.ones(T,bool))/rms(oc,x,np.ones(T,bool)))
        print(f"  {name:16s} {np.mean(c):9.3f} {np.mean(b):10.3f} {np.mean(a):9.3f}")
