"""scratch_05_qwalker.py -- the windowed unbounded scale-POSTERIOR walker (0008 style).

Scale posterior (mu, Sig) with UNBOUNDED growth predict:  mu_p=mu, Sig_p=Sig+q.
3-point Gauss-Hermite sigma points over the sensor log-scale, GPB1 collapse.
Sweep q on (i) sustained burst (steady optimum) and (ii) finite burst (reach).
This is the object whose growth-rate q we want to derive; s no longer sets reach,
q does. Also prints the finding-18 q_mu(s) reference values.
"""
from __future__ import annotations
import math, sys
import numpy as np
from scratch_core_vec import _sim_batch, OracleVec, rmse_ratio, steady_fisher_sensor

# 3-pt Gauss-Hermite for N(mu,Sig): nodes mu +- sqrt(3 Sig), center; weights:
GH_OFF = math.sqrt(3.0)
GH_W = np.array([1.0/6, 2.0/3, 1.0/6])


class PosteriorVec:
    """Unbounded windowed scale posterior, vectorized over S seeds."""
    def __init__(self, q, S, Q=1.0, R0=1.0, Sig0=0.09):
        self.q, self.Q, self.R0, self.S = q, Q, R0, S
        self.m = np.zeros(S); self.P = np.full(S, Q + R0)
        self.mu = np.zeros(S); self.Sig = np.full(S, Sig0)

    def step(self, x):
        self.P = self.P + self.Q
        self.Sig = self.Sig + self.q                       # unbounded growth
        a = GH_OFF * np.sqrt(self.Sig)                     # (S,)
        xi = np.stack([self.mu - a, self.mu, self.mu + a], 1)   # (S,3)
        R = self.R0 * np.exp(np.clip(xi, -60, 60))
        S_ = self.P[:, None] + R
        e = (x - self.m)[:, None]; e2 = e * e
        lg = -0.5 * (np.log(S_) + e2 / S_)
        lw = np.log(GH_W)[None, :] + lg
        mx = lw.max(1, keepdims=True)
        w = np.exp(lw - mx); w = w / w.sum(1, keepdims=True)   # posterior weights
        Kk = self.P[:, None] / S_
        Kbar = (w * Kk).sum(1)
        self.m = self.m + Kbar * (x - self.m)
        self.P = (w * (1 - Kk) * self.P[:, None]).sum(1) + \
                 e2[:, 0] * (w * (Kk - Kbar[:, None]) ** 2).sum(1)
        mu_new = (w * xi).sum(1)
        self.Sig = (w * (xi - mu_new[:, None]) ** 2).sum(1) + 1e-6
        self.mu = mu_new
        return self.m


def run_q(theta, x, eta, q, nodes_ignored=None):
    n, S = x.shape
    pw = PosteriorVec(q, S); orc = OracleVec(S)
    em = np.empty((n, S)); eo = np.empty((n, S)); mus = np.empty((n, S))
    for t in range(n):
        mm = pw.step(x[t]); oo = orc.step(x[t], eta[t])
        em[t] = mm - theta[t]; eo[t] = oo - theta[t]; mus[t] = pw.mu
    return em, eo, mus


def gen_perm(rng, S, n, B, t0, Q=1.0, R0=1.0):
    eta = np.zeros(n); eta[t0:] = B
    return _sim_batch(rng, S, eta, Q, R0)


QS = np.array([0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])


def sweep_q_perm(B, S=96, n=4000, t0=500, settle=400):
    rng = np.random.default_rng(321)
    theta, x, eta = gen_perm(rng, S, n, B, t0)
    tmask = np.zeros(n, bool); tmask[t0:t0+settle] = True
    smask = np.zeros(n, bool); smask[t0+settle:] = True
    sm = np.zeros(len(QS)); tm = np.zeros(len(QS)); reach = np.zeros(len(QS))
    sse = np.zeros(len(QS)); tse = np.zeros(len(QS))
    for i, q in enumerate(QS):
        em, eo, mus = run_q(theta, x, eta, q)
        sm[i], sse[i] = rmse_ratio(em, eo, smask)
        tm[i], tse[i] = rmse_ratio(em, eo, tmask)
        reach[i] = mus[t0+settle//2:t0+settle].mean()
    return sm, sse, tm, tse, reach


if __name__ == "__main__":
    print("############ Q4: windowed posterior walker, q-sweep ############")
    print("finding-18 q_mu(s) reference (phi=0.85): q_mu = K*^2/(I_char(1-K*))")
    for s in [0.15, 0.3, 0.5, 0.8]:
        Ich = steady_fisher_sensor(s, 0.85)
        Ks = (1-0.85)/4
        qmu = Ks**2/(Ich*(1-Ks))
        print(f"   s={s}: I_char={Ich:.4f}  q_mu={qmu:.4f}")
    print()
    for B in [2.0, 3.5, 5.0, 6.5, 8.0]:
        sm, sse, tm, tse, reach = sweep_q_perm(B)
        isb = int(np.argmin(sm)); itb = int(np.argmin(tm))
        print(f"=== B={B}  (permanent shift) ===")
        print("    q      steady-ratio      transient-ratio    reach")
        for i, q in enumerate(QS):
            tag = (" S*" if i==isb else "") + (" T*" if i==itb else "")
            print(f"  {q:7.4f}  {sm[i]:.3f}+-{sse[i]:.3f}   {tm[i]:.3f}+-{tse[i]:.3f}   {reach[i]:5.2f}{tag}")
        print(f"  --> steady q*={QS[isb]:.4f}   transient q*={QS[itb]:.4f}   (reach f_B at T*={reach[itb]/B:.2f})")
        sys.stdout.flush()
