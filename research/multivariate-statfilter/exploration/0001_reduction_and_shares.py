"""Prototype: multivariate statfilter with supplied H and full-symmetric Q0,R0.

Goal here is ONLY to validate the two load-bearing claims before productionising:
  (1) at n=m=1, H=[[1]] the multivariate recursion reduces to the shipped scalar
      AdaptiveFilter to ~1e-10 (mean, var, loglik, shares, mode coords);
  (2) the Mahalanobis shares sum to 1 and, in the scalar case, equal P/S, Qg/S, Rg/S.

Model:
  theta_t = theta_{t-1} + w_t,  w_t ~ N(0, Q0 exp(lamP_t))
  y_t     = H theta_t   + v_t,  v_t ~ N(0, R0 exp(lamM_t))
  lam^c   AR(1) with (phi_c, s_c); scalar scale channel per matrix (unchanged grid).
GPB1 collapse to one Gaussian (m, P) per step, exactly as scalar core.
"""
import math
import numpy as np

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter.core import _chain, AdaptiveFilter, Params

_LOG2PI = math.log(2.0 * math.pi)


class MvProto:
    def __init__(self, Q0, R0, H, phi_P=0.0, phi_M=0.0, s_P=0.0, s_M=0.0, order=5):
        self.Q0 = np.atleast_2d(np.asarray(Q0, float))
        self.R0 = np.atleast_2d(np.asarray(R0, float))
        self.H = np.atleast_2d(np.asarray(H, float))
        self.m_dim, self.n_dim = self.H.shape
        assert self.Q0.shape == (self.n_dim, self.n_dim)
        assert self.R0.shape == (self.m_dim, self.m_dim)
        self.phi_P, self.phi_M, self.s_P, self.s_M = phi_P, phi_M, s_P, s_M
        self.order = order
        lamP, wP, TP = _chain(phi_P, s_P, order)
        lamM, wM, TM = _chain(phi_M, s_M, order)
        self.LP = np.repeat(lamP, order)
        self.LM = np.tile(lamM, order)
        self.T = np.kron(TP, TM)
        self.pi0 = np.kron(wP, wM)
        # per-node covariance matrices (G, n, n) and (G, m, m)
        self.Qg = self.Q0[None] * np.exp(np.clip(self.LP, -60, 60))[:, None, None]
        self.Rg = self.R0[None] * np.exp(np.clip(self.LM, -60, 60))[:, None, None]
        self.reset()

    def reset(self):
        self.pi = None
        self.m = None
        self.P = None
        self.prev_lamP = 0.0
        self.prev_lamM = 0.0
        self.loglik = 0.0
        return self

    def update(self, y):
        y = np.atleast_1d(np.asarray(y, float))
        G, n, m = self.pi0.size, self.n_dim, self.m_dim
        H = self.H
        if self.pi is None:
            self.pi = self.pi0.copy()
            self.m = y @ np.linalg.pinv(H).T if np.all(np.isfinite(y)) else np.zeros(n)
            # diffuse start: widest process step + one obs of measurement, mapped up
            P0 = float(self.Rg.reshape(G, -1).max() + self.Qg.reshape(G, -1).max())
            self.P = np.eye(n) * P0 * n
        pi = self.pi @ self.T
        # predict
        Ppred = self.P[None] + self.Qg                       # (G, n, n)
        e = y - H @ self.m                                   # (m,) shared across nodes
        # per-node innovation cov, gain, posterior
        HPpre = np.einsum("ij,gjk->gik", H, Ppred)          # (G, m, n)
        S = np.einsum("gik,lk->gil", HPpre, H) + self.Rg    # (G, m, m)
        Sinv = np.linalg.inv(S)
        K = np.einsum("gnk,gkm->gnm", np.einsum("gij,kj->gik", Ppred, H), Sinv)  # (G,n,m)=Ppred H^T Sinv
        # loglik per node
        sign, logdet = np.linalg.slogdet(S)
        maha = np.einsum("i,gij,j->g", e, Sinv, e)
        lg = -0.5 * (m * _LOG2PI + logdet + maha)
        mx = float(lg.max())
        w = pi * np.exp(lg - mx)
        Z = float(w.sum())
        ll = math.log(Z) + mx
        pi = w / Z
        # posterior means per node, collapse
        mpost = self.m[None] + np.einsum("gnm,m->gn", K, e)  # (G, n)
        Kbar = np.einsum("g,gnm->nm", pi, K)
        m_new = self.m + Kbar @ e
        Ppost = Ppred - np.einsum("gnm,gml->gnl", np.einsum("gnm,ml->gnl", K, H), Ppred)  # (I-KH)Ppred
        dm = mpost - m_new                                    # (G, n)
        P_new = np.einsum("g,gnl->nl", pi, Ppost) + np.einsum("g,gn,gl->nl", pi, dm, dm)
        # mode coords (scalar channels, unchanged)
        lamP = float(pi @ self.LP)
        lamM = float(pi @ self.LM)
        # Variance-decomposition shares (innovation-independent, like the scalar
        # P/S, Qg/S, Rg/S): S = H P H^T + H Qg H^T + Rg -- three pieces summing to
        # S -- and share_* = tr(S^-1 piece)/m, which sums to 1 and reduces to the
        # scalar ratios at m=1.  (The e-weighted Mahalanobis form also reduces but
        # is 0/0 at e=0, e.g. the first step, so the trace form is the faithful one.)
        A = np.einsum("ij,gjk,lk->gil", H, self.P[None] + 0 * self.Qg, H)     # H P_prev H^T (broadcast)
        Bp = np.einsum("ij,gjk,lk->gil", H, self.Qg, H)                       # H Qg H^T
        Cp = self.Rg
        def tshare(M):
            return np.einsum("gij,gji->g", Sinv, M) / m                        # tr(S^-1 M)/m per node
        sp = float(pi @ tshare(A))
        spr = float(pi @ tshare(Bp))
        sm = float(pi @ tshare(Cp))
        self.pi, self.m, self.P = pi, m_new, P_new
        self.prev_lamP, self.prev_lamM = lamP, lamM
        self.loglik += ll
        return dict(mean=m_new.copy(), var=P_new.copy(), innovation=e.copy(), loglik=ll,
                    share_prior=sp, share_process=spr, share_measurement=sm,
                    process_scale=lamP, measurement_scale=lamM)


def scalar_check():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(200)) + rng.standard_normal(200) * 0.7
    p = Params(Q=0.8, s2=0.5, phi_P=0.9, phi_M=0.8, s_P=0.4, s_M=0.3)
    ref = AdaptiveFilter(p, order=5)
    mv = MvProto(Q0=[[0.8]], R0=[[0.5]], H=[[1.0]],
                 phi_P=0.9, phi_M=0.8, s_P=0.4, s_M=0.3, order=5)
    dmean = dvar = dll = dsh = 0.0
    ref.reset(); mv.reset()
    for v in x:
        rs = ref.update(float(v))
        ms = mv.update([float(v)])
        dmean = max(dmean, abs(rs.mean - float(ms["mean"][0])))
        dvar = max(dvar, abs(rs.var - float(ms["var"][0, 0])))
        dll = max(dll, abs(rs.loglik - ms["loglik"]))
        dsh = max(dsh, abs(rs.share_prior - ms["share_prior"]),
                  abs(rs.share_process - ms["share_process"]),
                  abs(rs.share_measurement - ms["share_measurement"]))
    print(f"scalar reduction (n=m=1, H=[[1]]):")
    print(f"  max |dmean|  = {dmean:.2e}")
    print(f"  max |dvar|   = {dvar:.2e}")
    print(f"  max |dloglik|= {dll:.2e}")
    print(f"  max |dshare| = {dsh:.2e}")
    print(f"  total loglik ref={ref._loglik:.6f}  mv={mv.loglik:.6f}  d={abs(ref._loglik-mv.loglik):.2e}")


def mv_smoke():
    rng = np.random.default_rng(1)
    n, m, T = 3, 2, 400
    # supplied H: observe component 0 and (component1 - component2)
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]])
    Q0 = np.array([[1.0, 0.3, 0.0], [0.3, 0.8, 0.1], [0.0, 0.1, 0.5]])
    R0 = np.array([[0.4, 0.1], [0.1, 0.6]])
    th = np.zeros(n); ys = np.zeros((T, m))
    LQ = np.linalg.cholesky(Q0); LR = np.linalg.cholesky(R0)
    for t in range(T):
        th = th + LQ @ rng.standard_normal(n)
        ys[t] = H @ th + LR @ rng.standard_normal(m)
    f = MvProto(Q0, R0, H, phi_P=0.85, phi_M=0.8, s_P=0.3, s_M=0.25, order=5)
    f.reset()
    finite = share_ok = True
    ssum = []
    for t in range(T):
        s = f.update(ys[t])
        finite = finite and np.all(np.isfinite(s["mean"])) and np.all(np.isfinite(s["var"]))
        ssum.append(s["share_prior"] + s["share_process"] + s["share_measurement"])
    ssum = np.array(ssum)
    print(f"\nmultivariate smoke (n={n}, m={m}):")
    print(f"  all finite: {finite}")
    print(f"  shares sum to 1: max|sum-1| = {np.max(np.abs(ssum-1.0)):.2e}")
    print(f"  final loglik = {f.loglik:.3f}")


if __name__ == "__main__":
    scalar_check()
    mv_smoke()
