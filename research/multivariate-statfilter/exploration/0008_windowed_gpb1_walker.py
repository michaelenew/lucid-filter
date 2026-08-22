"""Probe 0008 -- the windowed / GPB1 walker (sigma-point, linear in D).

0007 left the two point-estimate walks bracketing the grid: stationary under-reaches
(prior cap), unbounded drifts (coupling bias from the point-estimate state KF).  The
grid does both because its GPB1 collapse MIXES the state KF over the scale window.

This probe carries a scale posterior (mu, Sigma) [diagonal Sigma] and uses 2D+1
sigma points -- the linear-in-D realisation of "simplex for direction + a marginal":
  predict   mu- = phi mu ;  Sigma- = phi^2 Sigma + diag(nu)     (stationary AR(1))
  sigma pts s_j = mu- , mu- +/- c*sigma_k e_k        (2D+1 points, prior weights w_j)
  per point run the state KF -> loglik_j, (m_j, P_j)
  reweight  W_j ∝ w_j exp(loglik_j)                  (a 2D+1-particle scale posterior)
  scale     mu = sum W_j s_j ;  Sigma = sum W_j (s_j-mu)^2       (moment match, GPB1)
  state     m = sum W_j m_j ;  P  = sum W_j (P_j + (m_j-m)(m_j-m)^T)   (GPB1 collapse)
Mixing the state KF over the window should cancel the coupling bias (no drift); the
reweighting lets the window concentrate on a sustained regime (reach, not shrink).
Diagonal Sigma keeps it 2D+1 points -> linear in D.

Reuses the n=2, m=2, correlated-Q, mixing-H model + exact grid from 0006.
"""
import os
import sys
import importlib.util

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter.core import _LOG2PI  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "p6", os.path.join(os.path.dirname(__file__), "0006_walker_nge1_and_H.py"))
p6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p6)

np.set_printoptions(precision=3, suppress=True)
D, N, M, PHI, SS, LAM, RHO, H = p6.D, p6.N, p6.M, p6.PHI, p6.SS, p6.LAM, p6.RHO, p6.H
Q_of, R_of, exact_grid, steady_fisher = p6.Q_of, p6.R_of, p6.exact_grid, p6.steady_fisher


def gen(hot, amp, T=900, seed=1):
    rng = np.random.default_rng(seed)
    psi = np.zeros((T, D))
    if hot is not None:
        psi[T // 3: 2 * T // 3, hot] = amp
    th = np.zeros(N); Y = np.zeros((T, M))
    for t in range(T):
        LQ = np.linalg.cholesky(Q_of(psi[t, :N]) + 1e-12 * np.eye(N))
        th = th + LQ @ rng.standard_normal(N)
        Y[t] = H @ th + np.sqrt(np.diag(R_of(psi[t, N:]))) * rng.standard_normal(M)
    return Y


def kf_step(m, P, Qm, Rm, y):
    """One multivariate KF predict+update at fixed (Q, R).  Returns loglik, m_post, P_post."""
    Ppred = P + Qm
    e = y - H @ m
    S = H @ Ppred @ H.T + Rm + 1e-9 * np.eye(M)
    Si = np.linalg.inv(S)
    sgn, logdet = np.linalg.slogdet(S)
    ll = -0.5 * (M * _LOG2PI + logdet + float(e @ Si @ e))
    K = Ppred @ H.T @ Si
    return ll, m + K @ e, Ppred - K @ H @ Ppred


def walk_windowed(Y, c=1.0, w0=1.0 / 3.0, trunc_frac=0.05, mode="stationary", q=0.02):
    Ich = steady_fisher()
    active = Ich >= trunc_frac * Ich.max()
    nu = SS ** 2 * (1.0 - PHI ** 2)
    mu = np.zeros(D); Sig = SS ** 2                  # diagonal scale posterior
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max())
    wother = (1.0 - w0) / (2 * D)
    out = np.zeros((len(Y), D))
    for t, y in enumerate(Y):
        if mode == "unbounded":
            mu_p = mu.copy()                          # no reversion (reach a regime)
            Sig_p = Sig + q                           # random-walk scale (window stays open)
        else:
            mu_p = PHI * mu
            Sig_p = PHI ** 2 * Sig + nu
        sd = c * np.sqrt(np.maximum(Sig_p, 1e-12))
        # 2D+1 sigma points (center + +/- per active axis) and prior weights
        pts = [mu_p.copy()]; w = [w0]
        for k in range(D):
            if not active[k]:
                continue
            for sgn in (+1.0, -1.0):
                s = mu_p.copy(); s[k] = mu_p[k] + sgn * sd[k]
                pts.append(s); w.append(wother)
        pts = np.array(pts); w = np.array(w); w = w / w.sum()
        # run the state KF at each sigma point
        lls = np.empty(len(pts)); ms = np.empty((len(pts), N)); Ps = np.empty((len(pts), N, N))
        for j, s in enumerate(pts):
            lls[j], ms[j], Ps[j] = kf_step(m, P, Q_of(s[:N]), R_of(s[N:]), y)
        # reweight (particle GPB1 over the scale window)
        mx = lls.max()
        W = w * np.exp(lls - mx); W = W / W.sum()
        mu = W @ pts
        Sig = W @ (pts - mu) ** 2
        Sig = np.maximum(Sig, 1e-4)                  # floor so the window can re-open
        mu[~active] = 0.0; Sig[~active] = SS[~active] ** 2
        # collapse the state (GPB1)
        m = W @ ms
        dmv = ms - m
        P = np.einsum("j,jab->ab", W, Ps) + np.einsum("j,ja,jb->ab", W, dmv, dmv)
        P = 0.5 * (P + P.T)
        out[t] = mu
    return out


if __name__ == "__main__":
    T = 900
    lab = ["xi1(weak)", "xi2(strong)", "eta1", "eta2"]
    for mode in ("stationary", "unbounded"):
        print(f"\n===== mode={mode} =====")
        print(f"  STATIC t>100 (want ~0): {walk_windowed(gen(None, 0.0), mode=mode)[100:].mean(0)}")
        for ax, name in [(1, "strong xi2"), (3, "sensor eta2")]:
            Y = gen(ax, 1.4)
            ref = exact_grid(Y, order=5)
            w = walk_windowed(Y, mode=mode)
            b = slice(T // 3 + 40, 2 * T // 3 - 40)
            cr = [np.corrcoef(w[30:, k], ref[30:, k])[0, 1] if w[30:, k].std() > 1e-9 else np.nan
                  for k in range(D)]
            print(f"  {name} hot (truth 1.4): GRID {ref[b].mean(0)}  WIND {w[b].mean(0)}")
            print(f"      corr {lab}: {np.array(cr)}")
