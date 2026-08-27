"""Probe 0011 -- LINEAR-cost per-component walker (mean-field factored scale).

The shipped WalkingVectorFilter runs the state KF at every node of the tensor-product
scale grid: nodes**(#active axes), exponential.  0003 measured the scale-Fisher as
NEARLY BLOCK-DIAGONAL (process eigenmodes decouple, sensors decouple; only the one
~0.2 process<->measurement cross-term).  So the joint scale posterior should factor,
pi(psi) ~= prod_k pi_k(psi_k), and a mean-field representation is faithful at LINEAR
cost: one 1-D dense window per axis, D windows, D*nodes evaluations -- not nodes**D.

Construction (this probe):
  * one collapsed state (m, P);
  * per active axis k a 1-D dense window (walking centre mu_k), marginal pi_k;
  * SCALE update (mean-field): for axis k, vary only axis k over its window with the
    other axes held at their current means, recompute S and the obs likelihood, update
    pi_k, and walk mu_k by the finding-18 loop (the scalar WalkingFilter loop, one copy
    per axis, sharing the state);
  * STATE update: one KF at the posterior-mean scale (the axes are tracked by their
    windows, so a point-estimate state is enough) -- O(1).
Total per step: D * nodes single-observation KF evaluations = LINEAR in the axes.

Compared to the exact tensor grid (0006) on the n=2,m=2 mixing-H case.
"""
import os
import sys
import math
import importlib.util

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter.core import _LOG2PI  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "p6", os.path.join(os.path.dirname(__file__), "0006_walker_nge1_and_H.py"))
p6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p6)
D, N, M, PHI, SS, LAM, RHO, H, HV = p6.D, p6.N, p6.M, p6.PHI, p6.SS, p6.LAM, p6.RHO, p6.H, p6.HV
Q_of, R_of, exact_grid = p6.Q_of, p6.R_of, p6.exact_grid

np.set_printoptions(precision=3, suppress=True)
_GAP, _SPAN, _RIDGE = 1.5, 3.0, 1e-4


def gen(hot, amp, T=600, seed=1):
    rng = np.random.default_rng(seed); psi = np.zeros((T, D))
    if hot is not None:
        psi[T // 3: 2 * T // 3, hot] = amp
    th = np.zeros(N); Y = np.zeros((T, M))
    for t in range(T):
        LQ = np.linalg.cholesky(Q_of(psi[t, :N]) + 1e-12 * np.eye(N))
        th = th + LQ @ rng.standard_normal(N)
        Y[t] = H @ th + np.sqrt(np.diag(R_of(psi[t, N:]))) * rng.standard_normal(M)
    return Y


def scale_vec(mu, wmean):
    return mu + wmean            # absolute per-axis scale estimate


def Ichar():
    P = np.eye(N) * (LAM.max() + RHO.max()); Q0 = Q_of(np.zeros(N)); R0 = R_of(np.zeros(M))
    for _ in range(400):
        Pp = P + Q0; S = H @ Pp @ H.T + R0; K = Pp @ H.T @ np.linalg.inv(S); P = Pp - K @ H @ Pp
    Pp = P + Q0; S = H @ Pp @ H.T + R0; Si = np.linalg.inv(S)
    out = []
    for k in range(N):
        hv = HV[:, k]; dS = LAM[k] * np.outer(hv, hv); out.append(0.5 * np.trace(Si @ dS @ Si @ dS))
    for i in range(M):
        E = np.zeros((M, M)); E[i, i] = RHO[i]; out.append(0.5 * np.trace(Si @ E @ Si @ E))
    return np.array(out)


def star_walk(Y):
    """Linear-cost STAR-GPB1: state collapsed over the union-of-per-axis-windows star
    (D*nodes KF evals), which restores the joint de-mixing the point-estimate MF lost.
    The same per-node likelihoods drive the per-axis scale marginals."""
    Ich = Ichar(); Ifloor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2); active = Ich >= Ifloor
    Kstar = (1 - PHI) / 4.0; qmu = Kstar ** 2 / (Ich * (1 - Kstar))
    Kn = int(math.ceil(_SPAN / _GAP)); off = _GAP * SS[0] * np.arange(-Kn, Kn + 1)
    w0 = np.exp(-0.5 * (off / SS[0]) ** 2); w0 /= w0.sum()
    nu = max(SS[0] ** 2 * (1 - PHI[0] ** 2), 1e-12)
    Tw = np.exp(np.clip(-0.5 * (off[None, :] - PHI[0] * off[:, None]) ** 2 / nu, -700, 700)); Tw /= Tw.sum(1, keepdims=True)
    nnode = off.size; idx = np.where(active)[0]
    mu = np.zeros(D); Pmu = SS ** 2; pis = [w0.copy() for _ in range(D)]
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    out = np.zeros((len(Y), D)); evals = 0
    for t, y in enumerate(Y):
        for k in idx:
            pis[k] = pis[k] @ Tw
        wmean = np.array([float(pis[k] @ off) if active[k] else 0.0 for k in range(D)])
        base = mu + wmean; e = y - H @ m
        # per-axis window: KF at each node (axis k varied, others at mean) -> ll + posterior
        ll_ax = {}; m_ax = {}; P_ax = {}
        for k in idx:
            sc = base.copy(); lls = np.empty(nnode); ms = np.empty((nnode, N)); Ps = np.empty((nnode, N, N))
            for j in range(nnode):
                sc[k] = mu[k] + off[j]
                Pp = P + Q_of(sc[:N]); Sn = H @ Pp @ H.T + R_of(sc[N:]) + 1e-9 * np.eye(M)
                Si = np.linalg.inv(Sn); sgn, ld = np.linalg.slogdet(Sn)
                lls[j] = -0.5 * (ld + float(e @ Si @ e))
                Kk = Pp @ H.T @ Si; ms[j] = m + Kk @ e; Ps[j] = Pp - Kk @ H @ Pp
                evals += 1
            ll_ax[k] = lls; m_ax[k] = ms; P_ax[k] = Ps
            # scale marginal + finding-18 walk
            w = pis[k] * np.exp(lls - lls.max()); pis[k] = w / w.sum()
            wm = float(pis[k] @ off); info = float(Ich[k]) + _RIDGE
            K_mu = Pmu[k] / (Pmu[k] + 1.0 / info)
            mu[k] += float(np.clip(K_mu * wm, -_GAP * SS[k], _GAP * SS[k])); Pmu[k] = (1 - K_mu) * Pmu[k] + qmu[k]
        # STATE: GPB1 over the star (all per-axis window nodes), weight = prior*lik
        Ms = []; Ps = []; Ws = []
        for k in idx:
            w = pis[k] * np.exp(ll_ax[k] - ll_ax[k].max())
            Ms.append(m_ax[k]); Ps.append(P_ax[k]); Ws.append(w)
        Ms = np.concatenate(Ms); Ps = np.concatenate(Ps); Ws = np.concatenate(Ws); Ws /= Ws.sum()
        m = Ws @ Ms; dm = Ms - m
        P = np.einsum("g,gij->ij", Ws, Ps) + np.einsum("g,gi,gj->ij", Ws, dm, dm); P = 0.5 * (P + P.T)
        out[t] = mu + np.array([float(pis[k] @ off) if active[k] else 0.0 for k in range(D)])
    return out, evals, active, nnode


def meanfield_iter_walk(Y, sweeps=4):
    """Mean-field with SWEEPS: iterate the per-axis marginal updates to their joint
    fixed point each step (coordinate ascent), so the ambiguous process/measurement
    variance is not double-counted.  Still linear: sweeps * D * nodes."""
    Ich = Ichar(); Ifloor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2); active = Ich >= Ifloor
    Kstar = (1 - PHI) / 4.0; qmu = Kstar ** 2 / (Ich * (1 - Kstar))
    Kn = int(math.ceil(_SPAN / _GAP)); off = _GAP * SS[0] * np.arange(-Kn, Kn + 1)
    w0 = np.exp(-0.5 * (off / SS[0]) ** 2); w0 /= w0.sum()
    nu = max(SS[0] ** 2 * (1 - PHI[0] ** 2), 1e-12)
    Tw = np.exp(np.clip(-0.5 * (off[None, :] - PHI[0] * off[:, None]) ** 2 / nu, -700, 700)); Tw /= Tw.sum(1, keepdims=True)
    nnode = off.size; idx = np.where(active)[0]
    mu = np.zeros(D); Pmu = SS ** 2; pis = [w0.copy() for _ in range(D)]
    prior = [w0.copy() for _ in range(D)]
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    out = np.zeros((len(Y), D))
    for t, y in enumerate(Y):
        for k in idx:
            pis[k] = pis[k] @ Tw; prior[k] = pis[k].copy()
        e = y - H @ m
        for _sw in range(sweeps):
            for k in idx:
                wmean = np.array([float(pis[j] @ off) if active[j] else 0.0 for j in range(D)])
                sc = mu + wmean; lls = np.empty(nnode)
                for j in range(nnode):
                    sc[k] = mu[k] + off[j]
                    Sn = H @ (P + Q_of(sc[:N])) @ H.T + R_of(sc[N:]) + 1e-9 * np.eye(M)
                    Si = np.linalg.inv(Sn); sgn, ld = np.linalg.slogdet(Sn); lls[j] = -0.5 * (ld + float(e @ Si @ e))
                w = prior[k] * np.exp(lls - lls.max()); pis[k] = w / w.sum()
        for k in idx:
            wm = float(pis[k] @ off); info = float(Ich[k]) + _RIDGE
            K_mu = Pmu[k] / (Pmu[k] + 1.0 / info)
            mu[k] += float(np.clip(K_mu * wm, -_GAP * SS[k], _GAP * SS[k])); Pmu[k] = (1 - K_mu) * Pmu[k] + qmu[k]
        wmean = np.array([float(pis[k] @ off) if active[k] else 0.0 for k in range(D)]); sbar = mu + wmean
        Pp = P + Q_of(sbar[:N]); Sb = H @ Pp @ H.T + R_of(sbar[N:]) + 1e-9 * np.eye(M)
        Kk = Pp @ H.T @ np.linalg.inv(Sb); m = m + Kk @ e; P = Pp - Kk @ H @ Pp; P = 0.5 * (P + P.T)
        out[t] = sbar
    return out, sweeps * len(idx) * nnode, active, nnode


def meanfield_walk(Y):
    Ich = Ichar()
    Ifloor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2)
    active = Ich >= Ifloor
    Kstar = (1 - PHI) / 4.0; qmu = Kstar ** 2 / (Ich * (1 - Kstar))
    Kn = int(math.ceil(_SPAN / _GAP)); off = _GAP * SS[0] * np.arange(-Kn, Kn + 1)
    w0 = np.exp(-0.5 * (off / SS[0]) ** 2); w0 /= w0.sum()
    nu = max(SS[0] ** 2 * (1 - PHI[0] ** 2), 1e-12)
    Tw = np.exp(np.clip(-0.5 * (off[None, :] - PHI[0] * off[:, None]) ** 2 / nu, -700, 700)); Tw /= Tw.sum(1, keepdims=True)
    nnode = off.size
    mu = np.zeros(D); Pmu = SS ** 2
    pis = [w0.copy() for _ in range(D)]                 # per-axis marginals
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    out = np.zeros((len(Y), D)); evals = 0
    for t, y in enumerate(Y):
        for k in range(D):
            if active[k]:
                pis[k] = pis[k] @ Tw
        wmean = np.array([float(pis[k] @ off) if active[k] else 0.0 for k in range(D)])
        base = scale_vec(mu, wmean)                     # current mean scale
        e = y - H @ m
        # --- scale: mean-field per-axis window update + finding-18 walk
        for k in range(D):
            if not active[k]:
                continue
            sc = base.copy()
            # vary axis k over its window (others at mean); build S per node
            lg = np.empty(nnode); gS_list = np.empty(nnode); e2S = np.empty(nnode)
            for j in range(nnode):
                sc[k] = mu[k] + off[j]
                Qn = Q_of(sc[:N]); Rn = R_of(sc[N:]); Sn = H @ (P + Qn) @ H.T + Rn + 1e-9 * np.eye(M)
                Si = np.linalg.inv(Sn); sgn, ld = np.linalg.slogdet(Sn); mh = float(e @ Si @ e)
                lg[j] = -0.5 * (ld + mh)
                evals += 1
            sc[k] = base[k]
            w = pis[k] * np.exp(lg - lg.max()); Z = w.sum(); pis[k] = w / Z
            # finding-18 walk: shift toward the window's posterior mean, Kalman gain by info
            wm = float(pis[k] @ off)
            info = float(Ich[k]) + _RIDGE
            R_mu = 1.0 / info; K_mu = Pmu[k] / (Pmu[k] + R_mu)
            mu[k] += float(np.clip(K_mu * wm, -_GAP * SS[k], _GAP * SS[k]))
            Pmu[k] = (1.0 - K_mu) * Pmu[k] + qmu[k]
        # --- state: one KF at the posterior-mean scale
        wmean = np.array([float(pis[k] @ off) if active[k] else 0.0 for k in range(D)])
        sbar = scale_vec(mu, wmean)
        Qb = Q_of(sbar[:N]); Rb = R_of(sbar[N:]); Sb = H @ (P + Qb) @ H.T + Rb + 1e-9 * np.eye(M)
        Kk = (P + Qb) @ H.T @ np.linalg.inv(Sb); m = m + Kk @ e; P = (P + Qb) - Kk @ H @ (P + Qb); P = 0.5 * (P + P.T)
        out[t] = sbar
    return out, evals, active, nnode


if __name__ == "__main__":
    T = 600; b = slice(T // 3 + 40, 2 * T // 3 - 40); lab = ["xi1", "xi2", "eta1", "eta2"]
    _, evals, active, nnode = meanfield_walk(gen(None, 0.0, 60))
    n_active = int(active.sum())
    print(f"active axes: {n_active} of {D}; window nodes/axis: {nnode}")
    print(f"mean-field evals/step ~ D*nodes = {n_active}*{nnode} = {n_active*nnode}   "
          f"(exact tensor grid would be nodes^active = {nnode}**{n_active} = {nnode**n_active})")
    print()
    for name, fn in (("mean-field (point state)", meanfield_walk), ("STAR-GPB1 state", star_walk)):
        print(f"\n=== {name} ===")
        print("  STATIC:", fn(gen(None, 0.0, T))[0][150:].mean(0))
        for ax, nm in [(1, "xi2 hot"), (3, "eta2 hot")]:
            Y = gen(ax, 1.4, T); ref = exact_grid(Y, 5); w, _, _, _ = fn(Y)
            cr = [np.corrcoef(w[30:, k], ref[30:, k])[0, 1] if w[30:, k].std() > 1e-9 else np.nan for k in range(D)]
            print(f"  {nm}: GRID {ref[b].mean(0)}  W {w[b].mean(0)}  corr {np.array(cr)}")
