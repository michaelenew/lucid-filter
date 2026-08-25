"""Probe 0013 -- the CALTROP: axial-only grid points, walked to the joint truth.

The user's proposal, and the escape from 0011/0012's exponential wall. Instead of the
tensor window (nodes^r) or a factored posterior (loses the coupling), evaluate the
likelihood only along the AXES from the current caltrop origin mu:
    the origin mu, plus 2K points per axis (mu +/- k*gap * e_axis), others held at mu.
Cost 1 + r*2K -- LINEAR in the axes.  The filter WALKS mu (finding-18 loop per axis)
until every axis's axial profile is centred -- i.e. until the origin sits at the joint
truth.  It does not represent the joint density (no corners); it LOCATES its peak by
coordinate walking.  In the eigenbasis the directions are locally linearised and
well-conditioned, so the axial profiles should give clean per-axis direction and the
origin should converge to the joint peak -- faithful, unlike the marginal-mean of
mean-field (which double-counts).  The axial span can be large (it's just resolution;
the walk supplies unbounded reach), which also lets each axis see a far regime quickly.

State: one KF at the origin mu.  Reported scale: mu (the located joint peak).
Benchmarked vs the exact tensor grid (0006).
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
_K = 4                        # axial half-extent in cells (span +- _K*gap; the walk does the rest)


def gen(hot, amp, T=600, seed=1):
    rng = np.random.default_rng(seed); psi = np.zeros((T, D))
    if hot is not None:
        psi[T // 3: 2 * T // 3, hot] = amp
    th = np.zeros(N); Y = np.zeros((T, M))
    for t in range(T):
        th = th + np.linalg.cholesky(Q_of(psi[t, :N]) + 1e-12 * np.eye(N)) @ rng.standard_normal(N)
        Y[t] = H @ th + np.sqrt(np.diag(R_of(psi[t, N:]))) * rng.standard_normal(M)
    return Y


def Ichar():
    P = np.eye(N) * (LAM.max() + RHO.max()); Q0 = Q_of(np.zeros(N)); R0 = R_of(np.zeros(M))
    for _ in range(400):
        Pp = P + Q0; S = H @ Pp @ H.T + R0; K = Pp @ H.T @ np.linalg.inv(S); P = Pp - K @ H @ Pp
    Pp = P + Q0; S = H @ Pp @ H.T + R0; Si = np.linalg.inv(S)
    o = [0.5 * np.trace(Si @ (LAM[k] * np.outer(HV[:, k], HV[:, k])) @ Si @ (LAM[k] * np.outer(HV[:, k], HV[:, k]))) for k in range(N)]
    for i in range(M):
        E = np.zeros((M, M)); E[i, i] = RHO[i]; o.append(0.5 * np.trace(Si @ E @ Si @ E))
    return np.array(o)


def dS_k(sc, k):
    if k < N:
        hv = HV[:, k]; return LAM[k] * math.exp(min(sc[k], 60)) * np.outer(hv, hv)
    E = np.zeros((M, M)); E[k - N, k - N] = RHO[k - N] * math.exp(min(sc[k], 60)); return E


def score_info_at(sc, k, P, e):
    """Analytic grid-shift score and expected Fisher for axis k at scale vector sc."""
    S = H @ (P + Q_of(sc[:N])) @ H.T + R_of(sc[N:]) + 1e-9 * np.eye(M)
    Si = np.linalg.inv(S); d = dS_k(sc, k); Sie = Si @ e; SidS = Si @ d
    score = 0.5 * (Sie @ d @ Sie - np.trace(SidS))
    info = 0.5 * np.trace(SidS @ SidS)
    return score, info


def caltrop_walk(Y, sweeps=2):
    Ich = Ichar(); Ifloor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2); active = np.where(Ich >= Ifloor)[0]
    gap = _GAP * SS[0]; offs = gap * np.arange(-_K, _K + 1); nn = offs.size
    w0 = np.exp(-0.5 * (offs / SS[0]) ** 2); w0 /= w0.sum()
    Kstar = (1 - PHI) / 4.0; qmu = Kstar ** 2 / (Ich * (1 - Kstar))
    mu = np.zeros(D); Pmu = SS ** 2; m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    out = np.zeros((len(Y), D)); state = np.zeros((len(Y), N)); evals = 0
    for t, y in enumerate(Y):
        e = y - H @ m
        for _sw in range(sweeps):
            for k in active:
                base = mu.copy(); prof = np.empty(nn); sco = np.empty(nn); inf = np.empty(nn)
                for j in range(nn):
                    base[k] = mu[k] + offs[j]
                    prof[j] = -0.5 * (np.linalg.slogdet(H @ (P + Q_of(base[:N])) @ H.T + R_of(base[N:]) + 1e-9 * np.eye(M))[1]
                                      + float(e @ np.linalg.inv(H @ (P + Q_of(base[:N])) @ H.T + R_of(base[N:]) + 1e-9 * np.eye(M)) @ e))
                    sco[j], inf[j] = score_info_at(base, k, P, e); evals += 1
                pi = w0 * np.exp(prof - prof.max()); pi /= pi.sum()   # stationary-weighted (stability)
                grad = float(pi @ sco); info = float(pi @ inf) + _RIDGE   # gradient (reaches, zero-mean at truth)
                K_mu = Pmu[k] / (Pmu[k] + 1.0 / info)
                mu[k] += float(np.clip(K_mu * (grad / info), -gap, gap))
            # (Pmu updated once per observation, after the sweeps)
        for k in active:
            info = float(Ich[k]) + _RIDGE; K_mu = Pmu[k] / (Pmu[k] + 1.0 / info)
            Pmu[k] = (1 - K_mu) * Pmu[k] + qmu[k]
        # state: AXIAL-GPB1 -- collapse the KF over the caltrop star (centre + per-axis
        # points at the final mu), weighted by likelihood; restores the joint de-mixing
        # a point-estimate state lacks, at linear cost (the points are already the star).
        star = [mu.copy()]
        for k in active:
            for dd in (-1, 1):
                s = mu.copy(); s[k] = mu[k] + dd * gap; star.append(s)
        lls = np.empty(len(star)); ms = np.empty((len(star), N)); Ps = np.empty((len(star), N, N))
        for j, s in enumerate(star):
            Pp = P + Q_of(s[:N]); Sb = H @ Pp @ H.T + R_of(s[N:]) + 1e-9 * np.eye(M)
            Si = np.linalg.inv(Sb); sgn, ld = np.linalg.slogdet(Sb)
            lls[j] = -0.5 * (ld + float(e @ Si @ e))
            Kk = Pp @ H.T @ Si; ms[j] = m + Kk @ e; Ps[j] = Pp - Kk @ H @ Pp
        wj = np.exp(lls - lls.max()); wj /= wj.sum()
        m = wj @ ms; dmv = ms - m
        P = np.einsum("g,gij->ij", wj, Ps) + np.einsum("g,gi,gj->ij", wj, dmv, dmv); P = 0.5 * (P + P.T)
        out[t] = mu; state[t] = m
    return out, state, active.size, nn, evals // max(len(Y), 1)


if __name__ == "__main__":
    T = 600; b = slice(T // 3 + 40, 2 * T // 3 - 40); lab = ["xi1", "xi2", "eta1", "eta2"]
    _, _, r, nn, ev = caltrop_walk(gen(None, 0.0, 60))
    print(f"active axes r={r}; axial pts/axis={nn}; caltrop evals/step~{ev} "
          f"(sweeps*r*nn)  vs dense grid {5**r}")
    print("STATIC:", caltrop_walk(gen(None, 0.0, T), sweeps=4)[0][150:].mean(0))
    for ax, nm in [(1, "xi2 hot"), (3, "eta2 hot")]:
        Y = gen(ax, 1.4, T); ref = exact_grid(Y, 5); w, *_ = caltrop_walk(Y, sweeps=4)
        cr = [np.corrcoef(w[30:, k], ref[30:, k])[0, 1] if w[30:, k].std() > 1e-9 else np.nan for k in range(D)]
        print(f"{nm}: GRID {ref[b].mean(0)}  CALTROP {w[b].mean(0)}  corr {np.array(cr)}")
