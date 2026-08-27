"""Probe 0015 -- de-mix by a BLOCK (matrix) finding-18 walk on the full Fisher.

The caltrop leak is the per-axis walk ignoring the process<->measurement cross-Fisher.
The full expected Fisher over the active axes,
    F_kl = 0.5 tr(S^-1 dS_k S^-1 dS_l),
has exactly those off-diagonals; its inverse de-mixes.  Raw F^-1 grad is unstable
(0011) because F^-1 amplifies sloppy directions -- but a MATRIX finding-18 Kalman gain
    K = Pmu (Pmu + F^-1)^-1
damps them the way the scalar K_mu = Pmu/(Pmu+1/info) does.  F is ANALYTIC (the score
and Fisher need only S, e, and the dS_k at mu -- O(1), no arms).  The user's one-hot +
two-hot + N-1-hot arms are how one would MEASURE the same F if it weren't analytic;
this probe checks whether the analytic block-Kalman de-mixes the attribution to
grid-parity while staying stable.  Reduces to the scalar finding-18 when the block is 1x1.
"""
import os
import sys
import math
import importlib.util

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
_spec = importlib.util.spec_from_file_location(
    "p13", os.path.join(os.path.dirname(__file__), "0013_caltrop_walker.py"))
p13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p13)
D, N, M, PHI, SS, LAM, RHO, H, HV = p13.D, p13.N, p13.M, p13.PHI, p13.SS, p13.LAM, p13.RHO, p13.H, p13.HV
Q_of, R_of, exact_grid, gen, Ichar, dS_k = p13.Q_of, p13.R_of, p13.exact_grid, p13.gen, p13.Ichar, p13.dS_k

np.set_printoptions(precision=3, suppress=True)
_SPAN, _RIDGE = 3.0, 1e-4


def score_and_F(mu, m, P, y, act):
    """Analytic score vector and full expected Fisher over the active axes at mu."""
    e = y - H @ m
    S = H @ (P + Q_of(mu[:N])) @ H.T + R_of(mu[N:]) + 1e-9 * np.eye(M)
    Si = np.linalg.inv(S); Sie = Si @ e
    r = len(act)
    dS = [dS_k(mu, int(k)) for k in act]
    SidS = [Si @ d for d in dS]
    grad = np.array([0.5 * (Sie @ dS[a] @ Sie - np.trace(SidS[a])) for a in range(r)])
    F = np.array([[0.5 * np.trace(SidS[a] @ SidS[b]) for b in range(r)] for a in range(r)])
    return grad, F, e


def block_kalman(Y, cap_cells=1.0):
    Ich = Ichar(); Ifloor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2)
    act = np.where(Ich >= Ifloor)[0]; r = len(act)
    gap = 1.5 * SS[0]
    Kstar = (1 - PHI[0]) / 4.0
    qmu = np.array([Kstar ** 2 / (Ich[int(k)] * (1 - Kstar)) for k in act])   # per-axis derived drift
    mu = np.zeros(D); Pmu = np.diag(SS[act] ** 2).astype(float)               # block scale covariance
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    out = np.zeros((len(Y), D)); state = np.zeros((len(Y), N))
    I_r = np.eye(r)
    for t, y in enumerate(Y):
        grad, F, e = score_and_F(mu, m, P, y, act)
        Rmu = np.linalg.inv(F + _RIDGE * I_r)             # obs cov of the Newton estimate (= F^-1)
        offset = Rmu @ grad                              # Newton step delta = F^-1 grad (de-mixed)
        K = Pmu @ np.linalg.inv(Pmu + Rmu)               # MATRIX finding-18 gain (damps sloppy dirs)
        step = K @ offset
        step = np.clip(step, -cap_cells * gap, cap_cells * gap)
        mu[act] += step
        Pmu = (I_r - K) @ Pmu + np.diag(qmu); Pmu = 0.5 * (Pmu + Pmu.T)
        # state KF at the de-mixed origin
        Pp = P + Q_of(mu[:N]); Sb = H @ Pp @ H.T + R_of(mu[N:]) + 1e-9 * np.eye(M)
        Kk = Pp @ H.T @ np.linalg.inv(Sb); m = m + Kk @ e; P = Pp - Kk @ H @ Pp; P = 0.5 * (P + P.T)
        out[t] = mu; state[t] = m
    return out, state, r


if __name__ == "__main__":
    T = 600; b = slice(T // 3 + 40, 2 * T // 3 - 40); lab = ["xi1", "xi2", "eta1", "eta2"]
    _, _, r = block_kalman(gen(None, 0.0, 60))
    print(f"active r={r}; block-Kalman cost O(r^2..r^3), analytic F (no arms)")
    for seed in (1, 2, 3):
        st = block_kalman(gen(None, 0.0, T, seed))[0][150:].mean(0)
        Y = gen(1, 1.4, T, seed); ref = exact_grid(Y, 5); wx = block_kalman(Y)[0]
        Y2 = gen(3, 1.4, T, seed); ref2 = exact_grid(Y2, 5); we = block_kalman(Y2)[0]
        print(f"seed{seed} STATIC {st}")
        print(f"   xi2 hot : GRID {ref[b].mean(0)}  BLOCK {wx[b].mean(0)}")
        print(f"   eta2 hot: GRID {ref2[b].mean(0)}  BLOCK {we[b].mean(0)}")
