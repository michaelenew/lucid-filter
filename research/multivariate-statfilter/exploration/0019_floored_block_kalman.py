"""Probe 0019 -- block-Kalman with a SPECTRALLY-FLOORED Fisher (route 1, stable form).

0015 block-Kalman already de-mixes better than the exact grid vs the TRUE scales, with no
attribution flips, because it tracks Pmu SMOOTHLY as a full matrix.  0018 tried an explicit
eigenbasis caltrop and FLIPPED on ambiguous seeds -- not from the step (a matrix function of
F is sign-invariant) but from re-diagonalising Pmu in the rotating frame (lossy when Pmu and
F don't commute).

So keep the block-Kalman's smooth matrix Pmu, and get 0018's benefit -- freezing the sloppy
eigendirections where the leak lives -- by flooring F in the STEP only:
    F = U Lam U^T ;  Rmu = U diag( 1/lam_j if lam_j>=floor else 0 ) U^T   (floored pseudo-inv)
    offset = Rmu grad ;  K = Pmu (Pmu + Rmu)^-1 ;  mu += K offset
Sub-floor directions get zero inverse -> zero contribution to the step -> frozen, exactly the
derived identifiability truncation (floor = (1-phi)/(4 (SPAN s)^2)), now per eigendirection of
the full Fisher.  Pmu stays a smooth full matrix (no re-diagonalisation), so no flips.
Estimate mu is physical -> read "which sensor / mode is hot" directly.  Cost O(r^3).
"""
import os
import sys
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
_SPAN = 3.0


def score_and_F(mu, m, P, y, act):
    e = y - H @ m
    S = H @ (P + Q_of(mu[:N])) @ H.T + R_of(mu[N:]) + 1e-9 * np.eye(M)
    Si = np.linalg.inv(S); Sie = Si @ e
    dS = [dS_k(mu, int(k)) for k in act]
    SidS = [Si @ d for d in dS]
    grad = np.array([0.5 * (Sie @ dS[a] @ Sie - np.trace(SidS[a])) for a in range(len(act))])
    F = np.array([[0.5 * np.trace(SidS[a] @ SidS[b]) for b in range(len(act))] for a in range(len(act))])
    return grad, 0.5 * (F + F.T), e


def floored_block(Y):
    Ich = Ichar(); floor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2)
    act = np.where(Ich >= floor)[0]; r = len(act)
    gap = 1.5 * SS[0]; Kstar = (1 - PHI[0]) / 4.0
    qmu = np.array([Kstar ** 2 / (Ich[int(k)] * (1 - Kstar)) for k in act])
    mu = np.zeros(D); Pmu = np.diag(SS[act] ** 2).astype(float)
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    out = np.zeros((len(Y), D)); state = np.zeros((len(Y), N))
    for t, y in enumerate(Y):
        grad, F, e = score_and_F(mu, m, P, y, act)
        lam, U = np.linalg.eigh(F)
        inv = np.where(lam >= floor, 1.0 / np.maximum(lam, 1e-30), 0.0)   # floored pseudo-inverse
        Rmu = U @ np.diag(inv) @ U.T                                      # obs cov of Newton est, sloppy dirs frozen
        offset = Rmu @ grad
        K = Pmu @ np.linalg.inv(Pmu + Rmu)                               # smooth matrix finding-18 gain
        step = np.clip(K @ offset, -gap, gap)
        mu[act] += step
        Pmu = (np.eye(r) - K) @ Pmu + np.diag(qmu); Pmu = 0.5 * (Pmu + Pmu.T)
        Pp = P + Q_of(mu[:N]); Sb = H @ Pp @ H.T + R_of(mu[N:]) + 1e-9 * np.eye(M)
        Kk = Pp @ H.T @ np.linalg.inv(Sb); m = m + Kk @ e; P = Pp - Kk @ H @ Pp; P = 0.5 * (P + P.T)
        out[t] = mu; state[t] = m
    return out, state, r


def hammer():
    T = 600; b = slice(T // 3 + 40, 2 * T // 3 - 40)
    _, _, r = floored_block(gen(None, 0.0, 60))
    print(f"active r={r}; spectrally-floored block-Kalman, O(r^3), no grid\n")
    for hot, nm in [(1, "xi2 hot "), (3, "eta2 hot")]:
        tru = np.zeros(D); tru[hot] = 1.4
        ee = []; eg = []; drift = []
        for seed in range(6):
            st = floored_block(gen(None, 0.0, T, seed))[0][150:].mean(0)
            drift.append(np.abs(st[[1, 2, 3]]).max())
            Y = gen(hot, 1.4, T, seed)
            we = floored_block(Y)[0][b].mean(0); wg = exact_grid(Y, 5)[b].mean(0)
            ee.append(np.abs(we - tru)); eg.append(np.abs(wg - tru))
        ee = np.mean(ee, 0); eg = np.mean(eg, 0)
        print(f" {nm}: |FLOOR-truth| {ee}  sum {ee.sum():.2f}   (static drift max {max(drift):.2f})")
        print(f" {nm}: |GRID -truth| {eg}  sum {eg.sum():.2f}")
    print("\n per-seed (mixing H):")
    for seed in range(6):
        Yx = gen(1, 1.4, T, seed); wx = floored_block(Yx)[0]; gx = exact_grid(Yx, 5)
        Ye = gen(3, 1.4, T, seed); we = floored_block(Ye)[0]; ge = exact_grid(Ye, 5)
        print(f"  s{seed}: xi2hot eta2={wx[b,3].mean():+.2f}(g{gx[b,3].mean():+.2f}) | "
              f"eta2hot eta2={we[b,3].mean():.2f}(g{ge[b,3].mean():.2f}) eta1={we[b,2].mean():+.2f}(g{ge[b,2].mean():+.2f})")


if __name__ == "__main__":
    hammer()
