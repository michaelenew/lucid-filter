"""Probe 0018 -- the Fisher-eigenbasis caltrop (route 1, built right).

0015 showed the block-Kalman (matrix finding-18 on the full analytic Fisher) already
de-mixes BETTER than the exact grid when judged against the TRUE injected scales -- the
grid only looked better because it is span-capped and under-reaches.  But the block-Kalman
uses F^-1 grad (a Newton step): it moves in EVERY direction, sloppy ones included, only
damped by the matrix gain.  The residual leak lives in those sloppy directions.

Route 1 done cleanly: diagonalise the analytic scale Fisher  F = U Lam U^T  each step and
run the VALIDATED scalar caltrop (finding-18) INDEPENDENTLY along each eigendirection --
    * eigendirections are Fisher-decoupled, so the per-direction walk cannot leak into
      the others (that is the whole point of the eigenbasis);
    * critically-damped finding-18 (K* = (1-phi)/4) reaches the hot direction without the
      Newton over-reach;
    * FREEZE any eigendirection whose info lam_j is below the derived identifiability floor
      (1-phi)/(4 (SPAN s)^2) -- the same no-free-parameter spectral truncation as the
      caltrop, now applied per eigendirection.  The leak lives in the sloppy directions;
      freezing them (rather than damping via F^-1) is what removes it.
The estimate mu stays in PHYSICAL (xi_k, eta_i) coordinates -- U only chooses the walk
directions -- so "which sensor / which mode is hot" is read directly off mu.  The frame U
drifts each step (F depends on state and scales): a rotating-frame construction, tracked by
carrying Pmu as a physical matrix and re-diagonalising it in the current U each step.

Cost: one r x r eigendecomposition + r scalar walks per step = O(r^3) worst, O(r^2) typical
-- polynomial (the "settle for quadratic"), no exponential grid.
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
_SPAN, _RIDGE = 3.0, 1e-9


def score_and_F(mu, m, P, y, act):
    """Analytic score vector and full expected Fisher over the active axes at mu."""
    e = y - H @ m
    S = H @ (P + Q_of(mu[:N])) @ H.T + R_of(mu[N:]) + 1e-9 * np.eye(M)
    Si = np.linalg.inv(S); Sie = Si @ e
    dS = [dS_k(mu, int(k)) for k in act]
    SidS = [Si @ d for d in dS]
    grad = np.array([0.5 * (Sie @ dS[a] @ Sie - np.trace(SidS[a])) for a in range(len(act))])
    F = np.array([[0.5 * np.trace(SidS[a] @ SidS[b]) for b in range(len(act))] for a in range(len(act))])
    return grad, 0.5 * (F + F.T), e


def eigen_caltrop(Y):
    Ich = Ichar(); floor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2)
    act = np.where(Ich >= floor)[0]; r = len(act)
    gap = 1.5 * SS[0]; Kstar = (1 - PHI[0]) / 4.0
    mu = np.zeros(D); Pmu = np.diag(SS[act] ** 2).astype(float)     # physical scale covariance
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    out = np.zeros((len(Y), D)); state = np.zeros((len(Y), N))
    for t, y in enumerate(Y):
        grad, F, e = score_and_F(mu, m, P, y, act)
        lam, U = np.linalg.eigh(F)                     # F = U diag(lam) U^T (lam ascending)
        g = U.T @ grad                                 # score in the eigenbasis (decoupled)
        Pc = U.T @ Pmu @ U                             # walk covariance, current frame
        c = np.zeros(r); pcn = np.diag(Pc).copy()
        for j in range(r):
            if lam[j] < floor:                         # sloppy direction -> freeze (the leak lives here)
                continue
            Kj = pcn[j] / (pcn[j] + 1.0 / lam[j])      # scalar finding-18 gain along u_j
            c[j] = np.clip(Kj * g[j] / lam[j], -gap, gap)
            pcn[j] = (1 - Kj) * pcn[j] + Kstar ** 2 / (lam[j] * (1 - Kstar))
        mu[act] += U @ c
        Pmu = U @ np.diag(pcn) @ U.T; Pmu = 0.5 * (Pmu + Pmu.T)
        # state KF at the de-mixed origin
        Pp = P + Q_of(mu[:N]); Sb = H @ Pp @ H.T + R_of(mu[N:]) + 1e-9 * np.eye(M)
        Kk = Pp @ H.T @ np.linalg.inv(Sb); m = m + Kk @ e; P = Pp - Kk @ H @ Pp; P = 0.5 * (P + P.T)
        out[t] = mu; state[t] = m
    return out, state, r


def hammer():
    T = 600; b = slice(T // 3 + 40, 2 * T // 3 - 40); lab = ["xi1", "xi2", "eta1", "eta2"]
    _, _, r = eigen_caltrop(gen(None, 0.0, 60))
    print(f"active r={r}; Fisher-eigenbasis caltrop, O(r^2..r^3), no grid\n")
    # error vs the TRUE injected scale (hot axis = 1.4 in band, others 0)
    for hot, nm in [(1, "xi2 hot "), (3, "eta2 hot")]:
        tru = np.zeros(D); tru[hot] = 1.4
        ee = []; eg = []; drift = []
        for seed in range(6):
            st = eigen_caltrop(gen(None, 0.0, T, seed))[0][150:].mean(0)
            drift.append(np.abs(st[[1, 2, 3]]).max())
            Y = gen(hot, 1.4, T, seed)
            we = eigen_caltrop(Y)[0][b].mean(0); wg = exact_grid(Y, 5)[b].mean(0)
            ee.append(np.abs(we - tru)); eg.append(np.abs(wg - tru))
        ee = np.mean(ee, 0); eg = np.mean(eg, 0)
        print(f" {nm}: |EIGEN-truth| {ee}  sum {ee.sum():.2f}   (static drift max {max(drift):.2f})")
        print(f" {nm}: |GRID -truth| {eg}  sum {eg.sum():.2f}")
    # per-seed leak table (sensor read when a process mode is hot; clean sensor when the other is hot)
    print("\n per-seed (mixing H):")
    for seed in range(6):
        Yx = gen(1, 1.4, T, seed); wx = eigen_caltrop(Yx)[0]; gx = exact_grid(Yx, 5)
        Ye = gen(3, 1.4, T, seed); we = eigen_caltrop(Ye)[0]; ge = exact_grid(Ye, 5)
        print(f"  s{seed}: xi2hot eta2={wx[b,3].mean():+.2f}(g{gx[b,3].mean():+.2f}) | "
              f"eta2hot eta2={we[b,3].mean():.2f}(g{ge[b,3].mean():.2f}) eta1={we[b,2].mean():+.2f}(g{ge[b,2].mean():+.2f})")


if __name__ == "__main__":
    hammer()
