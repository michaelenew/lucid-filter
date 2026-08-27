"""Probe 0020 -- the caltrop along the Fisher eigendirections (route 1, final form).

Synthesis of 0015/0018/0019:
  * 0015: the full-Fisher walk de-mixes better than the grid vs TRUE scales -- the eigenbasis
    is the right frame (it diagonalises the process<->measurement AND sensor<->sensor coupling
    a mixing H induces).
  * 0018: an explicit per-direction walk FLIPPED on ambiguous seeds -- but only because it
    tracked Pmu as a matrix and re-diagonalised it in the rotating frame (lossy).
  * finding-18: the steady-state walk gain is a CONSTANT  K* = (1-phi)/4, independent of the
    info -- so we need not track Pmu at all.  That removes the rotating-frame bookkeeping and
    with it the flip.
  * 0019: the residual (seed1 process->sensor over-commit) lives in the ONE low-lambda active
    eigendirection; a local Newton step over-commits there.

Final: each step, diagonalise the analytic scale Fisher F = U Lam U^T and run the validated
caltrop axial profile ALONG each active eigendirection u_j -- evaluate the FULL likelihood at
offsets, take the profile peak (not the local gradient), and step K* toward it.  Because the
eigendirections are Fisher-decoupled, the per-direction profiles don't leak into each other
(the point of the eigenbasis); because we use the full likelihood (not local curvature), the
sloppy direction is HEDGED -- the profile is broad and its peak sits at the true partial
attribution, not the Newton over-shoot.  Freeze sub-floor directions (derived floor).  No Pmu
matrix, no free parameter.  mu stays physical -> read which sensor/mode is hot directly.
Cost: one r x r eigendecomposition + r x (grid) profiles per step = O(r^3) worst, linear grid.
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


def _loglik(mu, m, P, y):
    S = H @ (P + Q_of(mu[:N])) @ H.T + R_of(mu[N:]) + 1e-9 * np.eye(M)
    return -0.5 * (np.linalg.slogdet(S)[1] + float((y - H @ m) @ np.linalg.inv(S) @ (y - H @ m)))


def eigen_caltrop(Y):
    Ich = Ichar(); floor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2)
    act = np.where(Ich >= floor)[0]; r = len(act)
    gap = 1.5 * SS[0]; Kstar = (1 - PHI[0]) / 4.0
    off = gap * np.arange(-2, 3); no = off.size                      # axial profile offsets along u_j
    mu = np.zeros(D)
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    out = np.zeros((len(Y), D)); state = np.zeros((len(Y), N))
    for t, y in enumerate(Y):
        grad, F, e = score_and_F(mu, m, P, y, act)
        lam, U = np.linalg.eigh(F)
        c = np.zeros(r)
        for j in range(r):
            if lam[j] < floor:                                       # sloppy -> freeze
                continue
            uj = np.zeros(D); uj[act] = U[:, j]
            prof = np.array([_loglik(mu + o * uj, m, P, y) for o in off])   # FULL likelihood along u_j
            w = np.exp(prof - prof.max()); w /= w.sum()              # no stationary prior (avoid shrink)
            c[j] = Kstar * float(w @ off)                           # step K* toward the profile peak
        mu[act] += np.clip(U @ c, -gap, gap)
        # state KF at the de-mixed origin
        Pp = P + Q_of(mu[:N]); Sb = H @ Pp @ H.T + R_of(mu[N:]) + 1e-9 * np.eye(M)
        Kk = Pp @ H.T @ np.linalg.inv(Sb); m = m + Kk @ e; P = Pp - Kk @ H @ Pp; P = 0.5 * (P + P.T)
        out[t] = mu; state[t] = m
    return out, state, r


def hammer():
    T = 600; b = slice(T // 3 + 40, 2 * T // 3 - 40)
    _, _, r = eigen_caltrop(gen(None, 0.0, 60))
    print(f"active r={r}; caltrop along Fisher eigendirections, O(r^3)+linear grid, no Pmu\n")
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
        print(f" {nm}: |EIGCAL-truth| {ee}  sum {ee.sum():.2f}   (static drift max {max(drift):.2f})")
        print(f" {nm}: |GRID  -truth| {eg}  sum {eg.sum():.2f}")
    print("\n per-seed (mixing H):")
    for seed in range(6):
        Yx = gen(1, 1.4, T, seed); wx = eigen_caltrop(Yx)[0]; gx = exact_grid(Yx, 5)
        Ye = gen(3, 1.4, T, seed); we = eigen_caltrop(Ye)[0]; ge = exact_grid(Ye, 5)
        print(f"  s{seed}: xi2hot eta2={wx[b,3].mean():+.2f}(g{gx[b,3].mean():+.2f}) | "
              f"eta2hot eta2={we[b,3].mean():.2f}(g{ge[b,3].mean():.2f}) eta1={we[b,2].mean():+.2f}(g{ge[b,2].mean():+.2f})")


if __name__ == "__main__":
    hammer()
