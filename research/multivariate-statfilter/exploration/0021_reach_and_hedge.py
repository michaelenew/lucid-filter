"""Probe 0021 -- reach the stiff directions, hedge the sloppy one (route 1, resolved).

The tension made explicit by 0019 vs 0020:
  * 0019 (floored block-Kalman, smooth Pmu): REACHES the hot axis (Pmu inflation = finding-18
    reach) and never flips, but a local Newton step OVER-COMMITS the ambiguous seed -> leak.
  * 0020 (full-likelihood profile, no Pmu): the profile HEDGES -> leak ~0 on every seed, but
    the constant gain UNDER-REACHES every axis.

Key: a hot axis is a large signal, so its Fisher eigendirection is STIFF (high lambda) and
must be REACHED (Newton + Pmu).  The leak lives in the ONE low-lambda active eigendirection --
the process<->measurement ambiguity -- where the data genuinely can't decide; there,
under-reaching IS the correct hedge.  So run 0019's reaching walk, then OVERRIDE only the
lowest active eigendirection's component of the step with the full-likelihood profile hedge.
Stiff directions reach; the sloppy direction hedges.  No free parameter (the split is
lambda>=floor stiff-vs-sloppy by the ordering; only the single lowest active dir is hedged).
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


def reach_hedge(Y):
    Ich = Ichar(); floor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2)
    act = np.where(Ich >= floor)[0]; r = len(act)
    gap = 1.5 * SS[0]; Kstar = (1 - PHI[0]) / 4.0
    qmu = np.array([Kstar ** 2 / (Ich[int(k)] * (1 - Kstar)) for k in act])
    off = gap * np.arange(-2, 3)
    mu = np.zeros(D); Pmu = np.diag(SS[act] ** 2).astype(float)
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    out = np.zeros((len(Y), D)); state = np.zeros((len(Y), N))
    I_r = np.eye(r)
    for t, y in enumerate(Y):
        grad, F, e = score_and_F(mu, m, P, y, act)
        lam, U = np.linalg.eigh(F)                                   # ascending: col 0 = sloppiest
        inv = np.where(lam >= floor, 1.0 / np.maximum(lam, 1e-30), 0.0)
        Rmu = U @ np.diag(inv) @ U.T
        K = Pmu @ np.linalg.inv(Pmu + Rmu)
        step = K @ (Rmu @ grad)                                      # reaching walk (stiff dirs)
        # --- override the lowest ACTIVE eigendirection with the full-likelihood profile hedge
        act_dirs = np.where(lam >= floor)[0]
        if act_dirs.size:
            j0 = act_dirs[0]                                         # lowest-lambda active direction
            uj = np.zeros(D); uj[act] = U[:, j0]
            prof = np.array([_loglik(mu + o * uj, m, P, y) for o in off])
            w = np.exp(prof - prof.max()); w /= w.sum()
            c0 = Kstar * float(w @ off)                              # hedged step along u_j0
            step_r = U.T @ step                                     # step in eigenbasis
            step_r[j0] = c0                                          # replace sloppy component
            step = U @ step_r
        mu[act] += np.clip(step, -gap, gap)
        Pmu = (I_r - K) @ Pmu + np.diag(qmu); Pmu = 0.5 * (Pmu + Pmu.T)
        Pp = P + Q_of(mu[:N]); Sb = H @ Pp @ H.T + R_of(mu[N:]) + 1e-9 * np.eye(M)
        Kk = Pp @ H.T @ np.linalg.inv(Sb); m = m + Kk @ e; P = Pp - Kk @ H @ Pp; P = 0.5 * (P + P.T)
        out[t] = mu; state[t] = m
    return out, state, r


def hammer():
    T = 600; b = slice(T // 3 + 40, 2 * T // 3 - 40)
    _, _, r = reach_hedge(gen(None, 0.0, 60))
    print(f"active r={r}; reach-stiff + hedge-sloppy in the Fisher eigenbasis\n")
    for hot, nm in [(1, "xi2 hot "), (3, "eta2 hot")]:
        tru = np.zeros(D); tru[hot] = 1.4
        ee = []; eg = []; drift = []
        for seed in range(6):
            st = reach_hedge(gen(None, 0.0, T, seed))[0][150:].mean(0)
            drift.append(np.abs(st[[1, 2, 3]]).max())
            Y = gen(hot, 1.4, T, seed)
            we = reach_hedge(Y)[0][b].mean(0); wg = exact_grid(Y, 5)[b].mean(0)
            ee.append(np.abs(we - tru)); eg.append(np.abs(wg - tru))
        ee = np.mean(ee, 0); eg = np.mean(eg, 0)
        print(f" {nm}: |RH -truth| {ee}  sum {ee.sum():.2f}   (static drift max {max(drift):.2f})")
        print(f" {nm}: |GRID-truth| {eg}  sum {eg.sum():.2f}")
    print("\n per-seed (mixing H):")
    for seed in range(6):
        Yx = gen(1, 1.4, T, seed); wx = reach_hedge(Yx)[0]; gx = exact_grid(Yx, 5)
        Ye = gen(3, 1.4, T, seed); we = reach_hedge(Ye)[0]; ge = exact_grid(Ye, 5)
        print(f"  s{seed}: xi2hot eta2={wx[b,3].mean():+.2f}(g{gx[b,3].mean():+.2f}) xi2={wx[b,1].mean():.2f} | "
              f"eta2hot eta2={we[b,3].mean():.2f}(g{ge[b,3].mean():.2f}) eta1={we[b,2].mean():+.2f}")


if __name__ == "__main__":
    hammer()
