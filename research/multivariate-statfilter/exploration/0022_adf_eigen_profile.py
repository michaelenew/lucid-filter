"""Probe 0022 -- assumed-density filtering in the Fisher eigenbasis with full-likelihood
profiles (route 1, the construction).

The impasse: a low-lambda eigendirection is EITHER an ambiguous split (truth ~0, must HEDGE)
or a genuine slow signal (truth != 0, must REACH); lambda alone cannot tell them apart.  The
full-likelihood PROFILE can -- broad/flat (ambiguous) vs peaked-off-centre (real signal).

So run 0019's smooth matrix walk (Pmu gives the reach, and is smooth so it never flips), but
replace the LOCAL Gaussian likelihood (grad, F) with the FULL-likelihood profile per Fisher
eigendirection:
    each active u_j: evaluate loglik at mu + o u_j over an axial grid -> weights w(o);
        o_j*  = sum w o                      (profile peak offset; the "observation")
        v_j   = sum w (o-o_j*)^2  (+ floor)  (profile spread; the observation variance)
    frozen (sub-floor) dirs: v_j = inf.
    Rmu = U diag(v_j) U^T ;  K = Pmu (Pmu+Rmu)^-1 ;  mu += K (U o*) ;  Pmu = (I-K)Pmu + drift.
A stiff hot axis has a sharp off-centre profile -> small v, large o* -> Pmu reaches it.  An
ambiguous split has a broad flat profile -> large v -> small gain -> no commit (the hedge),
matching the exact grid's posterior spread.  Pmu stays a smooth physical matrix -> no flips.
No free parameter (floor = derived identifiability floor; grid = Sparrow gap).  Cost O(r^3)+
linear profiles.
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


def adf_eigen(Y):
    Ich = Ichar(); floor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2)
    act = np.where(Ich >= floor)[0]; r = len(act)
    gap = 1.5 * SS[0]; Kstar = (1 - PHI[0]) / 4.0
    qmu = np.array([Kstar ** 2 / (Ich[int(k)] * (1 - Kstar)) for k in act])
    off = gap * np.arange(-3, 4); vfloor = (gap / 3.0) ** 2       # +-3-gap profile window (labeled resolution)
    mu = np.zeros(D); Pmu = np.diag(SS[act] ** 2).astype(float)
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    out = np.zeros((len(Y), D)); state = np.zeros((len(Y), N)); I_r = np.eye(r)
    for t, y in enumerate(Y):
        _, F, e = score_and_F(mu, m, P, y, act)
        lam, U = np.linalg.eigh(F)
        ostar = np.zeros(r); vj = np.full(r, np.inf)
        for j in range(r):
            if lam[j] < floor:
                continue
            uj = np.zeros(D); uj[act] = U[:, j]
            prof = np.array([_loglik(mu + o * uj, m, P, y) for o in off])
            w = np.exp(prof - prof.max()); w /= w.sum()
            ostar[j] = float(w @ off)
            vj[j] = float(w @ (off - ostar[j]) ** 2) + vfloor       # profile spread = obs variance
        # matrix finding-18 gain with per-direction obs variance (frozen dirs contribute nothing)
        finite = np.isfinite(vj)
        Rmu = U[:, finite] @ np.diag(vj[finite]) @ U[:, finite].T + 1e6 * (I_r - U[:, finite] @ U[:, finite].T)
        K = Pmu @ np.linalg.inv(Pmu + Rmu)                          # frozen dirs: huge obs var -> gain ~0
        obs_phys = U[:, finite] @ ostar[finite]                     # profile-peak offsets, physical
        mu[act] += np.clip(K @ obs_phys, -gap, gap)
        Pmu = (I_r - K) @ Pmu + np.diag(qmu); Pmu = 0.5 * (Pmu + Pmu.T)
        Pp = P + Q_of(mu[:N]); Sb = H @ Pp @ H.T + R_of(mu[N:]) + 1e-9 * np.eye(M)
        Kk = Pp @ H.T @ np.linalg.inv(Sb); m = m + Kk @ e; P = Pp - Kk @ H @ Pp; P = 0.5 * (P + P.T)
        out[t] = mu; state[t] = m
    return out, state, r


def hammer():
    T = 600; b = slice(T // 3 + 40, 2 * T // 3 - 40)
    _, _, r = adf_eigen(gen(None, 0.0, 60))
    print(f"active r={r}; ADF in Fisher eigenbasis, full-likelihood profiles\n")
    for hot, nm in [(1, "xi2 hot "), (3, "eta2 hot")]:
        tru = np.zeros(D); tru[hot] = 1.4
        ee = []; eg = []; drift = []
        for seed in range(6):
            st = adf_eigen(gen(None, 0.0, T, seed))[0][150:].mean(0)
            drift.append(np.abs(st[[1, 2, 3]]).max())
            Y = gen(hot, 1.4, T, seed)
            we = adf_eigen(Y)[0][b].mean(0); wg = exact_grid(Y, 5)[b].mean(0)
            ee.append(np.abs(we - tru)); eg.append(np.abs(wg - tru))
        ee = np.mean(ee, 0); eg = np.mean(eg, 0)
        print(f" {nm}: |ADF -truth| {ee}  sum {ee.sum():.2f}   (static drift max {max(drift):.2f})")
        print(f" {nm}: |GRID-truth| {eg}  sum {eg.sum():.2f}")
    print("\n per-seed (mixing H):")
    for seed in range(6):
        Yx = gen(1, 1.4, T, seed); wx = adf_eigen(Yx)[0]; gx = exact_grid(Yx, 5)
        Ye = gen(3, 1.4, T, seed); we = adf_eigen(Ye)[0]; ge = exact_grid(Ye, 5)
        print(f"  s{seed}: xi2hot eta2={wx[b,3].mean():+.2f}(g{gx[b,3].mean():+.2f}) xi2={wx[b,1].mean():.2f} | "
              f"eta2hot eta2={we[b,3].mean():.2f}(g{ge[b,3].mean():.2f}) eta1={we[b,2].mean():+.2f}")


if __name__ == "__main__":
    hammer()
