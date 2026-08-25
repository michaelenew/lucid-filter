"""Probe 0007 -- the stationary-AR(1) walker + spectral truncation.

0006 diagnosed the fix: statfilter scales are stationary AR(1) reverting to 0 (the
prior the exact grid carries), not an unbounded random walk.  So each scale mu_k is
tracked by a stationary-AR(1) Kalman recursion, and only the identifiable
(high-Fisher) axes are walked (spectral truncation).

Per axis k, each step:
  predict  mu- = phi mu ;  Pmu- = phi^2 Pmu + s^2(1-phi^2)      (mean-reverting)
  observe  the local Newton offset e_est = score/info at mu-, obs var R = 1/info
  update   K = Pmu-/(Pmu- + R) ;  mu = mu- + K e_est ;  Pmu = (1-K)Pmu-
Frozen axes (Ichar below a fraction of the max) stay at 0.  Expected Fisher
throughout (0.5 tr(S^-1 dS_k S^-1 dS_k)) -- the stable curvature (0005).

Reuses the n=2, m=2, correlated-Q, mixing-H model + exact grid from 0006.
Tests: static data (must stay ~0, no drift), strong-mode hot, sensor hot; vs grid.
"""
import os
import sys
import importlib.util

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
_spec = importlib.util.spec_from_file_location(
    "p6", os.path.join(os.path.dirname(__file__), "0006_walker_nge1_and_H.py"))
p6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p6)

np.set_printoptions(precision=3, suppress=True)
D, N, M, PHI, SS = p6.D, p6.N, p6.M, p6.PHI, p6.SS
Q_of, R_of, H, LAM = p6.Q_of, p6.R_of, p6.H, p6.LAM
score_fisher, steady_fisher, exact_grid = p6.score_fisher, p6.steady_fisher, p6.exact_grid


def gen(hot_axis, amp, T=900, seed=1):
    rng = np.random.default_rng(seed)
    psi = np.zeros((T, D))
    if hot_axis is not None:
        psi[T // 3: 2 * T // 3, hot_axis] = amp
    th = np.zeros(N); Y = np.zeros((T, M))
    for t in range(T):
        LQ = np.linalg.cholesky(Q_of(psi[t, :N]) + 1e-12 * np.eye(N))
        th = th + LQ @ rng.standard_normal(N)
        Y[t] = H @ th + np.sqrt(np.diag(R_of(psi[t, N:]))) * rng.standard_normal(M)
    return Y


def walk_stationary(Y, trunc_frac=0.05):
    Ichar = steady_fisher()
    active = Ichar >= trunc_frac * Ichar.max()      # spectral truncation
    nu = SS ** 2 * (1.0 - PHI ** 2)                 # AR(1) innovation variance
    mu = np.zeros(D); Pmu = SS ** 2                 # stationary prior
    st = np.zeros(N); P = np.eye(N) * (LAM.max() + p6.RHO.max())
    out = np.zeros((len(Y), D))
    for t, y in enumerate(Y):
        # predict (mean-revert) the scale
        mu_pred = PHI * mu
        Pmu_pred = PHI ** 2 * Pmu + nu
        # local score/info at the predicted scale
        score, info, *_ = score_fisher(mu_pred, st, P, y)
        info = np.maximum(info, 1e-6)
        e_est = score / info
        R_mu = 1.0 / info
        K = Pmu_pred / (Pmu_pred + R_mu)
        mu = mu_pred + K * e_est
        Pmu = (1.0 - K) * Pmu_pred
        mu[~active] = 0.0; Pmu[~active] = SS[~active] ** 2
        out[t] = mu
        # state KF at the point estimate
        Ppred = P + Q_of(mu[:N]); e = y - H @ st
        Smat = H @ Ppred @ H.T + R_of(mu[N:]) + 1e-9 * np.eye(M)
        Sinv = np.linalg.inv(Smat); Kk = Ppred @ H.T @ Sinv
        st = st + Kk @ e; P = Ppred - Kk @ H @ Ppred; P = 0.5 * (P + P.T)
    return out, active


if __name__ == "__main__":
    T = 900
    lab = ["xi1(weak)", "xi2(strong)", "eta1", "eta2"]
    _, active = walk_stationary(gen(None, 0.0))
    print(f"active axes (spectral truncation, Ichar>=5% max): "
          f"{[lab[k] for k in range(D) if active[k]]}")

    print("\nSTATIC data (truth psi=0 -- must stay ~0, no drift):")
    w, _ = walk_stationary(gen(None, 0.0))
    print(f"  WALK mean over t>100: {w[100:].mean(0)}")

    for ax, name in [(1, "strong eigenmode xi2"), (3, "sensor eta2")]:
        Y = gen(ax, 1.4)
        ref = exact_grid(Y, order=5)
        w, _ = walk_stationary(Y)
        b = slice(T // 3 + 40, 2 * T // 3 - 40)
        print(f"\n{name} hot (truth 1.4; grid shrinks via the stationary prior):")
        print(f"  GRID band: {ref[b].mean(0)}")
        print(f"  WALK band: {w[b].mean(0)}")
        corr = [np.corrcoef(w[30:, k], ref[30:, k])[0, 1]
                if w[30:, k].std() > 1e-9 else np.nan for k in range(D)]
        rmse = np.sqrt(((w[30:] - ref[30:]) ** 2).mean(0))
        print(f"  corr {lab}: {np.array(corr)}")
        print(f"  rmse {lab}: {rmse}")
