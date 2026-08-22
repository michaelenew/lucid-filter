"""Probe 0004 -- the practical construction vs the exact grid.

Tests the walking-only multivariate plan on the cleanest per-component case:
n=1 (scalar state), m=2 sensors both reading it (H=[[1],[1]]), so
  psi = (xi [process log-scale], eta_1, eta_2 [per-sensor log-scales]),  D=3.
Truth: process scale steady; SENSOR 1 goes hot mid-stream while sensor 2 stays
clean.  The per-component question is whether we can say "sensor 1 is hot, sensor
2 is fine" -- which a single scalar measurement scale (VectorFilter today) cannot.

Two constructions, compared:
  REFERENCE  exact tensor grid over psi (order^3 nodes), GPB1 KF collapse -- the
             theory-only ground truth.  Reports per-step E[psi] and the shares.
  WALKER     a DIAGONAL simplex walker: one running estimate psi_hat, updated each
             step from a D+1 simplex estimate of grad log p(y|psi) with a per-axis
             critically-damped Kalman walk (the finding-18 loop, per axis, ignoring
             the ~0.2 process<->measurement cross-term 0003 found).  Linear in D.

Claims under test:
  (1) direction/tracking -- psi_hat tracks the exact grid's E[psi], including
      separating sensor 1 (hot) from sensor 2 (clean);
  (2) the diagonal walk (ignoring the one cross-term) is good enough to track.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter.core import _chain, _LOG2PI  # noqa: E402

np.set_printoptions(precision=3, suppress=True)

CH = ("xi", "eta1", "eta2")           # the D=3 log-scale channels
PHI = np.array([0.9, 0.9, 0.9])       # AR(1) persistence per channel (class)
S = np.array([0.4, 0.4, 0.4])         # AR(1) log-SD per channel (class)
Q0 = 1.0                              # base process variance (n=1)
RHO = np.array([1.0, 1.0])           # base per-sensor variances
H = np.array([[1.0], [1.0]])         # both sensors read the single state


def covs(psi):
    Q = np.array([[Q0 * math.exp(psi[0])]])
    R = np.diag(RHO * np.exp(psi[1:]))
    return Q, R


# ---------------------------------------------------------------- generation
def generate(T, seed):
    rng = np.random.default_rng(seed)
    psi_true = np.zeros((T, 3))
    # sensor 1 hot in the middle third; process & sensor 2 steady
    psi_true[T // 3: 2 * T // 3, 1] = 1.6
    th = 0.0
    Y = np.zeros((T, 2))
    for t in range(T):
        Q, R = covs(psi_true[t])
        th = th + math.sqrt(Q[0, 0]) * rng.standard_normal()
        Y[t] = (H @ np.array([th])) + np.sqrt(np.diag(R)) * rng.standard_normal(2)
    return Y, psi_true


# --------------------------------------------------- exact tensor grid (ref)
def exact_grid(Y, order=5):
    chains = [_chain(PHI[k], S[k], order) for k in range(3)]
    lams = [c[0] for c in chains]
    ws = [c[1] for c in chains]
    Ts = [c[2] for c in chains]
    # joint grid
    grid = np.array(np.meshgrid(*lams, indexing="ij")).reshape(3, -1).T  # (G, 3)
    pi0 = np.einsum("i,j,k->ijk", ws[0], ws[1], ws[2]).ravel()
    T = np.einsum("ia,jb,kc->ijkabc", Ts[0], Ts[1], Ts[2]).reshape(pi0.size, pi0.size)
    G = pi0.size
    Qg = np.array([covs(g)[0] for g in grid])       # (G,1,1)
    Rg = np.array([covs(g)[1] for g in grid])       # (G,2,2)
    m = 0.0
    P = float(Qg.max() + Rg.max())
    pi = pi0.copy()
    out_psi = np.zeros((len(Y), 3)); out_sh = np.zeros((len(Y), 3))
    for t, y in enumerate(Y):
        pi = pi @ T
        Ppred = P + Qg[:, 0, 0]                       # (G,)
        e = y - (H[:, 0] * m)                         # (2,)
        S_ = Ppred[:, None, None] * np.ones((1, 2, 2)) + Rg  # HPpredH^T + R, H=ones
        Sinv = np.linalg.inv(S_)
        sgn, logdet = np.linalg.slogdet(S_)
        maha = np.einsum("i,gij,j->g", e, Sinv, e)
        lg = -0.5 * (2 * _LOG2PI + logdet + maha)
        mx = lg.max(); w = pi * np.exp(lg - mx); Z = w.sum(); pi = w / Z
        # gain K_g = Ppred H^T Sinv  (n=1 -> (G,2))
        Ht = H[:, 0]                                  # (2,)
        K = Ppred[:, None] * np.einsum("j,gjk->gk", Ht, Sinv)   # (G,2)
        Kbar = np.einsum("g,gk->k", pi, K)
        m = m + float(Kbar @ e)
        # collapse P (scalar state)
        Ppost = Ppred * (1.0 - np.einsum("gk,k->g", K, Ht))     # (I-KH)Ppred
        dm = np.einsum("gk,k->g", K, e) - float(Kbar @ e)
        P = float(pi @ Ppost + pi @ (dm * dm))
        out_psi[t] = pi @ grid
        # trace shares
        HPHt = Ppred[:, None, None] * np.ones((1, 2, 2))
        sp = pi @ (np.einsum("gij,gji->g", Sinv, HPHt) / 2)
        # process piece = H Qg H^T ; but Ppred already includes Q so prior=P_prev
        # split: S = H P_prev H^T + H Qg H^T + R
        Pprev = Ppred - Qg[:, 0, 0]
        A = Pprev[:, None, None] * np.ones((1, 2, 2))
        B = Qg[:, 0, 0][:, None, None] * np.ones((1, 2, 2))
        s_pr = pi @ (np.einsum("gij,gji->g", Sinv, A) / 2)
        s_pc = pi @ (np.einsum("gij,gji->g", Sinv, B) / 2)
        s_ms = pi @ (np.einsum("gij,gji->g", Sinv, Rg) / 2)
        out_sh[t] = [s_pr, s_pc, s_ms]
    return out_psi, out_sh


# ------------------------------------------------- diagonal simplex walker
def obs_loglik(y, m, P, psi):
    Q, R = covs(psi)
    Ppred = P + Q[0, 0]
    e = y - H[:, 0] * m
    Smat = Ppred * np.ones((2, 2)) + R
    Sinv = np.linalg.inv(Smat)
    sgn, logdet = np.linalg.slogdet(Smat)
    return -0.5 * (2 * _LOG2PI + logdet + float(e @ Sinv @ e))


def walker(Y, delta=0.3):
    # per-axis critically-damped loop (finding 18): K* = (1-phi)/4
    phi = PHI
    Kstar = (1.0 - phi) / 4.0
    psi = np.zeros(3)
    Pmu = S ** 2                     # cold-start prior = stationary var
    m = 0.0
    P = float(Q0 + RHO.max())
    out_psi = np.zeros((len(Y), 3))
    for t, y in enumerate(Y):
        # mean-revert the walk estimate (AR(1) prior)
        psi = phi * psi
        # simplex gradient of the obs loglik at psi (central, per axis)
        g = np.zeros(3); info = np.zeros(3)
        base = obs_loglik(y, m, P, psi)
        for k in range(3):
            ek = np.zeros(3); ek[k] = delta
            lp = obs_loglik(y, m, P, psi + ek)
            lm = obs_loglik(y, m, P, psi - ek)
            g[k] = (lp - lm) / (2 * delta)
            info[k] = max(-(lp + lm - 2 * base) / delta ** 2, 1e-6)
        # per-axis natural-gradient step with critically-damped Kalman walk
        e_est = g / info                         # Newton offset per axis
        R_mu = 1.0 / info
        K_mu = Pmu / (Pmu + R_mu)
        step = np.clip(K_mu * e_est, -1.5 * S, 1.5 * S)
        psi = psi + step
        Pmu = (1.0 - K_mu) * Pmu + Kstar ** 2 / (info * (1.0 - Kstar) + 1e-9)
        out_psi[t] = psi
        # advance the state KF at the point estimate psi
        Q, R = covs(psi)
        Ppred = P + Q[0, 0]
        e = y - H[:, 0] * m
        Smat = Ppred * np.ones((2, 2)) + R
        Sinv = np.linalg.inv(Smat)
        K = Ppred * (H[:, 0] @ Sinv)
        m = m + float(K @ e)
        P = float(Ppred * (1.0 - K @ H[:, 0]))
    return out_psi


def walker_full(Y, delta=0.3, reversion=0.02):
    """Natural-gradient walker: full DxD simplex Hessian (de-mixes coupled
    channels), and a near-persistent walk (reversion ~0, so a sustained shift is
    held rather than pulled back to 0)."""
    psi = np.zeros(3)
    m = 0.0
    P = float(Q0 + RHO.max())
    out_psi = np.zeros((len(Y), 3))
    e_axis = np.eye(3)
    for t, y in enumerate(Y):
        psi = (1.0 - reversion) * psi
        base = obs_loglik(y, m, P, psi)
        g = np.zeros(3); Hmat = np.zeros((3, 3))
        fp = {}; fm = {}
        for k in range(3):
            fp[k] = obs_loglik(y, m, P, psi + delta * e_axis[k])
            fm[k] = obs_loglik(y, m, P, psi - delta * e_axis[k])
            g[k] = (fp[k] - fm[k]) / (2 * delta)
            Hmat[k, k] = (fp[k] + fm[k] - 2 * base) / delta ** 2
        for i in range(3):
            for j in range(i + 1, 3):
                pp = obs_loglik(y, m, P, psi + delta * (e_axis[i] + e_axis[j]))
                mm = obs_loglik(y, m, P, psi - delta * (e_axis[i] + e_axis[j]))
                Hmat[i, j] = Hmat[j, i] = (pp + mm - fp[i] - fm[i] - fp[j] - fm[j]
                                           + 2 * base) / (2 * delta ** 2)
        info = -Hmat + 1e-3 * np.eye(3)              # local Fisher (PD-regularised)
        step = np.linalg.solve(info, g)              # natural-gradient (Fisher^-1 grad)
        psi = psi + np.clip(step, -1.5 * S, 1.5 * S)
        out_psi[t] = psi
        Q, R = covs(psi)
        Ppred = P + Q[0, 0]
        e = y - H[:, 0] * m
        Smat = Ppred * np.ones((2, 2)) + R
        Sinv = np.linalg.inv(Smat)
        K = Ppred * (H[:, 0] @ Sinv)
        m = m + float(K @ e)
        P = float(Ppred * (1.0 - K @ H[:, 0]))
    return out_psi


if __name__ == "__main__":
    T = 1500
    Y, psi_true = generate(T, seed=0)
    ref_psi, ref_sh = exact_grid(Y, order=5)
    w_psi = walker(Y)
    wf_psi = walker_full(Y)

    def band(a, lo, hi):
        return a[lo:hi].mean(0)

    lo, mid_lo, mid_hi, hi = 50, T // 3 + 50, 2 * T // 3 - 50, T - 50
    print("mean psi by regime  [xi, eta1(sensor1), eta2(sensor2)]  (sensor1 hot in middle)")
    print(f"  truth  quiet : {psi_true[lo:mid_lo].mean(0)}")
    print(f"  truth  hot   : {psi_true[mid_lo:mid_hi].mean(0)}   <- eta1 = 1.6")
    print(f"  GRID   quiet : {band(ref_psi, lo, mid_lo)}")
    print(f"  GRID   hot   : {band(ref_psi, mid_lo, mid_hi)}")
    print(f"  WALK-diag quiet : {band(w_psi, lo, mid_lo)}")
    print(f"  WALK-diag hot   : {band(w_psi, mid_lo, mid_hi)}")
    print(f"  WALK-full quiet : {band(wf_psi, lo, mid_lo)}")
    print(f"  WALK-full hot   : {band(wf_psi, mid_lo, mid_hi)}")
    # does each correctly separate sensor1 from sensor2 in the hot band?
    print("\nhot-band sensor separation eta1 - eta2 (grid reaches ~0.96, span-capped):")
    print(f"  GRID      : {band(ref_psi, mid_lo, mid_hi)[1] - band(ref_psi, mid_lo, mid_hi)[2]:.3f}")
    print(f"  WALK-diag : {band(w_psi, mid_lo, mid_hi)[1] - band(w_psi, mid_lo, mid_hi)[2]:.3f}")
    print(f"  WALK-full : {band(wf_psi, mid_lo, mid_hi)[1] - band(wf_psi, mid_lo, mid_hi)[2]:.3f}")
    for tag, wp in (("diag", w_psi), ("full", wf_psi)):
        rmse = np.sqrt(((wp[50:] - ref_psi[50:]) ** 2).mean(0))
        corr = [np.corrcoef(wp[50:, k], ref_psi[50:, k])[0, 1] for k in range(3)]
        print(f"\nWALK-{tag} vs grid  RMSE [xi,eta1,eta2]: {rmse}   corr: {np.array(corr)}")
