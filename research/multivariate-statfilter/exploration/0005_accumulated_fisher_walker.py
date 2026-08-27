"""Probe 0005 -- the faithful multivariate walker (analytic expected Fisher).

0004 showed the two ways to get it wrong: a single-sample observed Hessian is too
noisy (natural gradient diverges), and a reverting per-axis step under-reaches.
The scalar walking filter (walking.py) avoids both: it uses the EXPECTED Fisher
`info = E[-d2 loglik]` (deterministic given S, not a sample Hessian), an UNBOUNDED
mu-walk (no reversion), and the derived critically-damped gain via a fixed drift
variance `q_mu = K*^2 / (I_char (1-K*))`, `K* = (1-phi)/4`.

This probe lifts that loop to D dimensions, analytically.  For log-scale axis k
that multiplies a covariance piece, with predictive S and innovation e:
    dS_k   = d S / d psi_k                     (a simple matrix)
    score_k = 0.5 ( e^T S^-1 dS_k S^-1 e  -  tr(S^-1 dS_k) )        (gradient)
    Fisher_kl = 0.5 tr( S^-1 dS_k S^-1 dS_l )                        (EXPECTED)
The expected Fisher is deterministic given S -- that is what makes it stable and
accumulable, exactly as in the scalar loop.  Two walkers:
  DIAG   per-axis mu-Kalman with Fisher_kk        (linear in D)
  BLOCK  natural-gradient with the full Fisher_kl  (quadratic in D)
Same n=1, m=2, H=[[1],[1]] per-sensor case as 0004; compared to the exact grid.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))

# reuse the generation + exact-grid reference from 0004
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "p0004", os.path.join(os.path.dirname(__file__), "0004_simplex_walk_vs_grid.py"))
_p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p)
generate, exact_grid, covs = _p.generate, _p.exact_grid, _p.covs
H, Q0, RHO, PHI, S = _p.H, _p.Q0, _p.RHO, _p.PHI, _p.S

np.set_printoptions(precision=3, suppress=True)


def dS_matrices(psi, Ppred):
    """dS/dpsi_k for k in (xi, eta1, eta2), at the current point."""
    Q = Q0 * math.exp(psi[0])
    dS_xi = Q * np.ones((2, 2))                       # d Ppred/dxi * H H^T
    dS_e1 = np.array([[RHO[0] * math.exp(psi[1]), 0.0], [0.0, 0.0]])
    dS_e2 = np.array([[0.0, 0.0], [0.0, RHO[1] * math.exp(psi[2])]])
    return [dS_xi, dS_e1, dS_e2]


def score_and_fisher(psi, m, P, y):
    Q, R = covs(psi)
    Ppred = P + Q[0, 0]
    e = y - H[:, 0] * m
    Smat = Ppred * np.ones((2, 2)) + R
    Sinv = np.linalg.inv(Smat)
    dS = dS_matrices(psi, Ppred)
    Sie = Sinv @ e
    score = np.array([0.5 * (Sie @ d @ Sie - np.trace(Sinv @ d)) for d in dS])
    F = np.zeros((3, 3))
    M = [Sinv @ d for d in dS]                        # S^-1 dS_k
    for i in range(3):
        for j in range(3):
            F[i, j] = 0.5 * np.trace(M[i] @ M[j])
    return score, F, Ppred, Sinv, e


def steady_fisher():
    """I_char_k: expected Fisher per axis at the base regime, steady-state P."""
    Q = Q0
    P = Q + RHO.max()
    for _ in range(200):                              # DARE fixed point at psi=0
        Ppred = P + Q
        Smat = Ppred * np.ones((2, 2)) + np.diag(RHO)
        K = Ppred * (H[:, 0] @ np.linalg.inv(Smat))
        P = Ppred * (1.0 - K @ H[:, 0])
    _, F, *_ = score_and_fisher(np.zeros(3), 0.0, P, np.zeros(2))
    return np.diag(F).copy(), P


def state_step(psi, m, P, y):
    Q, R = covs(psi)
    Ppred = P + Q[0, 0]
    e = y - H[:, 0] * m
    Smat = Ppred * np.ones((2, 2)) + R
    Sinv = np.linalg.inv(Smat)
    K = Ppred * (H[:, 0] @ Sinv)
    return m + float(K @ e), float(Ppred * (1.0 - K @ H[:, 0]))


def walk(Y, mode="diag", cap=None):
    Ichar, Psteady = steady_fisher()
    Kstar = (1.0 - PHI) / 4.0
    qmu = Kstar ** 2 / (Ichar * (1.0 - Kstar))        # fixed drift variance per axis
    if cap is None:
        cap = 1.5 * S
    mu = np.zeros(3)
    Pmu = S ** 2                                       # cold-start prior (finding 18)
    m, P = 0.0, float(Q0 + RHO.max())
    out = np.zeros((len(Y), 3))
    for t, y in enumerate(Y):
        score, F, *_ = score_and_fisher(mu, m, P, y)
        info = np.diag(F).copy()
        if mode == "diag":
            offset = score / np.maximum(info, 1e-6)    # per-axis Newton offset
            R_mu = 1.0 / np.maximum(info, 1e-6)
            K_mu = Pmu / (Pmu + R_mu)
            dmu = np.clip(K_mu * offset, -cap, cap)
            Pmu = (1.0 - K_mu) * Pmu + qmu
        else:  # block: full expected-Fisher natural gradient, per-axis Kalman gain
            offset = np.linalg.solve(F + 1e-6 * np.eye(3), score)
            R_mu = 1.0 / np.maximum(np.diag(F), 1e-6)
            K_mu = Pmu / (Pmu + R_mu)
            dmu = np.clip(K_mu * offset, -cap, cap)
            Pmu = (1.0 - K_mu) * Pmu + qmu
        mu = mu + dmu                                  # UNBOUNDED walk (no reversion)
        out[t] = mu
        m, P = state_step(mu, m, P, y)
    return out


if __name__ == "__main__":
    T = 1500
    Y, psi_true = generate(T, seed=0)
    ref, _ = exact_grid(Y, order=5)
    diag = walk(Y, "diag")
    block = walk(Y, "block")

    lo, mlo, mhi = 50, T // 3 + 50, 2 * T // 3 - 50

    def band(a):
        return a[mlo:mhi].mean(0)
    print("hot-band mean psi [xi, eta1(hot), eta2(clean)]  (truth eta1=1.6, grid span ~1.2)")
    print(f"  GRID  : {band(ref)}")
    print(f"  DIAG  : {band(diag)}")
    print(f"  BLOCK : {band(block)}")
    print("\nhot-band sensor separation eta1 - eta2 (grid ~0.96):")
    for tag, a in (("GRID", ref), ("DIAG", diag), ("BLOCK", block)):
        b = band(a); print(f"  {tag:5s}: {b[1] - b[2]:.3f}")
    print("\nquiet-band mean psi (should be ~0):")
    for tag, a in (("GRID", ref), ("DIAG", diag), ("BLOCK", block)):
        print(f"  {tag:5s}: {a[lo:mlo].mean(0)}")
    for tag, a in (("DIAG", diag), ("BLOCK", block)):
        rmse = np.sqrt(((a[50:] - ref[50:]) ** 2).mean(0))
        corr = [np.corrcoef(a[50:, k], ref[50:, k])[0, 1] for k in range(3)]
        print(f"\n{tag} vs grid  RMSE[xi,eta1,eta2]: {rmse}   corr: {np.array(corr)}")
