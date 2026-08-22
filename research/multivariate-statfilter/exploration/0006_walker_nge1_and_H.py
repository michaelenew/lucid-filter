"""Probe 0006 -- the walker with n>1 process eigenmodes and a real mixing H.

0005 validated the walker on a scalar process (only one xi axis) + two sensors.
The production case needs the PROCESS EIGENMODE axes too: Q symmetric PD, V fixed,
per-eigenmode scales xi_k walking on their own.  This probe:
  n=2, m=2; Q0 correlated (two eigenmodes), R0 diagonal, H a real mixing matrix.
  psi = (xi_1, xi_2 [process eigenmodes], eta_1, eta_2 [sensors]),  D=4.
Truth: process eigenmode 1 goes hot in window A; sensor 2 goes hot in window B
(disjoint).  Can the diagonal walker resolve BOTH axis types independently, and
match the exact per-component grid (order^4 = 625 nodes)?

Score / expected-Fisher per axis, at predictive S and innovation e:
  process eigenmode k:  dS/dxi_k  = lam_k e^{xi_k} (H v_k)(H v_k)^T
  sensor i:             dS/deta_i = rho_i e^{eta_i} E_ii
  score_k = 0.5 ( eᵀS⁻¹ dS_k S⁻¹ e − tr(S⁻¹ dS_k) );  F_kl = 0.5 tr(S⁻¹ dS_k S⁻¹ dS_l)
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter.core import _chain, _LOG2PI  # noqa: E402

np.set_printoptions(precision=3, suppress=True)

# ---- model constants
Q0 = np.array([[1.0, 0.6], [0.6, 1.0]])          # correlated process -> two eigenmodes
LAM, V = np.linalg.eigh(Q0)                       # eigenvalues (ascending), eigenvectors
RHO = np.array([1.0, 1.0])                        # base per-sensor variances (diagonal R)
H = np.array([[1.0, 0.0], [0.6, 1.0]])           # real mixing measurement matrix
N, M = 2, 2
D = N + M
PHI = np.full(D, 0.9)
SS = np.full(D, 0.4)
HV = H @ V                                        # (m, n): H times each eigenvector


def Q_of(xi):
    return V @ np.diag(LAM * np.exp(xi)) @ V.T


def R_of(eta):
    return np.diag(RHO * np.exp(eta))


# ---------------------------------------------------------------- generation
def generate(T, seed):
    rng = np.random.default_rng(seed)
    psi = np.zeros((T, D))
    psi[T // 5: 2 * T // 5, 0] = 1.4              # eigenmode-1 hot, window A
    psi[3 * T // 5: 4 * T // 5, 3] = 1.4          # sensor-2 hot, window B
    th = np.zeros(N); Y = np.zeros((T, M))
    for t in range(T):
        LQ = np.linalg.cholesky(Q_of(psi[t, :N]) + 1e-12 * np.eye(N))
        th = th + LQ @ rng.standard_normal(N)
        Y[t] = H @ th + np.sqrt(np.diag(R_of(psi[t, N:]))) * rng.standard_normal(M)
    return Y, psi


# --------------------------------------------------- exact per-component grid
def exact_grid(Y, order=5):
    chains = [_chain(PHI[k], SS[k], order) for k in range(D)]
    lams = [c[0] for c in chains]; ws = [c[1] for c in chains]; Ts = [c[2] for c in chains]
    mesh = np.array(np.meshgrid(*lams, indexing="ij")).reshape(D, -1).T   # (G, D)
    G = mesh.shape[0]
    pi0 = np.ones(G)
    for k in range(D):
        pi0 = pi0 * ws[k][np.searchsorted(lams[k], mesh[:, k])] if False else pi0
    # build pi0 and T as Kron
    pi0 = ws[0]
    T = Ts[0]
    for k in range(1, D):
        pi0 = np.kron(pi0, ws[k]); T = np.kron(T, Ts[k])
    Qg = np.array([Q_of(g[:N]) for g in mesh])   # (G,n,n)
    Rg = np.array([R_of(g[N:]) for g in mesh])   # (G,m,m)
    m = np.zeros(N); P = np.eye(N) * float(Qg.reshape(G, -1).max() + Rg.reshape(G, -1).max()) * N
    pi = pi0.copy()
    out = np.zeros((len(Y), D))
    for t, y in enumerate(Y):
        pi = pi @ T
        Ppred = P[None] + Qg
        e = y - H @ m
        PHt = np.einsum("gij,kj->gik", Ppred, H)
        Smat = np.einsum("ij,gjk->gik", H, PHt) + Rg
        Sinv = np.linalg.inv(Smat)
        sgn, logdet = np.linalg.slogdet(Smat)
        maha = np.einsum("i,gij,j->g", e, Sinv, e)
        lg = -0.5 * (M * _LOG2PI + logdet + maha)
        mx = lg.max(); w = pi * np.exp(lg - mx); Z = w.sum(); pi = w / Z
        K = np.einsum("gik,gkl->gil", PHt, Sinv)
        Kbar = np.einsum("g,gil->il", pi, K)
        m = m + Kbar @ e
        mpost = m[None] - (Kbar @ e)[None] + np.einsum("gil,l->gi", K, e)
        dm = mpost - m
        KH = np.einsum("gil,lj->gij", K, H)
        Ppost = Ppred - np.einsum("gij,gjk->gik", KH, Ppred)
        P = np.einsum("g,gij->ij", pi, Ppost) + np.einsum("g,gi,gj->ij", pi, dm, dm)
        P = 0.5 * (P + P.T)
        out[t] = pi @ mesh
    return out


# --------------------------------------------------------- diagonal walker
def dS_list(psi, Ppred_unused):
    dS = []
    for k in range(N):                            # process eigenmodes
        hv = HV[:, k]
        dS.append(LAM[k] * math.exp(psi[k]) * np.outer(hv, hv))
    for i in range(M):                            # sensors
        E = np.zeros((M, M)); E[i, i] = RHO[i] * math.exp(psi[N + i]); dS.append(E)
    return dS


def score_fisher(psi, m, P, y):
    Ppred = P + Q_of(psi[:N])
    e = y - H @ m
    Smat = H @ Ppred @ H.T + R_of(psi[N:]) + 1e-9 * np.eye(M)
    Sinv = np.linalg.inv(Smat)
    dS = dS_list(psi, Ppred)
    Sie = Sinv @ e
    score = np.array([0.5 * (Sie @ d @ Sie - np.trace(Sinv @ d)) for d in dS])
    Msi = [Sinv @ d for d in dS]
    info = np.array([0.5 * np.trace(Msi[k] @ Msi[k]) for k in range(D)])
    return score, info, Ppred, Sinv, e


def steady_fisher():
    P = np.eye(N) * (LAM.max() + RHO.max())
    for _ in range(300):
        Ppred = P + Q0
        Smat = H @ Ppred @ H.T + np.diag(RHO)
        K = Ppred @ H.T @ np.linalg.inv(Smat)
        P = Ppred - K @ H @ Ppred
    inf, _, _, _, _ = score_fisher(np.zeros(D), np.zeros(N), P, np.zeros(M))
    return inf


def walk(Y):
    Ichar = steady_fisher()
    Kstar = (1.0 - PHI) / 4.0
    qmu = Kstar ** 2 / (Ichar * (1.0 - Kstar))
    cap = 1.5 * SS
    mu = np.zeros(D); Pmu = SS ** 2
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max())
    out = np.zeros((len(Y), D))
    for t, y in enumerate(Y):
        score, info, Ppred, Sinv, e = score_fisher(mu, m, P, y)
        offset = score / np.maximum(info, 1e-6)
        R_mu = 1.0 / np.maximum(info, 1e-6)
        K_mu = Pmu / (Pmu + R_mu)
        mu = mu + np.clip(K_mu * offset, -cap, cap)
        Pmu = (1.0 - K_mu) * Pmu + qmu
        out[t] = mu
        # state KF at the point estimate
        Ppred = P + Q_of(mu[:N]); e = y - H @ m
        Smat = H @ Ppred @ H.T + R_of(mu[N:]) + 1e-9 * np.eye(M); Sinv = np.linalg.inv(Smat)
        K = Ppred @ H.T @ Sinv
        m = m + K @ e
        P = Ppred - K @ H @ Ppred; P = 0.5 * (P + P.T)
    return out


if __name__ == "__main__":
    T = 900
    Y, psi_true = generate(T, seed=1)
    ref = exact_grid(Y, order=5)
    w = walk(Y)
    a0, a1 = T // 5 + 30, 2 * T // 5 - 30        # window A (eigenmode-1 hot)
    b0, b1 = 3 * T // 5 + 30, 4 * T // 5 - 30    # window B (sensor-2 hot)
    lab = ["xi1", "xi2", "eta1", "eta2"]
    print(f"axes: {lab}   (eigenmode-1 hot in A, sensor-2 hot in B; truth peak 1.4)")
    print(f"window A  truth : {psi_true[a0:a1].mean(0)}")
    print(f"window A  GRID  : {ref[a0:a1].mean(0)}")
    print(f"window A  WALK  : {w[a0:a1].mean(0)}")
    print(f"window B  truth : {psi_true[b0:b1].mean(0)}")
    print(f"window B  GRID  : {ref[b0:b1].mean(0)}")
    print(f"window B  WALK  : {w[b0:b1].mean(0)}")
    rmse = np.sqrt(((w[30:] - ref[30:]) ** 2).mean(0))
    corr = [np.corrcoef(w[30:, k], ref[30:, k])[0, 1] for k in range(D)]
    print(f"\nwalk-vs-grid RMSE {lab}: {rmse}")
    print(f"walk-vs-grid corr {lab}: {np.array(corr)}")
