"""Probe 0003 -- the scale-Fisher geometry of the per-component multivariate filter.

Design locked with the user:
  * measurement R diagonal (each sensor an independent scale channel), full-R an open;
  * process Q symmetric PD, per-EIGENMODE scale channels with V FIXED (open: learn V);
  * eigenbasis fork -> COMPOSE: model scales in Q's eigenbasis + sensor axes, then
    allocate grid/simplex resolution by the Fisher spectrum in that basis.

This probe measures whether that composition is clean.  The scale vector is
  psi = (xi_1..xi_n  [process eigenmode log-scales],  eta_1..eta_m [sensor log-scales]),  D = n+m.
The covariances as functions of psi (V, lambda, rho fixed at the truth):
  Q(psi) = V diag(lambda_k e^{xi_k}) V^T          (always PD -- eigenbasis makes PD free)
  R(psi) = diag(rho_i e^{eta_i})                   (diagonal)
Homoscedastic (psi fixed) => a plain multivariate KF, so the marginal loglik(psi)
is exact and we can take its Hessian directly (no grid needed to MEASURE the Fisher).

We report, at the truth psi=0 on data generated there:
  * the Fisher eigenspectrum (sloppiness / condition number / effective rank),
  * how DIAGONAL the Fisher already is in the psi coordinates (are process-eigenmode
    and sensor axes close to the Fisher eigenbasis? -> composing is trivial),
  * the process<->measurement block coupling (the multivariate lift of the scalar
    s_P vs s_M confound),
  * how the spectrum spreads as the process correlation r grows (strong correlation
    -> spread eigenvalues -> fewer stiff DOF, the user's "less info about b,c once a").
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import VectorFilter, VecParams  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=120)


def Q_of(psi_P, V, lam):
    return V @ np.diag(lam * np.exp(psi_P)) @ V.T


def R_of(psi_M, rho):
    return np.diag(rho * np.exp(psi_M))


def loglik(Y, H, psi, V, lam, rho, n):
    Q0 = Q_of(psi[:n], V, lam)
    R0 = R_of(psi[n:], rho)
    f = VectorFilter(VecParams(Q0, R0, 0.0, 0.0, 0.0, 0.0), H=H, order=3)
    return f.loglik(Y)


def gen(H, V, lam, rho, T, seed):
    rng = np.random.default_rng(seed)
    n, m = V.shape[0], H.shape[0]
    Q0 = Q_of(np.zeros(n), V, lam)
    R0 = R_of(np.zeros(m), rho)
    LQ = np.linalg.cholesky(Q0)
    Lr = np.sqrt(np.diag(R0))
    th = np.zeros(n)
    Y = np.zeros((T, m))
    for t in range(T):
        th = th + LQ @ rng.standard_normal(n)
        Y[t] = H @ th + Lr * rng.standard_normal(m)
    return Y


def fisher(Y, H, V, lam, rho, n, h=1e-3):
    """Observed information: -Hessian of the mean loglik at psi=0, central differences."""
    D = n + H.shape[0]
    T = Y.shape[0]
    base = loglik(Y, H, np.zeros(D), V, lam, rho, n) / T
    Im = np.zeros((D, D))
    e = np.eye(D)
    ll = {}

    def L(v):
        key = tuple(np.round(v, 9))
        if key not in ll:
            ll[key] = loglik(Y, H, v, V, lam, rho, n) / T
        return ll[key]

    for i in range(D):
        Im[i, i] = -(L(h * e[i]) + L(-h * e[i]) - 2 * base) / h ** 2
    for i in range(D):
        for j in range(i + 1, D):
            pp = L(h * e[i] + h * e[j]); pm = L(h * e[i] - h * e[j])
            mp = L(-h * e[i] + h * e[j]); mm = L(-h * e[i] - h * e[j])
            Im[i, j] = Im[j, i] = -(pp - pm - mp + mm) / (4 * h ** 2)
    return Im


def analyse(tag, Im, n, m):
    D = n + m
    w, U = np.linalg.eigh(Im)
    w = w[::-1]; U = U[:, ::-1]
    cond = w[0] / max(w[-1], 1e-12)
    part = (w.sum() ** 2) / (w @ w)              # participation ratio ~ effective rank
    # diagonality in psi coords: how much Fisher mass is off-diagonal
    offfrac = (np.sum(Im ** 2) - np.sum(np.diag(Im) ** 2)) / np.sum(Im ** 2)
    # process<->measurement block coupling
    Bpm = Im[:n, n:]
    block_coup = np.sqrt(np.sum(Bpm ** 2)) / np.sqrt(np.sum(Im ** 2))
    # alignment of Fisher eigenvectors with psi axes: mean max |component|
    align = np.mean(np.max(np.abs(U), axis=0))
    print(f"\n=== {tag}  (n={n}, m={m}, D={D}) ===")
    print(" Fisher (psi coords: xi_1..xi_n | eta_1..eta_m):")
    print(Im)
    print(f" eigenspectrum:        {w}")
    print(f" condition number:     {cond:8.1f}   (sloppiness)")
    print(f" participation ratio:  {part:8.2f}   (effective # stiff DOF of {D})")
    print(f" off-diagonal mass:    {offfrac:8.3f}   (0 = Fisher already diagonal in psi coords)")
    print(f" proc<->meas coupling: {block_coup:8.3f}   (the multivariate s_P vs s_M confound)")
    print(f" eigvec axis-alignment:{align:8.3f}   (1 = eigenbasis == psi axes -> compose is trivial)")


if __name__ == "__main__":
    n = m = 2
    H_I = np.eye(2)
    H_mix = np.array([[1.0, 0.0], [1.0, 1.0]])
    theta45 = np.pi / 4
    V = np.array([[np.cos(theta45), -np.sin(theta45)],
                  [np.sin(theta45), np.cos(theta45)]])
    rho = np.array([0.5, 0.5])
    T = 6000

    for r in (0.0, 0.5, 0.9):
        lam_diag = np.array([1.0 + r, 1.0 - r])   # eigenvalues of [[1,r],[r,1]]
        Y = gen(H_I, V, lam_diag, rho, T, seed=100 + int(10 * r))
        Im = fisher(Y, H_I, V, lam_diag, rho, n)
        analyse(f"H=I, process corr r={r} (Q eigvals {lam_diag})", Im, n, m)

    # a mixing H, moderate correlation
    r = 0.5
    lam_diag = np.array([1.0 + r, 1.0 - r])
    Y = gen(H_mix, V, lam_diag, rho, T, seed=200)
    Im = fisher(Y, H_mix, V, lam_diag, rho, n)
    analyse(f"H=mixing [[1,0],[1,1]], r={r}", Im, n, m)
