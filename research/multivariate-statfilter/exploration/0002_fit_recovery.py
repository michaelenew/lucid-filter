"""Prototype: fit recovery for the multivariate filter with supplied H.

Fitted via log-Cholesky of Q0, R0 (guarantees PD, unconstrained) plus the four
scale-channel params.  Two checks:
  (A) homoscedastic (s_P=s_M=0): recover full-symmetric Q0, R0 given H.
  (B) with a live process-scale channel: recover Q0,R0 shape AND (phi_P, s_P).
H is SUPPLIED in both.  Optimiser is plain Nelder-Mead on the loglik (proto only;
production would stage it).
"""
import math
import numpy as np
from scipy.optimize import minimize

import os
import importlib.util
# load the sibling prototype (numeric filename -> load by path)
_spec = importlib.util.spec_from_file_location(
    "mv_proto", os.path.join(os.path.dirname(__file__), "0001_reduction_and_shares.py"))
_mv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mv)
MvProto = _mv.MvProto


def chol_to_vec(L):
    """lower-tri Cholesky -> unconstrained vector (log the diagonal)."""
    n = L.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1):
            out.append(math.log(L[i, i]) if i == j else L[i, j])
    return out


def vec_to_chol(v, n):
    L = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i + 1):
            L[i, j] = math.exp(v[k]) if i == j else v[k]
            k += 1
    return L


def n_chol(n):
    return n * (n + 1) // 2


def loglik_of(Y, Q0, R0, H, phi_P, phi_M, s_P, s_M, order=5):
    f = MvProto(Q0, R0, H, phi_P=phi_P, phi_M=phi_M, s_P=s_P, s_M=s_M, order=order)
    f.reset()
    ll = 0.0
    for t in range(Y.shape[0]):
        ll += f.update(Y[t])["loglik"]
    return ll


def gen(n, m, H, Q0, R0, T, seed, phi_P=0.0, s_P=0.0):
    rng = np.random.default_rng(seed)
    LQ = np.linalg.cholesky(Q0); LR = np.linalg.cholesky(R0)
    th = np.zeros(n); Y = np.zeros((T, m)); lam = 0.0
    nu = s_P * s_P * (1 - phi_P * phi_P)
    for t in range(T):
        if s_P > 0:
            lam = phi_P * lam + math.sqrt(max(nu, 0)) * rng.standard_normal()
        scale = math.exp(lam)
        th = th + math.sqrt(scale) * (LQ @ rng.standard_normal(n))
        Y[t] = H @ th + LR @ rng.standard_normal(m)
    return Y


def fit_homoscedastic(Y, H, n, m):
    nQ, nR = n_chol(n), n_chol(m)
    # init: identity-ish
    v0 = chol_to_vec(np.eye(n)) + chol_to_vec(np.eye(m))

    def neg(v):
        Q0 = (lambda L: L @ L.T)(vec_to_chol(v[:nQ], n))
        R0 = (lambda L: L @ L.T)(vec_to_chol(v[nQ:nQ + nR], m))
        try:
            return -loglik_of(Y, Q0, R0, H, 0, 0, 0, 0) / Y.shape[0]
        except np.linalg.LinAlgError:
            return 1e9
    r = minimize(neg, v0, method="L-BFGS-B",
                 options=dict(maxiter=200, ftol=1e-10))
    Q0 = (lambda L: L @ L.T)(vec_to_chol(r.x[:nQ], n))
    R0 = (lambda L: L @ L.T)(vec_to_chol(r.x[nQ:nQ + nR], m))
    return Q0, R0, -r.fun


if __name__ == "__main__":
    n, m, T = 2, 2, 1500
    H = np.array([[1.0, 0.0], [1.0, 1.0]])          # supplied, non-trivial mixing
    Q0 = np.array([[1.0, 0.4], [0.4, 0.6]])
    R0 = np.array([[0.5, -0.15], [-0.15, 0.3]])

    print("=== (A) homoscedastic recovery (s=0), H supplied ===")
    Y = gen(n, m, H, Q0, R0, T, seed=7)
    Qh, Rh, ll = fit_homoscedastic(Y, H, n, m)
    np.set_printoptions(precision=3, suppress=True)
    print("Q0 true:\n", Q0, "\nQ0 hat:\n", Qh)
    print("R0 true:\n", R0, "\nR0 hat:\n", Rh)
    print(f"loglik/pt = {ll/T:.4f}")
    print(f"max abs err  Q0: {np.max(np.abs(Qh-Q0)):.3f}   R0: {np.max(np.abs(Rh-R0)):.3f}")
