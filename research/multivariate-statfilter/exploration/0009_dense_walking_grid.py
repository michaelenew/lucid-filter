"""Probe 0009 -- the dense WALKING grid (the production mechanism).

Corrects 0008: the scalar WalkingFilter is a DENSE window that walks (nodes at 1.5s,
translated by mu), not sparse sigma points.  The multivariate per-component walker is
the same idea in D dims: a dense tensor window over the (spectrally-truncated)
significant axes, each axis translated by its own walking centre mu_k via the
finding-18 mu-loop.  This is the fixed exact grid (0006) made to WALK.

  * grid, per axis k: nodes lam at gap=1.5 s_k, span +/- SPAN_S * s_k (dense, NO order
    -- node count derived from span/spacing, the Sparrow criterion);
  * joint = tensor product; each node's scale = mu + lam-offsets; Q,R from it;
  * state KF: multivariate GPB1 collapse over the joint grid (as 0006);
  * walk mu_k: analytic residual score/expected-Fisher marginalised onto axis k,
    fed to the scalar mu-loop (K*=(1-phi)/4, q_mu=K*^2/(I_char(1-K*)), unbounded mu).
Cost nodes^D -- fine for a testbed at small (truncated) D; reaches (walk) and stays
dense (accurate).  Benchmarked vs the fixed exact grid on the 0006 model.
"""
import os
import sys
import math
import importlib.util

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter.core import _LOG2PI  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "p6", os.path.join(os.path.dirname(__file__), "0006_walker_nge1_and_H.py"))
p6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p6)

np.set_printoptions(precision=3, suppress=True)
D, N, M, PHI, SS, LAM, RHO, H, HV = p6.D, p6.N, p6.M, p6.PHI, p6.SS, p6.LAM, p6.RHO, p6.H, p6.HV
Q_of, R_of, exact_grid, steady_fisher = p6.Q_of, p6.R_of, p6.exact_grid, p6.steady_fisher


def gen(hot, amp, T=900, seed=1):
    rng = np.random.default_rng(seed)
    psi = np.zeros((T, D))
    if hot is not None:
        psi[T // 3: 2 * T // 3, hot] = amp
    th = np.zeros(N); Y = np.zeros((T, M))
    for t in range(T):
        LQ = np.linalg.cholesky(Q_of(psi[t, :N]) + 1e-12 * np.eye(N))
        th = th + LQ @ rng.standard_normal(N)
        Y[t] = H @ th + np.sqrt(np.diag(R_of(psi[t, N:]))) * rng.standard_normal(M)
    return Y

_GAP = 1.5
_SPAN_S = 3.0                       # window half-span in units of s (dense; derived count)


def _axis_window(s):
    K = int(math.ceil(_SPAN_S / _GAP))          # nodes to cover +/- SPAN_S*s at 1.5s
    off = _GAP * s * np.arange(-K, K + 1)
    w = np.exp(-0.5 * (off / s) ** 2); w = w / w.sum()
    nu = max(s * s * (1 - PHI[0] ** 2), 1e-12)  # (phi same across axes here)
    return off, w, 2 * K + 1


def _dS(scale):
    """dS/dpsi_k at a scale vector: process eigenmodes then sensors."""
    dS = []
    for k in range(N):
        hv = HV[:, k]; dS.append(LAM[k] * math.exp(scale[k]) * np.outer(hv, hv))
    for i in range(M):
        E = np.zeros((M, M)); E[i, i] = RHO[i] * math.exp(scale[N + i]); dS.append(E)
    return dS


def dense_walk(Y, active=None):
    if active is None:
        active = np.ones(D, bool)
    offs, ws, cnts = zip(*[_axis_window(SS[k]) for k in range(D)])
    # joint tensor window (offsets and stationary weights)
    mesh = np.array(np.meshgrid(*offs, indexing="ij")).reshape(D, -1).T   # (G, D)
    w0 = ws[0]
    for k in range(1, D):
        w0 = np.kron(w0, ws[k])
    G = mesh.shape[0]
    # per-axis AR(1) transition on each window, Kron -> joint T
    def axis_T(off, s):
        nu = max(s * s * (1 - PHI[0] ** 2), 1e-12)
        T = np.exp(np.clip(-0.5 * (off[None, :] - PHI[0] * off[:, None]) ** 2 / nu, -700, 700))
        return T / T.sum(1, keepdims=True)
    T = axis_T(offs[0], SS[0])
    for k in range(1, D):
        T = np.kron(T, axis_T(offs[k], SS[k]))

    Ich = steady_fisher()
    Kstar = (1 - PHI) / 4.0
    qmu = Kstar ** 2 / (Ich * (1 - Kstar))
    cap = _GAP * SS

    mu = np.zeros(D); Pmu = SS ** 2
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    pi = w0.copy()
    out = np.zeros((len(Y), D))
    for t, y in enumerate(Y):
        pi = pi @ T
        scales = mu[None, :] + mesh                        # (G, D) absolute scales
        Qg = np.array([Q_of(s[:N]) for s in scales])
        Rg = np.array([R_of(s[N:]) for s in scales])
        Ppred = P[None] + Qg
        e = y - H @ m
        PHt = np.einsum("gij,kj->gik", Ppred, H)
        S = np.einsum("ij,gjk->gik", H, PHt) + Rg
        Si = np.linalg.inv(S)
        sgn, logdet = np.linalg.slogdet(S)
        maha = np.einsum("i,gij,j->g", e, Si, e)
        lg = -0.5 * (M * _LOG2PI + logdet + maha)
        w = pi * np.exp(lg - lg.max()); Z = w.sum(); pi = w / Z
        # state GPB1 collapse
        K = np.einsum("gik,gkl->gil", PHt, Si)
        Kbar = np.einsum("g,gil->il", pi, K)
        m_new = m + Kbar @ e
        mpost = m[None] + np.einsum("gil,l->gi", K, e)
        dm = mpost - m_new
        KH = np.einsum("gil,lj->gij", K, H)
        Ppost = Ppred - np.einsum("gij,gjk->gik", KH, Ppred)
        P = np.einsum("g,gij->ij", pi, Ppost) + np.einsum("g,gi,gj->ij", pi, dm, dm)
        P = 0.5 * (P + P.T); m = m_new
        # walk each active axis: residual score / expected-Fisher, marginalised, mu-loop
        Sie = np.einsum("gij,j->gi", Si, e)
        for k in range(D):
            if not active[k]:
                continue
            dpk = np.array([_dS(s)[k] for s in scales])              # (G, m, m)
            score_g = 0.5 * (np.einsum("gi,gij,gj->g", Sie, dpk, Sie)
                             - np.einsum("gij,gji->g", Si, dpk))
            # per-node expected Fisher for axis k:
            SidS = np.einsum("gij,gjk->gik", Si, dpk)
            info_gk = 0.5 * np.einsum("gij,gji->g", SidS, SidS)
            grad = float(pi @ score_g)
            info = float(pi @ info_gk) + 1e-4
            R_mu = 1.0 / info
            K_mu = Pmu[k] / (Pmu[k] + R_mu)
            mu[k] += float(np.clip(K_mu * (grad / info), -cap[k], cap[k]))
            Pmu[k] = (1 - K_mu) * Pmu[k] + qmu[k]
        out[t] = mu + pi @ mesh                                       # centre + window mean
    return out, G


if __name__ == "__main__":
    T = 900
    lab = ["xi1(weak)", "xi2(strong)", "eta1", "eta2"]
    _, G = dense_walk(gen(1, 0.0, T))
    print(f"joint dense-window nodes G = {G} (D={D}, span +/-{_SPAN_S}s at 1.5s)")
    print("\nSTATIC (want ~0):")
    w, _ = dense_walk(gen(1, 0.0, T))
    print(f"  WALK t>100: {w[100:].mean(0)}")
    b = slice(T // 3 + 40, 2 * T // 3 - 40)
    for ax, name in [(1, "strong xi2"), (3, "sensor eta2")]:
        Y = gen(ax, 1.4, T)
        ref = exact_grid(Y, order=5)
        w, _ = dense_walk(Y)
        cr = [np.corrcoef(w[30:, k], ref[30:, k])[0, 1] if w[30:, k].std() > 1e-9 else np.nan
              for k in range(D)]
        print(f"\n{name} hot (truth 1.4; fixed grid span-caps ~0.95):")
        print(f"  FIXED GRID: {ref[b].mean(0)}")
        print(f"  DENSE WALK: {w[b].mean(0)}   corr {np.array(cr)}")
