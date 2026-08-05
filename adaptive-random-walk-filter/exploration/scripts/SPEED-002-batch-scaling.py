"""SPEED-002: if the recursion is dispatch-bound, is a *batch* of parameter
vectors nearly free?

SPEED-001 showed the per-step cost is flat in the grid size G: order 3 (G=9) to
order 13 (G=169) is 18.8x the arithmetic for 1.5x the time.  The recursion is
sequential in t and cannot be vectorised over time -- but nothing stops it being
vectorised over *parameter vectors*.  Every evaluation walks the same series
with the same number of numpy calls; only the array shape changes, from (G,) to
(B, G).

If the cost is flat in B over the range an optimiser actually needs (B ~ 10 for
a finite-difference gradient, B ~ 100 for a start scan), then the entire local
geometry of the 6-D surface costs one evaluation, and the parameter search
should be organised around batches rather than points.

Prototype only; the version that ships lives in statfilter.core.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "output"))

from statfilter import AdaptiveFilter, Params                       # noqa: E402
from statfilter.core import _gauss_hermite, _LOG2PI                 # noqa: E402


def chains(phi, s, n):
    """_chain, vectorised over a batch of (phi, s).  phi, s are (B,)."""
    z, w = _gauss_hermite(n)
    lam = s[:, None] * z                                            # (B, n)
    nu = np.maximum(s * s * (1.0 - phi * phi), 1e-12)[:, None, None]
    ex = (-0.5 * (lam[:, None, :] - phi[:, None, None] * lam[:, :, None]) ** 2 / nu
          + 0.5 * lam[:, None, :] ** 2 / (s * s)[:, None, None])
    T = w * np.exp(np.clip(ex, -700.0, 700.0))
    T /= T.sum(2, keepdims=True)
    flat = s <= 0.0
    if flat.any():
        lam[flat] = 0.0
        T[flat] = w
    return lam, np.broadcast_to(w, lam.shape).copy(), T


def loglik_batch(x, V, order=5):
    """Total log-likelihood of x for every unconstrained vector in V: (B, 6)."""
    V = np.atleast_2d(np.asarray(V, dtype=float))
    Q, S2 = np.exp(V[:, 0]), np.exp(V[:, 1])
    phP, phM = 1.0 / (1.0 + np.exp(-V[:, 2])), 1.0 / (1.0 + np.exp(-V[:, 3]))
    sP, sM = np.exp(V[:, 4]), np.exp(V[:, 5])
    n = order
    lamP, wP, TP = chains(phP, sP, n)
    lamM, wM, TM = chains(phM, sM, n)
    LP = np.repeat(lamP, n, axis=1)
    LM = np.tile(lamM, (1, n))
    T = (TP[:, :, None, :, None] * TM[:, None, :, None, :]).reshape(-1, n * n, n * n)
    pi0 = (wP[:, :, None] * wM[:, None, :]).reshape(-1, n * n)

    Qg = Q[:, None] * np.exp(np.clip(LP, -60.0, 60.0))
    Rg = S2[:, None] * np.exp(np.clip(LM, -60.0, 60.0))
    QR = Qg + Rg

    pi = pi0.copy()
    m = np.full(V.shape[0], float(x[0]))
    P = (Rg.max(1) + Qg.max(1))
    ll = np.zeros(V.shape[0])
    for t in range(x.size):
        pi = np.einsum("bi,bij->bj", pi, T)
        Pp = P[:, None] + Qg
        S = P[:, None] + QR
        e = x[t] - m
        lg = -0.5 * (np.log(S) + (e * e)[:, None] / S)
        mx = lg.max(1)
        w = pi * np.exp(lg - mx[:, None])
        Z = w.sum(1)
        ll += np.log(Z) + mx - 0.5 * _LOG2PI
        pi = w / Z[:, None]
        K = Pp / S
        Kbar = (pi * K).sum(1)
        m = m + Kbar * e
        P = (pi * ((1.0 - K) * Pp)).sum(1) + e * e * (pi * (K - Kbar[:, None]) ** 2).sum(1)
    return ll


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 1200
    x = np.cumsum(rng.standard_normal(n) * np.sqrt(0.05)) + rng.standard_normal(n)

    v0 = np.array([np.log(0.05), 0.0, 0.0, 0.0, np.log(0.6), np.log(0.6)])

    # agreement with the shipped scalar path
    p = Params(Q=0.05, s2=1.0, phi_P=0.5, phi_M=0.5, s_P=0.6, s_M=0.6)
    ref = AdaptiveFilter(p, order=5).loglik(x)
    got = float(loglik_batch(x, v0[None, :])[0])
    print(f"agreement with shipped loglik: {ref:.10f} vs {got:.10f} "
          f"(rel {abs(ref - got) / abs(ref):.2e})")

    print("\ncost of a batch of B parameter vectors, n = 1200, order 5 (G = 25)")
    print(f"   {'B':>5} {'B*G':>6} {'ms':>8} {'ms/vector':>11} {'vs B=1':>8}")
    one = None
    for B in (1, 2, 4, 7, 13, 25, 50, 100, 200, 400):
        V = v0 + rng.standard_normal((B, 6)) * 0.02
        loglik_batch(x[:20], V)
        t0 = time.perf_counter()
        loglik_batch(x, V)
        dt = time.perf_counter() - t0
        one = dt if one is None else one
        print(f"   {B:>5} {B * 25:>6} {1000 * dt:>8.1f} {1000 * dt / B:>11.2f} "
              f"{dt / one:>8.2f}x")
