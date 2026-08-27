"""scratch_03_decomp.py -- decompose s* into STEADY (jitter) vs TRANSIENT (reach).

A permanent shift eta: 0 -> B at t0, held to the end.
  - STEADY window: [t0 + settle, end]   -> jitter-optimal s (B-dependent?)
  - TRANSIENT window: [t0, t0 + settle]  -> reach-optimal s (favors large s)
Also: phi sweep of the steady optimum; nodes=1 (pure walk) vs nodes=7 (window).
"""
from __future__ import annotations
import math, sys
import numpy as np
from scratch_core_vec import (MemberVec, OracleVec, _sim_batch, run_vec, rmse_ratio)

SS = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.2, 1.5, 2.0])


def gen_perm(rng, S, n, B, t0, Q=1.0, R0=1.0):
    eta = np.zeros(n); eta[t0:] = B
    return _sim_batch(rng, S, eta, Q, R0)


def steady_transient(phi, B, S=64, n=4000, t0=500, settle=400, nodes=7):
    rng = np.random.default_rng(999)
    theta, x, eta = gen_perm(rng, S, n, B, t0)
    tmask = np.zeros(n, bool); tmask[t0:t0 + settle] = True
    smask = np.zeros(n, bool); smask[t0 + settle:] = True
    sm = np.zeros(len(SS)); sse = np.zeros_like(sm)
    tm = np.zeros_like(sm); tse = np.zeros_like(sm)
    for i, s in enumerate(SS):
        em, eo, mus = run_vec(theta, x, eta, phi, s, nodes)
        sm[i], sse[i] = rmse_ratio(em, eo, smask)
        tm[i], tse[i] = rmse_ratio(em, eo, tmask)
    return sm, sse, tm, tse


def argmin_pm(m, se):
    i = int(np.argmin(m))
    return SS[i], m[i], se[i]


if __name__ == "__main__":
    phi = 0.85
    print("############ Q3-core: STEADY vs TRANSIENT optima, phi=0.85, nodes=7 ###########")
    print("\n--- STEADY (jitter) optimum: is it B-independent? ---")
    print("  B      s_steady*   ratio         | s_transient*  ratio")
    steady_star = {}
    for B in [2.0, 3.5, 5.0, 6.5, 8.0]:
        sm, sse, tm, tse = steady_transient(phi, B)
        ss, sv, sse = argmin_pm(sm, sse)
        ts, tv, tsee = argmin_pm(tm, tse)
        steady_star[B] = ss
        print(f"  {B:4.1f}    {ss:5.2f}   {sv:.3f}+-{sse:.3f}   |  {ts:5.2f}   {tv:.3f}+-{tsee:.3f}")
    print("\nfull STEADY table (ratio vs s) at each B:")
    print("   s   " + "  ".join(f"B={B}" for B in [2.0,3.5,5.0,6.5,8.0]))
    tabs = {B: steady_transient(phi, B)[0] for B in [2.0,3.5,5.0,6.5,8.0]}
    for i, s in enumerate(SS):
        print(f" {s:4.2f}  " + "  ".join(f"{tabs[B][i]:.3f}" for B in [2.0,3.5,5.0,6.5,8.0]))
    sys.stdout.flush()

    print("\n--- phi sweep of steady optimum (B=5 permanent) ---")
    print("  phi    Kstar    s_steady*   ratio")
    for phi in [0.7, 0.85, 0.95, 0.99]:
        sm, sse, tm, tse = steady_transient(phi, 5.0)
        ss, sv, sse = argmin_pm(sm, sse)
        print(f"  {phi:4.2f}   {(1-phi)/4:.4f}  {ss:5.2f}   {sv:.3f}+-{sse:.3f}")

    print("\n--- nodes=1 (PURE WALK, no window): steady & transient, phi=0.85 ---")
    print("  B      s_steady*(n1)   s_transient*(n1)")
    for B in [2.0, 5.0, 8.0]:
        sm, sse, tm, tse = steady_transient(0.85, B, nodes=1)
        ss, *_ = argmin_pm(sm, sse); ts, *_ = argmin_pm(tm, tse)
        print(f"  {B:4.1f}    {ss:5.2f}            {ts:5.2f}")
