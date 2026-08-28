"""scratch_04_reach_ar1.py -- (a) reach-time law vs s,B ; (b) AR(1) self-consistency."""
from __future__ import annotations
import math, sys
import numpy as np
from scratch_core_vec import (MemberVec, OracleVec, _sim_batch, gen_batch_ar1,
                              run_vec, rmse_ratio)

SS = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.2, 1.5, 2.0])


def gen_perm(rng, S, n, B, t0, Q=1.0, R0=1.0):
    eta = np.zeros(n); eta[t0:] = B
    return _sim_batch(rng, S, eta, Q, R0)


def reach_time(phi, B, s, S=128, n=1500, t0=300, nodes=7, frac=0.9):
    """Median steps after t0 until mean mu crosses frac*B."""
    rng = np.random.default_rng(7)
    theta, x, eta = gen_perm(rng, S, n, B, t0)
    em, eo, mus = run_vec(theta, x, eta, phi, s, nodes)
    mbar = mus[t0:].mean(1)                 # mean mu across seeds after onset
    idx = np.argmax(mbar >= frac * B)
    if mbar.max() < frac * B:
        return None
    return int(idx)


def reach_section():
    phi = 0.85
    print("############ reach-time law: steps to climb to 0.9B (phi=0.85, nodes=7) ###")
    print("  climb-law prediction: cap-limited rate = 1.5*s/step -> t ~ B/(1.5 s)")
    print("   s      B=2   B=5   B=8    | B/(1.5s):  B=2  B=5  B=8")
    for s in [0.15, 0.2, 0.3, 0.5, 0.8, 1.2]:
        ts = [reach_time(phi, B, s) for B in [2.0, 5.0, 8.0]]
        pred = [B / (1.5 * s) for B in [2.0, 5.0, 8.0]]
        tt = "  ".join(f"{t if t is not None else -1:4d}" for t in ts)
        pp = "  ".join(f"{p:4.0f}" for p in pred)
        print(f"  {s:4.2f}    {tt}    |            {pp}")
    sys.stdout.flush()


def ar1_selfconsistency():
    print("\n############ Q2: AR(1) data family -- is member=(phi_d,s_d) optimal? ####")
    phis_d = [0.85, 0.95, 0.99]
    ss_d = [0.2, 0.4, 0.8, 1.5]
    n = 6000; S = 96; burn = 1000
    for phi_d in phis_d:
        print(f"\n--- data phi_d={phi_d}: member phi = phi_d, sweep member s ---")
        print("  s_true | " + "  ".join(f"sd={sd}" for sd in ss_d) + "   (row=member s; ratio)")
        # build data once per (phi_d, s_d)
        tabs = {}
        for sd in ss_d:
            rng = np.random.default_rng(int(1000*phi_d) + int(100*sd))
            theta, x, eta = gen_batch_ar1(rng, S, n, phi_d, sd)
            mask = np.zeros(n, bool); mask[burn:] = True
            col = np.zeros(len(SS))
            for i, sm in enumerate(SS):
                em, eo, mus = run_vec(theta, x, eta, phi_d, sm, nodes=7)
                col[i], _ = rmse_ratio(em, eo, mask)
            tabs[sd] = col
        for i, sm in enumerate(SS):
            marks = ""
            row = "  ".join(f"{tabs[sd][i]:.3f}" for sd in ss_d)
            # mark the s closest to each sd's column-min
            flags = []
            for sd in ss_d:
                flags.append("*" if i == int(np.argmin(tabs[sd])) else " ")
            print(f"  {sm:5.2f} | " + "  ".join(f"{tabs[sd][i]:.3f}{flags[k]}"
                                                for k, sd in enumerate(ss_d)))
        # report argmin s vs s_d
        print("  argmin member-s per data s_d:")
        for sd in ss_d:
            istar = int(np.argmin(tabs[sd]))
            print(f"     s_d={sd}: s*={SS[istar]:.2f}  (ratio {tabs[sd][istar]:.3f})")
        sys.stdout.flush()


if __name__ == "__main__":
    reach_section()
    ar1_selfconsistency()
