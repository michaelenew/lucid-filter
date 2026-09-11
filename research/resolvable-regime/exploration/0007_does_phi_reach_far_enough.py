"""0007 -- the last live question: does the shipped `phi` box reach near enough to 1?

0003's result is that a stationary class costs nothing on a WANDERING truth.  But
look at how it achieved that: the fit went to `phi = 0.995`, the top rung of
0003's own grid -- rail-pinned.  The class did not need to BE non-stationary; it
needed to be allowed close enough to the boundary.

The shipped box stops at `phi = 0.95` (a reversion timescale of 19.5 steps).
0003's grid reached 0.995 (195 steps).  If 0.95 is not close enough, then this
workstream's one surviving product consequence is not "drop stationarity" but the
much smaller "the persistence axis should reach nearer 1" -- and if 0.95 IS close
enough, the thread has no product consequence at all and should say so.

Same rig, same GPB1 grid filter, same wandering truths as 0003.  Three stationary
grids, differing only in how close to 1 they may reach:

    shipped-like   phi <= 0.95
    extended       phi <= 0.995   (0003's grid)
    boundary       phi <= 0.9995  (reversion timescale 2000 steps, half the record)

plus the correctly-specified wandering class as the reference.

Run: python3 0007_does_phi_reach_far_enough.py
"""
import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module                               # noqa: E402

_m = import_module("0003_what_stationarity_costs".replace("0003", "0003"))  # same dir
sim_wandering = _m.sim_wandering
sim_stationary = _m.sim_stationary
run_filter = _m.run_filter
T, SIGMAS, SS = _m.T, _m.SIGMAS, _m.SS
NSEED = 8

GRIDS = {
    "shipped-like  phi <= 0.95 ": np.array([0.60, 0.75, 0.85, 0.92, 0.95]),
    "extended      phi <= 0.995": np.array([0.60, 0.75, 0.85, 0.92, 0.96, 0.98, 0.995]),
    "boundary      phi <= 0.9995": np.array([0.60, 0.75, 0.85, 0.92, 0.96, 0.98,
                                             0.995, 0.9995]),
}


def fit_grid(xs, phis):
    best = (-np.inf, None, None, None)
    for phi in phis:
        for s in SS:
            ll, est = run_filter(xs, phi, s=s)
            if ll > best[0]:
                best = (ll, phi, s, est)
    return best


def fit_wander(xs):
    best = (-np.inf, None, None)
    for sg in SIGMAS:
        ll, est = run_filter(xs, 1.0, sigma=sg)
        if ll > best[0]:
            best = (ll, sg, est)
    return best


def cell(name, maker, args):
    out = {k: [] for k in GRIDS}
    out["wandering (correct class)"] = []
    rails = {k: [] for k in GRIDS}
    for seed in range(NSEED):
        xs, lams, oracle = maker(*args, seed)
        ref = None
        for k, phis in GRIDS.items():
            ll, phi, s, est = fit_grid(xs, phis)
            mse = float(np.mean((est - oracle) ** 2))
            out[k].append(mse)
            rails[k].append(1.0 if phi == phis[-1] else 0.0)
            if ref is None:
                ref = mse
        ll, sg, est = fit_wander(xs)
        out["wandering (correct class)"].append(float(np.mean((est - oracle) ** 2)))
    base = np.mean(out["shipped-like  phi <= 0.95 "])
    print(f"\n--- truth: {name}   ({NSEED} seeds, T={T})")
    print(f"  {'class':30s}{'MSE':>10s}{'vs shipped-like':>18s}{'rail-pinned':>13s}")
    for k in list(GRIDS) + ["wandering (correct class)"]:
        a = np.array(out[k])
        r = a.mean() / base
        sem = a.std(ddof=1) / math.sqrt(NSEED) / base
        rp = f"{np.mean(rails[k]) * 100:.0f}%" if k in rails else "-"
        print(f"  {k:30s}{a.mean():10.5f}{r:12.4f} +-{sem:5.4f}{rp:>13s}")
    return {k: np.mean(out[k]) / base for k in out}


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    cell("WANDERING sigma=0.03 (kappa = 0)", sim_wandering, (0.03,))
    cell("WANDERING sigma=0.06 (kappa = 0)", sim_wandering, (0.06,))
    cell("STATIONARY phi=0.96 s=0.55", sim_stationary, (0.96, 0.55))
    print("\nIf `shipped-like` matches `extended`/`boundary` on the WANDERING truths,")
    print("the phi axis already reaches far enough and this thread has no product")
    print("consequence.  If it loses, the consequence is one more phi rung, not a")
    print("new axiom.")
