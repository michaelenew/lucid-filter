"""SPEED-001: where does fit() actually spend its time?

SUMMARY.md asserts "~1,300 likelihood evaluations, 98% numpy dispatch on
25-element arrays".  Both halves of that claim are load-bearing for any speed
work, so measure them rather than inherit them.

Counts every likelihood evaluation fit() makes, by stage, and times the inner
recursion at several series lengths and quadrature orders.  If the per-step cost
is dispatch-bound it will be nearly flat in the grid size G = order^2, and the
cost per evaluation will be linear in n with a large constant.

Run from the workstream root:  python exploration/scripts/SPEED-001-baseline-profile.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ROOT))          # random-walk-filter
sys.path.insert(0, os.path.join(REPO, "lucid"))

from statfilter import AdaptiveFilter, Params          # noqa: E402
from statfilter import core as _core                   # noqa: E402


def series(n, seed=0, q=0.05, s2=1.0):
    rng = np.random.default_rng(seed)
    th = np.cumsum(rng.standard_normal(n) * np.sqrt(q))
    return th + rng.standard_normal(n) * np.sqrt(s2)


# ------------------------------------------------- 1. evaluation count by stage
def count_evals(x):
    """Wrap _run to count and time every likelihood evaluation fit() makes."""
    f = AdaptiveFilter(order=5)
    calls = {"n": 0, "t": 0.0}
    real = _core.AdaptiveFilter._run

    def counting(self, xx, want):
        t0 = time.perf_counter()
        out = real(self, xx, want)
        calls["t"] += time.perf_counter() - t0
        calls["n"] += 1
        return out

    _core.AdaptiveFilter._run = counting
    try:
        t0 = time.perf_counter()
        f.fit_(x)
        wall = time.perf_counter() - t0
    finally:
        _core.AdaptiveFilter._run = real
    return calls["n"], calls["t"], wall, f.params


# ---------------------------------------------- 2. is the inner loop dispatch-bound?
def time_eval(n, order, reps=3):
    x = series(n)
    f = AdaptiveFilter(Params(0.05, 1.0, 0.5, 0.5, 0.6, 0.6), order=order)
    f.loglik(x[:20])                                    # warm the grid cache
    t0 = time.perf_counter()
    for _ in range(reps):
        f.loglik(x)
    return (time.perf_counter() - t0) / reps


if __name__ == "__main__":
    print("1. fit() cost decomposition, n = 1200")
    x = series(1200)
    nev, tev, wall, p = count_evals(x)
    print(f"   likelihood evaluations : {nev}")
    print(f"   time inside them       : {tev:.1f} s  ({100 * tev / wall:.1f}% of fit)")
    print(f"   wall clock for fit()   : {wall:.1f} s")
    print(f"   per evaluation         : {1000 * tev / nev:.1f} ms")
    print(f"   fitted                 : {p}")

    print("\n2. per-step cost vs grid size G = order^2 (n = 1200)")
    print(f"   {'order':>6} {'G':>5} {'us/step':>9} {'us/step/G':>11}")
    base = None
    for order in (3, 5, 7, 9, 13):
        t = time_eval(1200, order)
        us = 1e6 * t / 1200
        base = us if base is None else base
        print(f"   {order:>6} {order ** 2:>5} {us:>9.2f} {us / order ** 2:>11.3f}")
    print("   flat us/step => dispatch-bound; flat us/step/G => arithmetic-bound")

    print("\n3. per-evaluation cost vs n (order 5)")
    print(f"   {'n':>7} {'ms':>8} {'us/step':>9}")
    for n in (300, 1200, 4800):
        t = time_eval(n, 5)
        print(f"   {n:>7} {1000 * t:>8.1f} {1e6 * t / n:>9.2f}")
