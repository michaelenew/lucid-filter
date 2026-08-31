"""0012 -- is the gain that falls off at large drifts the ladder's ceiling, or the transient?

`0005` closes 84% of the distance to an oracle at r = 0.14 and only 49% at r = 1.00, and
`0007` sees the same shape on a velocity-side drift: the estimate stays accurate (0.064 against
0.06, three times the class ladder's top rung) while the RMSE repair shrinks.  Two readings fit
that, and they call for opposite fixes:

  CEILING   the ladder's widest rung is one process sd per step, so a drift far above it is
            represented only at the edge of the grid and tracked sluggishly.  Fix: raise it.
  TRANSIENT the estimate has to be REACHED, and a bigger drift takes longer to reach while
            doing more damage on the way.  The steady state would then be fine and the
            averaged window is carrying the approach.  Fix: nothing here -- it is the price of
            learning, and `0005`'s window simply includes it.

They are distinguishable by measurement.  Splitting the post-onset window into its approach and
its tail separates them, and raising the ceiling directly says whether the top rung binds.

Run: python3 0012_the_ceiling.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from lucid import LucidFilter                                    # noqa: E402

Q_TRUE, R_TRUE, N, T0, SEEDS = 0.02, 1.0, 1400, 300, (11, 12, 13, 14)


def rig(rate, seed):
    rng = np.random.default_rng(seed)
    theta = np.cumsum(rng.normal(0, np.sqrt(Q_TRUE), N) + rate * (np.arange(N) >= T0))
    return theta, (theta + rng.normal(0, np.sqrt(R_TRUE), N))[:, None]


def oracle(Y, rate):
    x, P = float(Y[0, 0]), R_TRUE
    mean, var = np.empty(N), np.empty(N)
    for t, row in enumerate(Y):
        x, P = x + (rate if t >= T0 else 0.0), P + Q_TRUE
        S = P + R_TRUE
        K = P / S
        x, P = x + K * (float(row[0]) - x), (1.0 - K) * P
        mean[t], var[t] = x, P
    return mean, var


def run(rate, ceiling=None):
    """`ceiling` multiplies the ladder's top rung; None is the shipped ladder."""
    win = {"approach": (T0, T0 + 200), "tail": (N - 400, N)}
    acc = {k: np.zeros(3) for k in win}
    dh = 0.0
    for seed in SEEDS:
        theta, Y = rig(rate, seed)
        f = LucidFilter(offsets=True)
        if ceiling is not None:
            f.reset()
            mc = f._mean
            top = mc.cls[-1]
            step = (top * ceiling / mc.cls[0]) ** (1.0 / (mc.cls.shape[0] - 1))
            mc.cls = np.stack([mc.cls[0] * step ** j for j in range(mc.cls.shape[0])])
            mc.q = 1e-4 * mc.cls
            mc.reset()
        on = f.filter(Y)
        off = LucidFilter().filter(Y)
        om, ov = oracle(Y, rate)
        dh += on.offset[-1, 0] / len(SEEDS)
        for name, (lo, hi) in win.items():
            for j, (mm, vv) in enumerate(((off.mean[:, 0], off.var[:, 0, 0]),
                                          (on.mean[:, 0], on.var[:, 0, 0]), (om, ov))):
                e = mm[lo:hi] - theta[lo:hi]
                acc[name][j] += np.sqrt(np.mean(e ** 2)) / len(SEEDS)
    return acc, dh


def gap(a):
    return (a[0] - a[1]) / (a[0] - a[2]) if (a[0] - a[2]) > 1e-9 else float("nan")


def main():
    print("=" * 84)
    print("Is it the ceiling or the transient?  The shipped ladder, split by window")
    print("=" * 84)
    print(f"{'drift':>6} | {'approach (200 steps after onset)':>34} | {'tail (last 400)':>26}")
    print(f"{'':>6} | {'off':>8} {'on':>8} {'oracle':>8} {'gap':>6} | "
          f"{'off':>8} {'on':>8} {'gap':>6}")
    for rate in (0.14, 0.42, 1.00, 2.00):
        acc, dh = run(rate)
        a, t = acc["approach"], acc["tail"]
        print(f"{rate:6.2f} | {a[0]:8.3f} {a[1]:8.3f} {a[2]:8.3f} {gap(a):5.0%} | "
              f"{t[0]:8.3f} {t[1]:8.3f} {gap(t):5.0%}")
    print()
    print("  if the tail closes and the approach does not, the falloff is the price of")
    print("  reaching the estimate, not the grid's top rung.")

    print()
    print("=" * 84)
    print("And directly: does raising the ladder's ceiling change anything?")
    print("=" * 84)
    print(f"{'drift':>6} | {'ceiling x1 (shipped)':>28} | {'x10':>16} | {'x100':>16}")
    print(f"{'':>6} | {'tail gap':>10} {'d_hat':>8} | {'tail gap':>7} {'d_hat':>7} | "
          f"{'tail gap':>7} {'d_hat':>7}")
    for rate in (0.42, 1.00, 2.00):
        cells = []
        for mult in (None, 10.0, 100.0):
            acc, dh = run(rate, mult)
            cells.append((gap(acc["tail"]), dh))
        print(f"{rate:6.2f} | {cells[0][0]:9.0%} {cells[0][1]:8.3f} | "
              f"{cells[1][0]:6.0%} {cells[1][1]:7.3f} | {cells[2][0]:6.0%} {cells[2][1]:7.3f}")
    print()
    print("  the truth is in the `drift` column; a ceiling that binds would show as the")
    print("  wider ladders recovering more of it.")


if __name__ == "__main__":
    main()
