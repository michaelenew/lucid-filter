"""0010 -- the per-sensor read-out, shipped as an observer that cannot act.

`0004` and `0006` between them left one thing clearly worth having and clearly not safe to use.
The sensor entry is estimable to within a few percent at every `m`, and it is the one thing a
second-moment channel provably cannot report -- a scale sees only `e**2`, so a biased sensor
and its innocent neighbour move its `eta` the same way (`0001`: +0.71 / +0.74 at m = 2).  But
acting on it fails both ways: applied to the state it adopts the gauge convention and loses to
doing nothing, and merely left in the innovation it corrupts the process entry, which IS applied.

So it ships as an OBSERVER: the same two-stage recursion on the sensor entry's own quotient,
run beside the drift channel, whose every output is discarded.  It never corrects the state,
never corrects `y`, and never inflates what the members score against -- so it cannot change
the filter's behaviour, which is checked here bit-for-bit rather than argued.

Three things are measured:

  ACCURACY    across `m`, against the mean-centred truth.  The common mode is gauge on a
              random walk, so the read-out is "sensor i against the consensus" and that is what
              a caller can act on -- recalibrate it, drop it, or route around it.
  INVARIANCE  the filter's mean, variance, log-likelihood and drift estimate are bit-identical
              with the observer present and absent.
  THE STABLE CASE  where `H ker(F - I)` is empty, no bias is gauge and the read-out is
              ABSOLUTE rather than relative -- and it is exactly there that the drift channel
              is inert, so the two are complementary: whichever of the pair is identifiable is
              the one that is carried.
  CONVERGENCE the read-out sits 15-20% low at 400 steps of evidence, and the question is
              whether that is the prior's shrinkage or the estimate still arriving.  Widening
              the class ladder separates them: a wider one converges FASTER and lands in the
              same place, so it is the estimate arriving.  At 1700 steps the shipped ladder is
              within 1% of the truth.

Run: python3 0010_the_read_out.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from lucid import LucidFilter                                    # noqa: E402

Q_TRUE, R_TRUE, N, T0, BIAS, SEEDS = 0.02, 1.0, 700, 300, 2.0, (7, 8, 9, 10)


def rig(m, seed, total=N, drift=0.0):
    rng = np.random.default_rng(seed)
    theta = np.cumsum(rng.normal(0, np.sqrt(Q_TRUE), total)
                      + drift * (np.arange(total) >= T0))
    Y = np.stack([theta + rng.normal(0, np.sqrt(R_TRUE), total) for _ in range(m)], axis=1)
    Y[T0:, m - 1] += BIAS
    return theta, Y


def main():
    print("=" * 88)
    print(f"ACCURACY -- one level, m sensors, the last {BIAS} sigma off from t = {T0}")
    print("=" * 88)
    print(f"{'m':>2} | {'read-out (mean-centred)':<44} | truth")
    for m in (2, 3, 5, 8):
        acc = np.zeros(m)
        for seed in SEEDS:
            _, Y = rig(m, seed)
            r = LucidFilter(H=np.ones((m, 1)), measurement=np.ones(m),
                            offsets=True).filter(Y)
            acc += r.sensor_offset[-1] / len(SEEDS)
        want = np.full(m, -BIAS / m)
        want[-1] = BIAS * (m - 1) / m
        got = np.array2string(acc, precision=2, suppress_small=True)
        print(f"{m:2d} | {got:<44} | last {want[-1]:+.2f}, rest {want[0]:+.2f}")

    print()
    print("=" * 88)
    print("INVARIANCE -- the observer cannot change the filter, checked bit-for-bit")
    print("=" * 88)
    for m, drift in ((3, 0.0), (3, 0.10), (5, 0.10)):
        theta, Y = rig(m, 7, drift=drift)
        H, R0 = np.ones((m, 1)), np.ones(m)
        a = LucidFilter(H=H, measurement=R0, offsets=True)
        b = LucidFilter(H=H, measurement=R0, offsets=True)
        b._sensor = None
        ra, rb = a.filter(Y), b.filter(Y)
        same = (np.array_equal(ra.mean, rb.mean) and np.array_equal(ra.var, rb.var)
                and ra.loglik == rb.loglik and np.array_equal(ra.offset, rb.offset))
        print(f"  m = {m}, drift {drift:.2f}: mean / var / loglik / drift identical -> {same}"
              f"   (drift read {ra.offset[-1, 0]:+.3f})")

    print()
    print("=" * 88)
    print("THE STABLE CASE -- no bias is gauge there, and the drift channel is inert")
    print("=" * 88)
    F, H = np.array([[0.8]]), np.ones((2, 1))
    for bias in (0.0, 1.5):
        acc = np.zeros(2)
        inert = None
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            x, X = 0.0, []
            for _ in range(N):
                x = 0.8 * x + rng.normal(0, 0.3)
                X.append(x)
            X = np.array(X)
            Y = np.stack([X + rng.normal(0, 1.0, N) for _ in range(2)], axis=1)
            Y[T0:, 1] += bias
            f = LucidFilter(dynamics=F, H=H, measurement=np.ones(2), offsets=True)
            inert = f._mean is None
            acc += f.filter(Y).sensor_offset[-1] / len(SEEDS)
        print(f"  bias on sensor 2 = {bias:.1f}: read-out "
              f"{np.array2string(acc, precision=2, suppress_small=True):<14} "
              f"(ABSOLUTE, truth [0.0 {bias:.1f}]);  drift channel inert: {inert}")
    print()
    print("  the pair is complementary: where a drift is identifiable a sensor bias is gauge,")
    print("  and where a sensor bias is absolutely identifiable a drift is not carried at all.")

    print()
    print("=" * 88)
    print("CONVERGENCE -- is the shrinkage the prior's, or the estimate still arriving?")
    print("=" * 88)
    print(f"{'m':>2} {'steps of evidence':>18} | {'shipped':>9} {'ladder x10':>11} "
          f"{'x100':>8} | truth")
    for m in (3, 5):
        for total in (700, 2000):
            vals = []
            for mult in (1.0, 10.0, 100.0):
                acc = 0.0
                for seed in SEEDS:
                    _, Y = rig(m, seed, total)
                    f = LucidFilter(H=np.ones((m, 1)), measurement=np.ones(m), offsets=True)
                    if mult != 1.0:
                        f.reset()
                        so = f._sensor
                        step = (so.cls[-1] * mult / so.cls[0]) ** (1.0 / (so.cls.shape[0] - 1))
                        so.cls = np.stack([so.cls[0] * step ** j
                                           for j in range(so.cls.shape[0])])
                        so.q = 1e-4 * so.cls
                        so.reset()
                    acc += f.filter(Y).sensor_offset[-1, m - 1] / len(SEEDS)
                vals.append(acc)
            print(f"{m:2d} {total - T0:18d} | {vals[0]:9.3f} {vals[1]:11.3f} {vals[2]:8.3f} | "
                  f"{BIAS * (m - 1) / m:.3f}")
    print()
    print("  a wider ladder converges faster and lands in the SAME place, so the shrinkage at")
    print("  400 steps is the estimate arriving rather than a prior pulling it down.")


if __name__ == "__main__":
    main()
