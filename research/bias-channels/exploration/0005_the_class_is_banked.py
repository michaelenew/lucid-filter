"""0005 -- the offset's class cannot be chosen, so it is banked.

The two-stage channel of `0003` has exactly one quantity that is not fixed by the structure:
how big an offset is plausible a priori -- its CLASS.  This probe is the record of getting
that wrong twice and of what replaced it.

ATTEMPT 1, one noise sd per step (`cls = Q0`).  The scale-free convention the departure walker
uses -- "this part changed by about its own magnitude".  Measured: 71% of the distance to an
oracle told the drift, at an 11% RMSE premium on driftless data.

ATTEMPT 2, the memory's resolution floor (`cls = Q0 / T`, `T = 1/(1 - forget)`).  Derived, and
derived from the right kind of argument: over the filter's own memory a constant contributes
`T c` against the noise's `sqrt(V T)`, so `c^2 = V/T` is where "there is an offset" and "the
noise wandered" are the same claim.  Measured: the premium vanishes (0.6%) and so does the
channel -- the drift estimate saturates at a tenth of the truth.

The second is a DETECTABILITY limit used as a PRIOR WIDTH, which is the wrong instrument: it
places the prior at the edge of what can be seen rather than over what is plausible.  But the
first is not defensible either, and the sweep below says why -- its good behaviour comes from
the DEFAULT BASE being 50x looser than the truth on this rig.  A caller who supplied a tight,
accurate `process=` would get attempt 2's behaviour from attempt 1's rule.  There is no fixed
class that is right for both callers, because the class is not a property of the mechanism.

So it is a nuisance, and this filter has one way of handling a nuisance: grid it and let the
evidence weight it -- the same move as `lam_P`, `lam_M`, `lam_A` and the `(phi, s)` box.  The
shipped channel runs `_OFFSET_CLASSES` copies of the recursion at geometrically spaced widths
between the two ends above (both derived: the resolution floor and one noise sd per step) and
mixes them by their own predictive likelihood on the bank's `forget` timescale.  The sensitivity
`V` and the regressor `U` do not depend on the class, so a rung costs one k-dimensional update.

The result is the point of the file: the banked ladder is not a compromise between the two ends
but better than both -- the wide end's recovery AND the narrow end's premium.

Run: python3 0005_the_class_is_banked.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from lucid import LucidFilter                                    # noqa: E402

Q_TRUE, R_TRUE, N, T0, SEEDS = 0.02, 1.0, 900, 400, (11, 12, 13, 14)


def rig(rate, seed):
    rng = np.random.default_rng(seed)
    theta = np.cumsum(rng.normal(0, np.sqrt(Q_TRUE), N) + rate * (np.arange(N) >= T0))
    return theta, (theta + rng.normal(0, np.sqrt(R_TRUE), N))[:, None]


def oracle(y, drift):
    """A Kalman filter handed the true noises AND the true drift -- the achievable floor."""
    x, P = float(y[0, 0]), R_TRUE
    mean, var = np.empty(len(y)), np.empty(len(y))
    for t, row in enumerate(y):
        x, P = x + drift, P + Q_TRUE
        S = P + R_TRUE
        K = P / S
        x, P = x + K * (float(row[0]) - x), (1.0 - K) * P
        mean[t], var[t] = x, P
    return mean, var


def score(mean, var, theta, lo=T0 + 100):
    e = mean[lo:] - theta[lo:]
    return np.sqrt(np.mean(e ** 2)), np.mean(e ** 2 / var[lo:])


def run(rate, cls=None):
    """`cls=None` runs the shipped banked ladder; a float pins every rung to one width."""
    acc = np.zeros(5)
    for seed in SEEDS:
        theta, y = rig(rate, seed)
        f = LucidFilter(offsets=True)
        if cls is not None:
            f.reset()
            k = f._mean.k
            f._mean.cls = np.full((f._mean.cls.shape[0], k), cls)
            f._mean.q = 1e-4 * f._mean.cls
            f._mean.reset()
        r = f.filter(y)
        acc[:2] += np.array(score(r.mean[:, 0], r.var[:, 0, 0], theta)) / len(SEEDS)
        acc[4] += r.offset[-1, 0] / len(SEEDS)
        om, ov = oracle(y, rate)
        acc[2:4] += np.array(score(om, ov, theta)) / len(SEEDS)
    return acc


def baseline(rate):
    acc = np.zeros(2)
    for seed in SEEDS:
        theta, y = rig(rate, seed)
        r = LucidFilter().filter(y)
        acc += np.array(score(r.mean[:, 0], r.var[:, 0, 0], theta)) / len(SEEDS)
    return acc


def main():
    print("=" * 84)
    print("A FIXED class: the sweep that refuses every value  (base Q0 = 1.0, true Q = 0.02)")
    print("=" * 84)
    off0, off_d = baseline(0.0), baseline(0.42)
    print(f"{'class':>10} | {'driftless rmse':>14} {'premium':>8} | "
          f"{'r = 0.42 rmse':>13} {'d_hat':>7} | gap closed")
    for c in (1.0, 1e-1, 1e-2, 1e-3):
        a, b = run(0.0, c), run(0.42, c)
        gap = (off_d[0] - b[0]) / (off_d[0] - b[2])
        print(f"{c:10.0e} | {a[0]:14.3f} {a[0]/off0[0] - 1:+7.1%} | "
              f"{b[0]:13.3f} {b[4]:7.3f} | {gap:6.0%}")
    print(f"{'channel off':>10} | {off0[0]:14.3f} {0.0:+7.1%} | {off_d[0]:13.3f} "
          f"{'--':>7} | {0.0:6.0%}")
    print("\nThe wide end buys the recovery and costs 11%; the narrow end costs nothing and")
    print("buys nothing.  And the wide end is only good because the BASE is 50x loose here.")

    print()
    print("=" * 84)
    print("The BANKED ladder, which is better than both ends rather than between them")
    print("=" * 84)
    print(f"{'drift r':>8} | {'off rmse calib':>16} | {'on  rmse calib':>16} | "
          f"{'oracle':>7} | {'d_hat':>7} | gap")
    for rate in (0.0, 0.05, 0.14, 0.42, 1.0):
        b = run(rate)
        o = baseline(rate)
        gap = ((o[0] - b[0]) / (o[0] - b[2])) if (o[0] - b[2]) > 1e-9 else float("nan")
        tag = f"{gap:5.0%}" if np.isfinite(gap) else f"{b[0]/o[0] - 1:+5.1%}"
        print(f"{rate:8.2f} | {o[0]:9.3f} {o[1]:6.2f} | {b[0]:9.3f} {b[1]:6.2f} | "
              f"{b[2]:7.3f} | {b[4]:7.3f} | {tag}")
    print("\n(the last column is the fraction of the off-to-oracle gap closed, except at r = 0")
    print(" where there is no gap and it reports the premium instead)")

    print()
    print("=" * 84)
    print("Cost")
    print("=" * 84)
    rng = np.random.default_rng(3)
    y = rng.normal(size=(150, 1))
    for label, kw in (("channel off", {}), ("channel on", {"offsets": True})):
        LucidFilter(**kw).filter(y[:20])
        f = LucidFilter(**kw)
        t = time.time()
        f.filter(y)
        print(f"  {label:<12} {(time.time() - t) / len(y) * 1000:5.2f} ms/step")


if __name__ == "__main__":
    main()
