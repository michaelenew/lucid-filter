"""0008 -- the offset channel beside the filter's other two, and at arm scale.

The channel rides on the collapsed output, so it reaches the dynamics channel and the split
ladder BY CONSTRUCTION -- and that is not the same as measured, which is the SUMMARY's open 4
(and `sequence-demix` open 7 in a new place).  This probe closes it, and open 5 with it.

One of the two is already discharged and was not noticed: the SPLIT LADDER is on in every
scalar measurement in this workstream.  A scalar direct-observation rig is exactly the
structure the ladder switches on for -- one process mode read by exactly one sensor -- so
`LucidFilter()` carries 24 rungs and 360 members, and 0001, 0005, 0006 and 0007 all ran on it.
Printed below rather than asserted.

What is genuinely untested is the DYNAMICS channel, and it turned up a defect and its fix.  A
constant added to the prediction and a departure in `F` are two ways to explain the same
feature, so under FEEDBACK a departure walker adapts `F` to cancel the injected offset; the two
settle into a stable and wrong equilibrium (the offset estimate climbing to +0.09 on a driftless
series against -0.004 with the dynamics channel off), and the walker's adaptation registers as a
fault that the bank's thousand-step memory then keeps.  Over eight seeds, one locked `fault` at
1.000.

Two repairs were tried and measured before the third worked, and both are worth keeping:
masking the departure walkers out of the gain the channel reads (they are not the caller's axes
-- the rule the split ladder already follows) moved the mean fault 0.37 -> 0.36, and pushing the
offset's own variance into the members' predictive covariance, so a guess is not handed over as
a fact, did not move it either.  Neither touches the cause, which is that the two channels are
CONFOUNDED rather than interfering.

What works is structural: feedback is worth about twice the state repair of correcting only the
output (0.392 against 0.471, where doing nothing is 0.559), so the filter uses it -- and turns
it off exactly when the dynamics channel is on, where it is not available.  Both repairs above
are kept anyway: they are right on their own terms.

And ARM SCALE (open 5): the channel's cost is O(n^2 k + n m k) ONCE per step against the star's
per-member cost, so it should vanish into the noise as the rig grows -- the opposite of the
state augmentation `0002` priced at 2.06-2.87x on the same rig.  Measured here rather than
asserted.  The rig is a kinematic arm-scale model (10 states, 5 sensors), NOT the demo arm of
`multivariate-statfilter/0052`; that guard stays open.

Run: python3 0008_with_the_other_channels_on.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from lucid import LucidFilter                                    # noqa: E402

Q_TRUE, R_TRUE, N, T0, SEEDS = 0.02, 1.0, 700, 300, (11, 12, 13)


def drift_rig(rate, seed):
    rng = np.random.default_rng(seed)
    theta = np.cumsum(rng.normal(0, np.sqrt(Q_TRUE), N) + rate * (np.arange(N) >= T0))
    return theta, (theta + rng.normal(0, np.sqrt(R_TRUE), N))[:, None]


def score(kw, rate):
    acc = np.zeros(4)
    for seed in SEEDS:
        theta, Y = drift_rig(rate, seed)
        r = LucidFilter(**kw).filter(Y)
        lo = T0 + 100
        e = r.mean[lo:, 0] - theta[lo:]
        acc[0] += np.sqrt(np.mean(e ** 2)) / len(SEEDS)
        acc[1] += np.mean(e ** 2 / r.var[lo:, 0, 0]) / len(SEEDS)
        acc[2] += (r.offset[-1, 0] if r.offset is not None else 0.0) / len(SEEDS)
        acc[3] += (float(np.mean(r.fault)) if r.fault is not None else 0.0) / len(SEEDS)
    return acc


def kinematic(n_dof, dt=0.02):
    n = 2 * n_dof
    F = np.eye(n)
    for j in range(n_dof):
        F[2 * j, 2 * j + 1] = dt
    H = np.zeros((n_dof, n))
    for j in range(n_dof):
        H[j, 2 * j] = 1.0
    return F, H


def main():
    f = LucidFilter()
    print("=" * 92)
    print("OPEN 4a -- the split ladder was never off")
    print("=" * 92)
    print(f"  LucidFilter(): confounded pairs {f.groups}, {len(f.split_arr)} rungs, "
          f"{len(f._members)} members")
    print("  -- so 0001, 0005, 0006 and 0007's scalar rigs all ran with the ladder active.")

    print()
    print("=" * 92)
    print("OPEN 4b -- the DYNAMICS channel on, where the members carry an augmented state")
    print("=" * 92)
    print(f"{'configuration':<40} | {'rmse':>7} {'calib':>6} | {'d_hat':>7} | {'fault':>6}")
    for label, kw in (
        ("plain", {}),
        ("plain + offsets", {"offsets": True}),
        ("faults=1e-4", {"faults": 1e-4}),
        ("faults=1e-4 + offsets", {"faults": 1e-4, "offsets": True}),
        ("dynamics=None", {"dynamics": None}),
        ("dynamics=None + offsets", {"dynamics": None, "offsets": True}),
    ):
        for rate in (0.0, 0.14):
            a = score(kw, rate)
            tag = f"{label}, r={rate}"
            print(f"{tag:<40} | {a[0]:7.3f} {a[1]:6.2f} | {a[2]:+7.3f} | {a[3]:6.2f}")

    print()
    print("  the drift's truth is 0.14 in the second row of each pair, and 0 in the first.")
    print("  `fault` is the dynamics channel's own read-out: the offset channel must not")
    print("  provoke it, because a drift is not a change of dynamics.")

    print()
    print("=" * 92)
    print("OPEN 5 -- arm scale: the channel's cost should VANISH as the rig grows")
    print("=" * 92)
    print(f"{'rig':<34} | {'n':>3} {'m':>3} | {'off':>9} | {'on':>9} | overhead")
    rng = np.random.default_rng(3)
    for label, F, H in (("scalar (ladder on, 360 members)", np.eye(1), np.ones((1, 1))),
                        ("kinematic 2-DOF", *kinematic(2)),
                        ("kinematic 5-DOF (arm scale)", *kinematic(5))):
        n, m = F.shape[0], H.shape[0]
        Y = rng.normal(size=(60, m))
        ms = {}
        for tag, use in (("off", False), ("on", True)):
            LucidFilter(dynamics=F, H=H, offsets=use).filter(Y[:10])
            f2 = LucidFilter(dynamics=F, H=H, offsets=use)
            t = time.time()
            f2.filter(Y)
            ms[tag] = (time.time() - t) / len(Y) * 1000.0
        print(f"{label:<34} | {n:3d} {m:3d} | {ms['off']:6.2f} ms | {ms['on']:6.2f} ms | "
              f"{ms['on']/ms['off'] - 1:+.1%}")
    print()
    print("  against the state augmentation `0002` priced on the same rigs: 1.84x scalar,")
    print("  2.06-2.87x kinematic 5-DOF.")


if __name__ == "__main__":
    main()
