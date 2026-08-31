"""0011 -- the no-impingement guard, on the rig the repository actually demos.

`0008` measured the channel's cost at arm SCALE on a kinematic model of the right size.  This
runs the real thing: `multivariate-statfilter/0052`'s 5-DOF arm, imported rather than
reimplemented -- 15 states, 10 sensors, a commanded trajectory through known forcing, a
callable-free but genuinely coupled `H`, and phased noise bursts.  It is the guard every
change to this engine is held to (`sequence-demix`'s second acceptance benchmark), and the
question it answers is the narrow one that matters for a channel that is off by default:

    turning it ON must not damage a rig it has nothing to offer.

The arm has no drift and no miscalibrated sensor.  So the channel should find nothing, report
nothing, and cost little -- and if it instead invents an offset, this is where that shows up,
because the arm is driven hard by a known input and a filter that mistakes forcing for drift
would be visible immediately.

Reported per regime: angle and velocity RMSE with the channel off and on, the offset the
channel claims, and the per-step cost.  A regression here is a blocker; the SUMMARY records
this as the guard the workstream had not run.

Run: python3 0011_the_demo_arm_guard.py [n_seeds]
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

_p = os.path.join(REPO, "research", "multivariate-statfilter", "exploration",
                  "0052_lucid_arm5dof_profile.py")
_spec = importlib.util.spec_from_file_location("arm5dof", _p)
arm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arm)

from lucid import LucidFilter                                    # noqa: E402


def build(offsets):
    return LucidFilter(dynamics=arm.F, control=arm.B, H=arm.H,
                       process=arm.Q0, measurement=arm.R0, offsets=offsets)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    probe = build(True)
    print("=" * 92)
    print(f"The 5-DOF arm ({arm.N} states, {arm.M} sensors, {arm.T} steps), "
          f"{n_seeds} seeds per regime")
    print("=" * 92)
    print(f"  identifiable process offsets on this structure: k = "
          f"{0 if probe._mean is None else probe._mean.k}"
          f";  sensor read-out: k = {0 if probe._sensor is None else probe._sensor.k}")
    print()
    print(f"{'regime':<10} | {'angle off':>10} {'angle on':>10} {'ratio':>7} | "
          f"{'vel off':>9} {'vel on':>9} {'ratio':>7} | {'max |offset|':>12}")
    tot = {}
    for key, label in arm.REGIMES:
        acc = np.zeros(5)
        for seed in range(n_seeds):
            U, S, Y, jstd, pot, acc_sd = arm.sim(seed, key)
            for j, use in enumerate((False, True)):
                f = build(use)
                t0 = time.time()
                res = f.filter(Y, U=U)
                acc[2 * j] += arm.rms(res.mean, S, 0) / n_seeds
                acc[2 * j + 1] += arm.rms(res.mean, S, 1) / n_seeds
                if use:
                    off = 0.0 if res.offset is None else float(np.abs(res.offset).max())
                    acc[4] += off / n_seeds
                    tot.setdefault("on_ms", []).append((time.time() - t0) / arm.T * 1000)
                else:
                    tot.setdefault("off_ms", []).append((time.time() - t0) / arm.T * 1000)
        print(f"{label:<10} | {acc[0]:10.5f} {acc[2]:10.5f} {acc[2]/acc[0]:7.3f} | "
              f"{acc[1]:9.4f} {acc[3]:9.4f} {acc[3]/acc[1]:7.3f} | {acc[4]:12.4f}")
    print()
    print(f"  cost: {np.mean(tot['off_ms']):.1f} ms/step off, {np.mean(tot['on_ms']):.1f} on "
          f"({np.mean(tot['on_ms'])/np.mean(tot['off_ms']) - 1:+.1%})")
    print("  a ratio above 1.0 is the channel costing the arm something it did not need.")


if __name__ == "__main__":
    main()
