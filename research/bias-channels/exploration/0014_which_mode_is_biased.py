"""0014 -- a process bias on a multi-dimensional state: is it put on the right mode?

`0011` established that the channel finds NOTHING on the demo arm, which is the guard.  `0007`
established that it finds a velocity-side drift on a two-state rig, where there is only one
place a drift could be.  Neither asks the question a caller with a real machine actually has:
with fifteen states and five physical disturbance channels, does a process bias on ONE of them
get attributed to that one?

The structure says it should be askable.  On this arm the identifiable process-offset basis has
k = 5, and it spans exactly the columns of `GJ` -- the map from the five per-joint jerk
disturbances into the state -- to principal angles of 1.0000 on every direction.  So the
channel's coordinates ARE the rig's own noise-injection geometry, rediscovered from `(F, H)`
alone without being told what `GJ` is.  That is the structural half of the answer; this probe
measures the other half.

`r.offset` is reported in the caller's state coordinates, so the read is a least-squares
projection back onto the physical channels: `GJ g = offset` recovers a per-joint drift `g`.

ATTRIBUTION IS CLEAN AND THE STATE GETS WORSE, which is the finding.  The right joint carries
the offset to two decimals with negligible leak onto the others -- and the angle estimate
degrades 1.29x, the velocity 1.97x, with calibration going 1.45 -> 2.43.  The second table
locates it: the loss is absent before onset and absent in the 200 steps after it, and appears
only once the estimate has CONVERGED.  So it is not the transient, and the third table rules
out the other cheap explanation -- the fed-back estimate has a step-to-step jitter of 0.0000,
so it is not injecting noise.

What is left is that feeding back an offset that is accurate but not exact is not free when the
offset is far from the sensor in RELATIVE DEGREE.  Here a jerk bias reaches the potentiometer
through three integrations, so a residual of a few parts in a thousand becomes a systematic
angle error, while the covariance says the direction is known -- hence the overconfidence.  The
scalar rig, where the drift is ON the observed level, has the opposite sign of result (0.711 ->
0.457).  This is the same shape as `multivariate-statfilter`'s standing relative-degree open on
the SECOND-moment channel ("nothing in the construction currently prices relative degree in"),
now visible on the first.  Stated as a hypothesis consistent with three rigs at relative degree
0, 1 and 3, not as a demonstrated mechanism: one rig per degree is not a law.

Run: python3 0014_which_mode_is_biased.py [n_seeds]
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

_spec = importlib.util.spec_from_file_location(
    "arm5dof", os.path.join(REPO, "research", "multivariate-statfilter", "exploration",
                            "0052_lucid_arm5dof_profile.py"))
arm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arm)

from lucid import LucidFilter                                    # noqa: E402

T, T0 = arm.T, 300


def sim(seed, joint, rate):
    """The arm's own CALM rig, plus a constant jerk bias on one joint from T0."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) * arm.DT
    U = np.zeros((T, arm.NJ))
    for j in range(arm.NJ):
        for (a, w, ph) in [(2.0, 0.35 + 0.1 * j, j), (1.2, 0.7 + 0.13 * j, 2 * j)]:
            U[:, j] += a * np.sin(2 * np.pi * w * t + ph)
    bias = np.zeros(arm.NJ)
    bias[joint] = rate
    s = np.zeros(arm.N)
    S = np.zeros((T, arm.N))
    Y = np.zeros((T, arm.M))
    for k in range(T):
        d = bias if k >= T0 else np.zeros(arm.NJ)
        s = arm.F @ s + arm.B @ U[k] + arm.GJ @ (arm.JERK * rng.standard_normal(arm.NJ) + d)
        S[k] = s
        sd = np.where(arm.KIND == "pot", arm.POT, arm.ACC)
        Y[k] = arm.H @ s + sd * rng.standard_normal(arm.M)
    return U, S, Y


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print("=" * 90)
    print(f"A constant jerk bias on ONE joint of the 5-DOF arm (jerk sd = {arm.JERK}), "
          f"{n_seeds} seeds")
    print("=" * 90)
    print(f"{'injected':<22} | {'recovered per-joint drift g':<44} | {'angle rmse':>18}")
    print(f"{'':<22} | {'j0     j1     j2     j3     j4':<44} | {'off      on':>18}")
    for joint, rate in ((2, 0.6), (2, 1.2), (0, 1.2), (4, 1.2)):
        g_acc = np.zeros(arm.NJ)
        r_off = r_on = 0.0
        for seed in range(n_seeds):
            U, S, Y = sim(seed, joint, rate)
            on = LucidFilter(dynamics=arm.F, control=arm.B, H=arm.H, process=arm.Q0,
                             measurement=arm.R0, offsets=True).filter(Y, U=U)
            off = LucidFilter(dynamics=arm.F, control=arm.B, H=arm.H, process=arm.Q0,
                              measurement=arm.R0).filter(Y, U=U)
            # r.offset is in state coordinates; project back onto the physical channels
            g_acc += np.linalg.lstsq(arm.GJ, on.offset[-1], rcond=None)[0] / n_seeds
            sl = slice(T0 + 200, T)
            tt = S.reshape(T, arm.NJ, arm.ORDER)[sl, :, 0]
            for tag, res in (("off", off), ("on", on)):
                ee = res.mean.reshape(T, arm.NJ, arm.ORDER)[sl, :, 0]
                v = float(np.sqrt(((ee - tt) ** 2).mean())) / n_seeds
                if tag == "off":
                    r_off += v
                else:
                    r_on += v
        cells = "  ".join(f"{v:+5.2f}" for v in g_acc)
        star = "  <- injected on j%d" % joint
        print(f"joint {joint}, rate {rate:.1f}     | {cells:<44} | {r_off:8.5f} {r_on:9.5f}")
        print(f"{'':<22} | {'truth: ' + '  '.join('%+5.2f' % (rate if i == joint else 0.0) for i in range(arm.NJ)):<44} |{star}")
    print()
    print("  `g` is the least-squares read of `r.offset` back onto GJ's columns -- the five")
    print("  physical jerk channels the identifiable basis was found to span.")

    print()
    print("=" * 90)
    print("Where the loss is: by window, and whether the fed-back estimate is noisy")
    print("=" * 90)
    wins = (("before onset", slice(50, T0)), ("just after", slice(T0, 500)),
            ("settled", slice(700, T)))
    idx = [j * arm.ORDER for j in range(arm.NJ)]
    for rate in (0.0, 1.2):
        acc = {}
        for seed in range(n_seeds):
            U, S, Y = sim(seed, 2, rate)
            for tag, kw in (("off", {}), ("on", {"offsets": True})):
                r = LucidFilter(dynamics=arm.F, control=arm.B, H=arm.H, process=arm.Q0,
                                measurement=arm.R0, **kw).filter(Y, U=U)
                for wn, sl in wins:
                    e = r.mean[sl] - S[sl]
                    v = np.array([np.diag(P)[idx] for P in r.var[sl]])
                    acc[(wn, tag, "a")] = acc.get((wn, tag, "a"), 0.0) + float(
                        np.sqrt((e[:, idx] ** 2).mean())) / n_seeds
                    acc[(wn, tag, "c")] = acc.get((wn, tag, "c"), 0.0) + float(
                        np.mean(e[:, idx] ** 2 / v)) / n_seeds
                if tag == "on":
                    acc["jit"] = acc.get("jit", 0.0) + (
                        0.0 if r.offset is None
                        else float(np.abs(np.diff(r.offset[700:], axis=0)).mean())) / n_seeds
        print(f"  jerk bias on joint 2 = {rate}")
        for wn, _ in wins:
            ao, an = acc[(wn, "off", "a")], acc[(wn, "on", "a")]
            co, cn = acc[(wn, "off", "c")], acc[(wn, "on", "c")]
            print(f"    {wn:<14} angle {ao:.5f} -> {an:.5f} ({an / ao:.3f}x)   "
                  f"calibration {co:5.2f} -> {cn:5.2f}")
        print(f"    fed-back offset, mean |change| per step in the settled window: "
              f"{acc['jit']:.4f}")
    print()
    print("  the loss is absent before onset and absent in the 200 steps after it, and appears")
    print("  once the estimate has converged -- so it is not the transient; and the offset does")
    print("  not move once converged -- so it is not injected noise.  See the docstring.")


if __name__ == "__main__":
    main()
