"""0009 -- an offset that MOVES, which is what the walk is for and what nothing had tested.

Every rig in 0001-0008 sets the offset once and leaves it.  The channel carries a walk of
`rho * cls` per rung precisely so a changing offset is tracked -- "bounded, never frozen" --
and that clause had never been exercised.  The sibling workstream found its own missing
persistence axis exactly this way (`ode-filter` 0047/0051: the tau kernel could represent
impulse and diffusion but not directed drift, so the tracker staircased and undercovered while
remaining prequentially near-optimal, because predictive likelihood under-polices a coordinate
it can barely see).  This probe asks the same question of the offset channel.

Three schedules, each against the channel off and against a Kalman filter told `d_t` exactly:

  STEP      0 -> r -> 0.  The offset arrives and later leaves; the second edge is the harder
            one, because the channel has by then accumulated evidence for a nonzero value.
  RAMP      d_t climbing linearly.  This is the schedule the sibling's kernel could not
            represent: an offset with a velocity of its own.
  REVERSAL  +r -> -r.  The largest excursion, and the one where a slow tracker is worst.

Reported per segment: state RMSE, calibration, and how closely `d_hat` follows `d_t`.

Run: python3 0009_a_moving_offset.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from lucid import LucidFilter                                    # noqa: E402

Q_TRUE, R_TRUE, N, SEEDS = 0.02, 1.0, 1200, (11, 12, 13, 14)


def schedule(kind, r=0.25):
    d = np.zeros(N)
    if kind == "step":
        d[300:700] = r
    elif kind == "ramp":
        d[300:] = r * np.linspace(0.0, 1.0, N - 300)
    elif kind == "reversal":
        d[300:700] = r
        d[700:] = -r
    return d


def rig(d, seed):
    rng = np.random.default_rng(seed)
    theta = np.cumsum(rng.normal(0, np.sqrt(Q_TRUE), N) + d)
    return theta, (theta + rng.normal(0, np.sqrt(R_TRUE), N))[:, None]


def oracle(Y, d):
    x, P = float(Y[0, 0]), R_TRUE
    mean, var = np.empty(N), np.empty(N)
    for t, row in enumerate(Y):
        x, P = x + d[t], P + Q_TRUE
        S = P + R_TRUE
        K = P / S
        x, P = x + K * (float(row[0]) - x), (1.0 - K) * P
        mean[t], var[t] = x, P
    return mean, var


def segments(kind):
    if kind == "step":
        return [("before", 100, 300), ("on", 400, 700), ("after it leaves", 800, 1200)]
    if kind == "ramp":
        return [("before", 100, 300), ("early ramp", 400, 700), ("late ramp", 800, 1200)]
    return [("before", 100, 300), ("+r", 400, 700), ("after reversal", 800, 1200)]


def main():
    for kind in ("step", "ramp", "reversal"):
        d = schedule(kind)
        print("=" * 92)
        print(f"{kind.upper()}   (the offset's truth moves; the walk is what has to follow it)")
        print("=" * 92)
        acc = {}
        traces = np.zeros(N)
        for seed in SEEDS:
            theta, Y = rig(d, seed)
            off = LucidFilter().filter(Y)
            on = LucidFilter(offsets=True).filter(Y)
            om, ov = oracle(Y, d)
            traces += on.offset[:, 0] / len(SEEDS)
            for name, lo, hi in segments(kind):
                for tag, (mm, vv) in (("off", (off.mean[:, 0], off.var[:, 0, 0])),
                                      ("on", (on.mean[:, 0], on.var[:, 0, 0])),
                                      ("oracle", (om, ov))):
                    e = mm[lo:hi] - theta[lo:hi]
                    key = (name, tag)
                    cur = acc.get(key, np.zeros(2))
                    acc[key] = cur + np.array([np.sqrt(np.mean(e ** 2)),
                                               np.mean(e ** 2 / vv[lo:hi])]) / len(SEEDS)
        print(f"{'segment':<18} | {'off rmse/calib':>16} | {'on rmse/calib':>16} | "
              f"{'oracle':>8} | {'d_hat -> d':>18}")
        for name, lo, hi in segments(kind):
            o, n, r = acc[(name, "off")], acc[(name, "on")], acc[(name, "oracle")]
            print(f"{name:<18} | {o[0]:8.3f} {o[1]:7.2f} | {n[0]:8.3f} {n[1]:7.2f} | "
                  f"{r[0]:8.3f} | {traces[lo:hi].mean():+8.3f} -> {d[lo:hi].mean():+8.3f}")
        print()


if __name__ == "__main__":
    main()
