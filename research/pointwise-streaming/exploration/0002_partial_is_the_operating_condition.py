"""0002 -- partial rows against dropping the row, across the whole duty-cycle range.

The thing being retired.  Before this workstream the filter handled exactly one kind of
absence: an all-``NaN`` row (propagate, do not correct).  A row where SOME sensors read
had no path through the engine, so the only way to feed a multi-rate sensor set was to
throw away every reading that did not arrive on the common schedule -- which is most of
them, and the good ones as often as the bad.

That is not a rare-case bandaid: sensors sharing a schedule is the special case.  A GPS
at 5 Hz beside an IMU at 200 Hz, a camera that drops frames, a bus that arbitrates --
the general condition is that at any instant SOME subset reports.

Measured here on a two-sensor kinematic rig, sweeping the slow sensor's duty cycle:

    A. every sensor on every row          -- the unreachable upper bound
    B. partial rows, sub-selected H, R    -- what this change makes possible
    C. drop any row that is not complete  -- what the filter used to force

PREDICTION (before the run): B tracks A closely and degrades gracefully as the duty
cycle falls, because the fast sensor's rows are all still used; C degrades as 1/duty
because it is throwing away the fast sensor to punish the slow one.  The gap between
B and C is the whole content of the change and should be largest at LOW duty.

Second panel: the DIAGNOSIS must survive.  A sensor that fails is identified by its
measurement-scale chip; if it only reports one row in five, the chip must still find it,
just with fewer readings to do it from.

    python research/pointwise-streaming/exploration/0002_partial_is_the_operating_condition.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
from lucid import LucidFilter                                             # noqa: E402

OUT = os.path.join(HERE, "figures", "pw0002.json")
DT = 0.1
F = np.array([[1.0, DT], [0.0, 1.0]])
H = np.array([[1.0, 0.0], [0.0, 1.0]])          # a position sensor and a rate sensor
# The asymmetric pairing this is about, and the repo's own: a coarse ABSOLUTE sensor
# beside a precise DYNAMIC one (a 5 Hz GPS next to a 200 Hz IMU; the arm rig's bad
# potentiometer next to its good accelerometer).  The rate sensor is finer than one step
# of the velocity's own motion, so every one of its readings is worth having -- which is
# what makes throwing (k-1)/k of them away in order to punish the slow sensor expensive.
QS = np.array([0.01, 0.05])
RS = np.array([0.30, 0.02])


def rig(T, seed, hot=None):
    """Position/velocity truth read by a slow absolute sensor and a fast rate sensor.
    ``hot`` = (sensor, t0, t1, factor) degrades one of them over a window."""
    r = np.random.default_rng(seed)
    x = np.zeros(2)
    X = np.empty((T, 2)); Y = np.empty((T, 2)); sd = np.tile(RS, (T, 1))
    if hot is not None:
        i, t0, t1, fac = hot
        sd[t0:t1, i] *= fac
    for t in range(T):
        x = F @ x + r.standard_normal(2) * QS
        X[t] = x
        Y[t] = H @ x + r.standard_normal(2) * sd[t]
    return X, Y


def rmse(a, b, lo=40):
    return float(np.sqrt(np.mean((a[lo:] - b[lo:]) ** 2)))


def sweep(duty_list, seeds=8, T=400):
    """Sensor 0 (the coarse absolute one) reports on 1 row in ``k``; sensor 1 (the precise
    rate one) on every row.  Dropping any incomplete row throws the rate sensor away too."""
    rows = []
    for k in duty_list:
        acc = {t: {0: [], 1: []} for t in ("all", "part", "drop")}
        for sd in range(seeds):
            X, Y = rig(T, 100 + sd)
            keep = (np.arange(T) % k) == 0
            Yp = Y.copy(); Yp[~keep, 0] = np.nan       # slow sensor absent on most rows
            Yd = Y.copy(); Yd[~keep, :] = np.nan       # ... so the old filter loses the row
            for tag, YY in (("all", Y), ("part", Yp), ("drop", Yd)):
                mean = LucidFilter(dynamics=F, H=H).filter(YY).mean
                for j in (0, 1):
                    acc[tag][j].append(rmse(mean[:, j], X[:, j]))
        used_part = int(np.isfinite(Yp).sum())
        used_drop = int(np.isfinite(Yd).sum())
        rows.append(dict(
            duty=1.0 / k, every=k,
            readings_partial=used_part, readings_drop=used_drop,
            all_pos=float(np.mean(acc["all"][0])), all_vel=float(np.mean(acc["all"][1])),
            partial=float(np.mean(acc["part"][0])),
            partial_se=float(np.std(acc["part"][0]) / math.sqrt(seeds)),
            partial_vel=float(np.mean(acc["part"][1])),
            drop_row=float(np.mean(acc["drop"][0])),
            drop_se=float(np.std(acc["drop"][0]) / math.sqrt(seeds)),
            drop_vel=float(np.mean(acc["drop"][1]))))
    return rows


def diagnosis(seeds=8, T=400):
    """A sensor fails x8 over [180, 300).  Does its chip still light when it only
    reports one row in ``k``?  Reported as the mean learned log-scale inside the burst
    minus the mean outside it -- the truth is log(8**2) = 4.16."""
    rows = []
    for k in (1, 2, 5, 10):
        lift, lift_other = [], []
        for sd in range(seeds):
            X, Y = rig(T, 300 + sd, hot=(0, 180, 300, 8.0))
            keep = (np.arange(T) % k) == 0
            Yp = Y.copy(); Yp[~keep, 0] = np.nan
            ms = LucidFilter(dynamics=F, H=H).filter(Yp).measurement_scale
            inside = np.mean(ms[200:300, 0]); outside = np.mean(ms[60:180, 0])
            lift.append(float(inside - outside))
            lift_other.append(float(np.mean(ms[200:300, 1]) - np.mean(ms[60:180, 1])))
        rows.append(dict(every=k, lift=float(np.mean(lift)),
                         se=float(np.std(lift) / math.sqrt(seeds)),
                         leak_other=float(np.mean(lift_other))))
    return rows


if __name__ == "__main__":
    duty = [1, 2, 5, 10, 25]
    print("=" * 78)
    print("A. DUTY-CYCLE SWEEP -- the absolute sensor reports 1 row in k")
    print("   position RMSE (m); the rate sensor reports on every row throughout")
    print("=" * 78)
    rows = sweep(duty)
    print(f"{'k':>4} {'readings used':>15}   {'all sensors':>11} {'partial (new)':>16}"
          f" {'drop row (old)':>17}   {'old/new':>8}")
    for row in rows:
        print(f"{row['every']:>4} {row['readings_partial']:>7d}/{row['readings_drop']:<7d}"
              f"   {row['all_pos']:>11.4f}"
              f"   {row['partial']:.4f}+-{row['partial_se']:.4f}"
              f"   {row['drop_row']:.4f}+-{row['drop_se']:.4f}"
              f"   {row['drop_row'] / row['partial']:>7.2f}x")
    print()
    print("   the same rows, VELOCITY RMSE -- the state the discarded sensor reads")
    print(f"{'k':>4}   {'all sensors':>11} {'partial (new)':>16} {'drop row (old)':>17}"
          f"   {'old/new':>8}")
    for row in rows:
        print(f"{row['every']:>4}   {row['all_vel']:>11.4f}   {row['partial_vel']:>16.4f}"
              f" {row['drop_vel']:>17.4f}   {row['drop_vel'] / row['partial_vel']:>7.2f}x")

    print()
    print("=" * 78)
    print("B. THE DIAGNOSIS SURVIVES -- sensor 0 fails x8 while reporting 1 row in k")
    print("   learned log-scale lift inside the burst vs outside (truth 4.16)")
    print("=" * 78)
    dg = diagnosis()
    for row in dg:
        print(f"   1 row in {row['every']:<3d}  lift = {row['lift']:5.2f} +- {row['se']:.2f}"
              f"     leak onto the healthy sensor = {row['leak_other']:+.2f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(sweep=rows, diagnosis=dg), open(OUT, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
