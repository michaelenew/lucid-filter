"""0001 -- what the two empty first-moment cells cost the shipped filter.

The public `LucidFilter` carries every noise channel in VARIANCE currency: a log-scale
per process eigenmode and per sensor.  Both first-moment cells are empty --

    persistent process mean   = a drift / climbing bias   (shipped only in the fit-based
                                odefilter specimen, as `fit(unit_roots=2)`)
    persistent sensor mean    = a miscalibrated sensor    (nowhere in the repository)

-- and nothing has ever measured what that costs.  This probe prices both, on rigs the
filter is told nothing about, against the achievable floor in each case.

Three rigs:

  1  DRIFT (m = 1).  theta_t = theta_{t-1} + r + w_t, y_t = theta_t + v_t.  The floor is a
     Kalman filter handed the true (Q, R) AND the true r.  Prediction under test: the
     filter has no place to put r, so the process scale xi inflates to cover the lag, the
     state tracks with a residual lag, and the diagnostic reads "the process is hot" when
     the truth is "the process is drifting".

  2  SENSOR BIAS, m = 1.  y_t = theta_t + b*1{t >= t0} + v_t.  The bias is GAUGE here: the
     laws of (theta, b) and (theta + b, 0) are identical, so no filter can separate them.
     What is measurable is whether the filter's behaviour is the correct one -- absorb it
     as a level jump, stay calibrated, and leave nothing behind in the innovation sequence.

  3  SENSOR BIAS, m = 2 (the identifiable cell).  One level, two direct sensors, sensor 2
     develops a bias.  Only the m-1 RELATIVE biases are identifiable; the common mode stays
     gauge.  The floor is the same filter told the bias exactly.  Prediction under test:
     the bias is booked in variance currency -- eta_2 rises by about log(1 + b^2/R) -- so
     the sensor is DOWN-WEIGHTED rather than repaired, and the state keeps a residual bias
     equal to the biased sensor's remaining weight times b.

Run: python3 0001_what_the_empty_cells_cost.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from lucid import LucidFilter                                   # noqa: E402

N = 900
T0 = 400                    # where the step / the drift starts
SEEDS = (11, 12, 13, 14)
Q_TRUE = 0.02
R_TRUE = 1.0


# ------------------------------------------------------------------ reference filters
def kalman(y, Q, R, drift=0.0, H=None):
    """Textbook Kalman filter, handed everything.  Scalar state; y is (T, m)."""
    y = np.atleast_2d(y)
    m = y.shape[1]
    H = np.ones((m, 1)) if H is None else np.asarray(H, float)
    R = np.eye(m) * R if np.ndim(R) == 0 else np.diag(np.asarray(R, float))
    x, P = float(y[0, 0]), float(R[0, 0])
    mean = np.empty(len(y))
    var = np.empty(len(y))
    for t, row in enumerate(y):
        x, P = x + drift, P + Q
        h = H.ravel()
        S = P * np.outer(h, h) + R
        K = (P * h) @ np.linalg.inv(S)                       # (m,)
        e = row - h * x
        x = x + float(K @ e)
        P = P * (1.0 - float(K @ h))
        mean[t], var[t] = x, P
    return mean, var


def scores(mean, var, truth, lo):
    """RMSE and calibration E[e^2/S] over t >= lo."""
    e = mean[lo:] - truth[lo:]
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(e ** 2 / var[lo:])), float(np.mean(e))


# ------------------------------------------------------------------ rig 1: the drift cell
def rig_drift(r_rate):
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        w = rng.normal(0.0, np.sqrt(Q_TRUE), N)
        drift = r_rate * (np.arange(N) >= T0)
        theta = np.cumsum(w + drift)
        y = (theta + rng.normal(0.0, np.sqrt(R_TRUE), N))[:, None]

        told_nothing = LucidFilter().filter(y)
        km, kv = kalman(y, Q_TRUE, R_TRUE)                       # oracle noise, blind to r
        om, ov = kalman(y, Q_TRUE, R_TRUE, drift=r_rate)         # the floor: told r as well

        rows.append((
            scores(told_nothing.mean[:, 0], told_nothing.var[:, 0, 0], theta, T0),
            scores(km, kv, theta, T0),
            scores(om, ov, theta, T0),
            float(np.mean(told_nothing.process_scale[T0 + 100:, 0])),
            float(np.mean(told_nothing.process_scale[:T0, 0])),
        ))
    return rows


# ------------------------------------------------------------------ rig 2: gauge, m = 1
def rig_gauge(b):
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        theta = np.cumsum(rng.normal(0.0, np.sqrt(Q_TRUE), N))
        y = (theta + b * (np.arange(N) >= T0) + rng.normal(0.0, np.sqrt(R_TRUE), N))[:, None]
        res = LucidFilter().filter(y)
        # against the truth (uncorrectable), and against what the data can support: theta + b
        rows.append((
            scores(res.mean[:, 0], res.var[:, 0, 0], theta, T0 + 50),
            scores(res.mean[:, 0], res.var[:, 0, 0], theta + b * (np.arange(N) >= T0), T0 + 50),
            float(np.mean(res.measurement_scale[T0 + 100:, 0])),
            absorbed(res.mean[:, 0], theta, b),
        ))
    return rows


def absorbed(est, theta, b, frac=0.1):
    """Steps after t0 until the estimate has closed 90% of the step (nan if there is none)."""
    if b <= 0.0:
        return float("nan")
    hit = np.flatnonzero(np.abs(est[T0:] - (theta[T0:] + b)) < frac * b)
    return float(hit[0]) if hit.size else float(N - T0)


# ------------------------------------------------------------------ rig 3: the sensor-bias cell
def rig_two_sensors(b):
    H = np.array([[1.0], [1.0]])
    R0 = np.array([R_TRUE, R_TRUE])
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        theta = np.cumsum(rng.normal(0.0, np.sqrt(Q_TRUE), N))
        step = b * (np.arange(N) >= T0)
        y = np.stack([theta + rng.normal(0.0, np.sqrt(R_TRUE), N),
                      theta + step + rng.normal(0.0, np.sqrt(R_TRUE), N)], axis=1)

        told_nothing = LucidFilter(H=H, measurement=R0).filter(y)
        # the floor: the same filter handed the bias exactly (subtract it, keep both sensors)
        floor = LucidFilter(H=H, measurement=R0).filter(y - np.stack([np.zeros(N), step], 1))
        # the fallback a practitioner has today: throw the biased sensor away
        one = LucidFilter().filter(y[:, :1])

        rows.append((
            scores(told_nothing.mean[:, 0], told_nothing.var[:, 0, 0], theta, T0 + 100),
            scores(floor.mean[:, 0], floor.var[:, 0, 0], theta, T0 + 100),
            scores(one.mean[:, 0], one.var[:, 0, 0], theta, T0 + 100),
            told_nothing.measurement_scale[T0 + 100:].mean(axis=0),
        ))
    return rows


def mean_of(rows, i, j=None):
    vals = [r[i] if j is None else r[i][j] for r in rows]
    return float(np.mean(vals))


def main():
    t0 = time.time()

    print("=" * 78)
    print("RIG 1 -- the drift cell (m = 1).  Persistent process mean, no place to put it.")
    print("=" * 78)
    print(f"{'r/sd(w)':>8} | {'lucid':>22} | {'kalman blind':>14} | {'floor (told r)':>14} | xi")
    print(f"{'':>8} | {'rmse   calib   bias':>22} | {'rmse   calib':>14} | {'rmse   calib':>14} |")
    for r_rate in (0.0, 0.05, 0.14, 0.42):
        rows = rig_drift(r_rate)
        L, K, O = [tuple(mean_of(rows, i, j) for j in range(3)) for i in range(3)]
        xi_hot, xi_cold = mean_of(rows, 3), mean_of(rows, 4)
        print(f"{r_rate/np.sqrt(Q_TRUE):8.2f} | {L[0]:6.3f} {L[1]:7.2f} {L[2]:7.3f} | "
              f"{K[0]:6.3f} {K[1]:7.2f} | {O[0]:6.3f} {O[1]:7.2f} | "
              f"{xi_cold:+.2f} -> {xi_hot:+.2f}")

    print()
    print("=" * 78)
    print("RIG 2 -- a sensor bias at m = 1 is GAUGE.  Nothing to fix; check the behaviour.")
    print("=" * 78)
    print(f"{'b/sd(v)':>8} | {'rmse vs theta':>13} | {'rmse vs theta+b':>15} | {'calib':>6} | "
          f"{'eta':>6} | absorbed")
    for b in (0.0, 1.0, 3.0):
        rows = rig_gauge(b)
        vs_truth, vs_shifted = mean_of(rows, 0, 0), mean_of(rows, 1, 0)
        calib, eta, absorbed = mean_of(rows, 1, 1), mean_of(rows, 2), mean_of(rows, 3)
        print(f"{b:8.1f} | {vs_truth:13.3f} | {vs_shifted:15.3f} | {calib:6.2f} | "
              f"{eta:+6.2f} | {absorbed:.1f} steps")

    print()
    print("=" * 78)
    print("RIG 3 -- the sensor-bias cell (m = 2).  One level, two sensors, sensor 2 drifts off.")
    print("=" * 78)
    print(f"{'b/sd(v)':>8} | {'told nothing':>22} | {'floor (told b)':>14} | "
          f"{'sensor 1 alone':>14} | eta_1 / eta_2")
    print(f"{'':>8} | {'rmse   calib   bias':>22} | {'rmse   calib':>14} | {'rmse   calib':>14} |")
    for b in (0.0, 0.5, 1.0, 2.0, 4.0):
        rows = rig_two_sensors(b)
        L, F, S1 = [tuple(mean_of(rows, i, j) for j in range(3)) for i in range(3)]
        eta = np.mean([r[3] for r in rows], axis=0)
        print(f"{b:8.1f} | {L[0]:6.3f} {L[1]:7.2f} {L[2]:7.3f} | {F[0]:6.3f} {F[1]:7.2f} | "
              f"{S1[0]:6.3f} {S1[1]:7.2f} | {eta[0]:+.2f} / {eta[1]:+.2f}")

    print(f"\n[{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
