"""0007 -- the channel off the random walk: a general F, and the confound the frame predicts.

Everything in 0001-0006 was measured on `F = I`.  Two of the SUMMARY's opens live here, and
both are risks to the shipped channel rather than features it lacks:

  OPEN 3, a general F.  `_mean_basis` and the sensitivity recursion are written for any F and
  nothing has exercised one.  Two rigs: a DOUBLE INTEGRATOR whose drift is on the velocity
  (the offset is not where the sensor looks), and a STABLE AR(1).

  The stable case is the one the frame makes a prediction about, and it is uncomfortable.  On
  the stable spectrum a process mean and a sensor bias are confounded WITH EACH OTHER: `d`
  drives the state to the constant `(I - F)^-1 d`, whose reading is exactly a sensor bias of
  `d / (1 - phi)` at p = 1.  So the two hypotheses fit the data identically and imply
  DIFFERENT states -- under the drift reading the state really is displaced, under the bias
  reading it is not.  The shipped channel carries only the process entry, so it necessarily
  adopts the drift reading -- and measured BEFORE the fix this probe produced, that cost a real
  sensor bias 0.786 -> 0.960 RMSE with calibration 2.07 -> 5.25.  The fix is in the same
  algebra: `_mean_basis` now quotients the process entry by the SENSOR entry as well as by the
  free responses, so a drift is carried only where its signature GROWS and no constant offset
  can imitate it.  On a purely stable spectrum nothing survives, the channel correctly declines
  to act (`k = 0`, printed as `inert` below), and on a mixed spectrum it keeps exactly the
  unit-root part.

  OPEN 2, a sensor step under a genuine drift.  `0006`'s ESTIMATE variant showed a persistent
  innovation offset loading onto the process entry at +0.08.  The shipped channel is immune
  there only because it carries no sensor entry -- it has never been shown immune when a real
  drift and a real sensor step are present at once, which is the rig below.

Run: python3 0007_general_dynamics_and_the_confound.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from lucid import LucidFilter                                    # noqa: E402
from lucid.statfilter.lucid import _mean_basis                   # noqa: E402

SEEDS = (11, 12, 13, 14)


def simulate(F, H, Q, R, d, c, T, seed, t0=0):
    n, m = F.shape[0], H.shape[0]
    rng = np.random.default_rng(seed)
    Qc = np.linalg.cholesky(Q + np.eye(n) * 1e-18)
    sd = np.sqrt(np.diag(R))
    x = np.zeros(n)
    xs, ys = [], []
    for t in range(T):
        on = 1.0 if t >= t0 else 0.0
        x = F @ x + d * on + Qc @ rng.normal(size=n)
        xs.append(x.copy())
        ys.append(H @ x + c * on + sd * rng.normal(size=m))
    return np.array(xs), np.array(ys)


def oracle(Y, F, H, Q, R, d, c, t0):
    """A Kalman filter told the offsets exactly -- the floor, where one exists."""
    n, m = F.shape[0], H.shape[0]
    x, P = np.zeros(n), np.eye(n) * 1e3
    mean, var = np.empty((len(Y), n)), np.empty((len(Y), n, n))
    for t, y in enumerate(Y):
        on = 1.0 if t >= t0 else 0.0
        x, P = F @ x + d * on, F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (y - H @ x - c * on)
        P = P - K @ H @ P
        mean[t], var[t] = x, P
    return mean, var


def band(mean, var, truth, lo, axis=0):
    e = mean[lo:, axis] - truth[lo:, axis]
    return np.sqrt(np.mean(e ** 2)), np.mean(e ** 2 / var[lo:, axis, axis]), np.mean(e)


def compare(label, F, H, Q, R, d, c, T, t0, axis=0, n_state=None):
    n_state = F.shape[0] if n_state is None else n_state
    acc = np.zeros(10)
    for seed in SEEDS:
        theta, Y = simulate(F, H, Q, R, d, c, T, seed, t0)
        R0 = np.diag(R).copy()
        off = LucidFilter(dynamics=F, H=H, process=Q, measurement=R0).filter(Y)
        on = LucidFilter(dynamics=F, H=H, process=Q, measurement=R0, offsets=True).filter(Y)
        om, ov = oracle(Y, F, H, Q, R, d, c, t0)
        lo = t0 + (T - t0) // 3
        for j, (mm, vv) in enumerate(((off.mean, off.var), (on.mean, on.var), (om, ov))):
            r = band(mm, vv, theta, lo, axis)
            acc[3 * j:3 * j + 3] += np.array(r) / len(SEEDS)
        if on.offset is None:
            acc[9] = float("nan")                    # the channel declined to act -- see below
        else:
            # read the estimate along the direction the truth lies in, not along a component
            nd = float(np.linalg.norm(d))
            acc[9] += (float(on.offset[-1] @ (d / nd)) if nd > 0
                       else float(np.linalg.norm(on.offset[-1]))) / len(SEEDS)
    dh = "  inert" if not np.isfinite(acc[9]) else f"{acc[9]:+7.3f}"
    print(f"  {label:<34} off {acc[0]:7.3f}/{acc[1]:6.2f}   on {acc[3]:7.3f}/{acc[4]:6.2f}"
          f"   oracle {acc[6]:7.3f}   d_hat {dh:>7}")
    return acc


def main():
    print("=" * 96)
    print("OPEN 3a -- a general F where the offset is NOT where the sensor looks")
    print("=" * 96)
    F = np.array([[1.0, 0.05], [0.0, 1.0]])          # position / velocity
    H = np.array([[1.0, 0.0]])                       # a position sensor only
    Q, R = np.diag([1e-6, 4e-4]), np.eye(1) * 1.0
    B = _mean_basis(F, H)
    print(f"  identifiable process offsets: k = {B.shape[1]}, direction "
          f"{np.array2string(B[:2, 0], precision=3)}  (velocity-side, as the frame says)")
    print("  rmse / calibration on POSITION:")
    compare("no offset", F, H, Q, R, np.zeros(2), np.zeros(1), 900, 300)
    compare("velocity offset 0.02/step", F, H, Q, R, np.array([0.0, 0.02]), np.zeros(1),
            900, 300)
    compare("velocity offset 0.06/step", F, H, Q, R, np.array([0.0, 0.06]), np.zeros(1),
            900, 300)

    print()
    print("=" * 96)
    print("OPEN 3b -- the STABLE spectrum, where d and c are confounded with each other")
    print("=" * 96)
    phi = 0.8
    F, H = np.array([[phi]]), np.ones((1, 1))
    Q, R = np.eye(1) * 0.05, np.eye(1) * 1.0
    print(f"  a sensor bias c is matched exactly by a drift d = c (1 - phi) = {0.3 * (1 - phi):.2f}")
    print("  -- the two fit identically and imply DIFFERENT states.  Both readings measured:")
    compare("no offset", F, H, Q, R, np.zeros(1), np.zeros(1), 900, 300)
    compare("a real drift d = 0.06", F, H, Q, R, np.array([0.06]), np.zeros(1), 900, 300)
    compare("a real sensor bias c = 0.30", F, H, Q, R, np.zeros(1), np.array([0.30]), 900, 300)
    compare("a real sensor bias c = 1.00", F, H, Q, R, np.zeros(1), np.array([1.00]), 900, 300)

    print()
    print("=" * 96)
    print("OPEN 2 -- a sensor step UNDER a genuine drift: does the step corrupt the drift?")
    print("=" * 96)
    F, H = np.eye(1), np.ones((3, 1))
    Q, R = np.eye(1) * 0.02, np.eye(3)
    for rate, bias in ((0.14, 0.0), (0.14, 2.0), (0.14, 6.0)):
        c = np.zeros(3)
        c[2] = bias
        compare(f"drift {rate}, sensor-3 step {bias}", F, H, Q, R, np.array([rate]), c, 900, 300)
    print()
    print("  the drift's truth is 0.14 in all three rows; the sensor step is what varies.")


if __name__ == "__main__":
    main()
