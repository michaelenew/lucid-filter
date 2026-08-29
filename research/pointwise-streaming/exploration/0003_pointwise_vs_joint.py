"""0003 -- decomposing a synchronous row into m points at one timestamp.

If ``(sensor, timestamp, value)`` is to be the filter's native input, the vector row has
to be a SPECIAL CASE of it and not a different thing: feeding a row as ``m`` points that
share a timestamp must mean what feeding the row means.

For a plain Kalman filter with diagonal ``R`` that is an exact identity -- ``m``
sequential scalar corrections equal the joint one.  Here it is NOT exact, and the reason
is worth naming: between the sub-updates the engine does two things a Kalman filter does
not.  It GPB1-collapses the caltrop star back to a single (mean, covariance), and it
takes a walk step on every scale axis the sub-event can see.  So the m points see m
successive collapses instead of one, and the scale walk takes m smaller steps instead of
one larger one, on the same information.

PREDICTION (before the run): the two agree to within a small fraction of the filter's
own error -- the collapse is a projection onto the same first two moments and the walk
is a contraction -- and the pointwise route costs LESS arithmetic per instant, because
the joint update's ``m**3`` solve is replaced by ``m`` scalar ones.  Failure would be a
visible drift between them or a pointwise cost that is worse, either of which would make
the streaming API a second filter rather than the same one.

Panels: (A) agreement across sensor counts, (B) measured cost per instant, (C) the
predicted-density ledger -- the two routes' total log-likelihood of the same data.

    python research/pointwise-streaming/exploration/0003_pointwise_vs_joint.py
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
from lucid import LucidFilter                                             # noqa: E402

OUT = os.path.join(HERE, "figures", "pw0003.json")
DT = 0.1


def rig(ndof, T, seed):
    """``ndof`` independent joints, each position + velocity, each read by an absolute
    and a rate sensor -- n = 2*ndof states, m = 2*ndof sensors."""
    r = np.random.default_rng(seed)
    F = np.kron(np.eye(ndof), np.array([[1.0, DT], [0.0, 1.0]]))
    H = np.eye(2 * ndof)
    q = np.tile([0.02, 0.05], ndof)
    rr = np.tile([0.30, 0.10], ndof)
    x = np.zeros(2 * ndof)
    X = np.empty((T, 2 * ndof)); Y = np.empty((T, 2 * ndof))
    for t in range(T):
        x = F @ x + r.standard_normal(2 * ndof) * q
        X[t] = x
        Y[t] = x + r.standard_normal(2 * ndof) * rr
    return F, H, X, Y


def agreement(ndof, T=150, seed=11):
    F, H, X, Y = rig(ndof, T, seed)
    m = H.shape[0]
    j = LucidFilter(dynamics=F, H=H).filter(Y)
    pts = [(i, float(t), Y[t, i]) for t in range(T) for i in range(m)]
    s = LucidFilter(dynamics=F, H=H).stream(pts)
    at_instant = s.mean[m - 1::m]                      # after the last point of each instant
    err_j = np.sqrt(np.mean((j.mean - X) ** 2))
    err_s = np.sqrt(np.mean((at_instant - X) ** 2))
    disagree = np.abs(j.mean - at_instant).max()
    return dict(ndof=ndof, n=2 * ndof, m=m,
                rmse_joint=float(err_j), rmse_pointwise=float(err_s),
                ratio=float(err_s / err_j),
                max_disagreement=float(disagree),
                disagreement_over_rmse=float(disagree / err_j),
                loglik_joint=float(j.loglik), loglik_pointwise=float(s.loglik))


def cost(ndof, T=30, seed=11, reps=1):
    # Both columns are per-instant RATES, so the horizon only sets their precision.
    # It is short because the exact-Q path costs real time per event.
    F, H, X, Y = rig(ndof, T, seed)
    m = H.shape[0]
    f = LucidFilter(dynamics=F, H=H)
    t0 = time.perf_counter()
    for _ in range(reps):
        f.filter(Y)
    tj = (time.perf_counter() - t0) / (reps * T)
    pts = [(i, float(t), Y[t, i]) for t in range(T) for i in range(m)]
    g = LucidFilter(dynamics=F, H=H)
    t0 = time.perf_counter()
    for _ in range(reps):
        g.stream(pts)
    ts = (time.perf_counter() - t0) / (reps * T)
    # the arithmetic the README's cost model predicts, per instant
    n = 2 * ndof
    G = 1 + 4 * (n + m)
    flop_joint = G * (2 * n * n * m + 2 * n * m * m + m ** 3)
    flop_point = m * G * (2 * n * n + 2 * n + 1)
    return dict(ndof=ndof, n=n, m=m,
                ms_joint=1e3 * tj, ms_pointwise=1e3 * ts,
                flops_joint=flop_joint, flops_pointwise=flop_point,
                flop_ratio=flop_point / flop_joint)


if __name__ == "__main__":
    print("=" * 78)
    print("A. AGREEMENT -- one row as m points at one timestamp vs the joint row")
    print("=" * 78)
    ag = [agreement(k) for k in (1, 2, 3, 5)]
    print(f"{'n':>3} {'m':>3}   {'RMSE joint':>11} {'RMSE pointwise':>15} {'ratio':>7}"
          f"   {'max |disagree|':>15} {'/RMSE':>7}")
    for row in ag:
        print(f"{row['n']:>3} {row['m']:>3}   {row['rmse_joint']:>11.5f}"
              f"   {row['rmse_pointwise']:>13.5f} {row['ratio']:>7.4f}"
              f"   {row['max_disagreement']:>15.2e} {row['disagreement_over_rmse']:>6.1%}")

    print()
    print("=" * 78)
    print("B. COST PER INSTANT -- the joint m**3 solve replaced by m scalar ones")
    print("=" * 78)
    co = [cost(k) for k in (1, 2, 3, 5)]
    print(f"{'n':>3} {'m':>3}   {'ms/instant joint':>17} {'ms/instant pointwise':>21}"
          f"   {'multiply-adds joint':>20} {'pointwise':>12} {'ratio':>7}")
    for row in co:
        print(f"{row['n']:>3} {row['m']:>3}   {row['ms_joint']:>17.2f} {row['ms_pointwise']:>21.2f}"
              f"   {row['flops_joint']:>20,d} {row['flops_pointwise']:>12,d}"
              f" {row['flop_ratio']:>7.2f}")

    print()
    print("=" * 78)
    print("C. THE LEDGER -- total predictive log-likelihood of the same data")
    print("=" * 78)
    for row in ag:
        d = row['loglik_pointwise'] - row['loglik_joint']
        print(f"   n={row['n']:<3d} m={row['m']:<3d}  joint {row['loglik_joint']:11.2f}"
              f"   pointwise {row['loglik_pointwise']:11.2f}"
              f"   difference {d:+8.2f} nats over the run")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(agreement=ag, cost=co), open(OUT, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
