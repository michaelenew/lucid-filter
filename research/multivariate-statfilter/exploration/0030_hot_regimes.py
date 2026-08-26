"""Probe 0030 -- solve the HOT regimes (the absolute sensor fails) to the irreducible floor.

Reprofile (0029) found the expensive realistic regime is adaptation lag when the ABSOLUTE
position sensor (the bad pot) degrades: position observability collapses (the accelerometer
only integrates to a drifting position), and the filter LAGS in shedding the failing pot --
pot-hot 1.86x, process+pot 3.72x.  The lag is front-loaded: for the first ~1/beta steps of the
burst the scale hasn't ramped, so the filter trusts a sensor it should discard, and position
never recovers.

Floor = oracle-lagged (EMA-beta-lagged true Q,R -- the best a smooth beta-windowed estimator
can do).  A CHANGE-DETECTING estimator can beat even that: a per-sensor robust weight sheds an
outlier sensor at the FIRST corrupted sample instead of ramping an EMA over ~50 steps.

This probe measures the current gap, the floor, and (below) tests the robust fix.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402

np.set_printoptions(precision=4, suppress=True)
NJ, ORDER, DT = 5, 3, 0.01
POT, ACC, JERK = 0.06, 0.02, 0.6
T = 1000
ON = slice(400, 700)                                   # the hot window
G = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
GJ = np.kron(np.eye(NJ), G[:, None])


def build():
    return AdaptiveKalmanFilter.kinematic(NJ, ORDER, DT, process_var=JERK ** 2,
                                          meas_var={"pos": POT ** 2, "acc": ACC ** 2},
                                          measured=("pos", "acc"), control=True, s=0.5)


def sim(seed, regime, pot_mult=15.0, jerk_mult=20.0):
    f = build(); F, B, H, n, m = f.F, f.B, f.H, f.n, f.m
    rng = np.random.default_rng(seed); t = np.arange(T) * DT
    U = np.zeros((T, NJ))
    for j in range(NJ):
        for (a, w, ph) in [(2.0, 0.35 + 0.1 * j, j), (1.2, 0.7 + 0.13 * j, 2 * j)]:
            U[:, j] += a * np.sin(2 * np.pi * w * t + ph)
    jstd = np.full(T, JERK); pot = np.full(T, POT); acc = np.full(T, ACC)
    a0, b0 = ON.start, ON.stop
    if regime in ("pot", "procpot"):
        pot[a0:b0] = POT * pot_mult
    if regime in ("proc", "procpot"):
        jstd[a0:b0] = JERK * jerk_mult
    s = np.zeros(n); S = np.zeros((T, n)); Y = np.zeros((T, m))
    for k in range(T):
        s = F @ s + B @ U[k] + GJ @ (jstd[k] * rng.standard_normal(NJ)); S[k] = s
        sd = np.empty(m); sd[0::2] = pot[k]; sd[1::2] = acc[k]
        Y[k] = H @ s + sd * rng.standard_normal(m)
    return f, F, B, H, U, S, Y, jstd, pot, acc


def oracle(F, B, H, U, Y, jstd, pot, acc, n, m, beta=None):
    if beta is not None:                               # oracle-lagged: EMA-lag the true noise
        jl = np.empty(T); pl = np.empty(T); al = np.empty(T); cj = jstd[0] ** 2; cp = pot[0] ** 2; ca = acc[0] ** 2
        for k in range(T):
            cj = (1 - beta) * cj + beta * jstd[k] ** 2; cp = (1 - beta) * cp + beta * pot[k] ** 2
            ca = (1 - beta) * ca + beta * acc[k] ** 2; jl[k] = cj; pl[k] = cp; al[k] = ca
        jstd = np.sqrt(jl); pot = np.sqrt(pl); acc = np.sqrt(al)
    m0 = np.zeros(n); P = np.eye(n); out = np.zeros((T, n))
    for k, y in enumerate(Y):
        Q = jstd[k] ** 2 * (GJ @ GJ.T); sd = np.empty(m); sd[0::2] = pot[k]; sd[1::2] = acc[k]; R = np.diag(sd ** 2)
        mp = F @ m0 + B @ U[k]; Pp = F @ P @ F.T + Q
        K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R); m0 = mp + K @ (y - H @ mp); P = Pp - K @ H @ Pp; out[k] = m0
    return out


def rms(est, S):
    tt = S.reshape(T, NJ, ORDER)[:, ON, 0] if False else S.reshape(T, NJ, ORDER)[ON, :, 0]
    ee = est.reshape(T, NJ, ORDER)[ON, :, 0]
    return np.sqrt(((ee - tt) ** 2).mean())


def main():
    print("HOT REGIMES: joint-angle RMSE in the hot window (400:700), 4 seeds")
    print(f"  {'regime':10s} {'adaptive':>9} {'orc-lag':>9} {'oracle':>8}   {'AD/orc':>7} {'AD/lag':>7} {'lag/orc':>7}")
    for regime, tag in [("pot", "pot-hot"), ("procpot", "process+pot")]:
        ad = []; ol = []; oc = []
        for seed in range(4):
            f, F, B, H, U, S, Y, jstd, pot, acc = sim(seed, regime)
            ad.append(rms(f.filter(Y, U=U).mean, S))
            ol.append(rms(oracle(F, B, H, U, Y, jstd, pot, acc, f.n, f.m, beta=0.02), S))
            oc.append(rms(oracle(F, B, H, U, Y, jstd, pot, acc, f.n, f.m), S))
        a, l, o = np.mean(ad), np.mean(ol), np.mean(oc)
        print(f"  {tag:10s} {a:9.4f} {l:9.4f} {o:8.4f}   {a/o:7.2f} {a/l:7.2f} {l/o:7.2f}")


if __name__ == "__main__":
    main()
