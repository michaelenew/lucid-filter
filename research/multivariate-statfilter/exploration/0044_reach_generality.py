"""Probe 0044 -- generality/safety of the derived spatial reach across sensor structures.

The 0043 reach is net-positive on the pot+accel rig, where each joint has one integrated sensor (pot)
and one direct process-readout (accel). The "always-safe" claim rests on the eligibility degrading
gracefully when that structure is absent: with NO witnessing partner the reach must switch OFF (reduce
to the floor), never misfire. Two things to confirm:
  1. elig weights across sensor configs are sane (single sensor -> 0; redundant -> shared; etc).
  2. On a single-sensor-per-joint rig (no witness) the derived reach == floor on EVERY regime,
     including process (where a naive reach misfires).
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402

import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("p43", os.path.join(os.path.dirname(__file__), "0043_derived_reach.py"))
p43 = importlib.util.module_from_spec(_s); _s.loader.exec_module(p43)

NJ, ORDER, DT = 5, 3, 0.01
JERK, POT, ACC, VEL = 0.6, 0.06, 0.02, 0.03
T = 1000
ON = slice(400, 700)
G = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
GJ = np.kron(np.eye(NJ), G[:, None])
IDX = {"pos": 0, "vel": 1, "acc": 2}
BASEVAR = {"pos": POT ** 2, "vel": VEL ** 2, "acc": ACC ** 2}


def make(measured):
    return p43.DerivedReach.kinematic(NJ, ORDER, DT, process_var=JERK ** 2,
                                      meas_var={k: BASEVAR[k] for k in measured},
                                      measured=measured, control=True, s=0.5)


def show_elig():
    print("derived eligibility per sensor config (per-joint sensor list -> elig of joint 0's sensors):")
    for measured in [("pos",), ("acc",), ("pos", "acc"), ("pos", "vel", "acc"), ("pos", "pos")]:
        f = make(measured)
        Sdiag = np.diag(f.H @ np.eye(f.n) @ f.H.T) + f.rho
        f._prep(Sdiag)
        e = ", ".join(f"{nm}={f._elig[j]:.2f}" for j, nm in enumerate(measured))
        print(f"  {str(measured):22s} -> [{e}]")


def sim(seed, regime, measured):
    f = make(measured); F, B, H, n, m = f.F, f.B, f.H, f.n, f.m
    rng = np.random.default_rng(seed); t = np.arange(T) * DT
    U = np.zeros((T, NJ))
    for j in range(NJ):
        for (a, w, ph) in [(2.0, 0.35 + 0.1 * j, j), (1.2, 0.7 + 0.13 * j, 2 * j)]:
            U[:, j] += a * np.sin(2 * np.pi * w * t + ph)
    jstd = np.full(T, JERK); sensc = {k: np.full(T, math.sqrt(BASEVAR[k])) for k in measured}
    a0, b0 = ON.start, ON.stop
    if regime == "pot-hot" and "pos" in measured:
        sensc["pos"][a0:b0] *= 15.0
    if regime == "process":
        jstd[a0:b0] = JERK * 20.0
    s = np.zeros(n); S = np.zeros((T, n)); Y = np.zeros((T, m))
    for k in range(T):
        s = F @ s + B @ U[k] + GJ @ (jstd[k] * rng.standard_normal(NJ)); S[k] = s
        sd = np.empty(m); col = 0
        for d in range(NJ):
            for name in measured:
                sd[col] = sensc[name][k]; col += 1
        Y[k] = H @ s + sd * rng.standard_normal(m)
    return f, F, B, H, U, S, Y, jstd, sensc, measured


def oracle(F, B, H, U, Y, jstd, sensc, measured, n, m):
    m0 = np.zeros(n); P = np.eye(n); out = np.zeros((T, n))
    for k, y in enumerate(Y):
        Q = jstd[k] ** 2 * (GJ @ GJ.T)
        sd = np.empty(m); col = 0
        for d in range(NJ):
            for name in measured:
                sd[col] = sensc[name][k]; col += 1
        R = np.diag(sd ** 2)
        mp = F @ m0 + B @ U[k]; Pp = F @ P @ F.T + Q
        K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R); m0 = mp + K @ (y - H @ mp); P = Pp - K @ H @ Pp
        out[k] = m0
    return out


def rms(est, S):
    tt = S.reshape(T, NJ, ORDER)[ON, :, 0]; ee = est.reshape(T, NJ, ORDER)[ON, :, 0]
    return float(np.sqrt(((ee - tt) ** 2).mean()))


def safety(measured, nseed=15):
    print(f"\nsafety on {measured} (no witness -> reach must == floor): adaptive/oracle")
    print(f"  {'regime':10s} {'floor':>9} {'derived':>9} {'diff':>8}")
    for regime in ("pot-hot", "process"):
        fl = np.zeros(nseed); dv = np.zeros(nseed)
        for seed in range(nseed):
            f, F, B, H, U, S, Y, jstd, sensc, meas = sim(seed, regime, measured)
            oc = rms(oracle(F, B, H, U, Y, jstd, sensc, meas, f.n, f.m), S)
            f.qreach = 0.0; fl[seed] = rms(f.filter(Y, U=U).mean, S) / oc
            f2, *_ = sim(seed, regime, measured); f2.qreach = 4.0
            dv[seed] = rms(f2.filter(Y, U=U).mean, S) / oc
        print(f"  {regime:10s} {fl.mean():9.3f} {dv.mean():9.3f} {dv.mean()-fl.mean():+8.3f}")


def main():
    show_elig()
    safety(("pos",))          # single integrated sensor per joint -- NO witness
    safety(("acc",))          # single direct-readout sensor per joint -- NO witness


if __name__ == "__main__":
    main()
