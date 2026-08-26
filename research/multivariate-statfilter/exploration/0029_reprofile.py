"""Probe 0029 -- REPROFILE the backlog of discounted opens against the extended domain.

Premise (user): many "that doesn't matter much" verdicts were filed on SIMPLE domains (scalar,
H=I, quiet, orthogonal).  The domain has since exploded (5-DOF IMU fusion, phased noise, mixing
H, collinear modes, coupled dynamics).  The tell: the robotics case gives ~1.05x in quiet/
idealized regimes even with the suboptimalities in place, but gets much worse once realistic.
So a suboptimality dormant in the idealized regime may be load-bearing in the realistic one.

Method: for each open, measure its COST (mis-specified filter RMSE / an oracle that does not
have the suboptimality) in an IDEALIZED regime (quiet, constant) vs a REALISTIC one (phased
bursts), each with the open's violating feature present.  A verdict FLIPS if the cost is ~1 when
idealized but large when realistic.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
NJ, ORDER, DT = 5, 3, 0.01
POT, ACC, JERK = 0.06, 0.02, 0.6
T = 1400
BURST = [("SENSOR", 200, 400, "acc", 15), ("PROCESS", 600, 800, "jerk", 20),
         ("BOTH", 1000, 1200, "both", 15)]
G = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
GJ = np.kron(np.eye(NJ), G[:, None])                    # jerk-noise input


def build():
    return AdaptiveKalmanFilter.kinematic(NJ, ORDER, DT, process_var=JERK ** 2,
                                          meas_var={"pos": POT ** 2, "acc": ACC ** 2},
                                          measured=("pos", "acc"), control=True, s=0.5)


def sim(seed, phased, coupling=0.0, corrR=0.0):
    """Realistic rig with optional joint-coupling (violates fixed-V) and sensor-correlation
    (violates diagonal-R); phased=False is the idealized quiet regime."""
    f = build(); F, B, H, n, m = f.F, f.B, f.H, f.n, f.m
    rng = np.random.default_rng(seed); t = np.arange(T) * DT
    U = np.zeros((T, NJ))
    for j in range(NJ):
        for (a, w, ph) in [(2.0, 0.35 + 0.1 * j, j), (1.2, 0.7 + 0.13 * j, 2 * j)]:
            U[:, j] += a * np.sin(2 * np.pi * w * t + ph)
    jstd = np.full(T, JERK); pot = np.full(T, POT); acc = np.full(T, ACC)
    if phased:
        for nm, a, b, kind, mult in BURST:
            if kind in ("acc", "both"):
                acc[a:b] = ACC * mult
            if kind in ("jerk", "both"):
                jstd[a:b] = JERK * (20 if kind == "jerk" else mult)
    Cj = np.eye(NJ) + coupling * (np.ones((NJ, NJ)) - np.eye(NJ)); Lj = np.linalg.cholesky(Cj)
    Rc = np.eye(m)
    for i in range(1, m, 2):
        for j2 in range(1, m, 2):
            if i != j2:
                Rc[i, j2] = corrR
    Lr = np.linalg.cholesky(Rc)
    s = np.zeros(n); S = np.zeros((T, n)); Y = np.zeros((T, m))
    for k in range(T):
        jn = jstd[k] * (Lj @ rng.standard_normal(NJ))
        s = F @ s + B @ U[k] + GJ @ jn; S[k] = s
        sd = np.empty(m); sd[0::2] = pot[k]; sd[1::2] = acc[k]
        Y[k] = H @ s + sd * (Lr @ rng.standard_normal(m))
    return f, F, B, H, U, S, Y, jstd, pot, acc, Cj, Rc


def oracle(F, B, H, U, Y, jstd, pot, acc, Cj, Rc, n, m):
    m0 = np.zeros(n); P = np.eye(n); out = np.zeros((T, n))
    for k, y in enumerate(Y):
        Q = (jstd[k] ** 2) * (GJ @ Cj @ GJ.T)
        sd = np.empty(m); sd[0::2] = pot[k]; sd[1::2] = acc[k]
        R = (sd[:, None] * sd[None, :]) * Rc
        mp = F @ m0 + B @ U[k]; Pp = F @ P @ F.T + Q
        K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R); m0 = mp + K @ (y - H @ mp); P = Pp - K @ H @ Pp; out[k] = m0
    return out


def theta_rmse(est, S):
    tt = S.reshape(T, NJ, ORDER)[:, :, 0]; ee = est.reshape(T, NJ, ORDER)[:, :, 0]
    return np.sqrt(((ee - tt) ** 2).mean())


def cost(phased, coupling, corrR, seeds=4):
    a_e, o_e = [], []
    for seed in range(seeds):
        f, F, B, H, U, S, Y, jstd, pot, acc, Cj, Rc = sim(seed, phased, coupling, corrR)
        a_e.append(theta_rmse(f.filter(Y, U=U).mean, S))
        o_e.append(theta_rmse(oracle(F, B, H, U, Y, jstd, pot, acc, Cj, Rc, f.n, f.m), S))
    return np.mean(a_e), np.mean(o_e)


def main():
    print("REPROFILE: adaptive (block-diag Q0, diagonal R, learned scales) vs oracle-with-truth")
    print(f"  {'open / regime':40s} {'quiet cost':>11} {'bursty cost':>12}   flip?")
    rows = [
        ("baseline (no violation)", 0.0, 0.0),
        ("fixed V   (joints coupled 0.5)", 0.5, 0.0),
        ("diagonal R (sensors corr 0.6)", 0.0, 0.6),
        ("both violations (realistic)", 0.5, 0.6),
    ]
    for tag, cpl, cr in rows:
        aq, oq = cost(False, cpl, cr); ab, ob = cost(True, cpl, cr)
        cq, cb = aq / oq, ab / ob
        flip = "FLIPS" if (cb > 1.25 and cq < 1.1) else ("worse" if cb > 1.25 else "-")
        print(f"  {tag:40s} {cq:11.2f} {cb:12.2f}   {flip}")


if __name__ == "__main__":
    main()
