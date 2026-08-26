"""Probe 0034 -- HIGH-SEED paired profiling of a filter variant across all regimes.

The 0032 integration variants showed small per-regime moves at 4 seeds; whether they are real or
run-to-run noise needs error bars.  This runs N seeds on the 5-DOF rig for five regimes and reports
the adaptive/oracle ratio as mean +/- standard error.

Usage:
    python 0034_profile.py save <label>    # run N seeds on the CURRENTLY CHECKED-OUT filter, dump
                                           #   per-seed adaptive & oracle RMSE to 0034_<label>.npz
    python 0034_profile.py compare A B      # paired diff of two saved runs (same seeds -> the
                                           #   scenario/oracle variance cancels, tight error bar)

The oracle depends only on (regime, seed), not the filter, so the paired difference of the two
labels' ratios isolates the variant effect.  A variant that is within error bars of the baseline is
statistically indistinguishable -- and then the one that UNDERSTANDS the regime (the derived law)
wins on principle.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402

HERE = os.path.dirname(__file__)
NJ, ORDER, DT = 5, 3, 0.01
POT, ACC, JERK = 0.06, 0.02, 0.6
T = 1000
ON = slice(400, 700)
G = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
GJ = np.kron(np.eye(NJ), G[:, None])
N_SEEDS = 40
REGIMES = [("pot", "pot-hot"), ("procpot", "process+pot"), ("sens", "SENSOR"),
           ("proc", "PROCESS"), ("both", "BOTH")]


def build():
    return AdaptiveKalmanFilter.kinematic(NJ, ORDER, DT, process_var=JERK ** 2,
                                          meas_var={"pos": POT ** 2, "acc": ACC ** 2},
                                          measured=("pos", "acc"), control=True, s=0.5)


def sim(seed, regime, pot_mult=15.0, jerk_mult=20.0, acc_mult=15.0):
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
    if regime in ("proc", "procpot", "both"):
        jstd[a0:b0] = JERK * jerk_mult
    if regime in ("sens", "both"):
        acc[a0:b0] = ACC * acc_mult
    s = np.zeros(n); S = np.zeros((T, n)); Y = np.zeros((T, m))
    for k in range(T):
        s = F @ s + B @ U[k] + GJ @ (jstd[k] * rng.standard_normal(NJ)); S[k] = s
        sd = np.empty(m); sd[0::2] = pot[k]; sd[1::2] = acc[k]
        Y[k] = H @ s + sd * rng.standard_normal(m)
    return f, F, B, H, U, S, Y, jstd, pot, acc


def oracle(F, B, H, U, Y, jstd, pot, acc, n, m):
    m0 = np.zeros(n); P = np.eye(n); out = np.zeros((T, n))
    for k, y in enumerate(Y):
        Q = jstd[k] ** 2 * (GJ @ GJ.T); sd = np.empty(m); sd[0::2] = pot[k]; sd[1::2] = acc[k]
        R = np.diag(sd ** 2)
        mp = F @ m0 + B @ U[k]; Pp = F @ P @ F.T + Q
        K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R); m0 = mp + K @ (y - H @ mp); P = Pp - K @ H @ Pp
        out[k] = m0
    return out


def rms(est, S):
    tt = S.reshape(T, NJ, ORDER)[ON, :, 0]; ee = est.reshape(T, NJ, ORDER)[ON, :, 0]
    return float(np.sqrt(((ee - tt) ** 2).mean()))


def save(label):
    data = {}
    for regime, tag in REGIMES:
        ad = np.zeros(N_SEEDS); oc = np.zeros(N_SEEDS)
        for seed in range(N_SEEDS):
            f, F, B, H, U, S, Y, jstd, pot, acc = sim(seed, regime)
            ad[seed] = rms(f.filter(Y, U=U).mean, S)
            oc[seed] = rms(oracle(F, B, H, U, Y, jstd, pot, acc, f.n, f.m), S)
        data[f"{regime}_ad"] = ad; data[f"{regime}_oc"] = oc
        r = ad / oc
        print(f"  {tag:12s} adaptive/oracle = {r.mean():.3f} +/- {r.std(ddof=1)/math.sqrt(N_SEEDS):.3f}"
              f"   (oracle RMSE {oc.mean():.4f})")
    np.savez(os.path.join(HERE, f"0034_{label}.npz"), **data)
    print(f"saved 0034_{label}.npz  ({N_SEEDS} seeds)")


def compare(a, b):
    da = np.load(os.path.join(HERE, f"0034_{a}.npz")); db = np.load(os.path.join(HERE, f"0034_{b}.npz"))
    print(f"paired diff (adaptive/oracle):  {b} - {a}   [same seeds; +/- = SE of the paired diff]")
    print(f"  {'regime':12s} {a+' ratio':>14} {b+' ratio':>14} {'diff':>9} {'SE':>7} {'sigmas':>7}")
    for regime, tag in REGIMES:
        oc = da[f"{regime}_oc"]
        ra = da[f"{regime}_ad"] / oc; rb = db[f"{regime}_ad"] / da[f"{regime}_oc"]
        # guard: same seeds => same oracle; recompute rb against b's own oracle for safety
        rb = db[f"{regime}_ad"] / db[f"{regime}_oc"]
        d = rb - ra; se = d.std(ddof=1) / math.sqrt(len(d))
        sig = d.mean() / se if se > 0 else 0.0
        print(f"  {tag:12s} {ra.mean():14.3f} {rb.mean():14.3f} {d.mean():+9.3f} {se:7.3f} {sig:+7.1f}")


def floor(regime="both", nseed=20):
    """Achievable floor for the masked-Q regime: oracle-R (freeze sensors at the true time-varying
    scale, INFER the process) vs the full oracle.  Shows the full-oracle ratio overstates BOTH --
    inferring the masked Q is intrinsically ~3x, and the adaptive already sits below it."""
    acc_mult, jerk_mult = 15.0, 20.0
    ad = np.zeros(nseed); oc = np.zeros(nseed)
    for seed in range(nseed):
        f, F, B, H, U, S, Y, jstd, pot, acc = sim(seed, regime)
        n = f.n
        f.active = np.array([kk for kk in f.active if kk < n]); f.r = len(f.active)   # infer Q only
        est = np.zeros((T, n))
        for k in range(T):
            on = ON.start <= k < ON.stop
            f.mu[n:] = 0.0
            f.mu[n + 1::2] = 2 * math.log(acc_mult) if on else 0.0                     # true accel R
            est[k] = f.update(Y[k], u=U[k]).mean
        ad[seed] = rms(est, S); oc[seed] = rms(oracle(F, B, H, U, Y, jstd, pot, acc, n, f.m), S)
    r = ad / oc
    print(f"{regime} oracle-R (true R, INFER Q) / oracle = {r.mean():.3f} +/- "
          f"{r.std(ddof=1) / math.sqrt(nseed):.3f}   (the achievable floor for the masked Q)")


if __name__ == "__main__":
    if sys.argv[1] == "save":
        save(sys.argv[2])
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "floor":
        floor()
