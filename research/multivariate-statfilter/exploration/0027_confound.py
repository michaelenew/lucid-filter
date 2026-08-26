"""Probe 0027 -- MEASURE the simultaneous process<->measurement confound.

Hypothesis (user): in the critical BOTH regime the process and measurement modes are
correlated, so the whiteness gate cannot cleanly split them.  In this rig the correlation is
STRUCTURAL: the accelerometer reads alpha (the top derivative), and process noise is jerk ->
alpha.  So the process mode's innovation signature and the accel-sensor's live in the SAME
channel; the potentiometer (reads theta) is nearly orthogonal to the process mode.

We measure it three ways:
  (A) ANALYTIC -- the scale-Fisher correlation  C_kl = F_kl / sqrt(F_kk F_ll)  at steady state,
      F_kl = 0.5 tr(S^-1 dS_k S^-1 dS_l).  This is the exact geometric overlap of two scale
      axes; |C|~1 means the data cannot tell them apart.  Reported process<->accel vs
      process<->pot.
  (B) EMPIRICAL cross-talk -- drive ONE true mode at a time (process-jerk / accel-sensor /
      pot-sensor) and read how each LEARNED scale responds: the confusion matrix.
  (C) The COST -- adaptive-vs-oracle state RMSE and the attribution leak in BOTH, contrasted
      with a pot-only rig (no process<->sensor overlap) to isolate what the correlation costs.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
NJ, ORDER, DT = 3, 3, 0.01                              # 3 joints keeps the matrices readable
POT_STD, ACC_STD, JERK_STD = 0.06, 0.02, 0.6


def build(measured=("pos", "acc")):
    mv = {"pos": POT_STD ** 2, "acc": ACC_STD ** 2}
    return AdaptiveKalmanFilter.kinematic(n_dof=NJ, order=ORDER, dt=DT, process_var=JERK_STD ** 2,
                                          meas_var=mv, measured=measured, control=True, s=0.5)


def steady_S(f):
    """Steady-state innovation covariance S at the base regime."""
    n, m, H, F = f.n, f.m, f.H, f.F
    P = np.eye(n) * (f.lam.max() + f.rho.max())
    Q0, R0 = f._Q_of(np.zeros(n)), f._R_of(np.zeros(m))
    for _ in range(600):
        Pp = F @ P @ F.T + Q0; S = H @ Pp @ H.T + R0
        K = Pp @ H.T @ np.linalg.inv(S); P = Pp - K @ H @ Pp
    return H @ (F @ P @ F.T + Q0) @ H.T + R0


def fisher_correlation(f):
    """C_kl = F_kl/sqrt(F_kk F_ll) over the active scale axes; and labels for each axis."""
    S = steady_S(f); Si = np.linalg.inv(S); z = np.zeros(f.D)
    act = f.active
    dS = [f._dS(z, int(k)) for k in act]
    SidS = [Si @ d for d in dS]
    F = np.array([[0.5 * np.trace(SidS[a] @ SidS[b]) for b in range(len(act))] for a in range(len(act))])
    d = np.sqrt(np.clip(np.diag(F), 1e-30, None))
    C = F / np.outer(d, d)
    # label each active axis: process eigenmode "P<j>" or sensor "pot<j>"/"acc<j>"
    labels = []
    for k in act:
        if k < f.n:
            j = k // ORDER; labels.append(f"P{j}")        # process eigenmode (per joint block)
        else:
            i = k - f.n; j = i // 2
            labels.append(f"{'pot' if i % 2 == 0 else 'acc'}{j}")
    return C, labels


def sim(f, hot, amp_var, seed=0, T=500):
    """Drive one true mode hot in the middle third; return Y, U, true state, and phase band."""
    F, B, H, n, m = f.F, f.B, f.H, f.n, f.m
    rng = np.random.default_rng(seed); t = np.arange(T) * DT; on = slice(T // 3, 2 * T // 3)
    U = np.zeros((T, NJ))
    for j in range(NJ):
        U[:, j] += 2.0 * np.sin(2 * np.pi * (0.4 + 0.1 * j) * t + j)
    g = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
    Bn = np.kron(np.eye(NJ), g[:, None])
    jstd = np.full(T, JERK_STD); pot = np.full(T, POT_STD); acc = np.full(T, ACC_STD)
    if hot == "process":
        jstd[on] = JERK_STD * math.sqrt(amp_var)
    elif hot == "acc":
        acc[on] = ACC_STD * math.sqrt(amp_var)
    elif hot == "pot":
        pot[on] = POT_STD * math.sqrt(amp_var)
    elif hot == "both":                                   # process + the OVERLAPPING sensor (accel)
        jstd[on] = JERK_STD * math.sqrt(amp_var); acc[on] = ACC_STD * math.sqrt(amp_var)
    elif hot == "both_pot":                               # process + the ORTHOGONAL sensor (pot)
        jstd[on] = JERK_STD * math.sqrt(amp_var); pot[on] = POT_STD * math.sqrt(amp_var)
    s = np.zeros(n); S = np.zeros((T, n)); Y = np.zeros((T, m))
    for k in range(T):
        s = F @ s + B @ U[k] + Bn @ (jstd[k] * rng.standard_normal(NJ)); S[k] = s
        std = np.empty(m); std[0::2] = pot[k]; std[1::2] = acc[k]
        Y[k] = H @ s + std * rng.standard_normal(m)
    return Y, U, S, on, jstd, pot, acc


def main():
    f = build()
    C, labels = fisher_correlation(f)
    # dominant process modes: process axes whose base Fisher is non-negligible (skip the rank-
    # deficient near-zero eigenmodes of the rank-1 jerk Q0)
    proc_ax = [i for i, lab in enumerate(labels) if lab.startswith("P") and f._Ichar[f.active[i]] > 1e-6]
    acc_ax = [i for i, lab in enumerate(labels) if lab.startswith("acc")]
    pot_ax = [i for i, lab in enumerate(labels) if lab.startswith("pot")]
    print("(A) scale-Fisher correlation |C_kl| at steady state -- max over process modes per sensor:")
    ca = [max(abs(C[p, s]) for p in proc_ax) for s in acc_ax]
    cp = [max(abs(C[p, s]) for p in proc_ax) for s in pot_ax]
    print(f"     |corr(process, ACCEL)| per joint: {np.array(ca)}  mean {np.mean(ca):.3f}")
    print(f"     |corr(process, POT)|   per joint: {np.array(cp)}  mean {np.mean(cp):.3f}")
    print("     -> the accelerometer sits in the SAME channel as process (jerk->alpha); the pot does not.")

    print("\n(B) empirical cross-talk: drive ONE mode hot (var x100), read the learned scales (band mean):")
    print(f"     {'driven':10s} {'->accel-scale':>14} {'->pot-scale':>12} {'->process-scale':>16}")
    for hot in ("process", "acc", "pot"):
        Y, U, S, on, *_ = sim(f, hot, 100.0)
        r = f.filter(Y, U=U)
        ms, ps = r.measurement_scale, r.process_scale
        a = ms[on, 1::2].mean(); p = ms[on, 0::2].mean()
        pr = ps.reshape(len(Y), NJ, ORDER)[on, :, 2].mean()
        print(f"     {hot:10s} {a:14.2f} {p:12.2f} {pr:16.2f}")

    print("\n(C) cost of the confound (var x100 in the hot band), 4 seeds, SAME pot+acc rig.")
    print("    process+ACCEL is the correlated case; process+POT is the orthogonal control:")
    g = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
    for tag, hot in [("process + ACCEL  (correlated)", "both"),
                     ("process + POT    (orthogonal)", "both_pot"),
                     ("process alone", "process"), ("accel alone", "acc")]:
        ad_e, or_e, leak = [], [], []
        for seed in range(4):
            Y, U, S, on, jstd, pot, acc = sim(f, hot, 100.0, seed=seed)
            th_t = S.reshape(len(Y), NJ, ORDER)[:, :, 0]
            r = f.filter(Y, U=U); ad = r.mean.reshape(len(Y), NJ, ORDER)[:, :, 0]
            orc = p26_oracle(f, Y, U, jstd, pot, acc, g).reshape(len(Y), NJ, ORDER)[:, :, 0]
            ad_e.append(np.sqrt(((ad[on] - th_t[on]) ** 2).mean()))
            or_e.append(np.sqrt(((orc[on] - th_t[on]) ** 2).mean()))
            # mis-attribution: process driven -> accel scale should stay ~0 (and vice versa)
            leak.append(r.measurement_scale[on, 1::2].mean())
        print(f"     {tag:30s}  adaptive {np.mean(ad_e):.4f}  oracle {np.mean(or_e):.4f}  "
              f"gap {np.mean(ad_e)/np.mean(or_e):.2f}x  accel-scale {np.mean(leak):+.2f}")

    print("\n(D) WHY the temporal signature can't fix it: per-channel innovation lag-1 autocorr")
    print("    under PURE process noise (non-adaptive, fixed base noise, hot band):")
    fna = build(); fna.active = np.array([], dtype=int); fna.r = 0
    ac_pot, ac_acc = [], []
    for seed in range(6):
        Y, U, S, on, *_ = sim(f, "process", 100.0, seed=seed)
        e = fna.filter(Y, U=U).innovation[on]              # (band, m)
        for c in range(0, e.shape[1], 2):                  # pot channels
            x = e[:, c]; ac_pot.append(np.corrcoef(x[1:], x[:-1])[0, 1])
        for c in range(1, e.shape[1], 2):                  # accel channels
            x = e[:, c]; ac_acc.append(np.corrcoef(x[1:], x[:-1])[0, 1])
    print(f"     POT   channel lag-1 autocorr: {np.mean(ac_pot):+.3f}  (process is CORRELATED here -> detectable)")
    print(f"     ACCEL channel lag-1 autocorr: {np.mean(ac_acc):+.3f}  (process is ~WHITE here -> looks like sensor noise)")
    print("     => process noise is separable from the POT channel's correlation, but NOT from the")
    print("        accel channel -- the accelerometer sits on the same derivative the jerk enters.")


def p26_oracle(f, Y, U, jstd, pot, acc, g):
    F, B, H, n, m = f.F, f.B, f.H, f.n, f.m
    Bn = np.kron(np.eye(NJ), g[:, None]); m0 = np.zeros(n); P = np.eye(n); out = np.zeros((len(Y), n))
    for k, y in enumerate(Y):
        Q = (jstd[k] ** 2) * (Bn @ Bn.T)
        R = np.zeros((m, m)); R[np.arange(0, m, 2), np.arange(0, m, 2)] = pot[k] ** 2
        if m > NJ:
            R[np.arange(1, m, 2), np.arange(1, m, 2)] = acc[k] ** 2
        mp = F @ m0 + B @ U[k]; Pp = F @ P @ F.T + Q
        K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R); m0 = mp + K @ (y - H @ mp)
        P = Pp - K @ H @ Pp; out[k] = m0
    return out


if __name__ == "__main__":
    main()
