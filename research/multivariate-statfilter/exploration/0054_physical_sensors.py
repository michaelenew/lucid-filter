"""0054 -- sensors the arm could actually carry, and what the sensor-model shortcuts cost.

`0052` gave every joint an "accelerometer" reading that joint's own angular acceleration
through a constant, diagonal ``H``.  No sensor does that.  This probe replaced it in two
steps, and both are part of the record:

* **First cut, an angular sensor.**  A link-mounted angular-rate/acceleration sensor reads
  the CHAIN beneath it, ``sum_{i<=j} (a_j . a_i) al_i`` -- but on a serial chain the
  couplings are axis dot products, constant wherever axes are parallel or orthogonal.  On
  the old chain that was every coupling but one: a constant COUPLED ``H`` (the "simple sum")
  fixed links 1-4 exactly and left a single varying coefficient.  The honest headline there
  was "diagonal where it should have been a constant sum"; a reactive ``H`` was motivated,
  not forced.
* **The rig as it stands, a linear accelerometer** (`../scripts/arm5dof.py`) on the common
  5-DOF chain (yaw+pitch base, pitch+roll elbow, one wrist flex).  A MEMS accelerometer at
  each link's distal end reads proper acceleration: a configuration-dependent lever-arm map
  on the joint accelerations, centripetal/Coriolis terms quadratic in the rates, and
  **gravity resolved in the link frame** -- 9.81 m/s^2 that moves with every joint below.
  There is no constant ``H`` here at all, so the map reaches the shipped filter as
  ``LucidFilter(H=callable)`` and is linearised at every step.

Questions, in order:

  1. Is the complex-step Jacobian right?  (against central differences)
  2. On the physical rig, how does the filter do against an oracle told the true noise
     schedule, and against the same model frozen at the base noise?
  3. What does FREEZING the linearisation cost -- the same filter handed ``H_CHAR``, the
     map linearised once at the home pose and never again?  That is the ablation that shows
     the per-step linearisation is load-bearing, not decoration.
  4. Why the dynamic sensor is an accelerometer and not a rate gyro -- a relative-degree
     limit of the per-step scale walk, isolated on a diagonal control rig.

Run: python 0054_physical_sensors.py        (~25 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import arm5dof as A                                                    # noqa: E402
from lucid import LucidFilter                                          # noqa: E402

T = 1900
NS = 3
POT_MULT, ACC_MULT, JERK_MULT, FAIL_J = 15.0, 15.0, 20.0, 2
PHASES = [("calm", 0, 250), ("SENSOR", 250, 500), ("calm", 500, 650),
          ("PROCESS", 650, 900), ("calm", 900, 1050), ("POTFAIL", 1050, 1300),
          ("calm", 1300, 1450), ("BOTH", 1450, 1700), ("calm", 1700, 1900)]
SKIP = 40                                # each regime's own onset transient, not scored here


def se(v):
    v = np.asarray(v, float)
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if v.size > 1 else 0.0


def schedule():
    jstd = np.full(T, A.JERK)
    pot = np.full((T, A.NJ), A.POT)
    acc = np.full((T, A.NJ), A.ACC)
    for nm, a, b in PHASES:
        if nm == "SENSOR":
            acc[a:b] *= ACC_MULT
        elif nm == "PROCESS":
            jstd[a:b] *= JERK_MULT
        elif nm == "POTFAIL":
            pot[a:b, FAIL_J] *= POT_MULT
        elif nm == "BOTH":
            acc[a:b] *= ACC_MULT
            jstd[a:b] *= JERK_MULT
    return jstd, pot, acc


def tip(Z):
    return np.array([A.joints3d(t) for t in Z.reshape(len(Z), A.NJ, A.ORDER)[:, :, 0]])[:, -1]


def rms(P, Pt, sl):
    return float(np.sqrt(((P[sl] - Pt[sl]) ** 2).sum(1).mean()))


def windows():
    w = {}
    for nm, a, b in PHASES:
        w.setdefault(nm, []).append((a + SKIP, b))
    return w


def mask(spans):
    m = np.zeros(T, bool)
    for a, b in spans:
        m[a:b] = True
    return m


def jacobian_check():
    r = np.random.default_rng(0)
    worst = 0.0
    for _ in range(8):
        x = r.standard_normal(A.N) * 0.5
        Hm = A.measure(x)[0]
        num = np.zeros((A.M, A.N))
        for i in range(A.N):
            e = np.zeros(A.N)
            e[i] = 1e-6
            num[:, i] = (A.measure(x + e)[1] - A.measure(x - e)[1]) / 2e-6
        worst = max(worst, float(np.abs(Hm - num).max()))
    print(f"1. complex-step Jacobian vs central differences, worst abs error: {worst:.2e}")
    Hc = A.H_CHAR
    y0 = A.measure(np.zeros(A.N))[1]
    print("   what each accelerometer reads at the home pose:")
    print(f"   {'':>10} {'gravity (m/s^2)':>16}   {'d/d theta (per rad)':>28}   "
          f"{'lever arms d/d alpha (m)':>28}")
    for j in range(A.NJ):
        g = 2 * j + 1
        print(f"   link {j + 1:>4} {y0[g]:>16.2f}   "
              f"{np.array2string(np.round(Hc[g, 0::3], 1), floatmode='fixed'):>28}   "
              f"{np.array2string(np.round(Hc[g, 2::3], 2), floatmode='fixed'):>28}")
    print("   (the riser accelerometer reads ~nothing under yaw -- physically correct; yaw"
          "\n    redundancy rides on links 2-5 through the chain)")


def main():
    t0 = time.time()
    jacobian_check()

    W = windows()
    acc_r = {k: {b: [] for b in ("lucid", "fixed", "frozenH")} for k in W}
    burst = {b: [] for b in ("raw", "lucid", "fixed", "oracle", "frozenH")}
    Y0 = A.measure(np.zeros(A.N))[1]         # h(0): the gravity offset, computed once

    for sd in range(NS):
        jstd, pot, acc = schedule()
        U, S, Y = A.simulate(sd, jstd, pot, acc)
        Qs = [j ** 2 * (A.B @ A.B.T) for j in jstd]
        Rs = [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2]
                              for j in range(A.NJ)]) for k in range(T)]
        est = {"lucid": A.make_filter().filter(Y, U).mean,
               "fixed": A.kalman(U, Y),
               "oracle": A.kalman(U, Y, Qs, Rs),
               # the ablation: the same lucid filter, H linearised ONCE at the home pose.
               # (A constant H cannot carry h(0) either, so this contender is also spotted
               # the true gravity offset -- the handicap is purely the frozen Jacobian.)
               "frozenH": LucidFilter(dynamics=A.F, control=A.B,
                                      H=lambda x: (A.H_CHAR, A.H_CHAR @ x + Y0),
                                      process=A.Q0,
                                      measurement=A.R0).filter(Y, U).mean}
        Pt = tip(S)
        P = {k: tip(v) for k, v in est.items()}
        P["raw"] = np.array([A.joints3d(Y[k, 0::3])[-1] for k in range(T)])
        on = np.zeros(T, bool)
        for nm, a, b in PHASES:
            if nm != "calm":
                on[a:b] = True
        for b in burst:
            burst[b].append(rms(P[b], Pt, on))
        for key, spans in W.items():
            sl = mask(spans)
            o = rms(P["oracle"], Pt, sl)
            for b in ("lucid", "fixed", "frozenH"):
                acc_r[key][b].append(rms(P[b], Pt, sl) / o)
        print(f"   seed {sd} done ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n2/3. tip RMSE over the bursts ({NS} seeds), metres")
    for b in ("raw", "fixed", "oracle", "lucid", "frozenH"):
        print(f"     {b:<8} {np.mean(burst[b]):.4f} +- {se(burst[b]):.4f}")
    print("\n     ratio to an oracle told the true noise schedule "
          "(`frozen-H` is the same lucid filter, linearised once at home)")
    print(f"     {'regime':<10}{'lucid':>10}{'fixed':>10}{'frozen-H':>12}")
    for key in ("calm", "SENSOR", "PROCESS", "POTFAIL", "BOTH"):
        row = [np.mean(acc_r[key][b]) for b in ("lucid", "fixed", "frozenH")]
        print(f"     {key:<10}{row[0]:>10.2f}{row[1]:>10.2f}{row[2]:>12.2f}")

    print("\n4. why the dynamic sensor is an accelerometer, not a rate gyro (1 seed)")
    print("   A jerk disturbance reaches a rate sensor only through dt^2/2 and an angular-")
    print("   acceleration sensor through dt -- 200x more per step at this rate.  The scale")
    print("   walk scores per step, so on a rate sensor a process burst is nearly invisible")
    print("   in one step while the sensor axis is not, and the burst is blamed on the")
    print("   sensor.  Both control rigs are DIAGONAL; only the read derivative differs.")
    for lab, di, sig in (("angular accel (reads alpha)", 2, 0.020),
                         ("rate gyro     (reads omega)", 1, 0.010)):
        jstd, pot, _ = schedule()
        dyn = np.full((T, A.NJ), sig)
        for nm, a, b in PHASES:
            if nm in ("SENSOR", "BOTH"):
                dyn[a:b] *= ACC_MULT
        rows = []
        for j in range(A.NJ):
            e = np.zeros(A.N); e[A.ORDER * j] = 1.0; rows.append(e)
            e = np.zeros(A.N); e[A.ORDER * j + di] = 1.0; rows.append(e)
        Hm = np.array(rows)
        M4 = len(rows)
        R0 = np.tile([A.POT ** 2, sig ** 2], A.NJ)
        rng = np.random.default_rng(0)

        class _Servo2(A.Servo):                # the control rig interleaves 2 rows per joint
            def observe(self, y):
                A.Servo.observe(self, np.repeat(y[0::2], 3))

        servo = _Servo2(); s = np.zeros(A.N)
        S = np.zeros((T, A.N)); Y = np.zeros((T, M4)); U = np.zeros((T, A.NJ))
        rth, rom, ral, rjk = A.reference(T)
        sdv = np.empty(M4)
        sdv[0::2], sdv[1::2] = pot[0], dyn[0]
        y = Hm @ s + sdv * rng.standard_normal(M4)
        for k in range(T):
            servo.observe(y)
            U[k] = servo.command(rth[k], rom[k], ral[k], rjk[k])
            s = A.F @ s + A.B @ U[k] + A.B @ (jstd[k] * rng.standard_normal(A.NJ))
            sdv[0::2], sdv[1::2] = pot[k], dyn[k]
            y = Hm @ s + sdv * rng.standard_normal(M4)
            S[k], Y[k] = s, y
        lu = LucidFilter(dynamics=A.F, control=A.B, H=Hm, process=A.Q0,
                         measurement=R0).filter(Y, U).mean
        Qs = [j ** 2 * (A.B @ A.B.T) for j in jstd]
        m0 = np.zeros(A.N); Pk = np.eye(A.N); orc = np.zeros((T, A.N))
        for k in range(T):
            mp = A.F @ m0 + A.B @ U[k]
            Pp = A.F @ Pk @ A.F.T + Qs[k]
            Rk = np.diag(np.concatenate([[pot[k, j] ** 2, dyn[k, j] ** 2]
                                         for j in range(A.NJ)]))
            K = Pp @ Hm.T @ np.linalg.inv(Hm @ Pp @ Hm.T + Rk)
            m0 = mp + K @ (Y[k] - Hm @ mp)
            Pk = Pp - K @ Hm @ Pp
            orc[k] = m0
        Pt, Pl, Po = tip(S), tip(lu), tip(orc)
        W4 = windows()
        out = "  ".join(f"{nm} {rms(Pl, Pt, mask(W4[nm])) / rms(Po, Pt, mask(W4[nm])):.2f}"
                        for nm in ("calm", "SENSOR", "PROCESS", "POTFAIL", "BOTH"))
        print(f"   {lab}  lucid/oracle:  {out}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
