"""0054 -- sensors the arm could actually carry, and what the old diagonal H was worth.

`0052` gave every joint an "accelerometer" reading that joint's own angular acceleration
through a constant, diagonal ``H``.  No sensor does that.  An inertial sensor is bolted to a
*link*, and reads the motion of the whole chain beneath it in axes that rotate with the arm.
On this arm joints 2, 3 and 4 all rotate about their local y, so their axes are exactly
parallel and a sensor on link 4 reads ``al_2 + al_3 + al_4``; the joint 1 <-> joint 5 coupling
is not even constant, sweeping -0.15 to -0.53 across the trajectory.  Measured against 0052's
own sensor sigma, that model error is **7-13x** on joints 3, 4 and 5.

`../scripts/arm5dof.py` replaces it with the physical map -- coupled, state-dependent, and
with a rate-quadratic term, so ``h(x)`` is not ``H(x) x`` -- supplied to the shipped filter as
``LucidFilter(H=callable)``.  The servo also now flies on an alpha-beta-gamma tracker of the
potentiometers rather than on the true state, at a bandwidth the potentiometer can support.

Four questions, in order:

  1. Is the analytic Jacobian right?  (against central differences)
  2. On the physical rig, how does the filter do against an oracle told the true noise
     schedule, and against the same model frozen?
  3. What does the OLD diagonal ``H`` cost, run on this same physical data?  That is the
     price of the shortcut, and the reason the feature had to exist.
  4. Why the second sensor is an accelerometer and not a rate gyro -- a relative-degree
     result that is a genuine limit of the scale walk, isolated here.

Run: python 0054_physical_sensors.py        (~20 min)
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


def schedule(sig2=None):
    jstd = np.full(T, A.JERK)
    pot = np.full((T, A.NJ), A.POT)
    dyn = np.full((T, A.NJ), A.ACC if sig2 is None else sig2)
    for nm, a, b in PHASES:
        if nm == "SENSOR":
            dyn[a:b] *= ACC_MULT
        elif nm == "PROCESS":
            jstd[a:b] *= JERK_MULT
        elif nm == "POTFAIL":
            pot[a:b, FAIL_J] *= POT_MULT
        elif nm == "BOTH":
            dyn[a:b] *= ACC_MULT
            jstd[a:b] *= JERK_MULT
    return jstd, pot, dyn


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
    print(f"1. analytic Jacobian vs central differences, worst abs error: {worst:.2e}")
    Hc = A.H_CHAR
    print("   what each accelerometer actually reads (coefficients on alpha, at theta = 0):")
    for j in range(A.NJ):
        row = "  ".join(f"al{i + 1}={Hc[2 * j + 1, A.ORDER * i + 2]:+.2f}" for i in range(A.NJ))
        print(f"     link {j + 1}: {row}")


def main():
    t0 = time.time()
    jacobian_check()

    W = windows()
    acc = {k: {b: [] for b in ("lucid", "fixed", "diagH")} for k in W}
    burst = {b: [] for b in ("raw", "lucid", "fixed", "oracle", "diagH")}
    Hd = np.zeros((A.M, A.N))                # the retired model: diagonal, joint-local
    for j in range(A.NJ):
        Hd[2 * j, A.ORDER * j] = 1.0
        Hd[2 * j + 1, A.ORDER * j + 2] = 1.0

    for sd in range(NS):
        jstd, pot, dyn = schedule()
        U, S, Y = A.simulate(sd, jstd, pot, dyn)
        Qs = [j ** 2 * (A.B @ A.B.T) for j in jstd]
        Rs = [np.concatenate([[pot[k, j] ** 2, dyn[k, j] ** 2] for j in range(A.NJ)])
              for k in range(T)]
        est = {"lucid": A.make_filter().filter(Y, U).mean,
               "fixed": A.kalman(U, Y),
               "oracle": A.kalman(U, Y, Qs, Rs),
               "diagH": LucidFilter(dynamics=A.F, control=A.B, H=Hd, process=A.Q0,
                                    measurement=A.R0).filter(Y, U).mean}
        Pt = tip(S)
        P = {k: tip(v) for k, v in est.items()}
        P["raw"] = np.array([A.joints3d(Y[k, 0::2])[-1] for k in range(T)])
        on = np.zeros(T, bool)
        for nm, a, b in PHASES:
            if nm != "calm":
                on[a:b] = True
        for b in burst:
            burst[b].append(rms(P[b], Pt, on))
        for key, spans in W.items():
            sl = mask(spans)
            o = rms(P["oracle"], Pt, sl)
            for b in ("lucid", "fixed", "diagH"):
                acc[key][b].append(rms(P[b], Pt, sl) / o)
        print(f"   seed {sd} done ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n2/3. tip RMSE over the bursts ({NS} seeds), metres")
    for b in ("raw", "fixed", "oracle", "lucid", "diagH"):
        print(f"     {b:<8} {np.mean(burst[b]):.4f} +- {se(burst[b]):.4f}")
    print("\n     ratio to an oracle told the true noise schedule "
          "(`diag-H` is the RETIRED model, on this same physical data)")
    print(f"     {'regime':<10}{'lucid':>10}{'fixed':>10}{'diag-H':>12}")
    for key in ("calm", "SENSOR", "PROCESS", "POTFAIL", "BOTH"):
        row = [np.mean(acc[key][b]) for b in ("lucid", "fixed", "diagH")]
        print(f"     {key:<10}{row[0]:>10.2f}{row[1]:>10.2f}{row[2]:>12.2f}")

    print("\n4. why the dynamic sensor is an accelerometer, not a rate gyro (1 seed)")
    print("   A jerk disturbance reaches a rate sensor only through dt^2/2 and an angular-")
    print("   acceleration sensor through dt -- 200x more per step at this rate.  The scale")
    print("   walk scores per step, so on a rate sensor a process burst is nearly invisible")
    print("   in one step while the sensor axis is not, and the burst is blamed on the")
    print("   sensor.  Both rigs below are DIAGONAL, so only the read derivative differs.")
    for lab, di, sig in (("accelerometer (reads alpha)", 2, A.ACC),
                         ("rate gyro     (reads omega)", 1, 0.010)):
        jstd, pot, dyn = schedule(sig)
        rows = []
        for j in range(A.NJ):
            e = np.zeros(A.N); e[A.ORDER * j] = 1.0; rows.append(e)
            e = np.zeros(A.N); e[A.ORDER * j + di] = 1.0; rows.append(e)
        Hm = np.array(rows)
        R0 = np.tile([A.POT ** 2, sig ** 2], A.NJ)
        rng = np.random.default_rng(0)
        servo = A.Servo(); s = np.zeros(A.N)
        S = np.zeros((T, A.N)); Y = np.zeros((T, A.M)); U = np.zeros((T, A.NJ))
        rth, rom, ral, rjk = A.reference(T)
        sdv = np.empty(A.M)
        sdv[0::2], sdv[1::2] = pot[0], dyn[0]
        y = Hm @ s + sdv * rng.standard_normal(A.M)
        for k in range(T):
            servo.observe(y)
            U[k] = servo.command(rth[k], rom[k], ral[k], rjk[k])
            s = A.F @ s + A.B @ U[k] + A.B @ (jstd[k] * rng.standard_normal(A.NJ))
            sdv[0::2], sdv[1::2] = pot[k], dyn[k]
            y = Hm @ s + sdv * rng.standard_normal(A.M)
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
        out = "  ".join(f"{nm} {rms(Pl, Pt, mask(W[nm])) / rms(Po, Pt, mask(W[nm])):.2f}"
                        for nm in ("calm", "SENSOR", "PROCESS", "POTFAIL", "BOTH"))
        print(f"   {lab}  lucid/oracle:  {out}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
