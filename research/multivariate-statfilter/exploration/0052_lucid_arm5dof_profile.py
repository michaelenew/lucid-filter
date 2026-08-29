"""Probe 0052 -- profile the public LucidFilter on the 5-DOF arm rig (pot + accel fusion).

The README demo target is a 3D 5-DOF arm animation cycling through the noise regimes; before
building it, this profiler establishes that the PUBLIC filter (LucidFilter, the caltrop-star
engine) handles the case: every joint fuses a bad potentiometer (angle) with a good
accelerometer (angular acceleration), the arm is driven along a commanded trajectory (known
forcing B.u), and noise arrives in phases.  Same rig constants as 0026/0034 so results are
comparable to the AdaptiveKalmanFilter-era numbers.

Per (regime, seed) it runs three filters on the same data:
  - lucid   : LucidFilter(dynamics=F, control=B, H=H, process=Q0, measurement=R0) -- the
              public parameter-free filter, all noise inferred online;
  - oracle  : KF told the true time-varying (Q, R) schedule -- the lower bound;
  - fixed   : KF frozen at the base (Q0, R0) -- the non-adaptive floor the demo must beat.

Reported per regime: RMSE ratios lucid/oracle and fixed/oracle (mean +/- SE over seeds) on
joint angles in the burst window, plus the DIAGNOSTIC check the animation will visualise:
the learned per-sensor / per-eigenmode log-scales, hot channels vs cold, during the burst.

Usage:
    python 0052_lucid_arm5dof_profile.py save <label> [n_seeds]   # run + dump 0052_<label>.npz
    python 0052_lucid_arm5dof_profile.py compare A B              # paired diff of two runs
"""
import os
import sys
import math
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from lucid import LucidFilter  # noqa: E402

HERE = os.path.dirname(__file__)
NJ, ORDER, DT = 5, 3, 0.01
POT, ACC, JERK = 0.06, 0.02, 0.6
POT_MULT, ACC_MULT, JERK_MULT = 15.0, 15.0, 20.0
T = 1000
ON = slice(400, 700)
N_SEEDS_DEFAULT = 4
REGIMES = [("calm", "CALM"), ("sens", "SENSOR"), ("pot", "pot-hot"),
           ("proc", "PROCESS"), ("both", "BOTH"), ("procpot", "process+pot")]

# ---- the kinematic model (identical to AdaptiveKalmanFilter.kinematic, order 3, pos+acc) ----
Fb = np.eye(ORDER)
for i in range(ORDER):
    for j in range(i + 1, ORDER):
        Fb[i, j] = DT ** (j - i) / math.factorial(j - i)
G = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
F = np.kron(np.eye(NJ), Fb)
Q0 = np.kron(np.eye(NJ), JERK ** 2 * np.outer(G, G) + 1e-12 * np.eye(ORDER))
B = np.kron(np.eye(NJ), G[:, None])
GJ = np.kron(np.eye(NJ), G[:, None])
N = ORDER * NJ
_SPECS = {"potacc": (("pot", 0, POT), ("acc", 2, ACC)),   # per joint: which sensors exist
          "acc": (("acc", 2, ACC),),
          "pot": (("pot", 0, POT),)}
LAYOUT = "potacc"
H = R0 = KIND = None; M = 0


def set_layout(layout):
    """Sensor layout: 'potacc' (default), 'acc' (accelerometers only), 'pot' (pots only)."""
    global LAYOUT, H, R0, KIND, M
    rows, rvar, kind = [], [], []
    for d in range(NJ):
        for name, di, sd in _SPECS[layout]:
            e = np.zeros(N); e[d * ORDER + di] = 1.0
            rows.append(e); rvar.append(sd ** 2); kind.append(name)
    LAYOUT, H, R0 = layout, np.array(rows), np.array(rvar)
    KIND = np.array(kind); M = len(rows)


set_layout("potacc")


def build():
    return LucidFilter(dynamics=F, control=B, H=H, process=Q0, measurement=R0)


def sim(seed, regime):
    """Commanded 2-sinusoid jerk trajectory per joint + phased noise bursts."""
    rng = np.random.default_rng(seed); t = np.arange(T) * DT
    U = np.zeros((T, NJ))
    for j in range(NJ):
        for (a, w, ph) in [(2.0, 0.35 + 0.1 * j, j), (1.2, 0.7 + 0.13 * j, 2 * j)]:
            U[:, j] += a * np.sin(2 * np.pi * w * t + ph)
    jstd = np.full(T, JERK); pot = np.full(T, POT); acc = np.full(T, ACC)
    a0, b0 = ON.start, ON.stop
    if regime in ("pot", "procpot"):
        pot[a0:b0] = POT * POT_MULT
    if regime in ("proc", "procpot", "both"):
        jstd[a0:b0] = JERK * JERK_MULT
    if regime in ("sens", "both"):
        acc[a0:b0] = ACC * ACC_MULT
    s = np.zeros(N); S = np.zeros((T, N)); Y = np.zeros((T, M))
    for k in range(T):
        s = F @ s + B @ U[k] + GJ @ (jstd[k] * rng.standard_normal(NJ)); S[k] = s
        sd = np.where(KIND == "pot", pot[k], acc[k])
        Y[k] = H @ s + sd * rng.standard_normal(M)
    return U, S, Y, jstd, pot, acc


def kf(U, Y, Qs, Rs):
    """Plain KF with per-step (Q, R) supplied: the oracle (true schedule) or fixed (base)."""
    m0 = np.zeros(N); P = np.eye(N); out = np.zeros((T, N))
    for k, y in enumerate(Y):
        mp = F @ m0 + B @ U[k]; Pp = F @ P @ F.T + Qs[k]
        K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + Rs[k])
        m0 = mp + K @ (y - H @ mp); P = Pp - K @ H @ Pp
        out[k] = m0
    return out


def schedules(jstd, pot, acc):
    Qs = [j ** 2 * (GJ @ GJ.T) for j in jstd]
    Rs = []
    for k in range(T):
        sd = np.where(KIND == "pot", pot[k], acc[k])
        Rs.append(np.diag(sd ** 2))
    return Qs, Rs


def rms(est, S, d=0):
    """RMSE on derivative ``d`` (0 = angle, 1 = velocity) in the burst window."""
    tt = S.reshape(T, NJ, ORDER)[ON, :, d]; ee = est.reshape(T, NJ, ORDER)[ON, :, d]
    return float(np.sqrt(((ee - tt) ** 2).mean()))


def diagnostics(res, regime):
    """Mean learned log-scale, burst window vs calm prefix, on the channels the regime touches."""
    ms, ps = res.measurement_scale, res.process_scale
    calm_w, on_w = slice(100, ON.start), slice(ON.start + 50, ON.stop)
    nanpair = (float("nan"), float("nan"))
    pots, accs = ms[:, KIND == "pot"], ms[:, KIND == "acc"]
    # top-NJ Q eigenmodes (eigh sorts ascending; the NJ jerk directions are the last NJ)
    proc = ps[:, -NJ:]
    return {
        "pot":  nanpair if pots.shape[1] == 0 else
                (float(pots[calm_w].mean()), float(pots[on_w].mean())),
        "acc":  nanpair if accs.shape[1] == 0 else
                (float(accs[calm_w].mean()), float(accs[on_w].mean())),
        "proc": (float(proc[calm_w].mean()), float(proc[on_w].mean())),
    }


def regimes():
    """Regimes that exist under the current layout (pot regimes need pots)."""
    if "pot" not in KIND:
        return [(r, t) for r, t in REGIMES if r not in ("pot", "procpot")]
    return REGIMES


def save(label, n_seeds):
    data = {}
    f = build()
    eng = f._members[0]
    print(f"LucidFilter on the 5-DOF rig [{LAYOUT}]: n={N}, m={M}, D={N + M}; "
          f"bank nodes per member = {eng._G} (linear in D)")
    for regime, tag in regimes():
        lu = np.zeros(n_seeds); oc = np.zeros(n_seeds); fx = np.zeros(n_seeds)
        luv = np.zeros(n_seeds); ocv = np.zeros(n_seeds)
        dg = {"pot": [], "acc": [], "proc": []}
        for seed in range(n_seeds):
            U, S, Y, jstd, pot, acc = sim(seed, regime)
            t0 = time.time()
            res = build().filter(Y, U=U)
            el = time.time() - t0
            lu[seed] = rms(res.mean, S); luv[seed] = rms(res.mean, S, d=1)
            Qs, Rs = schedules(jstd, pot, acc)
            ockf = kf(U, Y, Qs, Rs)
            oc[seed] = rms(ockf, S); ocv[seed] = rms(ockf, S, d=1)
            Q0s, R0s = schedules(np.full(T, JERK), np.full(T, POT), np.full(T, ACC))
            fx[seed] = rms(kf(U, Y, Q0s, R0s), S)
            for kk, v in diagnostics(res, regime).items():
                dg[kk].append(v)
            if seed == 0:
                print(f"  [{tag}: {el:.1f}s/{T} steps = {1e3 * el / T:.1f} ms/step]")
        data[f"{regime}_lucid"] = lu; data[f"{regime}_oracle"] = oc; data[f"{regime}_fixed"] = fx
        data[f"{regime}_lucid_vel"] = luv; data[f"{regime}_oracle_vel"] = ocv
        rl, rf, rv = lu / oc, fx / oc, luv / ocv
        d = {kk: np.array(v).mean(0) for kk, v in dg.items()}
        print(f"  {tag:12s} lucid/oracle {rl.mean():.3f} +/- {rl.std(ddof=1)/math.sqrt(n_seeds):.3f}"
              f"   fixed/oracle {rf.mean():.3f}"
              f"   vel {rv.mean():.3f}"
              f"   angle-RMSE oracle {oc.mean():.4f}"
              f"   scales calm->burst: pot {d['pot'][0]:+.2f}->{d['pot'][1]:+.2f}"
              f"  acc {d['acc'][0]:+.2f}->{d['acc'][1]:+.2f}"
              f"  proc {d['proc'][0]:+.2f}->{d['proc'][1]:+.2f}")
    np.savez(os.path.join(HERE, f"0052_{label}.npz"), **data)
    print(f"saved 0052_{label}.npz  ({n_seeds} seeds, layout {LAYOUT})")


def compare(a, b):
    """Paired same-seed diff of two runs, on BOTH angle and velocity RMSE ratios.

    Velocity is part of the guard, not a footnote: 0053's regression showed up there first
    (vel/oracle 3.0 in PROCESS, 5.4 in BOTH) while some angle ratios still looked benign.
    """
    da = np.load(os.path.join(HERE, f"0052_{a}.npz")); db = np.load(os.path.join(HERE, f"0052_{b}.npz"))
    print(f"paired diff (lucid/oracle): {b} - {a}   [same seeds]   + = worse")
    worst = None
    for regime, tag in REGIMES:
        if f"{regime}_lucid" not in da or f"{regime}_lucid" not in db:
            continue
        line = f"  {tag:12s}"
        for what, suf in (("angle", ""), ("vel", "_vel")):
            ka, ko = f"{regime}_lucid{suf}", f"{regime}_oracle{suf}"
            if ka not in da or ka not in db:
                continue
            ra, rb = da[ka] / da[ko], db[ka] / db[ko]
            d = rb - ra; se = d.std(ddof=1) / math.sqrt(len(d))
            sig = d.mean() / se if se > 0 else 0.0
            line += (f"  {what} {ra.mean():7.3f} -> {rb.mean():7.3f} "
                     f"diff {d.mean():+7.3f} SE {se:.3f} ({sig:+.1f}s)")
            if worst is None or d.mean() - 2 * se > worst[1]:
                worst = (f"{tag}/{what}", d.mean() - 2 * se, d.mean(), se)
        print(line)
    if worst is not None:
        verdict = "PASS" if worst[1] <= 0 else "FAIL"
        print(f"  guard (every regime within +2 SE, velocity included): {verdict}"
              f"   worst {worst[0]}: diff {worst[2]:+.3f} SE {worst[3]:.3f} "
              f"-> diff-2SE {worst[1]:+.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "save":
        if len(sys.argv) > 4:
            set_layout(sys.argv[4])
        save(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else N_SEEDS_DEFAULT)
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
