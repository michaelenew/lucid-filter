"""Probe 0026 -- the hard realistic case: a 5-DOF arm in 3D, IMU-style fusion, phased noise.

Setup (per joint, order-3 kinematic state theta, omega, alpha):
  * a REALLY BAD potentiometer reads the joint angle (theta), std ~0.06 rad (~3.4 deg);
  * a GOOD accelerometer reads the joint angular acceleration (alpha) -- the main dynamic
    feedback -- std ~0.02 rad/s^2 (a link-mounted IMU, linearised to the joint);
  * the commanded jerk (top derivative of the trajectory) is the KNOWN forcing B u.
The arm is driven along a smooth multi-joint trajectory.  Noise arrives in phases:
    calm | SENSOR (accelerometers go noisy) | calm | PROCESS (disturbance torque / jerk) |
    calm | BOTH | calm
so the filter must (a) fuse a bad absolute sensor with a good dynamic one, (b) detect which
regime it is in and adapt, (c) never diverge.  The model is linear-Gaussian and matched, so the
test is the ONLINE NOISE IDENTIFICATION vs an oracle told the exact schedule -- not model
mismatch.  Metrics per phase: joint-angle / velocity / Cartesian-tip RMSE for raw pot,
non-adaptive, adaptive, oracle; plus the learned scales (do they light up in the right phase?).
"""
import os
import sys

import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
NJ, ORDER, DT = 5, 3, 0.01
POT_STD, ACC_STD, JERK_STD = 0.06, 0.02, 0.6           # base: bad pot, good accel, small jerk
T = 1400
PHASES = [("calm", 0, 200), ("SENSOR", 200, 400), ("calm", 400, 600),
          ("PROCESS", 600, 800), ("calm", 800, 1000), ("BOTH", 1000, 1200), ("calm", 1200, 1400)]


def build():
    return AdaptiveKalmanFilter.kinematic(
        n_dof=NJ, order=ORDER, dt=DT, process_var=JERK_STD ** 2,
        meas_var={"pos": POT_STD ** 2, "acc": ACC_STD ** 2},
        measured=("pos", "acc"), control=True, s=0.5)


def schedule():
    """Per-step true jerk-noise std and per-sensor measurement std (phase-dependent)."""
    jstd = np.full(T, JERK_STD); pot = np.full(T, POT_STD); acc = np.full(T, ACC_STD)
    for name, a, b in PHASES:
        if name == "SENSOR":
            acc[a:b] = ACC_STD * 15                     # accelerometers swamped (EM interference)
        elif name == "PROCESS":
            jstd[a:b] = JERK_STD * 20                   # disturbance torque / vibration
        elif name == "BOTH":
            acc[a:b] = ACC_STD * 15; jstd[a:b] = JERK_STD * 20
    return jstd, pot, acc


def simulate(seed=0):
    f = build(); F, B, H = f.F, f.B, f.H; n, m = f.n, f.m
    rng = np.random.default_rng(seed)
    t = np.arange(T) * DT
    # commanded jerk per joint: smooth band-limited multi-sine (a lifelike wander)
    U = np.zeros((T, NJ))
    for j in range(NJ):
        for (a, w, p) in [(2.0, 0.35 + 0.1 * j, j), (1.2, 0.7 + 0.13 * j, 2 * j)]:
            U[:, j] += a * np.sin(2 * np.pi * w * t + p)
    jstd, pot_s, acc_s = schedule()
    g = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
    Bn = np.kron(np.eye(NJ), g[:, None])                # jerk-noise input (== B)
    s = np.zeros(n); S = np.zeros((T, n)); Y = np.zeros((T, m))
    for k in range(T):
        s = F @ s + B @ U[k] + Bn @ (jstd[k] * rng.standard_normal(NJ))
        S[k] = s
        y = H @ s
        # per-sensor noise: rows are [dof0-pos, dof0-acc, dof1-pos, dof1-acc, ...]
        std = np.empty(m)
        std[0::2] = pot_s[k]; std[1::2] = acc_s[k]
        Y[k] = y + std * rng.standard_normal(m)
    return f, F, B, H, U, S, Y, jstd, pot_s, acc_s


def oracle(F, B, H, U, Y, jstd, pot_s, acc_s, g, n, m):
    m0 = np.zeros(n); P = np.eye(n) * 1.0; out = np.zeros((len(Y), n))
    Bn = np.kron(np.eye(NJ), g[:, None])
    for k, y in enumerate(Y):
        Q = (jstd[k] ** 2) * (Bn @ Bn.T)
        R = np.zeros((m, m)); R[np.arange(0, m, 2), np.arange(0, m, 2)] = pot_s[k] ** 2
        R[np.arange(1, m, 2), np.arange(1, m, 2)] = acc_s[k] ** 2
        mp = F @ m0 + B @ U[k]; Pp = F @ P @ F.T + Q
        Sm = H @ Pp @ H.T + R; K = Pp @ H.T @ np.linalg.inv(Sm)
        m0 = mp + K @ (y - H @ mp); P = Pp - K @ H @ Pp; out[k] = m0
    return out


def fk(theta):
    """5-DOF forward kinematics -> tip position (3D).  theta: (..., 5)."""
    axes = ["z", "y", "y", "y", "x"]; L = [0.30, 0.50, 0.40, 0.25, 0.15]
    def rot(ax, a):
        c, s = np.cos(a), np.sin(a)
        if ax == "z": return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        if ax == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    flat = theta.reshape(-1, NJ); tips = np.zeros((flat.shape[0], 3))
    for r in range(flat.shape[0]):
        Rm = np.eye(3); pos = np.zeros(3)
        for j in range(NJ):
            Rm = Rm @ rot(axes[j], flat[r, j]); pos = pos + Rm @ np.array([0, 0, L[j]])
        tips[r] = pos
    return tips.reshape(theta.shape[:-1] + (3,))


def main():
    f, F, B, H, U, S, Y, jstd, pot_s, acc_s = simulate(0)
    n, m = f.n, f.m
    g = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
    # --- filters ---
    adaptive = f.filter(Y, U=U)
    fna = build(); fna.active = np.array([], dtype=int); fna.r = 0
    nonad = fna.filter(Y, U=U)
    orc = oracle(F, B, H, U, Y, jstd, pot_s, acc_s, g, n, m)

    th_true = S.reshape(T, NJ, ORDER)[:, :, 0]
    th_ad = adaptive.mean.reshape(T, NJ, ORDER)[:, :, 0]
    th_na = nonad.mean.reshape(T, NJ, ORDER)[:, :, 0]
    th_or = orc.reshape(T, NJ, ORDER)[:, :, 0]
    pot = Y[:, 0::2]                                     # raw potentiometer angle
    tip_true = fk(th_true)
    print("active axes r =", f.r, " (n=%d states, m=%d sensors)  finite: %s"
          % (n, m, np.all(np.isfinite(adaptive.mean))))
    print("\njoint-angle RMSE (rad) by phase  [raw-pot / non-adaptive / ADAPTIVE / oracle]:")
    for name, a, b in PHASES:
        rms = lambda X: np.sqrt(((X[a:b] - th_true[a:b]) ** 2).mean())
        tip = lambda X: np.sqrt(((fk(X[a:b]) - tip_true[a:b]) ** 2).sum(-1).mean())
        print(f"  {name:8s}[{a:4d}:{b:4d}]  pot {rms(pot):.4f} | na {rms(th_na):.4f} | "
              f"AD {rms(th_ad):.4f} | or {rms(th_or):.4f}   tip(m) AD {tip(th_ad):.4f} na {tip(th_na):.4f}")
    # learned scales: do they light up in the right phase?
    ms = adaptive.measurement_scale                     # (T, m): pot cols 0::2, acc cols 1::2
    ps = adaptive.process_scale                         # (T, n)
    acc_scale = ms[:, 1::2].mean(1); pot_scale = ms[:, 0::2].mean(1)
    proc_scale = ps.reshape(T, NJ, ORDER)[:, :, 2].mean(1)   # top-derivative (jerk) scale
    print("\nlearned log-scales by phase  [accel-sensor / pot-sensor / process-jerk]:")
    for name, a, b in PHASES:
        print(f"  {name:8s}  accel {acc_scale[a:b].mean():+.2f}  pot {pot_scale[a:b].mean():+.2f}  "
              f"process {proc_scale[a:b].mean():+.2f}")


if __name__ == "__main__":
    main()
