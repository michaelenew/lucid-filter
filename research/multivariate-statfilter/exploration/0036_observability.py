"""Probe 0036 -- the STATIC process-coupling weight that retires the shed's _SHED / _WHITE_MIN.

The fast shed's danger is a process ONSET: the whiteness gate's rho1 EMA lags, so a fast shed
misfires on a process-coupled sensor before wg can drop.  The gentle _SHED and the hard _WHITE_MIN
floor are the empirical onset-safety balance.  The fix is a STATIC (no-lag) per-sensor weight that
says how safe a fast shed is -- fast-shed only channels that stay WHITE under a process disturbance
(they carry no process to misattribute); shed a process-coupled channel only at the slow rate.

First idea (observability criticality, tr P growth if the sensor is removed) FAILS: it is high for
any irreplaceable sensor, including a process-coupled one (a lone position sensor with process near
position).  The right quantity is the channel's PROCESS-COUPLING: the lag-1 innovation
autocorrelation it picks up when the process is under-modelled.  With the gain from the ASSUMED
(nominal) scale but a TRUE process cov Q0*excess, the actual a-priori error cov solves the
closed-loop Lyapunov M = A M A^T + F K R K^T F^T + Qtrue (A=F(I-KH)); the Mehra lag-1 innovation
autocov C1 = H A M H^T - H F K R; coupling_i = rho1 in channel i = C1_ii / S_ii.  Decoupling
weight = 1 - coupling_i in [0,1]: ~1 for an absolute reference (stays white), ~0 for a
process-coupled channel.  This probe checks it separates the two rigs correctly.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402


def coupling(f, excess):
    """Per sensor, the lag-1 innovation autocorr under true process cov Q0*excess (filter assumes
    Q0).  High = process-coupled channel."""
    H, F, n, m = f.H, f.F, f.n, f.m
    R0 = f._R_of(np.zeros(m)); Q0 = f._Q_of(np.zeros(n)); I = np.eye(n)
    K = f._gain_for(Q0, R0)                                 # gain from the assumed (nominal) scale
    A = F @ (I - K @ H); W = F @ K @ R0 @ K.T @ F.T + Q0 * excess
    M = I.copy()
    for _ in range(3000):
        M = A @ M @ A.T + W
    S = H @ M @ H.T + R0
    C1 = H @ A @ M @ H.T - H @ F @ K @ R0
    C1 = 0.5 * (C1 + C1.T)
    return np.array([C1[i, i] / (S[i, i] + 1e-12) for i in range(m)])


def rig_5dof():
    return AdaptiveKalmanFilter.kinematic(5, 3, 0.01, process_var=0.6 ** 2,
                                          meas_var={"pos": 0.06 ** 2, "acc": 0.02 ** 2},
                                          measured=("pos", "acc"), control=True, s=0.5)


def rig_2state():
    F = np.array([[1.0, 1.0], [0.0, 1.0]]); G = np.array([[0.5], [1.0]]); H = np.array([[1.0, 0.0]])
    return AdaptiveKalmanFilter(1e-3 * (G @ G.T) + 1e-9 * np.eye(2), R0=[0.04], H=H, F=F, s=0.5)


def main():
    for name, f, kinds in [("5-DOF (pot/accel x5)", rig_5dof(),
                            ["pot" if i % 2 == 0 else "accel" for i in range(10)]),
                           ("2-state single position sensor", rig_2state(), ["pos"])]:
        print(f"\n{name}:")
        for exc in (30.0, 100.0):
            c = coupling(f, exc)
            print(f"  true Q x{exc:5.0f}:  coupling rho1 = {np.round(c, 3)}")
            print(f"                 decoupling 1-rho1 = {np.round(np.clip(1 - c, 0, 1), 3)}")


if __name__ == "__main__":
    main()
