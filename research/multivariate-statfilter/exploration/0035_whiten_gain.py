"""Probe 0035 -- derive the process-walk gain (_Q_DRIVE) as Newton whitening at rate K*.

Sensitivity sweep (0034) found _Q_DRIVE = 0.2 is UNDER-tuned: 0.8 improves BOTH 2.38 -> 1.97 with
no regression, 1.6 breaks process+pot.  And 0.8 = K*/b with b ~ 0.03 -- i.e. the optimum is Newton
whitening at the SENSOR walk's own rate K*: the process mode steps mu += K* * sig / b, where
b = -d sig/d mu is the steady-state sensitivity of its lag-1 innovation autocorrelation to its
log-scale.  This unifies the two walks (both rate K*) and derives _Q_DRIVE.

b is a construction-time quantity: solve the steady-state (DARE) innovation statistics for the
scale, use the Mehra lag-1 autocovariance C1 = H F (I-KH) Pp H^T - H F K R, read the mode-direction
autocorr sig_k = (hv C1 hv)/(hv S hv), finite-difference in mu_k.  This probe checks the implied
gain K*/b_k lands near the empirical optimum ~0.8 for the jerk modes.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402

NJ, ORDER, DT = 5, 3, 0.01
POT, ACC, JERK = 0.06, 0.02, 0.6


def build():
    return AdaptiveKalmanFilter.kinematic(NJ, ORDER, DT, process_var=JERK ** 2,
                                          meas_var={"pos": POT ** 2, "acc": ACC ** 2},
                                          measured=("pos", "acc"), control=True, s=0.5)


def _gain(f, Q, R):
    """Steady-state Kalman gain the filter uses when it ASSUMES process cov Q, sensor cov R."""
    H, F, n = f.H, f.F, f.n
    P = np.eye(n) * (f.lam.max() + f.rho.max())
    for _ in range(600):
        Pp = F @ P @ F.T + Q
        K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R)
        P = Pp - K @ H @ Pp
    Pp = F @ P @ F.T + Q
    return Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R)


def sig_of(f, mu_k, k, Qtrue, R0):
    """Lag-1 innovation autocorr in mode k's direction when the filter ASSUMES scale mu_k on mode k
    but the TRUE process cov is Qtrue.  Gain from the assumed scale; actual a-priori error cov M
    from the true cov via the closed-loop Lyapunov M = A M A^T + F K R K^T F^T + Qtrue, A=F(I-KH)."""
    H, F, n = f.H, f.F, f.n
    mu = np.zeros(n); mu[k] = mu_k
    K = _gain(f, f._Q_of(mu), R0)                       # gain from the ASSUMED scale
    A = F @ (np.eye(n) - K @ H)
    W = F @ K @ R0 @ K.T @ F.T + Qtrue
    M = np.eye(n)
    for _ in range(2000):
        M = A @ M @ A.T + W
    S = H @ M @ H.T + R0
    C1 = H @ A @ M @ H.T - H @ F @ K @ R0               # Mehra lag-1, suboptimal gain
    C1 = 0.5 * (C1 + C1.T)
    hv = f.HV[:, k]
    return float(hv @ C1 @ hv) / (float(hv @ S @ hv) + 1e-12)


def main():
    f = build()
    Kstar = f._Kstar
    R0 = f._R_of(np.zeros(f.m))
    Qtrue = f._Q_of(np.zeros(f.n))                        # true process cov = nominal
    print(f"K* = {Kstar:.4f};  empirical _Q_DRIVE optimum ~ 0.8 (0034)\n")
    # a representative active mode (10) at two operating points: nominal true Q, and a disturbance
    # (true Q x 400 = jerk_mult^2), assumed scale ramping up from 0.  b is the LOCAL slope there.
    k = 10
    print("mode 10, b = -d sig/d(assumed mu), K*/b at several operating points:")
    print(f"  {'true Qx':>8} {'assumed mu0':>12} {'b_local':>9} {'K*/b':>8}")
    for qx in (1.0, 400.0):
        Qt = f._Q_of(np.zeros(f.n)) * qx
        for mu0 in (0.0, 2.0, 4.0):
            d = 0.5
            b = -(sig_of(f, mu0 + d, k, Qt, R0) - sig_of(f, mu0 - d, k, Qt, R0)) / (2 * d)
            print(f"  {qx:8.0f} {mu0:12.1f} {b:9.4f} {Kstar / max(abs(b), 1e-3):8.3f}")


if __name__ == "__main__":
    main()
