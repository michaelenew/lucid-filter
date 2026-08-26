"""Probe 0025 -- break the Q-vs-R confound with the innovation time-correlation (Mehra).

0024 showed the single-step likelihood walk DIVERGES when process noise dominates: a large
innovation is explained equally by inflating Q or R at one step, so the walk picks the stiffer
(sensor) axis, down-weights the good sensor to zero, goes open-loop and runs away.

The classical resolution (Mehra 1970): the innovation SEQUENCE separates them.
  * Sensor noise R inflates the innovation VARIANCE C0 = E[e e^T] but leaves it WHITE.
  * Process noise Q makes the filter LAG -> positive lag-1 autocorrelation C1 = E[e_t e_{t-1}^T].
So: adapt R to match C0 (given the state part), adapt Q to WHITEN C1.  C1 responds to process
noise but not to white sensor noise -> the confound is broken.

This prototype (1-DOF kinematic arm, crusher regime) tests moment-matching adaptation:
  R  <- diag( C0_hat - H Ppred H^T )_+            (innovation-based R; the white excess)
  xi <- walk to drive the lag-1 correlation C1_hat toward 0 (whiten -> correct Q)
across static / r-dominant / q-dominant / both, vs non-adaptive and the oracle.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True)


def sim(seed, q_on, r_on, q_base=1e-3, r_base=0.04, T=600):
    dt = 1.0; on = slice(T // 3, 2 * T // 3); rng = np.random.default_rng(seed)
    F = np.array([[1.0, dt], [0.0, 1.0]]); G = np.array([[0.5], [1.0]]); H = np.array([[1.0, 0.0]])
    qt = np.full(T, q_base); rt = np.full(T, r_base); qt[on] = q_on; rt[on] = r_on
    x = np.zeros(2); X = np.zeros((T, 2)); Y = np.zeros((T, 1))
    for t in range(T):
        a = np.sqrt(qt[t]) * rng.standard_normal(); x = F @ x + (G * a).ravel(); X[t] = x
        Y[t] = H @ x + np.sqrt(rt[t]) * rng.standard_normal()
    return X, Y, F, G, H, qt, rt, q_base, r_base, on


def kf_fixed(Y, F, H, Q, R, m0, P0):
    m = m0.copy(); P = P0.copy(); out = np.zeros((len(Y), F.shape[0]))
    for t, y in enumerate(Y):
        m = F @ m; P = F @ P @ F.T + Q[t]
        S = H @ P @ H.T + R[t]; K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ (y - H @ m); P = P - K @ H @ P; out[t] = m
    return out


def kf_adaptive(Y, F, G, H, q0, r0, m0, P0, beta=0.02):
    """Moment-matching adaptive KF: R from C0 (white excess), Q from whitening C1."""
    m = m0.copy(); P = P0.copy(); out = np.zeros((len(Y), F.shape[0]))
    logq = 0.0; logr = 0.0                                   # log-scale of Q, R around (q0, r0)
    C0 = np.array([[r0]]); C1 = np.array([[0.0]]); e_prev = None
    GGt = G @ G.T
    scales = np.zeros((len(Y), 2))
    for t, y in enumerate(Y):
        Q = q0 * math.exp(logq) * GGt if False else q0 * np.exp(logq) * GGt
        R = np.array([[r0 * np.exp(logr)]])
        m = F @ m; Ppred = F @ P @ F.T + Q
        e = (y - H @ m).reshape(-1)
        S = H @ Ppred @ H.T + R; K = Ppred @ H.T @ np.linalg.inv(S)
        m = m + K @ e; P = Ppred - K @ H @ Ppred
        out[t] = m; scales[t] = [logq, logr]
        # --- innovation statistics (EMA) ---
        C0 = (1 - beta) * C0 + beta * np.outer(e, e)
        if e_prev is not None:
            C1 = (1 - beta) * C1 + beta * np.outer(e, e_prev)
        e_prev = e.copy()
        # --- adapt R: match the WHITE innovation excess C0 - H Ppred H^T ---
        HPHt = float((H @ Ppred @ H.T).ravel()[0])
        r_target = max(float(C0.ravel()[0]) - HPHt, 1e-6)
        logr += 0.05 * (math.log(r_target / r0) - logr)
        # --- adapt Q: whiten C1 (positive lag-1 corr -> Q too small -> raise) ---
        rho1 = float(C1.ravel()[0]) / (float(C0.ravel()[0]) + 1e-12)              # normalised lag-1 autocorrelation
        logq += 0.5 * rho1                                  # drive rho1 -> 0
        logq = float(np.clip(logq, -5, 12)); logr = float(np.clip(logr, -5, 12))
    return out, scales


import math  # noqa: E402


def run():
    T = 600
    print("position RMSE (crusher-ON window), mean over 3 seeds:")
    for q_on, r_on, lbl in [(1e-3, 0.04, "static"), (1e-3, 4.0, "r-dom "),
                            (0.30, 0.04, "q-dom "), (0.30, 4.0, "both  ")]:
        na = []; ad = []; orc = []
        for s in range(3):
            X, Y, F, G, H, qt, rt, qb, rb, on = sim(s, q_on, r_on)
            Qb = qb * (G @ G.T); Rb = np.array([[rb]])
            na.append(kf_fixed(Y, F, H, np.array([Qb] * T), np.array([Rb] * T),
                               np.array([Y[0, 0], 0.]), np.eye(2)))
            orc.append(kf_fixed(Y, F, H, np.array([qt[t] * (G @ G.T) for t in range(T)]),
                                np.array([[[rt[t]]] for t in range(T)]),
                                np.array([Y[0, 0], 0.]), np.eye(2)))
            est, _ = kf_adaptive(Y, F, G, H, qb, rb, np.array([Y[0, 0], 0.]), np.eye(2))
            ad.append(est)
        rm = lambda lst: np.mean([np.sqrt(((e[on, 0] - sim(i, q_on, r_on)[0][on, 0]) ** 2).mean())
                                  for i, e in enumerate(lst)])
        print(f"  {lbl}: non-adapt {rm(na):.3f}  ADAPTIVE {rm(ad):.3f}  oracle {rm(orc):.3f}   "
              f"(gain {rm(na)/rm(ad):.2f}x)")


if __name__ == "__main__":
    run()
