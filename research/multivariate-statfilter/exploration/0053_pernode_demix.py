"""Probe 0053 -- what does the collinear accel<->jerk de-mix actually require?

0052 leaves one residual: the attribution on the collinear pair (the accelerometer reads the
very state the jerk drives; scale-Fisher correlation |C| = 1, research 0027).  During an accel
burst the process scale rides up too, and vice versa.  Candidate mechanisms, cheapest first:

  A. shared mean + shared P     (the caltrop engine pre-0053: per-step likelihood only)
  B. shared mean + PER-NODE P   (0050/0051's "per-node P propagation" -- each hypothesis
                                 accumulates the S it implies; measured in-engine: does NOT
                                 move the attribution, tested standalone here)
  C. full PER-NODE KFs          (per-node means: each hypothesis runs its own filter, so a
                                 high-Q node CHASES white sensor noise and mispredicts its own
                                 innovation statistics -- the lag-1 evidence enters through the
                                 mean, with no EMA and no whiteness statistic)

Rig: ONE joint of the 0052 arm (n = 3 kinematic state, m = 2: pot + accel), commanded forcing.
A static 2-D scale grid over (xi = jerk log-scale, eta = accel log-scale), pot scale fixed --
25 nodes, exact Bayes weights with a mild forgetting (a probe, not the production walk).
Regimes: SENSOR (accel x15 -> truth (0, 5.4)) and PROCESS (jerk x20 -> truth (6.0, 0)).
Read the posterior-mean (xi, eta) during the burst for each mechanism.
"""
import math
import sys

import numpy as np

NJ_ORDER, DT = 3, 0.01
POT, ACC, JERK = 0.06, 0.02, 0.6
T = 1000
ON = slice(400, 700)
FORGET = 0.99

Fb = np.eye(3)
for i in range(3):
    for j in range(i + 1, 3):
        Fb[i, j] = DT ** (j - i) / math.factorial(j - i)
G = np.array([DT ** (3 - i) / math.factorial(3 - i) for i in range(3)])
H = np.array([[1.0, 0, 0], [0, 0, 1.0]])
N, M = 3, 2

GRID = np.array([0.0, 1.5, 3.0, 4.5, 6.0])
NODES = [(xi, eta) for xi in GRID for eta in GRID]
NG = len(NODES)


def sim(seed, regime):
    rng = np.random.default_rng(seed); t = np.arange(T) * DT
    U = 2.0 * np.sin(2 * np.pi * 0.35 * t) + 1.2 * np.sin(2 * np.pi * 0.7 * t + 1)
    jstd = np.full(T, JERK); acc = np.full(T, ACC)
    if regime == "sens":
        acc[ON] = ACC * 15.0
    else:
        jstd[ON] = JERK * 20.0
    s = np.zeros(N); S = np.zeros((T, N)); Y = np.zeros((T, M))
    for k in range(T):
        s = Fb @ s + G * U[k] + G * (jstd[k] * rng.standard_normal()); S[k] = s
        Y[k] = H @ s + np.array([POT, acc[k]]) * rng.standard_normal(M)
    return U, S, Y


def QR(xi, eta):
    Q = JERK ** 2 * math.exp(xi) * np.outer(G, G) + 1e-12 * np.eye(N)
    R = np.diag([POT ** 2, ACC ** 2 * math.exp(eta)])
    return Q, R


def run(U, Y, mech):
    """mech: 'shared' (one m, one P), 'perP' (one m, per-node P), 'full' (per-node m and P)."""
    logw = np.zeros(NG)
    m1 = np.zeros(N); P1 = np.eye(N)
    mg = np.zeros((NG, N)); Pg = np.stack([np.eye(N)] * NG)
    QRs = [QR(*nd) for nd in NODES]
    post = np.zeros((T, 2))
    for k, y in enumerate(Y):
        lg = np.zeros(NG)
        if mech == "full":
            for g, (Q, R) in enumerate(QRs):
                mp = Fb @ mg[g] + G * U[k]; Pp = Fb @ Pg[g] @ Fb.T + Q
                Sm = H @ Pp @ H.T + R; Si = np.linalg.inv(Sm)
                e = y - H @ mp
                lg[g] = -0.5 * (np.linalg.slogdet(Sm)[1] + e @ Si @ e)
                K = Pp @ H.T @ Si
                mg[g] = mp + K @ e; Pg[g] = Pp - K @ H @ Pp
        else:
            mp = Fb @ m1 + G * U[k]; e = y - H @ mp
            wsum = np.exp(logw - logw.max()); wsum /= wsum.sum()
            mnew = np.zeros(N); Pnew = np.zeros((N, N))
            for g, (Q, R) in enumerate(QRs):
                Pbase = Pg[g] if mech == "perP" else P1
                Pp = Fb @ Pbase @ Fb.T + Q
                Sm = H @ Pp @ H.T + R; Si = np.linalg.inv(Sm)
                lg[g] = -0.5 * (np.linalg.slogdet(Sm)[1] + e @ Si @ e)
                K = Pp @ H.T @ Si
                if mech == "perP":
                    Pg[g] = Pp - K @ H @ Pp
        logw = FORGET * logw + lg
        w = np.exp(logw - logw.max()); w /= w.sum()
        if mech != "full":
            # GPB1 state at the posterior-weighted gain (shared mean/P collapse)
            mp = Fb @ m1 + G * U[k]; e = y - H @ mp
            Kbar = np.zeros((N, M)); Pbar = np.zeros((N, N))
            for g, (Q, R) in enumerate(QRs):
                Pbase = Pg[g] if mech == "perP" else P1
                Pp = Fb @ Pbase @ Fb.T + Q
                Sm = H @ Pp @ H.T + R; Si = np.linalg.inv(Sm)
                K = Pp @ H.T @ Si
                Kbar += w[g] * K; Pbar += w[g] * (Pp - K @ H @ Pp)
            m1 = mp + Kbar @ e; P1 = Pbar
        post[k] = np.array(NODES).T @ w
    return post


if __name__ == "__main__":
    for regime, truth in [("sens", (0.0, 5.4)), ("proc", (6.0, 0.0))]:
        U, S, Y = sim(0, regime)
        print(f"{regime.upper()}  truth (xi, eta) = {truth}")
        for mech in ("shared", "perP", "full"):
            po = run(U, Y, mech)[ON].mean(0)
            print(f"  {mech:6s}: posterior (xi, eta) = ({po[0]:+.2f}, {po[1]:+.2f})")
