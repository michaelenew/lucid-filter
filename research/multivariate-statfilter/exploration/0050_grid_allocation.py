"""Probe 0050 -- resolve the process/measurement confound with NO EMA: the core.py grid allocation.

_BETA (the C0/C1 innovation EMA) was added to the multivariate filter on this branch and is the wrong
foundation: the scalar core (statfilter.core.AdaptiveFilter) already solved process-vs-measurement with
NO EMA. It carries the two noise scales as latent log-AR(1) STATES on a quadrature grid and runs an
exact forward recursion: ONE GPB1 state (m, P) plus a scale posterior pi over the grid. Each node has
its own (Q_g, R_g); per step pi is propagated through the AR(1) transition T (the memory -- persistence
phi, NOT a lag EMA), then reweighted by the per-node innovation likelihood. The confound breaks because
a process node inflates Pp = F P F' + Q_g and so predicts CONTINUED large innovations, while a sensor
node (small Q_g, large R_g) explains one spike without inflating the state -- and the different
persistences phi_P vs phi_M penalise a persistent "it was the sensor". No threshold, no beta; and the
reach is automatic (a genuine sensor burst just moves pi onto high-R nodes).

This ports that to a vector state: single joint, order 3 (pos, vel, acc), jerk process, two sensors
(pot on pos, accel on acc -- the accel is collinear with the jerk process = the hard confound). Grid
over the three scale channels (process xi, pot eta_p, accel eta_a). Compare to floor (single-scale KF)
and oracle across the 0034 regimes. If the grid resolves process/BOTH AND reaches pot-hot/SENSOR with
no EMA, it is the right foundation.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter.core import _chain  # noqa: E402  -- the AR(1) log-scale grid + transition (no EMA)

ORDER, DT = 3, 0.01
JERK, POT, ACC = 0.6, 0.06, 0.02
PHI, S = 0.9, 0.5
NGRID = 5
T = 1000
ON = slice(400, 700)

Fb = np.eye(ORDER)
for i in range(ORDER):
    for j in range(i + 1, ORDER):
        Fb[i, j] = DT ** (j - i) / math.factorial(j - i)
Gv = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])  # jerk input map
GG = np.outer(Gv, Gv)
H = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # pot reads pos, accel reads acc
REGIMES = ["calm", "pot-hot", "SENSOR", "PROCESS", "BOTH", "process+pot"]


def sim(seed, regime):
    rng = np.random.default_rng(seed); t = np.arange(T) * DT
    u = 2.0 * np.sin(2 * np.pi * 0.4 * t) + 1.2 * np.sin(2 * np.pi * 0.75 * t)   # commanded jerk
    jstd = np.full(T, JERK); pot = np.full(T, POT); acc = np.full(T, ACC)
    a0, b0 = ON.start, ON.stop
    if regime in ("pot-hot", "process+pot"):
        pot[a0:b0] *= 15.0
    if regime in ("PROCESS", "BOTH", "process+pot"):
        jstd[a0:b0] *= 20.0
    if regime in ("SENSOR", "BOTH"):
        acc[a0:b0] *= 15.0
    s = np.zeros(ORDER); St = np.zeros((T, ORDER)); Y = np.zeros((T, 2))
    for k in range(T):
        s = Fb @ s + Gv * u[k] + Gv * (jstd[k] * rng.standard_normal()); St[k] = s
        Y[k] = H @ s + np.array([pot[k], acc[k]]) * rng.standard_normal(2)
    return St, Y, u, jstd, pot, acc


def oracle(Y, u, jstd, pot, acc):
    m = np.zeros(ORDER); P = np.eye(ORDER); out = np.zeros((T, ORDER))
    for k in range(T):
        Q = jstd[k] ** 2 * GG; R = np.diag([pot[k] ** 2, acc[k] ** 2])
        mp = Fb @ m + Gv * u[k]; Pp = Fb @ P @ Fb.T + Q
        S = H @ Pp @ H.T + R; K = Pp @ H.T @ np.linalg.inv(S)
        m = mp + K @ (Y[k] - H @ mp); P = Pp - K @ H @ Pp; out[k] = m
    return out


def floor_kf(Y, u):
    """Single fixed-scale KF at the nominal noise (no adaptation) -- the baseline."""
    m = np.zeros(ORDER); P = np.eye(ORDER); out = np.zeros((T, ORDER))
    Q = JERK ** 2 * GG; R = np.diag([POT ** 2, ACC ** 2])
    for k in range(T):
        mp = Fb @ m + Gv * u[k]; Pp = Fb @ P @ Fb.T + Q
        S = H @ Pp @ H.T + R; K = Pp @ H.T @ np.linalg.inv(S)
        m = mp + K @ (Y[k] - H @ mp); P = Pp - K @ H @ Pp; out[k] = m
    return out


def _grids():
    lam, w, Tm = _chain(PHI, S, NGRID)                 # per-channel: nodes, stationary wts, transition
    xi, ep, ea = (np.meshgrid(lam, lam, lam, indexing="ij"))
    xi, ep, ea = xi.ravel(), ep.ravel(), ea.ravel()
    pi0 = np.einsum("i,j,k->ijk", w, w, w).ravel()
    Tjoint = np.einsum("ij,kl,mn->ikmjln", Tm, Tm, Tm).reshape(NGRID ** 3, NGRID ** 3)
    return xi, ep, ea, pi0, Tjoint


def grid_filter(Y, u):
    xi, ep, ea, pi, Tj = _grids(); G = pi.size
    Qg = (JERK ** 2 * np.exp(xi))[:, None, None] * GG[None]                  # (G,3,3)
    Rg = np.zeros((G, 2, 2)); Rg[:, 0, 0] = POT ** 2 * np.exp(ep); Rg[:, 1, 1] = ACC ** 2 * np.exp(ea)
    m = np.zeros(ORDER); P = np.eye(ORDER); out = np.zeros((T, ORDER))
    I = np.eye(ORDER)
    for k in range(T):
        pi = pi @ Tj                                                        # propagate scale posterior
        mp = Fb @ m + Gv * u[k]; Pp0 = Fb @ P @ Fb.T
        Ppg = Pp0[None] + Qg                                                # (G,3,3) per-node prediction
        HPpH = np.einsum("ab,gbc,dc->gad", H, Ppg, H)                       # (G,2,2)
        Sg = HPpH + Rg
        e = Y[k] - H @ mp                                                   # (2,) shared innovation
        Sig = np.linalg.inv(Sg)                                            # (G,2,2)
        sign, logdet = np.linalg.slogdet(Sg)
        quad = np.einsum("a,gab,b->g", e, Sig, e)
        ll = -0.5 * (logdet + quad)
        w = pi * np.exp(ll - ll.max()); pi = w / w.sum()
        Kg = np.einsum("gab,cb,gcd->gad", Ppg, H, Sig)                      # (G,3,2)
        mg = mp[None] + np.einsum("gab,b->ga", Kg, e)                       # (G,3)
        mbar = pi @ mg                                                      # GPB1 mean
        Ppost = Ppg - np.einsum("gab,bc,gcd->gad", Kg, H, Ppg)             # (G,3,3)
        dm = mg - mbar[None]
        P = np.einsum("g,gab->ab", pi, Ppost) + np.einsum("g,ga,gb->ab", pi, dm, dm)  # GPB1 cov
        m = mbar; out[k] = m
    return out


def rms(est, St):
    d = (est[ON, 0] - St[ON, 0])
    return math.sqrt(float((d * d).mean()))


def main(nseed=6):
    print(f"grid allocation (NO EMA), single joint, {nseed} seeds -- adaptive/oracle:")
    print(f"  {'regime':13s} {'floor/orc':>10} {'grid/orc':>10}")
    for regime in REGIMES:
        fl = []; gr = []
        for seed in range(nseed):
            St, Y, u, jstd, pot, acc = sim(seed, regime)
            oc = rms(oracle(Y, u, jstd, pot, acc), St)
            fl.append(rms(floor_kf(Y, u), St) / oc)
            gr.append(rms(grid_filter(Y, u), St) / oc)
        print(f"  {regime:13s} {np.mean(fl):10.3f} {np.mean(gr):10.3f}")


if __name__ == "__main__":
    main()
