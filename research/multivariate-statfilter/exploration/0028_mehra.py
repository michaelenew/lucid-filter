"""Probe 0028 -- the Mehra solve: recover Q and R from the innovation moments, recursively.

The gated-drive heuristics (0025-0027) have drifted from an optimal tracking procedure; even the
collinear-confound gap to oracle (1.24x) is slack we should reclaim.  Mehra's principle: the
innovation covariance C0 and its lag structure C1 together identify Q and R.  Recursively (with a
forgetting factor for the time-varying regime):

    R_hat = C0 - H Pp H^T                 (innovation minus the predicted state part)
    Q_hat = K C0 K^T                      (innovations mapped through the gain -> state increment
                                           covariance; Mohamed-Schwarz 1999, the recursive Mehra)

projected onto the per-component structure (diagonal R, eigenmode Q) so we keep the diagnostic and
stay well-conditioned.  The fixed point is self-consistent: get Q right (so H Pp H^T carries the
process contribution to each channel) and R_hat is then automatically right -- no whiteness gate,
no drives.  This is what should separate the collinear process/accel modes: the gain K encodes the
temporal structure, so K C0 K^T attributes the correlated part to Q.

Compared here against the shipped heuristic filter and the oracle, on the 5-DOF phased rig (0026)
and the confound rig (0027).
"""
import os
import sys
import math
import importlib.util

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402

_p26 = importlib.util.spec_from_file_location("p26", os.path.join(os.path.dirname(__file__), "0026_arm5dof.py"))
p26 = importlib.util.module_from_spec(_p26); _p26.loader.exec_module(p26)
NJ, ORDER, DT, T, PHASES = p26.NJ, p26.ORDER, p26.DT, p26.T, p26.PHASES

np.set_printoptions(precision=4, suppress=True)
_BETA = 0.02


def mehra_filter(f, Y, U, beta=_BETA, kappa=0.15):
    """Recursive innovation-based (Mehra/Mohamed-Schwarz) Q,R estimate, per-component projected."""
    F, B, H, V, lam, rho = f.F, f.B, f.H, f.V, f.lam, f.rho
    n, m = f.n, f.m
    HV = H @ V
    floorR = 1e-8
    # only the structurally-observable process eigenmodes carry real noise (the rank-1 jerk Q0
    # leaves most eigenmodes null); freeze the rest at baseline so Q_hat can't leak onto them
    proc_active = np.where((lam > 1e-6 * lam.max()) & (np.linalg.norm(HV, axis=0) > 1e-8))[0]
    thr = 2.0 * math.sqrt(beta); Kstar = (1 - f.phi) / 4.0
    m0 = np.zeros(n); P = np.eye(n) * (lam.max() + rho.max()) * n
    xi = np.zeros(n); eta = np.zeros(m)
    C0 = np.diag(rho).astype(float); C1 = np.zeros((m, m)); e_prev = None
    out = np.zeros((len(Y), n)); ps = np.zeros((len(Y), n)); ms = np.zeros((len(Y), m))
    for t, y in enumerate(Y):
        Q = V @ np.diag(lam * np.exp(np.clip(xi, -8, 20))) @ V.T
        R = np.diag(rho * np.exp(np.clip(eta, -8, 20)))
        mp = F @ m0 + (B @ U[t] if B is not None else 0.0)
        Pp = F @ P @ F.T + Q
        e = y - H @ mp
        S = H @ Pp @ H.T + R + 1e-12 * np.eye(m); Si = np.linalg.inv(S)
        K = Pp @ H.T @ Si
        m0 = mp + K @ e; P = Pp - K @ H @ Pp; P = 0.5 * (P + P.T)
        out[t] = m0
        # --- innovation moments ---
        C0 = (1 - beta) * C0 + beta * np.outer(e, e)
        if e_prev is not None:
            C1 = (1 - beta) * C1 + beta * np.outer(e, e_prev)
        e_prev = e.copy(); C1s = 0.5 * (C1 + C1.T)
        # R: UNGATED innovation-based (Mehra: if Q is right, C0 - H Pp H^T is the sensor variance)
        Rhat = np.diag(C0) - np.diag(H @ Pp @ H.T)
        eta_t = np.clip(np.log(np.clip(Rhat, floorR, None) / rho), -2.0, 20.0)
        eta += np.clip(kappa * (eta_t - eta), -0.5, 0.5)
        # Q: WHITENING (self-limiting -> stable), per active eigenmode
        for k in proc_active:
            hv = HV[:, k]; c0k = float(hv @ C0 @ hv) + 1e-12
            sig = float(hv @ C1s @ hv) / c0k
            xi[k] += float(np.clip(Kstar * 8.0 * (np.sign(sig) * max(abs(sig) - thr, 0.0)), -0.5, 0.5))
        ps[t] = xi; ms[t] = eta
    return out, ps, ms, proc_active


def rmse(est, tru):
    return np.sqrt(((est - tru) ** 2).mean())


def oracle_lagged(F, B, H, U, Y, jstd, pot_s, acc_s, g, n, m, beta=_BETA):
    """The online-achievable bound: an oracle whose Q,R knowledge is EMA-lagged by beta -- the best
    ANY beta-windowed estimator can do.  (It is nearly equal to the full oracle here: the lag is
    cheap; the slack is estimation QUALITY, not lag.)"""
    Bn = np.kron(np.eye(NJ), g[:, None])
    jv = np.zeros(T); pv = np.zeros(T); av = np.zeros(T); cj = jstd[0] ** 2; cp = pot_s[0] ** 2; ca = acc_s[0] ** 2
    for k in range(T):
        cj = (1 - beta) * cj + beta * jstd[k] ** 2; cp = (1 - beta) * cp + beta * pot_s[k] ** 2
        ca = (1 - beta) * ca + beta * acc_s[k] ** 2; jv[k] = cj; pv[k] = cp; av[k] = ca
    m0 = np.zeros(n); P = np.eye(n); out = np.zeros((T, n))
    for k, y in enumerate(Y):
        Q = jv[k] * (Bn @ Bn.T); R = np.zeros((m, m))
        R[np.arange(0, m, 2), np.arange(0, m, 2)] = pv[k]; R[np.arange(1, m, 2), np.arange(1, m, 2)] = av[k]
        mp = F @ m0 + B @ U[k]; Pp = F @ P @ F.T + Q
        K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R); m0 = mp + K @ (y - H @ mp); P = Pp - K @ H @ Pp; out[k] = m0
    return out


def main():
    print("=== 5-DOF phased rig (0026): joint-angle RMSE by phase, 4 seeds ===")
    print("  heuristic = shipped filter; MEHRA = ungated-R + whitening-Q recursive solve;")
    print("  oracle-lagged = the online-achievable bound (EMA-lagged true Q,R).\n")
    agg = {ph[0] + str(ph[1]): {x: [] for x in ("heur", "mehra", "orlag", "orac")} for ph in PHASES}
    g = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
    for seed in range(4):
        f, F, B, H, U, S, Y, jstd, pot_s, acc_s = p26.simulate(seed)
        th_t = S.reshape(T, NJ, ORDER)[:, :, 0]
        heur = f.filter(Y, U=U).mean.reshape(T, NJ, ORDER)[:, :, 0]
        me = mehra_filter(f, Y, U, kappa=0.2)[0].reshape(T, NJ, ORDER)[:, :, 0]
        orl = oracle_lagged(F, B, H, U, Y, jstd, pot_s, acc_s, g, f.n, f.m).reshape(T, NJ, ORDER)[:, :, 0]
        orc = p26.oracle(F, B, H, U, Y, jstd, pot_s, acc_s, g, f.n, f.m).reshape(T, NJ, ORDER)[:, :, 0]
        for nm, a, b in PHASES:
            k = nm + str(a)
            agg[k]["heur"].append(rmse(heur[a:b], th_t[a:b])); agg[k]["mehra"].append(rmse(me[a:b], th_t[a:b]))
            agg[k]["orlag"].append(rmse(orl[a:b], th_t[a:b])); agg[k]["orac"].append(rmse(orc[a:b], th_t[a:b]))
    print(f"  {'phase':10s} {'heur':>8} {'MEHRA':>8} {'orc-lag':>8} {'oracle':>8}   {'heur/lag':>8} {'mehra/lag':>9}")
    tot = {x: 0.0 for x in ("heur", "mehra", "orlag", "orac")}
    for nm, a, b in PHASES:
        k = nm + str(a); v = {x: np.mean(agg[k][x]) for x in tot}
        for x in tot:
            tot[x] += v[x]
        print(f"  {nm:10s} {v['heur']:8.4f} {v['mehra']:8.4f} {v['orlag']:8.4f} {v['orac']:8.4f}   "
              f"{v['heur']/v['orlag']:8.2f} {v['mehra']/v['orlag']:9.2f}")
    print(f"  {'MEAN':10s} {tot['heur']/7:8.4f} {tot['mehra']/7:8.4f} {tot['orlag']/7:8.4f} {tot['orac']/7:8.4f}"
          f"   {tot['heur']/tot['orlag']:8.2f} {tot['mehra']/tot['orlag']:9.2f}")
    print("\n  Note: the Myers-Tapley (state-increment) Q estimator DIVERGES under the process burst")
    print("  (high Q -> trust the bad pot -> large increments -> higher Q_hat); ungated R hurts the")
    print("  process phases (process-elevated accel innovation inflates R). The heuristic's gated")
    print("  structure is load-bearing for stability; naive Mehra does not beat it here.")


if __name__ == "__main__":
    main()
