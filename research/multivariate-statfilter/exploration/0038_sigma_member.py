"""Probe 0038 -- the sigma-point scale POSTERIOR member on the 5-DOF rig (retire the shed).

The single-point walk under-reaches a jump (the AR(1) prior caps its gain) -- which is why the shed
was bolted on.  The parent's cure is a scale POSTERIOR (a windowed GPB1, 0008): carry (mu, Sig),
place 2D+1 sigma points, run the state KF at each, reweight by likelihood, moment-match.  On a jump
the reweighting shifts the posterior mean toward the far edge of the window -- a fast reaction by
likelihood, no threshold, no shed.  Everything is from the class: nu = s^2(1-phi^2) (stationary
AR(1)), reversion phi.  The one span control is the sigma spread c (a coverage budget, like the
SPAN_S window half-span the grid already uses).

This runs a single sigma-point member (stationary, one (phi,s)) with control input on the 5-DOF
rig, and prints adaptive/oracle on the five regimes next to the shed baseline.  If it reaches the
hot regimes parameter-free, the shed is retired (and a (phi,s) bank then only adds the family span).
"""
import os
import sys
import math
import importlib.util

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402
from statfilter.core import _LOG2PI  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "p34", os.path.join(os.path.dirname(__file__), "0034_profile.py"))
p34 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p34)
NJ, ORDER, DT, POT, ACC, JERK = p34.NJ, p34.ORDER, p34.DT, p34.POT, p34.ACC, p34.JERK
BASE = {"pot-hot": 1.189, "process+pot": 1.421, "SENSOR": 1.133, "PROCESS": 1.122, "BOTH": 2.378}


def member(phi=0.9, s=0.5):
    return AdaptiveKalmanFilter.kinematic(NJ, ORDER, DT, process_var=JERK ** 2,
                                          meas_var={"pos": POT ** 2, "acc": ACC ** 2},
                                          measured=("pos", "acc"), control=True, phi=phi, s=s)


def sigma_filter(f, Y, U, c=1.0):
    """Run the sigma-point scale-posterior walker using f's matrices.  Returns state means (T, n)."""
    F, B, H, n, m = f.F, f.B, f.H, f.n, f.m
    D = n + m
    phi, s = f.phi, f.s
    active = f.active
    nu = np.zeros(D)
    # per-axis stationary innovation on the log-scale AR(1): process modes carry lam-weighted swing,
    # sensors carry rho-weighted swing; both use the class (phi, s) -> nu = s^2 (1 - phi^2).
    nu[active] = s ** 2 * (1.0 - phi ** 2)
    Q_of, R_of = f._Q_of, f._R_of

    mu = np.zeros(D); Sig = np.zeros(D); Sig[active] = s ** 2
    mm = np.zeros(n); P = np.eye(n) * (f.lam.max() + f.rho.max()) * n
    out = np.zeros((len(Y), n))
    w0 = 1.0 / 3.0; wother = (1.0 - w0) / (2 * len(active))
    for t, y in enumerate(Y):
        bu = B @ U[t]
        mu_p = phi * mu
        Sig_p = phi ** 2 * Sig + nu
        sd = c * np.sqrt(np.maximum(Sig_p, 1e-12))
        pts = [mu_p.copy()]; w = [w0]
        for k in active:
            for sgn in (+1.0, -1.0):
                sp = mu_p.copy(); sp[k] = mu_p[k] + sgn * sd[k]
                pts.append(sp); w.append(wother)
        pts = np.array(pts); w = np.array(w); w = w / w.sum()
        lls = np.empty(len(pts)); ms = np.empty((len(pts), n)); Ps = np.empty((len(pts), n, n))
        for j, sp in enumerate(pts):
            Pp = F @ P @ F.T + Q_of(sp[:n])
            mpred = F @ mm + bu
            e = y - H @ mpred
            Sm = H @ Pp @ H.T + R_of(sp[n:]) + 1e-9 * np.eye(m)
            Si = np.linalg.inv(Sm)
            _, logdet = np.linalg.slogdet(Sm)
            lls[j] = -0.5 * (m * _LOG2PI + logdet + float(e @ Si @ e))
            K = Pp @ H.T @ Si
            ms[j] = mpred + K @ e; Ps[j] = Pp - K @ H @ Pp
        W = w * np.exp(lls - lls.max()); W = W / W.sum()
        mu = W @ pts; Sig = np.maximum(W @ (pts - mu) ** 2, 1e-6)
        mu[active] = np.clip(mu[active], -8.0, 20.0)
        mm = W @ ms
        dmv = ms - mm
        P = np.einsum("j,jab->ab", W, Ps) + np.einsum("j,ja,jb->ab", W, dmv, dmv)
        P = 0.5 * (P + P.T)
        out[t] = mm
    return out


def main(nseed=4, c=1.0, phi=0.9, s=0.5):
    print(f"sigma-point member (phi={phi}, s={s}, c={c}), {nseed} seeds:")
    print(f"  {'regime':12s} {'sigma/orc':>10} {'shed base':>10}")
    for regime, tag in p34.REGIMES:
        ad = np.zeros(nseed); oc = np.zeros(nseed)
        for seed in range(nseed):
            f, F, B, H, U, S, Y, jstd, pot, acc = p34.sim(seed, regime)
            fm = member(phi=phi, s=s)
            ad[seed] = p34.rms(sigma_filter(fm, Y, U, c=c), S)
            oc[seed] = p34.rms(p34.oracle(F, B, H, U, Y, jstd, pot, acc, f.n, f.m), S)
        print(f"  {tag:12s} {(ad/oc).mean():10.3f} {BASE[tag]:10.2f}")


if __name__ == "__main__":
    import sys as _s
    kw = dict(a.split("=") for a in _s.argv[1:]) if len(_s.argv) > 1 else {}
    main(c=float(kw.get("c", 1.0)), phi=float(kw.get("phi", 0.9)), s=float(kw.get("s", 0.5)))
