"""Probe 0032 -- chase the oracle across the sensor<->process continuum; find the in-between model.

The whiteness gate hard-switches: white -> attribute the excess innovation to the SENSOR (inflate
R), correlated -> to the PROCESS (inflate Q).  But "sensor failing" and "noisy-but-useful under
process" are a GRADIENT; a channel can be partly white.  The oracle never leaks because it KNOWS
the split; the adaptive's only evidence is the innovation's TIME structure -- the process
contribution is correlated (it persists through the closed-loop dynamics), the sensor contribution
is white.  So the innovation on a channel is  e = s (AR signal, process) + n (white, sensor), and
the derived interpolant between the two arms is the ALLOCATION of e between s and n.

Candidate (AR(1)-plus-white).  If s is AR(1) with coefficient a, C0 = var(s)+var(n) and the lag-1
covariance C1 = a var(s) (white contributes 0 at lag 1).  So the PROCESS fraction of the innovation
variance is
        f_proc = var(s)/C0 = (C1/a)/C0 = rho1 / a,        rho1 = C1/C0,
and the SENSOR fraction is 1 - rho1/a.  a is the process channel's OWN lag-1 coefficient (a = rho1
under pure process).  This is smooth, linear in rho1, and its two limits ARE the arms: rho1=0 ->
all sensor (the robust/scale-R arm), rho1=a -> all process (the Q arm).  No threshold, no sigmoid.

This probe (1 joint, pot on position + accel on acceleration, a burst whose split between process
jerk and accel-sensor noise is swept) tests: does rho1/a track the TRUE process fraction across
the continuum?  If yes, it is the derived transition.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402

np.set_printoptions(precision=4, suppress=True)
ORDER, DT = 3, 0.01
POT, ACC = 0.06, 0.02
JERK0 = 0.6
T = 1200
ON = slice(400, 1100)                                   # long burst -> steady correlation estimate
G = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])


def build():
    return AdaptiveKalmanFilter.kinematic(1, ORDER, DT, process_var=JERK0 ** 2,
                                          meas_var={"pos": POT ** 2, "acc": ACC ** 2},
                                          measured=("pos", "acc"), control=True, s=0.5)


def sim(seed, jerk_mult, acc_mult):
    """A burst with process jerk x jerk_mult AND accel-sensor noise x acc_mult in ON."""
    f = build(); F, B, H, n, m = f.F, f.B, f.H, f.n, f.m
    rng = np.random.default_rng(seed); t = np.arange(T) * DT
    U = 1.5 * np.sin(2 * np.pi * 0.4 * t)[:, None]
    jstd = np.full(T, JERK0); acc = np.full(T, ACC); pot = np.full(T, POT)
    jstd[ON] *= jerk_mult; acc[ON] *= acc_mult
    s = np.zeros(n); S = np.zeros((T, n)); Y = np.zeros((T, m))
    for k in range(T):
        s = F @ s + (B @ U[k] if B is not None else 0) + (G * (jstd[k] * rng.standard_normal()))
        S[k] = s
        Y[k, 0] = H[0] @ s + pot[k] * rng.standard_normal()
        Y[k, 1] = H[1] @ s + acc[k] * rng.standard_normal()
    return f, F, B, H, U, S, Y, jstd, pot, acc


def accel_rho1(f, Y, U):
    """Lag-1 autocorrelation of the accel-channel innovation, in the burst, at BASE noise
    (non-adaptive) -- the evidence the filter actually sees."""
    fna = build(); fna.active = np.array([], dtype=int); fna.r = 0
    e = fna.filter(Y, U=U).innovation[ON, 1]
    return float(np.corrcoef(e[1:], e[:-1])[0, 1])


def acf(e, K):
    e = e - e.mean(); c0 = float((e * e).mean())
    return [float((e[k:] * e[:-k]).mean()) / c0 for k in range(1, K + 1)]


def stats(seed, jm, am):
    """Return (rho1..rho4, f_proc_true) for the accel channel, at BASE noise (the evidence the
    filter sees).  f_proc_true = process share of the accel innovation variance = (total accel
    innovation var - known sensor var)/total (the sensor part is exactly the accel R)."""
    f, F, B, H, U, S, Y, jstd, pot, acc = sim(seed, jm, am)
    fna = build(); fna.active = np.array([], dtype=int); fna.r = 0
    e = fna.filter(Y, U=U).innovation[ON, 1]
    ev = float((e ** 2).mean()); sens = float((acc[ON] ** 2).mean())
    return acf(e, 4), max(ev - sens, 0.0) / max(ev, 1e-12)


def cross(seed, jm, am):
    """The CROSS-sensor separator.  Process = common state error, seen through H by BOTH the pot
    and the accel -> off-diagonal innovation covariance.  Sensor noise is diagonal (local).  So the
    accel innovation's process share should be recoverable from its covariance with the pot channel,
    which the accel's own sensor noise cannot touch.  Report the peak |corr(e_pot, e_accel)| over
    small lags (the pot->accel coupling is lagged: jerk integrates to position)."""
    f, F, B, H, U, S, Y, jstd, pot, acc = sim(seed, jm, am)
    fna = build(); fna.active = np.array([], dtype=int); fna.r = 0
    E = fna.filter(Y, U=U).innovation[ON]
    ep = E[:, 0] - E[:, 0].mean(); ea = E[:, 1] - E[:, 1].mean()
    sp = ep.std(); sa = ea.std()
    xc = []
    for lag in range(-6, 7):
        if lag >= 0:
            xc.append((ep[lag:] * ea[:len(ea) - lag]).mean() / (sp * sa))
        else:
            xc.append((ep[:lag] * ea[-lag:]).mean() / (sp * sa))
    return max(xc, key=abs)


def f_base_derived():
    """The NOMINAL process share of the accel innovation variance, f_base = 1 - R/S, from the
    filter's own steady-state innovation covariance S = H Pp H^T + R at nominal Q,R (no tuning)."""
    f = build()
    fna = build(); fna.active = np.array([], dtype=int); fna.r = 0
    # nominal accel innovation variance in a calm run
    _, _, _, _, U, _, Y, _, _, acc = sim(0, 1.0, 1.0)
    S_acc = float((fna.filter(Y, U=U).innovation[ON, 1] ** 2).mean())
    R_acc = ACC ** 2
    return 1.0 - R_acc / S_acc


def main():
    fb = f_base_derived()
    print("DERIVED transition:  process share of the accel innovation excess = clip(f_base + rho1),")
    print(f"f_base = 1 - R/S (nominal process share, from the model) = {fb:.3f}.")
    print("The whiteness gate wrongly maps rho1=0 -> pure sensor; the truth at rho1=0 is f_base.")
    print(f"  {'jerk x':>7} {'accS x':>7} {'rho1':>7} {'f_proc_pred':>12} {'f_proc_true':>12}"
          f" {'err':>7}")
    errs = []
    for jm, am in [(20, 1), (16, 4), (14, 6), (12, 8), (10, 10), (8, 12), (6, 14), (3, 17),
                   (1, 20), (1, 1)]:
        rows = [stats(seed, jm, am) for seed in range(4)]
        r1 = float(np.mean([x[0][0] for x in rows])); fp = float(np.mean([x[1] for x in rows]))
        pred = min(max(fb + r1, 0.0), 1.0); errs.append(abs(pred - fp))
        print(f"  {jm:7d} {am:7d} {r1:7.3f} {pred:12.3f} {fp:12.3f} {pred - fp:+7.3f}")
    print(f"\nmean |err| of the derived law across the continuum = {np.mean(errs):.3f}")


if __name__ == "__main__":
    main()
