"""Probe 0033 -- how much of the BOTH gap is IRREDUCIBLE observability loss (what the oracle
discounts), vs a fixable leak?

0032 (both.py) found: in BOTH (process jerk AND accel-sensor noise both hot) the process is nearly
unobservable from the innovations -- the accel that sees jerk directly is drowned by its own 20x
sensor noise (its process correlation dilutes +0.71 -> ~0), and the pot does not see jerk.  The
oracle is HANDED Q, so it discounts this.  Decompose the BOTH error by freezing one side at oracle:

  oracle        : true Q AND true R           -> the (unfair) full-information floor
  adaptive       : infer both                  -> what we ship
  oracle-Q       : true Q, infer R            -> removes the Q-observability problem
  oracle-R       : true R, infer Q            -> removes the R side; leaves Q inference

If oracle-Q ~ oracle (closes) but oracle-R ~ adaptive (stays open), the leak is Q-observability --
irreducible given the filter must INFER the masked jerk -- and the full-oracle gap overstates it.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402

ORDER, DT = 3, 0.01
POT, ACC, JERK0 = 0.06, 0.02, 0.6
T = 1200
ON = slice(400, 1100)
G = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
JM, AM = 20.0, 20.0                                    # BOTH: process x20 AND accel-sensor x20


def build():
    return AdaptiveKalmanFilter.kinematic(1, ORDER, DT, process_var=JERK0 ** 2,
                                          meas_var={"pos": POT ** 2, "acc": ACC ** 2},
                                          measured=("pos", "acc"), control=True, s=0.5)


def sim(seed, jm=JM, am=AM):
    f = build(); F, B, H, n, m = f.F, f.B, f.H, f.n, f.m
    rng = np.random.default_rng(seed); t = np.arange(T) * DT
    U = 1.5 * np.sin(2 * np.pi * 0.4 * t)[:, None]
    jstd = np.full(T, JERK0); acc = np.full(T, ACC); pot = np.full(T, POT)
    jstd[ON] *= jm; acc[ON] *= am
    s = np.zeros(n); S = np.zeros((T, n)); Y = np.zeros((T, m))
    for k in range(T):
        s = F @ s + (B @ U[k] if B is not None else 0) + G * (jstd[k] * rng.standard_normal())
        S[k] = s
        Y[k, 0] = H[0] @ s + pot[k] * rng.standard_normal()
        Y[k, 1] = H[1] @ s + acc[k] * rng.standard_normal()
    return f, U, S, Y


def true_mu():
    """Oracle log-scales in ON: process var x JM^2, accel var x AM^2 (pot unchanged)."""
    f = build()
    mu = np.zeros(f.n + f.m)
    # process eigenmodes: the jerk mode carries the scale; set all process axes to log(JM^2)
    mu[:f.n] = 2.0 * math.log(JM)
    mu[f.n + 0] = 0.0                                  # pot unchanged
    mu[f.n + 1] = 2.0 * math.log(AM)                  # accel
    return mu


def run(seed, mode):
    """Step through so the oracle scales are TIME-VARYING (baseline outside ON, true inside).
    freeze_proc / freeze_sens: overwrite that side with the true time-varying scale each step and
    drop it from the walk; the other side adapts."""
    f, U, S, Y = sim(seed)
    n = f.n
    tm = true_mu()
    freeze_proc = mode in ("oracle", "oracle-Q")
    freeze_sens = mode in ("oracle", "oracle-R")
    keep = [k for k in f.active if (k >= n or not freeze_proc) and (k < n or not freeze_sens)]
    f.active = np.array(keep, dtype=int); f.r = len(keep)
    est = np.zeros((T, n))
    on = np.zeros(T, bool); on[ON] = True
    for k in range(T):
        if freeze_proc:
            f.mu[:n] = tm[:n] if on[k] else 0.0
        if freeze_sens:
            f.mu[n:] = tm[n:] if on[k] else 0.0
        est[k] = f.update(Y[k], u=U[k]).mean
    return float(np.sqrt(np.mean((est[ON, 0] - S[ON, 0]) ** 2)))


def masking():
    """Show the jerk is unobservable in BOTH: its accel lag-1 correlation (base filter) is +0.71
    with process alone but dilutes to ~0 once the accel carries its own 20x sensor noise."""
    def acc_rho1(jm, am):
        rs = []
        for seed in range(6):
            f, U, _, Y = sim(seed, jm, am)
            f.active = np.array([], dtype=int); f.r = 0
            e = f.filter(Y, U=U).innovation[ON, 1]; e = e - e.mean()
            rs.append(float((e[1:] * e[:-1]).mean() / (e * e).mean()))
        return float(np.mean(rs))
    print(f"accel lag-1 corr:  PROCESS-only (jerk x20) = {acc_rho1(20, 1):+.3f}"
          f"   BOTH (jerk x20, accel x20) = {acc_rho1(20, 20):+.3f}  (diluted -> masked)\n")


def main():
    masking()
    seeds = range(6)
    res = {mode: float(np.mean([run(s, mode) for s in seeds]))
           for mode in ("oracle", "oracle-Q", "oracle-R", "adaptive")}
    orc = res["oracle"]
    print("BOTH-regime position RMSE (jerk x20 AND accel-sensor x20), 6 seeds:")
    print(f"  {'variant':>10} {'RMSE':>9} {'/oracle':>8}")
    for mode in ("oracle", "oracle-Q", "oracle-R", "adaptive"):
        print(f"  {mode:>10} {res[mode]:9.4f} {res[mode] / orc:8.2f}")
    print("\noracle-Q ~ oracle  => knowing the masked jerk closes it => the gap is Q-observability")
    print("oracle-R still open => inferring Q under the noisy accel is the irreducible part.")


if __name__ == "__main__":
    main()
