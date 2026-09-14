"""0004 -- C5: do the bank members above the resolution ceiling earn their keep?

C5 (SUMMARY): the Fisher information one Gaussian observation carries about its
own log-variance is exactly 1/2, so the one-event blur width on a log-scale is
sqrt(2) nats.  A class member whose log-scale moves further than that between
observations is asserting structure the record cannot resolve.  Across the
shipped 15-member `(phi, s)` box the per-step increment SD `s sqrt(1 - phi^2)`
runs 0.062 -> 2.285, so 2 members sit above sqrt(2):

    phi=0.70, s=3.20  ->  2.285        phi=0.85, s=3.20  ->  1.686

Prediction: those members are inert or harmful; removing them costs nothing.

The public API takes `phis` and `ss` as a PRODUCT grid, so the exact 13-member
prune is not expressible through it.  What is testable, and is the practically
interesting version, is three boxes:

    shipped      3 phi x 5 s = 15 members  (2 over the ceiling)
    no-top-s     3 phi x 4 s = 12 members  (drops all of s = 3.20)
    legal-5      1 phi x 5 s =  5 members  (phi = 0.95 only, where the whole
                                            s ladder sits under the ceiling --
                                            max increment SD 0.999)

`legal-5` is the interesting one: every member is resolution-legal and it is 3x
cheaper.  NOTE it is not a test of C2 -- 0002 killed C2, and this box is narrow
on the STIFF axis's partner rather than on the flat axis.  It is a test of
whether the flat axis needs three nodes at all.

Rig: the README hero series (README-004), which is the repo's own acceptance
gate -- steady stretch, level jump, sensor gets 3x noisier -- plus its three
published claims.  No figure, numbers only.

Run: python3 0004_resolution_pruning.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from lucid import LucidFilter                                      # noqa: E402

SEED, N, JUMP_AT, JUMP, NOISE_AT = 11, 900, 380, 9.0, 600
Q_TRUE, S2_A, S2_C = 0.02, 1.0, 9.0
NSEED = 12

SHIPPED_PHIS, SHIPPED_SS = (0.70, 0.85, 0.95), (0.20, 0.40, 0.80, 1.60, 3.20)
BOXES = {
    "shipped  (3 phi x 5 s = 15)": (SHIPPED_PHIS, SHIPPED_SS),
    "no-top-s (3 phi x 4 s = 12)": (SHIPPED_PHIS, (0.20, 0.40, 0.80, 1.60)),
    "legal-5  (1 phi x 5 s =  5)": ((0.95,), SHIPPED_SS),
    "legal-4  (1 phi x 4 s =  4)": ((0.95,), (0.20, 0.40, 0.80, 1.60)),
}


def generate(seed):
    rng = np.random.default_rng(seed)
    theta = np.cumsum(rng.normal(0.0, math.sqrt(Q_TRUE), N))
    theta[JUMP_AT:] += JUMP
    sd = np.where(np.arange(N) < NOISE_AT, math.sqrt(S2_A), math.sqrt(S2_C))
    return theta, theta + rng.normal(0.0, sd)


def kalman(y, Q, R):
    """Oracle: told the true Q and the regime-A measurement variance."""
    m, P = y[0], R
    out, var = np.empty(N), np.empty(N)
    for t in range(N):
        P += Q
        S = P + R
        g = P / S
        m = m + g * (y[t] - m)
        P = (1.0 - g) * P
        out[t], var[t] = m, P
    return out, var


def run_box(y, phis, ss):
    f = LucidFilter(phis=phis, ss=ss)
    r = f.filter(y.reshape(-1, 1))
    mean = np.asarray(r.mean).reshape(-1)
    cov = np.asarray(r.var).reshape(-1)
    return mean, cov


def regimes():
    """(name, slice) -- A steady, B the jump's absorption, C the noisy sensor."""
    return [("A steady", slice(80, JUMP_AT)),
            ("B jump+40", slice(JUMP_AT, JUMP_AT + 40)),
            ("C noisy", slice(NOISE_AT + 40, N))]


def settle_steps(err, thresh):
    """Steps after the jump until |error| is under `thresh` and stays under."""
    seg = np.abs(err[JUMP_AT:JUMP_AT + 120])
    below = seg < thresh
    for i in range(below.size):
        if below[i:].all():
            return i
    return below.size


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    acc = {k: {r[0]: [] for r in regimes()} for k in BOXES}
    for k in BOXES:
        acc[k]["settle"] = []
        acc[k]["calib"] = []
    orc = {r[0]: [] for r in regimes()}

    for seed in range(SEED, SEED + NSEED):
        theta, y = generate(seed)
        km, kv = kalman(y, Q_TRUE, S2_A)
        for nm, sl in regimes():
            orc[nm].append(float(np.mean((km[sl] - theta[sl]) ** 2)))
        for k, (phis, ss) in BOXES.items():
            m, c = run_box(y, phis, ss)
            for nm, sl in regimes():
                acc[k][nm].append(float(np.mean((m[sl] - theta[sl]) ** 2)))
            acc[k]["settle"].append(settle_steps(m - theta, 1.0))
            sl = regimes()[2][1]
            acc[k]["calib"].append(float(np.mean((m[sl] - theta[sl]) ** 2 / c[sl])))

    print(f"\nrig: README-004 hero series, {NSEED} seeds.  RMSE ratio to an "
          f"oracle Kalman told the truth.")
    print(f"\n{'box':30s}{'A steady':>10s}{'B jump':>10s}{'C noisy':>10s}"
          f"{'settle':>9s}{'E[e2/S]':>10s}")
    base = None
    for k in BOXES:
        row = []
        for nm, _ in regimes():
            row.append(math.sqrt(np.mean(acc[k][nm]) / np.mean(orc[nm])))
        st = np.mean(acc[k]["settle"])
        cb = np.mean(acc[k]["calib"])
        print(f"{k:30s}{row[0]:10.3f}{row[1]:10.3f}{row[2]:10.3f}{st:9.1f}{cb:10.3f}")
        if base is None:
            base = row + [st, cb]
    print("\nrelative to the shipped box (1.000 = identical):")
    for k in BOXES:
        row = [math.sqrt(np.mean(acc[k][nm]) / np.mean(orc[nm])) for nm, _ in regimes()]
        rel = [row[i] / base[i] for i in range(3)]
        print(f"{k:30s}{rel[0]:10.4f}{rel[1]:10.4f}{rel[2]:10.4f}"
              f"{np.mean(acc[k]['settle']) / base[3]:9.4f}"
              f"{np.mean(acc[k]['calib']) / base[4]:10.4f}")
    print("\nC5 predicts no-top-s ~ 1.0000 everywhere (the over-ceiling members are inert).")
