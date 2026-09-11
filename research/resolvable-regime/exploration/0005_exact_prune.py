"""0005 -- C5, the exact test: remove ONLY the members above the resolution ceiling.

0004 could not pose C5's actual claim.  The public API takes `phis` and `ss` as a
PRODUCT grid, so the smallest box that drops the 2 over-ceiling members
(phi=0.70 and 0.85 at s=3.20) also drops a third, LEGAL member
(phi=0.95, s=3.20, increment SD 0.999).  0004 measured that 3-member removal at
+12.7% on the sensor-degradation regime -- but could not say whether the cost
came from the 2 illegal members or from the 1 legal one.

This probe removes exactly the 2.  It loads a patched copy of `lucid.py` in which
the single `cells = [...]` line consults a module-level predicate, and it PINS
that copy against the shipped filter with the predicate off: the two must agree
bit for bit, or the probe reports nothing.

Boxes:
    shipped    15 members
    ceiling    13 members -- drops only  s*sqrt(1-phi^2) > sqrt(2)
    no-top-s   12 members -- drops all of s = 3.20          (0004's box, for reference)

Run: python3 0005_exact_prune.py
"""
import importlib.util
import math
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from lucid import LucidFilter as ShippedFilter                     # noqa: E402

SRC = os.path.join(ROOT, "lucid", "filter", "lucid.py")
OLD = ("        cells = [(ph, sv, bq, br) "
       "for ph in phis for sv in ss for (bq, br) in bases]")
NEW = ("        cells = [(ph, sv, bq, br) "
       "for ph in phis for sv in ss for (bq, br) in bases]\n"
       "        if CELL_FILTER is not None:\n"
       "            cells = [c for c in cells if CELL_FILTER(c[0], c[1])]")


def load_patched():
    src = open(SRC).read()
    if src.count(OLD) != 1:
        raise SystemExit("the cells line moved; re-point the patch before trusting this")
    src = src.replace(OLD, NEW).replace(
        'from __future__ import annotations',
        'from __future__ import annotations\nCELL_FILTER = None', 1)
    mod = types.ModuleType("lucid_patched")
    mod.__file__ = SRC
    sys.modules["lucid_patched"] = mod          # dataclass() resolves __module__ here
    mod.__dict__["__name__"] = "lucid_patched"
    exec(compile(src, SRC, "exec"), mod.__dict__)
    return mod


# ------------------------------------------------------------------- the rig
SEED, N, JUMP_AT, JUMP, NOISE_AT = 11, 900, 380, 9.0, 600
Q_TRUE, S2_A, S2_C = 0.02, 1.0, 9.0
NSEED = 12
PHIS, SS = (0.70, 0.85, 0.95), (0.20, 0.40, 0.80, 1.60, 3.20)
CEIL = math.sqrt(2.0)


def generate(seed):
    rng = np.random.default_rng(seed)
    theta = np.cumsum(rng.normal(0.0, math.sqrt(Q_TRUE), N))
    theta[JUMP_AT:] += JUMP
    sd = np.where(np.arange(N) < NOISE_AT, math.sqrt(S2_A), math.sqrt(S2_C))
    return theta, theta + rng.normal(0.0, sd)


def kalman(y, Q, R):
    m, P, out = y[0], R, np.empty(N)
    for t in range(N):
        P += Q
        g = P / (P + R)
        m = m + g * (y[t] - m)
        P = (1.0 - g) * P
        out[t] = m
    return out


def regimes():
    return [("A steady", slice(80, JUMP_AT)),
            ("B jump+40", slice(JUMP_AT, JUMP_AT + 40)),
            ("C noisy", slice(NOISE_AT + 40, N))]


def settle(err):
    seg = np.abs(err[JUMP_AT:JUMP_AT + 120]) < 1.0
    for i in range(seg.size):
        if seg[i:].all():
            return i
    return seg.size


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    mod = load_patched()

    print("per-step increment SD across the shipped box, against the sqrt(2) ceiling:")
    dropped = []
    for ph in PHIS:
        for sv in SS:
            inc = sv * math.sqrt(1.0 - ph * ph)
            if inc > CEIL:
                dropped.append((ph, sv, inc))
    for ph, sv, inc in dropped:
        print(f"    OVER CEILING: phi={ph}, s={sv}  ->  {inc:.3f}")

    # --- pin: patched-with-filter-off must equal the shipped filter exactly
    theta, y = generate(SEED)
    mod.CELL_FILTER = None
    a = np.asarray(mod.LucidFilter(phis=PHIS, ss=SS).filter(y.reshape(-1, 1)).mean)
    b = np.asarray(ShippedFilter(phis=PHIS, ss=SS).filter(y.reshape(-1, 1)).mean)
    dev = float(np.max(np.abs(a - b)))
    print(f"\npin, patched vs shipped with the predicate off: max |dev| = {dev:.3e}")
    if dev > 0.0:
        raise SystemExit("patched copy is not the shipped filter; nothing below is valid")
    print("  exact.  the patched copy is the shipped filter.")

    FILTERS = {
        "shipped   (15 members)": None,
        "ceiling   (13 members)": lambda ph, sv: sv * math.sqrt(1 - ph * ph) <= CEIL,
        "no-top-s  (12 members)": lambda ph, sv: sv < 3.0,
    }

    acc = {k: {r[0]: [] for r in regimes()} for k in FILTERS}
    for k in FILTERS:
        acc[k]["settle"], acc[k]["calib"] = [], []
    orc = {r[0]: [] for r in regimes()}

    for seed in range(SEED, SEED + NSEED):
        theta, y = generate(seed)
        km = kalman(y, Q_TRUE, S2_A)
        for nm, sl in regimes():
            orc[nm].append(float(np.mean((km[sl] - theta[sl]) ** 2)))
        for k, pred in FILTERS.items():
            mod.CELL_FILTER = pred
            r = mod.LucidFilter(phis=PHIS, ss=SS).filter(y.reshape(-1, 1))
            m = np.asarray(r.mean).reshape(-1)
            v = np.asarray(r.var).reshape(-1)
            for nm, sl in regimes():
                acc[k][nm].append(float(np.mean((m[sl] - theta[sl]) ** 2)))
            acc[k]["settle"].append(settle(m - theta))
            sl = regimes()[2][1]
            acc[k]["calib"].append(float(np.mean((m[sl] - theta[sl]) ** 2 / v[sl])))
    mod.CELL_FILTER = None

    print(f"\nRMSE ratio to an oracle Kalman told the truth, {NSEED} seeds:")
    print(f"\n{'box':26s}{'A steady':>10s}{'B jump':>10s}{'C noisy':>10s}"
          f"{'settle':>9s}{'E[e2/S]':>10s}")
    base = None
    for k in FILTERS:
        row = [math.sqrt(np.mean(acc[k][nm]) / np.mean(orc[nm])) for nm, _ in regimes()]
        st, cb = np.mean(acc[k]["settle"]), np.mean(acc[k]["calib"])
        print(f"{k:26s}{row[0]:10.3f}{row[1]:10.3f}{row[2]:10.3f}{st:9.1f}{cb:10.3f}")
        if base is None:
            base = row + [st, cb]
    print("\nrelative to the shipped box:")
    for k in FILTERS:
        row = [math.sqrt(np.mean(acc[k][nm]) / np.mean(orc[nm])) for nm, _ in regimes()]
        print(f"{k:26s}{row[0]/base[0]:10.4f}{row[1]/base[1]:10.4f}{row[2]/base[2]:10.4f}"
              f"{np.mean(acc[k]['settle'])/base[3]:9.4f}"
              f"{np.mean(acc[k]['calib'])/base[4]:10.4f}")
    print("\nC5 predicts the `ceiling` row is 1.0000 everywhere.")
