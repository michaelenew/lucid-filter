"""0010 -- the derived class ladder, tested against the shipped `(phi, s)` box.

THE DERIVATION (full statement in 0010_the_derivation.md).

The log-scale channel is the SAME object as the Q/R split channel, one level up:

  * The split channel: state theta, "process" Q, "sensor" R.  Per step only the
    total is visible; over a sequence the split is visible through the MA(1)
    structure, on the compact arclength t = arccos(1-K), with Fisher information
    1 per step (measured in 0009: 0.99-1.15 over a 400x range in q).

  * The scale channel: "state" lam, "process" nu = s^2 (1 - phi^2) per step,
    "sensor" = one observation's blur on lam, variance 1/I = 2 since I(lam) = 1/2.
    Per step only lam's level is visible -- that is what the walk tracks.  How
    FAST lam moves is not per-step identifiable, so it is a bank quantity, and it
    lives on the same compact arclength.

So the `(phi, s)` box is the scale channel's own split ladder, and it is built by
the same rule, with no new constants:

    q_lam = nu / 2,   K = gain from q_lam,   t = arccos(1 - K) in [0, pi/2]
    ends    : t in [blur, pi/2 - blur],  blur = sqrt(2/N),  N = 1/(1 - forget)
    spacing : the engine's own Sparrow factor, 1.5 * blur
    count   : follows.  Nothing is chosen.

Two free choices remain -- the window's WIDTH and the class's persistence -- and
the derivation fixes both, because the window must be the channel's own
predictive width:

    s   = sqrt(noise / (1 - K)) = sqrt(2 / cos t)          [predictive SD of lam]
    phi = sqrt(1 - K^2)                                    [forced by nu = s^2(1-phi^2)]

which is exact (residual 1e-16) and gives s >= sqrt(2) ALWAYS: the window can
never be finer than one observation can resolve.  Three of the shipped five
`s` rungs (0.20, 0.40, 0.80) are below that.

Tested here against the shipped box on the repo's own hero gate.

Run: python3 0010_derived_ladder.py
"""
import math
import os
import sys
import time
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
       "        if CELL_PAIRS is not None:\n"
       "            cells = [(ph, sv, bq, br) for (ph, sv) in CELL_PAIRS\n"
       "                     for (bq, br) in bases]")


def load_patched():
    src = open(SRC).read()
    if src.count(OLD) != 1:
        raise SystemExit("the cells line moved; re-point the patch")
    src = src.replace(OLD, NEW).replace(
        'from __future__ import annotations',
        'from __future__ import annotations\nCELL_PAIRS = None', 1)
    mod = types.ModuleType("lucid_pairs")
    mod.__file__ = SRC
    sys.modules["lucid_pairs"] = mod
    mod.__dict__["__name__"] = "lucid_pairs"
    exec(compile(src, SRC, "exec"), mod.__dict__)
    return mod


# ------------------------------------------------------------- the ladder
def derived_ladder(forget=0.999, gap_factor=1.5, stride=1):
    """The class ladder, from `forget` and the blur width alone."""
    N = 1.0 / (1.0 - forget)
    blur = math.sqrt(2.0 / N)
    gap = gap_factor * blur * stride
    lo, hi = blur, math.pi / 2.0 - blur
    n = int((hi - lo) // gap) + 1
    out = []
    for j in range(n):
        t = lo + j * gap
        c = math.cos(t)
        K = 1.0 - c
        out.append((math.sqrt(max(1.0 - K * K, 1e-12)), math.sqrt(2.0 / c)))
    return out


# ---------------------------------------------------------------- the rig
SEED, N_T, JUMP_AT, JUMP, NOISE_AT = 11, 900, 380, 9.0, 600
Q_TRUE, S2_A, S2_C = 0.02, 1.0, 9.0
NSEED = 12
PHIS, SS = (0.70, 0.85, 0.95), (0.20, 0.40, 0.80, 1.60, 3.20)


def generate(seed):
    rng = np.random.default_rng(seed)
    th = np.cumsum(rng.normal(0.0, math.sqrt(Q_TRUE), N_T))
    th[JUMP_AT:] += JUMP
    sd = np.where(np.arange(N_T) < NOISE_AT, math.sqrt(S2_A), math.sqrt(S2_C))
    return th, th + rng.normal(0.0, sd)


def kalman(y, Q, R):
    m, P, out = y[0], R, np.empty(N_T)
    for t in range(N_T):
        P += Q
        g = P / (P + R)
        m = m + g * (y[t] - m)
        P = (1.0 - g) * P
        out[t] = m
    return out


def regimes():
    return [("A steady", slice(80, JUMP_AT)),
            ("B jump+40", slice(JUMP_AT, JUMP_AT + 40)),
            ("C noisy", slice(NOISE_AT + 40, N_T))]


def settle(err):
    seg = np.abs(err[JUMP_AT:JUMP_AT + 120]) < 1.0
    for i in range(seg.size):
        if seg[i:].all():
            return i
    return seg.size


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    mod = load_patched()

    # pin: patched with the override off must be the shipped filter, exactly
    _, y = generate(SEED)
    mod.CELL_PAIRS = None
    a = np.asarray(mod.LucidFilter(phis=PHIS, ss=SS).filter(y.reshape(-1, 1)).mean)
    b = np.asarray(ShippedFilter(phis=PHIS, ss=SS).filter(y.reshape(-1, 1)).mean)
    dev = float(np.max(np.abs(a - b)))
    print(f"pin, patched vs shipped with the override off: max|dev| = {dev:.3e}")
    if dev > 0.0:
        raise SystemExit("patched copy is not the shipped filter")
    print("  exact.\n")

    lad = derived_ladder()
    lad2 = derived_ladder(stride=2)
    lad3 = derived_ladder(stride=3)
    print(f"derived ladder: {len(lad)} rungs, "
          f"phi {lad[-1][0]:.3f}..{lad[0][0]:.5f}, s {lad[0][1]:.2f}..{lad[-1][1]:.2f}")
    print(f"  stride 2: {len(lad2)} rungs;  stride 3: {len(lad3)} rungs")
    print(f"shipped box:   15 cells, phi 0.70..0.95, s 0.20..3.20\n")

    BOXES = {
        "shipped     (15 cells)": None,
        f"derived     ({len(lad)} cells)": lad,
        f"derived /2  ({len(lad2)} cells)": lad2,
        f"derived /3  ({len(lad3)} cells)": lad3,
    }

    acc = {k: {r[0]: [] for r in regimes()} for k in BOXES}
    for k in BOXES:
        acc[k]["settle"], acc[k]["calib"], acc[k]["secs"] = [], [], []
    orc = {r[0]: [] for r in regimes()}

    for seed in range(SEED, SEED + NSEED):
        theta, y = generate(seed)
        km = kalman(y, Q_TRUE, S2_A)
        for nm, sl in regimes():
            orc[nm].append(float(np.mean((km[sl] - theta[sl]) ** 2)))
        for k, pairs in BOXES.items():
            mod.CELL_PAIRS = pairs
            t0 = time.time()
            r = mod.LucidFilter(phis=PHIS, ss=SS).filter(y.reshape(-1, 1))
            acc[k]["secs"].append(time.time() - t0)
            m = np.asarray(r.mean).reshape(-1)
            v = np.asarray(r.var).reshape(-1)
            for nm, sl in regimes():
                acc[k][nm].append(float(np.mean((m[sl] - theta[sl]) ** 2)))
            acc[k]["settle"].append(settle(m - theta))
            sl = regimes()[2][1]
            acc[k]["calib"].append(float(np.mean((m[sl] - theta[sl]) ** 2 / v[sl])))
    mod.CELL_PAIRS = None

    print(f"RMSE ratio to an oracle Kalman told the truth, {NSEED} seeds.")
    print(f"gates: steady <= 1.10x, jump rise <= 4, calibration in [0.6, 1.5]\n")
    print(f"{'box':24s}{'A steady':>10s}{'B jump':>9s}{'C noisy':>9s}"
          f"{'settle':>8s}{'E[e2/S]':>9s}{'ms/step':>9s}")
    for k in BOXES:
        row = [math.sqrt(np.mean(acc[k][nm]) / np.mean(orc[nm])) for nm, _ in regimes()]
        print(f"{k:24s}{row[0]:10.3f}{row[1]:9.3f}{row[2]:9.3f}"
              f"{np.mean(acc[k]['settle']):8.1f}{np.mean(acc[k]['calib']):9.3f}"
              f"{1000 * np.mean(acc[k]['secs']) / N_T:9.2f}")
