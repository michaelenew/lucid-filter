"""0012 -- diagonalise the window trade: is the conflict real, or an artefact of one knob?

0011's sweep scaled the window geometry by a single constant `c` and found the
columns disagreeing -- steady state wants large `c`, settling wants small, regime
C peaks at 1.4-2.0, the jump is non-monotone.  That was read as a genuine
reach-versus-resolution trade with an interior optimum.

But `c` is not one thing.  The window has TWO geometric numbers:

    span = _SPAN_S * s     how far the window reaches      -> REACH
    gap  = _GAP_FACTOR * s how finely it is sampled        -> RESOLUTION

and the node count is `2 ceil(span/gap) + 1`, which the shipped defaults fix at 5
(`_SPAN_S = 3.0`, `_GAP_FACTOR = 1.5`).  Scaling by `c` moves BOTH and holds the
ratio -- so it slides along one diagonal of a two-dimensional plane and cannot
separate the two effects.  If reach and resolution are genuinely different axes,
the columns should diagonalise when the plane is opened up: each metric's optimum
may then sit at the SAME (span, gap) point, with 0011's disagreement an artefact
of being confined to the diagonal.

Two questions, in order:

 1. Sweep `(span, gap)` independently.  Does one point optimise every column?
 2. Diagonalise properly: PCA on the standardised metric matrix over grid points.
    If one component carries nearly all the variance, the metrics are one curve in
    disguise and "the optimum of both" is well posed.  If two components matter,
    the trade is real and the metrics are measuring different things.

The principled single scalar is the filter's own loss.  `optimality-proof`'s
Theorem A' removed the loss seam: both layers read under CODE LENGTH, and that is
what `fit()` optimises.  So the total predictive negative log-likelihood is
carried as its own column, and the question of whether its optimum coincides with
the RMSE columns' is the sharp form of the user's question.

Run: python3 0012_diagonalise.py
"""
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

SEED, N_T, JUMP_AT, JUMP, NOISE_AT = 11, 900, 380, 9.0, 600
Q_TRUE, S2_A, S2_C = 0.02, 1.0, 9.0
NSEED = 12
PHIS, SS = (0.70, 0.85, 0.95), (0.20, 0.40, 0.80, 1.60, 3.20)

SPANS = (1.5, 3.0, 6.0, 12.0)
GAPS = (0.75, 1.5, 3.0)


def load_copy():
    """A private copy of the shipped module whose module-level geometry constants
    can be moved.  Pinned against the installed filter at the defaults."""
    src = open(SRC).read()
    mod = types.ModuleType("lucid_geom")
    mod.__file__ = SRC
    sys.modules["lucid_geom"] = mod
    mod.__dict__["__name__"] = "lucid_geom"
    exec(compile(src, SRC, "exec"), mod.__dict__)
    return mod


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


REG = [("A steady", slice(80, JUMP_AT)),
       ("B jump", slice(JUMP_AT, JUMP_AT + 40)),
       ("C noisy", slice(NOISE_AT + 40, N_T))]


def settle(err):
    seg = np.abs(err[JUMP_AT:JUMP_AT + 120]) < 1.0
    for i in range(seg.size):
        if seg[i:].all():
            return i
    return seg.size


METRICS = ["A steady", "B jump", "C noisy", "settle", "calib", "codelen"]


def score(mod, data, orc):
    """All metrics, oriented so that LOWER IS BETTER."""
    acc = {k: [] for k in METRICS}
    for theta, y in data:
        r = mod.LucidFilter(phis=PHIS, ss=SS).filter(y.reshape(-1, 1))
        m = np.asarray(r.mean).reshape(-1)
        v = np.asarray(r.var).reshape(-1)
        ll = np.asarray(r.loglik).reshape(-1)
        for nm, sl in REG:
            acc[nm].append(float(np.mean((m[sl] - theta[sl]) ** 2)))
        acc["settle"].append(float(settle(m - theta)))
        sl = REG[2][1]
        cal = float(np.mean((m[sl] - theta[sl]) ** 2 / v[sl]))
        acc["calib"].append(abs(math.log(max(cal, 1e-6))))     # |log| : 1.0 is best
        acc["codelen"].append(-float(np.sum(ll[80:])) / (N_T - 80))
    out = {}
    for nm, _ in REG:
        out[nm] = math.sqrt(np.mean(acc[nm]) / np.mean(orc[nm]))
    out["settle"] = float(np.mean(acc["settle"]))
    out["calib"] = float(np.mean(acc["calib"]))
    out["codelen"] = float(np.mean(acc["codelen"]))
    return out


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    mod = load_copy()

    data = [generate(s) for s in range(SEED, SEED + NSEED)]
    orc = {nm: [] for nm, _ in REG}
    for theta, y in data:
        km = kalman(y, Q_TRUE, S2_A)
        for nm, sl in REG:
            orc[nm].append(float(np.mean((km[sl] - theta[sl]) ** 2)))

    a = np.asarray(mod.LucidFilter(phis=PHIS, ss=SS).filter(
        data[0][1].reshape(-1, 1)).mean)
    b = np.asarray(ShippedFilter(phis=PHIS, ss=SS).filter(
        data[0][1].reshape(-1, 1)).mean)
    print(f"pin, copy at defaults vs shipped: max|dev| = "
          f"{float(np.max(np.abs(a - b))):.3e}\n")

    rows, labels = [], []
    print(f"{'span':>6s}{'gap':>6s}{'nodes':>7s}" +
          "".join(f"{k:>10s}" for k in METRICS))
    for sp in SPANS:
        for gp in GAPS:
            mod._SPAN_S, mod._GAP_FACTOR = sp, gp
            nodes = 2 * int(math.ceil(sp / gp)) + 1
            sc = score(mod, data, orc)
            rows.append([sc[k] for k in METRICS])
            labels.append((sp, gp, nodes))
            star = "  <- shipped" if (sp, gp) == (3.0, 1.5) else ""
            print(f"{sp:6.2f}{gp:6.2f}{nodes:7d}" +
                  "".join(f"{sc[k]:10.3f}" for k in METRICS) + star)
    mod._SPAN_S, mod._GAP_FACTOR = 3.0, 1.5

    M = np.array(rows)
    print("\n--- 1. does one point optimise every column? ---")
    best = M.argmin(axis=0)
    for j, k in enumerate(METRICS):
        sp, gp, nd = labels[best[j]]
        print(f"  {k:>9s} best at span={sp:5.2f} gap={gp:5.2f} "
              f"({nd:2d} nodes)   value {M[best[j], j]:.3f}")
    uniq = sorted(set(best.tolist()))
    if len(uniq) == 1:
        sp, gp, nd = labels[uniq[0]]
        print(f"\n  ONE POINT optimises all six: span={sp}, gap={gp} ({nd} nodes).")
    else:
        print(f"\n  {len(uniq)} distinct optima -- not a single point on this grid.")

    print("\n--- 2. diagonalise: PCA on the standardised metric matrix ---")
    L = np.log(np.maximum(M, 1e-9))
    Z = (L - L.mean(0)) / np.maximum(L.std(0), 1e-12)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    print(f"  explained variance: " + "  ".join(f"{v:.3f}" for v in var[:4]))
    print(f"\n  {'metric':>9s}{'PC1':>9s}{'PC2':>9s}")
    for j, k in enumerate(METRICS):
        print(f"  {k:>9s}{Vt[0, j]:9.3f}{Vt[1, j]:9.3f}")
    print(f"\n  grid points ordered along PC1 (the dominant direction):")
    order = np.argsort(U[:, 0] * S[0])
    for i in order:
        sp, gp, nd = labels[i]
        print(f"    span={sp:5.2f} gap={gp:5.2f} ({nd:2d} nodes)   "
              f"PC1={U[i,0]*S[0]:+7.3f}  PC2={U[i,1]*S[1]:+7.3f}")
