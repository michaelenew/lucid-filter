"""0013 -- the arbiter: under the filter's OWN loss, is there one optimum?

0012 diagonalised the window geometry and the answer to "does one point optimise
every column" was no -- but the PCA said something sharper than 0011 did.

    PC1 (63.7% of the variance) orders the grid by GAP, and it loads
    calibration (+0.550) OPPOSITE to all three RMSE columns (-0.47, -0.38, -0.55).
    PC2 (25.7%) orders by SPAN: settle (+0.76) and jump (+0.55) want reach.

So the conflict is not reach-versus-resolution, as 0011 read it.  Reach (PC2) is
almost unconflicted -- more span helps the jump and the settling and costs steady
state almost nothing.  **The conflict is on the resolution axis, and it is between
being ACCURATE and being HONEST**: a fine gap calibrates well and scores worse
RMSE; a coarse gap scores better RMSE and is badly calibrated.

That is a conflict between two different losses, which is exactly the seam
`optimality-proof` closed.  Theorem A' removed it: BOTH layers read under code
length, and code length is the loss `fit()` optimises.  Code length prices
accuracy and honesty together -- an overconfident filter pays for its
overconfidence whatever its RMSE.

**So the sharp form of the question is whether the code-length optimum is a single
point.** If it is, then "one point is the optimum of both" is true in the only
sense the repository's own theory licenses, and the disagreement between the RMSE
and calibration columns is an artefact of scoring under a loss neither layer uses.

`r.loglik` is the SCALAR total for the record, so a regime's own prequential code
length is an exact difference of two runs:

    codelen(C) = [ -loglik(y) + loglik(y[:NOISE_AT]) ] / (N - NOISE_AT)

(0012 sliced the scalar as if it were a series and read 0.000 in every cell; that
column is void there and is replaced here.)

Run: python3 0013_code_length_arbiter.py
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
    mod = types.ModuleType("lucid_geom2")
    mod.__file__ = SRC
    sys.modules["lucid_geom2"] = mod
    mod.__dict__["__name__"] = "lucid_geom2"
    exec(compile(open(SRC).read(), SRC, "exec"), mod.__dict__)
    return mod


def generate(seed):
    rng = np.random.default_rng(seed)
    th = np.cumsum(rng.normal(0.0, math.sqrt(Q_TRUE), N_T))
    th[JUMP_AT:] += JUMP
    sd = np.where(np.arange(N_T) < NOISE_AT, math.sqrt(S2_A), math.sqrt(S2_C))
    return th, th + rng.normal(0.0, sd)


def code_lengths(mod, y):
    """(whole record, regime-A prefix, regime-C block, jump block) in nats/step."""
    def ll(seg):
        return float(np.asarray(
            mod.LucidFilter(phis=PHIS, ss=SS).filter(seg.reshape(-1, 1)).loglik))
    L_all = ll(y)
    L_pre = ll(y[:NOISE_AT])
    L_jmp = ll(y[:JUMP_AT])
    return (-L_all / N_T,
            -L_jmp / JUMP_AT,
            -(L_pre - L_jmp) / (NOISE_AT - JUMP_AT),
            -(L_all - L_pre) / (N_T - NOISE_AT))


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    mod = load_copy()
    data = [generate(s) for s in range(SEED, SEED + NSEED)]

    a = np.asarray(mod.LucidFilter(phis=PHIS, ss=SS).filter(
        data[0][1].reshape(-1, 1)).mean)
    b = np.asarray(ShippedFilter(phis=PHIS, ss=SS).filter(
        data[0][1].reshape(-1, 1)).mean)
    print(f"pin, copy at defaults vs shipped: max|dev| = "
          f"{float(np.max(np.abs(a - b))):.3e}\n")

    COLS = ["whole", "A pre-jump", "B jump-block", "C noisy"]
    rows, labels = [], []
    print("prequential code length, nats/step (LOWER IS BETTER, all four)\n")
    print(f"{'span':>6s}{'gap':>6s}{'nodes':>7s}" + "".join(f"{c:>14s}" for c in COLS))
    for sp in SPANS:
        for gp in GAPS:
            mod._SPAN_S, mod._GAP_FACTOR = sp, gp
            vals = np.mean([code_lengths(mod, y) for _, y in data], axis=0)
            sems = np.std([code_lengths(mod, y) for _, y in data], axis=0, ddof=1) \
                / math.sqrt(NSEED)
            rows.append(vals)
            labels.append((sp, gp, 2 * int(math.ceil(sp / gp)) + 1))
            star = "  <- shipped" if (sp, gp) == (3.0, 1.5) else ""
            print(f"{sp:6.2f}{gp:6.2f}{labels[-1][2]:7d}" +
                  "".join(f"{v:14.5f}" for v in vals) + star)
    mod._SPAN_S, mod._GAP_FACTOR = 3.0, 1.5

    M = np.array(rows)
    print("\n--- does ONE point minimise the code length everywhere? ---")
    best = M.argmin(axis=0)
    for j, c in enumerate(COLS):
        sp, gp, nd = labels[best[j]]
        print(f"  {c:>13s}  best at span={sp:5.2f} gap={gp:5.2f} ({nd:2d} nodes)"
              f"   {M[best[j], j]:.5f} nats/step")
    uniq = sorted(set(best.tolist()))
    if len(uniq) == 1:
        sp, gp, nd = labels[uniq[0]]
        print(f"\n  YES -- one point: span={sp}, gap={gp} ({nd} nodes).")
    else:
        print(f"\n  NO -- {len(uniq)} distinct optima under code length.")

    L = np.log(M)
    Z = (L - L.mean(0)) / np.maximum(L.std(0), 1e-12)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    print(f"\n--- diagonalised under code length ---")
    print(f"  explained variance: " + "  ".join(f"{v:.3f}" for v in var[:3]))
    print(f"  {'block':>13s}{'PC1':>9s}{'PC2':>9s}")
    for j, c in enumerate(COLS):
        print(f"  {c:>13s}{Vt[0, j]:9.3f}{Vt[1, j]:9.3f}")
    print(f"\n  all-same-sign on PC1 would mean the blocks agree on direction.")
    print(f"\n  grid ordered by TOTAL code length (the record's own score):")
    for i in np.argsort(M[:, 0]):
        sp, gp, nd = labels[i]
        print(f"    span={sp:5.2f} gap={gp:5.2f} ({nd:2d} nodes)   "
              f"{M[i, 0]:.5f} nats/step")
