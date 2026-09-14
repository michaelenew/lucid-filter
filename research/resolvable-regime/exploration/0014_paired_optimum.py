"""0014 -- paired error bars on 0013's code-length ranking.

0013 found the total prequential code length minimised at the shipped geometry
(span 3.0, gap 1.5), with all four `gap = 1.5` rows taking the top four places.
But the top four span 1.90533 .. 1.90782 -- 0.13% -- and the gap to fifth place
is 0.075%.  Without error bars that ranking is not readable.

Every configuration sees the SAME records here, so the comparison is paired and
its standard error is far smaller than either configuration's own scatter.

Run: python3 0014_paired_optimum.py
"""
import math
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "lucid", "filter", "lucid.py")

SEED, N_T, JUMP_AT, JUMP, NOISE_AT = 11, 900, 380, 9.0, 600
Q_TRUE, S2_A, S2_C = 0.02, 1.0, 9.0
NSEED = 24
PHIS, SS = (0.70, 0.85, 0.95), (0.20, 0.40, 0.80, 1.60, 3.20)
POINTS = [(3.0, 1.5), (1.5, 1.5), (6.0, 1.5), (12.0, 1.5),
          (12.0, 0.75), (6.0, 0.75), (3.0, 3.0), (3.0, 0.75)]


def load_copy():
    mod = types.ModuleType("lucid_pair")
    mod.__file__ = SRC
    sys.modules["lucid_pair"] = mod
    mod.__dict__["__name__"] = "lucid_pair"
    exec(compile(open(SRC).read(), SRC, "exec"), mod.__dict__)
    return mod


def generate(seed):
    rng = np.random.default_rng(seed)
    th = np.cumsum(rng.normal(0.0, math.sqrt(Q_TRUE), N_T))
    th[JUMP_AT:] += JUMP
    sd = np.where(np.arange(N_T) < NOISE_AT, math.sqrt(S2_A), math.sqrt(S2_C))
    return th + rng.normal(0.0, sd)


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    mod = load_copy()
    ys = [generate(s) for s in range(SEED, SEED + NSEED)]
    per = {}
    for sp, gp in POINTS:
        mod._SPAN_S, mod._GAP_FACTOR = sp, gp
        per[(sp, gp)] = np.array([
            -float(np.asarray(mod.LucidFilter(phis=PHIS, ss=SS)
                              .filter(y.reshape(-1, 1)).loglik)) / N_T
            for y in ys])
    mod._SPAN_S, mod._GAP_FACTOR = 3.0, 1.5

    base = per[(3.0, 1.5)]
    print(f"total prequential code length, nats/step, {NSEED} paired seeds.")
    print(f"reference: the SHIPPED geometry (span 3.0, gap 1.5) = "
          f"{base.mean():.5f}\n")
    print(f"{'span':>6s}{'gap':>6s}{'nodes':>7s}{'codelen':>11s}"
          f"{'vs shipped':>13s}{'paired SEM':>12s}{'t':>8s}")
    for sp, gp in POINTS:
        v = per[(sp, gp)]
        d = v - base
        sem = d.std(ddof=1) / math.sqrt(NSEED) if (sp, gp) != (3.0, 1.5) else 0.0
        t = d.mean() / sem if sem > 0 else 0.0
        nd = 2 * int(math.ceil(sp / gp)) + 1
        tag = "  <- shipped" if (sp, gp) == (3.0, 1.5) else ""
        print(f"{sp:6.2f}{gp:6.2f}{nd:7d}{v.mean():11.5f}{d.mean():+13.5f}"
              f"{sem:12.5f}{t:8.2f}{tag}")
    print("\npositive 'vs shipped' = worse than shipped.  |t| > 2 is resolved.")
