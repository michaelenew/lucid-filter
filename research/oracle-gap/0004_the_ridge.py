"""0004 -- The ridge: GPB1 measures the mean process variance and cannot
split it; IMM can.

`0003` B found the current fit no longer lands on s_P = 0 on 0029-style data
-- the speedup's staged fit escapes the boundary -- but it lands at
(Q = 0.39, s_P = 1.44) against a truth of (1.0, 0.8), and every screen
variant polishes to the same point.  The giveaway is the product:
Q e^{s_P^2/2} = 1.11, the truth's mean process variance (1 * e^{0.32} = 1.38,
within this window's draw).  So the likelihood pins the MEAN variance and is
nearly flat along the ridge that trades the median Q against the spread s_P.

That is the same degeneracy `0039` met at the boundary (s_P = 0 is the
ridge's endpoint), the same one that makes the two reported basins -- two
different (phi, s) splits of similar total variance -- an optimiser's choice,
and the same evidence the GPB1 collapse deletes: WHICH split is right is
visible only in the accumulated history of the covariance, which a shared-P
filter erases every step.

The measurement: walk the ridge Q(s) = Qeff * e^{-s^2/2} at the generating
Qeff and phi_P, score both filters.  If the reframe is right, GPB1 is flat
along it to a few millinats and IMM digs a minimum at the generating s_P.

Run:  python3 0004_the_ridge.py
"""
import json
import math
import os
import sys
from importlib import import_module

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lucid"))

from odefilter import Params  # noqa: E402

_m2 = import_module("0002_per_node_covariances")
ode, logscale, imm_run, gpb1_run = _m2.ode, _m2.logscale, _m2.imm_run, _m2.gpb1_run

ALPHA3 = np.array([2.785218519281637, -2.6855430450862655, 0.9003245225862656])
Q0, S20 = 1.0, 9.0
FIG = os.path.join(HERE, "figures")


def main():
    os.makedirs(FIG, exist_ok=True)
    out = {}
    grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.44, 1.6]
    print("THE RIDGE -- Q(s) = Qeff e^{-s^2/2}, phi_P = 0.9, nll/pt")
    for tag, seed, true_s in (("true s_P = 0.8", 19, 0.8),
                              ("true s_P = 0   ", 23, 0.0)):
        rng = np.random.default_rng(seed)
        n = 900
        lam = logscale(n, rng, 0.9, true_s) if true_s > 0 else np.zeros(n)
        y = (ode(n, ALPHA3, Q0 * np.exp(lam), rng)
             + math.sqrt(S20) * rng.standard_normal(n))
        qeff = Q0 * math.exp(true_s ** 2 / 2.0)

        gp, im = [], []
        for s in grid:
            pr = Params(ALPHA3, qeff * math.exp(-s * s / 2.0), S20,
                        phi_P=0.9, s_P=s)
            gp.append(gpb1_run(y, pr)[0])
            im.append(imm_run(y, pr)[0])
        out[tag.strip()] = dict(grid=grid, gpb1=gp, imm=im,
                                qeff=qeff, true_s=true_s)
        ag, ai = grid[int(np.argmin(gp))], grid[int(np.argmin(im))]
        rg = max(gp) - min(gp)
        ri = max(im) - min(im)
        print(f"  {tag}  (Qeff = {qeff:.3f})")
        print("    s_P    " + "  ".join(f"{s:7.2f}" for s in grid))
        print("    gpb1   " + "  ".join(f"{v:7.4f}" for v in gp)
              + f"   argmin {ag}, range {rg:.4f}")
        print("    imm    " + "  ".join(f"{v:7.4f}" for v in im)
              + f"   argmin {ai}, range {ri:.4f}")

    with open(os.path.join(FIG, "gap0004.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", os.path.join(FIG, "gap0004.json"))


if __name__ == "__main__":
    main()
